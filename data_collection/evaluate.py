#!/usr/bin/env python3
"""
Evaluate a trained diffusion policy on the real robot.

Loads a checkpoint, connects to the Franka arm + LEAP hand + cameras,
runs the policy in a loop, and logs the results for human review.

Supports both image-based (ResNet) and point-cloud-based (DP3) checkpoints.
The checkpoint config's 'encoder_type' field determines which mode to use.

Workflow per trial:
  1. Robot moves to a fixed reset pose (manual or automatic).
  2. [PC mode] Robot arm moves out of camera view; clean point cloud captured.
  3. Policy observes for obs_horizon steps to fill the observation buffer.
  4. Policy predicts a pred_horizon action chunk (joint position targets).
  5. First action_horizon steps of the chunk are executed on the robot.
  6. Steps 3-5 repeat until the episode ends (max steps or human stop).
  7. Human rates the grasp (good / bad) via keyboard.

Usage:
    python3 evaluate.py --checkpoint ../checkpoints/best.pt
    python3 evaluate.py --checkpoint ../checkpoints/best.pt --hold 0 --trials 10
    python3 evaluate.py --checkpoint ../checkpoints/best.pt --dry-run  # no robot
    python3 evaluate.py --checkpoint ../checkpoints/best.pt --grasp-type crimp
"""

import os
import sys
import time
import json
import argparse
import threading
from pathlib import Path
from collections import deque
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SCRIPT_DIR = Path(__file__).resolve().parent
TELE_ROOT = SCRIPT_DIR.parent

FRANKA_SCRIPTS_DIR = TELE_ROOT / "TeleoperationUnity" / "Robot Control - Python" / "Franka Scripts"
LEAP_DIR = TELE_ROOT / "TeleoperationUnity" / "LEAP" / "leaphandv1" / "for_transfer"
LEAP_API_DIR = LEAP_DIR / "LEAP_Hand_API" / "python"

for p in [str(FRANKA_SCRIPTS_DIR), str(LEAP_DIR), str(LEAP_API_DIR), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from episode_storage import resize_image, GRASP_TYPE_IDS, GRASP_TYPE_NAMES, N_GRASP_TYPES

import signal

CAMERA_NUMBERS = [2, 3, 4, 5]
CAMERA_RAW_W = 848
CAMERA_RAW_H = 480
CONTROL_FREQ = 10

# Verified 2026-03-18: joints where arm is fully clear of all 4 RealSense cameras.
PARK_ARM_JOINTS = np.array([-0.11426599, -0.56029082, -0.06635159, -2.17443357,
                              0.04112932,  2.15592909,  0.54378958], dtype=np.float64)

MAX_JOINT_STEP_RAD = 0.03

HOLD_NAMES = {0: "edge_A", 1: "edge_B", 2: "sloper", 3: "pinch", 4: "test_edge"}

DEFAULT_RESULTS_DIR = TELE_ROOT / "eval_results"

RESET_ARM_JOINTS = np.array([-0.1111, -0.1703, -0.0621, -2.3442, 0.0408, 2.1952, 0.1559],
                            dtype=np.float32)
RESET_HAND_ALLEGRO = np.array([0.078, 0.31, 0.41, 0.25, 0.14, 0.46, 0.50, 0.30,
                               0.43, 0.56, 0.55, 0.30, 0.17, -0.47, -0.10, 1.22],
                              dtype=np.float32)

# Number of frames averaged for point cloud capture
PC_CAPTURE_N_FRAMES = 5
PC_N_POINTS = 1024


def load_policy(ckpt_path, device):
    """Load policy from checkpoint. Returns (policy, cfg, norm_stats).

    Handles both image-based (encoder_type='vision') and
    point-cloud-based (encoder_type='point_cloud') checkpoints.
    """
    from train import DiffusionPolicy, PointCloudDiffusionPolicy

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt["config"]
    encoder_type = cfg.get("encoder_type", "vision")

    if encoder_type == "point_cloud":
        down_dims = tuple(cfg.get("down_dims", [256, 512, 1024]))
        policy = PointCloudDiffusionPolicy(
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            obs_horizon=cfg["obs_horizon"],
            pred_horizon=cfg["pred_horizon"],
            num_diffusion_steps=cfg["diffusion_steps"],
            down_dims=down_dims,
            n_grasp_types=cfg.get("n_grasp_types", N_GRASP_TYPES),
            use_grasp_conditioning=cfg.get("use_grasp_conditioning", True),
        ).to(device)
    else:
        down_dims = tuple(cfg.get("down_dims", [512, 1024, 2048]))
        img_channels = cfg.get("img_channels", 3)
        policy = DiffusionPolicy(
            state_dim=cfg["state_dim"],
            action_dim=cfg["action_dim"],
            n_cams=cfg["n_cams"],
            obs_horizon=cfg["obs_horizon"],
            pred_horizon=cfg["pred_horizon"],
            num_diffusion_steps=cfg["diffusion_steps"],
            down_dims=down_dims,
            img_channels=img_channels,
        ).to(device)

    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    norm_path = Path(ckpt_path).parent / "norm_stats.json"
    with open(norm_path) as f:
        norm = json.load(f)

    norm_type = norm.get("normalization", "zscore")
    if norm_type == "minmax":
        norm_stats = {
            "normalization": "minmax",
            "state_min": np.array(norm["state_min"], dtype=np.float32),
            "state_range": np.array(norm["state_range"], dtype=np.float32),
            "action_min": np.array(norm["action_min"], dtype=np.float32),
            "action_range": np.array(norm["action_range"], dtype=np.float32),
        }
    else:
        norm_stats = {
            "normalization": "zscore",
            "state_mean": np.array(norm["state_mean"], dtype=np.float32),
            "state_std": np.array(norm["state_std"], dtype=np.float32),
            "action_mean": np.array(norm["action_mean"], dtype=np.float32),
            "action_std": np.array(norm["action_std"], dtype=np.float32),
        }

    return policy, cfg, norm_stats


def build_state_vector_23(arm_joints, hand_joints):
    """arm(7) + hand(16) = 23-dim state (new pipeline)."""
    return np.concatenate([
        np.array(arm_joints, dtype=np.float32).ravel()[:7],
        np.array(hand_joints, dtype=np.float32).ravel()[:16],
    ])


def build_state_vector_30(arm_joints, ee_pos, ee_quat, hand_joints):
    """arm(7) + ee_pos(3) + ee_quat(4) + hand(16) = 30-dim state (legacy)."""
    return np.concatenate([
        np.array(arm_joints, dtype=np.float32).ravel()[:7],
        np.array(ee_pos, dtype=np.float32).ravel()[:3],
        np.array(ee_quat, dtype=np.float32).ravel()[:4],
        np.array(hand_joints, dtype=np.float32).ravel()[:16],
    ])


class PolicyEvaluator:
    def __init__(self, ckpt_path, hold_id=0, n_trials=5, max_steps=200,
                 action_horizon=8, dry_run=False, results_dir=DEFAULT_RESULTS_DIR,
                 num_inference_steps=10, grasp_type=None, zero_pc=False,
                 pull_dist=None, pull_stiffness=4000.0,
                 pull_lateral_stiffness=100.0, pull_z_stiffness=2000.0,
                 pull_z_bias=0.0):
        self.hold_id = hold_id
        self.ckpt_name = Path(ckpt_path).parent.name  # e.g. "pc_with_taxonomy"
        self.n_trials = n_trials
        self.max_steps = max_steps
        self.action_horizon = action_horizon
        self.dry_run = dry_run
        self.zero_pc = zero_pc
        self.num_inference_steps = num_inference_steps
        self.pull_dist             = pull_dist              # meters; None = pull test disabled
        self.pull_stiffness        = pull_stiffness         # N/m; X (pull axis) stiffness
        self.pull_lateral_stiffness = pull_lateral_stiffness  # N/m; Y (lateral) stiffness
        self.pull_z_stiffness      = pull_z_stiffness       # N/m; Z (vertical) stiffness
        self.pull_z_bias           = pull_z_bias            # meters upward Z bias to counter LEAP hand weight
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading policy from {ckpt_path}...")
        self.policy, self.cfg, self.norm = load_policy(ckpt_path, self.device)

        self.encoder_type = self.cfg.get("encoder_type", "vision")
        self.use_pc = (self.encoder_type == "point_cloud")
        self.img_size = self.cfg.get("img_size", 224)
        self.img_channels = self.cfg.get("img_channels", 3)
        self.use_rgbd = (self.img_channels == 4)
        self.state_dim = self.cfg["state_dim"]

        print(f"  encoder_type={self.encoder_type}, "
              f"state_dim={self.state_dim}, action_dim={self.cfg['action_dim']}, "
              f"obs_horizon={self.cfg['obs_horizon']}, "
              f"pred_horizon={self.cfg['pred_horizon']}")

        # Grasp type for PC mode
        if self.use_pc:
            if grasp_type is None:
                grasp_type = self._prompt_grasp_type()
            self.grasp_type = grasp_type
            self.grasp_type_id = GRASP_TYPE_IDS.get(grasp_type, 0)
            print(f"  grasp_type={grasp_type} (id={self.grasp_type_id})")
        else:
            self.grasp_type = grasp_type
            self.grasp_type_id = 0

        # Warm up GPU
        print("  Warming up GPU (one-time)...", end=" ", flush=True)
        self._warmup_gpu()
        print("done")

        self.cam_names = self.cfg.get("cam_names", [f"cam{n}" for n in CAMERA_NUMBERS])
        self.obs_horizon = self.cfg["obs_horizon"]
        self.pred_horizon = self.cfg["pred_horizon"]

        self.fa = None
        self.leap_dxl = None
        self.cameras = None
        self._live_active = False
        self._ros_pub = None
        self._ros_id = 0
        self._live_init_time = 0

        # Cached camera calibration for point cloud
        self._cam_intrinsics = None
        self._cam_extrinsics = None

    def _prompt_grasp_type(self):
        uses_cond = self.cfg.get("use_grasp_conditioning", True)
        label = "grasp type (fed to network)" if uses_cond else "grasp type (for logging — model ignores it)"
        print(f"\nSelect {label}:")
        for gid, gname in GRASP_TYPE_NAMES.items():
            print(f"  {gid}: {gname}")
        while True:
            try:
                choice = input("Enter grasp type ID or name: ").strip().lower()
                if choice.isdigit():
                    gid = int(choice)
                    if gid in GRASP_TYPE_NAMES:
                        return GRASP_TYPE_NAMES[gid]
                elif choice in GRASP_TYPE_IDS:
                    return choice
                print(f"  Invalid. Enter 0-{N_GRASP_TYPES-1} or a name.")
            except (EOFError, KeyboardInterrupt):
                print("Defaulting to 'crimp'")
                return "crimp"

    def _warmup_gpu(self):
        """One dummy forward pass to warm up CUDA kernels."""
        with torch.no_grad():
            if self.use_pc:
                dummy_s = torch.zeros(1, self.cfg["obs_horizon"], self.state_dim,
                                      device=self.device)
                dummy_pc = torch.zeros(1, PC_N_POINTS, 3, device=self.device)
                dummy_gt = torch.zeros(1, dtype=torch.long, device=self.device)
                self.policy.predict_action(
                    dummy_s, dummy_pc, dummy_gt,
                    num_inference_steps=self.num_inference_steps)
            else:
                n_cams = len(self.cam_names)
                dummy_s = torch.zeros(1, self.cfg["obs_horizon"], self.state_dim,
                                      device=self.device)
                dummy_i = torch.zeros(1, self.cfg["obs_horizon"], n_cams,
                                      self.img_channels, self.img_size, self.img_size,
                                      device=self.device)
                self.policy.predict_action(
                    dummy_s, dummy_i,
                    num_inference_steps=self.num_inference_steps)

    def setup(self):
        self._setup_franka()
        self._setup_leap()
        self._setup_cameras()
        print(f"\nReady to evaluate on hold {self.hold_id} "
              f"({HOLD_NAMES.get(self.hold_id, '?')}), {self.n_trials} trials\n")

    def _setup_franka(self):
        if self.dry_run:
            print("[Franka] dry-run, skipped")
            return
        from frankapy import FrankaArm
        self.fa = FrankaArm(with_gripper=False)
        try:
            self.fa.stop_skill()
        except Exception:
            pass
        time.sleep(0.5)
        print(f"[Franka] connected")

    def _setup_leap(self):
        if self.dry_run:
            print("[LEAP] dry-run, skipped")
            return
        from leap_pip_dip_teleop import find_leap_port
        from leap_hand_utils.dynamixel_client import DynamixelClient
        import leap_hand_utils.leap_hand_utils as lhu
        self._lhu = lhu
        self._leap_motors = list(range(16))

        port = find_leap_port()
        if port is None:
            print("[LEAP] WARNING: no LEAP hand USB device found")
            return

        try:
            self.leap_dxl = DynamixelClient(self._leap_motors, port, 4000000)
            self.leap_dxl.connect()
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 5, 11, 1)
            self.leap_dxl.set_torque_enabled(self._leap_motors, True)
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 800, 84, 2)
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 200, 80, 2)
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 600, 102, 2)
            print(f"[LEAP] connected on {port}")
        except Exception as e:
            print(f"[LEAP] WARNING: connection failed: {e}")
            self.leap_dxl = None

    def _setup_cameras(self):
        import robomail.vision as vis
        cam_nums = [int(c.replace("cam", "")) for c in self.cam_names]
        self._cam_nums = cam_nums
        self.cameras = vis.ThreadedCameras(
            cam_numbers=cam_nums, image_height=CAMERA_RAW_H, image_width=CAMERA_RAW_W,
            get_point_cloud=False, get_verts=False,
        )
        self.cameras.get_next_frames()
        print(f"[Cameras] {cam_nums} streaming")

        # Cache camera calibration for point cloud mode
        if self.use_pc:
            try:
                from point_cloud_utils import (
                    get_cam_intrinsics_from_realsense,
                    get_cam_extrinsics_from_realsense,
                )
                self._cam_intrinsics = [
                    get_cam_intrinsics_from_realsense(c) for c in self.cameras.cameras]
                self._cam_extrinsics = [
                    get_cam_extrinsics_from_realsense(c) for c in self.cameras.cameras]
                print(f"[Cameras] Calibration loaded for {len(self._cam_intrinsics)} cameras")
            except Exception as e:
                print(f"[Cameras] WARNING: calibration failed ({e})")

    def _capture_clean_point_cloud(self):
        """Capture fused point cloud from all cameras. Arm must be out of view.

        Returns:
            pc: (PC_N_POINTS, 3) float32 in world frame
        """
        from point_cloud_utils import fuse_multi_camera_points

        if self._cam_intrinsics is None or self._cam_extrinsics is None:
            print("  WARNING: No camera calibration. Returning zero PC.")
            return np.zeros((PC_N_POINTS, 3), dtype=np.float32)

        pcs = []
        for frame_i in range(PC_CAPTURE_N_FRAMES):
            raw_frames = self.cameras.get_next_frames()
            depth_images = []
            for i in range(len(self._cam_nums)):
                depth = raw_frames[i][1]
                if depth is None or depth.size == 0:
                    depth = np.zeros((CAMERA_RAW_H, CAMERA_RAW_W), dtype=np.uint16)
                depth_images.append(depth)
            pc = fuse_multi_camera_points(
                depth_images=depth_images,
                cam_intrinsics=self._cam_intrinsics,
                cam_extrinsics=self._cam_extrinsics,
                n_points=PC_N_POINTS,
                outlier_removal=(frame_i == 0),
            )
            pcs.append(pc)
            time.sleep(0.05)
        return pcs[-1]  # Return the last clean capture

    def _read_state(self):
        """Read robot state. Returns state vector matching the policy's state_dim."""
        if self.fa is not None:
            joints = np.array(self.fa.get_joints(), dtype=np.float32)
            pose = self.fa.get_pose()
            ee_pos = np.array(pose.translation, dtype=np.float32)
            ee_quat = np.array(pose.quaternion, dtype=np.float32)
        else:
            joints = np.zeros(7, dtype=np.float32)
            ee_pos = np.zeros(3, dtype=np.float32)
            ee_quat = np.array([1, 0, 0, 0], dtype=np.float32)

        if self.leap_dxl is not None:
            try:
                raw = self.leap_dxl.read_pos()
                hand = self._lhu.LEAPhand_to_allegro(raw, zeros=False).astype(np.float32)
            except Exception:
                hand = np.zeros(16, dtype=np.float32)
        else:
            hand = np.zeros(16, dtype=np.float32)

        if self.state_dim == 23:
            return build_state_vector_23(joints, hand)
        else:
            # Legacy 30-dim
            return build_state_vector_30(joints, ee_pos, ee_quat, hand)

    def _read_images(self):
        raw = self.cameras.get_frames()
        imgs = {}
        for i, name in enumerate(self.cam_names):
            color = raw[i][0]
            imgs[name] = resize_image(color, self.img_size, self.img_size)
            if self.use_rgbd:
                depth = raw[i][1]
                if depth is not None and depth.size > 0:
                    d = cv2.resize(depth, (self.img_size, self.img_size),
                                   interpolation=cv2.INTER_NEAREST)
                else:
                    d = np.zeros((self.img_size, self.img_size), dtype=np.uint16)
                imgs[name + "_depth"] = d
        return imgs

    def _normalize_state(self, s):
        norm = self.norm
        if norm["normalization"] == "minmax":
            return 2.0 * (s - norm["state_min"]) / norm["state_range"] - 1.0
        else:
            return (s - norm["state_mean"]) / norm["state_std"]

    def _unnormalize_action(self, a):
        norm = self.norm
        if norm["normalization"] == "minmax":
            return (a + 1.0) / 2.0 * norm["action_range"] + norm["action_min"]
        else:
            return a * norm["action_std"] + norm["action_mean"]

    def _start_live_control(self):
        if self.fa is None or self._live_active:
            return
        import rospy
        from frankapy import SensorDataMessageType, FrankaConstants as FC
        from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
        from frankapy.proto import JointPositionSensorMessage
        from franka_interface_msgs.msg import SensorDataGroup

        self._ros_pub = rospy.Publisher(
            FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
        self._ros_id = 0

        cur = np.array(self.fa.get_joints()).tolist()
        self.fa.goto_joints(cur, duration=1000, dynamic=True, buffer_time=10)
        self._live_init_time = rospy.Time.now().to_time()
        self._live_active = True
        time.sleep(0.1)
        print("[Arm] Live joint control started")

    def _stop_live_control(self):
        if not self._live_active:
            return
        try:
            self.fa.stop_skill()
        except Exception:
            pass
        self._live_active = False
        time.sleep(1.5)
        print("[Arm] Live joint control stopped")

    def _send_joint_target(self, joints):
        import rospy
        from frankapy import SensorDataMessageType
        from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
        from frankapy.proto import JointPositionSensorMessage
        from franka_interface_msgs.msg import SensorDataGroup

        timestamp = rospy.Time.now().to_time() - self._live_init_time
        msg = JointPositionSensorMessage(
            id=self._ros_id, timestamp=timestamp,
            joints=np.array(joints).tolist())
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                msg, SensorDataMessageType.JOINT_POSITION))
        self._ros_pub.publish(ros_msg)
        self._ros_id += 1

    def _execute_action(self, action_vec):
        """Send arm + hand joint targets to robot.

        Supports 23-dim (new: arm7 + hand16) and legacy 30/36-dim layouts.
        """
        if self.state_dim == 23:
            arm_target = np.array(action_vec[:7], dtype=np.float64)
            hand_target_allegro = action_vec[7:23]
        else:
            # Legacy 30-dim: arm(7) + ee_pos(3) + ee_quat(4) + hand(16)
            arm_target = np.array(action_vec[:7], dtype=np.float64)
            hand_target_allegro = action_vec[14:30]

        if self.fa is not None and self._live_active:
            cur_joints = np.array(self.fa.get_joints(), dtype=np.float64)
            delta = arm_target - cur_joints
            clipped = np.clip(delta, -MAX_JOINT_STEP_RAD, MAX_JOINT_STEP_RAD)
            clamped_target = cur_joints + clipped
            self._send_joint_target(clamped_target)

        if self.leap_dxl is not None:
            try:
                leap_target = self._lhu.allegro_to_LEAPhand(hand_target_allegro, zeros=False)
                self.leap_dxl.write_desired_pos(self._leap_motors, leap_target)
            except Exception:
                pass

    def _execute_pull(self, angle_deg=180.0):
        """Pull arm toward robot base (default 180° = -X direction) by self.pull_dist meters.

        Uses impedance control with differential per-axis stiffness:
          kx (pull axis, default 4000 N/m) — stiff, drives the motion
          ky (lateral,   default  100 N/m) — compliant, natural side flex
          kz (vertical,  default  100 N/m) — compliant, natural up/down flex

        Called after live control is already stopped. Arm is left at the displaced
        pose; the next _reset_robot() call brings it home.
        """
        if self.fa is None:
            print("  [pull] No FrankaArm — skipping")
            return
        import copy
        angle_rad = np.deg2rad(angle_deg)
        dx = self.pull_dist * np.cos(angle_rad)
        dy = self.pull_dist * np.sin(angle_rad)
        try:
            self.fa.stop_skill()
        except Exception:
            pass
        time.sleep(0.3)
        current_pose = self.fa.get_pose()
        target_pose = copy.deepcopy(current_pose)
        target_pose.translation = current_pose.translation + np.array([dx, dy, self.pull_z_bias])
        kx = float(self.pull_stiffness)
        ky = float(self.pull_lateral_stiffness)
        kz = float(self.pull_z_stiffness)
        print(f"  [pull] {self.pull_dist*100:.1f} cm at {angle_deg:.0f}° "
              f"(dx={dx*100:.1f}, dy={dy*100:.1f} cm) "
              f"[kx={kx:.0f} / ky={ky:.0f} / kz={kz:.0f} N/m] ...")
        try:
            self.fa.goto_pose(target_pose, duration=5.0, dynamic=False,
                              buffer_time=0.2, use_impedance=True,
                              cartesian_impedances=[kx, ky, kz, 10.0, 10.0, 10.0])
        except Exception as e:
            print(f"  [pull] WARNING: move failed: {e}")

    def _reset_robot(self, timeout=15):
        print("Resetting robot to home pose...")
        just_stopped = False
        if self._live_active:
            self._stop_live_control()
            just_stopped = True
        if self.fa is not None:
            if not just_stopped:
                try:
                    self.fa.stop_skill()
                except Exception:
                    pass
                time.sleep(1.0)
            done = threading.Event()
            error = [None]

            def _go():
                try:
                    self.fa.goto_joints(RESET_ARM_JOINTS.tolist(), duration=5.0,
                                        dynamic=False, buffer_time=0.2, block=True)
                except Exception as e:
                    error[0] = e
                finally:
                    done.set()

            t = threading.Thread(target=_go, daemon=True)
            t.start()
            if not done.wait(timeout):
                print(f"  WARNING: goto_joints timed out after {timeout}s, stopping skill...")
                try:
                    self.fa.stop_skill()
                except Exception:
                    pass
                time.sleep(0.5)
            elif error[0] is not None:
                print(f"  WARNING: goto_joints error: {error[0]}")
            else:
                time.sleep(0.5)
        if self.leap_dxl is not None:
            leap_target = self._lhu.allegro_to_LEAPhand(RESET_HAND_ALLEGRO, zeros=False)
            self.leap_dxl.write_desired_pos(self._leap_motors, leap_target)
            time.sleep(0.3)
        print("Reset done.")

    def run_trial(self, trial_idx):
        """Run a single evaluation trial. Returns dict with trial data."""
        print(f"\n--- Trial {trial_idx + 1}/{self.n_trials} ---")

        if not self.dry_run:
            self._reset_robot()

        cv2.namedWindow("Eval", cv2.WINDOW_NORMAL)

        # If point cloud mode: auto-park arm, capture PC, return to reset pose
        scene_pc = None
        if self.use_pc:
            # Auto-park
            if self.fa is not None:
                print("  Moving arm to park pose (out of camera view)...")
                try:
                    self.fa.stop_skill()
                except Exception:
                    pass
                time.sleep(0.3)
                self.fa.goto_joints(
                    PARK_ARM_JOINTS.tolist(), duration=4.0,
                    dynamic=False, buffer_time=0.2, block=True)
                time.sleep(1.5)
                print("  Arm at park pose.")
            else:
                print("  WARNING: No FrankaArm — ensure arm is manually out of camera view.")

            # Capture PC
            if self.zero_pc:
                print("  [--zero-pc] Feeding all-zeros to policy.")
                scene_pc = np.zeros((PC_N_POINTS, 3), dtype=np.float32)
            else:
                print("  Capturing point cloud...", end=" ", flush=True)
                scene_pc = self._capture_clean_point_cloud()
                n_valid = np.sum(np.any(scene_pc != 0, axis=-1))
                print(f"done ({n_valid}/{PC_N_POINTS} valid pts)")

            # Return arm to reset (approach) pose
            if self.fa is not None:
                print("  Returning arm to approach pose...")
                try:
                    self.fa.stop_skill()
                except Exception:
                    pass
                time.sleep(0.3)
                self.fa.goto_joints(
                    RESET_ARM_JOINTS.tolist(), duration=4.0,
                    dynamic=False, buffer_time=0.2, block=True)
                time.sleep(0.5)
                print("  Arm at approach pose.")

            print("Press SPACE to start policy rollout...")
            while True:
                imgs = self._read_images()
                canvas = self._make_canvas(imgs,
                    f"Trial {trial_idx+1} - SPACE to start")
                cv2.imshow("Eval", canvas)
                if (cv2.waitKey(50) & 0xFF) == ord(" "):
                    break
        else:
            print("Robot at reset pose. Press SPACE to start policy rollout...")
            while True:
                imgs = self._read_images()
                canvas = self._make_canvas(imgs,
                    f"Trial {trial_idx+1}/{self.n_trials} - SPACE to start")
                cv2.imshow("Eval", canvas)
                if (cv2.waitKey(50) & 0xFF) == ord(" "):
                    break

        # Fill observation buffer
        state_buf = deque(maxlen=self.obs_horizon)
        img_buf = deque(maxlen=self.obs_horizon)
        for _ in range(self.obs_horizon):
            state_buf.append(self._read_state())
            img_buf.append(self._read_images())
            time.sleep(1.0 / CONTROL_FREQ)

        if not self.dry_run:
            self._start_live_control()

        dt = 1.0 / CONTROL_FREQ
        states_log = []
        actions_log = []
        step = 0
        aborted = False
        print(f"Running policy (max {self.max_steps} steps)...")

        # Prepare grasp type tensor for PC mode
        grasp_type_t = None
        if self.use_pc:
            grasp_type_t = torch.tensor(
                [self.grasp_type_id], dtype=torch.long, device=self.device)

        while step < self.max_steps:
            t0 = time.time()

            # Build observation tensors
            obs_states = np.stack([self._normalize_state(s) for s in state_buf])
            obs_s_t = torch.from_numpy(obs_states).unsqueeze(0).to(self.device)

            # Predict action chunk
            t_inf = time.time()
            with torch.no_grad():
                if self.use_pc:
                    pc_t = torch.from_numpy(scene_pc).unsqueeze(0).to(self.device)
                    action_chunk = self.policy.predict_action(
                        obs_s_t, pc_t, grasp_type_t,
                        num_inference_steps=self.num_inference_steps)
                else:
                    obs_imgs_np = self._build_image_tensor(img_buf)
                    obs_i_t = torch.from_numpy(obs_imgs_np).unsqueeze(0).to(self.device)
                    action_chunk = self.policy.predict_action(
                        obs_s_t, obs_i_t,
                        num_inference_steps=self.num_inference_steps)

            action_chunk = action_chunk.squeeze(0).cpu().numpy()
            action_chunk = self._unnormalize_action(action_chunk)
            inf_time = time.time() - t_inf
            if step == 0:
                print(f"  First inference: {inf_time:.2f}s")

            # Execute action_horizon steps from the chunk
            n_exec = min(self.action_horizon, self.max_steps - step)
            for i in range(n_exec):
                action_vec = action_chunk[min(i, len(action_chunk) - 1)]
                if not self.dry_run:
                    self._execute_action(action_vec)

                new_state = self._read_state()
                new_imgs = self._read_images()
                state_buf.append(new_state)
                img_buf.append(new_imgs)

                states_log.append(new_state.copy())
                actions_log.append(action_vec.copy())
                step += 1

                elapsed = time.time() - t0
                t0 = time.time()
                sleep_t = dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

                status = f"Step {step}/{self.max_steps}"
                if self.use_pc:
                    status += f"  [{self.grasp_type}]"
                canvas = self._make_canvas(new_imgs, status, color=(0, 0, 255))
                cv2.imshow("Eval", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord(" "):
                    aborted = (key == ord("q"))
                    step = self.max_steps
                    break

        if not self.dry_run:
            self._stop_live_control()

        # Pull test (only if --pull-dist was set)
        ratchet_info = None
        pull_angle_used = None
        if self.pull_dist is not None and not self.dry_run and self.fa is not None:
            pull_angle_used = getattr(self, "pull_angle", 180.0) or 180.0
            self._execute_pull(pull_angle_used)
            # Ratchet reading — operator enters tooth count, script computes displacement + force
            print("  Ratchet teeth (0–11, 0=no movement, Enter=skip): ", end="", flush=True)
            try:
                raw = input().strip()
                if raw.isdigit():
                    teeth = int(raw)
                    if 0 <= teeth <= 11:
                        disp_mm   = round(teeth * 9.3, 1)
                        disp_in   = disp_mm / 25.4
                        force_lbf = round(2 * (0.59 + 0.8 * disp_in), 3)
                        force_N   = round(force_lbf * 4.44822, 2)
                        ratchet_info = {
                            "teeth":           teeth,
                            "displacement_mm": disp_mm,
                            "force_lbf":       force_lbf,
                            "force_N":         force_N,
                        }
                        print(f"    → {disp_mm:.1f} mm  |  F ≈ {force_lbf:.2f} lbf ({force_N:.1f} N)")
            except (EOFError, ValueError):
                pass

        print("Trial done. Rate the grasp:  g = good,  b = bad,  s = skip")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("g"):
                rating = "good"
                break
            elif key == ord("b"):
                rating = "bad"
                break
            elif key == ord("s"):
                rating = "skip"
                break

        cv2.destroyAllWindows()

        result = {
            "trial": trial_idx,
            "hold_id": self.hold_id,
            "hold_name": HOLD_NAMES.get(self.hold_id, "unknown"),
            "rating": rating,
            "num_steps": len(states_log),
            "aborted": aborted,
            "timestamp": datetime.now().isoformat(),
            "encoder_type": self.encoder_type,
            "model": self.ckpt_name,
        }
        if self.pull_dist is not None:
            result["pull_dist_m"]    = self.pull_dist
            result["pull_angle_deg"] = pull_angle_used if pull_angle_used is not None else 180.0
            result["pull_stiffness"] = {
                "kx": self.pull_stiffness,
                "ky": self.pull_lateral_stiffness,
                "kz": self.pull_z_stiffness,
            }
            result["ratchet"] = ratchet_info
        if self.use_pc:
            result["grasp_type"] = self.grasp_type
            result["grasp_type_id"] = self.grasp_type_id
        print(f"  Result: {rating} ({len(states_log)} steps)")
        return result

    def _build_image_tensor(self, img_buf):
        """Convert img_buf (deque of dicts) → (T_o, n_cams, C, H, W) float32."""
        obs_imgs_list = []
        for img_dict in img_buf:
            cams = []
            for cam in self.cam_names:
                rgb = img_dict[cam].astype(np.float32) / 255.0
                rgb = np.transpose(rgb, (2, 0, 1))
                for ch_i in range(3):
                    rgb[ch_i] = (rgb[ch_i] - IMAGENET_MEAN[ch_i]) / IMAGENET_STD[ch_i]
                if self.use_rgbd:
                    d = img_dict.get(cam + "_depth",
                                     np.zeros((self.img_size, self.img_size), dtype=np.uint16))
                    d_norm = np.clip(d.astype(np.float32) * 0.001 / 2.0, 0, 1)[np.newaxis]
                    cams.append(np.concatenate([rgb, d_norm], axis=0))
                else:
                    cams.append(rgb)
            obs_imgs_list.append(np.stack(cams))
        return np.stack(obs_imgs_list).astype(np.float32)  # (T_o, n_cams, C, H, W)

    def _make_canvas(self, imgs, text, color=(0, 255, 0)):
        """Build a preview canvas from image dict."""
        panels = []
        for cam in self.cam_names:
            if cam in imgs:
                p = cv2.resize(imgs[cam], (224, 224))
                panels.append(p)
        if not panels:
            canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        elif len(panels) <= 2:
            canvas = np.hstack(panels)
        else:
            ncols = 2
            while len(panels) % ncols != 0:
                panels.append(np.zeros_like(panels[0]))
            rows = [np.hstack(panels[i:i + ncols]) for i in range(0, len(panels), ncols)]
            canvas = np.vstack(rows)
        cv2.putText(canvas, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return canvas

    def run(self):
        results = []
        for i in range(self.n_trials):
            result = self.run_trial(i)
            results.append(result)
            if result.get("aborted"):
                print("Evaluation aborted by user.")
                break

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.results_dir / f"eval_{ts}_{self.ckpt_name}_hold{self.hold_id}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 50}")
        print(f"Evaluation complete — {len(results)} trials")
        n_good = sum(1 for r in results if r["rating"] == "good")
        n_bad = sum(1 for r in results if r["rating"] == "bad")
        n_skip = sum(1 for r in results if r["rating"] == "skip")
        print(f"  Good: {n_good}  Bad: {n_bad}  Skip: {n_skip}")
        if n_good + n_bad > 0:
            print(f"  Success rate: {n_good / (n_good + n_bad) * 100:.0f}%")
        print(f"  Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion policy on robot")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hold", type=int, default=0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--max-joint-step", type=float, default=0.05,
                        help="Max radians per step per joint (motion clamping)")
    parser.add_argument("--dry-run", action="store_true",
                        help="No robot — cameras only")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--inference-steps", type=int, default=10,
                        help="DDIM inference steps at evaluation time (default 10). "
                             "10 DDIM steps ≈ 100 DDPM quality but ~10x faster — "
                             "required to stay within 10 Hz control loop. "
                             "Use 100 only for offline quality comparison.")
    parser.add_argument("--grasp-type", type=str, default=None,
                        choices=list(GRASP_TYPE_IDS.keys()),
                        help="Grasp type for point-cloud policy conditioning. "
                             "If omitted, will be prompted for PC checkpoints.")
    parser.add_argument("--zero-pc", action="store_true",
                        help="Feed all-zeros as the point cloud instead of capturing. "
                             "Use to test whether the model ignores the PC input.")
    parser.add_argument("--pull-dist", type=float, default=None,
                        help="If set, execute a spring-testbed pull test after each rollout: "
                             "arm moves this many meters toward robot base (180°, -X). "
                             "Standard value: 0.130 (13 cm). Ratchet range is 0–102.3 mm (11 teeth); "
                             "strong grips that travel past 102.3 mm read 11 teeth (≥33.9 N). "
                             "After the pull, enter the ratchet tooth count (0–11); "
                             "the script computes and logs displacement + slip force.")
    parser.add_argument("--pull-angle", type=float, default=180.0,
                        help="Direction of the pull in degrees (0=+X, 90=+Y, 180=-X toward robot). "
                             "Default 180° (spring testbed standard). "
                             "Only change for non-testbed use.")
    parser.add_argument("--pull-stiffness", type=float, default=4000.0,
                        help="X-axis (pull direction) Cartesian impedance stiffness in N/m. "
                             "Default 4000. At 4000 vs spring k≈280 N/m, arm reaches ~97 mm "
                             "for a perfect grip (≈10 ratchet teeth).")
    parser.add_argument("--pull-lateral-stiffness", type=float, default=100.0,
                        help="Y-axis (lateral) Cartesian impedance stiffness in N/m. "
                             "Default 100 — compliant, allows natural side-to-side hand flex.")
    parser.add_argument("--pull-z-stiffness", type=float, default=2000.0,
                        help="Z-axis (vertical) Cartesian impedance stiffness in N/m. "
                             "Default 2000. Must be high enough to hold the LEAP hand (~1 kg, "
                             "~10 N gravity load) against gravity without sagging; at 100 N/m "
                             "the arm droops ~10 cm. 2000 N/m → ~5 mm static droop, negligible.")
    parser.add_argument("--pull-z-bias", type=float, default=0.0,
                        help="Upward Z offset (meters) added to pull target pose. "
                             "Compensates for LEAP hand weight not in Franka gravity model. "
                             "Tune empirically: start at 0.02. (default: 0.0)")
    args = parser.parse_args()

    global MAX_JOINT_STEP_RAD
    MAX_JOINT_STEP_RAD = args.max_joint_step

    evaluator = PolicyEvaluator(
        ckpt_path=args.checkpoint,
        hold_id=args.hold,
        n_trials=args.trials,
        max_steps=args.max_steps,
        action_horizon=args.action_horizon,
        dry_run=args.dry_run,
        results_dir=args.results_dir,
        num_inference_steps=args.inference_steps,
        grasp_type=args.grasp_type,
        zero_pc=args.zero_pc,
        pull_dist=args.pull_dist,
        pull_stiffness=args.pull_stiffness,
        pull_lateral_stiffness=args.pull_lateral_stiffness,
        pull_z_stiffness=args.pull_z_stiffness,
        pull_z_bias=args.pull_z_bias,
    )
    # Store pull_angle on the evaluator for _execute_pull default override
    evaluator.pull_angle = args.pull_angle

    def _cleanup(signum=None, frame=None):
        # Called from SIGINT, exception, AND normal completion.
        print("\nCleaning up hardware...")

        def _do_cleanup():
            try:
                evaluator._stop_live_control()
            except Exception:
                pass
            try:
                if evaluator.fa is not None:
                    evaluator.fa.stop_skill()
            except Exception:
                pass
            try:
                if evaluator.leap_dxl is not None:
                    evaluator.leap_dxl.set_torque_enabled(evaluator._leap_motors, False)
            except Exception:
                pass
            cv2.destroyAllWindows()

        cleanup_thread = threading.Thread(target=_do_cleanup, daemon=True)
        cleanup_thread.start()
        cleanup_thread.join(timeout=5)
        if cleanup_thread.is_alive():
            print("Cleanup timed out — force exiting.")
        else:
            print("Cleanup done.")
        os._exit(0)

    signal.signal(signal.SIGINT, _cleanup)

    evaluator.setup()
    try:
        evaluator.run()
    except KeyboardInterrupt:
        _cleanup()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        _cleanup()
    else:
        # Normal completion: tear down pyrealsense / FrankaArm threads cleanly
        # so the C++ destructors don't throw "terminate called without an
        # active exception" and hang the process. _cleanup() calls os._exit(0).
        print("\nEvaluation complete.")
        _cleanup()


if __name__ == "__main__":
    main()
