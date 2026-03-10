#!/usr/bin/env python3
"""
Evaluate a trained diffusion policy on the real robot.

Loads a checkpoint, connects to the Franka arm + LEAP hand + cameras,
runs the policy in a loop, and logs the results for human review.

Workflow per trial:
  1. Robot moves to a fixed reset pose (manual or automatic).
  2. Policy observes for obs_horizon steps to fill the observation buffer.
  3. Policy predicts a pred_horizon action chunk (joint position targets).
  4. First action_horizon steps of the chunk are executed on the robot.
  5. Steps 2-4 repeat until the episode ends (max steps or human stop).
  6. Human rates the grasp (good / bad) via keyboard.

Usage:
    python3 evaluate.py --checkpoint ../checkpoints/best.pt
    python3 evaluate.py --checkpoint ../checkpoints/best.pt --hold 0 --trials 10
    python3 evaluate.py --checkpoint ../checkpoints/best.pt --dry-run  # no robot
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

from episode_storage import resize_image
from hold_detector import detect_hold_multi_camera

import signal

CAMERA_NUMBERS = [2, 3, 4, 5]
CAMERA_RAW_W = 848
CAMERA_RAW_H = 480
CONTROL_FREQ = 10

MAX_JOINT_STEP_RAD = 0.03

HOLD_NAMES = {0: "edge_A", 1: "edge_B", 2: "sloper", 3: "pinch", 4: "test_edge"}

DEFAULT_RESULTS_DIR = TELE_ROOT / "eval_results"

RESET_ARM_JOINTS = np.array([-0.1111, -0.1703, -0.0621, -2.3442, 0.0408, 2.1952, 0.1559],
                            dtype=np.float32)
RESET_HAND_ALLEGRO = np.array([0.078, 0.31, 0.41, 0.25, 0.14, 0.46, 0.50, 0.30,
                               0.43, 0.56, 0.55, 0.30, 0.17, -0.47, -0.10, 1.22],
                              dtype=np.float32)


def load_policy(ckpt_path, device):
    from train import DiffusionPolicy
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt["config"]

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
    norm_stats = {
        "state_mean": np.array(norm["state_mean"], dtype=np.float32),
        "state_std": np.array(norm["state_std"], dtype=np.float32),
        "action_mean": np.array(norm["action_mean"], dtype=np.float32),
        "action_std": np.array(norm["action_std"], dtype=np.float32),
    }
    return policy, cfg, norm_stats


def build_state_vector(arm_joints, ee_pos, ee_quat, hand_joints, hold_pose=None):
    parts = [
        np.array(arm_joints, dtype=np.float32).ravel()[:7],
        np.array(ee_pos, dtype=np.float32).ravel()[:3],
        np.array(ee_quat, dtype=np.float32).ravel()[:4],
        np.array(hand_joints, dtype=np.float32).ravel()[:16],
    ]
    if hold_pose is not None:
        parts.append(np.array(hold_pose, dtype=np.float32).ravel()[:6])
    return np.concatenate(parts)


class PolicyEvaluator:
    def __init__(self, ckpt_path, hold_id=0, n_trials=5, max_steps=200,
                 action_horizon=8, dry_run=False, results_dir=DEFAULT_RESULTS_DIR,
                 num_inference_steps=100):
        self.hold_id = hold_id
        self.n_trials = n_trials
        self.max_steps = max_steps
        self.action_horizon = action_horizon
        self.dry_run = dry_run
        self.num_inference_steps = num_inference_steps
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading policy from {ckpt_path}...")
        self.policy, self.cfg, self.norm = load_policy(ckpt_path, self.device)
        self.img_size = self.cfg.get("img_size", 224)
        self.img_channels = self.cfg.get("img_channels", 3)
        self.use_rgbd = (self.img_channels == 4)
        print(f"  state_dim={self.cfg['state_dim']}, action_dim={self.cfg['action_dim']}, "
              f"obs_horizon={self.cfg['obs_horizon']}, pred_horizon={self.cfg['pred_horizon']}, "
              f"img_size={self.img_size}, img_channels={self.img_channels}")

        # Warmup CUDA kernels with a dummy forward pass
        print("  Warming up GPU (one-time)...", end=" ", flush=True)
        n_cams = len(self.cfg["cam_names"])
        dummy_s = torch.zeros(1, self.cfg["obs_horizon"], self.cfg["state_dim"],
                              device=self.device)
        dummy_i = torch.zeros(1, self.cfg["obs_horizon"], n_cams,
                              self.img_channels, self.img_size, self.img_size,
                              device=self.device)
        with torch.no_grad():
            self.policy.predict_action(dummy_s, dummy_i, num_inference_steps=self.num_inference_steps)
        print("done")

        self.cam_names = self.cfg["cam_names"]
        self.obs_horizon = self.cfg["obs_horizon"]
        self.pred_horizon = self.cfg["pred_horizon"]

        self.fa = None
        self.leap_dxl = None
        self.cameras = None
        self.hold_pose = np.zeros(6, dtype=np.float32)
        self._live_active = False
        self._ros_pub = None
        self._ros_id = 0
        self._live_init_time = 0

    def setup(self):
        self._setup_franka()
        self._setup_leap()
        self._setup_cameras()
        if self.cfg.get("state_dim", 30) > 30:
            self.scan_hold_pose()
        print(f"\nReady to evaluate on hold {self.hold_id} "
              f"({HOLD_NAMES.get(self.hold_id, '?')}), {self.n_trials} trials\n")

    def _setup_franka(self):
        if self.dry_run:
            print("[Franka] dry-run, skipped")
            return
        from frankapy import FrankaArm
        self.fa = FrankaArm(with_gripper=False)
        # Clear any lingering skill from a previous crashed run
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
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 800, 84, 2)  # kP
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 200, 80, 2)  # kD
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 600, 102, 2)  # curr_lim
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

    def scan_hold_pose(self, cam_index=0, n_samples=5):
        """Detect hold 3D pose using depth from a designated camera."""
        print(f"\nScanning hold pose from camera {self._cam_nums[cam_index]}...")
        poses = []
        for _ in range(n_samples):
            raw = self.cameras.get_next_frames()
            pose, n_pts = detect_hold_multi_camera(raw, cam_index=cam_index)
            if n_pts > 50:
                poses.append(pose)
            time.sleep(0.1)

        if not poses:
            print("  WARNING: No hold detected — using zeros.")
            self.hold_pose = np.zeros(6, dtype=np.float32)
        else:
            self.hold_pose = np.mean(poses, axis=0).astype(np.float32)
            c, n = self.hold_pose[:3], self.hold_pose[3:]
            print(f"  Hold detected ({len(poses)}/{n_samples} frames):")
            print(f"    Centroid: [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}] m")
            print(f"    Normal:   [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
        return self.hold_pose

    def _read_state(self):
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

        hp = self.hold_pose if self.cfg.get("state_dim", 30) > 30 else None
        return build_state_vector(joints, ee_pos, ee_quat, hand, hold_pose=hp)

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
        return (s - self.norm["state_mean"]) / self.norm["state_std"]

    def _unnormalize_action(self, a):
        return a * self.norm["action_std"] + self.norm["action_mean"]

    def _start_live_control(self):
        """Start a long-running dynamic joint skill for streaming targets."""
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
        """Stop the live dynamic skill."""
        if not self._live_active:
            return
        try:
            self.fa.stop_skill()
        except Exception:
            pass
        self._live_active = False
        time.sleep(1.5)  # frankapy action server needs time to fully reset
        print("[Arm] Live joint control stopped")

    def _send_joint_target(self, joints):
        """Send a joint position update to the running dynamic skill."""
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
        """Send a 36-dim action to the robot (arm joints + hand joints).
        Layout: arm(7) + ee_pos(3) + ee_quat(4) + hand(16) + hold_pose(6)
        Only arm joints [0:7] and hand joints [14:30] are actuated."""
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

    def _reset_robot(self, timeout=15):
        """Move arm and hand to a safe reset pose."""
        print("Resetting robot to home pose...")
        just_stopped = False
        if self._live_active:
            self._stop_live_control()
            just_stopped = True
        if self.fa is not None:
            if not just_stopped:
                # Only stop skill if _stop_live_control didn't just do it
                try:
                    self.fa.stop_skill()
                except Exception:
                    pass
                time.sleep(1.0)
            # Run goto_joints in a thread with timeout to avoid hanging forever
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

        print("Robot at reset pose. Press SPACE to start policy rollout...")
        while True:
            imgs = self._read_images()
            panels = []
            for cam in self.cam_names:
                p = cv2.resize(imgs[cam], (224, 224))
                panels.append(p)
            if len(panels) <= 2:
                canvas = np.hstack(panels)
            else:
                ncols = 2
                while len(panels) % ncols != 0:
                    panels.append(np.zeros_like(panels[0]))
                rows = [np.hstack(panels[i:i + ncols]) for i in range(0, len(panels), ncols)]
                canvas = np.vstack(rows)
            cv2.putText(canvas, f"Trial {trial_idx+1}/{self.n_trials} - SPACE to start",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
        while step < self.max_steps:
            t0 = time.time()

            # Build observation tensors
            obs_states = np.stack([self._normalize_state(s) for s in state_buf])
            obs_imgs_list = []
            for img_dict in img_buf:
                cams = []
                for cam in self.cam_names:
                    rgb = img_dict[cam].astype(np.float32) / 255.0
                    rgb = np.transpose(rgb, (2, 0, 1))  # (3, H, W)
                    for ch_i in range(3):
                        rgb[ch_i] = (rgb[ch_i] - IMAGENET_MEAN[ch_i]) / IMAGENET_STD[ch_i]
                    if self.use_rgbd:
                        d = img_dict.get(cam + "_depth",
                                         np.zeros((self.img_size, self.img_size), dtype=np.uint16))
                        d_norm = np.clip(d.astype(np.float32) * 0.001 / 2.0, 0, 1)[np.newaxis]
                        cams.append(np.concatenate([rgb, d_norm], axis=0))  # (4, H, W)
                    else:
                        cams.append(rgb)
                obs_imgs_list.append(np.stack(cams))
            obs_imgs = np.stack(obs_imgs_list)

            obs_s_t = torch.from_numpy(obs_states).unsqueeze(0).to(self.device)
            obs_i_t = torch.from_numpy(obs_imgs).unsqueeze(0).to(self.device)

            # Predict action chunk
            t_inf = time.time()
            with torch.no_grad():
                action_chunk = self.policy.predict_action(obs_s_t, obs_i_t, num_inference_steps=self.num_inference_steps)
            action_chunk = action_chunk.squeeze(0).cpu().numpy()  # (Tp, D)
            action_chunk = self._unnormalize_action(action_chunk)
            inf_time = time.time() - t_inf
            if step == 0:
                print(f"  First inference took {inf_time:.1f}s")
            elif step <= self.action_horizon:
                print(f"  Inference: {inf_time:.2f}s")

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

                # Show live feed
                panels = [cv2.resize(new_imgs[c], (224, 224)) for c in self.cam_names]
                if len(panels) <= 2:
                    canvas = np.hstack(panels)
                else:
                    ncols = 2
                    while len(panels) % ncols != 0:
                        panels.append(np.zeros_like(panels[0]))
                    rows = [np.hstack(panels[j:j + ncols]) for j in range(0, len(panels), ncols)]
                    canvas = np.vstack(rows)

                status = f"Step {step}/{self.max_steps}"
                cv2.putText(canvas, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Eval", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord(" "):
                    aborted = (key == ord("q"))
                    step = self.max_steps
                    break

        if not self.dry_run:
            self._stop_live_control()

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
        }
        print(f"  Result: {rating} ({len(states_log)} steps)")
        return result

    def run(self):
        results = []
        for i in range(self.n_trials):
            result = self.run_trial(i)
            results.append(result)
            if result.get("aborted"):
                print("Evaluation aborted by user.")
                break

        # Save results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.results_dir / f"eval_{ts}_hold{self.hold_id}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        # Summary
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
    parser.add_argument("--inference-steps", type=int, default=100,
                        help="Inference steps (default 100 = full DDPM, use fewer for DDIM)")
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
    )

    def _cleanup(signum=None, frame=None):
        print("\nCaught interrupt — cleaning up...")
        # Run cleanup in a thread so we can force-exit if it hangs
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


if __name__ == "__main__":
    main()
