#!/usr/bin/env python3
"""
Paired evaluation: with-taxonomy vs. without-taxonomy diffusion policies.

Loads BOTH checkpoints once at startup, shares all robot hardware, and
runs interlaced paired trials. A fresh point cloud is captured BEFORE
EACH trial (not once per pair) because the hand from trial 1 will
typically nudge the hold even when the operator is careful; between
trials the operator is prompted to re-align the hold and we re-scan.
Per-trial PC centroids + point counts are logged to the JSON so any
drift between the two trials of a pair is auditable post-hoc.

The model that goes first in pair 1 is randomized; afterwards the order
strictly alternates across ALL pairs in the session (not just within a
single grasp-type batch) so each model goes first half the time.

The session is organised into BATCHES. Each batch fixes (grasp_type,
hold_id) for a user-chosen number of pairs — this lets you test, e.g.,
5 crimps, then 5 jugs, then 5 slopers, then 5 pinches without restarting
the script. Every pair is logged with its grasp_type + hold_id so the
no-taxonomy model's per-grasp-type performance is still measurable even
though it ignores the label.

A full statistical analysis is printed at the end (overall + per
grasp type), with success rates, Wilson 95% CIs, and McNemar's paired
test, and all raw pairs are saved to one JSON file for later plotting.

Usage:
    # Interactive: prompts for grasp_type / hold / pairs before each batch.
    python3 paired_eval.py

    # Single batch (backwards-compatible with the old one-shot mode):
    python3 paired_eval.py --hold 1 --grasp-type crimp --pairs 10

    # Scripted multi-batch (format: grasp:hold:pairs,grasp:hold:pairs,...):
    python3 paired_eval.py --batches crimp:1:5,jug:0:5,sloper:2:5,pinch:3:5

    # Dry-run (no robot, cameras only — for testing script flow):
    python3 paired_eval.py --dry-run

Hold mapping: 0=edge_A/jug  1=edge_B/crimp  2=sloper  3=pinch  4=test_edge
"""

import os
import sys
import math
import time
import json
import random
import signal
import argparse
import threading
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import torch
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
TELE_ROOT  = SCRIPT_DIR.parent

FRANKA_SCRIPTS_DIR = TELE_ROOT / "TeleoperationUnity" / "Robot Control - Python" / "Franka Scripts"
LEAP_DIR           = TELE_ROOT / "TeleoperationUnity" / "LEAP" / "leaphandv1" / "for_transfer"
LEAP_API_DIR       = LEAP_DIR / "LEAP_Hand_API" / "python"

for p in [str(FRANKA_SCRIPTS_DIR), str(LEAP_DIR), str(LEAP_API_DIR), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from episode_storage import GRASP_TYPE_IDS, GRASP_TYPE_NAMES, N_GRASP_TYPES

# ── Hardware constants ──────────────────────────────────────────────────────
CAMERA_NUMBERS    = [2, 3, 4, 5]
CAMERA_RAW_W      = 848
CAMERA_RAW_H      = 480
CONTROL_FREQ      = 10          # Hz
PC_N_POINTS       = 1024
PC_CAPTURE_FRAMES = 5
MAX_JOINT_STEP    = 0.05        # rad per step

PARK_ARM_JOINTS = np.array(
    [-0.11426599, -0.56029082, -0.06635159, -2.17443357,
      0.04112932,  2.15592909,  0.54378958], dtype=np.float64)

RESET_ARM_JOINTS = np.array(
    [-0.1111, -0.1703, -0.0621, -2.3442,
      0.0408,  2.1952,  0.1559], dtype=np.float32)

RESET_HAND_ALLEGRO = np.array(
    [0.078, 0.31, 0.41, 0.25,
     0.14,  0.46, 0.50, 0.30,
     0.43,  0.56, 0.55, 0.30,
     0.17, -0.47,-0.10, 1.22], dtype=np.float32)

HOLD_NAMES = {0: "edge_A", 1: "edge_B", 2: "sloper", 3: "pinch", 4: "test_edge"}

DEFAULT_RESULTS_DIR = TELE_ROOT / "eval_results"
DEFAULT_WITH_CKPT   = TELE_ROOT / "checkpoints" / "pc_with_taxonomy" / "best.pt"
DEFAULT_NO_CKPT     = TELE_ROOT / "checkpoints" / "pc_no_taxonomy"  / "best.pt"

MODEL_WITH = "WITH_TAXONOMY"
MODEL_NO   = "WITHOUT_TAXONOMY"


class QuitRequested(Exception):
    """Raised when the operator requests a clean save-and-exit at a prompt."""
    pass


# ── Policy loader ──────────────────────────────────────────────────────────

def load_policy(ckpt_path, device):
    """Load a PointCloudDiffusionPolicy from a checkpoint file."""
    from train import PointCloudDiffusionPolicy
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    encoder_type = cfg.get("encoder_type", "vision")
    if encoder_type != "point_cloud":
        raise ValueError(
            f"paired_eval only supports point_cloud checkpoints; got {encoder_type!r}")

    policy = PointCloudDiffusionPolicy(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        obs_horizon=cfg["obs_horizon"],
        pred_horizon=cfg["pred_horizon"],
        num_diffusion_steps=cfg["diffusion_steps"],
        down_dims=tuple(cfg.get("down_dims", [256, 512, 1024])),
        n_grasp_types=cfg.get("n_grasp_types", N_GRASP_TYPES),
        use_grasp_conditioning=cfg.get("use_grasp_conditioning", True),
    ).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    norm_path = Path(ckpt_path).parent / "norm_stats.json"
    with open(norm_path) as f:
        raw = json.load(f)

    if raw.get("normalization", "zscore") == "minmax":
        norm = {
            "normalization": "minmax",
            "state_min":    np.array(raw["state_min"],    dtype=np.float32),
            "state_range":  np.array(raw["state_range"],  dtype=np.float32),
            "action_min":   np.array(raw["action_min"],   dtype=np.float32),
            "action_range": np.array(raw["action_range"], dtype=np.float32),
        }
    else:
        norm = {
            "normalization": "zscore",
            "state_mean": np.array(raw["state_mean"], dtype=np.float32),
            "state_std":  np.array(raw["state_std"],  dtype=np.float32),
            "action_mean": np.array(raw["action_mean"], dtype=np.float32),
            "action_std":  np.array(raw["action_std"],  dtype=np.float32),
        }
    return policy, cfg, norm


# ── Statistics ────────────────────────────────────────────────────────────

def wilson_ci(n_success, n_total, z=1.96):
    """Wilson score 95% confidence interval. Returns (proportion, (lo, hi))."""
    if n_total == 0:
        return 0.0, (0.0, 1.0)
    p     = n_success / n_total
    denom  = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt(p*(1-p)/n_total + z**2/(4*n_total**2)) / denom
    return p, (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_test(n01, n10):
    """
    McNemar's test for paired binary outcomes.
      n01 = pairs where WITH succeeded, WITHOUT failed
      n10 = pairs where WITH failed,   WITHOUT succeeded
    Returns (chi2_stat, p_value).  chi2_stat is None when using exact test.
    """
    n = n01 + n10
    if n == 0:
        return None, None
    if n < 10:
        # Exact binomial two-tailed
        from math import comb
        larger = max(n01, n10)
        p = 2 * sum(comb(n, k) * 0.5**n for k in range(larger, n + 1))
        return None, min(1.0, p)
    # Continuity-corrected chi-squared
    chi2 = (abs(n01 - n10) - 1.0)**2 / n
    try:
        from scipy.stats import chi2 as chi2_dist
        p_val = float(1 - chi2_dist.cdf(chi2, df=1))
    except ImportError:
        # Fallback: erfc approximation for chi2(1)
        p_val = math.erfc(math.sqrt(chi2 / 2))
    return chi2, p_val


# ── Main evaluator class ───────────────────────────────────────────────────

class PairedEvaluator:

    def __init__(self, with_ckpt, no_ckpt,
                 max_steps=200, action_horizon=8,
                 dry_run=False, results_dir=DEFAULT_RESULTS_DIR,
                 num_inference_steps=10,
                 pull_dist=None, pull_stiffness=4000.0,
                 pull_lateral_stiffness=100.0, pull_z_stiffness=100.0,
                 pull_z_bias=0.0):

        # Batch state — set at start of each batch via _set_batch()
        self.hold_id            = None
        self.grasp_type         = None
        self.grasp_type_id      = None

        self.max_steps          = max_steps
        self.action_horizon     = action_horizon
        self.dry_run            = dry_run
        self.num_inf_steps      = num_inference_steps
        self.pull_dist              = pull_dist              # meters; None = pull test disabled
        self.pull_stiffness         = pull_stiffness         # N/m; X (pull axis) stiffness
        self.pull_lateral_stiffness = pull_lateral_stiffness  # N/m; Y (lateral) stiffness
        self.pull_z_stiffness       = pull_z_stiffness       # N/m; Z (vertical) stiffness
        self.pull_z_bias            = pull_z_bias            # meters upward Z bias to counter LEAP hand weight
        self.results_dir        = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Global session state (persists across batches)
        self._global_pair_idx   = 0          # strict alternation spans all batches
        self._first_model       = None       # randomised on the very first pair
        self._all_pairs         = []         # accumulates every pair in the session
        self._session_id        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._batch_configs     = []         # per-batch state incl. completed_pairs
        self._planned_batches   = []         # scripted-mode plan (unchanged once set)
        self._mode              = None       # "scripted" / "interactive" / "single"
        self._resumed           = False
        self._with_ckpt_path    = str(with_ckpt)
        self._no_ckpt_path      = str(no_ckpt)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\nLoading WITH-taxonomy policy  ({with_ckpt}) ...")
        self.with_policy, self.with_cfg, self.with_norm = load_policy(with_ckpt, self.device)
        assert self.with_cfg.get("use_grasp_conditioning", True), \
            "--with-ckpt must have use_grasp_conditioning=True"

        print(f"Loading WITHOUT-taxonomy policy ({no_ckpt}) ...")
        self.no_policy, self.no_cfg, self.no_norm = load_policy(no_ckpt, self.device)
        assert not self.no_cfg.get("use_grasp_conditioning", True), \
            "--no-ckpt must have use_grasp_conditioning=False"

        self.state_dim   = self.with_cfg["state_dim"]
        self.obs_horizon = self.with_cfg["obs_horizon"]
        self.cam_names   = self.with_cfg.get("cam_names",
                               [f"cam{n}" for n in CAMERA_NUMBERS])

        print("Warming up GPU for both policies ...", end=" ", flush=True)
        self._warmup(self.with_policy)
        self._warmup(self.no_policy)
        print("done")

        # Hardware handles (populated in setup())
        self.fa               = None
        self.leap_dxl         = None
        self._lhu             = None
        self._leap_motors     = None
        self.cameras          = None
        self._cam_intrinsics  = None
        self._cam_extrinsics  = None
        self._live_active     = False
        self._ros_pub         = None
        self._ros_id          = 0
        self._live_init_time  = 0

    # ── GPU warmup ──────────────────────────────────────────────────────────

    def _warmup(self, policy):
        with torch.no_grad():
            dummy_s  = torch.zeros(1, self.obs_horizon, self.state_dim, device=self.device)
            dummy_pc = torch.zeros(1, PC_N_POINTS, 3, device=self.device)
            dummy_gt = torch.zeros(1, dtype=torch.long, device=self.device)
            policy.predict_action(dummy_s, dummy_pc, dummy_gt,
                                  num_inference_steps=self.num_inf_steps)

    # ── Hardware setup ──────────────────────────────────────────────────────

    def setup(self):
        if not self.dry_run:
            self._setup_franka()
            self._setup_leap()
        self._setup_cameras()
        print(f"\nHardware ready. Session ID: {self._session_id}\n")

    def _set_batch(self, grasp_type, hold_id):
        self.grasp_type    = grasp_type
        self.grasp_type_id = GRASP_TYPE_IDS[grasp_type]
        self.hold_id       = hold_id

    def _wait_or_quit(self, prompt, quit_hint=True):
        """Block until the user presses Enter, or types q/quit to save & exit.
        Raises QuitRequested on quit. Ctrl-C is also treated as quit."""
        suffix = "  (Enter = continue, q = save & quit)" if quit_hint else ""
        try:
            raw = input(prompt + suffix + "\n").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise QuitRequested()
        if raw in ("q", "quit", "exit"):
            raise QuitRequested()

    def _load_resume(self, path):
        """Restore session state from a previously-saved paired_session JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        self._session_id      = data.get("session_id",
                                         datetime.now().strftime("%Y%m%d_%H%M%S"))
        self._first_model     = data.get("first_model")
        self._all_pairs       = data.get("pairs", [])
        self._batch_configs   = data.get("batches", [])
        self._planned_batches = [tuple(b) for b in data.get("planned_batches", [])]
        self._mode            = data.get("mode", self._mode)
        self._global_pair_idx = len(self._all_pairs)
        self._resumed         = True

        # Ensure each saved batch config has completed_pairs (for older files)
        by_idx = {bc.get("batch", i): bc for i, bc in enumerate(self._batch_configs)}
        counts = {}
        for r in self._all_pairs:
            counts[r.get("batch", 0)] = counts.get(r.get("batch", 0), 0) + 1
        for bidx, bc in by_idx.items():
            if "completed_pairs" not in bc:
                bc["completed_pairs"] = counts.get(bidx, 0)

        print(f"\n  Resumed session {self._session_id}")
        print(f"    Pairs already recorded : {len(self._all_pairs)}")
        print(f"    First model (pair 1)   : {self._first_model}")
        if self._planned_batches:
            print(f"    Planned batches        :")
            for i, spec in enumerate(self._planned_batches):
                gt, hid, n = spec[0], spec[1], spec[2]
                orient = spec[3] if len(spec) > 3 else None
                orient_tag = f" @ {orient}°" if orient is not None else ""
                done = by_idx.get(i, {}).get("completed_pairs", 0)
                marker = "✓" if done >= n else ("…" if done > 0 else " ")
                print(f"      [{marker}] batch {i+1}: {gt}/hold={hid}{orient_tag}  "
                      f"{done}/{n} pairs done")

    def _setup_franka(self):
        from frankapy import FrankaArm
        self.fa = FrankaArm(with_gripper=False)
        try:
            self.fa.stop_skill()
        except Exception:
            pass
        time.sleep(0.5)
        print("[Franka] connected")

    def _setup_leap(self):
        from leap_pip_dip_teleop import find_leap_port
        from leap_hand_utils.dynamixel_client import DynamixelClient
        import leap_hand_utils.leap_hand_utils as lhu
        self._lhu         = lhu
        self._leap_motors = list(range(16))
        port = find_leap_port()
        if port is None:
            print("[LEAP] WARNING: no device found — hand commands skipped")
            return
        try:
            self.leap_dxl = DynamixelClient(self._leap_motors, port, 4000000)
            self.leap_dxl.connect()
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 5,   11, 1)
            self.leap_dxl.set_torque_enabled(self._leap_motors, True)
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 800, 84, 2)
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 200, 80, 2)
            self.leap_dxl.sync_write(self._leap_motors, np.ones(16) * 600, 102, 2)
            print(f"[LEAP] connected on {port}")
        except Exception as e:
            print(f"[LEAP] WARNING: connection failed ({e})")
            self.leap_dxl = None

    def _setup_cameras(self):
        import robomail.vision as vis
        cam_nums = [int(c.replace("cam", "")) for c in self.cam_names]
        self._cam_nums = cam_nums
        self.cameras = vis.ThreadedCameras(
            cam_numbers=cam_nums,
            image_height=CAMERA_RAW_H, image_width=CAMERA_RAW_W,
            get_point_cloud=False, get_verts=False)
        self.cameras.get_next_frames()
        print(f"[Cameras] {cam_nums} streaming")
        try:
            from point_cloud_utils import (get_cam_intrinsics_from_realsense,
                                            get_cam_extrinsics_from_realsense)
            self._cam_intrinsics = [get_cam_intrinsics_from_realsense(c)
                                     for c in self.cameras.cameras]
            self._cam_extrinsics = [get_cam_extrinsics_from_realsense(c)
                                     for c in self.cameras.cameras]
            print(f"[Cameras] calibration loaded ({len(self._cam_intrinsics)} cameras)")
        except Exception as e:
            print(f"[Cameras] WARNING: calibration failed ({e})")

    # ── Robot actions ───────────────────────────────────────────────────────

    def _capture_clean_point_cloud(self):
        from point_cloud_utils import fuse_multi_camera_points
        if self._cam_intrinsics is None:
            print("  WARNING: no calibration — returning zero PC")
            return np.zeros((PC_N_POINTS, 3), dtype=np.float32)
        pcs = []
        for fi in range(PC_CAPTURE_FRAMES):
            frames = self.cameras.get_next_frames()
            depths = []
            for i in range(len(self._cam_nums)):
                d = frames[i][1]
                if d is None or d.size == 0:
                    d = np.zeros((CAMERA_RAW_H, CAMERA_RAW_W), dtype=np.uint16)
                depths.append(d)
            pc = fuse_multi_camera_points(
                depth_images=depths,
                cam_intrinsics=self._cam_intrinsics,
                cam_extrinsics=self._cam_extrinsics,
                n_points=PC_N_POINTS,
                outlier_removal=(fi == 0))
            pcs.append(pc)
            time.sleep(0.05)
        return pcs[-1]

    def _park_and_capture_pc(self):
        """Park arm → capture PC → return arm to approach pose. Returns PC array."""
        if self.fa is not None:
            print("  Parking arm for point cloud capture ...")
            try:
                self.fa.stop_skill()
            except Exception:
                pass
            time.sleep(0.3)
            self.fa.goto_joints(PARK_ARM_JOINTS.tolist(), duration=4.0,
                                dynamic=False, buffer_time=0.2, block=True)
            time.sleep(1.5)

        print("  Capturing point cloud ...", end=" ", flush=True)
        pc = self._capture_clean_point_cloud()
        n_valid = int(np.sum(np.any(pc != 0, axis=-1)))
        print(f"done  ({n_valid}/{PC_N_POINTS} valid pts)")

        if self.fa is not None:
            print("  Returning arm to approach pose ...")
            try:
                self.fa.stop_skill()
            except Exception:
                pass
            time.sleep(0.3)
            self.fa.goto_joints(RESET_ARM_JOINTS.tolist(), duration=4.0,
                                dynamic=False, buffer_time=0.2, block=True)
            time.sleep(0.5)
        return pc

    def _execute_pull(self):
        """Pull arm 10.5 cm toward robot base (180° = -X direction).

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
        # Spring testbed: pull is always 180° (-X, toward robot base)
        dx = -self.pull_dist
        dy = 0.0
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
        print(f"  [pull] {self.pull_dist*100:.1f} cm toward robot base "
              f"[kx={kx:.0f} / ky={ky:.0f} / kz={kz:.0f} N/m] ...")
        try:
            self.fa.goto_pose(target_pose, duration=5.0, dynamic=False,
                              buffer_time=0.2, use_impedance=True,
                              cartesian_impedances=[kx, ky, kz, 10.0, 10.0, 10.0])
        except Exception as e:
            print(f"  [pull] WARNING: move failed: {e}")

    def _reset_robot(self):
        """Bring arm + hand back to the start-of-trial pose."""
        if self._live_active:
            self._stop_live_control()
        if self.fa is not None:
            try:
                self.fa.stop_skill()
            except Exception:
                pass
            time.sleep(0.8)
            done = threading.Event()

            def _go():
                try:
                    self.fa.goto_joints(RESET_ARM_JOINTS.tolist(), duration=5.0,
                                        dynamic=False, buffer_time=0.2, block=True)
                except Exception as e:
                    print(f"  WARNING: reset error: {e}")
                finally:
                    done.set()

            t = threading.Thread(target=_go, daemon=True)
            t.start()
            if not done.wait(15):
                print("  WARNING: reset timed out")
                try:
                    self.fa.stop_skill()
                except Exception:
                    pass
            time.sleep(0.3)
        if self.leap_dxl is not None:
            leap_t = self._lhu.allegro_to_LEAPhand(RESET_HAND_ALLEGRO, zeros=False)
            self.leap_dxl.write_desired_pos(self._leap_motors, leap_t)
            time.sleep(0.3)

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
        self._live_active    = True
        time.sleep(0.1)

    def _stop_live_control(self):
        if not self._live_active:
            return
        try:
            self.fa.stop_skill()
        except Exception:
            pass
        self._live_active = False
        time.sleep(1.5)

    def _send_joint_target(self, joints):
        import rospy
        from frankapy import SensorDataMessageType
        from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
        from frankapy.proto import JointPositionSensorMessage
        from franka_interface_msgs.msg import SensorDataGroup
        ts = rospy.Time.now().to_time() - self._live_init_time
        msg = JointPositionSensorMessage(id=self._ros_id, timestamp=ts,
                                          joints=np.array(joints).tolist())
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                msg, SensorDataMessageType.JOINT_POSITION))
        self._ros_pub.publish(ros_msg)
        self._ros_id += 1

    def _execute_action(self, action_vec):
        arm_target  = np.array(action_vec[:7],  dtype=np.float64)
        hand_target = action_vec[7:23]
        if self.fa is not None and self._live_active:
            cur = np.array(self.fa.get_joints(), dtype=np.float64)
            clamped = cur + np.clip(arm_target - cur, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            self._send_joint_target(clamped)
        if self.leap_dxl is not None:
            try:
                leap_t = self._lhu.allegro_to_LEAPhand(hand_target, zeros=False)
                self.leap_dxl.write_desired_pos(self._leap_motors, leap_t)
            except Exception:
                pass

    # ── State / image reads ─────────────────────────────────────────────────

    def _read_state(self):
        joints = (np.array(self.fa.get_joints(), dtype=np.float32)
                  if self.fa is not None else np.zeros(7, dtype=np.float32))
        if self.leap_dxl is not None:
            try:
                raw  = self.leap_dxl.read_pos()
                hand = self._lhu.LEAPhand_to_allegro(raw, zeros=False).astype(np.float32)
            except Exception:
                hand = np.zeros(16, dtype=np.float32)
        else:
            hand = np.zeros(16, dtype=np.float32)
        return np.concatenate([joints[:7], hand[:16]])

    def _read_images(self):
        from episode_storage import resize_image
        img_size = self.with_cfg.get("img_size", 224)
        raw  = self.cameras.get_frames()
        imgs = {}
        for i, name in enumerate(self.cam_names):
            imgs[name] = resize_image(raw[i][0], img_size, img_size)
        return imgs

    def _normalize_state(self, s, norm):
        if norm["normalization"] == "minmax":
            return 2.0 * (s - norm["state_min"]) / norm["state_range"] - 1.0
        return (s - norm["state_mean"]) / norm["state_std"]

    def _unnormalize_action(self, a, norm):
        if norm["normalization"] == "minmax":
            return (a + 1.0) / 2.0 * norm["action_range"] + norm["action_min"]
        return a * norm["action_std"] + norm["action_mean"]

    # ── Display ─────────────────────────────────────────────────────────────

    def _make_canvas(self, imgs, line1, line2="", model_label=""):
        panels = [cv2.resize(imgs[c], (224, 224))
                  for c in self.cam_names if c in imgs]
        if not panels:
            canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        elif len(panels) <= 2:
            canvas = np.hstack(panels)
        else:
            while len(panels) % 2:
                panels.append(np.zeros_like(panels[0]))
            canvas = np.vstack([np.hstack(panels[i:i+2])
                                for i in range(0, len(panels), 2)])

        color = (0, 220, 0) if model_label == MODEL_WITH else (0, 160, 255)
        cv2.putText(canvas, line1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if line2:
            cv2.putText(canvas, line2, (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        return canvas

    # ── Single trial ────────────────────────────────────────────────────────

    def _run_single_trial(self, policy, norm, cfg, grasp_type_id, scene_pc,
                           model_label, pair_idx, overlay_tag=""):
        """
        Execute one model's rollout. Arm must already be at RESET_ARM_JOINTS.
        Uses scene_pc captured just before this trial.
        `overlay_tag` is a short string drawn in the camera overlay (e.g.
        "B2 pair 3/5 [crimp]") so the operator knows where they are.
        Returns (rating, aborted).
        """
        obs_horizon = cfg["obs_horizon"]
        cv2.namedWindow("PairedEval", cv2.WINDOW_NORMAL)

        header = f"{overlay_tag}  —  {model_label}" if overlay_tag else \
                 f"Pair {pair_idx+1}  —  {model_label}"

        # Wait for user to confirm ready
        print(f"\n  Press SPACE to start rollout ...")
        while True:
            imgs   = self._read_images()
            canvas = self._make_canvas(
                imgs,
                header,
                "SPACE to start",
                model_label)
            cv2.imshow("PairedEval", canvas)
            if (cv2.waitKey(50) & 0xFF) == ord(" "):
                break

        # Fill observation buffer
        state_buf = deque(maxlen=obs_horizon)
        for _ in range(obs_horizon):
            state_buf.append(self._read_state())
            time.sleep(1.0 / CONTROL_FREQ)

        if not self.dry_run:
            self._start_live_control()

        pc_t         = torch.from_numpy(scene_pc).unsqueeze(0).to(self.device)
        grasp_type_t = torch.tensor([grasp_type_id], dtype=torch.long, device=self.device)
        dt           = 1.0 / CONTROL_FREQ
        step         = 0
        aborted      = False

        while step < self.max_steps:
            t0    = time.time()
            obs_s = np.stack([self._normalize_state(s, norm) for s in state_buf])
            obs_t = torch.from_numpy(obs_s).unsqueeze(0).to(self.device)

            with torch.no_grad():
                chunk = policy.predict_action(obs_t, pc_t, grasp_type_t,
                                              num_inference_steps=self.num_inf_steps)
            chunk = self._unnormalize_action(chunk.squeeze(0).cpu().numpy(), norm)

            n_exec = min(self.action_horizon, self.max_steps - step)
            for i in range(n_exec):
                av = chunk[min(i, len(chunk) - 1)]
                if not self.dry_run:
                    self._execute_action(av)
                state_buf.append(self._read_state())
                step += 1

                elapsed = time.time() - t0
                t0 = time.time()
                sl = dt - elapsed
                if sl > 0:
                    time.sleep(sl)

                imgs   = self._read_images()
                canvas = self._make_canvas(
                    imgs,
                    header,
                    f"Step {step}/{self.max_steps}   SPACE=stop  Q=abort",
                    model_label)
                cv2.imshow("PairedEval", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    aborted = True
                    step = self.max_steps
                    break
                elif key == ord(" "):
                    step = self.max_steps
                    break

        if not self.dry_run:
            self._stop_live_control()

        # Pull test (only if --pull-dist was set)
        ratchet_info = None
        if self.pull_dist is not None and not self.dry_run and self.fa is not None:
            self._execute_pull()
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

        # Collect rating
        print(f"  Rate this trial:   g = good   b = bad   s = skip")
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
        symbol = "✓" if rating == "good" else ("✗" if rating == "bad" else "–")
        print(f"  {symbol} {model_label}: {rating.upper()}")
        return rating, aborted, ratchet_info

    # ── Pair loop ────────────────────────────────────────────────────────────

    def run_pair(self, pair_idx, order, batch_idx=0, batch_pair_idx=0, batch_n_pairs=1,
                 orientation_deg=None):
        """Run one full pair (fresh PC captured BEFORE EACH trial). Returns pair dict.

        `pair_idx` is the GLOBAL session pair counter (used for strict model
        alternation and stored in the JSON). `batch_idx`, `batch_pair_idx`,
        `batch_n_pairs` describe this pair's position inside its current batch.

        Design note: we capture a fresh point cloud for each of the two trials
        in a pair, not one shared PC. The hand from trial 1 will invariably
        nudge the hold by a few mm, so trial 2's model needs a PC that
        matches whatever the hold actually looks like at the moment trial 2
        starts. Between the two trials we prompt the operator to reposition
        the hold as close to its trial-1 location as possible, then re-scan.
        Per-PC centroids are logged in the JSON so drift can be audited.
        """
        orient_str = f"{orientation_deg}°" if orientation_deg is not None else "not recorded"
        print(f"\n{'='*62}")
        print(f"  SESSION PAIR {pair_idx+1}   |   "
              f"BATCH {batch_idx+1}, pair {batch_pair_idx+1}/{batch_n_pairs}")
        print(f"  Hold : {HOLD_NAMES.get(self.hold_id,'?')} (id={self.hold_id})   "
              f"Grasp : {self.grasp_type}   Orientation : {orient_str}")
        print(f"  Order: {order[0]}  →  {order[1]}")
        print(f"{'='*62}")

        ratings    = {}
        pc_stats   = {}        # model_label -> {"n_valid": int, "centroid": [x,y,z]}
        aborted    = False

        for trial_num, model_label in enumerate(order, start=1):
            print(f"\n  ── Trial {trial_num}/2: {model_label} ──")

            # Between trials of the same pair, remind operator to re-align
            # the hold before we re-scan (trial 1's rollout may have bumped it).
            if trial_num == 2:
                print(f"\n  *** Reposition the {self.grasp_type} hold to match its "
                      f"location/orientation from trial 1 as closely as you can. ***")
                # If the user wants to quit mid-pair, let QuitRequested propagate
                # up through run_pair/run_batch — the completed trial-1 rating is
                # NOT appended to _all_pairs (we only append on pair completion),
                # so the session stays consistent on restart.
                self._wait_or_quit("  Press Enter when the hold is back in place ...")

            # Reset arm, then park + fresh PC capture + return to approach pose.
            if not self.dry_run:
                self._reset_robot()
                scene_pc = self._park_and_capture_pc()
            else:
                print("  [dry-run] Using zero PC")
                scene_pc = np.zeros((PC_N_POINTS, 3), dtype=np.float32)

            # Log PC summary stats for later drift audit
            valid_mask = np.any(scene_pc != 0, axis=-1)
            n_valid    = int(valid_mask.sum())
            if n_valid > 0:
                centroid = scene_pc[valid_mask].mean(axis=0).tolist()
            else:
                centroid = [0.0, 0.0, 0.0]
            pc_stats[model_label] = {"n_valid": n_valid, "centroid": centroid}

            policy     = self.with_policy if model_label == MODEL_WITH else self.no_policy
            norm       = self.with_norm   if model_label == MODEL_WITH else self.no_norm
            cfg        = self.with_cfg    if model_label == MODEL_WITH else self.no_cfg
            # grasp_type_id is fed to WITH model; ignored internally by WITHOUT model
            gt_id      = self.grasp_type_id

            overlay_tag = (f"B{batch_idx+1} "
                           f"pair {batch_pair_idx+1}/{batch_n_pairs} "
                           f"[{self.grasp_type}]")
            rating, trial_aborted, ratchet_info = self._run_single_trial(
                policy, norm, cfg, gt_id, scene_pc, model_label, pair_idx,
                overlay_tag=overlay_tag)

            ratings[model_label] = rating
            pc_stats[model_label]["pull_angle_deg"] = 180.0
            if ratchet_info is not None:
                pc_stats[model_label]["ratchet"] = ratchet_info
            aborted = aborted or trial_aborted
            if trial_aborted:
                break

        print(f"\n  Pair {pair_idx+1} result:")
        print(f"    WITH taxonomy   : {ratings.get(MODEL_WITH, '—').upper()}")
        print(f"    WITHOUT taxonomy: {ratings.get(MODEL_NO,   '—').upper()}")

        # Report drift between the two PCs so the operator sees the scale of
        # hold movement and can decide whether to drop the pair.
        if MODEL_WITH in pc_stats and MODEL_NO in pc_stats:
            c1 = np.array(pc_stats[order[0]]["centroid"], dtype=np.float64)
            c2 = np.array(pc_stats[order[1]]["centroid"], dtype=np.float64)
            drift_m = float(np.linalg.norm(c1 - c2))
            print(f"    PC centroid drift between trials: {drift_m*1000:.1f} mm")

        result = {
            "pair":            pair_idx,
            "batch":           batch_idx,
            "batch_pair":      batch_pair_idx,
            "hold_id":         self.hold_id,
            "hold_name":       HOLD_NAMES.get(self.hold_id, "unknown"),
            "grasp_type":      self.grasp_type,
            "grasp_type_id":   self.grasp_type_id,
            "orientation_deg": orientation_deg,
            "order":           order,
            "with_rating":     ratings.get(MODEL_WITH, "skip"),
            "no_rating":       ratings.get(MODEL_NO,   "skip"),
            "pc_stats":        pc_stats,
            "aborted":         aborted,
            "timestamp":       datetime.now().isoformat(),
        }
        if self.pull_dist is not None:
            result["pull_dist_m"]    = self.pull_dist
            result["pull_angle_deg"] = 180.0
            result["pull_stiffness"] = {
                "kx": self.pull_stiffness,
                "ky": self.pull_lateral_stiffness,
                "kz": self.pull_z_stiffness,
            }
        return result

    # ── Batch / session orchestration ────────────────────────────────────────

    def _next_order(self):
        """Return [first_model, second_model] for the next global pair."""
        if self._first_model is None:
            self._first_model = random.choice([MODEL_WITH, MODEL_NO])
            print(f"\n  Random coin-flip: pair 1 starts with {self._first_model}\n"
                  f"  (order then strictly alternates for every subsequent pair)")
        other = MODEL_NO if self._first_model == MODEL_WITH else MODEL_WITH
        if self._global_pair_idx % 2 == 0:
            return [self._first_model, other]
        return [other, self._first_model]

    def _get_or_create_batch_config(self, grasp_type, hold_id, n_pairs, batch_idx,
                                    orientation_deg=None):
        """Return the batch config dict for this batch_idx, creating it if needed.

        If the batch was previously started (resume), reuse the existing entry
        so `completed_pairs` is preserved. Does NOT mutate completed_pairs."""
        for bc in self._batch_configs:
            if bc.get("batch") == batch_idx:
                return bc
        bc = {
            "batch":           batch_idx,
            "grasp_type":      grasp_type,
            "hold_id":         hold_id,
            "hold_name":       HOLD_NAMES.get(hold_id, "unknown"),
            "n_pairs":         n_pairs,
            "orientation_deg": orientation_deg,
            "completed_pairs": 0,
        }
        self._batch_configs.append(bc)
        return bc

    def run_batch(self, grasp_type, hold_id, n_pairs, batch_idx,
                  orientation_deg=None):
        """Run (n_pairs - already_completed) pairs with fixed (grasp_type, hold_id,
        orientation_deg). Propagates QuitRequested if the user types 'q' at a prompt."""
        self._set_batch(grasp_type, hold_id)
        bc = self._get_or_create_batch_config(grasp_type, hold_id, n_pairs,
                                               batch_idx, orientation_deg)
        start_bpi = bc.get("completed_pairs", 0)
        # Use saved orientation if resuming and none was passed
        if orientation_deg is None:
            orientation_deg = bc.get("orientation_deg")

        if start_bpi >= n_pairs:
            print(f"\n  Batch {batch_idx+1} ({grasp_type} @ "
                  f"{orientation_deg}°) already complete "
                  f"({start_bpi}/{n_pairs}) — skipping.")
            return

        orient_str = f"{orientation_deg}°" if orientation_deg is not None else "orientation not set"
        print(f"\n{'#'*62}")
        print(f"  BATCH {batch_idx+1} — grasp={grasp_type}, "
              f"hold={HOLD_NAMES.get(hold_id,'?')} (id={hold_id}), "
              f"orientation={orient_str}, {n_pairs} pairs" +
              (f"  [RESUMING from pair {start_bpi+1}]" if start_bpi > 0 else ""))
        print(f"{'#'*62}")
        print(f"  Set the {grasp_type.upper()} hold to {orient_str} on the testbed.")
        self._wait_or_quit("  Press Enter when the hold is in position ...")

        for bpi in range(start_bpi, n_pairs):
            order  = self._next_order()
            result = self.run_pair(self._global_pair_idx, order,
                                   batch_idx=batch_idx,
                                   batch_pair_idx=bpi,
                                   batch_n_pairs=n_pairs,
                                   orientation_deg=orientation_deg)
            self._all_pairs.append(result)
            self._global_pair_idx += 1
            bc["completed_pairs"] = bpi + 1

            # Incremental save after every pair so a crash/segfault/power-loss
            # at worst loses the single in-progress pair, never a whole session.
            self._save_only()

            if result["aborted"]:
                print("\n  Trial aborted — treating as save-point and quitting.")
                raise QuitRequested()

            if bpi < n_pairs - 1:
                print(f"\n{'─'*62}")
                print(f"  Batch {batch_idx+1} — pair {bpi+1}/{n_pairs} complete "
                      f"({len(self._all_pairs)} pairs total this session).")
                print(f"  REPOSITION the {grasp_type} hold to {orient_str} "
                      f"(same orientation, fresh position).")
                self._wait_or_quit("  Press Enter when ready for the next pair ...")

        print(f"\n  ✓ Batch {batch_idx+1} ({grasp_type} @ {orient_str}) complete: "
              f"{n_pairs}/{n_pairs} pairs recorded.")

    def run_multi_batch(self, batches):
        """batches: list of (grasp_type, hold_id, n_pairs[, orientation_deg]) tuples.

        Preserves an existing mode if one was already set (e.g. 'single' or a
        resumed session). Otherwise defaults to 'scripted'."""
        if not self._mode:
            self._mode = "scripted"
        self._planned_batches = list(batches)
        for i, spec in enumerate(batches):
            gt, hid, n = spec[0], spec[1], spec[2]
            orientation_deg = spec[3] if len(spec) > 3 else None
            self.run_batch(gt, hid, n, batch_idx=i, orientation_deg=orientation_deg)
        self._save_and_analyze()

    def run_interactive(self):
        """Prompt the user for each batch's config in turn until they choose to stop.
        Type 'q' at any prompt to save & exit cleanly."""
        self._mode = "interactive"
        print("\n  Interactive batch mode. Type 'q' at any prompt to save & quit.\n")
        # Start batch numbering from wherever the resumed session left off
        batch_idx = len(self._batch_configs)
        while True:
            print(f"\n{'·'*62}")
            print(f"  Configure batch {batch_idx+1}  "
                  f"(so far this session: {len(self._all_pairs)} pairs)")
            print(f"{'·'*62}")

            gt = self._prompt_grasp_type()
            if gt is None:
                raise QuitRequested()
            hid = self._prompt_hold_id(default_for_grasp=gt)
            if hid is None:
                raise QuitRequested()
            orientation_deg = self._prompt_orientation()
            if orientation_deg is None:
                raise QuitRequested()
            n = self._prompt_int("How many pairs in this batch?", default=4, min_val=1)
            if n is None:
                raise QuitRequested()

            self.run_batch(gt, hid, n, batch_idx=batch_idx,
                           orientation_deg=orientation_deg)
            batch_idx += 1

            cont = self._prompt_yes_no(
                "\n  Run another batch (different grasp type / hold)?", default=True)
            if not cont:
                break

        self._save_and_analyze()

    # ── Interactive prompts ──────────────────────────────────────────────────

    STANDARD_ORIENTATIONS = [-45.0, -22.5, 0.0, 22.5, 45.0]

    def _prompt_orientation(self, default=None):
        """Prompt operator for hold orientation on the testbed (degrees).
        Standard values: -45, -22.5, 0, 22.5, 45 relative to pull axis."""
        opts = ", ".join(str(int(v) if v == int(v) else v)
                         for v in self.STANDARD_ORIENTATIONS)
        suffix = f" (default {default})" if default is not None else ""
        print(f"  Hold orientation on testbed [{opts}]{suffix}: ", end="", flush=True)
        while True:
            try:
                raw = input().strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if raw in ("q", "quit", "exit"):
                return None
            if raw == "" and default is not None:
                return float(default)
            try:
                val = float(raw)
                if val not in self.STANDARD_ORIENTATIONS:
                    print(f"  Note: {val}° is not a standard orientation {self.STANDARD_ORIENTATIONS}.")
                return val
            except ValueError:
                print(f"  Enter a number (e.g. -45, -22.5, 0, 22.5, 45): ", end="", flush=True)

    def _prompt_grasp_type(self):
        names = list(GRASP_TYPE_IDS.keys())
        print(f"  Grasp types: {', '.join(names)}")
        while True:
            try:
                raw = input("  Grasp type for this batch: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if raw in ("q", "quit", "exit"):
                return None
            if raw in GRASP_TYPE_IDS:
                return raw
            print(f"  '{raw}' is not a valid grasp type. Choose from: {', '.join(names)}")

    def _prompt_hold_id(self, default_for_grasp=None):
        # Suggest the "canonical" hold for this grasp type as the default
        GRASP_TO_DEFAULT_HOLD = {"jug": 0, "crimp": 1, "sloper": 2, "pinch": 3}
        default = GRASP_TO_DEFAULT_HOLD.get(default_for_grasp, 0)
        hold_list = ", ".join(f"{k}={v}" for k, v in HOLD_NAMES.items())
        while True:
            try:
                raw = input(f"  Hold ID [{hold_list}] (default {default}): ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if raw in ("q", "quit", "exit"):
                return None
            if raw == "":
                return default
            try:
                val = int(raw)
                if val in HOLD_NAMES:
                    return val
            except ValueError:
                pass
            print(f"  Invalid. Choose an integer in {list(HOLD_NAMES.keys())}.")

    def _prompt_int(self, label, default=5, min_val=1):
        while True:
            try:
                raw = input(f"  {label} (default {default}): ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if raw in ("q", "quit", "exit"):
                return None
            if raw == "":
                return default
            try:
                val = int(raw)
                if val >= min_val:
                    return val
            except ValueError:
                pass
            print(f"  Invalid. Enter an integer ≥ {min_val}.")

    def _prompt_yes_no(self, label, default=True):
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            try:
                raw = input(f"  {label} {suffix}: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            if raw == "":
                return default
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no", "q", "quit", "exit"):
                return False

    # ── Save + analyze ───────────────────────────────────────────────────────

    def _save_only(self):
        """Write the session JSON. Safe to call multiple times / from signal handler."""
        if not self._all_pairs:
            return None
        out_path = self.results_dir / f"paired_session_{self._session_id}.json"
        payload = {
            "session_id":       self._session_id,
            "mode":             self._mode,
            "with_ckpt":        self._with_ckpt_path,
            "no_ckpt":          self._no_ckpt_path,
            "max_steps":        self.max_steps,
            "action_horizon":   self.action_horizon,
            "inference_steps":  self.num_inf_steps,
            "first_model":      self._first_model,
            "planned_batches":  list(self._planned_batches),
            "batches":          self._batch_configs,
            "pairs":            self._all_pairs,
            "last_saved":       datetime.now().isoformat(),
        }
        try:
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\n  Raw session results -> {out_path}")
            return out_path
        except Exception as e:
            # Last-ditch fallback to /tmp so we never lose recorded pairs
            fallback = Path("/tmp") / f"paired_session_{self._session_id}.json"
            try:
                with open(fallback, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"\n  [warn] primary save failed ({e}); wrote fallback -> {fallback}")
                return fallback
            except Exception:
                print(f"\n  [error] could not save session: {e}")
                return None

    def _save_and_analyze(self):
        out_path = self._save_only()
        if out_path is None:
            return
        try:
            self._print_analysis(self._all_pairs, out_path)
        except Exception as e:
            print(f"\n  [warn] analysis print failed: {e}")
            print(f"  JSON was saved to {out_path} — rerun analysis offline.")

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _print_subset_analysis(self, label, pair_results):
        """Print success-rate table + McNemar for an arbitrary subset of pairs."""
        n = len(pair_results)
        if n == 0:
            print(f"\n  [{label}] no pairs — skipping.")
            return

        def tally(key):
            g = sum(1 for r in pair_results if r[key] == "good")
            b = sum(1 for r in pair_results if r[key] == "bad")
            s = sum(1 for r in pair_results if r[key] == "skip")
            return g, b, s

        wg, wb, ws = tally("with_rating")
        ng, nb, ns = tally("no_rating")
        wp, wci    = wilson_ci(wg, wg + wb)
        np_, nci   = wilson_ci(ng, ng + nb)

        n_paired = n01 = n10 = n11 = n00 = 0
        for r in pair_results:
            wr, nr = r["with_rating"], r["no_rating"]
            if wr in ("good", "bad") and nr in ("good", "bad"):
                n_paired += 1
                wok, nok = wr == "good", nr == "good"
                if   wok and not nok: n01 += 1
                elif not wok and nok: n10 += 1
                elif wok and nok:     n11 += 1
                else:                 n00 += 1

        chi2_stat, p_val = mcnemar_test(n01, n10)

        print(f"\n{'='*62}")
        print(f"  {label}  —  {n} pairs")
        print(f"{'='*62}")
        print(f"  {'Model':<24} {'G':>4} {'B':>4} {'S':>4}  {'Rate':>6}  {'95% CI'}")
        print(f"  {'─'*58}")
        print(f"  {'WITH taxonomy':<24} {wg:>4} {wb:>4} {ws:>4}  "
              f"{wp*100:>5.1f}%  [{wci[0]*100:.1f}%, {wci[1]*100:.1f}%]")
        print(f"  {'WITHOUT taxonomy':<24} {ng:>4} {nb:>4} {ns:>4}  "
              f"{np_*100:>5.1f}%  [{nci[0]*100:.1f}%, {nci[1]*100:.1f}%]")
        print(f"  {'─'*58}")
        print(f"  Δ (WITH − WITHOUT)              {(wp - np_)*100:>+6.1f}%")

        print(f"\n  Paired contingency ({n_paired} complete pairs, "
              f"{n - n_paired} pair(s) with a skip):")
        print(f"    Both good              : {n11}")
        print(f"    WITH only good   (n01) : {n01}")
        print(f"    WITHOUT only good (n10): {n10}")
        print(f"    Both bad               : {n00}")

        print(f"\n  McNemar's test (H0: no difference):")
        if p_val is None:
            print("    Insufficient discordant pairs (n01 + n10 = 0) — no test")
        else:
            if chi2_stat is not None:
                print(f"    chi^2 = {chi2_stat:.3f},  p = {p_val:.4f}", end="")
            else:
                print(f"    Exact binomial  p = {p_val:.4f}", end="")
            if p_val < 0.05:
                print("  <- SIGNIFICANT (p < 0.05)")
            elif p_val < 0.10:
                print("  <- marginal (p < 0.10)")
            else:
                print("  <- not significant")

    def _print_analysis(self, pair_results, saved_path):
        # Overall
        self._print_subset_analysis("OVERALL (all grasp types)", pair_results)

        # Per grasp type
        by_grasp = {}
        for r in pair_results:
            by_grasp.setdefault(r["grasp_type"], []).append(r)
        for gt in sorted(by_grasp.keys()):
            self._print_subset_analysis(f"GRASP TYPE = {gt}", by_grasp[gt])

        print(f"\n  Raw JSON -> {saved_path}")
        print(f"{'='*62}\n")


# ── Entry point ───────────────────────────────────────────────────────────

def _parse_batches(spec):
    """Parse a batches spec like 'jug:0:4:-45,jug:0:4:0,crimp:1:4:22.5'.

    Format: grasp:hold:pairs[:orientation_deg]
    orientation_deg is optional (e.g. -45, -22.5, 0, 22.5, 45).
    Returns list of (grasp_type, hold_id, n_pairs, orientation_deg) tuples.
    """
    batches = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(
                f"Bad batch spec '{chunk}' — expected grasp:hold:pairs or "
                f"grasp:hold:pairs:orientation_deg  (e.g. jug:0:4:-45)")
        gt  = parts[0].strip().lower()
        hid = int(parts[1])
        n   = int(parts[2])
        orientation_deg = float(parts[3]) if len(parts) == 4 else None
        if gt not in GRASP_TYPE_IDS:
            raise ValueError(f"Unknown grasp type '{gt}' in '{chunk}'. "
                             f"Valid: {list(GRASP_TYPE_IDS.keys())}")
        if hid not in HOLD_NAMES:
            raise ValueError(f"Unknown hold id {hid} in '{chunk}'. "
                             f"Valid: {list(HOLD_NAMES.keys())}")
        if n < 1:
            raise ValueError(f"Pair count must be >=1, got {n} in '{chunk}'")
        batches.append((gt, hid, n, orientation_deg))
    if not batches:
        raise ValueError("Empty --batches spec")
    return batches


def main():
    parser = argparse.ArgumentParser(
        description="Paired evaluation: with-taxonomy vs without-taxonomy "
                    "(interactive multi-batch, or scripted via --batches)")
    # Single-batch shortcut (backwards-compatible)
    parser.add_argument("--hold", type=int, default=None,
                        help="Hold ID  (0=edge_A  1=edge_B  2=sloper  3=pinch  4=test_edge). "
                             "If given with --grasp-type and --pairs, runs a single batch.")
    parser.add_argument("--grasp-type", type=str, default=None,
                        choices=list(GRASP_TYPE_IDS.keys()),
                        help="Grasp type for single-batch mode")
    parser.add_argument("--pairs", type=int, default=None,
                        help="Number of pairs for single-batch mode")
    parser.add_argument("--orientation", type=float, default=None,
                        help="Hold orientation in degrees for single-batch mode "
                             "(-45, -22.5, 0, 22.5, 45). Logged per pair. "
                             "If omitted, you will be prompted interactively.")
    # Scripted multi-batch
    parser.add_argument("--batches", type=str, default=None,
                        help="Scripted multi-batch spec. Format: "
                             "'grasp:hold:pairs,grasp:hold:pairs,...' "
                             "e.g. 'crimp:1:5,jug:0:5,sloper:2:5,pinch:3:5'")
    # Resume a previous session
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a paired_session_*.json from a previous run. "
                             "Restores session_id / first_model / completed pairs, "
                             "then continues from where that session left off. "
                             "If the resumed session was scripted, its planned "
                             "batches are replayed (already-completed ones are "
                             "skipped).")
    # Shared
    parser.add_argument("--max-steps",       type=int, default=200)
    parser.add_argument("--action-horizon",  type=int, default=8)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--with-ckpt",   default=str(DEFAULT_WITH_CKPT))
    parser.add_argument("--no-ckpt",     default=str(DEFAULT_NO_CKPT))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--dry-run", action="store_true",
                        help="No robot — cameras only, for testing the script flow")
    parser.add_argument("--pull-dist", type=float, default=None,
                        help="If set, execute a spring-testbed pull test after each rollout: "
                             "arm moves this many meters toward robot base (180°, -X, hardcoded). "
                             "Standard value: 0.105 (10.5 cm, 11 ratchet teeth). "
                             "After each pull, enter the ratchet tooth count (0–11); "
                             "the script computes and logs displacement + slip force.")
    parser.add_argument("--pull-stiffness", type=float, default=4000.0,
                        help="X-axis (pull direction) Cartesian impedance stiffness in N/m. "
                             "Default 4000. At 4000 vs spring k≈280 N/m, arm equilibrates "
                             "at ~97 mm for a perfect grip (≈10 ratchet teeth).")
    parser.add_argument("--pull-lateral-stiffness", type=float, default=100.0,
                        help="Y-axis (lateral) Cartesian impedance stiffness in N/m. "
                             "Default 100 — compliant, allows natural side-to-side hand flex.")
    parser.add_argument("--pull-z-stiffness", type=float, default=100.0,
                        help="Z-axis (vertical) Cartesian impedance stiffness in N/m. "
                             "Default 100 — compliant, allows natural up/down hand flex.")
    parser.add_argument("--pull-z-bias", type=float, default=0.0,
                        help="Upward Z offset (meters) added to pull target pose. "
                             "Compensates for LEAP hand weight not in Franka gravity model. "
                             "Tune empirically: start at 0.02. (default: 0.0)")
    args = parser.parse_args()

    # Decide run mode from CLI args (may be overridden by --resume below)
    single_batch_args = [args.hold is not None,
                         args.grasp_type is not None,
                         args.pairs is not None]
    if args.batches is not None:
        if any(single_batch_args):
            parser.error("--batches cannot be combined with "
                         "--hold/--grasp-type/--pairs")
        try:
            scripted_batches = _parse_batches(args.batches)
        except ValueError as e:
            parser.error(str(e))
        run_mode = "scripted"
    elif all(single_batch_args):
        scripted_batches = [(args.grasp_type, args.hold, args.pairs,
                             getattr(args, "orientation", None))]
        run_mode = "single"
    elif not any(single_batch_args):
        scripted_batches = None
        run_mode = "interactive"
    else:
        parser.error("Specify EITHER --hold + --grasp-type + --pairs (all three), "
                     "OR --batches, OR none of them (interactive mode).")

    evaluator = PairedEvaluator(
        with_ckpt=args.with_ckpt,
        no_ckpt=args.no_ckpt,
        max_steps=args.max_steps,
        action_horizon=args.action_horizon,
        dry_run=args.dry_run,
        results_dir=args.results_dir,
        num_inference_steps=args.inference_steps,
        pull_dist=args.pull_dist,
        pull_stiffness=args.pull_stiffness,
        pull_lateral_stiffness=args.pull_lateral_stiffness,
        pull_z_stiffness=args.pull_z_stiffness,
        pull_z_bias=args.pull_z_bias,
    )

    # If resuming, load the JSON and pull planned_batches / mode from it.
    # Any fresh --batches / --hold/etc. args take precedence over the saved
    # plan only if explicitly given; otherwise the saved plan is replayed.
    if args.resume:
        # Friendly path resolution: try the given path as-is, then fall back
        # to results_dir + basename (in case the user ran from a different cwd).
        resume_path = Path(args.resume)
        candidates = [resume_path]
        if not resume_path.is_absolute():
            candidates.append(Path(args.results_dir) / resume_path.name)
            candidates.append(Path(args.results_dir) / resume_path)
        resolved = next((p for p in candidates if p.exists()), None)
        if resolved is None:
            tried = "\n    ".join(str(p) for p in candidates)
            parser.error(f"--resume file not found. Tried:\n    {tried}")
        try:
            evaluator._load_resume(resolved)
        except Exception as e:
            parser.error(f"--resume failed to load '{resolved}': {e}")

        if args.batches is None and not all(single_batch_args):
            # No explicit plan on CLI — reuse whatever was saved
            if evaluator._planned_batches:
                scripted_batches = list(evaluator._planned_batches)
                run_mode = evaluator._mode or "scripted"
            else:
                # Was an interactive session; continue interactively
                scripted_batches = None
                run_mode = "interactive"

    def _cleanup(signum=None, frame=None):
        print("\n  Interrupt — cleaning up ...")

        # 1) SAVE FIRST, in the main thread, before touching any hardware.
        #    Hardware teardown (pyrealsense / FrankaArm threads) often
        #    segfaults on exit and would kill a daemon thread before
        #    the JSON gets written. Saving synchronously here guarantees
        #    no collected pairs are ever lost to a teardown crash.
        try:
            evaluator._save_only()
        except Exception as e:
            print(f"  [warn] save_only failed: {e}")

        # 2) THEN run hardware teardown in a daemon thread with a timeout,
        #    so a hung / crashy stop_skill can't block exit.
        def _do():
            try:
                evaluator._stop_live_control()
            except Exception:
                pass
            try:
                if evaluator.fa:
                    evaluator.fa.stop_skill()
            except Exception:
                pass
            try:
                if evaluator.leap_dxl:
                    evaluator.leap_dxl.set_torque_enabled(evaluator._leap_motors, False)
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        t = threading.Thread(target=_do, daemon=True)
        t.start()
        t.join(timeout=5)
        os._exit(0)

    signal.signal(signal.SIGINT, _cleanup)

    # Seed the mode so the first _save_only() call records it even if the
    # user quits before the first pair completes.
    if evaluator._mode is None:
        evaluator._mode = run_mode

    evaluator.setup()
    try:
        if run_mode == "interactive":
            evaluator.run_interactive()
        else:
            evaluator.run_multi_batch(scripted_batches)
    except QuitRequested:
        print("\n  Quit requested — saving and shutting down cleanly.")
        print(f"  {len(evaluator._all_pairs)} pair(s) saved this session.")
        print(f"  To resume later:\n"
              f"    python3 paired_eval.py --resume "
              f"{evaluator.results_dir}/paired_session_{evaluator._session_id}.json")
        _cleanup()
    except KeyboardInterrupt:
        _cleanup()
    except Exception as e:
        print(f"\n  Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        try:
            evaluator._save_only()
        except Exception:
            pass
        _cleanup()


if __name__ == "__main__":
    main()
