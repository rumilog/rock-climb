#!/usr/bin/env python3
"""
Diagnostic: does the captured point cloud change when the hold is moved?

Moves the arm to the park pose (out of camera view), captures a point cloud
using the exact same pipeline as evaluate.py, prints stats, saves a top-down
PNG and a raw .npy file. Run three times with different hold positions and compare.

Usage:
    python3 check_pc_sensitivity.py --label pos_a   # hold in position A
    python3 check_pc_sensitivity.py --label pos_b   # hold in new location
    python3 check_pc_sensitivity.py --label pos_c   # hold rotated differently

    # Outputs: pos_a.png + pos_a.npy  (etc.)

    # Skip arm movement (cameras only):
    python3 check_pc_sensitivity.py --no-robot --label pos_a
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
TELE_ROOT = SCRIPT_DIR.parent
FRANKA_SCRIPTS_DIR = TELE_ROOT / "TeleoperationUnity" / "Robot Control - Python" / "Franka Scripts"
for p in [str(FRANKA_SCRIPTS_DIR), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import robomail.vision as vis
from point_cloud_utils import (
    fuse_multi_camera_points,
    get_cam_intrinsics_from_realsense,
    get_cam_extrinsics_from_realsense,
    DEFAULT_WORKSPACE_BOUNDS,
)

CAMERA_NUMBERS = [2, 3, 4, 5]
CAMERA_RAW_W = 848
CAMERA_RAW_H = 480
PC_CAPTURE_N_FRAMES = 5
PC_N_POINTS = 1024

# Same reset joints used in evaluate.py — arm tucks back and up, mostly out of view.
# Override with --park-joints if this still clips a camera.
PARK_ARM_JOINTS = np.array([-0.11426599, -0.56029082, -0.06635159, -2.17443357,
                             0.04112932,  2.15592909,  0.54378958], dtype=np.float64)


def move_arm_to_park(fa, joints, settle_s=1.5):
    try:
        fa.stop_skill()
    except Exception:
        pass
    time.sleep(0.3)
    print(f"  Moving arm to park pose...")
    fa.goto_joints(joints.tolist(), duration=4.0, dynamic=False, buffer_time=0.2, block=True)
    time.sleep(settle_s)
    print("  Arm at park pose.")


def capture_pc(cameras, cam_intrinsics, cam_extrinsics):
    """Same logic as evaluate.py _capture_clean_point_cloud()."""
    pcs = []
    for frame_i in range(PC_CAPTURE_N_FRAMES):
        raw_frames = cameras.get_next_frames()
        depth_images = []
        for i in range(len(CAMERA_NUMBERS)):
            depth = raw_frames[i][1]
            if depth is None or depth.size == 0:
                depth = np.zeros((CAMERA_RAW_H, CAMERA_RAW_W), dtype=np.uint16)
            depth_images.append(depth)
        pc = fuse_multi_camera_points(
            depth_images=depth_images,
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
            n_points=PC_N_POINTS,
            outlier_removal=(frame_i == 0),
        )
        pcs.append(pc)
        time.sleep(0.05)
    return pcs[-1]


def save_topdown_png(pc, out_path, label, bounds=DEFAULT_WORKSPACE_BOUNDS):
    """Top-down (X-Y) scatter; color = Z height (blue=low, red=high)."""
    W, H = 640, 480
    x_min, x_max = bounds["x_min"], bounds["x_max"]
    y_min, y_max = bounds["y_min"], bounds["y_max"]
    z_min, z_max = bounds["z_min"], bounds["z_max"]

    img = np.ones((H, W, 3), dtype=np.uint8) * 30

    valid = np.any(pc != 0, axis=-1)
    pts = pc[valid]
    if len(pts) == 0:
        cv2.putText(img, "NO VALID POINTS", (20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        z_norm = np.clip((pts[:, 2] - z_min) / max(z_max - z_min, 1e-6), 0, 1)
        colors_gray = (z_norm * 255).astype(np.uint8)
        colors = cv2.applyColorMap(colors_gray[:, np.newaxis], cv2.COLORMAP_JET)[:, 0, :]

        px = ((pts[:, 0] - x_min) / max(x_max - x_min, 1e-6) * (W - 1)).astype(int)
        py = (H - 1 - (pts[:, 1] - y_min) / max(y_max - y_min, 1e-6) * (H - 1)).astype(int)
        in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        for i in np.where(in_bounds)[0]:
            cv2.circle(img, (px[i], py[i]), 2, colors[i].tolist(), -1)

        cx_w, cy_w, cz_w = pts.mean(axis=0)
        cpx = int((cx_w - x_min) / max(x_max - x_min, 1e-6) * (W - 1))
        cpy = int(H - 1 - (cy_w - y_min) / max(y_max - y_min, 1e-6) * (H - 1))
        cv2.drawMarker(img, (cpx, cpy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img, f"centroid ({cx_w:.3f}, {cy_w:.3f}, {cz_w:.3f})",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(img, f"X: {x_min:.2f}..{x_max:.2f} m", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(img, f"Y: {y_min:.2f}..{y_max:.2f} m", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(img, f"Z: blue={z_min:.2f}m  red={z_max:.2f}m", (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(img, label, (5, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

    cv2.imwrite(str(out_path), img)
    print(f"Saved top-down PNG → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="snap", help="Label shown on output image")
    parser.add_argument("--out", default=None, help="Output PNG path (default: <label>.png)")
    parser.add_argument("--no-robot", action="store_true",
                        help="Skip arm movement (cameras only)")
    parser.add_argument("--park-joints", type=float, nargs=7, default=None,
                        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
                        help="Override park joint angles (7 floats in radians)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(f"{args.label}.png")
    park_joints = np.array(args.park_joints, dtype=np.float64) if args.park_joints else PARK_ARM_JOINTS

    # --- Arm: move to park pose ---
    fa = None
    if not args.no_robot:
        from frankapy import FrankaArm
        print("Connecting to Franka arm...")
        fa = FrankaArm(with_gripper=False)
        move_arm_to_park(fa, park_joints)

    # --- Cameras ---
    print(f"\nInitialising cameras {CAMERA_NUMBERS}...")
    cameras = vis.ThreadedCameras(
        cam_numbers=CAMERA_NUMBERS,
        image_height=CAMERA_RAW_H,
        image_width=CAMERA_RAW_W,
        get_point_cloud=False,
        get_verts=False,
    )
    cameras.get_next_frames()
    print("Cameras ready.\n")

    # --- Per-camera extrinsic sanity check ---
    print("=== Camera extrinsics (identity = calibration missing!) ===")
    cam_intrinsics = []
    cam_extrinsics = []
    for i, cam_obj in enumerate(cameras.cameras):
        intr = get_cam_intrinsics_from_realsense(cam_obj)
        extr = get_cam_extrinsics_from_realsense(cam_obj)
        cam_intrinsics.append(intr)
        cam_extrinsics.append(extr)
        is_identity = np.allclose(extr, np.eye(4), atol=1e-3)
        t = extr[:3, 3]
        print(f"  Cam {CAMERA_NUMBERS[i]}: fx={intr['fx']:.1f} fy={intr['fy']:.1f} | "
              f"t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]"
              + ("  *** IDENTITY — CALIBRATION MISSING ***" if is_identity else ""))

    # --- Capture ---
    print(f"\nCapturing point cloud ({PC_CAPTURE_N_FRAMES} frames)...")
    pc = capture_pc(cameras, cam_intrinsics, cam_extrinsics)

    n_valid = int(np.sum(np.any(pc != 0, axis=-1)))
    print(f"\n=== Point cloud result ===")
    print(f"  Valid points : {n_valid} / {PC_N_POINTS}")

    if n_valid > 0:
        pts = pc[np.any(pc != 0, axis=-1)]
        centroid = pts.mean(axis=0)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        print(f"  Centroid     : X={centroid[0]:.4f}  Y={centroid[1]:.4f}  Z={centroid[2]:.4f}")
        print(f"  Bbox X       : [{bbox_min[0]:.4f}, {bbox_max[0]:.4f}]")
        print(f"  Bbox Y       : [{bbox_min[1]:.4f}, {bbox_max[1]:.4f}]")
        print(f"  Bbox Z       : [{bbox_min[2]:.4f}, {bbox_max[2]:.4f}]")
    else:
        print("  WARNING: all points are zero — pipeline is broken.")
        print("  Check extrinsics above (identity = calibration file missing).")

    save_topdown_png(pc, out_path, label=args.label)

    npy_path = out_path.with_suffix(".npy")
    np.save(str(npy_path), pc)
    print(f"Saved raw point cloud → {npy_path}  (shape {pc.shape}, dtype {pc.dtype})")

    try:
        cameras.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
