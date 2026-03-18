#!/usr/bin/env python3
"""
Capture a single RGB+Depth screenshot from all RealSense cameras.

This script uses the same camera configuration and tiling logic as
`view_rgbd_live.py`, but instead of showing a live window it:

  1. Grabs one set of frames from all cameras.
  2. Builds the RGB+Depth canvas (all cameras tiled).
  3. Saves it to a PNG file (default: rgbd_screenshot.png).
  4. Exits.

Usage (from repo root of this worktree):

    source ~/franka/bin/activate
    python3 capture_rgbd_screenshot.py --out rgbd_screenshot.png
"""

import argparse
import os

import cv2
import numpy as np

try:
    import robomail.vision as vis
except ImportError as e:
    raise RuntimeError(
        "Failed to import robomail.vision. Make sure the robomail package "
        "is installed in the current environment."
    ) from e

# Match the viewer/data_collection configuration
CAMERA_NUMBERS = [2, 3, 4, 5]
CAMERA_RAW_W = 848
CAMERA_RAW_H = 480
DEPTH_SCALE = 0.001  # mm -> meters


def colorize_depth(depth, max_depth_m=1.0):
    """Convert a uint16 depth map in millimeters to a color image."""
    if depth is None or depth.size == 0:
        return np.zeros((CAMERA_RAW_H, CAMERA_RAW_W, 3), dtype=np.uint8)

    depth_m = depth.astype(np.float32) * DEPTH_SCALE
    depth_norm = np.clip(depth_m / max_depth_m, 0.0, 1.0)
    depth_8u = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    return depth_color


def make_rgbd_tiles(frames, cam_numbers):
    """Build per-camera RGB+depth tiles and arrange them in a grid."""
    tiles = []
    for i, (color, depth, _, _) in enumerate(frames):
        # RGB pane
        if color is None or color.size == 0:
            rgb_small = np.zeros((180, 320, 3), dtype=np.uint8)
            rgb_label = f"Cam {cam_numbers[i]} (NO RGB)"
        else:
            rgb_small = cv2.resize(color, (320, 180))
            rgb_label = f"Cam {cam_numbers[i]}"

        cv2.putText(
            rgb_small,
            rgb_label,
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Depth pane
        depth_color = colorize_depth(depth, max_depth_m=1.0)
        depth_small = cv2.resize(depth_color, (320, 180))
        cv2.putText(
            depth_small,
            f"Depth {cam_numbers[i]}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        tile = np.vstack([rgb_small, depth_small])
        tiles.append(tile)

    if not tiles:
        return np.zeros((360, 320, 3), dtype=np.uint8)

    # Arrange tiles in a grid (2 columns by default)
    ncols = 2
    while len(tiles) % ncols != 0:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + ncols]) for i in range(0, len(tiles), ncols)]
    panel = np.vstack(rows)
    return panel


def main():
    parser = argparse.ArgumentParser(description="Capture RGB+Depth screenshot from all cameras")
    parser.add_argument(
        "--out",
        type=str,
        default="rgbd_screenshot.png",
        help="Output PNG path (default: rgbd_screenshot.png)",
    )
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    print(f"Starting cameras {CAMERA_NUMBERS} to capture one RGBD screenshot...")
    cameras = vis.ThreadedCameras(
        cam_numbers=CAMERA_NUMBERS,
        image_height=CAMERA_RAW_H,
        image_width=CAMERA_RAW_W,
        get_point_cloud=False,
        get_verts=False,
    )

    try:
        # Warm-up and capture one set of frames
        print("Waiting for first frames...")
        frames = cameras.get_next_frames()
        print("Frames received. Building canvas...")

        canvas = make_rgbd_tiles(frames, CAMERA_NUMBERS)

        # Optional: add a small status text
        status = "RGB+Depth snapshot per camera"
        cv2.putText(
            canvas,
            status,
            (5, canvas.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Save to disk
        cv2.imwrite(out_path, canvas)
        print(f"Saved RGBD screenshot to: {out_path}")

    finally:
        try:
            cameras.record = False
            if hasattr(cameras, "_thread") and cameras._thread.is_alive():
                cameras._thread.join(timeout=3.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()

