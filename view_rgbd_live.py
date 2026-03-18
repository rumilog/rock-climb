#!/usr/bin/env python3
"""
Live RGB + depth viewer for the Intel RealSense cameras used in this project.

Shows, for **each** camera:
  - Top: RGB image
  - Bottom: matching depth visualization (colormap)

All cameras are tiled into a single window.

Usage (from repo root):
    source ~/franka/bin/activate
    python3 view_rgbd_live.py

Press 'q' or ESC to quit.
"""

import cv2
import numpy as np

try:
    import robomail.vision as vis
except ImportError as e:
    raise RuntimeError(
        "Failed to import robomail.vision. Make sure the robomail package "
        "is installed in the current environment."
    ) from e


# Match the data_collection configuration
CAMERA_NUMBERS = [2, 3, 4, 5]
CAMERA_RAW_W = 848
CAMERA_RAW_H = 480

# Depth scaling: RealSense default is uint16 in mm
DEPTH_SCALE = 0.001  # mm -> meters


def colorize_depth(depth, max_depth_m=1.0):
    """
    Convert a uint16 depth map in millimeters to a color image for visualization.

    Args:
        depth: (H, W) uint16 depth image (mm)
        max_depth_m: depths beyond this are clamped for visualization

    Returns:
        depth_color: (H, W, 3) uint8 BGR image with colormap applied.
    """
    if depth is None or depth.size == 0:
        return np.zeros((CAMERA_RAW_H, CAMERA_RAW_W, 3), dtype=np.uint8)

    depth_m = depth.astype(np.float32) * DEPTH_SCALE
    depth_norm = np.clip(depth_m / max_depth_m, 0.0, 1.0)
    depth_8u = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    return depth_color


def make_rgbd_tiles(frames, cam_numbers):
    """
    Build per-camera RGB+depth tiles and arrange them in a grid.

    For each camera:
        [ RGB 320x180 ]
        [ DEPTH 320x180 ]
    """
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
    print(f"Starting cameras {CAMERA_NUMBERS} for live RGBD view...")
    cameras = vis.ThreadedCameras(
        cam_numbers=CAMERA_NUMBERS,
        image_height=CAMERA_RAW_H,
        image_width=CAMERA_RAW_W,
        get_point_cloud=False,
        get_verts=False,
    )

    # Warm up
    print("Waiting for first frames...")
    frames = cameras.get_next_frames()
    print("Cameras streaming. Press 'q' or ESC to quit.")

    cv2.namedWindow("RGB + Depth (all cameras)", cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = cameras.get_frames()

            # Build RGB+depth tiles for all cameras
            canvas = make_rgbd_tiles(frames, CAMERA_NUMBERS)

            # Status bar
            status = "Live RGB+Depth per camera  |  Press 'q' or ESC to quit"
            cv2.putText(
                canvas,
                status,
                (5, canvas.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("RGB + Depth (all cameras)", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break

    finally:
        try:
            cameras.record = False
            if hasattr(cameras, "_thread") and cameras._thread.is_alive():
                cameras._thread.join(timeout=3.0)
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Viewer closed.")


if __name__ == "__main__":
    main()

