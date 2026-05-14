"""
Point cloud utilities for diffusion policy data collection and inference.

Handles:
  - Multi-camera depth → point cloud conversion
  - Multi-camera fusion into world frame using calibration transforms
  - Workspace bounding box cropping
  - Farthest Point Sampling (FPS) to fixed number of points
  - Statistical outlier removal

Design choices (following DP3 / DexCap):
  - XYZ only (no color) for better appearance generalization
  - 1024 points via FPS (less random than uniform sampling)
  - Workspace crop removes table/background noise
"""

import numpy as np

# Default workspace bounds in world frame (meters).
# Calibrated 2026-05-11 for the spring displacement testbed (supersedes the
# 2026-03-16 flat-table calibration):
#   - Spring testbed seats each hold ~2 cm above the original table surface,
#     so the hold base now sits at Z ≈ 0.027 m (vs. 0.006 m on the bare table)
#   - z_max=0.40 gives headroom for the rig + a tall jug
# History (do NOT regress past these):
#   - z_min=-0.02 (table-era) → 95% of 1024 FPS points on the table; model
#     ignored the PC. Unrecoverable; dataset had to be rebuilt.
#   - z_min=0.006 (table-era, correct for flat-table collection)
#   - z_min=0.027 (CURRENT, spring testbed era — verify with
#     check_pc_sensitivity.py before each fresh data-collection session;
#     centroid Z should sit above the rig, full hold geometry visible)
DEFAULT_WORKSPACE_BOUNDS = {
    "x_min": 0.30, "x_max": 0.85,
    "y_min": -0.35, "y_max": 0.35,
    "z_min": 0.0265, "z_max": 0.40,
}

DEPTH_SCALE = 0.001  # RealSense default: uint16 mm → meters


def depth_to_points(depth_img, fx, fy, cx, cy, depth_scale=DEPTH_SCALE):
    """Convert a depth image to (N, 3) points in camera frame.

    Args:
        depth_img: (H, W) uint16 depth from RealSense
        fx, fy, cx, cy: camera intrinsics
        depth_scale: conversion factor (default 0.001 for mm→m)

    Returns:
        points: (N, 3) float32 array of valid 3D points
    """
    h, w = depth_img.shape[:2]
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    z = depth_img.astype(np.float32) * depth_scale
    valid = z > 0

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return points


def farthest_point_sampling(points, n_samples):
    """Farthest Point Sampling (FPS) to select n_samples from points.

    More uniform coverage than random sampling. O(N * n_samples).

    Args:
        points: (N, 3) array
        n_samples: number of points to select

    Returns:
        sampled: (n_samples, 3) array
    """
    N = len(points)
    if N <= n_samples:
        # Pad by repeating if not enough points
        if N == 0:
            return np.zeros((n_samples, 3), dtype=np.float32)
        indices = np.arange(N)
        pad_indices = np.random.choice(N, n_samples - N, replace=True)
        indices = np.concatenate([indices, pad_indices])
        return points[indices].astype(np.float32)

    selected = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(N, np.inf, dtype=np.float32)

    # Start from a random point
    selected[0] = np.random.randint(N)

    for i in range(1, n_samples):
        last = points[selected[i - 1]]
        dist = np.sum((points - last) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = np.argmax(distances)

    return points[selected].astype(np.float32)


def crop_to_workspace(points, bounds=None):
    """Remove points outside a 3D bounding box.

    Args:
        points: (N, 3) array in world frame
        bounds: dict with x_min, x_max, y_min, y_max, z_min, z_max

    Returns:
        cropped: (M, 3) array
    """
    if bounds is None:
        bounds = DEFAULT_WORKSPACE_BOUNDS

    mask = (
        (points[:, 0] >= bounds["x_min"]) & (points[:, 0] <= bounds["x_max"]) &
        (points[:, 1] >= bounds["y_min"]) & (points[:, 1] <= bounds["y_max"]) &
        (points[:, 2] >= bounds["z_min"]) & (points[:, 2] <= bounds["z_max"])
    )
    return points[mask]


def remove_statistical_outliers(points, nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers based on mean distance to neighbors.

    A simplified version that doesn't require Open3D.

    Args:
        points: (N, 3) array
        nb_neighbors: number of neighbors to consider
        std_ratio: standard deviation threshold

    Returns:
        filtered: (M, 3) array
    """
    if len(points) < nb_neighbors + 1:
        return points

    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    mean_dists = dists[:, 1:].mean(axis=1)  # exclude self

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std

    mask = mean_dists < threshold
    return points[mask]


def fuse_multi_camera_points(depth_images, cam_intrinsics, cam_extrinsics,
                              workspace_bounds=None, n_points=1024,
                              outlier_removal=True):
    """Full pipeline: multiple depth images → fused, cropped, downsampled point cloud.

    Args:
        depth_images: list of (H, W) uint16 depth images, one per camera
        cam_intrinsics: list of dicts with keys 'fx', 'fy', 'cx', 'cy'
        cam_extrinsics: list of (4, 4) camera-to-world transform matrices
        workspace_bounds: dict with x/y/z min/max, or None for defaults
        n_points: target number of output points (FPS)
        outlier_removal: whether to run statistical outlier removal

    Returns:
        point_cloud: (n_points, 3) float32 array in world frame
    """
    all_points = []

    for depth, intrinsics, extrinsics in zip(depth_images, cam_intrinsics, cam_extrinsics):
        if depth is None or depth.size == 0:
            continue

        # Depth → camera-frame points
        pts_cam = depth_to_points(
            depth,
            fx=intrinsics["fx"], fy=intrinsics["fy"],
            cx=intrinsics["cx"], cy=intrinsics["cy"],
        )

        if len(pts_cam) == 0:
            continue

        # Transform to world frame: p_world = R @ p_cam + t
        # extrinsics is a 4×4 matrix [R|t; 0 1]
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        pts_world = (R @ pts_cam.T).T + t

        all_points.append(pts_world)

    if not all_points:
        return np.zeros((n_points, 3), dtype=np.float32)

    points = np.concatenate(all_points, axis=0).astype(np.float32)

    # Crop to workspace
    points = crop_to_workspace(points, workspace_bounds)

    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    # Remove outliers
    if outlier_removal and len(points) > 50:
        points = remove_statistical_outliers(points)

    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    # Pre-random-sample before FPS to keep FPS tractable.
    # FPS is O(N × n_points); on 700k+ points it would take minutes.
    # Sampling to 20k first has negligible quality impact at 1024 output points.
    PRE_SAMPLE = 20000
    if len(points) > PRE_SAMPLE:
        idx = np.random.choice(len(points), PRE_SAMPLE, replace=False)
        points = points[idx]

    # FPS downsample to target count
    point_cloud = farthest_point_sampling(points, n_points)

    return point_cloud


def get_cam_intrinsics_from_realsense(camera_obj):
    """Extract intrinsics dict from a robomail CameraClass instance.

    Reads live intrinsics directly from the active pyrealsense2 pipeline at the
    current capture resolution. The calibration YAML files stored in robomail
    may be from a different resolution and should not be trusted.
    """
    try:
        import pyrealsense2 as rs2
        pipeline = camera_obj.pipeline
        profile = pipeline.get_active_profile()
        depth_stream = profile.get_stream(rs2.stream.depth).as_video_stream_profile()
        live = depth_stream.get_intrinsics()
        return {
            "fx": float(live.fx),
            "fy": float(live.fy),
            "cx": float(live.ppx),
            "cy": float(live.ppy),
        }
    except Exception:
        # Fallback to calibration file if live query fails
        intr = camera_obj.get_cam_intrinsics()
        return {
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.cx),
            "cy": float(intr.cy),
        }


def get_cam_extrinsics_from_realsense(camera_obj):
    """Extract 4×4 camera-to-world transform from a robomail CameraClass.

    Returns:
        (4, 4) numpy array
    """
    extr = camera_obj.get_cam_extrinsics()
    if isinstance(extr, np.ndarray) and extr.shape == (4, 4):
        return extr
    # If it's a RigidTransform, convert
    if hasattr(extr, 'matrix'):
        return np.array(extr.matrix)
    return np.eye(4)
