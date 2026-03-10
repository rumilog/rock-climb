"""
Hold 3D pose detection from RealSense depth cameras.

Detects a climbing hold's centroid (x, y, z) and surface normal (nx, ny, nz)
using depth data from one or more RealSense cameras. Returns a 6-dim vector
to append to the robot state, providing explicit spatial grounding for the
diffusion policy.

Detection approach:
  1. Capture depth + color from a designated camera
  2. Segment the hold region (depth-based foreground in a workspace ROI)
  3. Build a small point cloud of the hold surface
  4. Compute centroid and principal surface normal via PCA
"""

import numpy as np
import cv2

# Intrinsics for RealSense D415/D455 at 848x480 (approximate defaults)
# These are overwritten at runtime if camera intrinsics are available.
DEFAULT_FX = 421.0
DEFAULT_FY = 421.0
DEFAULT_CX = 424.0
DEFAULT_CY = 240.0
DEPTH_SCALE = 0.001  # RealSense default: depth values in mm -> meters


def depth_to_pointcloud(depth_img, fx=DEFAULT_FX, fy=DEFAULT_FY,
                        cx=DEFAULT_CX, cy=DEFAULT_CY, depth_scale=DEPTH_SCALE):
    """Convert a depth image to an (N, 3) point cloud in camera frame."""
    h, w = depth_img.shape[:2]
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

    z = depth_img.astype(np.float64) * depth_scale
    valid = z > 0

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return points.astype(np.float32)


def segment_hold_region(depth_img, color_img=None,
                        min_depth_m=0.15, max_depth_m=0.80,
                        workspace_roi=None):
    """
    Segment the hold as the dominant foreground cluster in the depth image.

    Args:
        depth_img: (H, W) uint16 depth in mm (RealSense default)
        color_img: (H, W, 3) optional BGR image (unused for now)
        min_depth_m: minimum depth in meters to consider
        max_depth_m: maximum depth in meters to consider
        workspace_roi: optional (y1, y2, x1, x2) pixel ROI to restrict search

    Returns:
        mask: (H, W) bool array where True = hold pixels
    """
    depth_m = depth_img.astype(np.float64) * DEPTH_SCALE

    mask = (depth_m > min_depth_m) & (depth_m < max_depth_m)

    if workspace_roi is not None:
        y1, y2, x1, x2 = workspace_roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = True
        mask = mask & roi_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_uint8 = mask.astype(np.uint8) * 255
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(depth_img.shape[:2], dtype=bool)

    largest = max(contours, key=cv2.contourArea)
    result = np.zeros(depth_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(result, [largest], -1, 255, -1)

    return result > 0


def compute_hold_pose(points):
    """
    Compute hold centroid and surface normal from a set of 3D points.

    Args:
        points: (N, 3) array of hold surface points in camera frame

    Returns:
        centroid: (3,) float32 — mean position
        normal: (3,) float32 — unit surface normal (smallest PCA component)
    """
    if len(points) < 10:
        return np.zeros(3, dtype=np.float32), np.array([0, 0, 1], dtype=np.float32)

    centroid = points.mean(axis=0)

    centered = points - centroid
    cov = centered.T @ centered / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    normal = eigenvectors[:, 0].astype(np.float32)
    if normal[2] < 0:
        normal = -normal

    return centroid.astype(np.float32), normal


def detect_hold_from_depth(depth_img, color_img=None,
                           fx=DEFAULT_FX, fy=DEFAULT_FY,
                           cx=DEFAULT_CX, cy=DEFAULT_CY,
                           min_depth_m=0.15, max_depth_m=0.80,
                           workspace_roi=None):
    """
    Full pipeline: depth image -> hold centroid + surface normal.

    Args:
        depth_img: (H, W) uint16 depth from RealSense
        color_img: (H, W, 3) optional BGR image
        fx, fy, cx, cy: camera intrinsics
        min_depth_m, max_depth_m: depth range for hold detection
        workspace_roi: optional (y1, y2, x1, x2) pixel ROI

    Returns:
        hold_pose: (6,) float32 array [cx, cy, cz, nx, ny, nz]
                   centroid in camera frame + unit surface normal
        n_points: int — number of points used (0 if detection failed)
    """
    mask = segment_hold_region(depth_img, color_img,
                               min_depth_m, max_depth_m, workspace_roi)
    n_hold_pixels = mask.sum()
    if n_hold_pixels < 50:
        return np.zeros(6, dtype=np.float32), 0

    # Build point cloud only for hold pixels
    h, w = depth_img.shape[:2]
    v_coords, u_coords = np.where(mask)
    z = depth_img[v_coords, u_coords].astype(np.float64) * DEPTH_SCALE
    valid = z > 0
    u_coords = u_coords[valid].astype(np.float64)
    v_coords = v_coords[valid].astype(np.float64)
    z = z[valid]

    x = (u_coords - cx) * z / fx
    y = (v_coords - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    if len(points) < 10:
        return np.zeros(6, dtype=np.float32), 0

    centroid, normal = compute_hold_pose(points)
    hold_pose = np.concatenate([centroid, normal]).astype(np.float32)
    return hold_pose, len(points)


def detect_hold_multi_camera(raw_frames, cam_index=0,
                             fx=DEFAULT_FX, fy=DEFAULT_FY,
                             cx=DEFAULT_CX, cy=DEFAULT_CY,
                             min_depth_m=0.15, max_depth_m=0.80):
    """
    Detect hold using a specific camera from multi-camera frame list.

    Args:
        raw_frames: list of (color, depth, ...) tuples from ThreadedCameras
        cam_index: which camera to use for hold detection (default: first)
        fx, fy, cx, cy: intrinsics for the detection camera

    Returns:
        hold_pose: (6,) float32 array [cx, cy, cz, nx, ny, nz]
        n_points: int
    """
    if cam_index >= len(raw_frames):
        return np.zeros(6, dtype=np.float32), 0

    color, depth = raw_frames[cam_index][0], raw_frames[cam_index][1]

    if depth is None or depth.size == 0:
        return np.zeros(6, dtype=np.float32), 0

    return detect_hold_from_depth(
        depth, color, fx, fy, cx, cy, min_depth_m, max_depth_m)
