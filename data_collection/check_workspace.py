"""
Quick diagnostic: captures one point cloud and reports the Z/X/Y ranges.
Run once with empty table, once with hold in workspace.
Usage:
    python3 check_workspace.py
"""
import sys, time
sys.path.insert(0, '/home/rumi/Desktop/tele/data_collection')
import robomail.vision as vis
import numpy as np
from point_cloud_utils import (
    depth_to_points, get_cam_intrinsics_from_realsense, get_cam_extrinsics_from_realsense,
    crop_to_workspace, DEFAULT_WORKSPACE_BOUNDS
)

print("Initialising cameras (2-5)...")
cams = vis.ThreadedCameras(cam_numbers=[2,3,4,5], image_height=480, image_width=848,
                           get_point_cloud=False, get_verts=False)
frames = cams.get_next_frames()
print("Got frames.\n")

all_pts = []
for i, cam_obj in enumerate(cams.cameras):
    intr = get_cam_intrinsics_from_realsense(cam_obj)
    extr = get_cam_extrinsics_from_realsense(cam_obj)
    depth = frames[i][1]
    pts_cam = depth_to_points(depth, **intr)
    R, t = extr[:3,:3], extr[:3,3]
    all_pts.append((R @ pts_cam.T).T + t)

pts = np.concatenate(all_pts, axis=0)

print("=== FULL SCENE (no crop) ===")
print(f"  Points: {len(pts):,}")
print(f"  X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
print(f"  Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
print(f"  Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")

print(f"\n=== AFTER DEFAULT CROP {DEFAULT_WORKSPACE_BOUNDS} ===")
cropped = crop_to_workspace(pts, DEFAULT_WORKSPACE_BOUNDS)
print(f"  Points: {len(cropped):,}")
if len(cropped) > 0:
    print(f"  X: [{cropped[:,0].min():.3f}, {cropped[:,0].max():.3f}]")
    print(f"  Y: [{cropped[:,1].min():.3f}, {cropped[:,1].max():.3f}]")
    print(f"  Z: [{cropped[:,2].min():.3f}, {cropped[:,2].max():.3f}]")

    # Histogram of Z values to see where mass concentrates
    hist, edges = np.histogram(cropped[:,2], bins=10)
    print("\n  Z histogram (low→high):")
    for j in range(len(hist)):
        bar = '#' * (hist[j] // max(hist[j] for h in [hist] for _ in [h]) * 20 // 1000 + 1)
        print(f"    [{edges[j]:.3f}, {edges[j+1]:.3f}]: {hist[j]:,}")

cams.stop()
