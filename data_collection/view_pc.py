#!/usr/bin/env python3
"""
Quick 3D viewer for a saved point cloud .npy file.

Usage:
    python3 view_pc.py pos_a.npy
    python3 view_pc.py pos_a.npy pos_b.npy   # overlay multiple clouds

Drag to rotate, scroll to zoom. Color = Z height (blue low, red high).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load(path):
    pc = np.load(path)
    valid = np.any(pc != 0, axis=-1)
    return pc[valid]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 view_pc.py <file.npy> [file2.npy ...]")
        sys.exit(1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmaps = ["jet", "cool", "autumn"]
    for i, path in enumerate(sys.argv[1:]):
        pts = load(path)
        if len(pts) == 0:
            print(f"  {path}: no valid points, skipping")
            continue
        z = pts[:, 2]
        z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)
        cmap = plt.get_cmap(cmaps[i % len(cmaps)])
        colors = cmap(z_norm)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=colors, s=2, alpha=0.7, label=path)
        print(f"  {path}: {len(pts)} points | "
              f"X=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
              f"Y=[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] "
              f"Z=[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Point Cloud Viewer — drag to rotate")
    if len(sys.argv) > 2:
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
