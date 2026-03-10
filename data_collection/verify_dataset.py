#!/usr/bin/env python3
"""
Verify and visualize a collected zarr dataset.

Checks structural integrity, prints statistics per hold/quality,
shows sample images and state trajectories for manual inspection.

Usage:
    python3 verify_dataset.py                              # default path
    python3 verify_dataset.py --zarr ../datasets/climbing_holds.zarr
    python3 verify_dataset.py --zarr ../datasets/climbing_holds.zarr --episode 3
    python3 verify_dataset.py --zarr ../datasets/climbing_holds.zarr --plot
"""

import argparse
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TELE_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_ZARR = os.path.join(TELE_ROOT, "datasets", "climbing_holds.zarr")

STATE_LABELS = (
    [f"arm_j{i}" for i in range(7)]
    + ["ee_x", "ee_y", "ee_z"]
    + ["ee_qw", "ee_qx", "ee_qy", "ee_qz"]
    + [f"hand_j{i}" for i in range(16)]
    + ["hold_cx", "hold_cy", "hold_cz", "hold_nx", "hold_ny", "hold_nz"]
)


def load_dataset(zarr_path):
    import zarr
    root = zarr.open(zarr_path, mode="r")
    return root


def print_structure(root):
    print("\n=== Dataset Structure ===")

    def _print_group(g, indent=0):
        for key in sorted(g.keys()):
            obj = g[key]
            prefix = "  " * indent
            if hasattr(obj, "shape"):
                print(f"{prefix}{key}: shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"{prefix}{key}/")
                _print_group(obj, indent + 1)
    _print_group(root)


def print_statistics(root):
    print("\n=== Statistics ===")
    ep_ends = root["meta/episode_ends"][:]
    n_eps = len(ep_ends)
    n_steps = int(ep_ends[-1]) if n_eps > 0 else 0
    print(f"  Episodes: {n_eps}")
    print(f"  Total timesteps: {n_steps}")

    if n_eps == 0:
        print("  (empty dataset)")
        return

    starts = np.concatenate([[0], ep_ends[:-1]])
    lengths = ep_ends - starts
    print(f"  Episode lengths: min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}, median={np.median(lengths):.0f}")

    hold_ids = root["meta/hold_id"][:] if "hold_id" in root["meta"] else None
    quality = root["meta/quality"][:] if "quality" in root["meta"] else None

    if hold_ids is not None and quality is not None:
        print("\n  Per-hold breakdown:")
        for h in np.unique(hold_ids):
            mask = hold_ids == h
            n_good = int(np.sum(quality[mask] == 1))
            n_bad = int(np.sum(quality[mask] == 0))
            avg_len = float(lengths[mask].mean())
            print(f"    hold {h}: {int(mask.sum())} episodes "
                  f"({n_good} good, {n_bad} bad), avg length {avg_len:.0f}")


def verify_alignment(root):
    """Check that all data arrays have consistent lengths."""
    print("\n=== Alignment Check ===")
    ep_ends = root["meta/episode_ends"][:]
    expected_n = int(ep_ends[-1]) if len(ep_ends) > 0 else 0

    errors = []
    for key in ["data/state", "data/action", "data/timestamps"]:
        actual = root[key].shape[0]
        ok = "OK" if actual == expected_n else "MISMATCH"
        if actual != expected_n:
            errors.append(key)
        print(f"  {key}: {actual} (expected {expected_n}) [{ok}]")

    img_group = root["data/img"]
    for cam in sorted(img_group.keys()):
        actual = img_group[cam].shape[0]
        ok = "OK" if actual == expected_n else "MISMATCH"
        if actual != expected_n:
            errors.append(f"data/img/{cam}")
        print(f"  data/img/{cam}: {actual} (expected {expected_n}) [{ok}]")

    if errors:
        print(f"\n  ERRORS in: {errors}")
    else:
        print(f"\n  All arrays aligned at {expected_n} timesteps.")
    return len(errors) == 0


def verify_state_ranges(root):
    """Sanity-check that state values are in plausible ranges."""
    print("\n=== State Range Check ===")
    states = root["data/state"][:]
    if len(states) == 0:
        print("  (no data)")
        return

    for i, label in enumerate(STATE_LABELS[:states.shape[1]]):
        col = states[:, i]
        mn, mx, mu, std = col.min(), col.max(), col.mean(), col.std()
        flag = ""
        if std < 1e-8:
            flag = " [CONSTANT — check sensor]"
        print(f"  {label:12s}: min={mn:+8.4f}  max={mx:+8.4f}  "
              f"mean={mu:+8.4f}  std={std:.4f}{flag}")


def verify_timestamps(root):
    """Check timestamps are monotonic within episodes and spacing is plausible."""
    print("\n=== Timestamp Check ===")
    ts = root["data/timestamps"][:]
    ep_ends = root["meta/episode_ends"][:]
    if len(ts) == 0:
        print("  (no data)")
        return

    starts = np.concatenate([[0], ep_ends[:-1]])
    all_dts = []
    problems = 0
    for i, (s, e) in enumerate(zip(starts, ep_ends)):
        ep_ts = ts[s:e]
        if len(ep_ts) < 2:
            continue
        dt = np.diff(ep_ts)
        if np.any(dt <= 0):
            print(f"  Episode {i}: NON-MONOTONIC timestamps!")
            problems += 1
        all_dts.append(dt)

    if all_dts:
        all_dts = np.concatenate(all_dts)
        freq = 1.0 / np.median(all_dts)
        print(f"  Median dt: {np.median(all_dts)*1000:.1f} ms  ({freq:.1f} Hz)")
        print(f"  Max dt: {all_dts.max()*1000:.1f} ms  Min dt: {all_dts.min()*1000:.1f} ms")
    if problems == 0:
        print("  Timestamps OK.")


def show_episode(root, ep_idx):
    """Display images and state trajectory for one episode."""
    import cv2

    ep_ends = root["meta/episode_ends"][:]
    if ep_idx >= len(ep_ends):
        print(f"Episode {ep_idx} does not exist (dataset has {len(ep_ends)} episodes)")
        return

    start = int(ep_ends[ep_idx - 1]) if ep_idx > 0 else 0
    end = int(ep_ends[ep_idx])
    n = end - start

    hold_id = root["meta/hold_id"][ep_idx] if "hold_id" in root["meta"] else "?"
    quality = root["meta/quality"][ep_idx] if "quality" in root["meta"] else "?"
    qual_str = "GOOD" if quality == 1 else ("BAD" if quality == 0 else str(quality))

    print(f"\n=== Episode {ep_idx}: {n} steps, hold={hold_id}, quality={qual_str} ===")

    cam_names = sorted(root["data/img"].keys())

    for t_offset in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        t = start + t_offset
        panels = []
        for cam in cam_names:
            img = root[f"data/img/{cam}"][t]
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (320, 320))
            cv2.putText(img, f"{cam} t={t_offset}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panels.append(img)

        ncols = 2
        while len(panels) % ncols != 0:
            panels.append(np.zeros_like(panels[0]))
        rows = [np.hstack(panels[i:i + ncols])
                for i in range(0, len(panels), ncols)]
        canvas = np.vstack(rows)

        label = f"Ep {ep_idx} | step {t_offset}/{n} | hold={hold_id} | {qual_str}"
        cv2.putText(canvas, label, (5, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Episode Viewer", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def plot_episode_trajectories(root, ep_idx):
    """Plot state trajectories for one episode using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    ep_ends = root["meta/episode_ends"][:]
    start = int(ep_ends[ep_idx - 1]) if ep_idx > 0 else 0
    end = int(ep_ends[ep_idx])

    states = root["data/state"][start:end]
    actions = root["data/action"][start:end]
    ts = root["data/timestamps"][start:end]
    ts = ts - ts[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i in range(7):
        axes[0].plot(ts, states[:, i], label=STATE_LABELS[i])
    axes[0].set_ylabel("Arm joints (rad)")
    axes[0].legend(fontsize=7, ncol=4)
    axes[0].set_title(f"Episode {ep_idx}")

    for i in range(7, 14):
        axes[1].plot(ts, states[:, i], label=STATE_LABELS[i])
    axes[1].set_ylabel("EE pose")
    axes[1].legend(fontsize=7, ncol=4)

    for i in range(14, min(30, states.shape[1])):
        axes[2].plot(ts, states[:, i], label=STATE_LABELS[i] if i < len(STATE_LABELS) else f"d{i}")
    axes[2].set_ylabel("Hand joints (rad)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=6, ncol=4)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Verify a climbing-hold dataset")
    parser.add_argument("--zarr", default=DEFAULT_ZARR, help="Path to .zarr dataset")
    parser.add_argument("--episode", type=int, default=None,
                        help="Show images + trajectory for a specific episode")
    parser.add_argument("--plot", action="store_true",
                        help="Plot state trajectories (requires matplotlib)")
    args = parser.parse_args()

    if not os.path.exists(args.zarr):
        print(f"Dataset not found: {args.zarr}")
        sys.exit(1)

    root = load_dataset(args.zarr)
    print_structure(root)
    print_statistics(root)
    aligned = verify_alignment(root)
    verify_state_ranges(root)
    verify_timestamps(root)

    if args.episode is not None:
        show_episode(root, args.episode)
        if args.plot:
            plot_episode_trajectories(root, args.episode)
    elif args.plot and root["meta/episode_ends"].shape[0] > 0:
        plot_episode_trajectories(root, 0)

    print("\n" + ("ALL CHECKS PASSED" if aligned else "ALIGNMENT ERRORS FOUND"))


if __name__ == "__main__":
    main()
