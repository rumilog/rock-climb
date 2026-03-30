#!/usr/bin/env python3
"""
Remove episodes from a zarr dataset.

Usage:
    python3 trim_dataset.py --last                  # remove last episode
    python3 trim_dataset.py --last 3                # remove last 3 episodes
    python3 trim_dataset.py --episode 45            # remove episode 45 (0-indexed)
    python3 trim_dataset.py --zarr path/to.zarr --last

    Add --dry-run to preview without modifying anything.
"""

import argparse
import os
import sys
import numpy as np
import zarr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TELE_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_ZARR = "/mnt/ssd/rumi_tele_datasets/climbing_holds.zarr"


def print_summary(root):
    ends = root["meta/episode_ends"][:]
    n = len(ends)
    total = int(ends[-1]) if n > 0 else 0
    print(f"  {n} episodes, {total} total timesteps")
    if n > 0 and "hold_id" in root["meta"] and "quality" in root["meta"]:
        holds = root["meta/hold_id"][:]
        quality = root["meta/quality"][:]
        good = int(np.sum(quality == 1))
        bad = int(np.sum(quality == 0))
        print(f"  {good} good, {bad} bad")


def remove_last_n(root, n, dry_run=False):
    ends = root["meta/episode_ends"][:]
    total_eps = len(ends)

    if n > total_eps:
        print(f"ERROR: only {total_eps} episodes, can't remove {n}")
        return

    if total_eps - n > 0:
        new_total_steps = int(ends[total_eps - n - 1])
    else:
        new_total_steps = 0

    old_total_steps = int(ends[-1])
    removed_steps = old_total_steps - new_total_steps

    print(f"Removing last {n} episode(s):")
    for i in range(total_eps - n, total_eps):
        s = int(ends[i - 1]) if i > 0 else 0
        e = int(ends[i])
        hold = root["meta/hold_id"][i] if "hold_id" in root["meta"] else "?"
        qual = root["meta/quality"][i] if "quality" in root["meta"] else "?"
        qual_str = "good" if qual == 1 else ("bad" if qual == 0 else str(qual))
        print(f"  ep {i}: {e - s} steps, hold={hold}, quality={qual_str}")

    print(f"Total: removing {removed_steps} timesteps")

    if dry_run:
        print("(dry run — no changes made)")
        return

    confirm = input("Type 'y' or 'yes' to confirm deletion: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Aborted.")
        return

    root["data/state"].resize(new_total_steps, root["data/state"].shape[1])
    root["data/action"].resize(new_total_steps, root["data/action"].shape[1])
    root["data/timestamps"].resize(new_total_steps)
    for cam in sorted(root["data/img"].keys()):
        ds = root[f"data/img/{cam}"]
        root[f"data/img/{cam}"].resize(new_total_steps, *ds.shape[1:])

    new_n = total_eps - n
    root["meta/episode_ends"].resize(new_n)
    if "hold_id" in root["meta"]:
        root["meta/hold_id"].resize(new_n)
    if "quality" in root["meta"]:
        root["meta/quality"].resize(new_n)
    if "grasp_type" in root["meta"]:
        root["meta/grasp_type"].resize(new_n)

    print(f"Done.")


def remove_episode(root, ep_idx, dry_run=False):
    """Remove a specific episode by index (rebuilds arrays without that episode)."""
    ends = root["meta/episode_ends"][:]
    total_eps = len(ends)

    if ep_idx < 0 or ep_idx >= total_eps:
        print(f"ERROR: episode {ep_idx} does not exist (dataset has {total_eps} episodes)")
        return

    starts = np.concatenate([[0], ends[:-1]])
    s = int(starts[ep_idx])
    e = int(ends[ep_idx])
    n_remove = e - s

    hold = root["meta/hold_id"][ep_idx] if "hold_id" in root["meta"] else "?"
    qual = root["meta/quality"][ep_idx] if "quality" in root["meta"] else "?"
    qual_str = "good" if qual == 1 else ("bad" if qual == 0 else str(qual))
    print(f"Removing episode {ep_idx}: {n_remove} steps, hold={hold}, quality={qual_str}")

    if dry_run:
        print("(dry run — no changes made)")
        return

    confirm = input("Type 'y' or 'yes' to confirm deletion: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Aborted.")
        return

    total_steps = int(ends[-1])

    for key in ["data/state", "data/action"]:
        arr = root[key][:]
        arr = np.delete(arr, range(s, e), axis=0)
        root[key].resize(*arr.shape)
        root[key][:] = arr

    ts = root["data/timestamps"][:]
    ts = np.delete(ts, range(s, e))
    root["data/timestamps"].resize(len(ts))
    root["data/timestamps"][:] = ts

    for cam in sorted(root["data/img"].keys()):
        arr = root[f"data/img/{cam}"][:]
        arr = np.delete(arr, range(s, e), axis=0)
        root[f"data/img/{cam}"].resize(arr.shape[0], *arr.shape[1:])
        root[f"data/img/{cam}"][:] = arr

    new_ends = np.delete(ends, ep_idx)
    new_ends[ep_idx:] -= n_remove
    root["meta/episode_ends"].resize(len(new_ends))
    root["meta/episode_ends"][:] = new_ends

    for meta_key in ["hold_id", "quality", "grasp_type"]:
        if meta_key in root["meta"]:
            arr = root[f"meta/{meta_key}"][:]
            arr = np.delete(arr, ep_idx)
            root[f"meta/{meta_key}"].resize(len(arr))
            root[f"meta/{meta_key}"][:] = arr

    print(f"Done.")


def main():
    parser = argparse.ArgumentParser(description="Remove episodes from a zarr dataset")
    parser.add_argument("--zarr", default=DEFAULT_ZARR, help="Path to .zarr dataset")
    parser.add_argument("--last", nargs="?", const=1, type=int,
                        help="Remove last N episodes (default: 1)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Remove a specific episode by index (0-based)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be removed without changing anything")
    args = parser.parse_args()

    if not os.path.exists(args.zarr):
        print(f"Dataset not found: {args.zarr}")
        sys.exit(1)

    root = zarr.open(args.zarr, mode="r+" if not args.dry_run else "r")

    print(f"Dataset: {args.zarr}")
    print_summary(root)
    print()

    if args.last is not None:
        remove_last_n(root, args.last, dry_run=args.dry_run)
    elif args.episode is not None:
        remove_episode(root, args.episode, dry_run=args.dry_run)
    else:
        print("Specify --last or --episode. Use --help for usage.")
        sys.exit(1)

    if not args.dry_run:
        print()
        print_summary(root)


if __name__ == "__main__":
    main()
