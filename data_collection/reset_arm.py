#!/usr/bin/env python3
"""
Quick reset utility for the Franka arm (and optionally the LEAP hand).

Usage:
    # Source env first (as always on this machine):
    #   source ~/franka/bin/activate
    #   source ~/frankapy/catkin_ws/devel/setup.bash

    # Default: go to the "approach pose" used by evaluate.py / collect_data.py
    python3 reset_arm.py

    # Go to the fully-clear PARK pose (arm out of all 4 camera views)
    python3 reset_arm.py --park

    # Franka's own factory home pose (fa.reset_joints)
    python3 reset_arm.py --home

    # Also reset the LEAP hand to the neutral open pose
    python3 reset_arm.py --hand

    # Just print the pose it would go to and exit (no motion)
    python3 reset_arm.py --dry-run
"""

import argparse
import sys
import time

import numpy as np

# Same poses used by evaluate.py / collect_data.py — keep in sync.

# Approach pose: arm slightly above the workspace, ready to run a policy.
RESET_ARM_JOINTS = np.array(
    [-0.1111, -0.1703, -0.0621, -2.3442, 0.0408, 2.1952, 0.1559],
    dtype=np.float64,
)

# Park pose: verified 2026-03-18 to be fully clear of all 4 RealSense cameras.
# Use this for clean point-cloud capture.
PARK_ARM_JOINTS = np.array(
    [-0.11426599, -0.56029082, -0.06635159, -2.17443357,
      0.04112932,  2.15592909,  0.54378958],
    dtype=np.float64,
)

# Neutral open hand (Allegro/LEAP joint layout used elsewhere in the repo).
RESET_HAND_JOINTS = np.array(
    [0.078, 0.31, 0.41, 0.25,
     0.14, 0.46, 0.50, 0.30,
     0.43, 0.56, 0.55, 0.30,
     0.17, -0.47, -0.10, 1.22],
    dtype=np.float64,
)


def _reset_franka(mode, duration, dry_run):
    if mode == "approach":
        target = RESET_ARM_JOINTS
        label = "approach pose (RESET_ARM_JOINTS)"
    elif mode == "park":
        target = PARK_ARM_JOINTS
        label = "park pose (PARK_ARM_JOINTS — fully clear of cameras)"
    elif mode == "home":
        target = None
        label = "Franka factory home (fa.reset_joints())"
    else:
        raise ValueError(f"unknown mode: {mode}")

    print(f"  Target: {label}")
    if target is not None:
        print(f"          {np.array2string(target, precision=4, suppress_small=True)}")

    if dry_run:
        print("  [dry-run] skipping motion.")
        return

    from frankapy import FrankaArm
    print("  Connecting to Franka ... ", end="", flush=True)
    fa = FrankaArm()
    print("connected.")

    print(f"  Moving to target over {duration:.1f}s ...")
    if mode == "home":
        fa.reset_joints()
    else:
        fa.goto_joints(target.tolist(), duration=duration, ignore_virtual_walls=True)
    print("  ✓ Arm at target.")
    return fa


def _reset_leap():
    print("  Resetting LEAP hand ...")
    try:
        from dynamixel_client import DynamixelClient
    except ImportError:
        print("  [skip] dynamixel_client not importable — cannot reset hand.")
        return

    port = "/dev/ttyUSB0"
    motors = list(range(16))
    dxl = DynamixelClient(motors, port, 4000000)
    dxl.connect()
    dxl.set_torque_enabled(motors, True)
    dxl.write_desired_pos(motors, RESET_HAND_JOINTS)
    time.sleep(1.0)
    dxl.set_torque_enabled(motors, False)
    dxl.disconnect()
    print("  ✓ LEAP hand at neutral open pose.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--approach", action="store_true",
                     help="Go to the approach pose (default).")
    grp.add_argument("--park", action="store_true",
                     help="Go to the fully-clear PARK pose (out of all camera views).")
    grp.add_argument("--home", action="store_true",
                     help="Franka's own factory home pose via fa.reset_joints().")
    p.add_argument("--hand", action="store_true",
                   help="Also reset the LEAP hand to a neutral open pose.")
    p.add_argument("--duration", type=float, default=5.0,
                   help="Trajectory duration in seconds (default: 5.0).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print target pose and exit without moving.")
    args = p.parse_args()

    if args.park:
        mode = "park"
    elif args.home:
        mode = "home"
    else:
        mode = "approach"

    print(f"\n=== reset_arm.py  ({mode}) ===")
    _reset_franka(mode, duration=args.duration, dry_run=args.dry_run)

    if args.hand and not args.dry_run:
        _reset_leap()

    print("Done.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
