# Progress Update: Grasp-Taxonomy-Aware 3D Diffusion Policy

**Date:** March 2026  
**Purpose:** Status update for professor / slide deck material

---

## 1. Project Overview

We are building a **grasp-taxonomy-aware 3D diffusion policy** for dexterous climbing hold manipulation:

- **Robot:** Franka Emika Panda arm + LEAP Hand (16-DoF)
- **Observation:** Point clouds (1024 pts) from 4 RealSense cameras, fused into world frame
- **Conditioning:** Grasp type (crimp / sloper / pinch / jug) — the policy learns type-specific trajectories
- **Benchmark:** Climbing holds — zero prior work on robotic dexterous grasping of holds

---

## 2. Current Status

### Pipeline Status: **Ready for Training**

| Component | Status |
|-----------|--------|
| Data collection (point cloud + 23-dim state) | ✅ Working |
| Episode storage (zarr) | ✅ Working |
| Point cloud fusion (4 cams → 1024 pts) | ✅ Calibrated |
| Workspace bounds | ✅ Calibrated (empty table + hold verified) |
| Training script (PointNet + grasp conditioning) | ✅ Implemented |
| Evaluation script (real robot) | ✅ Implemented |

### Data Collection Progress

| Metric | Value |
|--------|-------|
| **Episodes collected** | 0 (valid) |
| **Hold** | 0 (edge_A) — ready to collect |
| **Grasp type** | jug |
| **Pipeline status** | ✅ Fixed and verified |

**Previous 50 jug episodes were discarded.** The point cloud workspace bounds had `z_min=-0.02`, causing 95% of 1024 sampled points to land on the flat table rather than the hold. The trained pilot model (`checkpoints/pc_large/best.pt`) confirmed this: it produced identical arm motion regardless of hold position or zero-PC input, meaning it learned to ignore the point cloud entirely.

**Fix applied (2026-03-18):** `z_min` corrected to `0.006` in `DEFAULT_WORKSPACE_BOUNDS`. Verified with `check_pc_sensitivity.py` — full hold geometry captured, Z centroid ≈ 0.034 m, zero table noise.

**Target per hold:** 50–80 good episodes (RESEARCH_PLAN).
**Current:** 0 / 50 for hold 0 — ready to recollect with fixed pipeline.

---

## 3. What We've Done This Semester

1. **Upgraded pipeline from RGB to point clouds**
   - Replaced ResNet image encoder with PointNet
   - Multi-camera depth fusion → world-frame point cloud
   - Workspace cropping + FPS downsampling to 1024 points

2. **Added grasp-type conditioning**
   - Policy conditions on crimp / sloper / pinch / jug
   - One-hot embedding → 64-d vector fused with observation

3. **Reduced state/action space**
   - 30-dim → 23-dim (dropped redundant ee_pos / ee_quat)
   - Cleaner action space for diffusion

4. **Fixed camera calibration issues**
   - Live intrinsics from pyrealsense2 (calibration YAMLs were stale)
   - Calibrated workspace bounds with physical measurement (empty table + hold)

5. **Started data collection**
   - 37 jug-grasp episodes on hold 0
   - All episodes marked good; point cloud geometry verified

---

## 4. Two-Part Research Architecture

The project is explicitly decomposed into two independent components:

| Part | Goal | Status |
|------|------|--------|
| **Part 1: Hold Identifier** | VLM classifies hold → grasp type (crimp/sloper/pinch/jug) | Not started |
| **Part 2: Grasp Policy** | DP3-style diffusion policy executes the grasp | Pipeline complete, data collection in progress |

At deployment: Part 1 feeds its label into Part 2 as the conditioning signal.

**Part 1 approach:** VLM-based image classifier. Climbing hold taxonomy is standard industry language — substantial labeled data exists online. Zero-shot GPT-4V or Claude is the starting point; fine-tune CLIP if needed. Completely independent of the zarr data collection pipeline.

## 5. Next Steps

### Part 2 (Grasp Policy) — Immediate
1. ⏳ **Recollect hold 0 data** — 50 jug episodes with fixed z_min=0.006 pipeline
2. **Upload to HuggingFace** and retrain on cluster
3. **Pilot eval on robot** — verify model uses PC: `--zero-pc` vs normal must produce different actions
4. **Collect data for holds 1–3** — crimp, sloper, pinch (50 each) — only after hold 0 eval confirms PC is used
5. **Scale and ablations** — full dataset, compare with/without grasp conditioning, PC vs RGB baseline

### Part 1 (Hold Identifier) — Future
1. Benchmark zero-shot VLM on 4-class hold classification
2. Collect in-lab hold photos if fine-tuning needed (independent of robot pipeline)
3. Integrate: RGB snapshot → Part 1 → label → Part 2

---

## 6. Slide-Ready Bullets

- **Problem:** Dexterous grasping needs different hand configurations per object geometry; no prior work on grasp-type-conditioned diffusion policies for climbing holds.
- **Two-part approach:** (1) VLM classifier identifies hold type from RGB image; (2) DP3-style point cloud policy executes the grasp conditioned on that type.
- **Hardware:** Franka + LEAP Hand, 4× RealSense, VR teleoperation.
- **Progress:** Full pipeline implemented and calibrated; workspace bounds verified (`z_min=0.006`); previous bad-PC dataset discarded; ready to recollect with correct pipeline.
- **Next:** Recollect 50 jug episodes (hold 0) with fixed pipeline → retrain → verify model uses PC → collect holds 1–3 → scale → Part 1 VLM identifier.
