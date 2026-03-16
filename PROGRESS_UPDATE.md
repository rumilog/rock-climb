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
| **Episodes collected** | 37 |
| **Total timesteps** | 6,435 |
| **Hold** | 0 (edge_A) |
| **Grasp type** | jug |
| **Quality** | 37 good, 0 bad |

**Target per hold:** 50–80 good episodes (RESEARCH_PLAN).  
**Current:** ~37 / 60 for hold 0 (jug) — about halfway for this hold.

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
1. **Finish data collection for hold 0** — ~23 more episodes to reach 60
2. **Collect data for holds 1–3** — crimp, sloper, pinch (50–80 each)
3. **Pilot training** — train on hold 0 only, validate pipeline
4. **Real-robot evaluation** — test policy on Franka + LEAP
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
- **Progress:** Full pipeline implemented and calibrated; 37 jug episodes collected for hold 0; ready for pilot training once we reach ~60 episodes.
- **Next:** Complete hold 0 data → pilot train → real-robot eval → scale to all 4 hold types → implement Part 1 VLM identifier.
