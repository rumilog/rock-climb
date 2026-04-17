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
| Evaluation script (real robot, single model) | ✅ Implemented |
| Paired evaluation (WITH vs WITHOUT taxonomy, real robot) | ✅ Implemented (2026-04-17) |
| Both policies trained (`pc_with_taxonomy`, `pc_no_taxonomy`) | ✅ Completed 2026-04-16 |

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

6. **Started paper draft (`Paper writing/main.tex`)**
   - Full draft: abstract, intro, related work, problem formulation, method, benchmark, discussion, conclusion
   - Related work covers all key competitors: Dexonomy (RSS 2025, closest), Grasp as You Say (NeurIPS 2024), DexGraspVLA (AAAI 2026), OmniDexVLG, UniDexFPM, GenDP, CrossDex
   - Novelty gap table (Table 1) — 5-method comparison across 4 axes
   - Key differentiator from Dexonomy in §2.3: continuous imitation-learning execution trajectories vs static pose snapshots for a motion planner
   - Discussion §6 covers two-part deployment architecture (VLM classifier → diffusion policy)
   - **Outstanding:** 7 new bib entries have `FIXME` author placeholders — fill in before submission

7. **Trained both ablation policies** (completed 2026-04-16)
   - `checkpoints/pc_with_taxonomy/best.pt` — grasp-type-conditioned (our method)
   - `checkpoints/pc_no_taxonomy/best.pt` — no-conditioning ablation (baseline)
   - Identical architecture, data, training schedule except for the grasp-type encoder branch

8. **Built paired evaluation tooling** (`data_collection/paired_eval.py`, 2026-04-17)
   - Both models loaded once; every "pair" runs trial A + trial B on the same hold position so only model identity varies
   - Coin-flip + strict global alternation controls for order bias (each model goes first 50% of the time per grasp type)
   - Fresh point cloud captured **before each trial** (trial 1's hand contact nudges the hold a few mm; between-trial re-scan keeps the comparison faithful); per-trial centroid + point count logged, drift in mm printed per pair for post-hoc filtering
   - Batched session model — one command evaluates all 4 grasp types in sequence (interactive / scripted / single-batch modes)
   - Clean-quit button (`q` at any prompt) + incremental save after every pair + `--resume <json>` for seamless continuation across sittings; alternation parity + `completed_pairs` per batch are all restored on resume
   - Reports Wilson 95% CIs + McNemar's paired test, overall and per grasp type
   - **Smoke test (2 crimp pairs):** WITH 2/2, WITHOUT 0/2 — directionally promising but data-lost to a first-version shutdown segfault (since fixed). Ready for full 80-pair session.

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
1. ✅ Data recollected with fixed z_min=0.006 pipeline (all 4 grasp types)
2. ✅ Both policies trained on cluster: `pc_with_taxonomy`, `pc_no_taxonomy` (2026-04-16)
3. ⏳ **Paired evaluation run** — 20 pairs × 4 grasp types = 80 pairs / 160 rollouts using `paired_eval.py`, supports mid-session quit + resume across sittings
4. **Drift filtering + analysis** — drop pairs with PC centroid drift > ~15 mm (paired_eval logs this per pair), report Wilson CIs + McNemar p-values overall and per grasp type
5. **Final ablations** — PC vs RGB baseline (if time permits); sensitivity to number of demos per grasp type

### Part 1 (Hold Identifier) — Future
1. Benchmark zero-shot VLM on 4-class hold classification
2. Collect in-lab hold photos if fine-tuning needed (independent of robot pipeline)
3. Integrate: RGB snapshot → Part 1 → label → Part 2

---

## 6. Slide-Ready Bullets

- **Problem:** Dexterous grasping needs different hand configurations per object geometry; no prior work on grasp-type-conditioned diffusion policies for climbing holds.
- **Two-part approach:** (1) VLM classifier identifies hold type from RGB image; (2) DP3-style point cloud policy executes the grasp conditioned on that type.
- **Hardware:** Franka + LEAP Hand, 4× RealSense, VR teleoperation.
- **Progress:** Full pipeline implemented and calibrated; workspace bounds verified (`z_min=0.006`); data collected for all 4 grasp types; both ablation policies trained (`pc_with_taxonomy`, `pc_no_taxonomy`, 2026-04-16); paired evaluation tool built with mid-session save/resume (2026-04-17).
- **Next:** Full 80-pair paired evaluation (20 pairs × 4 grasp types) — paired protocol controls for hold drift and order bias, reports per-grasp-type success rates + Wilson CIs + McNemar p-values → write up results → Part 1 VLM identifier.
