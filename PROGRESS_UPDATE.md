# Progress Update: Grasp-Taxonomy-Aware 3D Diffusion Policy

**Date:** May 2026  
**Purpose:** Status update for professor / slide deck material

---

## 1. Project Overview

A **grasp-taxonomy-aware 3D diffusion policy** for dexterous climbing hold manipulation:

- **Robot:** Franka Emika Panda arm + LEAP Hand (16-DoF)
- **Observation:** Point clouds (1024 pts, XYZ only) from 4 RealSense cameras fused into world frame
- **Conditioning:** Grasp type label (crimp / sloper / pinch / jug) — policy generates type-specific trajectories
- **Benchmark:** Climbing holds — zero prior work on robotic dexterous grasping of climbing holds

---

## 2. ✅ Evaluation Complete (2026-05-15)

### Dataset
- **80 paired trials** — exactly 4 pairs × 5 hold orientations (−45°, −22.5°, 0°, +22.5°, +45°) × 4 hold types
- **Session file:** `eval_results/paired_session_20260515_074256.json`
- Spring displacement testbed, 13 cm pull at 180°, impedance control
- **Primary metric:** ratchet slip force (N) — continuous, from two springs in parallel

### Key Results (Wilcoxon signed-rank, paired, two-sided)

| Grasp type | WITH taxonomy (med) | WITHOUT (med) | Δ med | p-value | sig | Cohen's d |
|------------|---------------------|---------------|-------|---------|-----|-----------|
| **Crimp**  | 15.7 N              | 9.2 N         | +6.5 N | 0.0007 | *** | 1.05 |
| **Jug**    | 15.7 N              | 14.4 N        | +1.3 N | 0.0278 | *   | 0.55 |
| **Sloper** | 7.9 N               | 5.2 N         | +2.6 N | 0.0012 | **  | 1.08 |
| **Pinch**  | 15.7 N              | 9.2 N         | +6.5 N | <0.001 | *** | 1.82 |
| **OVERALL**| 13.1 N              | 7.9 N         | +5.2 N | <0.001 | *** | 0.95 |

**Interpretation:** Taxonomy conditioning significantly improves grip strength (force at slip) across all
hold types. Effect sizes are large (d > 1) for crimp, sloper, and pinch — the three precision-critical
holds requiring exact finger placement. The jug result is the weakest (d=0.55) but still significant,
consistent with the jug's forgiving geometry (deep pocket, mm-scale slack). This matches the
**precision-regularizer** framing: conditioning tightens the sampling distribution around the correct
hand configuration, with the biggest gain where geometry leaves no room for imprecision.

### Force reference
```
0 teeth =  5.2 N (preload)   5 teeth = 18.3 N
3 teeth = 13.1 N              9 teeth = 28.7 N   11 teeth = 33.9 N (max)
```

---

## 3. Generated Figures (eval_results/figures/)

| File | Description | Status |
|------|-------------|--------|
| `fig_ratchet_1_boxplots.png` | Slip force box+jitter plots per grasp type, WITH vs WITHOUT, Wilcoxon brackets | ✅ PRIMARY FIGURE |
| `fig_ratchet_2_scatter.png` | Per-pair scatter (WITH vs WITHOUT force), colored by grasp type | ✅ |
| `fig_ratchet_3_by_orientation.png` | Mean force vs hold orientation (−45°→+45°), both models | ✅ |
| `fig_ratchet_4_per_pair_delta.png` | Per-pair force delta (WITH − WITHOUT), signed bars | ✅ |
| `fig_latent_4panel.png` | Latent space PCA + silhouette + kNN metrics | ✅ (old ckpt) |
| `fig_action_distribution.png` | Within-input sampling variance comparison | ✅ (old ckpt) |
| `fig_novelty_mode_commitment.png` | Mode commitment / precision-regularizer evidence | ✅ (old ckpt) |
| `fig3_z_centroid.png` | PC centroid Z by grasp type (shows geometric separability) | ✅ |

**Note:** figures marked "(old ckpt)" were generated from the pre-rig checkpoints
(`pc_with_taxonomy`, `pc_no_taxonomy`). Regenerate from rig checkpoints if the paper
requires exact alignment — see HANDOFF.md §Analysis Reproduction.

---

## 4. What Was Built

1. **Point cloud pipeline** — multi-camera depth → world-frame fusion → workspace crop → FPS (1024 pts)
2. **DP3-style architecture** — PointNet encoder + grasp-type conditioning (one-hot → 64-d MLP) + 1D U-Net DDPM
3. **Spring displacement testbed** — linear single-axis pull, linear ratchet for continuous peak-displacement measurement
4. **Paired evaluation protocol** — both models loaded once, per-trial fresh PC, strict global order alternation, incremental save, resume, ratchet force logging
5. **Training** — 3000 epochs each, rig-collected dataset (spring testbed, 5 orientations per hold)

---

## 5. What Remains for a Full Paper Submission

### Completed ✅
- Primary paired eval (80 pairs, all 4 hold types, all 5 orientations)
- Ratchet force figures (fig_ratchet_1–4)
- Latent / action-distribution / novelty figures (from pre-rig checkpoints — may regenerate)

### Pending ⏳
- **Wrong-label ablation** — run paired_eval with deliberately swapped taxonomy labels (~10 pairs per mislabeled condition). Proves conditioning causally controls behavior, not just noise. Use `--batches` with the correct hold but wrong `grasp_type` field in the paired_eval batch spec. See tasks/todo.md.
- **Held-out hold generalization** — evaluate taxonomy-conditioned model on a NEW crimp hold (test_edge / hold 4) never seen during training. Tests within-type generalization. No demos needed — eval only.
- **Regenerate latent/novelty figures from rig checkpoints** — run `generate_latent_viz.py`, `generate_action_dist_viz.py`, `generate_novelty_viz.py` pointing at `pc_with_taxonomy_rig` and `pc_no_taxonomy_rig`.
- **Training curves** — plot loss vs epoch from `checkpoints/pc_with_taxonomy_rig_train.log` and `checkpoints/pc_no_taxonomy_rig/train.log`.
- **Paper write-up** — update results section with ratchet force numbers, replace binary-success figures with ratchet figures.

---

## 6. Slide-Ready Bullets

- **Problem:** Dexterous grasping requires fundamentally different hand configurations per hold geometry. No prior work on taxonomy-conditioned imitation learning for dexterous hands.
- **Method:** DP3-style diffusion policy (PointNet + 1D U-Net) conditioned on grasp type label. Point clouds from 4 fused RealSense cameras. 23-dim arm+hand state/action space.
- **New evaluation:** Spring displacement testbed measures continuous slip force per trial (ratchet readout), not binary pass/fail. Stronger grip = hold displaced further before slipping.
- **Result:** Taxonomy conditioning improves grip strength on ALL 4 hold types (Wilcoxon, all p < 0.05). Largest gains on precision-critical holds: pinch d=1.82, sloper d=1.08, crimp d=1.05. Jug (forgiving geometry) smallest gain but still significant: d=0.55.
- **Mechanism (precision regularizer):** Unconditioned model's MEAN actions still separate grasp types (kNN=99% in latent space). The conditioning reduces per-rollout VARIANCE — tightens the sampling distribution around the correct configuration. Analogy: classifier-free guidance in image diffusion / low-temperature LLM sampling.
- **Differentiator from Dexonomy (RSS 2025):** They generate static grasp poses for a motion planner. We learn continuous visuomotor execution trajectories (arm + hand, approach through contact) via imitation learning and characterise what the taxonomy label does to the diffusion decoder.
