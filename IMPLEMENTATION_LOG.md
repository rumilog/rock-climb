# Implementation Log: Point Cloud Diffusion Policy Pipeline

This document tracks all changes made to convert the data collection and training
pipeline from RGB images to point clouds, and add grasp-type conditioning.

**If you are a new agent picking this up**, read this file first, then read
`RESEARCH_PLAN.md` for the full research context.

## Environment Setup (IMPORTANT)

Before running ANY python commands on the robot machine:
```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
```

## Overview of Changes Needed

### 1. Data Collection (`data_collection/collect_data.py`)
- [x] Architecture fix: ObservationEncoder now concatenates all obs_horizon image
  features (standard DP approach) instead of last-frame-only
- [x] Enable point cloud capture from all 4 cameras (`--point-cloud` flag)
- [x] Semi-automatic arm clear: operator moves arm out of view (SPACE to confirm),
  script captures clean PC, operator returns arm (SPACE to start recording)
- [x] Capture clean scene point cloud (hold only, no robot) at episode start
- [x] Add `--grasp-type` argument for labeling each episode (prompts if omitted)
- [x] Reduce state/action dim from 30 to 23 (drop ee_pos/ee_quat, keep arm 7 + hand 16)

### 2. Episode Storage (`data_collection/episode_storage.py`)
- [x] Add `data/point_cloud` dataset: `(N, 1024, 3)` float32
- [x] Add `meta/grasp_type_id` integer metadata (0=crimp, 1=sloper, 2=pinch, 3=jug)
- [x] Add `GRASP_TYPE_NAMES`, `GRASP_TYPE_IDS` constants
- [x] `EpisodeBuffer.set_initial_point_cloud(pc)`: store once, repeat for all timesteps

### 3. Point Cloud Processing (`data_collection/point_cloud_utils.py`)
- [x] Multi-camera point cloud fusion using existing calibration transforms
- [x] Calibrated workspace bounding box crop in world frame
- [x] Farthest Point Sampling (FPS) to downsample to 1024 points
- [x] Random pre-sampling to 20k points before FPS to keep runtime ≈1s
- [x] Statistical outlier removal (scipy-based, no Open3D needed)
- [x] `get_cam_intrinsics_from_realsense` / `get_cam_extrinsics_from_realsense` helpers
- [x] `get_cam_intrinsics_from_realsense` reads **live** intrinsics from the active
  `pyrealsense2` pipeline at 848×480 (fallback to YAML only if needed)

### 4. Training (`data_collection/train.py`)
- [x] ObservationEncoder concatenates all obs_horizon vision features (fixed earlier)
- [x] `PointNetEncoder`: (B, N, 3) → (B, 256) via per-point MLP + global max pool
- [x] `GraspTypeEncoder`: one-hot(4) → MLP → 64-d embedding
- [x] `PointCloudObservationEncoder`: PointNet + State + GraspType → 512-d cond
- [x] `PointCloudDiffusionPolicy`: full DP3-style policy class
- [x] `PointCloudGraspDataset`: loads point clouds + grasp_type_ids, min-max norm
- [x] Switch to min-max [-1, 1] normalization in PC mode (following DP3)
- [x] Updated state/action dim to 23
- [x] `--point-cloud` flag selects PC mode (backward compatible with image mode)
- [x] `encoder_type` saved in checkpoint config for correct loading

### 5. Evaluation (`data_collection/evaluate.py`)
- [x] Load correct policy type based on `encoder_type` in checkpoint config
- [x] Capture point clouds at runtime (arm-out SPACE flow, same as collect_data)
- [x] Pass grasp type conditioning via `--grasp-type` arg (or interactive prompt)
- [x] 23-dim state for new checkpoints, 30-dim legacy support for old checkpoints
- [x] `_execute_action` correctly dispatches to arm[:7] + hand[7:23] for 23-dim

## Files Modified

| File | Status | Description |
|------|--------|-------------|
| `data_collection/train.py` | COMPLETE | PointNet + grasp conditioning + PC dataset + minmax norm |
| `data_collection/collect_data.py` | COMPLETE | 23-dim state, PC capture, grasp type, 3-step SPACE flow |
| `data_collection/episode_storage.py` | COMPLETE | point_cloud zarr + grasp_type_id meta |
| `data_collection/point_cloud_utils.py` | COMPLETE | NEW - PC processing utilities |
| `data_collection/evaluate.py` | COMPLETE | PC policy loading, runtime capture, grasp type arg |
| `data_collection/paired_eval.py` | COMPLETE (2026-04-17) | Side-by-side WITH-vs-WITHOUT-taxonomy evaluator: paired trials with shared hold position, per-trial PC capture, batched multi-grasp sessions, incremental save, clean `q` quit, `--resume` continuation, Wilson-CI + McNemar analysis (overall + per grasp type) |
| `data_collection/evaluate.py` | UPDATED (2026-04-20) | Added disturbance rejection pull test: `--pull-dist` (meters) + `--pull-angle` (degrees, or prompted per trial). After rollout, arm executes `goto_pose` displacement in X,Y; human visually confirms hold moved. Angle/dist logged in JSON. Off by default. |
| `data_collection/paired_eval.py` | UPDATED (2026-04-20) | Same pull test added. Per-trial `pull_angle_deg` stored in `pc_stats[model_label]` within each pair's JSON entry. |
| `RESEARCH_PLAN.md` | CREATED | Full research plan with citations |
| `IMPLEMENTATION_LOG.md` | UPDATED | This file |

## Architecture Summary

### New DP3-Style Pipeline (--point-cloud mode)

```
collect_data.py --point-cloud --hold 0 --grasp-type crimp
  → Per episode: arm out (SPACE) → capture PC (1024×3) → arm back (SPACE) → record
  → State: arm(7) + hand(16) = 23-dim
  → Zarr: data/point_cloud (N, 1024, 3), meta/grasp_type_id

train.py --point-cloud --epochs 3000 --batch 128 --augment --good-only
  → PointNet(1024×3) → 256-d
  → State(2×23) → MLP → 128-d
  → GraspType(one-hot 4) → MLP → 64-d
  → Fuse → 512-d conditioning vector
  → DDPM 1D temporal U-Net → action chunk (16 × 23-dim)
  → Min-max normalization to [-1, 1]
  → Checkpoint config: encoder_type="point_cloud"

evaluate.py --checkpoint best.pt --hold 0 --grasp-type crimp
  → Detects encoder_type from config → loads PointCloudDiffusionPolicy
  → Same arm-out flow as collection for clean scene PC
  → 23-dim state, min-max unnorm for actions
```

### Legacy Image Pipeline (unchanged, backward compatible)

```
collect_data.py --hold 0   (no --point-cloud)
  → State: arm(7) + hand(16) = 23-dim  [now consistent]

train.py   (no --point-cloud)
  → ResNet-18 × n_cams → VisionEncoder → 512-d
  → State → MLP → 256-d
  → encoder_type="vision" in checkpoint

evaluate.py --checkpoint old_best.pt
  → Detects encoder_type="vision" → loads DiffusionPolicy (image-based)
```

## Progress Log

### Session 1 (2026-03-16)
- Diagnosed architecture mismatch between local train.py and training machine
- Fixed ObservationEncoder to use standard concatenation approach
- Researched field: point clouds are the standard for dexterous hand policies
- Created RESEARCH_PLAN.md with full research design
- Created point_cloud_utils.py with multi-camera fusion + FPS + outlier removal

### Session 2 (2026-03-16)
- Implemented full point cloud pipeline across all 4 files
- **episode_storage.py**: Added point_cloud zarr dataset, grasp_type_id meta,
  EpisodeBuffer.set_initial_point_cloud(), GRASP_TYPE_NAMES/IDS constants
- **collect_data.py**: 23-dim state (dropped ee_pos/ee_quat), --point-cloud flag,
  --grasp-type flag with interactive prompt, 3-step SPACE flow for clean PC capture
  (arm out → capture → arm back → record), _capture_clean_point_cloud()
- **train.py**: PointNetEncoder, GraspTypeEncoder, PointCloudObservationEncoder,
  PointCloudDiffusionPolicy, PointCloudGraspDataset with min-max normalization,
  --point-cloud flag, encoder_type in checkpoint config
- **evaluate.py**: load_policy() dispatches on encoder_type, 23/30-dim state support,
  PC capture flow at trial start, --grasp-type flag, _execute_action() for 23-dim
- **point_cloud_utils.py**: intrinsics now read live from `pyrealsense2`, workspace
  bounds calibrated with `check_workspace.py`, and FPS made fast via 20k pre-sample

### Session 4 (2026-04-01) — Paper Draft + Related Work Overhaul

**Files changed:** `Paper writing/main.tex`, `Paper writing/references.bib`

**Problem identified:** Draft related work was missing its closest competitor (Dexonomy, RSS 2025) and several other directly relevant papers. iDP3 venue was wrong. DexCap listed as arXiv instead of RSS 2024.

**Changes made to `Paper writing/main.tex`:**
- **§1 Introduction:** Added two-sentence citation of Dexonomy + Grasp as You Say to establish the execution-policy gap upfront
- **§2.1 Diffusion Policies:** Added GenDP (CoRL 2024, 3D semantic fields) sentence
- **§2.2 Dexterous Manipulation:** Added UniDexFPM and CrossDex (ICLR 2025)
- **§2.3 Grasp Taxonomy:** Complete rewrite — now covers Dexonomy with 2-sentence differentiation (static pose-gen vs continuous execution policy), plus Grasp as You Say, OmniDexVLG, DexGraspVLA
- **Table 1 (new):** Novelty gap table — 5 methods vs ours on 4 axes (point cloud / grasp type cond. / execution policy / dexterous hand)
- **§3 Problem Formulation:** Added sentence: `g` can come from VLM classifier (Part 1) or manual label at deployment; forward-ref to §6
- **§6 Discussion:** Added `\label{sec:discussion}` + new paragraph on two-part deployment architecture (VLM classifier → diffusion policy)

**Changes made to `Paper writing/references.bib`:**
- **DexCap:** Fixed venue from `arXiv preprint` → `RSS 2024`
- **iDP3:** Changed key `ze2025idp3` → `ze2024idp3`, venue `IROS 2025` → `CoRL 2024` (verify comment added)
- **DexDiffuser:** Updated to `IEEE Robotics and Automation Letters` (verify comment added)
- **Added 7 new entries:** `dexonomy2025`, `tian2024graspasyousay`, `dexgraspvla2026`, `omnidexvlg2024`, `wu2024unidexfpm`, `wang2024gendp`, `crossdex2025`

**⚠️ ACTION REQUIRED before submission:** All 7 new bib entries have `FIXME: see arXiv:XXXXX` placeholder in the `author` field. Fill in real author lists from each arXiv page. CrossDex also needs its arXiv URL confirmed.

### Session 6 (2026-04-20) — Latent/Action Visualisations + Paper Reframe

**Files created:**
- `eval_results/generate_latent_viz.py` (pre-existing, extended)
- `eval_results/generate_action_dist_viz.py` (new)
- `eval_results/generate_novelty_viz.py` (new)

**Files modified:**
- `eval_results/generate_figures.py` — readability pass on all five paper figures.

**What got built.**

1. **Readability pass on `fig1`–`fig5`.** User flagged overlapping labels /
   titles / table cells in the paper-quality figures. Fixed:
   - `fig1_success_rates.png` — removed redundant `n=20` labels, added bar-top
     `k/N` counts, widened canvas + raised ylim so significance stars stop
     colliding with the legend.
   - `fig2_pair_heatmap.png` — flipped from one-tall-column to a row of per-grasp
     narrow 2-column blocks, shortened "With/Without taxonomy" to "With/W/o
     tax.", widened subplots_adjust for the suptitle.
   - `fig3_z_centroid.png` — repositioned annotation arrow so it no longer
     crosses the jug cluster.
   - `fig4_win_tie_loss.png` — raised ylim, moved McNemar p-value annotations
     out of the title strip, pulled the legend outside the axes.
   - `fig5_delta_summary.png` — widened canvas + table col widths so the
     summary table stops clipping "Primary finding" text.

2. **`generate_latent_viz.py` extensions** (Task 1 from previous request).
   - Added `cluster_quality()` → (silhouette score, stratified-5-fold kNN accuracy).
     Results now printed in panel titles so the visual clustering is backed by numbers.
   - Added `fig_latent_eval_overlay.png` — overlays paired-eval success/failure
     markers onto the action-trajectory PCA, matched to nearest training
     episode by PC centroid. Shows that WITHOUT-taxonomy failures are broadly
     distributed across the action space (not concentrated in one sub-region),
     consistent with a precision failure rather than a mode-selection failure.

3. **`generate_action_dist_viz.py`** (Task 2, new file).
   For 32 training inputs (8 per grasp type), samples K=8 DDIM rollouts from
   both checkpoints with the *same* noise seed sequence per model so the
   comparison is fair. Produces `fig_action_distribution.png` with three panels:
   - A. Joint-space PCA of predicted actions, filled ○ = WITH, open △ = W/o.
   - B. Within-input sampling variance per grasp type.
   - C. Between-model L2 distance of mean predictions per grasp type.
   Key empirical finding: means per grasp type are similar between the two
   models (both cluster correctly by grasp type); the WITHOUT model's **within-input**
   variance is 15–30% higher on crimp / sloper / pinch.

4. **`generate_novelty_viz.py`** (Task 4, new file).
   Two figures intended to sell the paper's novelty:
   - `fig_novelty_benchmark.png`: 2×4 grid, column per grasp type. Row 1 is a
     side-view scatter of one representative training PC with a floating
     info-box (Z centroid, peak height, X×Y footprint). Row 2 is the mean
     final-step LEAP hand pose across demos of that type. Establishes that
     four geometrically distinct holds map to four distinct target hand poses.
   - `fig_novelty_mode_commitment.png`: top row = one-joint ribbon plot over
     the 16-step prediction horizon (demo ±1σ vs With vs W/o, all four grasp
     types side by side). Bottom row = final-timestep hand pose fingerprint
     per source (Demos / With / W/o) with between-type-spread metric in the title.

**Quantitative results from N2 (this is load-bearing for the paper reframe):**

| Source | Between-grasp-type hand-pose spread @ final step |
|---|---|
| Demonstrations | 0.352 rad |
| With-taxonomy model | 0.314 rad (89% of demo) |
| Without-taxonomy model | 0.316 rad (90% of demo) |

**What this tells us — paper framing reframe.** The "Without" model *does* commit
to four distinct grasp types. It does **not** crudely collapse to a single mode.
The top-row ribbons show With and W/o tracking the demonstrator mean to within a
few percent on all four types. The PointNet-fused latent already linearly
separates grasp types at kNN=99% with no label (from Session 5 latent viz).

Yet W/o fails ~80% on crimp/sloper/pinch in the paired eval. The mechanism is
**not** "wrong mode selected" — it's a **precision effect**: without the label,
the diffusion decoder samples with 15–30% higher within-input variance. On real
hardware that extra per-rollout wobble misses the hold by a few mm. Means don't
grasp; individual rollouts do.

This is a stronger, more publishable claim than "mode collapse":
- Explains why the effect scales with geometric difficulty (Jug p=0.688 ns, the
  other three highly significant) — Jug has mm-scale slack, Crimp does not.
- Explains why the latent space looks identical with / without conditioning.
- Matches the diffusion-literature pattern that conditioning mostly helps with
  precision, not mode selection.
- Reframes taxonomy conditioning as a **precision regularizer** that tightens the
  sampling distribution around the right subtype, analogous to CFG in image
  diffusion or low-temperature sampling in LLMs.

**RESEARCH_PLAN.md and tasks/todo.md Paper-Framing section updated to reflect this.**
A new `tasks/lessons.md` entry was added: "measure mode structure before asserting
mode collapse — the latent space and per-type means can look fine while sampling
tails are what actually fails on hardware."

**Next high-leverage viz (proposed, not yet run):** a "wrong-label ablation"
figure — run With-taxonomy model with a deliberately swapped label (feed `jug`
when facing a crimp hold, etc.) and show that sampling variance goes back up
AND success rate drops. That would causally pin the precision effect on the
label rather than on any other downstream difference. ~30 eval trials.

### Session 5 (2026-04-17) — Paired Evaluation Tooling

**File created:** `data_collection/paired_eval.py` (~1200 lines)

**Motivation.** Both diffusion policies (`pc_with_taxonomy` and `pc_no_taxonomy`) finished training 2026-04-16. Evaluating them serially with per-model `evaluate.py` runs is biased: the physical hold drifts between a 20-trial WITH-only block and a 20-trial WITHOUT-only block, and within-run operator variation bleeds into only one model. We need both policies evaluated back-to-back on the SAME hold position per pair so only the model identity varies.

**Design decisions (in roughly the order they were made):**

1. **Paired protocol.** Both checkpoints loaded once. For each "pair" the script runs trial 1 with model A, prompts the operator, then runs trial 2 with model B. The first model of pair 1 is a coin-flip; every subsequent pair alternates strictly across the entire session so each model goes first exactly half the time per grasp type. This controls for both order bias and hand-on-hold fatigue.

2. **Per-trial point cloud capture (not shared per pair).** Initial design captured one PC per pair and used it for both trials. First attempt at the real flow made clear that trial 1's hand contact always nudges the hold a few mm — trial 2's model would then be receiving a stale PC that doesn't match the actual scene. Changed to capture a fresh PC before each of the two trials; between trials the operator is prompted to re-align the hold, the arm parks, the 4 cameras re-capture, and the `centroid` + `n_valid` of each PC is logged. After the pair, the script prints the **centroid drift (mm)** between the two scans so high-drift pairs can be filtered out during analysis. Smoke test saw 28.9 mm and 57.4 mm drift across 2 crimp pairs — the 57 mm pair is a clear outlier worth dropping during final analysis.

3. **Batched session model.** A single invocation can evaluate multiple grasp types (crimp, jug, sloper, pinch) in one session without restarting. Each "batch" fixes `(grasp_type, hold_id, n_pairs)`. Three run modes:
   - `paired_eval.py` → interactive: prompts per batch for grasp_type / hold / n_pairs.
   - `paired_eval.py --batches crimp:1:20,jug:0:20,sloper:2:20,pinch:3:20` → scripted.
   - `paired_eval.py --hold 1 --grasp-type crimp --pairs 10` → single-batch, backwards compatible.
   Alternation parity spans the whole session (not per batch) so fairness holds globally.

4. **Clean quit button.** First smoke test ran 2 crimp pairs cleanly, then hit Ctrl-C at the next batch's "Press Enter" prompt → `pyrealsense2` or `FrankaArm` C-extension segfaulted during teardown → the daemon thread that was supposed to write the JSON died before save ran → **2 completed pairs lost**. Fix: `QuitRequested` exception raised by `_wait_or_quit()` at any prompt if user types `q` / `quit`. In the signal and quit handlers, `_save_only()` runs SYNCHRONOUSLY in the main thread BEFORE any hardware teardown. Hardware teardown happens in a daemon thread with a 5-second `join` timeout; if it segfaults, `os._exit(0)` still fires and the JSON is already on disk. A `/tmp/paired_session_<ts>.json` fallback write is attempted if the primary path fails.

5. **Incremental save.** `_save_only()` is called after every completed pair (not just at session end), overwriting `eval_results/paired_session_<session_id>.json` in place. Worst-case loss from any crash/interrupt/power-outage is a single in-progress pair.

6. **Resume support.** `--resume <path>` restores:
   - `session_id` (so the existing file is overwritten, no duplicates)
   - `first_model` (so alternation parity continues: if pair 14 was `WITHOUT→WITH`, pair 15 will correctly be `WITH→WITHOUT`)
   - `_all_pairs` and `_global_pair_idx`
   - `planned_batches` from the original `--batches` spec — fully-completed batches get a `✓` marker and are skipped; partial ones resume at `completed_pairs + 1`
   - `_batch_configs` with per-batch `completed_pairs` counter
   
   Re-invoking `--resume <file>` alone is enough — the saved plan is replayed.

7. **Analysis.** Wilson 95% CI per model and McNemar's paired test (exact binomial for n_discordant < 10, continuity-corrected χ² otherwise), computed both overall and per grasp type. The no-taxonomy model still receives and logs `grasp_type` (the model ignores it internally) so its per-grasp-type performance is measurable.

**Auto-park integration.** Reused the existing `/tmp/franka_park_*` IPC protocol (from `collect_data.py`) so the arm parks for PC capture without killing the FrankaArm `GotoPoseLive` skill. No changes to `VR_Teleoperation_Minimum.py` were needed for paired_eval, since paired_eval runs the policy directly (no VR teleop terminal involved during evaluation).

**Smoke test (2026-04-17, 2 crimp pairs).** WITH taxonomy: 2/2, WITHOUT taxonomy: 0/2. Pattern held regardless of who went first. Centroid drifts: 28.9 mm (pair 1), 57.4 mm (pair 2 — above the ~15 mm drift threshold we'd want for clean apples-to-apples comparisons; may be dropped in final analysis). Data was lost to the segfault bug — motivated fix #4 above. Smoke test cannot be repeated until a full session is run with the fix in place.

**Next step.** Real evaluation session: 20 pairs per grasp type × 4 grasp types = 80 pairs = ~160 rollouts. Expected operator time ~2.5 hours. The `q` quit + `--resume` workflow means this can be spread across multiple sittings.

**Files also updated:** `tasks/todo.md` (workflow description), `tasks/lessons.md` (3 new entries: paired protocol motivation; per-trial PC capture; save-first-on-signal + resume design).

### Session 3 (2026-03-16) — Research Architecture Discussion
- Clarified that 1024-pt XYZ point cloud is appropriate for the diffusion policy (trajectory planning) but marginal for classifying hold grasp type from shape alone (~30-150 points land on the hold itself).
- Confirmed that the diffusion policy does NOT need to predict grasp type — it receives it as a conditioning label. Existing data collection workflow is correct and untouched.
- Decided on a **two-part research architecture**:
  - **Part 1 (Hold Identifier):** Separate VLM-based classifier. Input: RGB image of hold. Output: grasp type label. Trained independently using online climbing hold images (jug/sloper/crimp/pinch are standard industry taxonomy — data is available). Zero-shot VLM is the starting point before fine-tuning.
  - **Part 2 (Grasp Policy):** Existing DP3-style diffusion policy. Unchanged.
- Part 1 data collection is independent: photograph holds (no robot, no zarr). Existing 37 episodes are unaffected and remain valid.
- Updated RESEARCH_PLAN.md Section 5 with full two-part architecture documentation.
- Updated tasks/todo.md with Part 1 as a future workstream.

---
## Commands

### Data Collection (Point Cloud Mode — NEW)
```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
cd /home/rumi/Desktop/tele/data_collection
python3 collect_data.py --hold 0 --point-cloud --grasp-type crimp
```

### Data Collection (Image Mode — Legacy)
```bash
python3 collect_data.py --hold 0
```

### Training (Point Cloud / DP3-style — NEW)
```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
cd /home/rumi/Desktop/tele/data_collection
python3 train.py --point-cloud --epochs 3000 --batch 128 --augment --good-only
```

### Training (Image Mode — Legacy)
```bash
python3 train.py --epochs 3000 --batch 128 --augment --good-only
```

### Evaluation (Point Cloud — NEW)
```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
cd /home/rumi/Desktop/tele/data_collection
python3 evaluate.py --checkpoint ../checkpoints/best.pt --hold 0 --grasp-type crimp
```

### Evaluation (Legacy / Dry Run)
```bash
python3 evaluate.py --checkpoint ../checkpoints/best.pt --hold 0 --dry-run
```

### Paired Evaluation — WITH vs WITHOUT taxonomy (NEW, 2026-04-17)
```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
cd /home/rumi/Desktop/tele/data_collection

# Interactive (prompts per batch). Type 'q' at any prompt to save+quit cleanly.
python3 paired_eval.py

# Scripted (20 pairs per grasp type, 80 pairs / 160 rollouts total)
python3 paired_eval.py --batches crimp:1:20,jug:0:20,sloper:2:20,pinch:3:20

# Resume an interrupted session (restores session_id, first_model, completed pairs, plan)
python3 paired_eval.py --resume eval_results/paired_session_<timestamp>.json
```
