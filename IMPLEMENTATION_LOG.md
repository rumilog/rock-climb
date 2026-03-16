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
