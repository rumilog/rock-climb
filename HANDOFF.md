# Technical Handoff: Grasp-Taxonomy-Aware 3D Diffusion Policy (LEAP Hand + Franka)

## Quick-Start Commands

```bash
# Check LEAP hand USB
ls /dev/ttyUSB*

# Start ROS1
bash ~/frankapy/bash_scripts/start_control_pc.sh -u franka -i franka-Alienware-Area-51-R5 -g 0

# Always source both before running any Python
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
```

---

## 1. Project Overview

**Research goal:** Train a **grasp-taxonomy-aware 3D diffusion policy** (DP3-style) that uses point cloud observations and is conditioned on grasp type (crimp/sloper/pinch/jug) to autonomously grasp climbing holds with a LEAP Hand on a Franka arm.

**Current status (2026-03-16):**
- The full point cloud pipeline is implemented end-to-end (collect → store → train → eval)
- **50 good jug episodes collected on hold 0 (edge_A)** — ready for pilot training
- Dataset uploaded to HuggingFace: `rlogh/climbing-holds-pointcloud` (point cloud only, no images)
- Old image-based pipeline exists and is fully backward compatible:
  - `checkpoints/overnight_224/best.pt` — 30-dim, image-based, evaluated on real robot (grasped but not precise)

**Pipeline:**
```
collect_data.py --point-cloud   →   zarr (23-dim state + 1024-pt PC + grasp_type_id)
    ↓
train.py --point-cloud          →   PointCloudDiffusionPolicy checkpoint
    ↓
evaluate.py                     →   auto-detects checkpoint type, runs on robot
```

---

## 2. Hardware

| Component | Details |
|-----------|---------|
| **Arm** | Franka Emika Panda (7-DoF), controlled via `frankapy` |
| **Hand** | LEAP Hand v1 (16 Dynamixel motors), 4 fingers: Index, Middle, Pinky, Thumb |
| **Cameras** | 4× Intel RealSense D415/D455, IDs [2, 3, 4, 5], 848×480 RGB+D |
| **Calibration** | Per-camera intrinsics + world-frame extrinsics stored in robomail |
| **VR Controller** | Meta Quest 2, communicating over UDP |
| **Computer** | Ubuntu, Python 3.8/3.10, NVIDIA GPU, ROS Noetic |

---

## 3. Repository Layout

```
/home/rumi/Desktop/tele/
├── data_collection/
│   ├── collect_data.py         # Main data collection script (Terminal 2)
│   ├── episode_storage.py      # EpisodeBuffer + ZarrDatasetWriter (PC-aware)
│   ├── point_cloud_utils.py    # NEW: PC fusion, FPS, workspace crop, outlier removal
│   ├── hold_detector.py        # 3D hold pose detection from depth (legacy; still works)
│   ├── train.py                # Diffusion policy training (image mode + PC mode)
│   ├── evaluate.py             # Real robot policy evaluation (auto-detects checkpoint type)
│   ├── verify_dataset.py       # Dataset integrity checks
│   └── trim_dataset.py         # Dataset trimming utility
├── TeleoperationUnity/
│   ├── Robot Control - Python/
│   │   └── Franka Scripts/
│   │       ├── VR_Teleoperation_Minimum.py  # Terminal 1: Franka arm + LEAP data forwarding
│   │       └── Teleoperation.py             # Core teleoperation class
│   └── LEAP/leaphandv1/for_transfer/
│       ├── leap_pip_dip_teleop.py           # LeapPipDipTeleop: motor control + UDP
│       ├── UdpComms.py                      # UDP communication
│       └── LEAP_Hand_API/python/
│           └── leap_hand_utils/
│               ├── dynamixel_client.py      # Low-level Dynamixel serial protocol
│               └── leap_hand_utils.py       # Allegro↔LEAP coordinate conversions
├── datasets/
│   ├── climbing_holds.zarr      # 44 episodes, 30-dim state, RGB only (old image pipeline)
│   └── img_cache/               # Memory-mapped numpy image caches (auto-built by train.py)
├── checkpoints/
│   ├── quick_sanity/            # 50-epoch sanity check (96px images)
│   └── overnight_224/           # 300-epoch overnight run (224px), best.pt = real-robot tested
├── RESEARCH_PLAN.md             # Full research design with citations
├── IMPLEMENTATION_LOG.md        # Detailed changelog for the PC pipeline
└── tasks/
    ├── todo.md                  # Current next steps
    └── lessons.md               # Session lessons learned
```

---

## 4. Two-Terminal Architecture (Data Collection — unchanged)

### Terminal 1: `VR_Teleoperation_Minimum.py`
- Controls Franka arm via `GotoPoseLive` (cartesian impedance)
- Receives finger angles from Quest 2 over UDP
- **Forwards finger data** to `localhost:8002` — does NOT control LEAP hand directly

### Terminal 2: `collect_data.py`
- `LeapHandRecorder` wraps `LeapPipDipTeleop`:
  - Opens Dynamixel serial to LEAP hand (auto-detects `/dev/ttyUSB*`)
  - Background thread receives UDP on port 8002 → writes motor commands
  - `get_joint_positions()` returns cached positions (no serial I/O on main thread)
- Main 10 Hz loop: reads Franka arm joints (ROS topic), LEAP joints (cached), cameras
- Records to zarr

### Data Flow
```
Quest 2 → (UDP WiFi) → VR_Teleoperation_Minimum.py [Terminal 1]
                              │
                              ├── Franka arm control (GotoPoseLive)
                              └── Forwards gripper_data → localhost:8002
                                                              │
                                                              ▼
                                            collect_data.py [Terminal 2]
                                              ├── LeapHandRecorder (UDP → motors)
                                              ├── Reads Franka joints via ROS
                                              ├── Reads 4 cameras + optional depth
                                              └── Records to zarr at 10 Hz
```

**Note:** `server_env.py` is NOT part of this pipeline — it is legacy code.

---

## 5. State / Action Space

### New Pipeline (--point-cloud mode) — 23-dim
```
[0:7]   = Franka arm joint positions (7 DoF, radians)
[7:23]  = LEAP hand joint positions (16 DoF, Allegro convention, radians)
```
- **ee_pos and ee_quat are dropped** — they are fully determined by arm joints via FK
- This simplifies the action space and removes a redundant representation
- `_execute_action(action_vec)` in evaluate.py: arm ← `action[:7]`, hand ← `action[7:23]`

### Legacy Image Pipeline — 30-dim (old datasets/checkpoints still work)
```
[0:7]   = Franka arm joint positions
[7:10]  = End-effector position (x, y, z) in meters
[10:14] = End-effector quaternion (w, x, y, z)
[14:30] = LEAP hand joint positions (16 DoF, Allegro convention)
```
- evaluate.py auto-detects which layout to use from the checkpoint's `state_dim` field

### Action Space (both modes)
- Action = next-timestep state (shifted +1 by `finalize_actions()`)
- At evaluation, only the arm joints and hand joints are actually sent to hardware

---

## 6. Key Files in Detail

### `collect_data.py`

**New (--point-cloud) episode flow — 3-step SPACE sequence:**
1. SPACE → "Move arm out of all camera views, then SPACE to capture clean PC"
2. SPACE (arm is clear) → captures fused 4-camera PC (1024 pts, world frame)
3. SPACE (arm returned) → begin recording teleoperation episode

**Flags:**
- `--point-cloud` — enables PC capture + 23-dim state + PC zarr storage
- `--grasp-type crimp|sloper|pinch|jug` — labels all episodes; interactive prompt if omitted with --point-cloud
- `--hold 0..4` — hold ID stored per episode
- `--rgbd` — also stores depth images (not needed for PC mode, but compatible)

**State:** 23-dim `build_state_vector(arm_joints, hand_joints)` — no ee_pos/ee_quat

### `episode_storage.py`

**Zarr layout:**
```
dataset.zarr/
  data/
    state     (N, 23)        float32
    action    (N, 23)        float32
    timestamps (N,)          float64
    img/
      cam2    (N, H, W, 3)   uint8
      ...
    point_cloud (N, 1024, 3) float32   ← NEW: clean scene scan, same PC repeated per episode
  meta/
    episode_ends  (E,)  int64
    hold_id       (E,)  int64    0=edge_A, 1=edge_B, 2=sloper, 3=pinch, 4=test_edge
    quality       (E,)  int64    1=good, 0=bad
    grasp_type    (E,)  str      "crimp" | "sloper" | "pinch" | "jug"
    grasp_type_id (E,)  int64    0=crimp, 1=sloper, 2=pinch, 3=jug     ← NEW
```

`EpisodeBuffer.set_initial_point_cloud(pc)` — set once per episode, stored repeated for all N timesteps.

### `point_cloud_utils.py` (NEW)

Key functions:
- `fuse_multi_camera_points(depth_images, cam_intrinsics, cam_extrinsics, n_points=1024)` — full pipeline: depth → cam-frame points → world frame → workspace crop → outlier removal → FPS
- `farthest_point_sampling(points, n_samples)` — deterministic-ish coverage, better than random
- `crop_to_workspace(points, bounds)` — remove table/background; adjust `DEFAULT_WORKSPACE_BOUNDS` to match your table
- `get_cam_intrinsics_from_realsense(cam_obj)` / `get_cam_extrinsics_from_realsense(cam_obj)` — helpers for robomail `CameraClass` objects

**Important:** `self.cameras.cameras` (on a `ThreadedCameras` instance) is the list of individual `CameraClass` objects — use this to get per-camera calibration.

**Workspace bounds `z_min` history (critical):**
- `z_min=-0.02` (original) → 95% of 1024 points land on flat table, model ignores PC — **DO NOT USE**
- `z_min=0.008` → captures hold but clips hold base detail
- `z_min=0.006` → **correct value**, verified 2026-03-18: full hold geometry, Z centroid ≈ 0.034 m, zero table noise

Always run `check_pc_sensitivity.py` before collecting data to confirm PC looks correct.

### `train.py`

Two modes, selected by `--point-cloud`:

| | Image Mode (default) | Point Cloud Mode (DP3-style) |
|-|---|---|
| Encoder | ResNet-18 per camera → 512-d | PointNet → 256-d |
| Conditioning | Vision + State | PointNet + State + GraspType |
| Normalization | Z-score | Min-max to [-1, 1] |
| Dataset class | `GraspDataset` | `PointCloudGraspDataset` |
| Policy class | `DiffusionPolicy` | `PointCloudDiffusionPolicy` |
| Checkpoint `encoder_type` | `"vision"` | `"point_cloud"` |

**PC mode architecture** (matches RESEARCH_PLAN exactly):
```
PointNet(1024×3) → 256-d
State(obs_horizon × 23) → MLP → 128-d
GraspType(one-hot 4) → MLP → 64-d
Concat → MLP → 512-d conditioning vector
DDPM 1D U-Net → action chunk (16 × 23-dim)
```

**Norm stats:** saved to `norm_stats.json` alongside checkpoint. Contains `"normalization": "minmax"` or `"zscore"` so evaluate.py can detect which to apply.

**Image cache:** `datasets/img_cache/*.npy` — delete to force rebuild if dataset changes.

### `evaluate.py`

- `load_policy(ckpt_path, device)` — reads `cfg["encoder_type"]` from checkpoint, instantiates either `DiffusionPolicy` or `PointCloudDiffusionPolicy`
- Reads `"normalization"` from `norm_stats.json` — handles both min-max and z-score
- Reads state using `build_state_vector_23` (23-dim) or `build_state_vector_30` (30-dim) based on checkpoint's `state_dim`
- PC trial flow: arm out → SPACE → capture PC → arm back → SPACE → start policy
- `--grasp-type` arg required for PC checkpoints (or interactive prompt)
- **Inference speed:** default `--inference-steps 100` (full DDPM). For PC mode use `--inference-steps 10` (DDIM, per RESEARCH_PLAN recommendation) for faster 10 Hz execution
- **Pull test** (`--pull-dist`, `--pull-angle`): after the rollout ends, arm executes a Cartesian displacement of `pull_dist` meters in the X,Y plane at `pull_angle` degrees. You then visually confirm whether the hold moved with the arm (`g`/`b`). If `--pull-angle` is not set, you are prompted for the angle before each pull. See CLAUDE.md "Pull test angle convention" for the coordinate diagram. Pull angle and distance are logged per trial in the result JSON.

### `paired_eval.py` (canonical tool for WITH-vs-WITHOUT-taxonomy comparison)

Created 2026-04-17. Loads BOTH checkpoints once at startup, shares all robot hardware, and runs interlaced paired trials so both models see the same physical hold position per pair. This is the intended tool for the final comparison experiment (RESEARCH_PLAN §Ablations).

Key design choices:

- **Per-trial point cloud capture** — fresh PC is taken before each of the two trials in a pair (not once per pair). Trial 1's hand contact always nudges the hold a few mm; between-trial operator prompt + re-scan keeps the comparison faithful. Per-trial centroid + `n_valid` count are logged, and centroid drift (mm) is printed at pair end so outlier pairs can be filtered post-hoc.
- **Strict alternation across all pairs** — pair 1's first model is a coin-flip; every subsequent pair (across all batches in the session) alternates, so each model starts first exactly half the time per grasp type. Alternation parity is preserved on resume.
- **Batched session model** — a session consists of one or more batches, each fixing `(grasp_type, hold_id)` for N pairs. You can run 20 crimp, then 20 jug, then 20 sloper, then 20 pinch in a single invocation.
- **Three modes:** interactive (prompt per batch), scripted (`--batches crimp:1:20,jug:0:20,...`), single-batch (`--hold 1 --grasp-type crimp --pairs 10`).
- **Clean quit / save-and-walk-away** — type `q` + Enter at any "Press Enter ..." prompt (between pairs, between batches, or between the two trials of a pair). The script saves the session JSON synchronously (in the main thread, before any hardware teardown, so a pyrealsense/Franka exit segfault can never clobber it), tears down hardware in a daemon thread with 5 s timeout, prints the exact `--resume` command, and exits.
- **Incremental save** — `_save_only()` is called after every completed pair, overwriting `eval_results/paired_session_<session_id>.json` in place. Worst-case data loss from any crash/interrupt is a single in-progress pair.
- **`--resume <path>`** — restores `session_id` (to overwrite same file), `first_model` (to preserve alternation parity), per-batch `completed_pairs`, and the original `planned_batches` spec. Fully-completed batches are skipped, partial batches resume at `completed_pairs + 1`. Re-invoking `--resume` alone is enough — no need to repeat `--batches`.
- **Analysis** — success rates + Wilson 95% CIs + McNemar's paired test, printed both overall and per grasp type. The no-taxonomy model still receives `grasp_type` in the JSON (ignored by the model internally) so its per-grasp-type performance is measurable.

Saved JSON fields: `session_id`, `mode`, `first_model`, `planned_batches`, `batches` (each with `completed_pairs`), `pairs` (each with `grasp_type`, `hold_id`, `order`, `with_rating`, `no_rating`, `pc_stats` (including `pull_angle_deg` per trial if pull test was used), `pull_dist_m`, `timestamp`), `last_saved`.

**Pull test** (`--pull-dist`, `--pull-angle`): same as evaluate.py. After each rollout the arm executes a Cartesian displacement; you visually confirm whether the hold moved (`g`/`b`). Prompted per trial if `--pull-angle` is omitted (recommended — holds rotate between trials). See CLAUDE.md "Pull test angle convention" for the coordinate diagram.

Default checkpoints: `checkpoints/pc_with_taxonomy/best.pt` and `checkpoints/pc_no_taxonomy/best.pt`. Override with `--with-ckpt` / `--no-ckpt`.

---

## 7. Full Workflow

### Data Collection (Point Cloud mode)
```bash
source ~/franka/bin/activate && source ~/frankapy/catkin_ws/devel/setup.bash

# Terminal 1
cd ~/Desktop/tele/"TeleoperationUnity/Robot Control - Python/Franka Scripts"
python3 VR_Teleoperation_Minimum.py

# Terminal 2
cd ~/Desktop/tele/data_collection
python3 collect_data.py --hold 0 --point-cloud --grasp-type crimp
# SPACE flow: arm out → SPACE → PC captured → arm back → SPACE → record → g/b/d
```

### Training (Point Cloud / DP3-style)
```bash
cd ~/Desktop/tele/data_collection
python3 train.py --point-cloud --epochs 3000 --batch 128 --augment --good-only \
    --zarr ../datasets/climbing_holds.zarr --ckpt-dir ../checkpoints/pc_v1
```

### Evaluation (Point Cloud, single model)
```bash
cd ~/Desktop/tele/data_collection
python3 evaluate.py --checkpoint ../checkpoints/pc_v1/best.pt \
    --hold 0 --grasp-type crimp
# --inference-steps defaults to 10 DDIM (correct for 10 Hz); use 100 only for offline comparison

# With disturbance rejection pull test (5 cm pull, angle prompted each trial):
python3 evaluate.py --checkpoint ../checkpoints/pc_v1/best.pt \
    --hold 0 --grasp-type crimp --pull-dist 0.05
```

### Paired Evaluation (WITH vs WITHOUT taxonomy — the main comparison experiment)
```bash
cd ~/Desktop/tele/data_collection

# Recommended: interactive mode — prompts for grasp_type/hold/pairs per batch
python3 paired_eval.py

# Or scripted, for a planned full session (20 pairs per hold = 80 total pairs, ~160 rollouts):
python3 paired_eval.py --batches crimp:1:20,jug:0:20,sloper:2:20,pinch:3:20

# With disturbance rejection pull test (angle prompted per trial):
python3 paired_eval.py --batches crimp:1:20,jug:0:20,sloper:2:20,pinch:3:20 --pull-dist 0.05
```

During the run, type `q` + Enter at any "Press Enter ..." prompt to save and
exit cleanly (no Ctrl-C needed). To continue later:
```bash
python3 paired_eval.py --resume eval_results/paired_session_<timestamp>.json
```
Alternation parity, completed batches, and planned batches are all restored
so the resume is seamless. Every pair is saved incrementally, so any crash /
power-loss at worst loses the single in-progress pair.

### Legacy Image Pipeline (still works)
```bash
# Collection (no --point-cloud)
python3 collect_data.py --hold 0

# Training
python3 train.py --zarr ../datasets/climbing_holds.zarr --epochs 300 --batch 64 --img-size 224

# Eval (auto-detects image checkpoint)
python3 evaluate.py --checkpoint ../checkpoints/overnight_224/best.pt --hold 0
```

**Quest 2 IP:** update in `VR_Teleoperation_Minimum.py` line 72/74 when it changes.

---

## 8. Training History

| Checkpoint | Dataset | State | Mode | Epochs | Notes |
|-----------|---------|-------|------|--------|-------|
| `checkpoints/quick_sanity/best.pt` | climbing_holds_legacy_image.zarr | 30-dim | Image | 50 | Sanity check only |
| `checkpoints/overnight_224/best.pt` | climbing_holds_legacy_image.zarr | 30-dim | Image | 300 | Real-robot tested: arm approached hold but grasp imprecise |
| `checkpoints/pc_large/best.pt` | ~~climbing_holds.zarr~~ (DELETED) | 23-dim | PC | 5000 | **INVALID** — trained on bad PC data (z_min=-0.02, 95% table noise). Robot did same motion regardless of hold position or zero-PC input. Checkpoint must not be used. |

**Dataset status (2026-03-18):**
- `datasets/climbing_holds_legacy_image.zarr` — 44 eps, 30-dim, image-only. **VALID.** Use for RGB ablation baseline.
- `datasets/climbing_holds_v2.zarr` — 73 eps, 30-dim, image-only. **VALID.** Legacy.
- `datasets/climbing_holds.zarr` — **DOES NOT EXIST** — old 50-ep PC dataset was deleted (bad z_min). Will be recreated by fresh collection with z_min=0.006.

**Next training run:** recollect 50 jug episodes → retrain PC model → verify robot uses PC (zero-PC vs real-PC actions must differ).

---

## 9. Known Issues & Quirks

1. **`server_env.py` is NOT used** — legacy file; some comments still reference it.
2. **Dead code in `leap_pip_dip_teleop.py`** — lines 358-407 are unreachable after `return` on line 349/356. Harmless but confusing.
3. **robomail import warnings** on startup (tensorflow, pylibfreenect2, etc.) — harmless. Cameras work fine.
4. **Force-killing `collect_data.py`** can lock `/dev/ttyUSB*`. Always try graceful `q` or single Ctrl+C first.
5. **`DEFAULT_WORKSPACE_BOUNDS`** in `point_cloud_utils.py` — adjust these x/y/z ranges to match your actual table/hold placement. Wrong bounds → empty point clouds.
6. **Image cache** in `datasets/img_cache/` — delete to force rebuild when dataset changes.
7. **Quest 2 IP** — changes periodically on WiFi. Update `VR_Teleoperation_Minimum.py` lines 72/74.
8. **PC mode inference speed** — default is 10 DDIM steps (correct for 10 Hz control). Only pass `--inference-steps 100` if doing offline quality comparison; it will miss the control loop timing on real hardware.
9. **Process exit can segfault** — on this box, `pyrealsense2` and/or `FrankaArm` C-extension teardown occasionally segfaults during process exit (observed 2026-04-17 in paired_eval.py). Harmless as long as your script SAVES DATA BEFORE hardware teardown and uses `os._exit(0)` after a bounded-timeout cleanup thread. `paired_eval.py` follows this pattern and additionally incrementally-saves after every pair; `evaluate.py` (single-model, short runs) has not triggered it in practice. If you write new tools: save first, teardown second, never rely on `atexit` alone.

---

## 10. Current Goals (Priority Order)

1. **Run pilot training on cluster** — 50 jug episodes on hold 0 are ready (see README for commands)
2. **Evaluate pilot on robot** — load checkpoint, run evaluate.py, confirm PC inference works end-to-end
3. **Collect data for holds 1–3** — crimp, sloper, pinch (50 good episodes each)
4. **Scale training** — full dataset across all hold types
5. **Run ablations** — with/without grasp type conditioning; PC vs RGB baseline; 1024 vs other point counts

**Research target:** RESEARCH_PLAN.md describes the full research design, related work citations, and the two-part architecture (VLM identifier + diffusion policy).
