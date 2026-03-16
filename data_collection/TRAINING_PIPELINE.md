# Training Pipeline — Diffusion Policy for Climbing Hold Grasps

Full reference for data collection, training, and evaluation. Written after the March 2026 improvements.

---

## Overview

The policy is a **Diffusion Policy** (DDPM):
- **Vision**: ResNet-18 per camera, pre-extracted features (frozen, ImageNet-pretrained)
- **Policy**: 1D temporal U-Net, channel dims `(256, 512, 1024)` or `(512, 1024, 2048)`
- **Inputs**: 4 cameras (`cam2–5`) + robot joint state (30-dim or 36-dim with hold pose)
- **Output**: Action chunk of 16 predicted joint targets, 8 of which are executed per step

---

## Data Collection

### Standard pipeline (old — 30-dim state)

```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
cd ~/Desktop/tele/data_collection

python collect_data.py --hold 0 --task climbing_holds
```

State vector: `arm_joints(7) + ee_pos(3) + ee_quat(4) + hand_joints(16) = 30`

**Problem**: robot grasps the correct hold *type* but lands a few cm off the exact location.

---

### New pipeline (recommended — 36-dim state with hold pose scan)

```bash
python collect_data.py --hold 0 --task climbing_holds_v2 --scan-hold
```

State vector: same 30-dim + `hold_centroid(3) + hold_normal(3) = 36`

**Workflow per episode:**
1. Press **SPACE** → prompted to move hand out of the way
2. Press **SPACE** again → 1-second depth scan fires (5 frames averaged), hold 3D pose is detected
3. Recording starts automatically → move hand back and begin demonstration
4. Press **SPACE** to stop, then `g` (good) / `b` (bad) / `d` (discard)

The 6D hold pose (centroid + surface normal in camera frame) is appended to every timestep's state, giving the policy explicit spatial grounding for where the hold is in 3D space.

**Other useful flags:**
```bash
--cameras 2 3 4 5     # which cameras to use (default: all four)
--freq 10             # recording Hz (default: 10)
--rgbd                # also store depth images alongside RGB
--no-franka           # skip arm (cameras-only dry run)
```

---

## Training

### Environment setup (training machine)
```bash
# torch is installed under python3.10 (not the system python3 which is 3.8)
python3.10 --version  # should say 3.10.x
```

### Standard training run

```bash
cd ~/Desktop/tele
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3.10 -u data_collection/train.py \
    --epochs 3000 --amp --batch 64 --workers 0 \
    2>&1 | tee checkpoints/train.log &
```

Monitor:
```bash
tail -f ~/Desktop/tele/checkpoints/train.log
cat ~/Desktop/tele/checkpoints/training_status.md
```

### Key training decisions

| Choice | Value | Reason |
|---|---|---|
| Epochs | 3000 | Vision features are frozen so per-epoch compute is low; 3000 epochs = ~2.5 hrs and converges well |
| Batch size | 64 | Fits GPU (2080 Ti, 11GB) with frozen ResNet |
| U-Net channels | `(512, 1024, 2048)` | Larger capacity; run2_large_unet achieved best loss (0.0016 vs 0.0038 baseline) |
| obs_horizon | 2 | Two observation timesteps per policy step |
| pred_horizon | 16 | Predicts 16 steps ahead, executes 8 |
| Vision | Frozen ResNet-18, pre-extracted | 224px image cache is 15.8GB — too large for 16GB RAM. Features are 54MB, fit trivially |
| AMP | Yes | Mixed precision, ~2x faster |

### Why frozen ResNet features?

The 224px image cache (15.8GB) doesn't fit in the 16GB RAM available on this machine (Chrome + Slack + Cursor consume ~6GB). Loading it from disk each epoch via memmap caused ~20-30 min/epoch (GPU at 0% utilization). Pre-extracting ResNet features to a 54MB file brings epoch time to ~3 seconds.

Tradeoff: ResNet weights don't adapt to the task. Mitigated by using 4 cameras + obs_horizon=2 (8 ResNet feature vectors per step).

### Resume training

```bash
python3.10 -u data_collection/train.py \
    --epochs 5000 --amp --batch 64 --workers 0 \
    --resume checkpoints/run2_large_unet/best.pt \
    2>&1 | tee -a checkpoints/train.log
```

### Overnight multi-run experiments

Checkpoints from the March 2026 overnight run:

| Run | Location | Config | Best loss |
|---|---|---|---|
| Baseline | `checkpoints/run1_baseline/` | U-Net (256,512,1024), pred=16 | 0.003812 |
| **Best** | `checkpoints/run2_large_unet/` | U-Net (512,1024,2048), pred=16 | **0.001566** |
| Short pred | `checkpoints/run3_pred8/` | U-Net (256,512,1024), pred=8 | 0.004271 |

Use `run2_large_unet/best.pt` for robot evaluation.

---

## Evaluation on Robot

### Environment setup
```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
cd ~/Desktop/tele/data_collection
```

Note: use `python` (not `python3.10`) inside the franka conda environment.

### Run evaluation

```bash
# Old model (30-dim state, no hold scan) — auto-detected:
python evaluate.py --checkpoint ../checkpoints/run2_large_unet/best.pt --hold 0 --trials 5

# New model (36-dim state, with hold scan) — auto-detected:
python evaluate.py --checkpoint ../checkpoints/NEW_RUN/best.pt --hold 0 --trials 5

# Override auto-detection explicitly:
python evaluate.py --checkpoint ... --scan-hold no   # force skip scan
python evaluate.py --checkpoint ... --scan-hold yes  # force scan

# Dry run (cameras only, no robot):
python evaluate.py --checkpoint ... --dry-run

# Faster inference (DDIM, ~5x speedup, slight quality tradeoff):
python evaluate.py --checkpoint ... --inference-steps 20
```

The `--scan-hold` flag defaults to `auto`, which reads `state_dim` from the checkpoint:
- `state_dim=30` → no scan (old pipeline)
- `state_dim=36` → scan hold at startup (new pipeline)

### Evaluation workflow
1. Robot moves to reset pose
2. If new pipeline: hold depth is scanned automatically at startup
3. Press **SPACE** to start each trial
4. Policy runs for up to 200 steps (20 seconds at 10 Hz)
5. Press `g` / `b` / `s` to rate the grasp (good / bad / skip)
6. Results saved to `eval_results/eval_<timestamp>_hold<id>.json`

---

## Architecture Summary

```
Images (B, T_o=2, 4 cams, 3, 224, 224)
    → ResNet-18 per camera (frozen, ImageNet-pretrained)
    → 512-dim feature per camera per timestep
    → concat all: 2 × 4 × 512 = 4096-dim visual condition

State (B, T_o=2, 30 or 36)
    → linear projection
    → concat with visual condition → obs_cond

Noise (B, T_p=16, action_dim=30)
    → 1D U-Net with FiLM conditioning on obs_cond
    → DDPM/DDIM denoising (100 or 20 steps)
    → predicted action chunk (16 steps)
    → execute first 8 steps on robot
```
