# Rock Climb — Grasp-Taxonomy-Aware 3D Diffusion Policy

Trains a DP3-style point cloud diffusion policy conditioned on grasp type (crimp/sloper/pinch/jug)
to autonomously grasp climbing holds with a Franka arm + LEAP Hand.

---

## Quick Start (Training Machine)

### 1. Clone the repo

```bash
git clone https://github.com/rumilog/rock-climb.git tele
cd tele
```

### 2. Create a Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Install PyTorch with CUDA (adjust to match your GPU driver):

```bash
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

### 3. Download the dataset from Hugging Face

```bash
mkdir -p datasets
hf download rlogh/climbing-holds-pointcloud --repo-type dataset --local-dir ./datasets/climbing_holds_rig.zarr
```

(Note: older HuggingFace CLI was `huggingface-cli`; current is `hf`. Both still work.)

Verify the download:

```bash
python3 -c "
import zarr
z = zarr.open('datasets/climbing_holds_rig.zarr', 'r')
print('Episodes:', z['meta/episode_ends'].shape[0])
print('Timesteps:', z['data/state'].shape[0])
print('Point cloud shape:', z['data/point_cloud'].shape)
print('Grasp type IDs:', z['meta/grasp_type_id'][:5], '...')
"
```

Expected output:
```
Episodes: 200
Timesteps: 27821
Point cloud shape: (27821, 1024, 3)
Grasp type IDs: [0 0 0 0 0] ...
```

The rig-era dataset: 200 episodes (50 per grasp type, 10 episodes per orientation × 5 orientations).
Workspace bounds: `z_min=0.027, z_max=0.40` (clipping the spring testbed rig deck while
preserving full hold geometry).

### 4. Run training

**Model A — with grasp taxonomy conditioning:**
```bash
cd data_collection

python3 train.py \
    --point-cloud \
    --zarr ../datasets/climbing_holds_rig.zarr \
    --ckpt-dir ../checkpoints/pc_with_taxonomy_rig \
    --epochs 3000 \
    --batch 128 \
    --augment \
    --good-only \
    --save-every 100
```

**Model B — ablation without taxonomy (same data, no grasp type input):**
```bash
python3 train.py \
    --point-cloud \
    --no-grasp-conditioning \
    --zarr ../datasets/climbing_holds_rig.zarr \
    --ckpt-dir ../checkpoints/pc_no_taxonomy_rig \
    --epochs 3000 \
    --batch 128 \
    --augment \
    --good-only \
    --save-every 100
```

Pre-trained checkpoints are also available on HuggingFace if you want to skip training:
```bash
hf download rlogh/climbing-holds-rig-with-taxonomy --local-dir checkpoints/pc_with_taxonomy_rig
hf download rlogh/climbing-holds-rig-no-taxonomy   --local-dir checkpoints/pc_no_taxonomy_rig
```

Training writes to the checkpoint directory:
- `best.pt` — EMA weights with lowest training loss (use this for evaluation)
- `epoch_XXXX.pt` — periodic snapshots
- `norm_stats.json` — min-max normalization stats (required by evaluate.py)
- `training_status.md` — live progress updated every 10 epochs

### 5. Monitor training

```bash
cat ../checkpoints/pc_with_taxonomy/training_status.md
```

---

## Training Details

| Setting | Value |
|---------|-------|
| Architecture | PointNet encoder + 1D temporal U-Net (DP3-style) |
| PointNet output | 256-d |
| Grasp type conditioning | one-hot(4) → MLP → 64-d, fused with observation |
| Conditioning vector | 512-d (PointNet 256 + State 128 + GraspType 64 + MLP) |
| U-Net dims | (256, 512, 1024) |
| Optimizer | AdamW, lr=1e-4 |
| LR schedule | 500-step cosine warmup |
| EMA | Power-law warmup (power=0.75) |
| Normalization | Min-max to [-1, 1] (DP3 convention) |
| Diffusion | 100-step cosine DDPM (train), 10-step DDIM (inference) |
| Obs horizon | 2 timesteps |
| Pred horizon | 16 timesteps |
| Action horizon | 8 timesteps |
| Action dim | 23 (7 arm joints + 16 hand joints) |
| Point cloud | 1024 pts, XYZ only, world frame, FPS downsampled |
| Dataset | 200 episodes (50 per grasp type × 5 orientations × 10 reps), 27,821 timesteps, 4 holds — spring-testbed rig (z_min=0.027) |

---

## Architecture

```
Point Cloud (1024×3) → PointNet → 256-d
Robot State (2×23)   → MLP     → 128-d
Grasp Type (one-hot) → MLP     → 64-d
                       Concat → MLP → 512-d conditioning vector
                                          ↓
                              DDPM 1D Temporal U-Net
                                          ↓
                           Action chunk (16 × 23-dim)
```

Grasp type IDs: `0=crimp, 1=sloper, 2=pinch, 3=jug`

---

## Evaluation

Run on the robot machine (or same machine if training locally):

```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
cd ~/Desktop/tele/data_collection

# Single model (no pull test)
python3 evaluate.py --checkpoint ../checkpoints/pc_with_taxonomy_rig/best.pt \
    --hold 0 --grasp-type jug

# Spring testbed pull test (13 cm pull at 180°, ratchet readout per trial)
python3 evaluate.py --checkpoint ../checkpoints/pc_with_taxonomy_rig/best.pt \
    --hold 0 --grasp-type jug --pull-dist 0.130
```

For the paired with-vs-without comparison (recommended), use `paired_eval.py`:
```bash
# Full session: 4 pairs × 5 orientations × 4 hold types = 80 paired trials
python3 paired_eval.py --pull-dist 0.130 \
    --batches jug:0:4:-45,jug:0:4:-22.5,jug:0:4:0,jug:0:4:22.5,jug:0:4:45,crimp:1:4:-45,crimp:1:4:-22.5,crimp:1:4:0,crimp:1:4:22.5,crimp:1:4:45,sloper:2:4:-45,sloper:2:4:-22.5,sloper:2:4:0,sloper:2:4:22.5,sloper:2:4:45,pinch:3:4:-45,pinch:3:4:-22.5,pinch:3:4:0,pinch:3:4:22.5,pinch:3:4:45
```

After collection, analyse and display:
```bash
python3 eval_results/display_results.py          # terminal table + figure
python3 eval_results/generate_ratchet_figures.py # 8 force-based figures
python3 eval_results/analyze_observations.py     # failure-mode breakdown + order-effect check
python3 eval_results/generate_training_curves.py # loss-vs-epoch for both models
```

See `HANDOFF.md` §11 for full analysis-reproduction instructions and `OBSERVATIONS.md`
for pre-written paragraphs with all numbers.

---

## Dataset Structure (zarr)

```
climbing_holds.zarr/
  data/
    state        (N, 23)       float32  — arm(7) + hand(16) joint positions
    action       (N, 23)       float32  — same layout, shifted +1 timestep
    point_cloud  (N, 1024, 3)  float32  — clean scene scan per episode, repeated per timestep
    timestamps   (N,)          float64
  meta/
    episode_ends  (E,)  int64
    hold_id       (E,)  int64   — 0=edge_A, 1=edge_B, 2=sloper, 3=pinch, 4=test_edge
    quality       (E,)  int64   — 1=good, 0=bad
    grasp_type    (E,)  str     — "crimp" | "sloper" | "pinch" | "jug"
    grasp_type_id (E,)  int64   — 0=crimp, 1=sloper, 2=pinch, 3=jug
```

Note: images are NOT included in this dataset — the policy uses point clouds only.

---

## Documentation

| File | Purpose |
|------|---------|
| `PAPER_METHODOLOGY.md` | Paper-ready methodology reference (compute, hardware, dataset, training, statistics, reproducibility, limitations) |
| `OBSERVATIONS.md` | Pre-written paper paragraphs with all results numbers |
| `HANDOFF.md` | Technical reference (hardware, file layout, known quirks, analysis reproduction) |
| `RESEARCH_PLAN.md` | Full research design with related work citations |
| `PROGRESS_UPDATE.md` | Slide-ready summary with results table |
| `IMPLEMENTATION_LOG.md` | Chronological code-change log |
| `tasks/todo.md` / `tasks/lessons.md` | Current next steps + lessons learned |

---

## Hardware notes (Franka workstation)

- Cameras 2/3 and 5/4 are 24.5 in apart; cameras 4/3 and 5/2 are 35.75 in apart
- Hold mount sits 7 in from the front of the holder (+X) and 31 cm from the wall on the arm side