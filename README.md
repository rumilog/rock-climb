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
huggingface-cli download rlogh/climbing-holds-pointcloud --repo-type dataset --local-dir ./datasets/climbing_holds.zarr
```

Verify the download:

```bash
python3 -c "
import zarr
z = zarr.open('datasets/climbing_holds.zarr', 'r')
print('Episodes:', z['meta/episode_ends'].shape[0])
print('Timesteps:', z['data/state'].shape[0])
print('Point cloud shape:', z['data/point_cloud'].shape)
print('Grasp type IDs:', z['meta/grasp_type_id'][:5], '...')
"
```

Expected output:
```
Episodes: 50
Timesteps: 8381
Point cloud shape: (8381, 1024, 3)
Grasp type IDs: [3 3 3 3 3] ...
```

(grasp_type_id=3 is jug — this is a jug-only pilot dataset on hold 0)

### 4. Run training

```bash
cd data_collection

python3 train.py \
    --point-cloud \
    --zarr ../datasets/climbing_holds.zarr \
    --ckpt-dir ../checkpoints/pc_pilot \
    --epochs 3000 \
    --batch 128 \
    --augment \
    --good-only \
    --save-every 100
```

Training writes to `../checkpoints/pc_pilot/`:
- `best.pt` — EMA weights with lowest validation loss (use this for evaluation)
- `epoch_XXXX.pt` — periodic snapshots
- `norm_stats.json` — min-max normalization stats (required by evaluate.py)
- `training_status.md` — live progress updated every 10 epochs

### 5. Monitor training

```bash
cat ../checkpoints/pc_pilot/training_status.md
```

---

## Training Details

| Setting | Value |
|---------|-------|
| Architecture | PointNet encoder + 1D temporal U-Net (DP3-style) |
| PointNet output | 256-d |
| Grasp type conditioning | one-hot(4) → MLP → 64-d, fused with observation |
| Conditioning vector | 512-d (PointNet 256 + State 128 + GraspType 64 + MLP) |
| U-Net dims | (512, 1024, 2048) |
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
| Dataset | 50 episodes, 8381 timesteps, hold 0 (jug) — pilot |

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

## Copying Checkpoints Back

After training, copy the checkpoint back to the robot machine for evaluation:

```bash
scp -r checkpoints/pc_pilot/ user@robot-machine:/path/to/tele/checkpoints/pc_pilot/
```

Then on the robot machine:

```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
cd ~/Desktop/tele/data_collection
python3 evaluate.py --checkpoint ../checkpoints/pc_pilot/best.pt --hold 0 --grasp-type jug
```

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
