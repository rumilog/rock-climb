# Rock Climb — Diffusion Policy for Climbing Hold Grasps

Trains a DDPM diffusion policy (ResNet-18 vision encoder + 1D temporal U-Net) to
autonomously grasp climbing holds using a Franka arm + LEAP hand.

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

Install PyTorch with CUDA (adjust the CUDA version to match your GPU driver):

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
pip install huggingface_hub
mkdir -p datasets
huggingface-cli download rlogh/climbing-holds-v2 --repo-type dataset --local-dir ./datasets/climbing_holds_v2.zarr
```

Verify the download:

```bash
python3 -c "
import zarr
z = zarr.open('datasets/climbing_holds_v2.zarr', 'r')
print('Episodes:', z['meta/episode_ends'].shape[0])
print('Timesteps:', z['data/state'].shape[0])
for cam in sorted(z['data/img'].keys()):
    print(f'  {cam}: {z[\"data/img/\" + cam].shape}')
"
```

Expected output:
```
Episodes: 73
Timesteps: 9923
  cam2: (9923, 224, 224, 3)
  cam3: (9923, 224, 224, 3)
  cam4: (9923, 224, 224, 3)
  cam5: (9923, 224, 224, 3)
```

### 4. Run training

```bash
cd data_collection

python3 train.py \
    --zarr ../datasets/climbing_holds_v2.zarr \
    --ckpt-dir ../checkpoints/v2_fixed \
    --epochs 600 \
    --batch 64 \
    --img-size 224 \
    --diffusion-steps 100 \
    --augment \
    --good-only \
    --amp \
    --save-every 50
```

Training writes checkpoints to `../checkpoints/v2_fixed/`:
- `best.pt` — EMA weights with lowest loss (used for evaluation)
- `epoch_XXXX.pt` — periodic snapshots of training weights
- `norm_stats.json` — dataset normalization statistics
- `training_status.md` — live training progress (updated every 10 epochs)

### 5. Monitor training

Check `training_status.md` for live progress:

```bash
cat ../checkpoints/v2_fixed/training_status.md
```

Or watch the terminal output — each epoch prints loss, learning rate, and timing.

---

## Training Details

| Setting | Value |
|---------|-------|
| Architecture | Per-camera ResNet-18 (GroupNorm) + 1D temporal U-Net |
| U-Net dims | (512, 1024, 2048) |
| Optimizer | AdamW, lr=1e-4, betas=(0.95, 0.999) |
| LR schedule | 500-step linear warmup + cosine decay |
| EMA | Power-law warmup (power=0.75, max=0.9999) |
| Image normalization | ImageNet mean/std |
| Diffusion | 100-step cosine beta schedule (DDPM) |
| Obs horizon | 2 timesteps |
| Pred horizon | 16 timesteps |
| Action dim | 30 (7 arm + 3 pos + 4 quat + 16 hand) |
| Cameras | 4x Intel RealSense (224x224 RGB) |
| Dataset | 73 episodes, 9923 timesteps, hold 0 only |

---

## Copying Checkpoints Back

After training, copy the checkpoint back to the robot machine for evaluation:

```bash
scp -r checkpoints/v2_fixed/ user@robot-machine:/path/to/tele/checkpoints/v2_fixed/
```

Then on the robot machine:

```bash
cd data_collection
python3 evaluate.py --checkpoint ../checkpoints/v2_fixed/best.pt --hold 0
```
