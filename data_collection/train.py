#!/usr/bin/env python3
"""
Diffusion Policy training for climbing-hold grasps.

Reads zarr data collected by collect_data.py and trains a DDPM-based
diffusion policy that predicts action chunks (joint position targets)
conditioned on multi-view images + proprioceptive state.

Architecture:
  observation encoder:
    - Per-camera ResNet-18 (separate weights, GroupNorm, pretrained) → 512-d each
    - ImageNet mean/std normalization on input images
    - State MLP → 256-d
  noise prediction network:
    - 1-D temporal U-Net over action sequence (like Pushing-T paper)
  training:
    - EMA with power-law warmup (matches real-stanford/diffusion_policy)
    - AdamW with 500-step linear LR warmup + cosine decay

Usage:
    python3 train.py                                # defaults
    python3 train.py --epochs 200 --batch 64        # custom
    python3 train.py --zarr ../datasets/climbing_holds.zarr --good-only
"""

import sys
import math
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import zarr
import cv2
from copy import deepcopy

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TELE_ROOT = SCRIPT_DIR.parent
DEFAULT_ZARR = TELE_ROOT / "datasets" / "climbing_holds.zarr"
DEFAULT_CKPT_DIR = TELE_ROOT / "checkpoints"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===========================================================================
# Dataset
# ===========================================================================

class GraspDataset(Dataset):
    """
    Loads zarr dataset and yields (obs, action_chunk) pairs.

    obs = {
        "state":  (T_o, state_dim)      float32   — proprioception history
        "images": (T_o, n_cams, 3, H, W) float32  — multi-view images, normalized
    }
    action_chunk = (T_p, action_dim)    float32   — future action targets
    """

    def __init__(self, zarr_path, obs_horizon=2, pred_horizon=16,
                 good_only=False, hold_ids=None, img_size=224,
                 augment=False, use_rgbd=False):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.img_size = img_size
        self.augment = augment
        self.use_rgbd = use_rgbd

        root = zarr.open(str(zarr_path), mode="r")
        ep_ends = root["meta/episode_ends"][:]
        quality = root["meta/quality"][:] if "quality" in root["meta"] else np.ones(len(ep_ends))
        hold_arr = root["meta/hold_id"][:] if "hold_id" in root["meta"] else np.zeros(len(ep_ends))

        self.states = root["data/state"][:]
        self.actions = root["data/action"][:]

        self.cam_names = sorted(root["data/img"].keys())
        self.has_depth = "depth" in root["data"] and len(root["data/depth"].keys()) > 0

        if use_rgbd and not self.has_depth:
            print("WARNING: --rgbd requested but no depth data in zarr. Using RGB only.")
            self.use_rgbd = False

        self.img_channels = 4 if self.use_rgbd else 3
        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]

        N = self.states.shape[0]
        n_cams = len(self.cam_names)
        ch = self.img_channels
        cache_dir = Path(zarr_path).parent / "img_cache"
        suffix = "rgbd" if self.use_rgbd else "rgb"
        cache_path = cache_dir / f"imgs_{n_cams}x{img_size}_{suffix}_inet.npy"
        shape = (N, n_cams, ch, img_size, img_size)

        if cache_path.exists():
            print(f"Loading image cache: {cache_path} ...")
            self.all_images = np.load(str(cache_path), mmap_mode="r")
            if self.all_images.shape != shape:
                print(f"  Cache shape mismatch {self.all_images.shape} vs {shape}, rebuilding ...")
                cache_path.unlink()
                self.all_images = self._build_image_cache(
                    root, N, n_cams, img_size, cache_dir, cache_path, shape)
            else:
                print(f"  Loaded {N} x {n_cams} images at {img_size}px, {ch}ch (memory-mapped)")
        else:
            self.all_images = self._build_image_cache(
                root, N, n_cams, img_size, cache_dir, cache_path, shape)

        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0).clip(min=1e-6)
        self.action_mean = self.actions.mean(axis=0)
        self.action_std = self.actions.std(axis=0).clip(min=1e-6)

        # Build valid sample indices
        starts = np.concatenate([[0], ep_ends[:-1]])
        self.sample_indices = []
        for ep_i, (s, e) in enumerate(zip(starts, ep_ends)):
            if good_only and quality[ep_i] != 1:
                continue
            if hold_ids is not None and hold_arr[ep_i] not in hold_ids:
                continue
            ep_len = int(e - s)
            min_len = obs_horizon + pred_horizon - 1
            if ep_len < min_len:
                continue
            for t in range(int(s) + obs_horizon - 1, int(e) - pred_horizon + 1):
                self.sample_indices.append(t)

        print(f"Dataset: {len(self.sample_indices)} samples from "
              f"{len(ep_ends)} episodes, state_dim={self.state_dim}, "
              f"cams={self.cam_names}")

    def _build_image_cache(self, root, N, n_cams, img_size, cache_dir, cache_path, shape):
        """Read images from zarr, resize, normalize to float32, save as .npy cache."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        ch = shape[2]
        print(f"Building image cache ({N} x {n_cams} @ {img_size}px, {ch}ch) → {cache_path}")
        print("  This is a one-time cost; subsequent runs load instantly.")

        native_size = root[f"data/img/{self.cam_names[0]}"].shape[1]
        need_resize = (img_size != native_size)

        arr_out = np.empty(shape, dtype=np.float32)
        BATCH = 200
        t0 = time.time()
        for ci, cam in enumerate(self.cam_names):
            zarr_arr = root[f"data/img/{cam}"]
            depth_arr = root[f"data/depth/{cam}"] if self.use_rgbd else None
            for start in range(0, N, BATCH):
                end = min(start + BATCH, N)
                raw = zarr_arr[start:end]  # (batch, H, W, 3) uint8
                raw_depth = depth_arr[start:end] if depth_arr is not None else None
                for j in range(raw.shape[0]):
                    frame = raw[j]
                    if need_resize:
                        frame = cv2.resize(frame, (img_size, img_size),
                                           interpolation=cv2.INTER_AREA)
                    rgb = frame.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)
                    for ch_i in range(3):
                        rgb[ch_i] = (rgb[ch_i] - IMAGENET_MEAN[ch_i]) / IMAGENET_STD[ch_i]

                    if self.use_rgbd and raw_depth is not None:
                        d = raw_depth[j]  # (H, W) uint16
                        if need_resize:
                            d = cv2.resize(d, (img_size, img_size),
                                           interpolation=cv2.INTER_NEAREST)
                        # Normalize depth to [0, 1] range (0-2m typical range)
                        d_norm = np.clip(d.astype(np.float32) * 0.001 / 2.0, 0, 1)
                        arr_out[start + j, ci] = np.concatenate(
                            [rgb, d_norm[np.newaxis]], axis=0)  # (4, H, W)
                    else:
                        arr_out[start + j, ci] = rgb

                elapsed = time.time() - t0
                rate = (ci * N + end) / max(elapsed, 1)
                remaining = ((n_cams * N) - (ci * N + end)) / max(rate, 1)
                print(f"  {cam}: {end}/{N}  "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", end="\r")
            print(f"  {cam}: {N}/{N} done" + " " * 40)

        print(f"  Saving cache ({arr_out.nbytes / 1e9:.1f} GB) ...")
        np.save(str(cache_path), arr_out)
        print(f"  Cache saved. Total build time: {time.time() - t0:.0f}s")
        return np.load(str(cache_path), mmap_mode="r")

    def normalize_state(self, s):
        return (s - self.state_mean) / self.state_std

    def unnormalize_action(self, a):
        return a * self.action_std + self.action_mean

    def normalize_action(self, a):
        return (a - self.action_mean) / self.action_std

    def __len__(self):
        return len(self.sample_indices)

    def _augment_images(self, imgs):
        """Apply random color jitter + random crop to a (To, n_cams, C, H, W) float32 array.
        Images are in ImageNet-normalized space; we un-normalize to [0,1], augment, re-normalize.
        """
        imgs = imgs.copy()
        To, n_cams, C, H, W = imgs.shape
        in_mean = IMAGENET_MEAN.reshape(3, 1, 1)
        in_std = IMAGENET_STD.reshape(3, 1, 1)
        for t in range(To):
            for c in range(n_cams):
                frame = imgs[t, c]  # (C, H, W)
                rgb = frame[:3] * in_std + in_mean  # back to [0,1]

                brightness = np.random.uniform(0.85, 1.15)
                contrast = np.random.uniform(0.85, 1.15)
                rgb = rgb * brightness
                mean = rgb.mean()
                rgb = (rgb - mean) * contrast + mean
                rgb = np.clip(rgb, 0.0, 1.0)

                crop_frac = np.random.uniform(0.85, 1.0)
                crop_size = int(H * crop_frac)
                y0 = np.random.randint(0, H - crop_size + 1)
                x0 = np.random.randint(0, W - crop_size + 1)
                rgb = rgb[:, y0:y0 + crop_size, x0:x0 + crop_size]
                if crop_size != H:
                    frame_hwc = np.transpose(rgb, (1, 2, 0))
                    frame_hwc = cv2.resize(frame_hwc, (W, H), interpolation=cv2.INTER_LINEAR)
                    rgb = np.transpose(frame_hwc, (2, 0, 1))

                rgb = (rgb - in_mean) / in_std  # re-normalize
                if C > 3:
                    depth = frame[3:]
                    depth = depth[:, y0:y0 + crop_size, x0:x0 + crop_size]
                    if crop_size != H:
                        d_hwc = np.transpose(depth, (1, 2, 0))
                        d_hwc = cv2.resize(d_hwc, (W, H), interpolation=cv2.INTER_NEAREST)
                        depth = np.transpose(d_hwc, (1, 2, 0)) if d_hwc.ndim == 3 else d_hwc[np.newaxis]
                    frame = np.concatenate([rgb, depth], axis=0)
                else:
                    frame = rgb

                imgs[t, c] = frame
        return imgs

    def __getitem__(self, idx):
        t = self.sample_indices[idx]
        To, Tp = self.obs_horizon, self.pred_horizon

        obs_states = self.states[t - To + 1: t + 1].copy()
        obs_states = (obs_states - self.state_mean) / self.state_std

        obs_imgs = self.all_images[t - To + 1: t + 1].copy()  # (To, n_cams, 3, H, W)

        if self.augment:
            obs_imgs = self._augment_images(obs_imgs)

        action_chunk = self.actions[t: t + Tp].copy()
        action_chunk = (action_chunk - self.action_mean) / self.action_std

        return {
            "obs_state": torch.from_numpy(obs_states),
            "obs_images": torch.from_numpy(obs_imgs),
            "action": torch.from_numpy(action_chunk),
        }


# ===========================================================================
# Model components
# ===========================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def _group_norm(num_channels, num_groups=8):
    while num_channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(max(num_groups, 1), num_channels)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=5):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            _group_norm(out_ch),
            nn.Mish(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
            _group_norm(out_ch),
            nn.Mish(),
        )
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.blocks(x)
        h = h + self.cond_proj(cond).unsqueeze(-1)
        return h + self.residual(x)


class ConditionalUnet1D(nn.Module):
    """1-D temporal U-Net for denoising action sequences."""

    def __init__(self, action_dim, cond_dim, diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024), kernel_size=5):
        super().__init__()
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        all_dims = [action_dim] + list(down_dims)
        full_cond_dim = cond_dim + diffusion_step_embed_dim

        self.down_blocks = nn.ModuleList()
        for i in range(len(down_dims)):
            self.down_blocks.append(
                ResidualBlock1D(all_dims[i], all_dims[i + 1], full_cond_dim, kernel_size))

        self.mid_block = ResidualBlock1D(down_dims[-1], down_dims[-1], full_cond_dim, kernel_size)

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(down_dims))):
            self.up_blocks.append(
                ResidualBlock1D(all_dims[i + 1] * 2, all_dims[i], full_cond_dim, kernel_size))

        self.final_conv = nn.Sequential(
            nn.Conv1d(action_dim, action_dim, 1),
        )

    def forward(self, x, timestep, cond):
        """
        x:        (B, action_dim, T)  — noisy actions
        timestep: (B,)                — diffusion timestep
        cond:     (B, cond_dim)       — observation encoding
        """
        t_emb = self.diffusion_step_encoder(timestep.float())
        global_cond = torch.cat([cond, t_emb], dim=-1)

        skips = []
        h = x
        for block in self.down_blocks:
            h = block(h, global_cond)
            skips.append(h)

        h = self.mid_block(h, global_cond)

        for block in self.up_blocks:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, global_cond)

        return self.final_conv(h)


def _replace_bn_with_gn(module):
    """Walk a module tree and replace all BatchNorm2d with GroupNorm (16 groups)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_ch = child.num_features
            num_groups = 16
            while num_ch % num_groups != 0:
                num_groups //= 2
            setattr(module, name, nn.GroupNorm(max(num_groups, 1), num_ch))
        else:
            _replace_bn_with_gn(child)


class VisionEncoder(nn.Module):
    """Per-camera ResNet-18 with separate encoders per camera."""

    def __init__(self, n_cams, out_dim=512, pretrained=True, in_channels=3):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone_dim = 512
        self.n_cams = n_cams

        self.cam_backbones = nn.ModuleList()
        for _ in range(n_cams):
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)

            if in_channels != 3:
                old_conv = backbone.conv1
                backbone.conv1 = nn.Conv2d(
                    in_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None)
                if pretrained:
                    with torch.no_grad():
                        backbone.conv1.weight[:, :3] = old_conv.weight
                        if in_channels > 3:
                            backbone.conv1.weight[:, 3:] = old_conv.weight[:, :in_channels - 3]

            features = nn.Sequential(*list(backbone.children())[:-1])
            _replace_bn_with_gn(features)
            self.cam_backbones.append(features)

        self.proj = nn.Linear(self.backbone_dim * n_cams, out_dim)

    def forward(self, imgs):
        """imgs: (B, n_cams, C, H, W) → (B, out_dim)"""
        B = imgs.shape[0]
        feats = []
        for ci in range(self.n_cams):
            x = self.cam_backbones[ci](imgs[:, ci]).squeeze(-1).squeeze(-1)
            feats.append(x)
        x = torch.cat(feats, dim=-1)  # (B, n_cams * 512)
        return self.proj(x)


class ObservationEncoder(nn.Module):
    """Encodes multi-step observations (images + state) → conditioning vector."""

    def __init__(self, state_dim, n_cams, obs_horizon, vision_dim=512, state_mlp_dim=256,
                 img_channels=3):
        super().__init__()
        self.vision_encoder = VisionEncoder(n_cams, vision_dim, in_channels=img_channels)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim * obs_horizon, 256),
            nn.ReLU(),
            nn.Linear(256, state_mlp_dim),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(vision_dim + state_mlp_dim, vision_dim + state_mlp_dim),
            nn.ReLU(),
        )
        self.out_dim = vision_dim + state_mlp_dim

    def forward(self, obs_state, obs_images):
        """
        obs_state:  (B, T_o, state_dim)
        obs_images: (B, T_o, n_cams, 3, H, W)
        """
        B, To = obs_state.shape[:2]
        last_imgs = obs_images[:, -1]
        vis_feat = self.vision_encoder(last_imgs)  # (B, vision_dim)
        state_flat = obs_state.reshape(B, -1)
        state_feat = self.state_mlp(state_flat)  # (B, state_mlp_dim)
        return self.fuse(torch.cat([vis_feat, state_feat], dim=-1))


# ===========================================================================
# Diffusion noise schedule (DDPM)
# ===========================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DDPMScheduler:
    def __init__(self, num_timesteps=100):
        self.num_timesteps = num_timesteps
        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.betas = betas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        posterior_variance = betas * (1.0 - F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = posterior_variance

    def add_noise(self, x0, noise, t):
        s1 = self.sqrt_alphas_cumprod.to(x0.device)[t].reshape(-1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod.to(x0.device)[t].reshape(-1, 1, 1)
        return s1 * x0 + s2 * noise

    def step(self, noise_pred, t_scalar, x_t):
        """Single DDPM reverse step (for inference)."""
        t = t_scalar
        alpha_t = self.alphas_cumprod[t].to(x_t.device)
        alpha_prev = self.alphas_cumprod[t - 1].to(x_t.device) if t > 0 else torch.tensor(1.0, device=x_t.device)
        beta_t = self.betas[t].to(x_t.device)

        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -5.0, 5.0)

        mean = torch.sqrt(alpha_prev) * beta_t / (1 - alpha_t) * x0_pred + \
               torch.sqrt(1 - beta_t) * (1 - alpha_prev) / (1 - alpha_t) * x_t

        if t > 0:
            var = self.posterior_variance[t].to(x_t.device)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(var) * noise
        return mean

    def ddim_step(self, noise_pred, t_scalar, t_prev_scalar, x_t, eta=0.0):
        """Single DDIM reverse step — deterministic when eta=0."""
        alpha_t = self.alphas_cumprod[t_scalar].to(x_t.device)
        alpha_prev = self.alphas_cumprod[t_prev_scalar].to(x_t.device) if t_prev_scalar >= 0 else torch.tensor(1.0, device=x_t.device)

        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -5.0, 5.0)

        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
        x_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt

        if eta > 0 and t_prev_scalar > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)
        return x_prev


# ===========================================================================
# Full policy
# ===========================================================================

class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, n_cams, obs_horizon=2,
                 pred_horizon=16, num_diffusion_steps=100,
                 down_dims=(256, 512, 1024), img_channels=3):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps

        self.obs_encoder = ObservationEncoder(
            state_dim=state_dim, n_cams=n_cams, obs_horizon=obs_horizon,
            img_channels=img_channels)
        self.noise_net = ConditionalUnet1D(
            action_dim=action_dim,
            cond_dim=self.obs_encoder.out_dim,
            down_dims=down_dims,
        )
        self.scheduler = DDPMScheduler(num_diffusion_steps)

    def compute_loss(self, batch):
        obs_state = batch["obs_state"]
        obs_images = batch["obs_images"]
        action = batch["action"]  # (B, Tp, action_dim)

        cond = self.obs_encoder(obs_state, obs_images)

        noise = torch.randn_like(action)
        t = torch.randint(0, self.num_diffusion_steps, (action.shape[0],),
                          device=action.device).long()

        action_t = action.permute(0, 2, 1)  # (B, D, Tp)
        noise_t = noise.permute(0, 2, 1)
        noisy = self.scheduler.add_noise(action_t, noise_t, t)

        pred = self.noise_net(noisy, t, cond)
        return F.mse_loss(pred, noise_t)

    @torch.no_grad()
    def predict_action(self, obs_state, obs_images, num_inference_steps=None):
        """Inference: denoise from pure noise → action chunk.

        If num_inference_steps is set (and < num_diffusion_steps), uses DDIM
        sampling for faster inference. Otherwise uses full DDPM.
        """
        self.eval()
        cond = self.obs_encoder(obs_state, obs_images)
        B = obs_state.shape[0]

        x = torch.randn(B, self.action_dim, self.pred_horizon, device=obs_state.device)

        if num_inference_steps is not None and num_inference_steps < self.num_diffusion_steps:
            # DDIM: subsample timesteps evenly
            step_ratio = self.num_diffusion_steps // num_inference_steps
            timesteps = list(range(self.num_diffusion_steps - 1, -1, -step_ratio))[:num_inference_steps]
            for i, t in enumerate(timesteps):
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)
                noise_pred = self.noise_net(x, t_batch, cond)
                x = self.scheduler.ddim_step(noise_pred, t, t_prev, x, eta=0.0)
        else:
            # Full DDPM
            for t in reversed(range(self.num_diffusion_steps)):
                t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)
                noise_pred = self.noise_net(x, t_batch, cond)
                x = self.scheduler.step(noise_pred, t, x)

        return x.permute(0, 2, 1)  # (B, Tp, action_dim)


# ===========================================================================
# Training loop
# ===========================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    dataset = GraspDataset(
        zarr_path=args.zarr,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        good_only=args.good_only,
        img_size=args.img_size,
        augment=args.augment,
        use_rgbd=args.rgbd,
    )

    if len(dataset) == 0:
        print("ERROR: dataset has 0 valid samples. Collect more data.")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        drop_last=True)

    n_cams = len(dataset.cam_names)
    img_channels = dataset.img_channels
    down_dims = (128, 256, 512) if getattr(args, 'quick', False) else (512, 1024, 2048)
    policy = DiffusionPolicy(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        n_cams=n_cams,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        num_diffusion_steps=args.diffusion_steps,
        down_dims=down_dims,
        img_channels=img_channels,
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Model parameters: {n_params:,}")

    ema_policy = deepcopy(policy)
    ema_policy.eval()
    for p in ema_policy.parameters():
        p.requires_grad_(False)

    ema_step_counter = [0]
    ema_power = args.ema_power
    ema_max_value = 0.9999

    @torch.no_grad()
    def update_ema():
        step = max(0, ema_step_counter[0] - 1)
        if step <= 0:
            decay = 0.0
        else:
            decay = 1 - (1 + step) ** -ema_power
        decay = min(decay, ema_max_value)

        for ema_p, p in zip(ema_policy.parameters(), policy.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)
        ema_step_counter[0] += 1

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                   betas=(0.95, 0.999), weight_decay=1e-6)

    total_steps = len(loader) * args.epochs
    warmup_steps = 500

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    norm_stats = {
        "state_mean": dataset.state_mean.tolist(),
        "state_std": dataset.state_std.tolist(),
        "action_mean": dataset.action_mean.tolist(),
        "action_std": dataset.action_std.tolist(),
    }
    with open(ckpt_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f)

    # Mixed precision training (AMP)
    use_amp = getattr(args, 'amp', False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("  Mixed precision (AMP) enabled")

    print(f"\nTraining: {args.epochs} epochs, {len(loader)} batches/epoch, "
          f"batch={args.batch}")
    print(f"  obs_horizon={args.obs_horizon}, pred_horizon={args.pred_horizon}, "
          f"diffusion_steps={args.diffusion_steps}")
    print(f"  cams={dataset.cam_names}, state_dim={dataset.state_dim}, "
          f"action_dim={dataset.action_dim}\n")

    best_loss = float("inf")
    train_start_time = time.time()
    loss_history = []

    def write_status(epoch, avg_loss, best_loss, loss_history):
        """Write a training_status.md file for remote monitoring."""
        elapsed = time.time() - train_start_time
        elapsed_h = elapsed / 3600
        if epoch > 0:
            eta = elapsed / epoch * (args.epochs - epoch)
            eta_h = eta / 3600
        else:
            eta_h = 0
        status_path = ckpt_dir / "training_status.md"
        with open(status_path, "w") as f:
            f.write(f"# Training Status\n\n")
            f.write(f"**Epoch:** {epoch} / {args.epochs}\n\n")
            f.write(f"**Current loss:** {avg_loss:.6f}\n\n")
            f.write(f"**Best loss:** {best_loss:.6f}\n\n")
            f.write(f"**Elapsed:** {elapsed_h:.1f} hours\n\n")
            f.write(f"**ETA:** {eta_h:.1f} hours\n\n")
            f.write(f"**Dataset:** {args.zarr} ({dataset.state_dim}-dim, "
                    f"{len(dataset)} samples, {len(dataset.cam_names)} cams)\n\n")
            f.write(f"## Recent Losses\n\n")
            f.write(f"| Epoch | Loss |\n|---|---|\n")
            for ep, lo in loss_history[-20:]:
                marker = " *" if lo <= best_loss else ""
                f.write(f"| {ep} | {lo:.6f}{marker} |\n")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_ema()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}  "
              f"lr={lr:.2e}  time={elapsed:.1f}s")
        loss_history.append((epoch, avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_config = {
                "state_dim": dataset.state_dim,
                "action_dim": dataset.action_dim,
                "n_cams": n_cams,
                "cam_names": dataset.cam_names,
                "obs_horizon": args.obs_horizon,
                "pred_horizon": args.pred_horizon,
                "diffusion_steps": args.diffusion_steps,
                "img_size": args.img_size,
                "img_channels": img_channels,
                "down_dims": list(down_dims),
            }
            # Save EMA weights as the primary checkpoint (used at inference)
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema_policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": ckpt_config,
            }, ckpt_dir / "best.pt")

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": policy.state_dict(),
                "loss": avg_loss,
            }, ckpt_dir / f"epoch_{epoch:04d}.pt")

        if epoch % 10 == 0 or epoch == 1:
            write_status(epoch, avg_loss, best_loss, loss_history)

    write_status(args.epochs, avg_loss, best_loss, loss_history)
    torch.save(ema_policy.state_dict(), ckpt_dir / "final.pt")
    print(f"\nDone. Best loss: {best_loss:.6f}  (EMA warmup power={ema_power})")
    print(f"Checkpoints: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train diffusion policy for climbing holds")
    parser.add_argument("--zarr", type=str, default=str(DEFAULT_ZARR))
    parser.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--good-only", action="store_true",
                        help="Train only on episodes marked as good quality")
    parser.add_argument("--augment", action="store_true",
                        help="Apply image augmentation (color jitter, random crop)")
    parser.add_argument("--rgbd", action="store_true",
                        help="Use RGBD (4-channel) images if depth data is available")
    parser.add_argument("--ema-power", type=float, default=0.75,
                        help="EMA warmup power (0.75 matches reference; higher = faster warmup)")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision training (faster on supported GPUs)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check: 96x96 images, 50 epochs, "
                             "smaller U-Net, 20 diffusion steps")
    args = parser.parse_args()

    if args.quick:
        args.img_size = 96
        args.epochs = min(args.epochs, 50)
        args.diffusion_steps = min(args.diffusion_steps, 20)
        args.batch = min(args.batch, 32)
        args.save_every = min(args.save_every, 10)
        print("=== QUICK MODE: 96px images, 20 diffusion steps, 50 epochs ===")

    train(args)


if __name__ == "__main__":
    main()
