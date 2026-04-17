# TODO

## Current State (2026-04-14)

- ✅ Point cloud pipeline implemented end-to-end
- ✅ Workspace bounds calibrated: `z_min=0.006` verified with `check_pc_sensitivity.py`
- ✅ **200 episodes collected** — 50 per grasp type, all marked good quality
  - Hold 0 (edge_A): 50 jug episodes
  - Hold 1 (edge_B): 50 crimp episodes
  - Hold 2 (sloper): 50 sloper episodes
  - Hold 3 (pinch): 50 pinch episodes
- ✅ `--no-grasp-conditioning` flag implemented in train.py (ablation-ready)
- ✅ Per-grasp LEAP hand offsets saved in memory (jug, sloper, pinch, crimp)
- ✅ DIP/PIP max clamp added to `leap_pip_dip_teleop.py`
- ✅ Verified training fits on local RTX 2080 Ti (0.8 GB VRAM, ~11 hours for 3000 epochs)

---

## Immediate Next Steps

1. **Train Model A — WITH taxonomy** (on this machine)
   ```bash
   cd ~/Desktop/tele/data_collection
   python3 train.py --point-cloud --epochs 3000 --batch 128 --augment --good-only \
       --zarr ../datasets/climbing_holds.zarr --ckpt-dir ../checkpoints/pc_with_taxonomy \
       --save-every 100
   ```

2. **Train Model B — WITHOUT taxonomy** (ablation, on this machine)
   ```bash
   cd ~/Desktop/tele/data_collection
   python3 train.py --point-cloud --no-grasp-conditioning --epochs 3000 --batch 128 \
       --augment --good-only --zarr ../datasets/climbing_holds.zarr \
       --ckpt-dir ../checkpoints/pc_no_taxonomy --save-every 100
   ```

3. **Evaluate BOTH models on robot** — 20+ trials per grasp type per model (160+ total)

   Both checkpoints finished training (2026-04-16). Use `paired_eval.py`
   — one terminal command runs both models back-to-back on the SAME hold
   position per pair, with a FRESH point cloud captured before each
   trial (trial 1's hand contact always nudges the hold a few mm; the
   between-trial reposition prompt + re-scan keeps the comparison
   faithful). The first model of pair 1 is randomised; order strictly
   alternates across every subsequent pair (across all batches) so each
   model goes first half the time per grasp type.

   **Interactive (recommended):** prompts for grasp_type / hold / n_pairs
   before each batch, so you can swap between, say, 5 crimps → 5 jugs →
   5 slopers → 5 pinches without restarting:

   ```bash
   cd ~/Desktop/tele/data_collection
   python3 paired_eval.py
   ```

   **Scripted (for planned sessions):**
   ```bash
   python3 paired_eval.py --batches crimp:1:20,jug:0:20,sloper:2:20,pinch:3:20
   ```

   **Single batch (backwards-compatible):**
   ```bash
   python3 paired_eval.py --hold 1 --grasp-type crimp --pairs 10
   ```

   **Clean quit / save-and-walk-away:** at any "Press Enter ..." prompt
   (between pairs, between batches, or between trials in a pair), type
   `q` + Enter instead of just Enter. The script saves, tears down
   hardware, and prints the exact `--resume` command to continue later.
   No Ctrl-C needed, no segfault risk.

   **Resume:**
   ```bash
   python3 paired_eval.py --resume eval_results/paired_session_<ts>.json
   ```
   The script restores `session_id`, `first_model`, the completed-pair
   history, and (for scripted runs) the original `--batches` plan. It
   then skips already-complete batches and picks up partial batches at
   the next pair. Alternation continues correctly — so if pair 14 is
   `WITHOUT → WITH`, pair 15 after resume will be `WITH → WITHOUT`.

   Every pair is saved **incrementally** (not just at end) with
   `grasp_type`, `hold_id`, `order`, per-trial `pc_stats` (n_valid +
   centroid), and both ratings to
   `eval_results/paired_session_<timestamp>.json`. Worst-case data loss
   from any crash/interrupt is a single in-progress pair. Each pair
   prints its centroid drift (mm) between the two trials' PCs for
   drift auditing. The no-taxonomy model receives `grasp_type` only for
   internal logging (it ignores it), so its per-grasp-type performance
   is still measurable. At the end the script prints success rates +
   Wilson 95% CIs + McNemar p-values both overall and per grasp type.

   Record per-trial: grasp success (0/1), grasp type correctness, hold stability, contact time.

4. **Build results table and figures** — see RESEARCH_PLAN.md §3 Evaluation Metrics

5. **Upload dataset to HuggingFace** (for reproducibility / cluster backup)
   ```bash
   huggingface-cli upload rlogh/climbing-holds-pointcloud \
       ./datasets/climbing_holds.zarr --repo-type dataset
   ```

---

## Training Parameter Audit (2026-04-14)

All parameters verified against DP3 (Ze et al., RSS 2024), original Diffusion Policy
(Chi et al., RSS 2023), and DexCap (Wang et al., RSS 2024).

| Parameter | Value | Source / Justification |
|-----------|-------|----------------------|
| obs_horizon | 2 | Chi et al. 2023, DP3 — standard for diffusion policy |
| pred_horizon | 16 | Chi et al. 2023 — standard for CNN-based DP |
| action_horizon | 8 | Chi et al. 2023 — execute 8 of 16 predicted steps, then re-plan |
| diffusion_steps | 100 (train), 10 DDIM (inference) | DP3, Chi et al. — standard |
| batch_size | 128 | DP3 — fits in 0.8 GB VRAM on RTX 2080 Ti |
| learning_rate | 1e-4 | DP3, Chi et al. — standard for AdamW |
| AdamW betas | (0.95, 0.999) | DP3 — β1=0.95 is standard for diffusion models |
| weight_decay | 1e-6 | Standard — minimal regularization |
| grad_clip | 1.0 | Standard for diffusion training — prevents exploding gradients |
| warmup | 500 steps | DP3 — linear warmup before cosine decay |
| LR schedule | cosine decay after warmup | Standard |
| EMA power | 0.75 | Power-law warmup EMA, matches reference implementations |
| EMA max | 0.9999 | Standard |
| epochs | 3000 | DP3 — sufficient for convergence on ~200 episodes |
| PointNet dims | 3→64→128→256→proj(256) | Simplified PointNet (no T-Net, sufficient for fixed-frame PCs) |
| U-Net down_dims | (256, 512, 1024) | Chi et al. 2023 — standard for original DP |
| normalization | min-max to [-1, 1] | DP3 recommendation for point cloud mode |
| PC augmentation | jitter σ=0.002 + 5% dropout | DP3 uses σ=0.002; dropout is standard |
| n_points | 1024 | DP3 default |
| grasp_type_dim | 64-d (one-hot→MLP) | Our design — 64-d is sufficient for 4-class embedding |

**Note on U-Net dims:** README says (512, 1024, 2048) but code uses (256, 512, 1024).
The smaller dims match Chi et al. 2023 and are appropriate for 200 episodes — the larger
dims risk overfitting on this dataset size. Scale up if dataset grows to 500+ episodes.

**Note on validation:** No train/val split — best.pt saved on lowest training loss.
This matches DP3 and Chi et al. Real validation is on-robot success rate.
Training loss should decrease smoothly; a spike/plateau after initial decrease
could indicate learning rate issues.

---

## Evaluation Protocol (2026-04-14)

### Per-trial recording
For EVERY evaluation trial, record:
1. **Grasp success** — binary: did the hand achieve stable contact? (0/1)
2. **Grasp type correctness** — did the hand form the correct grip shape? (0/1)
3. **Hold stability** — can the grasp sustain a gentle tug? (0/1)
4. **Contact time** — seconds from episode start to first contact
5. **Point cloud used** — save the PC for post-hoc analysis

### Trial counts
- Per model × per grasp type: **minimum 20 trials** (ideally 30)
- Total: 2 models × 4 grasp types × 20 trials = **160 trials minimum**
- Held-out test hold (hold 4, test_edge): 20 trials per model = 40 more

### Statistical analysis
- Report success rates with **95% confidence intervals** (Wilson score)
- Use **Fisher's exact test** for pairwise comparisons (p < 0.05 = significant)
- Primary figure of merit: **Δ success rate** (taxonomy model − no-taxonomy model)

### Key figures to generate
1. **Bar chart**: success rate per grasp type, two bars per type (with/without taxonomy)
2. **Training curves**: loss vs epoch for both models on same plot
3. **Point cloud visualizations**: example PCs for each hold type
4. **Architecture diagram**: PointNet + State + GraspType → Fuse → U-Net
5. **Novelty gap table**: our method vs DP3, Dexonomy, etc. across 4 axes

---

## Part 1: Hold Identifier (Separate Workstream — Future)

**Goal:** VLM-based classifier that predicts grasp type from an RGB image of a hold.
**Status:** Not started. Existing Part 2 data unaffected — this is fully independent.

---

## Part 2 Ablations

- [x] Add `--no-grasp-conditioning` flag to train.py (2026-04-14)
- [ ] Train with-taxonomy model (3000 epochs, ~11 hours)
- [ ] Train without-taxonomy model (3000 epochs, ~11 hours)
- [ ] Evaluate both models per grasp type on robot → build comparison table
- [ ] Point cloud vs RGB (ResNet baseline — legacy zarrs preserved for this)
- [ ] 1024 vs 512 vs 2048 points (if time permits)
