# TODO

## Current State (2026-05-15) — PRIMARY EVALUATION COMPLETE

- ✅ Point cloud pipeline implemented end-to-end
- ✅ `--no-grasp-conditioning` flag implemented in train.py (ablation-ready)
- ✅ Per-grasp LEAP hand offsets saved in memory (jug, sloper, pinch, crimp)
- ✅ DIP/PIP max clamp added to `leap_pip_dip_teleop.py`
- ✅ Verified training fits on local RTX 2080 Ti (0.8 GB VRAM, ~11 hours for 3000 epochs)
- ✅ Spring displacement testbed built (linear single-axis pull toward robot base, ratcheting peak-displacement readout)
- ✅ Workspace bounds re-calibrated for spring testbed: `z_min=0.027`, `z_max=0.40` in `point_cloud_utils.py` (was `z_min=0.006`, `z_max=0.30` on the bare table)
- ✅ Impedance pull: kx=4000 / ky=100 / kz=2000 N/m (Z stiffened to hold LEAP hand weight)
- ✅ Both rig-trained checkpoints ready: `pc_with_taxonomy_rig/best.pt`, `pc_no_taxonomy_rig/best.pt`
- ✅ **PRIMARY PAIRED EVALUATION COMPLETE** (2026-05-15)
  - 80 pairs: 4 pairs × 5 orientations × 4 hold types
  - Session: `eval_results/paired_session_20260515_074256.json`
  - Primary metric: ratchet slip force (N) — spring testbed, 13 cm pull
  - Results (Wilcoxon signed-rank, paired):
    - Crimp:  WITH 15.7 N  vs WITHOUT 9.2 N  | Δ+6.5 N | p=0.0007 | d=1.05 ***
    - Jug:    WITH 15.7 N  vs WITHOUT 14.4 N | Δ+1.3 N | p=0.0278 | d=0.55 *
    - Sloper: WITH  7.9 N  vs WITHOUT 5.2 N  | Δ+2.6 N | p=0.0012 | d=1.08 **
    - Pinch:  WITH 15.7 N  vs WITHOUT 9.2 N  | Δ+6.5 N | p<0.001  | d=1.82 ***
    - OVERALL: WITH 13.1 N vs WITHOUT 7.9 N  | Δ+5.2 N | p<0.001  | d=0.95 ***
  - Ratchet figures: `eval_results/figures/fig_ratchet_1–4.png`
- ⚠️ Legacy checkpoints (`pc_with_taxonomy/`, `pc_no_taxonomy/`) superseded — pre-rig data

**Plan reframe (2026-05-11):** training and evaluation both happen on the spring
testbed. Training = teleoperate the LEAP hand + Franka to grip the hold (no pull
during collection); 10 episodes × 5 orientations (−45°, −22.5°, 0°, +22.5°, +45°)
× per hold = 50 episodes per hold. The 5 orientations are pure pose diversity in
the training distribution — they are NOT stored in the zarr or consumed by the
policy. Eval = run the policy to grasp, then pull at 180° (the rig's fixed axis)
in position control. The linear ratchet records the peak displacement; slip
force is computed offline via `F = k · x`. No force or displacement signal is
fed into the policy — both live only in the eval JSON / lab notebook.

Per-orientation eval analysis (if wanted) is achieved by randomizing the hold
orientation per eval trial and logging it in `paired_eval.py` at the time of
each pair — not by storing it in the training zarr.

---

## Immediate Next Steps (paper completion)

### 1. Wrong-label ablation (highest priority)
Run ~10 pairs per mislabeled condition to prove conditioning causally controls behavior.
Give each model the WRONG grasp type label while evaluating the correct physical hold.
```bash
# Example: jug hold (hold 0) but tell both models it's a crimp
python3 paired_eval.py --pull-dist 0.130 --batches jug:0:10:0
# Then at the batch start, override the grasp_type label manually in the JSON, OR
# edit paired_eval.py to pass a wrong grasp_type_id to the models
```
Hypothesis: WITH-taxonomy model executes wrong grip shape → low ratchet force.
WITHOUT-taxonomy model ignores label → same force as correct-label condition.
This proves the conditioning signal is causal, not a noise term.

### 2. Held-out hold generalization
Get a new physical crimp hold never seen in training. Evaluate taxonomy-conditioned
model only with `grasp_type=crimp`. No demos needed — eval only.
```bash
python3 evaluate.py --checkpoint ../checkpoints/pc_with_taxonomy_rig/best.pt \
    --hold 4 --grasp-type crimp --pull-dist 0.130 --trials 20
```
Tests whether model learned a general crimp strategy vs memorized edge_B geometry.

### 3. Regenerate latent/action/novelty figures from rig checkpoints
The existing figures in `eval_results/figures/` were made from the pre-rig checkpoints.
Update the checkpoint paths in each script and rerun:
```bash
cd ~/Desktop/tele/data_collection
# edit generate_latent_viz.py / generate_action_dist_viz.py / generate_novelty_viz.py
# to point at checkpoints/pc_with_taxonomy_rig/best.pt + checkpoints/pc_no_taxonomy_rig/best.pt
python3 ../eval_results/generate_latent_viz.py
python3 ../eval_results/generate_action_dist_viz.py
python3 ../eval_results/generate_novelty_viz.py
```

### 4. Training curves
Plot loss vs epoch from the training logs:
- `checkpoints/pc_with_taxonomy_rig_train.log`
- `checkpoints/pc_no_taxonomy_rig/train.log`

5. **Wrong-label ablation** — after main eval is done
   - Run paired_eval with deliberately incorrect taxonomy labels
   - e.g. give `grasp_type=jug` when facing the crimp hold, `grasp_type=crimp` when facing the sloper
   - Hypothesis: model executes wrong grasp type → fails
   - This proves the conditioning signal is causally controlling behavior, not just a noise term
   - ~10 pairs per mislabeled condition is sufficient

3. **Held-out hold generalization** — get any NEW crimp hold never seen during training
   - Do NOT collect training demos — evaluation only
   - Run taxonomy-conditioned model with `grasp_type=crimp` on new physical hold
   - Tests whether model learned general crimp strategy vs memorized edge_B geometry
   - This is hold 4 (test_edge) from the original plan — just needs a physical hold acquired

4. **Build results table and figures** — see `eval_results/eval_data_description.md` for visualization spec

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
1. **Grasp success (disturbance rejection)** — after rollout converges, arm moves 5 cm in a specified X,Y direction (`--pull-dist 0.05`); success = hold visibly moves with the arm (0/1). This is the primary success criterion.
2. **Grasp type correctness** — did the hand form the correct grip shape? (0/1)
3. **Contact time** — seconds from episode start to first contact

**How to run with pull test:**
```bash
python3 paired_eval.py --pull-dist 0.130
```
Pull angle is hardcoded at 180° (toward robot base, -X). After each pull, enter the ratchet tooth count (0–11) when prompted; the script computes and logs displacement + force. See CLAUDE.md "Spring testbed protocol" for the force table.

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

## Ablation Checklist

- [x] Add `--no-grasp-conditioning` flag to train.py (2026-04-14)
- [x] Train with-taxonomy model (finished 2026-04-16)
- [x] Train without-taxonomy model (finished 2026-04-16)
- [ ] Complete paired eval — sloper + pinch batches
- [ ] Wrong-label ablation (~10 pairs per mislabeled condition)
- [ ] Held-out hold generalization (test_edge, new physical crimp hold)
- [ ] Point cloud vs RGB (ResNet baseline — legacy zarrs preserved for this)
- [ ] 1024 vs 512 vs 2048 points (if time permits)

---

## Paper Framing (reframed 2026-04-20 after novelty viz)

**Target venue:** CoRL (ambitious) or RA-L/ICRA (realistic fallback)

**Core claim (reframed — precision regularizer, NOT mode collapse):**
Both with- and without-taxonomy diffusion policies trained on mixed grasp-type
data learn a representation that separates grasp types (latent kNN 99% without
the label) AND produce mean action predictions that commit to four distinct
grasp configurations (90% of demonstrator between-type spread). The unconditioned
model does **not** crudely collapse to a single behaviour mode.

The failure mode is a **precision effect**: without explicit taxonomy conditioning,
the diffusion decoder samples with 15–30% higher *within-input* variance — the
means are right but individual rollouts are jittery. On real dexterous hardware
this extra per-rollout wobble misses the hold. The taxonomy label acts as a
**precision regularizer**, tightening the sampling distribution around the
correct subtype (analogous to classifier-free guidance in image diffusion or
low-temperature sampling in LLMs), not as a mode selector.

Evidence:
- Paired eval: W/o ≈ 80% failure on crimp/sloper/pinch, ≈ same as With on jug.
  Jug has mm-scale slack, crimps don't — exactly the geometric-difficulty
  pattern a precision-regularizer account predicts.
- `fig_novelty_mode_commitment.png`: W/o means track demo means; W/o between-type
  spread = 0.316 rad ≈ With 0.314 rad ≈ Demos 0.352 rad.
- `fig_action_distribution.png` Panel B: W/o within-input sampling std is
  15–30% higher on non-jug types, same on jug.
- Latent (`fig_latent_4panel.png`): W/o fused encoder gets silhouette ≈ 0.32 and
  kNN ≈ 99% with no conditioning info — class structure is already in the
  representation.

**What makes this publishable (reframed):**
1. Benchmark — climbing holds with 4 taxonomically distinct grasp types on real dexterous hardware
2. Paired evaluation protocol — controls for hold position drift, randomised order, McNemar's test
3. **New mechanistic finding** — taxonomy label as a precision regularizer for dexterous diffusion policies, with joint-space + latent-space + sampling-variance evidence
4. Wrong-label ablation — if label is a precision knob around the *correct* subtype, a wrong label should both move the mean AND re-inflate the variance. Pending.
5. Held-out generalisation — shows conditioning enables within-type generalisation to new hold instances

**What NOT to claim:**
- Do NOT claim "the unconditioned model collapses to one mode". Empirically false — it produces four distinct means at 90% of demo spread.
- Do NOT claim VLM-in-the-loop is novel (DexGraspVLA, DexVLA, OmniDexVLG already do this).
- Do NOT overclaim the architecture — it is DP3 + grasp-type conditioning, applied to a new domain. The *mechanistic* finding (precision, not mode) is what earns publication.

**Key differentiator from Dexonomy (RSS 2025):** They generate static grasp poses.
We learn continuous visuomotor execution trajectories, and we characterise what
the taxonomy label is actually doing to the diffusion decoder. Different problem entirely.

---

## Visualisation / Figure Status (2026-04-20)

### Generated figures (in `eval_results/figures/`)
- `fig1_success_rates.png` — readability-fixed
- `fig2_pair_heatmap.png` — rotated + readability-fixed
- `fig3_z_centroid.png` — annotation-fixed
- `fig4_win_tie_loss.png` — p-value placement fixed, legend moved out
- `fig5_delta_summary.png` — table sizing fixed
- `fig_latent_4panel.png` — with silhouette + kNN metrics in titles
- `fig_latent_with_vs_without.png` — with metrics
- `fig_latent_trajectories.png` — PCA of raw joint trajectories
- `fig_latent_eval_overlay.png` — paired-eval outcomes over action PCA
- `fig_action_distribution.png` — per-input sampling variance + between-model L2
- `fig_novelty_benchmark.png` — 4-column benchmark overview
- `fig_novelty_mode_commitment.png` — ribbons + final-pose fingerprint (load-bearing reframe evidence)

### Scripts
- `eval_results/generate_figures.py` — fig1–fig5
- `eval_results/generate_latent_viz.py` — latent + trajectory + eval-overlay
- `eval_results/generate_action_dist_viz.py` — action-distribution comparison
- `eval_results/generate_novelty_viz.py` — novelty / mode-commitment figures

### Proposed next viz (not yet run)
- **Wrong-label fingerprint.** Same `fig_novelty_mode_commitment.png` layout but
  with deliberately swapped labels. If label is a precision knob, expect both
  (a) mean shifts towards the wrong-subtype mean and (b) sampling variance
  partially re-inflates. Strongest single causal figure for the reframe.
  Requires ~30 physical eval trials.
