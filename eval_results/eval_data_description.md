# Evaluation Data Description
# For: Data visualization and figure generation
# Primary session: paired_session_20260515_074256.json  (80 pairs, complete, rig-trained checkpoints)
# Historical session: paired_session_20260417_131412.json  (partial, pre-rig checkpoints, kept for reference)

## What this data is

A paired evaluation of two diffusion policies on a real robot (Franka arm + LEAP dexterous hand):
- **WITH_TAXONOMY**: policy conditioned on grasp type label (one-hot, 4-class)
- **WITHOUT_TAXONOMY**: same architecture, grasp type conditioning removed (ablation)

Each "pair" runs both models on the **same physical hold at the same position**, back-to-back,
with model order strictly alternating across pairs so each model goes first 50% of the time.
A fresh point cloud is captured before each individual trial (not shared within a pair).

The 2026-05-15 session is the **primary evaluation** — 80 pairs / 160 rollouts at all 5 hold
orientations (−45°, −22.5°, 0°, +22.5°, +45°) across 4 grasp types, using the spring-testbed
ratchet readout (slip force in Newtons) as the primary metric.

---

## JSON Structure (current schema, 2026-05-15)

```
{
  "session_id":               "20260515_074256",
  "mode":                     "scripted",
  "with_ckpt":                "checkpoints/pc_with_taxonomy_rig/best.pt",
  "no_ckpt":                  "checkpoints/pc_no_taxonomy_rig/best.pt",
  "max_steps":                200,
  "action_horizon":           8,
  "inference_steps":          10,
  "pull_dist_m":              0.13,
  "pull_stiffness":           4000.0,
  "pull_lateral_stiffness":   100.0,
  "pull_z_stiffness":         2000.0,
  "pull_z_bias":              0.0,
  "first_model":              "WITH_TAXONOMY" | "WITHOUT_TAXONOMY",
  "planned_batches":          [["jug", 0, 4, -45.0], ["jug", 0, 4, -22.5], ...],
  "batches":                  [ ... ],   // summary per batch (grasp_type × hold × orientation × n_pairs)
  "pairs":                    [ ... ],   // one entry per pair — the main data
  "last_saved":               "2026-05-15T07:42:56..."
}
```

### Each pair entry:
```json
{
  "pair": 0,                       // global pair index (0..79)
  "batch": 0,                      // batch index (one per orientation × hold combo)
  "batch_pair": 0,                 // pair index within this batch
  "hold_id": 0,                    // physical hold (0=edge_A, 1=edge_B, 2=sloper, 3=pinch)
  "hold_name": "edge_A",
  "grasp_type": "jug",             // intended grasp type
  "grasp_type_id": 3,              // 0=crimp, 1=sloper, 2=pinch, 3=jug
  "orientation_deg": -45.0,        // hold orientation relative to pull axis
  "order": ["WITH_TAXONOMY", "WITHOUT_TAXONOMY"],   // which model ran first
  "with_rating": "good",           // binary g/b rating — DEGENERATE in this session (all "good", do not analyse)
  "no_rating":   "good",
  "pc_stats": {
    "WITH_TAXONOMY": {
      "n_valid": 1024,
      "centroid": [x, y, z],
      "pull_angle_deg": 180.0,
      "ratchet": {                 // PRIMARY METRIC — present on all 80 pairs
        "teeth":           4,      // 0..11 (operator-entered, 9.3 mm per tooth)
        "displacement_mm": 37.2,
        "force_lbf":       3.523,
        "force_N":         15.67
      }
    },
    "WITHOUT_TAXONOMY": { ...same fields... }
  },
  "aborted": false,
  "timestamp": "2026-05-15T07:42:56...",
  "pull_dist_m":  0.13,
  "pull_angle_deg": 180.0,
  "pull_stiffness": { "kx": 4000.0, "ky": 100.0, "kz": 2000.0 }
}
```

---

## Data completeness (as of 2026-05-15) — PRIMARY SESSION COMPLETE

`paired_session_20260515_074256.json`: 80 paired trials, **all complete**.
Each (grasp_type, orientation) cell has exactly 4 paired trials:

| Grasp / orientation | −45° | −22.5° | 0° | +22.5° | +45° | Total |
|---|---|---|---|---|---|---|
| jug    (hold 0) | 4 | 4 | 4 | 4 | 4 | 20 |
| crimp  (hold 1) | 4 | 4 | 4 | 4 | 4 | 20 |
| sloper (hold 2) | 4 | 4 | 4 | 4 | 4 | 20 |
| pinch  (hold 3) | 4 | 4 | 4 | 4 | 4 | 20 |
| **Total** | | | | | | **80** |

---

## Key results summary (for context when building figures)

**Primary metric: ratchet slip force (continuous, Newtons).** Wilcoxon signed-rank, paired,
two-sided. Cohen's d on paired differences with bootstrap 95% CI.

| Grasp | n  | WITH median | WITHOUT median | Δ median | Wilcoxon p | Cohen's d |
|-------|----|-------------|----------------|----------|------------|-----------|
| Crimp | 20 | 15.7 N      | 9.2 N          | +6.5 N   | 0.0007     | 1.05 ***  |
| Jug   | 20 | 15.7 N      | 14.4 N         | +1.3 N   | 0.0278     | 0.55 *    |
| Sloper| 20 | 7.9 N       | 5.2 N          | +2.6 N   | 0.0012     | 1.08 **   |
| Pinch | 20 | 15.7 N      | 9.2 N          | +6.5 N   | <0.001     | 1.82 ***  |
| **Overall** | **80** | **13.1 N** | **7.9 N** | **+5.2 N** | **<0.001** | **0.95 ***  |

**Failure-mode breakdown** (force buckets: complete ≤5.3 N, weak 5.3–13.5 N, moderate 13.5–23.0 N, strong >23 N):

| Bucket           | WITH | WITHOUT |
|------------------|------|---------|
| Complete failure | 10%  | 44%     |
| Weak             | 44%  | 35%     |
| Moderate         | 38%  | 20%     |
| Strong           |  9%  |  1%     |

**Per-pair winners:** 61 WITH / 14 tie / 5 WITHOUT (pinch was a perfect 20/20 WITH sweep).

**Methodological controls passed:** Mann–Whitney U on order effect → p > 0.69 both models;
no pegged-ratchet (right-censored) trials.

---

## ⚠️ Binary `with_rating` / `no_rating` field — DO NOT USE for analysis

The binary g/b rating was the original primary metric. During this evaluation, the operator
entered the ratchet tooth count after every pull and stopped using the binary rating as a
discriminative judgement — every trial was rated `"good"`. As a result:

- `with_rating == "good"` for all 80 pairs in this session
- `no_rating == "good"` for all 80 pairs in this session
- Any McNemar's test or Fisher's exact test on these fields will return "no discordant pairs"
- All analyses MUST use `pc_stats[model].ratchet.force_N` as the primary metric

The binary fields are kept for backward-compatibility with the original eval JSON schema
and for any future session that wants to use a discriminative binary on top of the ratchet
reading. They are degenerate in `paired_session_20260515_074256.json`.

---

## Figures to generate (current pipeline)

All scripts live in `eval_results/`. Run from the repo root.

### Primary force-based figures (`generate_ratchet_figures.py`)
- `fig_ratchet_1_boxplots.png` — main result figure: box+jitter per grasp type, WITH vs WITHOUT, Wilcoxon brackets
- `fig_ratchet_2_scatter.png` — combined per-pair scatter (one dot per pair)
- `fig_ratchet_3_by_orientation.png` — line plot of mean force vs orientation per grasp type
- `fig_ratchet_4_per_pair_delta.png` — signed bar per pair (WITH − WITHOUT)
- `fig_ratchet_5_cdf.png` — empirical CDF of slip force per grasp type
- `fig_ratchet_6_effect_sizes.png` — Cohen's d bars with bootstrap 95% CI
- `fig_ratchet_7_per_grasp_scatter.png` — 4-panel head-to-head per grasp type
- `fig_ratchet_8_orientation_heatmap.png` — force × orientation × model heatmap

### Observation analysis (`analyze_observations.py`)
- `fig_obs_1_failure_modes.png` — stacked-bar failure-mode breakdown per grasp type
- `fig_obs_2_variance.png` — IQR comparison per grasp type
- `fig_obs_3_order_check.png` — order-effect Mann–Whitney validation
- `fig_obs_4_orientation_dive.png` — per-orientation × grasp deep dive

### Display / table (`display_results.py`)
- `fig_results_table.png` — publication-quality results table + Cohen's d bar chart

### Training curves (`generate_training_curves.py`)
- `fig_training_curves.png` — loss vs epoch for both models (linear + log-scale panels)

### Legacy binary-based figures (`generate_figures.py`) — DEPRECATED for this session
- The binary g/b analysis is degenerate (all "good") and these figures are no longer informative.
  Only `fig3_z_centroid.png` (which uses PC centroid Z, independent of g/b rating) remains valid.

---

## Grasp type ID mapping
- 0 = crimp (curled fingers on thin edge)
- 1 = sloper (open-hand friction press)
- 2 = pinch (thumb + fingers on narrow feature)
- 3 = jug (full hand wrap on large hold)

## Hold ID mapping
- 0 = edge_A (jug hold)
- 1 = edge_B (crimp hold)
- 2 = sloper
- 3 = pinch
- 4 = test_edge (held-out crimp, not yet evaluated)

## Notes for the visualization AI

- **Primary metric:** `pc_stats[model].ratchet.force_N` (continuous, range 5.2–33.9 N)
- **Statistical test:** Wilcoxon signed-rank, paired, two-sided (use `scipy.stats.wilcoxon`)
- **Effect size:** Cohen's d on paired differences (mean(diff) / std(diff)). Bootstrap 95% CI via percentile method, 2000 iterations.
- **Excluded pairs:** any with `aborted: true` (none in this session) or `ratchet is None` (none in this session)
- `grasp_type_id` in the JSON uses: 0=crimp, 1=sloper, 2=pinch, 3=jug (sloper=1 and pinch=2 in the ID scheme, not alphabetical)
- `orientation_deg` is the hold orientation relative to the pull axis: −45 / −22.5 / 0 / +22.5 / +45
- Point cloud centroids are in robot world frame (meters): x = forward, y = lateral, z = height above rig deck
- **Spring formula:** `F_lbf = 1.18 + 1.6 × (teeth × 9.3 / 25.4)` (two springs in parallel); `F_N = F_lbf × 4.448`
- The binary `with_rating` / `no_rating` fields are degenerate in this session (all "good") — DO NOT analyse
