# Evaluation Data Description
# For: Data visualization and figure generation
# Session: paired_session_20260417_131412.json

## What this data is

A paired evaluation of two diffusion policies on a real robot (Franka arm + LEAP dexterous hand):
- **WITH_TAXONOMY**: policy conditioned on grasp type label (one-hot, 4-class)
- **WITHOUT_TAXONOMY**: same architecture, grasp type conditioning removed (ablation)

Each "pair" runs both models on the **same physical hold at the same position**, back-to-back,
with the order randomized and strictly alternating across pairs so each model goes first 50% of the time.
A fresh point cloud is captured before each trial.

---

## JSON Structure

```
{
  "session_id": "20260417_131412",
  "with_ckpt": "checkpoints/pc_with_taxonomy/best.pt",
  "no_ckpt": "checkpoints/pc_no_taxonomy/best.pt",
  "batches": [ ... ],   // summary per grasp type
  "pairs": [ ... ]      // one entry per pair, the main data
}
```

### Each pair entry:
```json
{
  "pair": 0,                     // global pair index
  "batch": 0,                    // batch index (0=crimp, 1=jug, 2=sloper, 3=pinch)
  "batch_pair": 0,               // pair index within this batch
  "hold_id": 1,                  // physical hold (0=edge_A, 1=edge_B, 2=sloper, 3=pinch)
  "hold_name": "edge_B",
  "grasp_type": "crimp",         // intended grasp type
  "grasp_type_id": 0,            // 0=crimp, 1=sloper, 2=pinch, 3=jug
  "order": ["WITHOUT_TAXONOMY", "WITH_TAXONOMY"],  // which ran first
  "with_rating": "good",         // human rating: "good" or "bad"
  "no_rating": "bad",
  "pc_stats": {
    "WITHOUT_TAXONOMY": { "n_valid": 1024, "centroid": [x, y, z] },
    "WITH_TAXONOMY":    { "n_valid": 1024, "centroid": [x, y, z] }
  },
  "aborted": false,
  "timestamp": "2026-04-17T13:18:32.651058"
}
```

---

## Data completeness (as of 2026-04-20)

| Batch | Grasp type | Hold | Pairs complete |
|-------|-----------|------|---------------|
| 0     | crimp     | edge_B (hold 1) | 20/20 |
| 1     | jug       | edge_A (hold 0) | 20/20 |
| 2     | sloper    | sloper (hold 2) | IN PROGRESS |
| 3     | pinch     | pinch (hold 3)  | not started |

Pairs 0–19: crimp
Pairs 20–39: jug
Pairs 40+: sloper (ongoing)

---

## Key results summary (for context when building figures)

### Crimp (20 pairs)
- WITH_TAXONOMY: 11/20 good (55%)
- WITHOUT_TAXONOMY: 0/20 good (0%)
- WITHOUT never won a single pair. Observation: collapses to jug-style open hand grip.

### Jug (20 pairs)
- WITH_TAXONOMY: 14/20 good (70%)
- WITHOUT_TAXONOMY: 12/20 good (60%)
- Models perform similarly on jug — the jug grip is what WITHOUT_TAXONOMY defaults to,
  so it accidentally succeeds here.

### Sloper (in progress)
- Early results: WITH_TAXONOMY winning every pair so far
- WITHOUT_TAXONOMY observation: converging to jug-style grip (wrong behavior)

---

## Figures to generate

### Figure 1 — Main result bar chart
- X-axis: grasp type (crimp, jug, sloper, pinch)
- Y-axis: success rate (0–100%)
- Two bars per group: WITH_TAXONOMY (blue) and WITHOUT_TAXONOMY (orange/red)
- Error bars: 95% Wilson confidence intervals
- Add significance markers (Fisher's exact test) above pairs where p < 0.05

### Figure 2 — Per-pair outcome heatmap
- Rows: pairs 0–N
- Columns: WITH_TAXONOMY outcome, WITHOUT_TAXONOMY outcome
- Color: green=good, red=bad
- Group by grasp type with a dividing line
- This shows the paired structure and visually shows WITH winning consistently on crimp

### Figure 3 — Point cloud Z centroid by grasp type
- X-axis: grasp type
- Y-axis: Z centroid (meters)
- Scatter plot of all individual trial centroids
- Color by grasp type
- Shows sloper (~0.034m) and jug (~0.034m) overlap in Z, while crimp (~0.021m) is distinct
- Motivates the "geometry isn't always sufficient" argument

### Figure 4 — Win/tie/loss breakdown
- Stacked bar or pie per grasp type showing:
  - WITH wins (WITH=good, WITHOUT=bad)
  - Tie-good (both good)
  - Tie-bad (both bad)
  - WITHOUT wins (WITHOUT=good, WITH=bad)

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
- `grasp_type_id` in the JSON uses: 0=crimp, 1=sloper, 2=pinch, 3=jug
  (note: sloper=1 and pinch=2 in the ID scheme, not alphabetical)
- Ratings are strings: "good" or "bad" (no numeric scores)
- Each pair has exactly one rating per model
- `aborted: true` pairs should be excluded from analysis
- Point cloud centroids are in robot world frame (meters), roughly:
  x = forward distance from robot base (0.4–0.75m typical)
  y = lateral position (-0.25 to +0.25m typical)
  z = height above table (crimp ~0.021m, jug/sloper ~0.034m)
