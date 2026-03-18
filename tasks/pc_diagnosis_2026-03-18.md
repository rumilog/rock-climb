# Point Cloud Pipeline Diagnosis — 2026-03-18

**Requesting second opinion on:** whether the current z_min bug is fatal to the trained model, or whether the model can still extract hold position from sparse hold points amid table noise.

---

## What We Observed on the Robot

Trained a DP3-style diffusion policy (PointNet encoder, 1024-pt PC, grasp-type conditioning) on 50 jug episodes. During eval, the robot did the **exact same motion regardless of where the climbing hold was placed**. This prompted the diagnostic below.

---

## Diagnostic: `check_pc_sensitivity.py`

Moved the arm to a park pose (fully out of camera view), captured the full point cloud pipeline output (same code as `evaluate.py`) with the hold in 3 different positions.

### Terminal output — centroids barely move

| Run | Hold position | Centroid X | Centroid Y | Centroid Z | Valid pts |
|-----|--------------|-----------|-----------|-----------|-----------|
| pos_a | center of workspace | 0.5754 | 0.0220 | -0.0064 | 1024/1024 |
| pos_b | same area, different orientation | 0.5746 | 0.0231 | -0.0065 | 1024/1024 |
| pos_c | moved significantly (different corner) | 0.5761 | 0.0251 | -0.0063 | 1024/1024 |

Centroid X range across all three: **1.5 mm**. Physical displacement was ~20+ cm.

### Z histogram (pos_a representative)

```
Z ∈ [-0.020, -0.011)  →  359 pts  (35%)  ← below table
Z ∈ [-0.011, -0.002)  →  490 pts  (48%)  ← table surface
Z ∈ [-0.002, +0.007)  →  122 pts  (12%)  ← table surface
Z ∈ [+0.007, +0.069)  →   53 pts   (5%)  ← actual hold
```

**~95% of the 1024 FPS points land on the flat table. ~50 points are on the hold.**

### Root cause

`DEFAULT_WORKSPACE_BOUNDS` had `z_min = -0.02`. The table surface is at Z ∈ [-0.02, 0.006]. FPS samples the entire flat workspace uniformly, spending 95% of its budget on table. The hold gets ~50 points.

---

## The Counterargument (why this might not be fatal)

Looking at the top-down PNGs tells a different story from the centroid numbers:

- **pos_a / pos_b**: Hold point cluster appears center-right of workspace
- **pos_c**: Hold point cluster jumps to upper-left of workspace — clearly a different location

The ~50 hold points **do** encode the hold's position. The centroid is just a bad summary statistic because it's dominated by the 950 table points.

**The question is: can PointNet extract hold position from 50 signal points amid 950 noise points?**

Arguments **for** it working:
- PointNet uses local feature extraction + global max-pooling. Max-pooling is specifically designed to pick out the most distinctive features — the hold cluster may be more distinctive than flat table points.
- 50 geometrically coherent points in a cluster is non-trivial signal.
- The model trained for 5000 epochs with loss 0.000560 — it did *learn something*.

Arguments **against** it working:
- 5% signal-to-noise is terrible. Any position-sensitive PointNet feature has to survive being max-pooled against 950 table points.
- The pre-random-sampling step (`np.random.choice(len_pts, 20000)`) before FPS introduces randomness — the 50 hold points could be under-sampled on any given capture.
- **Bigger problem:** During training, the hold was always at the **same fixed location** (50 jug episodes, all on `hold 0 / edge_A`). The model never needed to learn position conditioning — it could have just memorized the trajectory and ignored the PC entirely.
- The robot doing the same thing in every eval trial is more consistent with "model ignores PC" than "model can't parse sparse PC."

---

## Fixes Applied

### 1. z_min fix
Raised `z_min` from `-0.02` to `0.008` (2mm above the table surface at Z=0.006) in `point_cloud_utils.py`. This forces all 1024 FPS points onto the hold geometry instead of the table.

**Not yet verified** — need to re-run `check_pc_sensitivity.py` with the fix and confirm centroids now track hold position.

### 2. Auto-park via IPC (no second FrankaArm)
During `--point-cloud` data collection, the arm auto-parks for clean PC capture using file-based IPC between collect_data.py (Terminal 2) and GotoPoseLive (Terminal 1). This avoids creating a second FrankaArm or calling stop_skill() externally, which would permanently kill VR teleop. See CLAUDE.md "Auto-park IPC protocol" for details.

---

## Open Questions for Second Opinion

1. **Is 50/1024 hold points enough for PointNet to learn position-sensitive grasping?** Or is 5% SNR guaranteed to fail?

2. **Did the model learn to use the PC at all, or did it ignore it?** Best test: run `evaluate.py --dry-run` with `scene_pc = np.zeros((1024,3))` vs real PC and see if actions differ. If actions are identical, the model learned to ignore the PC.

3. **Is the training data recoverable?** The 50-episode zarr (`climbing_holds_upload.zarr`) only stores processed 1024-pt PCs — no raw depth. Can't re-extract with new z_min. Options:
   - Recollect 50 jug episodes with fixed z_min → retrain (clean solution, ~1-2 hours of work)
   - Accept current model as a z_min=-0.02 model and use the same bounds at inference (consistent but bad SNR)

4. **Does the shape of the 50 hold points encode enough for grasp conditioning?** Even with perfect position info, 50 pts might not capture hold shape/orientation well enough to condition the grasp type.

---

## Recommended Next Steps

**Before recollecting data**, run one test to answer question 2:

```python
# In evaluate.py, temporarily replace scene_pc capture with zeros:
scene_pc = np.zeros((1024, 3), dtype=np.float32)
```

Run 3 trials with real PC, 3 with zero PC. If the robot does the same thing both ways, the model is ignoring the PC and recollecting data is the right call. If actions differ, the 50-pt signal is being used.

**If model ignores PC:**
1. Fix z_min (already done)
2. Recollect 50 jug episodes
3. Retrain

**If model uses PC despite 5% SNR:**
1. Fix z_min anyway (raises SNR from 5% → ~100%)
2. Retrain with better data
3. Current model may be partially functional but noisy

---

## Files

| File | Description |
|------|-------------|
| `data_collection/pos_a.npy` / `.png` | Hold in center-right, old z_min |
| `data_collection/pos_b.npy` / `.png` | Hold same area, different orientation |
| `data_collection/pos_c.npy` / `.png` | Hold moved to upper-left, clearly different |
| `data_collection/point_cloud_utils.py` | z_min fix applied (0.008) |
| `data_collection/check_pc_sensitivity.py` | Diagnostic script |
| `checkpoints/pc_large/best.pt` | Trained model (trained with old z_min=-0.02) |
