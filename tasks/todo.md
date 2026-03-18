# TODO

## Current State (2026-03-18 — updated after PC pipeline fix)

- ✅ Point cloud pipeline implemented end-to-end
- ✅ Workspace bounds calibrated: `z_min=0.006` verified with `check_pc_sensitivity.py`
  - Hold geometry fully captured, zero table noise, Z centroid ~0.034 m
- ✅ `check_pc_sensitivity.py` diagnostic tool available for future verification
- ❌ Previous 50 jug episodes (`climbing_holds.zarr`) **deleted** — bad PC data (95% table noise, z_min was -0.02)
- ❌ Pilot checkpoint (`checkpoints/pc_large/best.pt`) is invalid — trained on bad data, model ignored PC
- ⏳ **Ready to recollect clean data** with fixed pipeline

---

## Immediate Next Steps

1. **Recollect 50 jug episodes on hold 0 (edge_A)**
   ```bash
   python3 collect_data.py --hold 0 --point-cloud --grasp-type jug
   ```
   - Pre-flight check: run `check_pc_sensitivity.py` once to confirm PC looks correct before wasting recording time
   - Verify Z centroid > 0.01 m and centroids shift when hold is moved

2. **Upload new dataset to HuggingFace** and retrain on cluster
   ```bash
   python3 train.py --point-cloud --epochs 5000 --batch 128 --augment --good-only \
       --zarr ../datasets/climbing_holds.zarr --ckpt-dir ../checkpoints/pc_v2
   ```

3. **Pilot Eval** — Test new checkpoint on robot
   ```bash
   python3 evaluate.py --checkpoint ../checkpoints/pc_v2/best.pt --hold 0 --grasp-type jug
   ```
   - Run 2 trials: one with `--zero-pc`, one normal. Actions MUST differ — if identical, PC is still broken.

4. **Data Collection — Holds 1–3** — 50 good episodes each (only after hold 0 eval passes)
   - Hold 1 (edge_B, crimp): `python3 collect_data.py --hold 1 --point-cloud --grasp-type crimp`
   - Hold 2 (sloper): `python3 collect_data.py --hold 2 --point-cloud --grasp-type sloper`
   - Hold 3 (pinch): `python3 collect_data.py --hold 3 --point-cloud --grasp-type pinch`

5. **Full Training** — Retrain on all holds once data is collected

---

## Part 1: Hold Identifier (Separate Workstream — Future)

**Goal:** VLM-based classifier that predicts grasp type from an RGB image of a hold.
**Status:** Not started. Existing Part 2 data unaffected — this is fully independent.

Steps:
1. **Benchmark zero-shot first** — prompt GPT-4V or Claude with the 4 hold types on a handful of test images. If >90%, ship it.
2. **If fine-tuning needed:** Scrape labeled images from gear sites / Kilter Board / Moonboard (jug/sloper/crimp/pinch are standard terms). Add 20-50 in-lab photos per class from RealSense.
3. **Fine-tune CLIP or lightweight VLM** on the combined dataset.
4. **Integration:** At deployment, pipe RGB snapshot of hold → Part 1 → grasp_type label → Part 2 (diffusion policy conditioning).

Note: In-lab photo collection requires no robot or zarr — just place hold, take picture, label.

---

## Part 2 Ablations
   - With vs without grasp type conditioning
   - Point cloud vs RGB (ResNet baseline — legacy zarrs preserved for this)
   - 1024 vs 512 vs 2048 points
   - Effect of number of demonstrations
