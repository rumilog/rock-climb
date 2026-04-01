# TODO

## Current State (2026-03-26)

- ✅ Point cloud pipeline implemented end-to-end
- ✅ Workspace bounds calibrated: `z_min=0.006` verified with `check_pc_sensitivity.py`
- ✅ Trained model uploaded to HuggingFace: `rlogh/climbing-holds-diffusion-policy`
  - Downloaded to `checkpoints/hf_model/best.pt` + `norm_stats.json`, ready to eval
  - Config: encoder_type=point_cloud, state/action_dim=23, obs_horizon=2, pred_horizon=16
- ✅ Arrow key thumb joint tuning added to data collection (up/down=nudge, left/right=cycle joints)
  - MCP_Flex starts at +0.9 rad offset to clear hardware clip from -1.3 post-scale offset
- ✅ **Thumb IK test file created**: `TeleoperationUnity/LEAP/leaphandv1/for_transfer/leap_thumb_ik_test.py`
  - Drop-in replacement for `leap_pip_dip_teleop.py`; only thumb handling differs
  - Index/Middle/Pinky: identical joint-angle pipeline to the original
  - Thumb: ThumbFK (OVR bone quaternions → 3D positions) + BeaVR-style coord transform + PyBullet IK
  - Uses LEAP URDF from `beavr-bot-reference/assets/urdf/leap_hand/leap_hand_right.urdf`
  - `pybullet` installed in `~/franka` venv (3.2.7)
  - Reads thumb bone quaternions from values 16-27 in the 28-value UDP packet (previously discarded)
  - Key constants marked `# TUNE` at top of file; prints FK/IK diagnostics every 100 frames
- ✅ `beavr-bot-reference/` cloned and gitignored — does not affect git push
- ⏳ **Thumb IK test NOT yet run on hardware** — needs live Quest + LEAP session to tune constants

---

## Immediate Next Steps

1. **Test Thumb IK** — run `leap_thumb_ik_test.py` in a live session to tune FK constants
   ```bash
   cd ~/Desktop/tele/TeleoperationUnity/LEAP/leaphandv1/for_transfer
   python3 leap_thumb_ik_test.py
   ```
   - Watch the `[THUMB IK DBG]` printouts (every 100 frames) to see FK positions and IK results
   - If thumb motion looks geometrically wrong, adjust in order:
     1. `THUMB_BONE_AXIS` — try `[1,0,0]` or `[0,0,1]` if `[0,1,0]` is wrong
     2. `THUMB_BASE_POS` — shift until rest-pose FK puts the thumb tip ~12 cm from wrist
     3. `THUMB_BONE_LENS` — scale up/down uniformly if the reach is too short/long
     4. `THUMB_XY_ROT` / `THUMB_Y_ROT` — adjust if thumb IK targets land in the wrong sector
   - Goal: thumb motion should be smoother and more anatomically correct than `leap_pip_dip_teleop.py`
   - Once tuned, promote constants into `leap_pip_dip_teleop.py` (or replace it outright)

3. **Recollect 50 jug episodes on hold 0 (edge_A)**
   ```bash
   python3 collect_data.py --hold 0 --point-cloud --grasp-type jug
   ```
   - Pre-flight check: run `check_pc_sensitivity.py` once to confirm PC looks correct before wasting recording time
   - Verify Z centroid > 0.01 m and centroids shift when hold is moved

4. **Upload new dataset to HuggingFace** and retrain on cluster
   ```bash
   python3 train.py --point-cloud --epochs 5000 --batch 128 --augment --good-only \
       --zarr ../datasets/climbing_holds.zarr --ckpt-dir ../checkpoints/pc_v2
   ```

5. **Pilot Eval** — Test new checkpoint on robot
   ```bash
   python3 evaluate.py --checkpoint ../checkpoints/pc_v2/best.pt --hold 0 --grasp-type jug
   ```
   - Run 2 trials: one with `--zero-pc`, one normal. Actions MUST differ — if identical, PC is still broken.

6. **Data Collection — Holds 1–3** — 50 good episodes each (only after hold 0 eval passes)
   - Hold 1 (edge_B, crimp): `python3 collect_data.py --hold 1 --point-cloud --grasp-type crimp`
   - Hold 2 (sloper): `python3 collect_data.py --hold 2 --point-cloud --grasp-type sloper`
   - Hold 3 (pinch): `python3 collect_data.py --hold 3 --point-cloud --grasp-type pinch`

7. **Full Training** — Retrain on all holds once data is collected

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

**⚠️ The with/without grasp type conditioning ablation is load-bearing for the CoRL paper.**
If conditioning doesn't improve success rate, the main technical claim collapses.
`--no-grasp-conditioning` flag does NOT exist yet in train.py — must be added before running this ablation.
The ablation is simply: same model, same dataset, same hyperparams — just drop the 64-d grasp type embedding branch.

   - [ ] Add `--no-grasp-conditioning` flag to train.py (drops GraspTypeEncoder branch, concat dim 384 instead of 448)
   - [ ] Train unconditioned model on full mixed-grasp-type dataset
   - [ ] Evaluate both models per grasp type on robot → build comparison table
   - [ ] Point cloud vs RGB (ResNet baseline — legacy zarrs preserved for this)
   - [ ] 1024 vs 512 vs 2048 points
   - [ ] Effect of number of demonstrations

## CoRL Paper — Key Action Items (2026-03-20)

- [ ] Verify current cluster training runs cover the ablation sweep (what configs are running?)
- [ ] Add `--no-grasp-conditioning` to train.py
- [x] Add Dexonomy, OmniDexVLG, DexGraspVLA, Grasp as You Say, UniDexFPM, GenDP, CrossDex to related work (2026-04-01)
- [x] Prepare 2-sentence differentiator from Dexonomy (pose gen vs execution policy) — now in §2.3 (2026-04-01)
- [x] Add novelty gap table (Table 1) — 5-method comparison across 4 axes (2026-04-01)
- [x] Fix reference venues: DexCap → RSS 2024, iDP3 → CoRL 2024, DexDiffuser → RA-L (2026-04-01)
- [x] Add two-part deployment architecture paragraph to Discussion §6 (2026-04-01)
- [ ] ⚠️ Fill in FIXME author fields in references.bib for 7 new entries (see IMPLEMENTATION_LOG Session 4)
- [ ] Run Part 1 VLM zero-shot benchmark (GPT-4V or Claude on 4-class hold classification)
