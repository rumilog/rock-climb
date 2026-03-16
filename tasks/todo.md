# TODO

## Session 2 Complete — Point Cloud Pipeline Implemented

All items from IMPLEMENTATION_LOG.md have been implemented and verified.

### Next Steps (Phase 3 onwards per RESEARCH_PLAN.md)

1. **Data Collection** — Run collect_data.py with --point-cloud on the robot
   - Hold 0 (edge_A, crimp): 50-80 good episodes
   - Hold 1 (edge_B, crimp): 50-80 good episodes
   - Hold 2 (sloper): 50-80 good episodes
   - Hold 3 (pinch): 50-80 good episodes
   ```bash
   python3 collect_data.py --hold 0 --point-cloud --grasp-type crimp
   ```

2. **Pilot Train** — Train on first hold to validate pipeline
   ```bash
   python3 train.py --point-cloud --epochs 3000 --batch 128 --augment --good-only
   ```

3. **Pilot Eval** — Test on robot to confirm PC inference works
   ```bash
   python3 evaluate.py --checkpoint ../checkpoints/best.pt --hold 0 --grasp-type crimp
   ```

4. **Full Data Collection** — Scale to all holds once pilot validated

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
   - Point cloud vs RGB (ResNet baseline)
   - 1024 vs 512 vs 2048 points
   - Effect of number of demonstrations
