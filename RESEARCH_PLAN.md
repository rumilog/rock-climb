# Grasp-Taxonomy-Aware 3D Diffusion Policy for Dexterous Climbing Hold Manipulation

## 1. Research Problem

Current diffusion policies for dexterous manipulation treat grasping as a binary
success/failure task — the policy learns a single distribution of "good grasps"
regardless of grasp type. However, real-world dexterous grasping requires
fundamentally different hand configurations depending on object geometry: a crimp
grip for thin edges, an open-hand press for slopers, a pinch grip for narrow
features. No existing work explicitly conditions the diffusion policy on grasp
taxonomy or evaluates grasp-type-specific performance.

We propose a **grasp-taxonomy-aware 3D diffusion policy** that:
1. Uses **point cloud observations** (following DP3) instead of RGB images
2. **Conditions on grasp type** — the policy generates type-specific trajectories
3. Is evaluated on a new **climbing hold benchmark** with natural grasp diversity

**Central mechanistic claim (updated 2026-04-20, see §4 Phase 5):**
An ablation without the taxonomy label does **not** exhibit classical mode
collapse — its representation still separates grasp types and its mean
predicted actions still cluster by grasp type at ~90% of the demonstrator's
between-type spread. Instead, the unconditioned model samples with 15–30%
higher within-input variance on precision-critical grasp types. Taxonomy
conditioning therefore acts as a **precision regularizer** on the diffusion
decoder (tightening the sampling distribution around the correct subtype,
analogous to classifier-free guidance in image diffusion or low-temperature
sampling in LLMs), not as a mode selector. This mechanistic distinction is a
primary contribution of the paper.

### Why Climbing Holds?

Climbing holds are uniquely suited as a dexterous manipulation benchmark because:
- Each hold type (crimp, sloper, pinch, jug) demands a different grasp strategy
- Hold geometry varies within each type, requiring generalization
- Grasp quality matters — a weak crimp fails under load even if contact is made
- Commercially standardized objects enable reproducibility
- **Zero prior work** exists on robotic dexterous grasping of climbing holds

## 2. Related Work (Papers to Review)

### 3D Diffusion Policies (Core Method)
- **DP3**: Ze et al., "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations," RSS 2024. [arXiv:2403.03954](https://arxiv.org/abs/2403.03954)
  - Point cloud + PointNet encoder + diffusion policy
  - 40 demos, 85% success on Allegro hand tasks
  - Key reference: our architecture follows DP3
- **iDP3**: Ze et al., "Improved 3D Diffusion Policy," 2024. [arXiv:2410.10803](https://arxiv.org/abs/2410.10803)
  - Egocentric 3D, no camera calibration needed
  - Works with just 10 demos
  - Uses 4096 points, pred_horizon=16
- **Original Diffusion Policy**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
  - Foundational work: DDPM for action prediction
  - obs_horizon=2, pred_horizon=16, action_horizon=8
  - 10 Hz control, cosine beta schedule

### Dexterous Hand Manipulation
- **DexCap**: Wang et al., "DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation," RSS 2024. [arXiv:2403.07788](https://arxiv.org/abs/2403.07788)
  - Uses LEAP hand + Franka (same as our setup!)
  - 10,000 point clouds per frame
  - 55-201 demos per task, collected at 60Hz downsampled to 20Hz
- **DexDiffuser**: Weng et al., "DexDiffuser: Generating Dexterous Grasps with Diffusion Models," RA-L 2024. [arXiv:2402.02989](https://arxiv.org/abs/2402.02989)
  - Diffusion for grasp pose generation on Allegro hand
  - 9-19% improvement over baselines
- **UniDexFPM**: Wu et al., "UniDexFPM: Universal Dexterous Functional Pre-Grasp Manipulation via Diffusion Policy," 2024. [arXiv:2403.12421](https://arxiv.org/abs/2403.12421)
  - Relative joint position changes, obs_horizon=2, pred_horizon=4
  - 1026 training objects, teacher-student distillation

### Grasp Taxonomy and Type-Aware Grasping
- **Cutkosky/GRASP Taxonomy**: Feix et al., "The GRASP Taxonomy of Human Grasp Types," IEEE THMS 2016.
  - 33 canonical human grasp types
  - We map climbing hold grasps to this taxonomy
- **Dexonomy**: RSS 2025. [arXiv:2504.18829](https://arxiv.org/abs/2504.18829) ⚠️ CLOSEST COMPETITOR — MUST CITE AND DIFFERENTIATE
  - Conditions on GRASP taxonomy (31 types) from single-view point clouds; Shadow Hand; 82.3% real-world success
  - **Critical difference**: Dexonomy generates *static grasp poses* (three snapshots: pre-grasp, grasp, squeeze) handed to a motion planner. No imitation learning. No trajectory execution.
  - Our system learns *continuous visuomotor execution trajectories* (arm + hand from approach through contact) via diffusion policy trained on human demonstrations. Fundamentally different problem.
  - Reviewers WILL notice this. Have a crisp 2-sentence differentiation ready.
- **Grasp as You Say**: Tian et al., NeurIPS 2024. [arXiv:2405.19291](https://arxiv.org/abs/2405.19291)
  - Language-conditioned dexterous grasp generation
  - Static poses only — no execution policy
- **OmniDexVLG**: Dec 2024. [arXiv:2512.03874](https://arxiv.org/abs/2512.03874)
  - VLM multi-agent reasoning to infer grasp semantics including taxonomy type
  - Downstream is a grasp pose generator, NOT an execution policy
- **DexGraspVLA**: AAAI 2026. [arXiv:2502.20900](https://arxiv.org/abs/2502.20900)
  - VLM high-level planner + diffusion low-level execution policy
  - VLM plans *what* to grasp (object identity/location), NOT *which grasp type from taxonomy*
  - Closest to our architecture but no taxonomy conditioning
- **CrossDex**: ICLR 2025.
  - Universal grasping across hand morphologies
  - Eigengrasp action space — not the same as functional taxonomy categories

### Multi-Modal and Point Cloud Methods
- **FPV-Net**: Feb 2025. [arXiv:2502.12320](https://arxiv.org/abs/2502.12320)
  - RGB + point cloud fusion with AdaLN
  - Only tested on parallel grippers, not dexterous hands
- **GenDP**: CoRL 2024. [arXiv:2410.17488](https://arxiv.org/abs/2410.17488)
  - 3D semantic fields for category-level generalization
  - 20% → 93% success on unseen instances
- **Point Cloud Matters**: NeurIPS 2024.
  - Systematic study of point cloud representations for manipulation

### Reactive/Tactile Policies (Future Extension)
- **Reactive Diffusion Policy**: 2025. [arXiv:2503.02881](https://arxiv.org/abs/2503.02881)
  - Slow-fast visual-tactile corrections within action chunks
  - Parallel grippers only — gap for dexterous hands

## 3. Experimental Setup

### Hardware
- **Robot arm**: Franka Emika Panda (7-DoF)
- **Dexterous hand**: LEAP Hand v1 (16-DoF, 4 fingers)
- **Cameras**: 4x Intel RealSense D415/D455 (cameras 2-5)
  - Capture: 848×480 RGB + depth at 30fps
  - Intrinsics: read live from the active `pyrealsense2` pipeline at 848×480
  - Extrinsics: fixed camera-to-world transforms from an offline calibration
- **Point cloud generation**: depth → XYZ via intrinsics, fused across 4 cameras
  into world frame using the extrinsics, then cropped to a calibrated workspace
  bounding box and downsampled with FPS
- **Spring displacement testbed** (built and used in primary evaluation, 2026-05-15):
  a linearly constrained spring platform that seats each climbing hold and restricts
  displacement to a single axis — 180° (toward the robot base, -X direction). The
  testbed uses 2 identical compression springs in parallel, empirically calibrated as
  F = 0.59 + 0.8 × x per spring (lbf, x in inches) → F_total = 1.18 + 1.6 × x_in lbf.
  Holds are placed at 5 discrete orientations relative to the pull axis: 0°, ±22.5°,
  ±45°. Orientations beyond ±45° are not tested because frictional slip is certain
  at those angles regardless of grasp quality.
- **Linear ratcheting mechanism**: integrated into the spring testbed; captures
  the hold's maximum displacement even when it slips mid-trial. 11 teeth at
  9.3 mm/tooth (max travel 102.3 mm). The operator enters the tooth count after
  each pull; force is computed analytically. This converts evaluation from a binary
  pass/fail into a continuous slip-force metric in Newtons (range 5.2 N preload to
  33.9 N at the 11-tooth ceiling), enabling Wilcoxon-signed-rank comparisons and
  Cohen's d effect-size estimation rather than McNemar's test on a binary outcome.

### Climbing Holds

We use **4 hold categories** for training, with a held-out category for evaluation:

| Hold ID | Type       | Grasp Strategy              | Training/Test |
|---------|------------|-----------------------------|---------------|
| 0       | edge_A     | Crimp (curled fingers)      | Train         |
| 1       | edge_B     | Crimp (variant)             | Train         |
| 2       | sloper     | Open-hand press (friction)  | Train         |
| 3       | pinch      | Pinch grip (thumb+fingers)  | Train         |
| 4       | test_edge  | Crimp (held out)            | **Test**      |

For each category, use **2-3 physical hold variants** (different manufacturers/shapes
that require the same grasp strategy). This tests within-category generalization.

**Total target**: ~3-5 physical holds per category × 4 categories = 12-20 holds

### Data Collection Protocol

**Per hold:**
- Place hold on the spring testbed at each of the 5 orientations in turn
- Collect **10 episodes per orientation × 5 orientations = 50 episodes per hold**
- Each episode: approach → grasp → (optionally) pull to confirm quality
- Mark quality as good/bad after each episode
- Record grasp type label (crimp/sloper/pinch/jug) and hold orientation (degrees) per episode

**Episode structure:**
1. Robot resets to home pose
2. Robot arm moves out of camera view (automated)
3. System captures a **clean point cloud snapshot** of the hold (no robot in scene)
4. Robot returns to approach pose
5. Teleoperation begins — operator demonstrates the grasp
6. Operator rates the grasp quality (good/bad)
7. Data saved to zarr

**Current dataset (2026-05-13, rig-collected):**
- 200 episodes collected on the spring testbed at 5 orientations × 10 reps per orientation × 4 hold types
- All 200 marked "good" quality
  - Hold 0 (edge_A): 50 jug | Hold 1 (edge_B): 50 crimp
  - Hold 2 (sloper): 50 sloper | Hold 3 (pinch): 50 pinch
- 27,821 total timesteps, stored at `/mnt/ssd/rumi_tele_datasets/climbing_holds_rig.zarr`
- HuggingFace mirror: `rlogh/climbing-holds-pointcloud` (rig version)
- Held-out test hold (hold 4, test_edge): not yet collected

**Note (2026-05-15):** an older flat-table 200-episode dataset (`climbing_holds.zarr`,
29,647 timesteps, no orientation variation, z_min=0.006) is retained on disk for
RGB-ablation reference only; it is NOT used for any current training.

### Observation Space

| Component | Dimensions | Source |
|-----------|-----------|--------|
| Point cloud (scene) | 1024 × 3 (XYZ) | 4 cameras fused, workspace-cropped, FPS downsampled |
| Robot state | 23 (7 arm joints + 16 hand joints) | Proprioception |
| Grasp type (conditioning) | 4 (one-hot) | Human label |
| **Total** | 1024×3 + 23 + 4 | |

**Note**: Following DP3, we use **XYZ only** (no color) in point clouds for
better appearance generalization. The point cloud is cropped to a **calibrated
workspace bounding box** (x=[0.30,0.85], y=[-0.35,0.35], z=[0.027,0.40] in the
spring-testbed configuration) to remove the rig deck, table/background, and
ceiling/wall points while preserving the full hold geometry. (Earlier flat-table
era used z=[0.006,0.30]; the rig elevated the hold ~2 cm, prompting the bounds
update.)

#### Static-scene assumption and single point cloud per episode

The climbing hold and table are **static** throughout each episode; only the
robot configuration changes. We therefore capture **one clean fused point cloud
per episode** at the start (after moving the arm out of view), and reuse that
fixed scene point cloud for all timesteps in the episode. During data
collection we log only robot state, actions, and grasp type over time.

This design:
- Matches DP3-style setups where the 3D scene is effectively static during an
  episode and the policy reasons about **relative motion** of the robot with
  respect to a fixed 3D geometry.
- Keeps on-robot computation tractable: multi-camera fusion + FPS takes ≈1s,
  so doing it once per episode (rather than at 10 Hz) preserves real-time
  control during teleoperation and evaluation.
- Is appropriate for this benchmark, where the core challenge is learning
  **grasp-type-specific hand configurations** relative to a fixed climbing hold,
  not tracking moving objects.

### Action Space

| Component | Dimensions | Format |
|-----------|-----------|--------|
| Arm joint targets | 7 | Absolute positions (rad) |
| Hand joint targets | 16 | Absolute positions (rad) |
| **Total** | 23 | |

**Note**: We drop ee_pos (3) and ee_quat (4) from the action space since they
are redundant with arm joint positions via forward kinematics. This reduces
action dim from 30 to 23, which is cleaner.

### Policy Architecture (DP3-based)

```
Point Cloud (1024×3) → PointNet encoder → 256-d
Robot State (23) → MLP → 128-d
Grasp Type (4) → Embedding → 64-d

Fuse: concat → MLP → 512-d conditioning vector

Conditioning + Diffusion timestep → 1D Temporal U-Net → Action chunk (16 steps × 23-dim)
```

### Training Configuration

| Parameter | Value | Source / Justification |
|-----------|-------|----------------------|
| obs_horizon | 2 | Chi et al. 2023, DP3 — provides velocity info without blowing up input |
| pred_horizon | 16 | Chi et al. 2023 — standard for CNN-based DP |
| action_horizon | 8 | Chi et al. 2023 — execute 8 of 16 predicted, then re-plan for smoothness |
| diffusion_steps | 100 (train), 10 DDIM (inference) | DP3, Chi et al. — cosine beta schedule |
| batch_size | 128 | DP3 — fits in 0.8 GB VRAM on RTX 2080 Ti |
| learning_rate | 1e-4 | DP3, Chi et al. — standard for AdamW with diffusion |
| AdamW betas | (0.95, 0.999) | DP3 — β1=0.95 standard for diffusion (noisier loss than typical) |
| weight_decay | 1e-6 | Minimal regularization, standard |
| grad_clip | 1.0 | Prevents exploding gradients from large noise samples |
| epochs | 3000 | DP3 — sufficient for convergence on 200 episodes |
| LR schedule | 500-step linear warmup + cosine decay | DP3, standard |
| EMA | power-law warmup (power=0.75), max=0.9999 | Reference implementations |
| normalization | min-max to [-1, 1] | DP3 recommendation for PC mode |
| point cloud points | 1024 | DP3 default |
| downsampling | Farthest Point Sampling (FPS) | DP3 |
| U-Net down_dims | (256, 512, 1024) | Chi et al. 2023 — appropriate for 200-episode dataset |
| PointNet | 3→64→128→256, no T-Net | Simplified — T-Net unnecessary for fixed-frame PCs |
| PC augmentation | jitter σ=0.002 + 5% dropout | DP3 jitter value; dropout simulates partial occlusion |

### Evaluation Metrics

1. **Grasp success rate** — binary: did the hand achieve stable contact?
2. **Grasp type accuracy** — did the robot use the correct grasp strategy?
3. **Hold stability (spring displacement test)** — after policy convergence, the arm
   pulls the hold 10.5 cm along the testbed's fixed linear axis (hardcoded 180°,
   toward the robot base, -X direction) using impedance control (kx=4000 N/m pull
   axis, ky=kz=100 N/m lateral/vertical compliance). Five hold orientations are
   tested per grasp type: 0°, ±22.5°, ±45°. The **linear ratchet** (9.3 mm/tooth,
   11 teeth) captures peak hold displacement; the operator enters the tooth count
   into the script after each pull. Slip force is computed from two springs in
   parallel: `F = (1.18 + 1.6 × x_in) × 4.448 N` where x_in = displacement/25.4.
   Range: 5.2 N (0 teeth, preload) to 33.9 N (11 teeth). The per-trial `ratchet`
   dict is logged in the eval JSON; binary success threshold can be set post-hoc.
   This is the primary success criterion — adapted from the standard lift test for
   holds that cannot be picked up.
4. **Cross-category generalization** — success on held-out test_edge
5. **Ablations**:
   - With vs without grasp type conditioning ← **LOAD-BEARING** — preliminary results confirm conditioning wins. `--no-grasp-conditioning` flag implemented 2026-04-14.
   - **Wrong-label ablation** ← planned — give model deliberately incorrect taxonomy label; hypothesis: model executes wrong grasp type, proving conditioning is causally responsible
   - **Held-out hold generalization** ← planned — evaluate on new physical crimp hold (test_edge) never seen in training; tests within-type generalization
   - Point cloud vs RGB (ResNet baseline)
   - 1024 vs 512 vs 2048 points
   - Effect of number of demonstrations

### Evaluation Protocol

**Trial counts:** 20+ trials per model per grasp type = 160+ total trials minimum.

**Per-trial metrics:**
1. Grasp success (binary 0/1 — disturbance rejection test: arm displaced ~5 cm, did hold move with it?)
2. Grasp type correctness (binary 0/1 — human judgment of hand configuration)
3. Contact time (seconds from start to first contact)

**Spring displacement test protocol:** The spring testbed constrains hold motion to
a single axis (180°, toward the robot base, -X direction). Pull angle is hardcoded
in both eval scripts — no `--pull-angle` flag needed. Max pull distance is 10.5 cm
(`--pull-dist 0.105`). After each trial the operator enters the ratchet tooth count
(0–11); the script logs displacement and slip force automatically.
The varying dimension is the **hold orientation**: 0°, ±22.5°, ±45° relative to
the pull axis. For each grasp type, run at least 4 trials per orientation (20 pairs
total across the 5 orientations). Log orientation per trial in the lab notebook;
it is not stored in the zarr.

**Statistical tests:** Fisher's exact test for pairwise success rate comparisons.
Report 95% Wilson confidence intervals on all success rates.
Primary figure of merit: **delta success rate** (taxonomy model minus no-taxonomy model).

### Novelty Gap Table (verified via literature search, 2026-03-20)

| Paper | Point Cloud | Grasp Type Cond | Execution Policy | Dexterous Hand |
|---|---|---|---|---|
| DP3 (RSS 2024) | ✅ | ❌ | ✅ | ✅ (Allegro) |
| Dexonomy (RSS 2025) | ✅ | ✅ (31 types) | ❌ pose gen only | ✅ (Shadow) |
| Grasp as You Say (NeurIPS 2024) | ✅ | ❌ free-form lang | ❌ pose gen only | ✅ |
| OmniDexVLG (Dec 2024) | partial | ✅ VLM-guided | ❌ pose gen only | ✅ |
| DexGraspVLA (AAAI 2026) | ❌ | ❌ object ID only | ✅ | ✅ |
| **Ours** | **✅** | **✅ 4-class discrete** | **✅ arm+hand traj** | **✅ LEAP** |

The combination of all four simultaneously is the novel contribution. Confidence: ~82%.

## 4. Implementation Status (Updated 2026-05-15)

### Phase 1: Point Cloud Data Collection Pipeline — COMPLETE
- ✅ Multi-camera PC fusion (4 cameras → world frame → workspace crop → FPS → 1024 pts)
- ✅ 23-dim state/action (dropped ee_pos/ee_quat)
- ✅ `--grasp-type` and `--point-cloud` flags in collect_data.py
- ✅ Auto-park IPC protocol for clean PC capture (arm out of camera view)
- ✅ `point_cloud` + `grasp_type_id` stored in zarr

### Phase 2: Point Cloud Policy Architecture — COMPLETE
- ✅ PointNet encoder (3→64→128→256, max pool, proj→256-d)
- ✅ GraspTypeEncoder (one-hot→MLP→64-d)
- ✅ PointCloudObservationEncoder (PointNet + State + GraspType → 512-d)
- ✅ Min-max normalization to [-1, 1]
- ✅ `--no-grasp-conditioning` ablation flag (drops GraspTypeEncoder branch)

### Phase 3: Spring Testbed Data Collection — COMPLETE (2026-05-13)
- ✅ Spring testbed built (2 parallel springs, linear-axis constraint, 11-tooth ratchet)
- ✅ Workspace bounds recalibrated for testbed (z_min=0.027, z_max=0.40)
- ✅ Rig-era dataset collected: 200 episodes (50 per grasp type × 5 orientations × 10 reps)
- ✅ Stored at `/mnt/ssd/rumi_tele_datasets/climbing_holds_rig.zarr` + HuggingFace mirror
- ⏳ Held-out test hold (hold 4, test_edge) not yet collected

### Phase 4: Training and Primary Evaluation — COMPLETE (2026-05-15)
- ✅ Train Model A (with taxonomy) on rig dataset — `pc_with_taxonomy_rig`, best loss 0.001670 (epoch 2939)
- ✅ Train Model B (without taxonomy, ablation) on rig dataset — `pc_no_taxonomy_rig`, best loss 0.001777 (epoch 2859)
- ✅ Both checkpoints on HuggingFace: `rlogh/climbing-holds-rig-with-taxonomy`, `rlogh/climbing-holds-rig-no-taxonomy`
- ✅ **Primary paired evaluation complete**: 80 paired trials (4 pairs × 5 orientations × 4 hold types)
  - Session: `eval_results/paired_session_20260515_074256.json`
  - Primary metric: ratchet slip force (continuous, N)
  - All grasp types significant at $p < 0.05$ (Wilcoxon signed-rank, paired)
  - Overall: median +5.2 N, $p < 0.001$, Cohen's $d = 0.95$
  - Per-type effect size: pinch d=1.82 (20/20 wins), sloper d=1.08, crimp d=1.05, jug d=0.55
  - Failure-mode rescue: complete-failure rate 44% → 10% (4.4× reduction)
  - No order bias (Mann–Whitney p > 0.69 both models); no pegged-ratchet (right-censored) trials
- ⏳ Wrong-label ablation — pending (causal pin on conditioning signal)
- ⏳ Held-out hold generalization (test_edge, new physical crimp hold, eval only)
- ⏳ Regenerate latent / action-distribution / novelty figures from rig checkpoints
- ⏳ RGB baseline comparison (legacy zarrs preserved)

### Phase 5: Mechanistic Analysis — COMPLETE (2026-04-20)
- ✅ Latent-space visualisations of PointNet + fused encoders (`generate_latent_viz.py`).
  Silhouette score and stratified-5-fold kNN accuracy computed for each latent.
- ✅ Action-distribution comparison under matched inputs (`generate_action_dist_viz.py`):
  K=8 rollouts per model per input, 32 inputs.
- ✅ Novelty / mode-commitment figures (`generate_novelty_viz.py`): per-type
  hand-pose fingerprints and single-joint ribbons over the prediction horizon.
- ✅ Eval-overlay figure: paired-eval success/failure mapped onto the action-PCA space.

**Key empirical finding (load-bearing for the paper's reframe):**
The Without-taxonomy model separates grasp types in its *representation*
(fused-encoder silhouette ≈ 0.32, kNN ≈ 99%) and produces *mean action
predictions* that differ by grasp type at 90% of the demonstrator between-type
spread (0.316 rad vs 0.352 rad demo, With-taxonomy 0.314 rad). It therefore
does **not** undergo classical mode collapse.

What it *does* is sample with 15–30% higher **within-input variance** on the
non-jug grasp types. On real dexterous hardware this extra per-rollout wobble
misses the hold. The taxonomy label is best characterised as a **precision
regularizer** that tightens the sampling distribution around the correct
subtype (analogous to CFG in image diffusion / low-temperature sampling in
LLMs), not as a mode selector.

The mechanistic prediction is confirmed by the Phase-4 hardware evaluation:
Cohen's d ordering across grasp types (pinch 1.82 > sloper 1.08 ≈ crimp 1.05
≫ jug 0.55) recovers the precision-tolerance ordering exactly, with the
geometrically slack jug producing the smallest (though still statistically
significant) effect.

This is the central mechanistic claim of the paper. See
`IMPLEMENTATION_LOG.md` Session 6 (latent / action / novelty analysis) and
Session 7 (primary hardware evaluation) for full numbers and figures.

## 5. Grasp Type Label — Manual Input

The grasp type label `g` is provided manually by the operator at both data collection
time and deployment time. No vision classifier is implemented or planned. This keeps
the paper focused on the mechanistic contribution (precision regularizer) rather than
the deployment pipeline.

**Key design decision:** Grasp type is input (not predicted from the point cloud),
keeping the policy's job focused on *how* to grasp rather than *what* grasp to use.

---

## 6. Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Point cloud capture too slow for 10Hz control | Pre-compute a single clean scene PC before each episode; during the episode, only read depth and run the trained policy (no fusion needed) |
| Too few demos per hold | Start with 2 holds, validate pipeline works before scaling |
| Grasp type conditioning doesn't help | Still publishable as a benchmark paper with ablation showing this negative result |
| Camera calibration drift | Re-calibrate periodically; iDP3 shows egocentric 3D can avoid this |
| FPS downsampling too slow | Randomly pre-sample to ~20k points before FPS; this keeps runtime ≈1s while preserving coverage at 1024 output points |

## 7. Timeline

| Week | Task | Status |
|------|------|--------|
| 1 | Update data collection pipeline for point clouds | ✅ Done |
| 2 | Collect pilot data (1 hold, 50 demos), validate pipeline | ✅ Done |
| 3 | Implement PointNet encoder + grasp type conditioning in train.py | ✅ Done |
| 4 | Train pilot model, debug, iterate | ✅ Done |
| 5–7 | Initial 200-episode flat-table dataset | ✅ Done (2026-04-14, later superseded) |
| 8 | Mechanistic / latent / action-distribution analysis | ✅ Done (2026-04-20) |
| 9 | Build spring testbed, recollect rig dataset | ✅ Done (2026-05-13) |
| 10 | Train both rig models, upload checkpoints to HuggingFace | ✅ Done (2026-05-15) |
| 11 | Primary paired evaluation (80 trials, ratchet slip-force) | ✅ Done (2026-05-15) |
| 12 | Analysis figures + paper-ready methodology / observations docs | ✅ Done (2026-05-15) |
| 13+ | Wrong-label ablation, held-out hold, final paper writing | ⏳ Future |
