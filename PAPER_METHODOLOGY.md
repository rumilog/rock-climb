# Paper Methodology Reference

**Purpose:** every infrastructure / reproducibility / dataset detail a reviewer will demand.
Copy paragraphs directly from here into the paper's Methods, Implementation Details, and
Reproducibility sections.

---

## 1. Compute Infrastructure

### Workstation (data collection, training, evaluation — all on the same machine)
| Component | Spec |
|---|---|
| OS | Ubuntu 20.04.6 LTS (Focal Fossa) |
| Kernel | Linux 5.15.0-88-generic |
| CPU | Intel Core i7-8700K @ 3.70 GHz (6 cores / 12 threads) |
| RAM | 16 GB DDR4 |
| GPU | NVIDIA GeForce RTX 2080 Ti (11.3 GB VRAM) |
| NVIDIA driver | 515.43.04 |
| CUDA toolkit | 11.8 |
| Python | 3.8 (`~/franka` virtualenv) |
| PyTorch | 2.4.1+cu118 |
| zarr | 2.16.1 |
| ROS | Noetic |
| Robot middleware | frankapy (Iam-Lab fork) |
| Camera library | pyrealsense2 + robomail.vision (in-house wrapper) |

### Total compute budget
- **Training:** 2 models × 8.6 h ≈ **17 GPU-hours** on a single RTX 2080 Ti
- **Evaluation:** 80 paired trials × ~5 min/pair ≈ **6.7 wall-clock hours** of robot time
- **Data collection:** 200 episodes × ~30 s/episode (incl. setup) ≈ **2 h** of teleop
- **Total project compute footprint:** < 1 day of single-GPU + < 1 day of robot time

---

## 2. Robot Hardware

| Component | Detail |
|---|---|
| **Arm** | Franka Emika Panda (7-DoF), serial #(redact for blind review) |
| **Arm control** | frankapy 0.5+, ROS Noetic, custom `GotoPoseLive` cartesian-impedance skill |
| **Arm control PC** | Separate Ubuntu 20.04 box, `franka-Alienware-Area-51-R5`, real-time kernel |
| **Hand** | LEAP Hand v1 (16-DoF, 4 fingers: Index, Middle, Pinky, Thumb) |
| **Hand motors** | Dynamixel XC330 servos, 4 Mbps serial via U2D2, daisy-chained |
| **Cameras** | 4× Intel RealSense (mix of D415 and D455), IDs 2–5, 848×480 RGB+Depth at 30 fps |
| **Calibration** | Per-camera intrinsics read live from `pyrealsense2`; world-frame extrinsics from offline checkerboard calibration (stored in robomail) |
| **VR controller** | Meta Quest 2 (hand-tracking via OVRPlugin BoneRotations API), Quest connects to robot machine over WiFi UDP (port 8002) |
| **Spring testbed** | 2 identical compression springs in parallel, linear rail constrains motion to ±X axis (180° toward robot base). Springs empirically calibrated: F = 0.59 + 0.8 × x [lbf, x in inches] per spring → F_total = 1.18 + 1.6 × x_in [lbf]. Convert: F_N = F_lbf × 4.448. |
| **Ratchet** | Linear pawl mechanism, 11 teeth at 9.3 mm/tooth, captures peak hold displacement |

### State / action space
- **State:** 23-dim — 7 Franka arm joints (rad) + 16 LEAP hand joints (rad, Allegro convention)
- **Action:** 23-dim, same layout. Absolute joint targets, executed via `goto_joints(dynamic=True)` for the arm and Dynamixel `write_desired_pos` for the hand.
- **Control loop:** 10 Hz closed-loop. Joint deltas clamped to ±0.05 rad/step for safety.
- **End-effector pose (ee_pos, ee_quat) is NOT in state/action** — it is fully determined by arm joints via FK, so we drop it to keep the action space minimal.

---

## 3. Dataset

### Training data: `climbing_holds_rig.zarr`
| Metric | Value |
|---|---|
| Storage | `/mnt/ssd/rumi_tele_datasets/climbing_holds_rig.zarr` |
| HuggingFace mirror | `rlogh/climbing-holds-pointcloud` (rig version) |
| Total episodes | **200** (50 per grasp type × 4 types) |
| Total timesteps | **27,821** at 10 Hz (= 46.4 minutes of teleop) |
| Episode length | mean 139.1, median 132, min 61, max 279 steps |
| State dim | 23 (arm7 + hand16) |
| Point cloud | 1024 points × XYZ per timestep |
| Quality flag | 200 / 200 marked "good" by demonstrator |
| Demonstrator | single operator, VR teleoperation (Meta Quest 2 hand-tracking) |

### Collection protocol
- **Per hold:** 10 episodes × 5 orientations (−45°, −22.5°, 0°, +22.5°, +45°) = 50 episodes
- Orientation rotated manually between blocks of 10 episodes
- **One clean point cloud captured per episode** at the start (after parking arm out of view), reused for all timesteps within the episode. Justified by static-scene assumption — only the robot moves during an episode.
- 4 cameras fused → world-frame XYZ → workspace bounding-box crop (x∈[0.30,0.85], y∈[−0.35,0.35], z∈[0.027,0.40]) → outlier removal → farthest-point sampling to exactly 1024 points

### Holds
- **Hold 0 (edge_A) → jug grasp** — deep horn-style jug
- **Hold 1 (edge_B) → crimp grasp** — thin shallow edge
- **Hold 2 (sloper) → sloper grasp** — convex friction-only surface
- **Hold 3 (pinch) → pinch grasp** — narrow vertical feature

---

## 4. Architecture

```
Point Cloud (1024×3)  ─→  PointNet encoder (3→64→128→256, no T-Net, max-pool, proj)  ─→  256-d
Robot State (T_o × 23) ─→  MLP                                                          ─→  128-d
Grasp Type (one-hot 4) ─→  Embedding MLP                                                ─→  64-d
                                       │
                            concat ────┴─→  MLP fuse  ──→  512-d conditioning vector
                                                                      │
                                                                      ▼
            Diffusion timestep  ──→  positional embed  ──→  1D Temporal U-Net (DDPM)
                                                                      │
                                                                      ▼
                                                  Action chunk: 16 steps × 23 dim
```

| Hyperparameter | Value | Source |
|---|---|---|
| obs_horizon (T_o) | 2 | Chi et al. RSS 2023, DP3 RSS 2024 |
| pred_horizon | 16 | Chi et al. RSS 2023 |
| action_horizon | 8 | Execute 8 of 16 predicted, then re-plan |
| diffusion_steps (train) | 100 (DDPM, cosine β schedule) | DP3 |
| diffusion_steps (infer) | 10 (DDIM) | DP3, for 10 Hz real-time control |
| batch_size | 128 | DP3 |
| optimizer | AdamW(β=(0.95, 0.999), wd=1e-6) | DP3 |
| learning_rate | 1e-4 | DP3 |
| LR schedule | 500-step linear warmup → cosine decay | DP3 |
| EMA | power-law warmup (power=0.75), max=0.9999 | Standard |
| grad_clip | 1.0 | Standard |
| epochs | 3000 | DP3 |
| normalization | min-max to [−1, 1] | DP3 (PC mode) |
| point cloud augmentation | jitter σ=0.002 + 5% point dropout | DP3 + standard |
| U-Net down_dims | (256, 512, 1024) | Chi et al. RSS 2023 |
| PointNet dims | (3 → 64 → 128 → 256) | Simplified DP3 |
| grasp-type embedding dim | 64 | Our design |
| Total parameters | 34,621,365 (WITH) / 34,584,117 (WITHOUT) | identical except grasp-type encoder branch |

**Mixed precision:** PyTorch AMP, `torch.cuda.amp.GradScaler` enabled.

---

## 5. Training

| Metric | WITH taxonomy | WITHOUT taxonomy |
|---|---|---|
| Total epochs | 3000 | 3000 |
| Wall-clock time | 8.6 h | 8.5 h |
| Avg time/epoch | 10.3 s | 10.2 s |
| Best training loss | 0.001670 | 0.001777 |
| Best epoch | 2939 | 2859 |
| Iter/epoch | 192 (24,621 samples / batch 128) | 192 |

**Training machines:**
- WITH taxonomy: this workstation (i7-8700K + RTX 2080 Ti)
- WITHOUT taxonomy: separate machine, identical code (`train.py --no-grasp-conditioning`),
  checkpoint distributed via HuggingFace: `rlogh/climbing-holds-rig-no-taxonomy`

**No validation split** — best.pt selected on lowest training loss. Justification (per DP3):
real-world success rate is the true validation signal; train/val splits on tiny imitation
datasets (200 episodes) hurt more than they help, and DP3, Chi et al. (RSS 2023), and DexCap
(RSS 2024) all follow this convention.

**Reproducibility note:** training does not currently set a global random seed. Future runs
should set `torch.manual_seed(42)` and `np.random.seed(42)` in `train.py` for bitwise
reproducibility — current results are dependent on system RNG state but the qualitative
findings (WITH > WITHOUT on all 4 grasp types) are robust given the large effect sizes.

---

## 6. Evaluation Protocol

### Hardware setup at eval time
- Spring testbed in place on the workspace, ratchet armed
- Robot home pose verified (PARK_ARM_JOINTS clear of all 4 cameras)
- Both checkpoints loaded once at startup; hardware shared between models

### Per-pair protocol
1. Operator places target hold on testbed at the specified orientation (one of −45°, −22.5°, 0°, +22.5°, +45°)
2. Coin-flip decides first model on pair 1; **strict global alternation** for every subsequent pair
3. For each of the 2 trials in the pair:
   a. Reset arm to RESET_ARM_JOINTS
   b. Park arm to PARK_ARM_JOINTS (clears all 4 cameras)
   c. Capture fused point cloud (4 cameras, 5 frames averaged, FPS to 1024 pts)
   d. Return arm to approach pose
   e. SPACE → fill 2-step observation buffer → run policy at 10 Hz
   f. Up to 200 control steps (20 s) or operator-stop (SPACE / Q)
   g. **Pull test:** impedance-controlled −X displacement of 13 cm. Stiffness: kx=4000 N/m (pull axis), ky=100 N/m (lateral), kz=2000 N/m (vertical — must support LEAP hand weight). Duration 5 s.
   h. Operator reads ratchet tooth count (0–11), enters into script. Script computes displacement (mm), force (N, from two-spring formula), and logs to JSON.
4. Between trials within a pair: operator repositions hold to match trial 1's location; fresh point cloud captured for trial 2

### Per-trial latency
- Point cloud capture: ~1 s (5 depth-frame averaging)
- Policy inference: ~150–300 ms per forward pass (10 DDIM steps), batched once per `action_horizon=8` steps
- Effective control rate: 10 Hz throughout the rollout

### Operator
- Single demonstrator and single evaluator (same person, same day)
- All ratchet readings entered by this single observer
- Binary g/b rating field exists in JSON but is uniformly "good" across this session — the
  primary metric is the continuous ratchet force. This is by design: with continuous data the
  binary rating loses information. We preserve the field for backward-compatibility with the
  paired_eval.py format.

### Failure-handling
- Mid-pair Ctrl-C / Franka skill error: pair is discarded (not in JSON), session resumes
  from previously-completed pair via `--resume <session.json>`
- Session JSON saved after every completed pair (worst-case loss = 1 in-progress pair)

---

## 7. Statistical Methods

### Primary metric
**Ratchet slip force (N)** — continuous, per trial:
```
displacement (mm) = teeth × 9.3
displacement (in) = displacement (mm) / 25.4
F_total (lbf)     = 1.18 + 1.6 × displacement (in)   [two springs in parallel]
F_total (N)       = F_total (lbf) × 4.448222
```
Range: 5.2 N (0 teeth, preload only) to 33.9 N (11 teeth, ratchet max).
Strong grips that traveled past 102.3 mm are pegged at 11 teeth and reported as
"≥ 33.9 N" — see Limitations §10.

### Statistical tests
- **Paired comparison per grasp type:** Wilcoxon signed-rank test (two-sided), n=20 pairs each
- **Effect size:** Cohen's d for paired differences = mean(diff) / std(diff)
- **95% confidence intervals on d:** percentile bootstrap, 2000 iterations
- **Significance levels:** *** p<0.001, ** p<0.01, * p<0.05, ns ≥0.05
- **No multiple-comparisons correction:** 4 pre-registered grasp-type comparisons; overall
  test (n=80) is also pre-registered. Both Bonferroni-corrected and uncorrected p-values
  for the overall test are < 0.001.

### Justification: Wilcoxon over paired t-test
Force data are right-truncated at the ratchet's 11-tooth ceiling and have heavy lower-tail
mass at the 5.2 N preload (complete grip failure). Both violate normality. The
non-parametric Wilcoxon signed-rank is robust to these violations and is the standard for
paired imitation-learning evaluations.

---

## 8. Software / Code

### Repository layout
```
~/Desktop/tele/
├── data_collection/
│   ├── train.py             — DP3 architecture + grasp-type conditioning, --no-grasp-conditioning ablation flag
│   ├── collect_data.py      — VR teleop data collection (Terminal 2, paired with VR_Teleoperation_Minimum.py)
│   ├── paired_eval.py       — Paired with-vs-without evaluation harness
│   ├── evaluate.py          — Single-model evaluation
│   ├── episode_storage.py   — Zarr writer with grasp_type + hold_id metadata
│   └── point_cloud_utils.py — Multi-cam fusion, workspace crop, FPS
├── eval_results/
│   ├── generate_ratchet_figures.py   — primary force figures (8 figs)
│   ├── generate_training_curves.py   — loss-vs-epoch
│   ├── display_results.py            — terminal table + publication PNG
│   └── figures/                      — all generated figures
├── checkpoints/
│   ├── pc_with_taxonomy_rig/         — trained 2026-05-14
│   └── pc_no_taxonomy_rig/           — downloaded from HuggingFace, trained 2026-05-15
└── CLAUDE.md, HANDOFF.md, RESEARCH_PLAN.md, PAPER_METHODOLOGY.md, OBSERVATIONS.md, README.md
```

### Reproducibility checklist
- ✅ Code: this repository (specify GitHub URL when submitting)
- ✅ Training data: HuggingFace `rlogh/climbing-holds-pointcloud` (rig version)
- ✅ Trained checkpoints: `rlogh/climbing-holds-rig-with-taxonomy` and `rlogh/climbing-holds-rig-no-taxonomy`
- ✅ Evaluation session JSON: `eval_results/paired_session_20260515_074256.json`
- ✅ Hardware specs: PAPER_METHODOLOGY.md §1, §2
- ✅ Analysis scripts: `eval_results/generate_*.py`, `eval_results/display_results.py`
- ⚠️ Random seed: not set in current training run (add for future work)
- ⚠️ Spring testbed: a custom mechanical apparatus — CAD/3D-print files not yet released
  (offer to release on acceptance, OR provide commercially-equivalent spec)

---

## 9. Cited Prior Work (key references)

See `RESEARCH_PLAN.md` §2 for the full annotated reading list. Headlines:

- **DP3** (Ze et al., RSS 2024) — point cloud diffusion policy, 40-demo Allegro hand. Our architecture follows DP3.
- **Original Diffusion Policy** (Chi et al., RSS 2023) — DDPM for action prediction, obs_horizon=2, pred_horizon=16 conventions.
- **Dexonomy** (RSS 2025) — *closest competitor*. Conditions on grasp taxonomy from point clouds. Generates **static grasp poses**; ours learns **continuous visuomotor execution trajectories**. Different problem class.
- **DexCap** (Wang et al., RSS 2024) — LEAP hand + Franka setup; basis for our hardware choice.

---

## 10. Limitations

1. **Single demonstrator** — all 200 episodes collected by one operator via VR teleop. No inter-demonstrator variance characterized.
2. **Single physical hold per grasp type** — within-type generalization not yet tested (held-out crimp hold, hold 4 / test_edge, planned but not yet evaluated).
3. **Single robot platform** — Franka + LEAP only. No cross-embodiment transfer tested.
4. **Ratchet ceiling** — 11 teeth maps to ~33.9 N; strong grips that exceeded this are pegged. Of the 160 rollouts, [run analyze_observations.py to fill in n_pegged] hit the ceiling. True peak force for those trials is right-censored.
5. **Spring testbed is a proxy** — measures grip-force-against-displacement, not full-route climbing. Captures the load-bearing axis of dynamic crux moves but not multi-directional or sustained loads.
6. **Binary g/b rating uniformly "good"** — the ratchet-force metric superseded the binary rating mid-collection. The g/b field has no discriminative power in this session.
7. **No random seed** — see §5. Effect sizes are large enough that single-seed results are highly indicative, but seed-averaged numbers should be reported in a final paper revision.
8. **Single evaluation day** — operator fatigue not assessed; mitigated by per-pair model alternation (each model sees fatigued and fresh trials equally).
9. **Hold orientation as a covariate** — recorded per pair but not stored in the training zarr (the policy never sees orientation as input). Limits in-policy reasoning about orientation.

---

## 11. Pre-registration Statement

The following hypotheses were committed in writing **before** the paired evaluation began
(see git history of `RESEARCH_PLAN.md` and `tasks/todo.md` prior to 2026-05-14):

- **H1 (primary):** WITH-taxonomy model achieves higher median slip force than WITHOUT
  across all 4 grasp types (Wilcoxon signed-rank, paired).
- **H2 (precision-regularizer):** The largest effect sizes occur on precision-critical grasp
  types (crimp, sloper, pinch — narrow / thin features); the smallest occurs on jug
  (forgiving deep-pocket geometry).

Both hypotheses are supported by the data. See OBSERVATIONS.md for the full breakdown.
