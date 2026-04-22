# Related Work: Paper Summaries

For each paper recommended in RESEARCH_PLAN.md: a summary, why it belongs in the related work section, and how it differs from our work.

---

## Core Method: 3D Diffusion Policies

### Diffusion Policy — Chi et al., RSS 2023 ([arXiv:2303.04137](https://arxiv.org/abs/2303.04137))

**What it is:** The foundational paper for using DDPM (denoising diffusion probabilistic models) to predict action sequences in robot manipulation. The policy takes in observations and iteratively denoises a random action chunk into a clean trajectory. Key hyperparameters — obs_horizon=2, pred_horizon=16, action_horizon=8 — come directly from this paper.

**Why it belongs in related work:** Our policy *is* a diffusion policy. This is the intellectual ancestor of the entire approach — the loss function, the temporal U-Net backbone, the beta schedule, the chunk-execution strategy. You cannot write a paper on diffusion policy for manipulation without citing it.

**How it differs from our work:** Chi et al. use RGB image observations (not point clouds), target table-top pick-and-place (not dexterous grasping), and have no notion of grasp taxonomy conditioning. They treat the action distribution as unimodal per object. Our work adds point cloud input, a 16-DoF dexterous hand, and explicit grasp-type conditioning as a precision regularizer on that distribution.

---

### DP3 — Ze et al., RSS 2024 ([arXiv:2403.03954](https://arxiv.org/abs/2403.03954))

**What it is:** Extends Diffusion Policy to use 3D point cloud observations instead of RGB images. A PointNet++ encoder compresses the point cloud to a fixed-size vector that conditions the diffusion U-Net. Demonstrates strong performance on Allegro hand tasks (85% success) with only 40 demos.

**Why it belongs in related work:** Our architecture is a direct descendant of DP3. We use the same PointNet encoder, the same FPS downsampling to 1024 points, the same min-max normalization, the same U-Net architecture. This is our primary methodological citation — reviewers will immediately see the lineage and expect it to be cited and positioned against.

**How it differs from our work:** DP3 has no grasp taxonomy conditioning — it learns a single distribution over "good grasps" for each task. It does not address multi-mode grasp type variation. Its point cloud is used for geometric localization of a target object, not for discriminating between fundamentally different contact strategies. We add the taxonomy conditioning branch and show it functions as a precision regularizer rather than a mode selector.

---

### iDP3 — Ze et al., 2024 ([arXiv:2410.10803](https://arxiv.org/abs/2410.10803))

**What it is:** An improved variant of DP3 that uses an egocentric 3D representation — the point cloud is expressed in the robot's wrist frame rather than a fixed world frame. This eliminates the need for precise camera calibration and works with as few as 10 demonstrations. Uses 4096 points and pred_horizon=16.

**Why it belongs in related work:** We must compare our design choices against iDP3. We chose world-frame point clouds (requiring camera calibration) rather than egocentric ones. Reviewers will ask why. Citing iDP3 lets us explicitly address this: our static climbing hold setup makes world-frame cleaner (the hold does not move, so a single pre-episode scan is sufficient), and we have a fixed extrinsics calibration.

**How it differs from our work:** iDP3 solves the calibration problem but does not address grasp taxonomy conditioning. It also does not target a multi-type dexterous grasping problem — it improves the 3D representation, not the conditioning structure of the policy.

---

## Dexterous Hand Manipulation

### DexCap — Wang et al., RSS 2024 ([arXiv:2403.07788](https://arxiv.org/abs/2403.07788))

**What it is:** A mocap-based data collection system for dexterous manipulation using the LEAP Hand + Franka — the same hardware we use. Collects 10,000-point clouds per frame at 60Hz (downsampled to 20Hz), with 55–201 demos per task.

**Why it belongs in related work:** This is the closest paper to our hardware setup. Same robot arm, same LEAP hand, same point cloud modality. It legitimizes our hardware choice and gives reviewers a reference point for understanding our system. It also sets a scale reference — DexCap uses 55–201 demos per task; our 200 across 4 types is in range for the simpler per-type problem.

**How it differs from our work:** DexCap does not condition on grasp taxonomy. Each task has a single intended grasp mode — they are not trying to learn multiple qualitatively different contact strategies from the same hardware. Their focus is on the *data collection system* itself (the mocap hardware pipeline), not on the policy architecture or taxonomy conditioning.

---

### DexDiffuser — Weng et al., RA-L 2024 ([arXiv:2402.02989](https://arxiv.org/abs/2402.02989))

**What it is:** Uses a diffusion model to *generate* dexterous grasp poses on the Allegro hand. Given an object point cloud, the model denoises a joint configuration into a stable grasp. Reports 9–19% improvement over non-diffusion baselines.

**Why it belongs in related work:** DexDiffuser establishes that diffusion models are effective for dexterous grasping — a claim our work builds on. It also covers Allegro-hand grasp generation, which is adjacent to our LEAP-hand execution policy. Citing it grounds the claim that diffusion is the right framework for this problem.

**How it differs from our work:** DexDiffuser generates *static grasp poses* (a single hand configuration, not a trajectory). It is a grasp pose generator, not a visuomotor execution policy. It does not learn from human demonstrations, does not condition on grasp taxonomy, and does not execute a multi-step arm + hand trajectory. Our policy generates continuous multi-step trajectories (approach → contact → hold) from imitation learning.

---

### UniDexFPM — Wu et al., 2024 ([arXiv:2403.12421](https://arxiv.org/abs/2403.12421))

**What it is:** A diffusion policy for dexterous functional pre-grasp manipulation — repositioning objects into a graspable pose before grasping. Trains on 1,026 objects using teacher-student distillation. Uses relative joint position changes as actions rather than absolute positions.

**Why it belongs in related work:** UniDexFPM demonstrates diffusion policies working with dexterous hands on complex multi-step tasks. It is part of the expanding body of work showing that diffusion-based imitation learning scales to high-DoF manipulation. Citing it helps position our contribution in terms of what the current frontier of dexterous diffusion policies can and cannot do.

**How it differs from our work:** UniDexFPM focuses on pre-grasp object repositioning (a separate sub-problem), uses a simulation-trained teacher policy, and does not condition on grasp taxonomy. Its 1,026-object diversity addresses category generalization, not grasp-type specificity. We address the complementary question: given the object is positioned, which grasp type to use and how to execute it.

---

## Grasp Taxonomy and Type-Aware Grasping

### The GRASP Taxonomy — Feix et al., IEEE THMS 2016

**What it is:** The canonical taxonomic classification of human grasps. Through systematic analysis, Feix et al. identified 33 distinct human grasp types organized by opposition type and virtual fingers. This is the foundational reference for any work that discusses grasp categories — crimp, sloper, pinch, and jug all map onto entries in this taxonomy.

**Why it belongs in related work:** Our central contribution is grasp-taxonomy-aware policy conditioning. We need to ground our 4-class taxonomy in an established scientific framework rather than appearing to have invented ad-hoc categories. Citing Feix et al. shows that our crimp/sloper/pinch/jug categories are not arbitrary — they correspond to recognized functional grasp types with distinct biomechanical signatures.

**How it differs from our work:** Feix et al. are doing human hand science, not robotics. Their taxonomy is descriptive and does not address how a robot should plan or execute grasp-type-specific trajectories. We take their framework as a prior and operationalize it for robot policy conditioning.

---

### Dexonomy — RSS 2025 ([arXiv:2504.18829](https://arxiv.org/abs/2504.18829)) — CLOSEST COMPETITOR

**What it is:** Conditions grasp generation on GRASP taxonomy categories (31 types) from a single-view point cloud using a Shadow Hand. Given a point cloud and a requested grasp type, the model generates a static grasp configuration (pre-grasp, grasp, and squeeze poses). Reports 82.3% real-world success.

**Why it belongs in related work:** Dexonomy is the closest competitor in the literature. Both papers condition on grasp taxonomy and use point cloud inputs with a dexterous hand. They share the same positioning in the novelty gap. Reviewers will compare against this paper directly. It must be cited with a crisp differentiation.

**How it differs from our work:** Dexonomy generates three static snapshots (poses) that are handed to a motion planner. It has no imitation learning, no continuous trajectory, and no arm control — only the hand configuration at a few key moments. Our system learns the full visuomotor execution trajectory (arm approach + dexterous hand closure + force application) from human demonstrations via diffusion policy. The problem is fundamentally different: pose generation vs. trajectory execution. Additionally, Dexonomy learns across 31 generic household grasp types; we specialize to 4 climbing-specific types and study the within-distribution precision regularization effect — the finding that taxonomy conditioning tightens the sampling distribution rather than selecting between modes.

---

### Grasp as You Say — Tian et al., NeurIPS 2024 ([arXiv:2405.19291](https://arxiv.org/abs/2405.19291))

**What it is:** Conditions dexterous grasp generation on free-form natural language descriptions rather than discrete taxonomy labels. Given an object point cloud and a language instruction ("hold this like a pen"), the model generates a grasp pose.

**Why it belongs in related work:** This paper is directly relevant to both our Part 1 (VLM-based grasp type classifier) and our taxonomy conditioning story. It shows the community is interested in semantic/functional conditioning of dexterous grasps, and validates that the "which grasp type" question is scientifically interesting. It also provides a contrast case for our discrete label approach.

**How it differs from our work:** Language conditioning is open-vocabulary and ambiguous — "hold firmly" could map to many grasp types. Our discrete 4-class taxonomy is explicit and unambiguous. More importantly, like DexDiffuser and Dexonomy, this generates static grasp *poses* — not execution trajectories. There is no imitation learning, no arm control, no continuous policy.

---

### OmniDexVLG — Dec 2024 ([arXiv:2512.03874](https://arxiv.org/abs/2512.03874))

**What it is:** A multi-agent VLM reasoning pipeline that infers grasp semantics (including taxonomy type) from visual inputs and passes them downstream to a grasp pose generator. Uses chained VLM calls for scene understanding, grasp type inference, and pose generation.

**Why it belongs in related work:** OmniDexVLG addresses the same "VLM → grasp type" inference step that our Part 1 (VLM classifier) tackles. It shows that VLMs can reason about taxonomy-level grasp categories from visual inputs — directly supporting the feasibility of our Part 1 design. It also appears in the novelty gap table.

**How it differs from our work:** The VLM reasoning feeds into a static pose generator, not an execution policy. Like the others in this group, there is no imitation learning or continuous trajectory. OmniDexVLG treats the VLM as the primary intelligence and the grasp generator as a downstream module; our work uses a simple discrete label (output of Part 1) to condition a learned diffusion policy (Part 2) that handles the physically complex execution.

---

### DexGraspVLA — AAAI 2026 ([arXiv:2502.20900](https://arxiv.org/abs/2502.20900))

**What it is:** A two-tier architecture where a VLM high-level planner decides *what* to grasp (object identity and location) and a diffusion-based low-level policy executes the grasp trajectory. VLM output conditions the diffusion policy.

**Why it belongs in related work:** DexGraspVLA is the closest architectural match — it also uses a VLM-to-diffusion pipeline with a dexterous hand. Our Part 1 + Part 2 design is structurally similar. Citing it lets us explain exactly where we depart: the VLM in DexGraspVLA selects *which object* to grasp (object identity/location), while our Part 1 selects *which grasp type from taxonomy* to apply to a given object.

**How it differs from our work:** DexGraspVLA has no taxonomy conditioning. The VLM tells the policy "grasp object X at location Y," not "use a crimp grip." The diffusion policy then executes whatever grasp it learned for that object class — a single implicit mode per class. We condition explicitly on the functional grasp type, allowing the same object to be grasped in multiple valid ways, and we show that this conditioning acts as a precision regularizer on the sampling distribution.

---

### CrossDex — ICLR 2025

**What it is:** A method for transferring dexterous grasps across different hand morphologies using an eigengrasp action space — a low-dimensional basis of hand configurations derived from PCA of human and robot joint spaces.

**Why it belongs in related work:** CrossDex addresses the problem of cross-morphology grasp generalization, which is adjacent to our grasp taxonomy work. Both papers ask "what is the right parameterization of dexterous grasp space?" CrossDex answers with eigengrasps; we answer with functional taxonomy labels. Citing it positions our label-based approach as a deliberate, interpretable alternative to learned latent action spaces.

**How it differs from our work:** CrossDex's eigengrasp space is learned from data and is morphology-agnostic but semantically opaque — you cannot easily say "this eigengrasp vector means crimp." Our 4-class taxonomy is human-interpretable and grounded in functional contact requirements. CrossDex also focuses on grasp *transfer* across hands, not on conditioning a continuous execution policy on grasp type.

---

## Multi-Modal and Point Cloud Methods

### GenDP — CoRL 2024 ([arXiv:2410.17488](https://arxiv.org/abs/2410.17488))

**What it is:** Uses 3D semantic feature fields (dense per-point semantic descriptors from foundation models) instead of raw XYZ point clouds. The semantic features provide category-level geometric correspondences, enabling generalization from 20% to 93% success on unseen object instances of the same category.

**Why it belongs in related work:** GenDP demonstrates what happens when you add semantic information to 3D point cloud policies — dramatic generalization improvement. This is relevant to the discussion of why point clouds are the right observation modality and what their limits are. It also connects to our Part 1/Part 2 discussion: our grasp taxonomy label is a form of semantic conditioning, and GenDP shows that semantic information added to point clouds meaningfully changes policy behavior.

**How it differs from our work:** GenDP's semantic fields are dense (per-point) and derived from vision foundation models — they encode object identity, not grasp type. Our conditioning is a single discrete scalar (the grasp type label) appended to the observation, not a modification of the point cloud itself. GenDP addresses category-level instance generalization; we address grasp-type-specific trajectory precision.

---

### FPV-Net — Feb 2025 ([arXiv:2502.12320](https://arxiv.org/abs/2502.12320))

**What it is:** Fuses RGB images with point clouds for robot manipulation policy learning, using AdaLN (adaptive layer normalization) to blend the two modalities. Tests only on parallel gripper tasks.

**Why it belongs in related work:** FPV-Net is directly relevant to our design decision to use point cloud *only* (no RGB) for our diffusion policy. We can cite it to acknowledge the multi-modal alternative and explain why we chose the pure-point-cloud approach: appearance generalization (DP3's motivation), avoidance of lighting sensitivity, and alignment with DexCap/DP3 baselines. FPV-Net's restriction to parallel grippers also highlights a gap our work fills.

**How it differs from our work:** FPV-Net does not address dexterous hands, taxonomy conditioning, or multi-mode grasping. It is a modality fusion paper, not a grasp type conditioning paper.

---

### Point Cloud Matters — NeurIPS 2024

**What it is:** A systematic ablation study of point cloud representation choices for robot manipulation — number of points, sampling strategy, coordinate frame, normalization — and their effects on policy performance.

**Why it belongs in related work:** This paper directly supports our design choices (1024 points, FPS downsampling, world-frame coordinates, min-max normalization). When reviewers ask "why 1024 points?" or "why FPS?" this paper shows those are not arbitrary choices. It also validates that point cloud representation choices meaningfully impact policy success rates, which contextualizes our workspace-bound tuning work.

**How it differs from our work:** It is a representation study, not a new architecture or conditioning method. It does not address grasp taxonomy or dexterous hands. We build on its findings to make our representation choices, then go further by adding taxonomy conditioning.

---

## Reactive/Tactile Policies (Future Extension)

### Reactive Diffusion Policy — 2025 ([arXiv:2503.02881](https://arxiv.org/abs/2503.02881))

**What it is:** Adds a fast reactive correction loop inside diffusion policy action chunks using visual-tactile feedback. A slow diffusion planner generates action chunks; a fast feedback controller corrects them based on contact signals. Currently demonstrated on parallel grippers only.

**Why it belongs in related work:** Best cited as a limitation and future direction: our current policy has no tactile feedback, and Reactive Diffusion Policy shows the path to adding it. It also contextualizes the gap in dexterous tactile policy learning — the approach is shown on parallel grippers but not yet extended to multi-fingered hands.

**How it differs from our work:** Our policy is open-loop at the action chunk level (no within-chunk correction). Reactive DP adds a closed-loop correction mechanism. Applying it to a 16-DoF LEAP hand with climbing-specific contact patterns is an open problem we can identify as future work.

---

## Novelty Summary

| Paper | Point Cloud | Taxonomy Cond | Execution Policy | Dexterous Hand |
|---|---|---|---|---|
| Chi et al. (DP, RSS 2023) | RGB | — | yes | no |
| Ze et al. (DP3, RSS 2024) | yes | — | yes | yes |
| Wang et al. (DexCap, RSS 2024) | yes | — | yes | yes (LEAP) |
| Dexonomy (RSS 2025) | yes | 31 types | **no** (pose gen) | yes |
| Grasp as You Say (NeurIPS 2024) | yes | language | **no** (pose gen) | yes |
| DexGraspVLA (AAAI 2026) | no | object ID only | yes | yes |
| **Ours** | **yes** | **4-class discrete** | **yes (arm+hand traj)** | **yes (LEAP)** |

The central novelty is the simultaneous combination of all four columns — no prior work has all of them — plus the mechanistic finding that taxonomy conditioning acts as a *precision regularizer* (tightening the sampling distribution around the correct subtype) rather than a mode selector. This is the primary contribution beyond the benchmark itself.
