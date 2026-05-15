## SESSION START

1. Read tasks/lessons.md — apply all lessons before touching anything

2. Read tasks/todo.md — understand current state

3. If neither exists, create them before starting

## PROJECT DOCUMENTS — READ THESE TO GET UP TO SPEED

| File | Purpose | When to read |
|------|---------|--------------|
| `tasks/todo.md` | Current status + immediate next steps. Always read first. | Every session |
| `tasks/lessons.md` | Mistakes made + rules to prevent recurrence. | Every session |
| `HANDOFF.md` | Full technical reference: hardware, file layout, state/action space, data flow, known quirks. The most complete single doc. | When doing anything non-trivial |
| `RESEARCH_PLAN.md` | Full research design: problem statement, related work citations, two-part architecture (VLM identifier + diffusion policy), experimental setup, training config, ablations, timeline. | When making architecture or research decisions |
| `PAPER_METHODOLOGY.md` | Paper-ready methodology reference: exact compute specs, robot hardware versions, dataset stats, training-time numbers, statistical methods, reproducibility checklist, limitations. **Use directly when writing the Methods section.** | When writing the paper |
| `OBSERVATIONS.md` | Pre-written observation paragraphs with numbers/tables already filled in (failure-mode breakdown, win counts, orientation effects, order-check, quote-ready sentences). **Use directly when writing Results and Discussion.** | When writing the paper |
| `IMPLEMENTATION_LOG.md` | Chronological changelog of every code change made, with session-by-session summaries. | When debugging or resuming code work |
| `PROGRESS_UPDATE.md` | Slide-ready status summary: what's been built, data collection numbers, next steps. | When presenting to professor or writing slides |
| `README.md` | Instructions for the training cluster: clone repo, install deps, download HF dataset, run training. | When setting up the cluster machine |



## WORKFLOW



### 1. Plan First

- Enter plan mode for any non-trivial task (3+ steps)

- Write plan to tasks/todo.md before implementing

- If something goes wrong, STOP and re-plan — never push through



### 2. Subagent Strategy

- Use subagents to keep main context clean

- One task per subagent

- Throw more compute at hard problems



### 3. Self-Improvement Loop

- After any correction: update tasks/lessons.md

- Format: [date] | what went wrong | rule to prevent it

- Review lessons at every session start



### 4. Verification Standard

- Never mark complete without proving it works

- Run tests, check logs, diff behavior

- Ask: "Would a staff robotics research engineer approve this?"



### 5. Demand Elegance

- For non-trivial changes: is there a more elegant solution?

- If a fix feels hacky: rebuild it properly

- Don't over-engineer simple things



### 6. Autonomous Bug Fixing

- When given a bug: just fix it

- Go to logs, find root cause, resolve it

- No hand-holding needed



## CORE PRINCIPLES

- Simplicity First — touch minimal code

- No Laziness — root causes only, no temp fixes

- Never Assume — verify paths, APIs, variables before using

- Ask Once — one question upfront if unclear, never interrupt mid-task



## TASK MANAGEMENT

1. Plan → tasks/todo.md

2. Verify → confirm before implementing

3. Track → mark complete as you go

4. Explain → high-level summary each step

5. Learn → tasks/lessons.md after corrections



## LEARNED

### Two-terminal data collection architecture (DO NOT suggest a third terminal)

Data collection requires exactly **two terminals** on this Linux machine:

- **Terminal 1**: `VR_Teleoperation_Minimum.py`
  - Path: `TeleoperationUnity/Robot Control - Python/Franka Scripts/`
  - Controls Franka arm; receives Quest 2 finger data over UDP WiFi;
    forwards finger data to `localhost:8002`

- **Terminal 2**: `collect_data.py`
  - Path: `data_collection/`
  - Wraps `LeapPipDipTeleop` internally — this is what controls the LEAP
    hand motors directly over `/dev/ttyUSB*`
  - Listens on `localhost:8002` for finger data from Terminal 1
  - Records Franka joints, LEAP joints, point cloud to zarr

`leap_pip_dip_teleop.py` (in `TeleoperationUnity/LEAP/leaphandv1/for_transfer/`)
is **NOT needed** during data collection — it is only used when doing VR teleop
without recording. Never suggest running it as a third terminal for data collection.

### IP addresses

Only one IP ever needs updating: **`Oculus_IP`** in `VR_Teleoperation_Minimum.py`
lines 72 and 74 — this is the Quest 2's current WiFi IP, which changes periodically.

- `franka_IP = "172.26.67.113"` — fixed (hardwired Franka control box), never changes
- `Oculus_IP = "172.26.39.18"` — Quest 2 WiFi IP, update when Quest reconnects to WiFi
- `Hand_IP` — only used when `USE_ROBOHAND = True`, which is currently `False`; ignore it

`USE_ROBOHAND = False` → only the `else` branch on line 74 is executed.

### Commands (always source both before running any Python on this machine)

```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
```

Terminal 1:
```bash
cd ~/Desktop/tele/"TeleoperationUnity/Robot Control - Python/Franka Scripts"
python3 VR_Teleoperation_Minimum.py
```

Terminal 2 (after Terminal 1 prints "Teleoperation started"):
```bash
cd ~/Desktop/tele/data_collection
python3 collect_data.py --hold 0 --point-cloud --grasp-type jug --task climbing_holds_rig
```

Valid values for `--grasp-type` (must match `GRASP_TYPE_IDS` in `episode_storage.py`):

- `crimp`  → 0
- `sloper` → 1
- `pinch`  → 2
- `jug`    → 3

Always use one of these four exact strings when running `collect_data.py --point-cloud`.

### `--task` is the dataset name — pick one and never mix eras (CRITICAL)

`collect_data.py` writes to `{DEFAULT_DATASET_DIR}/{task_name}.zarr`. Defaults
are `DEFAULT_DATASET_DIR=/mnt/ssd/rumi_tele_datasets` and
`DEFAULT_TASK_NAME=climbing_holds`. The zarr is opened with `zarr.open(..., mode="a")`,
which means **`collect_data.py` APPENDS to whatever `<task>.zarr` already exists
on disk** — it does not overwrite, but new episodes are tacked onto the back of
the existing arrays with no separator.

This is dangerous because the legacy `climbing_holds.zarr` on disk is the
**flat-table 200-episode dataset** (collected at `z_min=0.006`, no rig). Running
`collect_data.py` with no `--task` will silently merge new rig episodes into
that legacy dataset, producing a frankenstein zarr that mixes two distributions.

**Rule:** for any **spring-testbed-era collection**, always pass
`--task climbing_holds_rig` (or another distinct name) so the new data lives
in its own zarr file (`/mnt/ssd/rumi_tele_datasets/climbing_holds_rig.zarr`).
Use the SAME `--task` name across every collection session for the same dataset
(all 4 hold/grasp combos = one zarr). NEVER omit `--task` on this machine; the
default name points at the legacy flat-table dataset.

Existing zarrs on disk (2026-05-13) — do not touch with `--task` overrides:
- `climbing_holds.zarr` — flat-table 200-ep PC dataset (legacy)
- `climbing_holds_v2.zarr` / `climbing_holds_v2_backup_36dim.zarr` — legacy image-era
- `climbing_holds_legacy_image.zarr` — legacy image-era
- `climbing_holds_upload.zarr` — HuggingFace upload snapshot

### Arm park pose (fully out of all camera views for clean point cloud capture)

```python
PARK_ARM_JOINTS = [-0.11426599, -0.56029082, -0.06635159, -2.17443357, 0.04112932, 2.15592909, 0.54378958]
```

This is the joint configuration where the Franka arm is **fully clear of all 4 RealSense cameras** (cams 2–5), verified 2026-03-18. Use this pose whenever a clean point cloud capture is needed (evaluate.py, check_pc_sensitivity.py, etc.). The previous RESET_ARM_JOINTS (`[-0.1111, -0.1703, ...]`) sits too low and clips the cameras.

### Workspace bounds — current value (2026-05-11, spring testbed era)

```python
DEFAULT_WORKSPACE_BOUNDS = {
    "x_min": 0.30, "x_max": 0.85,
    "y_min": -0.35, "y_max": 0.35,
    "z_min": 0.027, "z_max": 0.40,
}
```

The **spring testbed lifts each hold ~2 cm above the original table surface**, so the lower bound moves from the table-era `z_min=0.006` to `z_min=0.027` to clip the rig deck while keeping the full hold geometry. `z_max=0.40` gives headroom for the rig + tall jugs.

**History (do not regress past these):**
- `z_min=-0.02` (table-era) — catastrophic; 95% table noise, model ignored PC.
- `z_min=0.006` (table-era, correct for the original flat-table dataset).
- `z_min=0.027` (CURRENT, spring testbed era — set 2026-05-11).
- `z_max=0.30` (table-era) → `z_max=0.40` (testbed-era; rig is taller than bare table).

**Before collecting any new data:** always run `check_pc_sensitivity.py` first with the spring testbed in place and a hold mounted. Verify:
- Z centroid sits above the rig deck (not at it),
- the full hold shape is captured (not just the top),
- centroids shift by cm when the hold is rotated through the 5 orientations.

### Auto-park IPC protocol (collect_data.py ↔ GotoPoseLive)

During data collection with `--point-cloud`, the arm must be parked for clean PC
capture. **Cannot** create a second FrankaArm or call `stop_skill()` externally —
doing so permanently kills GotoPoseLive's live skill (self.initialize stays False,
no recovery). Instead, file-based IPC via `/tmp/franka_park_*`:

1. `collect_data.py` writes `/tmp/franka_park_request` with `{"joints": [...]}`
2. `GotoPoseLive.run()` detects file → stops skill → `goto_joints(park)` → writes `/tmp/franka_park_done`
3. `collect_data.py` sees done → captures point cloud → writes `/tmp/franka_park_resume`
4. `GotoPoseLive` sees resume → cleans up files → sets `self.initialize = True` → restarts live skill

This way Terminal 1 (VR teleop) never needs to be restarted.

### Thumb IK test file (2026-03-26)

`TeleoperationUnity/LEAP/leaphandv1/for_transfer/leap_thumb_ik_test.py` is a
drop-in replacement for `leap_pip_dip_teleop.py` that replaces thumb retargeting
with PyBullet IK. **The working code is NOT modified** — this is a separate test file.

Key facts:
- Index/Middle/Pinky: identical to `leap_pip_dip_teleop.py` (no change)
- Thumb: uses the 12 raw bone quaternions (UDP values 16-27) that were previously
  discarded; FK → BeaVR-style Cartesian transform → PyBullet IK on LEAP URDF
- Requires `pybullet` (installed in `~/franka` venv as of 2026-03-26)
- URDF: `beavr-bot-reference/assets/urdf/leap_hand/leap_hand_right.urdf`
- `beavr-bot-reference/` is gitignored — does not affect pushes

To run the IK test (same two terminals as normal teleop, just swap Terminal 2 script):
```bash
cd ~/Desktop/tele/TeleoperationUnity/LEAP/leaphandv1/for_transfer
python3 leap_thumb_ik_test.py
```

Tunable constants are marked `# TUNE` at the top of the file. Print diagnostics
every 100 frames; adjust `VERBOSE_THUMB_EVERY` to change frequency.

### Pull test angle convention (evaluate.py + paired_eval.py)

When `--pull-dist` is set, the arm moves in a straight line in the world X,Y plane
after the policy converges. The angle is entered in degrees (0–360):

```
                 0° / 360°
                  (+X, away from robot)
                      ↑
                      |
                      |
  90°  (+Y left) ←----+----→  270° (-Y right)
                      |
                      |
                      ↓
                 180° (-X)
               [ROBOT BASE]
                 [YOU]
```

- **0°** — arm pulls hold away from the robot (toward you)
- **90°** — arm pulls to the robot's left
- **180°** — arm pulls toward the robot base (away from you)
- **270°** — arm pulls to the robot's right

### Spring testbed protocol (active plan as of 2026-05-11)

The spring displacement testbed constrains hold motion to a single axis fixed at
**180° (toward the robot base, -X direction)**. The pull angle is **hardcoded at
180° in both eval scripts** — no `--pull-angle` flag needed or accepted by
`paired_eval.py`; `evaluate.py` still accepts it for non-testbed use but defaults
to 180°.

The varying experimental dimension is the **hold orientation** on the testbed:
five discrete angles relative to the pull axis — **−45°, −22.5°, 0°, +22.5°, +45°**.
Orientations beyond ±45° are not tested (frictional slip is certain).

**Ratchet and force measurement:** the linear ratchet captures the hold's peak
displacement even when it slips mid-trial. After each pull, the script prompts for
the ratchet tooth count (0–11). Each tooth = **9.3 mm**; max travel = 11 teeth =
102.3 mm. The two springs in parallel are empirically calibrated as:

```
F_total = 2 × (0.59 + 0.8 × x_in)  [lbf]   where x_in = displacement_mm / 25.4
        = (1.18 + 1.6 × x_in) × 4.448       [N]
```

| Teeth | Disp (mm) | F (lbf) | F (N)  |
|-------|-----------|---------|--------|
| 0     | 0         | 1.18    | 5.2    |
| 1     | 9.3       | 1.77    | 7.9    |
| 3     | 27.9      | 2.94    | 13.1   |
| 5     | 46.5      | 4.11    | 18.3   |
| 7     | 65.1      | 5.28    | 23.5   |
| 9     | 83.7      | 6.45    | 28.7   |
| 11    | 102.3     | 7.62    | 33.9   |

**No force / displacement signal is fed into the policy or stored in the zarr** —
it lives only in the eval JSON (`ratchet.teeth`, `ratchet.displacement_mm`,
`ratchet.force_lbf`, `ratchet.force_N` fields per trial).

Training demonstrations (teleoperated only — operator drives the arm + LEAP hand
to grip the hold while it sits on the rig, no pull during collection):
**10 episodes per orientation × 5 orientations = 50 episodes per hold**. Collect
all 10 reps at one orientation before rotating the hold to the next angle.

Evaluation: at least 4 pairs per orientation per grasp type (20 pairs total
across the 5 orientations). Orientation is not stored in the zarr (policy
doesn't consume it as input); randomize the orientation per eval trial and
log it in the per-pair eval JSON / lab notebook instead.

**Pull control:** impedance control with **differential per-axis stiffness**:
- X (pull axis, -X direction): `kx = 4000 N/m` — stiff, drives the 10.5 cm motion
- Y (lateral): `ky = 100 N/m` — compliant, allows natural hand flex side-to-side
- Z (vertical): `kz = 2000 N/m` — must hold LEAP hand (~1 kg = ~10 N) against gravity.
  At kz=100 the arm droops ~10 cm under hand weight; at kz=2000 droop ≈ 5 mm.

Defaults are baked in; override with `--pull-stiffness` (kx), `--pull-lateral-stiffness`
(ky), `--pull-z-stiffness` (kz), `--pull-z-bias` (upward Z offset to counter LEAP
hand weight). At kx=4000, a perfect grip equilibrates at ~97 mm (≈10 ratchet teeth)
due to spring back-force; this is expected — readings ≥ 9 = strong grip.

```bash
# Full evaluation session — all 5 orientations per hold, 4 pairs each (80 pairs total)
# Format: grasp:hold:pairs:orientation_deg
# IMPORTANT: --batches value MUST be quoted and continuation lines MUST start at column 0
# (no leading whitespace), otherwise bash splits the comma-list into separate argv tokens.
python3 paired_eval.py --pull-dist 0.130 --batches "\
jug:0:4:-45,jug:0:4:-22.5,jug:0:4:0,jug:0:4:22.5,jug:0:4:45,\
crimp:1:4:-45,crimp:1:4:-22.5,crimp:1:4:0,crimp:1:4:22.5,crimp:1:4:45,\
sloper:2:4:-45,sloper:2:4:-22.5,sloper:2:4:0,sloper:2:4:22.5,sloper:2:4:45,\
pinch:3:4:-45,pinch:3:4:-22.5,pinch:3:4:0,pinch:3:4:22.5,pinch:3:4:45"

# Single hold session (one hold, all 5 orientations):
python3 paired_eval.py --pull-dist 0.130 --batches \
  jug:0:4:-45,jug:0:4:-22.5,jug:0:4:0,jug:0:4:22.5,jug:0:4:45

# Single model evaluation with spring testbed
python3 evaluate.py --checkpoint ... --pull-dist 0.130
```

**Data-collection (teleop) — no `--pull-dist` flag.** Pulls happen only at
evaluation. During collection, the operator teleoperates the grasp, ends the
episode, and rotates the hold on the rig to the next of the 5 orientations
after 10 reps.

### UDP packet format from Unity (28 values)

`HandController.cs` (the LEAP version in `TeleoperationUnity/LEAP/leaphandv1/`) sends
**28 tab-separated values** when hand tracking is active:

- Values 0–15: 16 joint angles in **degrees**, ordered as
  Index[DIP, PIP, MCP_Flex, MCP_Abd] × Middle × Pinky × Thumb
- Values 16–19: Thumb base quaternion (metacarpal, `BoneRotations[3]`) as x,y,z,w
- Values 20–23: Thumb mid quaternion (proximal, `BoneRotations[4]`) as x,y,z,w
- Values 24–27: Thumb tip quaternion (distal, `BoneRotations[5]`) as x,y,z,w

`leap_pip_dip_teleop.py` strips to 16 values (discards 16-27).
`leap_thumb_ik_test.py` uses all 28.


3 bad for the jug,
0 bad for the crimp
2 bad sloper
2 bad pinch