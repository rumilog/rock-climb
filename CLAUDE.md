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
python3 collect_data.py --hold 0 --point-cloud --grasp-type crimp
```

Valid values for `--grasp-type` (must match `GRASP_TYPE_IDS` in `episode_storage.py`):

- `crimp`  → 0
- `sloper` → 1
- `pinch`  → 2
- `jug`    → 3

Always use one of these four exact strings when running `collect_data.py --point-cloud`.

### Arm park pose (fully out of all camera views for clean point cloud capture)

```python
PARK_ARM_JOINTS = [-0.11426599, -0.56029082, -0.06635159, -2.17443357, 0.04112932, 2.15592909, 0.54378958]
```

This is the joint configuration where the Franka arm is **fully clear of all 4 RealSense cameras** (cams 2–5), verified 2026-03-18. Use this pose whenever a clean point cloud capture is needed (evaluate.py, check_pc_sensitivity.py, etc.). The previous RESET_ARM_JOINTS (`[-0.1111, -0.1703, ...]`) sits too low and clips the cameras.

### Workspace bounds — verified correct value (2026-03-18)

```python
DEFAULT_WORKSPACE_BOUNDS = {
    "x_min": 0.30, "x_max": 0.85,
    "y_min": -0.35, "y_max": 0.35,
    "z_min": 0.006, "z_max": 0.30,
}
```

`z_min=0.006` is the **empirically verified correct value**. This is at the table surface (table tops at Z=0.006), so all 1024 FPS points land on the climbing hold geometry. Confirmed with `check_pc_sensitivity.py`: Z centroid ≈ 0.034 m, full hold shape captured, zero table noise.

**History:** z_min was previously -0.02 (catastrophic — 95% table noise, model ignored PC) and then briefly 0.008 (slightly too high, clipped hold base). The correct value is 0.006.

**Before collecting any new data:** always run `check_pc_sensitivity.py` first and verify Z centroid > 0.01 m and centroids shift when the hold moves.

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

- **0°** — arm pulls hold away from the robot (toward you) — most common
- **90°** — arm pulls to the robot's left
- **180°** — arm pulls toward the robot base (away from you)
- **270°** — arm pulls to the robot's right

Pick the angle perpendicular to the hold's main axis so you're testing grip
friction, not sliding along a smooth face. If `--pull-angle` is not set, the
terminal prompts you for the angle before each pull (recommended since holds rotate).

```bash
# 5 cm pull, prompt for angle each trial (recommended — holds rotate)
python3 evaluate.py --checkpoint ... --pull-dist 0.05

# 5 cm pull, fixed angle for whole session
python3 evaluate.py --checkpoint ... --pull-dist 0.05 --pull-angle 0

# Same for paired_eval
python3 paired_eval.py --pull-dist 0.05
```

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
