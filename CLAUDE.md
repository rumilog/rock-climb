## SESSION START

1. Read tasks/lessons.md — apply all lessons before touching anything

2. Read tasks/todo.md — understand current state

3. If neither exists, create them before starting



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
