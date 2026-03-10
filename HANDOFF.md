# Technical Handoff: Diffusion Policy for LEAP Hand + Franka Arm Grasping

ls /dev/ttyUSB*
run this in order to check for usb plugin for LEAP hand

bash ~/frankapy/bash_scripts/start_control_pc.sh -u franka -i franka-Alienware-Area-51-R5 -g 0
run this to start ros1

source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
for sourcing workspaces for interacting with franka arm



## 1. Project Overview

**Goal:** Train a diffusion policy (imitation learning) to autonomously grasp rock-like climbing holds using a LEAP Hand mounted on a Franka Emika arm. The pipeline is: collect human teleoperation demonstrations → store in zarr → train diffusion policy → evaluate on real robot.

**Current status:** The training pipeline works end-to-end. An initial model was trained on 44 episodes (old 30-dim dataset at `climbing_holds.zarr`). A new 36-dim dataset (`climbing_holds_v2.zarr`) with RGBD and 3D hold pose was started but has 0 episodes because the LEAP hand motor control is broken during data collection (see Section 8).

---

## 2. Hardware

| Component | Details |
|-----------|---------|
| **Arm** | Franka Emika Panda (7-DOF), controlled via `frankapy` |
| **Hand** | LEAP Hand v1 (16 Dynamixel motors), 4 fingers: Index, Middle, Pinky, Thumb |
| **Cameras** | 4× Intel RealSense D415/D455, IDs [2, 3, 4, 5], 848×480 RGB+D |
| **VR Controller** | Meta Quest 2 headset, communicating over UDP |
| **Computer** | Ubuntu (Python 3.8/3.10), NVIDIA GPU, ROS Noetic |

---

## 3. Repository Layout

```
/home/rumi/Desktop/tele/
├── data_collection/
│   ├── collect_data.py         # Main data collection script (Terminal 2)
│   ├── episode_storage.py      # EpisodeBuffer + ZarrDatasetWriter
│   ├── hold_detector.py        # 3D hold pose detection from depth
│   ├── train.py                # Diffusion policy training
│   ├── evaluate.py             # Real robot policy evaluation
│   ├── verify_dataset.py       # Dataset integrity checks
│   └── trim_dataset.py         # Dataset trimming utility
├── TeleoperationUnity/
│   ├── Robot Control - Python/
│   │   └── Franka Scripts/
│   │       ├── VR_Teleoperation_Minimum.py  # Terminal 1: Franka arm control + LEAP data forwarding
│   │       └── Teleoperation.py             # Core teleoperation class (used by above)
│   └── LEAP/leaphandv1/for_transfer/
│       ├── leap_pip_dip_teleop.py           # LeapPipDipTeleop class (motor control + UDP)
│       ├── UdpComms.py                      # Generic UDP communication
│       ├── server_env.py                    # NOT used in current pipeline (legacy)
│       └── LEAP_Hand_API/python/
│           └── leap_hand_utils/
│               ├── dynamixel_client.py      # Low-level Dynamixel serial protocol
│               └── leap_hand_utils.py       # Allegro↔LEAP coordinate conversions
├── datasets/
│   ├── climbing_holds.zarr      # Original 44 episodes, 30-dim state, RGB only
│   ├── climbing_holds_v2.zarr   # New dataset (0 episodes so far), 36-dim state, RGBD
│   └── img_cache/               # Memory-mapped numpy image caches for training
└── checkpoints/
    ├── quick_sanity/             # 50-epoch sanity check run (96px)
    └── overnight_224/            # 300-epoch overnight run (224px), best.pt is here
```

---

## 4. Two-Terminal Architecture (Data Collection)

This is the most critical architectural detail to understand:

### Terminal 1: `VR_Teleoperation_Minimum.py`
- Location: `TeleoperationUnity/Robot Control - Python/Franka Scripts/`
- Activates `frankapy` environment: `source ~/franka/bin/activate && source ~/frankapy/catkin_ws/devel/setup.bash`
- Connects to Quest 2 via UDP (Quest IP set in `Teleoperation(Oculus_IP="...")` call)
- Controls Franka arm position via `GotoPoseLive` (cartesian impedance)
- **Crucially:** Receives finger angle data from Quest and **forwards it** to `localhost:8002` via `self.leap_forward_sock.SendData(gripper_data)` (see `Teleoperation.py` line ~145)
- This script does NOT control the LEAP hand directly

### Terminal 2: `collect_data.py`
- Location: `data_collection/`
- Same environment activation as Terminal 1
- `LeapHandRecorder` creates a `LeapPipDipTeleop` instance which:
  - Opens Dynamixel serial connection to LEAP hand (auto-detects `/dev/ttyUSB*`)
  - Listens on UDP `localhost:8002` for finger data forwarded by Terminal 1
  - Background thread receives UDP data → calls `process_udp_data()` → calls `update_full_joints_deg16()` → sends motor commands
- Main loop reads: Franka arm state (via ROS topic subscriber), LEAP hand positions (cached from control thread), camera images (4× RealSense)
- Records at 10 Hz into zarr dataset

### Data Flow
```
Quest 2 → (UDP over WiFi) → VR_Teleoperation_Minimum.py [Terminal 1]
                                  │
                                  ├── Franka arm control (GotoPoseLive)
                                  │
                                  └── Forwards gripper_data → localhost:8002
                                                                    │
                                                                    ▼
                                                  collect_data.py [Terminal 2]
                                                    ├── LeapHandRecorder._control_loop
                                                    │     reads UDP → process_udp_data → motor write
                                                    ├── Reads Franka state via ROS subscriber
                                                    ├── Reads 4 cameras
                                                    └── Records to zarr at 10 Hz
```

**Important:** `server_env.py` is NOT part of this pipeline. It was a legacy script from an older setup. All messages in the code mentioning `server_env.py` are misleading (I partially fixed these but some remain in comments/docstrings).

---

## 5. State/Action Vector (36-dim)

```
[0:7]   = Franka arm joint positions (7 DOF)
[7:10]  = End-effector position (x, y, z) in meters
[10:14] = End-effector quaternion (w, x, y, z)
[14:30] = LEAP hand joint positions (16 DOF, Allegro convention)
[30:36] = Hold pose: centroid (x, y, z) + surface normal (nx, ny, nz) in camera frame
```

The old dataset (`climbing_holds.zarr`) uses 30-dim (no hold pose). The new dataset (`climbing_holds_v2.zarr`) uses 36-dim.

Action = next-timestep state (actions are shifted by 1 timestep via `finalize_actions()`). During training, only the first 30 dims (arm + hand) are actuated; the hold pose (dims 30-36) is observational context.

---

## 6. Key Files in Detail

### `collect_data.py`
- `LeapHandRecorder`: Wraps `LeapPipDipTeleop` in a background thread. The control thread handles all serial I/O (reads + writes to Dynamixel motors). `get_joint_positions()` returns cached data without touching serial port.
- `DataCollector.setup()`: Connects to Franka (ROS subscriber), initializes LEAP hand, starts cameras, opens zarr dataset, runs hold pose scan.
- `DataCollector.run()`: Main 10 Hz loop. Reads all sensors, optionally records to `EpisodeBuffer`, shows preview window.
- Keyboard: SPACE=start/stop recording, g=save good, b=save bad, d=discard, h=re-scan hold, q=quit.
- CLI: `--hold 0 --rgbd --task climbing_holds_v2`

### `leap_pip_dip_teleop.py`
- `LeapPipDipTeleop.__init__`: Auto-detects USB port, connects DynamixelClient at 4Mbaud, configures PID gains, opens UDP listener on port 8002.
- `update_full_joints_deg16(angles_deg16)`: Core function. Takes 16 degree values from Quest, applies zero-offsets, scales, converts to radians, maps to Allegro convention, converts to LEAP convention, clips to safety limits, writes to motors. Also reads current motor positions (for the mapping function).
- Scaling parameters: `pip_scale=2`, `dip_scale=2`, per-finger multipliers, zero-offsets in degrees, post-scaling offsets in radians. These are tuned for the Quest-to-LEAP mapping.
- `process_udp_data`: Parses tab-separated float values. Expects 28 values (takes first 16) or 16 or 10.
- Dead code: Lines 358-407 (unreachable `try` block after a `return` on line 349/356). Harmless but should be cleaned up.

### `train.py`
- `GraspDataset`: Loads zarr, builds a memory-mapped numpy image cache (one-time cost, ~30s), yields (obs, action_chunk) pairs.
- `DiffusionPolicy`: DDPM-based. ResNet-18 per camera (shared weights) → 512-d. State MLP → 256-d. Fused conditioning → 1D temporal U-Net. Cosine beta schedule, 100 diffusion steps.
- Architecture: obs_horizon=2, pred_horizon=16, action_horizon=8 (during eval).
- Normalization: state and actions are z-score normalized. Stats saved in `norm_stats.json`.
- Supports `--augment` (color jitter + random crop), `--rgbd` (4-channel images), `--good-only`, `--quick` (96px, 50 epochs, 20 diffusion steps).

### `evaluate.py`
- Loads checkpoint, connects to Franka via `frankapy`, LEAP hand via direct `DynamixelClient`, cameras via `robomail.vision`.
- Reset robot to fixed pose between trials.
- Live control: starts a `goto_joints(dynamic=True, duration=1000)` skill, then streams joint targets via ROS `SensorDataGroup` messages.
- Motion clamping: `MAX_JOINT_STEP_RAD = 0.05` rad/step (~3°/step at 10Hz).
- LEAP hand: direct `write_desired_pos` (no teleop/UDP layer).
- Trial workflow: reset → SPACE to start → policy runs → SPACE/q to stop → rate g/b/s → repeat.

### `episode_storage.py`
- `EpisodeBuffer`: In-memory accumulator for one episode. `finalize_actions()` shifts states forward by 1 to create next-state actions.
- `ZarrDatasetWriter`: Appends episodes to zarr store with metadata (hold_id, quality, grasp_type).

### `hold_detector.py`
- Segments climbing hold from depth image (depth thresholding + morphology + largest contour).
- Converts to 3D point cloud, computes centroid + surface normal via PCA.
- Used by both `collect_data.py` and `evaluate.py` at startup.

---

## 7. Environment Setup

```bash
# Activate the franka virtualenv (required for both terminals)
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash

# Terminal 1: Start VR teleoperation
cd ~/Desktop/tele/"TeleoperationUnity/Robot Control - Python/Franka Scripts"
python3 VR_Teleoperation_Minimum.py

# Terminal 2: Start data collection
cd ~/Desktop/tele/data_collection
python3 collect_data.py --hold 0 --rgbd --task climbing_holds_v2

# Training (can be in any terminal with GPU access)
cd ~/Desktop/tele/data_collection
python3 train.py --zarr ../datasets/climbing_holds_v2.zarr --ckpt-dir ../checkpoints/v2 --epochs 300 --img-size 224

# Evaluation
cd ~/Desktop/tele/data_collection
python3 evaluate.py --checkpoint ../checkpoints/v2/best.pt --hold 0 --trials 5
```

**IP addresses to update when Quest 2 IP changes:**
- `VR_Teleoperation_Minimum.py` line 72/74: `Oculus_IP = "172.26.84.138"` (change to Quest's current IP)
- `leap_pip_dip_teleop.py` line 32: `OCULUS_IP` (only relevant if running standalone, not used during data collection)

---

## 8. CRITICAL OPEN BUG: LEAP Hand Stops Responding

### Symptom
During data collection, the LEAP hand moves 3-5 times at startup, then freezes. UDP messages are received at ~70 Hz, `write_desired_pos` calls report success, but motors don't move. On restart, init may hang indefinitely.

### Debug Evidence
```
[LEAP DBG] write #500:  current[0:4]=[3.118583  3.4591267 3.4667966 3.273515]
[LEAP DBG] write #1000: current[0:4]=[3.118583  3.4591267 3.4667966 3.273515]  ← IDENTICAL = FROZEN
[LEAP DBG] write #1500: current[0:4]=[3.118583  3.4591267 3.4667966 3.273515]  ← STILL FROZEN
```

Motor positions returned by `read_pos()` are exactly identical across 500+ writes. Target positions vary by up to 1.7 rad but motors don't move.

### Root Cause Analysis (Hypotheses Explored)

1. **Serial port contention (partially addressed):**
   - The original working code had a threading lock around `process_udp_data + read_pos` in the control thread, and `get_joint_positions` also acquired the lock (but only read cached data).
   - During debugging, the lock was removed and `get_joint_positions` was changed to do its own `read_pos()` from the main thread — causing concurrent serial access from two threads.
   - **Fix applied:** `get_joint_positions` now returns cached data only (no serial I/O). Control thread uses `self.teleop.last_read_pos` cached during `update_full_joints_deg16`.
   - **But the hand still freezes**, so this may not be the full cause.

2. **Dynamixel operating mode not set (partially addressed):**
   - After force-kill (`Ctrl+C Ctrl+C`), motors retain torque-enabled state. Next startup writes operating mode (address 11, value 5) while torque is still on — this write is silently ignored by Dynamixel protocol.
   - **Fix applied:** Added `set_torque_enabled(False, retries=5)` before writing operating mode in `init_leap_hand()` and `reconnect()`.
   - **But the freeze happens DURING a session**, not just at startup, so this is likely not the primary cause.

3. **DynamixelClient read() returns stale cached data on failure:**
   - `DynamixelReader.read()` (dynamixel_client.py line 385-418): if `txRxPacket()` fails, it returns `self._get_data()` which is the previously cached values. Errors are logged at `logging.ERROR` level.
   - `collect_data.py` was suppressing all logs with `logging.getLogger("root").setLevel(logging.CRITICAL)`.
   - **Fix applied:** Changed to `logging.WARNING` level so serial errors are visible.

4. **Possible Dynamixel SDK bug or half-duplex timing issue:**
   - At 70 Hz, each `update_full_joints_deg16` call does `read_pos()` + `write_desired_pos()` = 2 serial transactions per cycle = 140/sec.
   - The Dynamixel SDK's `GroupSyncRead` uses `txRxPacket()` which is a half-duplex operation. At high rates on a single-wire RS-485 bus, timing margins shrink.
   - The `fastSyncRead()` method throws an exception (triggering "Update your Dynamixel_SDK from Github faster reads!" message), falling back to slower `txRxPacket()`.

5. **Something external locking the serial port:**
   - After force-kills, the serial port `/dev/ttyUSB*` may be held by zombie processes. This causes the next init to hang.
   - **Recovery:** `kill -9 $(pgrep -f collect_data)` then `sudo lsof /dev/ttyUSB*` to find and kill port holders.

### What the Original Working Code Looked Like

The original `_control_loop` that worked for 44 episodes:
```python
def _control_loop(self):
    while self._running:
        if self.teleop.sock is None:
            time.sleep(0.01)
            continue
        data = self.teleop.sock.ReadReceivedData()
        if data is not None:
            with self._lock:
                self.teleop.process_udp_data(data)
                if self.teleop.dxl_client is not None:
                    try:
                        raw = self.teleop.dxl_client.read_pos()
                        self._last_positions_allegro = lhu.LEAPhand_to_allegro(
                            raw, zeros=False).astype(np.float32)
                    except Exception:
                        pass
        time.sleep(0.001)

def get_joint_positions(self):
    with self._lock:
        return self._last_positions_allegro.copy()
```

Key difference from current code: The lock was present and `get_joint_positions` did NOT do serial I/O. The current code has removed the lock entirely and the `_control_loop` no longer does a separate `read_pos()` after the write (it uses `last_read_pos` cached inside `update_full_joints_deg16`).

### Recommended Next Steps for Debugging

1. **Add the lock back** around `process_udp_data` in `_control_loop` (matching the original working pattern).
2. Run with `logging.WARNING` to see if serial errors are being thrown.
3. Check the `[LEAP DBG]` output — new diagnostics print "CHANGED" vs "FROZEN" for `current_leap` and a `[VERIFY]` line after the first 5 writes showing actual motor movement in 20ms.
4. If the `read_pos()` inside `update_full_joints_deg16` (line 287) itself returns stale data, the problem is at the Dynamixel SDK level — possibly the `GroupSyncRead` object's internal buffer getting corrupted.
5. Consider reducing the control rate by adding `time.sleep(0.005)` after each write, or using `read_pos_vel_cur()` instead of separate reads.
6. As a nuclear option: revert `_control_loop` to the exact original code and only change `get_joint_positions` to use cached data without `read_pos()`.

---

## 9. Training Details

### Existing Trained Models

| Checkpoint | Dataset | Image Size | Epochs | Notes |
|-----------|---------|-----------|--------|-------|
| `checkpoints/quick_sanity/best.pt` | climbing_holds.zarr (30-dim) | 96×96 | 50 | Sanity check |
| `checkpoints/overnight_224/best.pt` | climbing_holds.zarr (30-dim) | 224×224 | 300 | Overnight run, used for real eval |

### Evaluation Results
The overnight model was evaluated on the real robot. It successfully rotated the hand toward the hold but did not grab precisely or perform the pulling motion. This led to the plan to add 3D hold pose to state, use RGBD, and collect more data — which is where the LEAP hand bug was encountered.

### Training Command (for reference)
```bash
# Quick sanity check
python3 train.py --zarr ../datasets/climbing_holds.zarr --ckpt-dir ../checkpoints/quick --quick

# Full overnight training
python3 train.py --zarr ../datasets/climbing_holds.zarr \
    --ckpt-dir ../checkpoints/overnight_224 \
    --epochs 300 --batch 64 --img-size 224 --diffusion-steps 100

# New v2 training (once data is collected)
python3 train.py --zarr ../datasets/climbing_holds_v2.zarr \
    --ckpt-dir ../checkpoints/v2 \
    --epochs 300 --batch 64 --img-size 224 --rgbd --augment
```

---

## 10. Known Issues & Quirks

1. **`server_env.py` is NOT used** — but many print statements and docstrings still reference it. These are misleading.
2. **Dead code in `leap_pip_dip_teleop.py`** — lines 358-407 are unreachable (after `return True` / `return False`). Harmless but confusing.
3. **`robomail.vision` warnings** on import (tensorflow, pylibfreenect2, openni2, phoxi) — these are harmless, the RealSense cameras work fine.
4. **Force-killing** `collect_data.py` can leave the serial port locked. Always try graceful shutdown (q key, or single Ctrl+C) first. If hung, use `kill -9` then `sudo lsof /dev/ttyUSB*`.
5. **`set_torque_enabled` retries** — the DynamixelClient's default is `retries=-1` (infinite). The init now uses `retries=5` to prevent hanging.
6. **Image cache** — training creates a `.npy` memmap in `datasets/img_cache/`. If the dataset changes, delete this directory to force a rebuild.
7. **Quest 2 IP changes** — the Quest's WiFi IP changes periodically. Update in `VR_Teleoperation_Minimum.py` lines 72/74.

---

## 11. User's Goals (Priority Order)

1. **Fix LEAP hand motor control** so teleoperation works reliably during data collection
2. **Collect 100+ episodes** of the new 36-dim + RGBD data on a single hold
3. **Retrain overnight** on the new dataset
4. **Evaluate** the new model on the real robot
5. Optionally: collect data on multiple holds, train multi-hold policy
