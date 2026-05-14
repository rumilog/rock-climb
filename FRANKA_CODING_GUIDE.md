# Franka Emika Panda — Coding & Operation Guide

This document onboards a new coding agent to write control code for the Franka Emika
Panda arm on this machine. The project uses **no LEAP hand** — the end effector is a
**GelSight Mini** tactile sensor. The overhead scene camera is an **Intel RealSense D555**.

---

## 1. Machine Setup — Do This Every Session
w
Two commands must be sourced **before running any Python** that touches the arm, cameras, or ROS:

```bash
source ~/franka/bin/activate
source ~/frankapy/catkin_ws/devel/setup.bash
```

ROS must already be running. If it is not, start it first in a dedicated terminal:

```bash
bash ~/frankapy/bash_scripts/start_control_pc.sh -u franka -i franka-Alienware-Area-51-R5 -g 0
```

Leave that terminal open. Everything else runs after sourcing the two lines above.

---

## 2. Connecting to the Arm

```python
from frankapy import FrankaArm

fa = FrankaArm(with_gripper=False)   # no gripper on this project
```

`FrankaArm()` is a blocking call that establishes the ROS/FCI connection. Only one
`FrankaArm` instance should exist per process. Creating a second one while the first
is live will corrupt the controller state — if you need to reset, call `fa.stop_skill()`
first, wait ~0.5 s, then create a new instance.

Fixed IP addresses on this machine:
- Franka control box: `172.26.67.113` — hardwired, never changes
- Everything else (cameras, GelSight, host PC) is on the same subnet

---

## 3. Reading Arm State

```python
# 7-element array, joint angles in radians, q1..q7
joints = fa.get_joints()   # returns list; wrap with np.array() if needed

# Current end-effector pose as an autolab_core RigidTransform
pose = fa.get_pose()
xyz  = pose.translation    # np.array([x, y, z]) in metres, world frame
quat = pose.quaternion     # np.array([w, x, y, z])

# Rotation matrix (3×3)
R = pose.rotation
```

The world frame origin is at the **robot base**. X points away from the robot (toward
the workspace), Y points to the robot's left, Z points up.

---

## 4. Blocking Joint-Space Move

```python
import numpy as np

target_joints = np.array([-0.11, -0.56, -0.07, -2.17, 0.04, 2.16, 0.54])

fa.goto_joints(
    target_joints.tolist(),   # must be a plain Python list
    duration=5.0,             # seconds to complete the move
    dynamic=False,            # True = streaming mode (see §6); False = blocking
    buffer_time=0.2,          # extra seconds FCI waits before timing out
)
```

`goto_joints` is **blocking** when `dynamic=False`. The call returns after the arm
reaches the target or the duration expires.

Safe conservative duration formula: allow ~1 s per 0.3 rad of maximum joint delta.

---

## 5. Blocking Cartesian-Space Move

```python
import copy
from autolab_core import RigidTransform

# Approach: read current pose, modify translation, send it
current_pose = fa.get_pose()
target_pose  = copy.deepcopy(current_pose)
target_pose.translation = np.array([0.55, 0.0, 0.12])  # x, y, z in metres

fa.goto_pose(
    target_pose,
    duration=5.0,
    dynamic=False,
    buffer_time=0.2,
)
```

To move relative to the current position:

```python
current_pose = fa.get_pose()
target_pose  = copy.deepcopy(current_pose)
target_pose.translation = current_pose.translation + np.array([0.05, 0.0, -0.03])
fa.goto_pose(target_pose, duration=3.0, dynamic=False)
```

To change orientation, modify `target_pose.rotation` (3×3 numpy array) or
`target_pose.quaternion` (w, x, y, z). For Z-down approach, set:

```python
import scipy.spatial.transform as st

# End-effector pointing straight down (Z-axis of EE frame aligned with world -Z)
R_down = np.array([[1,  0,  0],
                   [0, -1,  0],
                   [0,  0, -1]], dtype=float)
target_pose.rotation = R_down
```

---

## 6. Stopping / Aborting a Skill

```python
fa.stop_skill()
```

Call this whenever you need to interrupt a blocking move or before starting a new
skill type. Always wrap in try/except — it raises if no skill is active:

```python
try:
    fa.stop_skill()
except Exception:
    pass
import time; time.sleep(0.5)   # let FCI settle
```

---

## 7. Live Streaming Control (Continuous Trajectory)

For scanning tasks that need smooth continuous motion, use the **live controller** from
robomail instead of chaining many `goto_joints` calls.

### Joint-space streaming

```python
from robomail.motion import GotoJointsLive
import time, numpy as np

controller = GotoJointsLive(ignore_virtual_walls=True)

goal = np.array(fa.get_joints(), dtype=float)
controller.set_goal_joints(goal)
controller.start()
time.sleep(0.5)               # let the skill initialise

# Update the target at your control rate
for new_joints in trajectory:
    controller.set_goal_joints(new_joints)
    time.sleep(0.1)           # 10 Hz

controller.stop()
```

### Cartesian-space streaming

```python
from robomail.motion import GotoPoseLive
import time, numpy as np

controller = GotoPoseLive(step_size=0.05)  # max 5 cm per control step

controller.set_goal_translation(np.array([0.55, 0.0, 0.15]))
controller.start()
time.sleep(1.0)

for target_xyz in waypoints:
    controller.set_goal_translation(target_xyz)
    time.sleep(0.05)          # 20 Hz

controller.stop()
```

`GotoPoseLive` uses cartesian impedance internally. `step_size` caps how far the
commanded pose moves per control cycle (safety).

---

## 8. Low-Level Live Streaming via ROS (used in evaluate.py)

For the tightest control loop, publish joint targets directly to the FCI ROS topic.
This is what evaluate.py does at 10 Hz:

```python
import rospy
from frankapy import SensorDataMessageType, FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
msg_id = 0

# Start the dynamic skill (duration=1000 s keeps it alive)
cur = np.array(fa.get_joints()).tolist()
fa.goto_joints(cur, duration=1000, dynamic=True, buffer_time=10)
init_time = rospy.Time.now().to_time()

def send_joint_target(joints):
    global msg_id
    timestamp = rospy.Time.now().to_time() - init_time
    msg = JointPositionSensorMessage(id=msg_id, timestamp=timestamp,
                                     joints=np.array(joints).tolist())
    ros_msg = make_sensor_group_msg(
        trajectory_generator_sensor_msg=sensor_proto2ros_msg(
            msg, SensorDataMessageType.JOINT_POSITION))
    pub.publish(ros_msg)
    msg_id += 1

# Call send_joint_target(joints) at your loop rate, then:
fa.stop_skill()
```

Important: always clamp joint deltas to a safe max per step (e.g. 0.03 rad) to
prevent the arm from lurching if a stale target is sent.

---

## 9. Key Joint Configurations

```python
# Arm fully clear of all overhead cameras — use before any camera capture
PARK_JOINTS = np.array([-0.11426599, -0.56029082, -0.06635159,
                         -2.17443357,  0.04112932,  2.15592909,  0.54378958])

# Neutral upright home position
RESET_JOINTS = np.array([-0.1111, -0.1703, -0.0621, -2.3442,
                           0.0408,  2.1952,  0.1559])
```

To move to a named configuration:

```python
fa.goto_joints(PARK_JOINTS.tolist(), duration=7.0, dynamic=False, buffer_time=0.2)
```

---

## 10. Intel RealSense D555 — Single Camera

```python
import pyrealsense2 as rs

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

profile = pipeline.start(config)

frames      = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

color_image = np.asanyarray(color_frame.get_data())  # (720, 1280, 3) BGR uint8
depth_image = np.asanyarray(depth_frame.get_data())  # (720, 1280) uint16 mm

pipeline.stop()
```

Alternatively, use the robomail `ThreadedCameras` wrapper (already used in this repo)
if you need multi-camera streaming and prefer calibrated extrinsics from the robomail
database:

```python
import robomail.vision as vis

cameras = vis.ThreadedCameras(
    cam_numbers=[5],         # D555 assigned to camera slot 5 — verify with ls /dev/video*
    image_height=480, image_width=848,
    get_point_cloud=False, get_verts=False,
)
cameras.get_next_frames()

frames = cameras.get_frames()
color  = frames[0][0]   # (480, 848, 3) BGR uint8
depth  = frames[0][1]   # (480, 848) uint16 mm
```

---

## 11. GelSight Mini — Tactile Capture

The GelSight Mini appears as a standard USB camera (V4L2 device). Capture with OpenCV:

```python
import cv2

gel = cv2.VideoCapture(0)   # change device index if another camera is /dev/video0
gel.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
gel.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = gel.read()     # frame is (480, 640, 3) BGR uint8

gel.release()
```

Identify the correct device index before starting:

```bash
ls /dev/video*              # list all video devices
v4l2-ctl --list-devices     # human-readable names
```

Allow a few frames to flush after `VideoCapture()` before calling `read()` — the first
1–3 frames may be stale:

```python
for _ in range(5):
    gel.read()              # flush buffer
ret, gelsight_frame = gel.read()
```

---

## 12. Weld-Bead Scan — Application Overview

The arm scans a grid of metal billets that have had weld beads ground down. For each
billet position the sequence is:

1. **Move to approach waypoint** above the billet (Z clearance ~0.10 m above surface)
2. **Descend slowly** until the GelSight makes contact (or target Z reached)
3. **Wait / settle** (~0.5 s) then capture a GelSight frame
4. **Lift back** to approach waypoint
5. **Move horizontally** to the D555 overhead view position for that billet
6. **Capture D555 RGB + depth frame** (the camera sees the exact contact patch)
7. **Move to the next billet** (step to the right; when row ends, step to next row)
8. **Repeat** until all billets are scanned

Suggested data structure per billet:

```python
{
    "billet_id":      int,           # sequential index across the grid
    "row":            int,           # 0-indexed row
    "col":            int,           # 0-indexed column
    "nominal_xyz":    [x, y, z],     # expected contact position in world frame (m)
    "actual_joints":  [q1..q7],      # arm joints at moment of GelSight capture
    "actual_xyz":     [x, y, z],     # fa.get_pose().translation at capture
    "gelsight_frame": np.ndarray,    # (H, W, 3) uint8 BGR
    "d555_color":     np.ndarray,    # (H, W, 3) uint8 BGR
    "d555_depth":     np.ndarray,    # (H, W) uint16 mm
    "timestamp":      float,         # time.time() at GelSight capture
}
```

---

## 13. Safe Approach Pattern for Touch-Down

Never command the arm directly to a surface contact Z in one call — use a two-phase
approach to avoid impacting the billet:

```python
APPROACH_Z  = 0.10   # metres above table — safe transit height
CONTACT_Z   = 0.005  # metres — surface contact (tune per table height)
TOUCH_SPEED = 3.0    # seconds for the final descent (slow)

def approach_and_touch(fa, x, y):
    # Phase 1: move above target XY at safe Z
    pose = fa.get_pose()
    target = copy.deepcopy(pose)
    target.translation = np.array([x, y, APPROACH_Z])
    fa.goto_pose(target, duration=5.0, dynamic=False, buffer_time=0.2)

    # Phase 2: descend slowly to contact Z
    pose = fa.get_pose()
    target = copy.deepcopy(pose)
    target.translation = np.array([x, y, CONTACT_Z])
    fa.goto_pose(target, duration=TOUCH_SPEED, dynamic=False, buffer_time=0.2)

    time.sleep(0.5)   # settle

def lift_to_approach(fa, x, y):
    pose = fa.get_pose()
    target = copy.deepcopy(pose)
    target.translation = np.array([x, y, APPROACH_Z])
    fa.goto_pose(target, duration=2.0, dynamic=False, buffer_time=0.2)
```

---

## 14. Grid Traversal Pattern

```python
# Define billet grid in world frame
COL_PITCH = 0.05    # metres between billets along Y axis
ROW_PITCH = 0.05    # metres between rows along X axis (negative = toward robot)
N_COLS    = 6
N_ROWS    = 3

# Origin: position (X, Y) of billet (row=0, col=0)
GRID_ORIGIN_X = 0.55
GRID_ORIGIN_Y = 0.15

# D555 camera view offset from contact point (horizontal)
CAM_OFFSET_X = 0.0
CAM_OFFSET_Y = 0.0
CAM_Z        = 0.25  # height for overhead view (must clear GelSight cable)

billet_id = 0
for row in range(N_ROWS):
    for col in range(N_COLS):
        bx = GRID_ORIGIN_X - row * ROW_PITCH
        by = GRID_ORIGIN_Y - col * COL_PITCH

        # 1–4: GelSight touch
        approach_and_touch(fa, bx, by)
        gel_frame = capture_gelsight()
        lift_to_approach(fa, bx, by)

        # 5–6: D555 overhead view
        pose = fa.get_pose()
        cam_target = copy.deepcopy(pose)
        cam_target.translation = np.array([bx + CAM_OFFSET_X,
                                           by + CAM_OFFSET_Y,
                                           CAM_Z])
        fa.goto_pose(cam_target, duration=3.0, dynamic=False, buffer_time=0.2)
        color, depth = capture_d555()

        # 7: save
        save_billet_data(billet_id, row, col, [bx, by, CONTACT_Z],
                         fa.get_joints(), fa.get_pose().translation,
                         gel_frame, color, depth)
        billet_id += 1
```

---

## 15. Error Handling & Safety Rules

1. **Always `stop_skill()` before any new movement call** — especially when switching
   between `goto_joints`, `goto_pose`, and live-streaming modes.

2. **Always lift to APPROACH_Z before moving horizontally** — never translate XY while
   at contact Z; the GelSight or billet will collide.

3. **Never create two `FrankaArm()` instances** — it silently corrupts the FCI.
   Reuse the one `fa` object for the whole script lifetime.

4. **Wrap every `goto_pose` / `goto_joints` in try/except** in long scanning loops —
   a timeout or soft E-stop should abort the current billet and move to the next, not
   crash the whole script.

5. **`process exit can segfault`** — pyrealsense2 and frankapy C-extensions occasionally
   segfault on teardown. Save all data before closing hardware. Use `os._exit(0)` after
   cleanup if clean exit is needed.

6. **Stiffness limit** — the Panda's default cartesian stiffness is high. For contact
   tasks (pressing GelSight against a surface), optionally lower it:
   ```python
   fa.goto_pose(contact_target, duration=3.0, dynamic=False,
                cartesian_impedances=[600, 600, 100, 50, 50, 50])
   # [Tx, Ty, Tz, Rx, Ry, Rz] — lower Tz = compliant in Z
   ```

---

## 16. Complete Minimal Example

```python
#!/usr/bin/env python3
"""
Minimal scan: one billet, GelSight touch + D555 capture.
Run: source ~/franka/bin/activate && source ~/frankapy/catkin_ws/devel/setup.bash
     python3 scan_one_billet.py
"""
import copy, time
import numpy as np
import cv2
from frankapy import FrankaArm

BILLET_X    = 0.55
BILLET_Y    = 0.00
CONTACT_Z   = 0.005
APPROACH_Z  = 0.10
GEL_DEV     = 0       # /dev/video0 — adjust if needed

fa = FrankaArm(with_gripper=False)

# --- home ---
fa.reset_joints()
time.sleep(1.0)

# --- open gelsight ---
gel = cv2.VideoCapture(GEL_DEV)
for _ in range(5): gel.read()   # flush

# --- approach ---
pose   = fa.get_pose()
target = copy.deepcopy(pose)
target.translation = np.array([BILLET_X, BILLET_Y, APPROACH_Z])
fa.goto_pose(target, duration=5.0, dynamic=False, buffer_time=0.2)

# --- descend ---
target.translation = np.array([BILLET_X, BILLET_Y, CONTACT_Z])
fa.goto_pose(target, duration=3.0, dynamic=False, buffer_time=0.2)
time.sleep(0.5)

# --- capture gelsight ---
ret, gel_frame = gel.read()
cv2.imwrite("gelsight.png", gel_frame)

# --- lift ---
target.translation = np.array([BILLET_X, BILLET_Y, APPROACH_Z])
fa.goto_pose(target, duration=2.0, dynamic=False, buffer_time=0.2)

# --- D555 overhead ---
import pyrealsense2 as rs
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipe.start(cfg)
for _ in range(5): pipe.wait_for_frames()   # flush
frames = pipe.wait_for_frames()
color = np.asanyarray(frames.get_color_frame().get_data())
depth = np.asanyarray(frames.get_depth_frame().get_data())
cv2.imwrite("d555_color.png", color)
pipe.stop()

gel.release()
print("Done. GelSight and D555 frames saved.")
```
