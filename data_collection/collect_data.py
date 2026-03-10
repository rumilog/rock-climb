#!/usr/bin/env python3
"""
Diffusion Policy Data Collection — Climbing Hold Grasps

Records teleoperated demonstrations of a LEAP Hand (on a Franka Emika arm)
grasping climbing-style holds.  Each episode captures:
  - Franka arm joints (7) + EE pose (3 pos + 4 quat) + LEAP hand joints (16)
    = 30-dim state vector, at 10 Hz
  - 4 camera views (480x640 RGB), at 10 Hz
  - Per-episode metadata: hold_id, quality (good/bad), grasp_type

Architecture (two terminals):
  Terminal 1:  VR_Teleoperation_Minimum.py   (controls Franka, forwards finger data)
  Terminal 2:  python3 collect_data.py --hold 0   (this script: LEAP hand + cameras + recording)

Do NOT run leap_pip_dip_teleop.py — this script takes over LEAP hand control.

Keyboard (click the preview window):
    SPACE  = start / stop recording
    g      = save episode as GOOD
    b      = save episode as BAD
    d      = discard episode
    i      = dataset info
    q      = quit
"""

import os
import sys
import time
import threading
import signal
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TELE_ROOT = os.path.dirname(SCRIPT_DIR)

FRANKA_SCRIPTS_DIR = os.path.join(
    TELE_ROOT, "TeleoperationUnity", "Robot Control - Python", "Franka Scripts")
LEAP_DIR = os.path.join(
    TELE_ROOT, "TeleoperationUnity", "LEAP", "leaphandv1", "for_transfer")
LEAP_API_DIR = os.path.join(LEAP_DIR, "LEAP_Hand_API", "python")

for p in [FRANKA_SCRIPTS_DIR, LEAP_DIR, LEAP_API_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Always-available imports only; hardware deferred to setup methods.
# ---------------------------------------------------------------------------
import robomail.vision as vis
from episode_storage import ZarrDatasetWriter, EpisodeBuffer, resize_image

# ===========================================================================
# Configuration
# ===========================================================================

CAMERA_NUMBERS = [2, 3, 4, 5]
CAMERA_RAW_W = 848
CAMERA_RAW_H = 480
IMG_SAVE_H = 480
IMG_SAVE_W = 640

CONTROL_FREQ = 10
LEAP_PORT = None  # Auto-detect from /dev/ttyUSB*

# state = arm_joints(7) + ee_pos(3) + ee_quat(4) + hand_joints(16) = 30
STATE_DIM = 30
ACTION_DIM = 30

HOLD_NAMES = {
    0: "edge_A",
    1: "edge_B",
    2: "sloper",
    3: "pinch",
    4: "test_edge",   # held-out for evaluation
}

DEFAULT_DATASET_DIR = os.path.join(TELE_ROOT, "datasets")
DEFAULT_TASK_NAME = "climbing_holds"


# ===========================================================================
# LEAP Hand Controller + Recorder
# ===========================================================================

class LeapHandRecorder:
    def __init__(self, port=LEAP_PORT):
        from leap_pip_dip_teleop import LeapPipDipTeleop
        import leap_hand_utils.leap_hand_utils as lhu
        import logging
        logging.getLogger("root").setLevel(logging.WARNING)
        self._lhu = lhu
        self.teleop = LeapPipDipTeleop(port=port)
        self._lock = threading.Lock()
        self._last_positions_allegro = np.zeros(16, dtype=np.float32)
        self._running = False
        self._thread = None
        if self.teleop.dxl_client is not None:
            raw = self.teleop.dxl_client.read_pos()
            self._last_positions_allegro = self._lhu.LEAPhand_to_allegro(
                raw, zeros=False).astype(np.float32)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        print("LEAP hand control thread started (receiving on localhost:8002)")

    def _control_loop(self):
        msg_count = 0
        write_ok = 0
        write_fail = 0
        last_report = time.time()
        while self._running:
            if self.teleop.sock is None:
                time.sleep(0.01)
                continue
            data = self.teleop.sock.ReadReceivedData()
            if data is not None:
                msg_count += 1
                success = self.teleop.process_udp_data(data)
                if success:
                    write_ok += 1
                else:
                    write_fail += 1
                # update_full_joints_deg16 already did a read_pos() and cached
                # it in self.teleop.last_read_pos — reuse it (no extra serial I/O)
                if hasattr(self.teleop, 'last_read_pos') and self.teleop.last_read_pos is not None:
                    self._last_positions_allegro = self._lhu.LEAPhand_to_allegro(
                        self.teleop.last_read_pos, zeros=False).astype(np.float32)
            now = time.time()
            if now - last_report >= 5.0:
                if msg_count == 0:
                    print(f"  [LEAP UDP] No finger data received in last 5s — "
                          f"is VR_Teleoperation_Minimum.py running?")
                else:
                    print(f"  [LEAP UDP] {msg_count} msgs, "
                          f"{write_ok} writes OK, {write_fail} failed "
                          f"({msg_count/5.0:.1f} Hz)")
                msg_count = 0
                write_ok = 0
                write_fail = 0
                last_report = now
            time.sleep(0.001)

    def get_joint_positions(self):
        """Return cached positions — all serial I/O happens in _control_loop."""
        return self._last_positions_allegro.copy()

    @property
    def connected(self):
        return self.teleop.dxl_client is not None

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.teleop.dxl_client is not None:
            try:
                self.teleop.cleanup()
            except Exception:
                pass


# ===========================================================================
# Helpers
# ===========================================================================

def build_state_vector(arm_joints, ee_pos, ee_quat, hand_joints):
    """arm(7) + ee_pos(3) + ee_quat(4) + hand(16) = 30"""
    parts = [
        np.array(arm_joints, dtype=np.float32).ravel()[:7],
        np.array(ee_pos, dtype=np.float32).ravel()[:3],
        np.array(ee_quat, dtype=np.float32).ravel()[:4],
        np.array(hand_joints, dtype=np.float32).ravel()[:16],
    ]
    return np.concatenate(parts)


# ===========================================================================
# Main Data Collector
# ===========================================================================

class DataCollector:
    def __init__(self, hold_id=0, task_name=DEFAULT_TASK_NAME,
                 dataset_dir=DEFAULT_DATASET_DIR, cam_numbers=None,
                 skip_franka=False, skip_leap=False,
                 control_freq=CONTROL_FREQ, img_h=IMG_SAVE_H, img_w=IMG_SAVE_W,
                 show_preview=True, store_depth=False):
        self.hold_id = hold_id
        self.task_name = task_name
        self.dataset_dir = dataset_dir
        self.skip_franka = skip_franka
        self.skip_leap = skip_leap
        self.control_freq = control_freq
        self.img_h = img_h
        self.img_w = img_w
        self.show_preview = show_preview
        self.store_depth = store_depth
        self.running = False
        self.recording = False

        self.cam_numbers = cam_numbers or CAMERA_NUMBERS
        self.cam_names = [f"cam{n}" for n in self.cam_numbers]

        self.fa = None
        self.leap_recorder = None
        self.cameras = None
        self.dataset = None
        self.episode_buf = None

    # -------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------
    def setup(self):
        hold_name = HOLD_NAMES.get(self.hold_id, f"hold_{self.hold_id}")
        print("=" * 60)
        print(f"  DATA COLLECTION — Hold {self.hold_id}: {hold_name}")
        print("=" * 60)

        self._setup_franka()
        self._setup_leap_hand()
        self._setup_cameras()
        self._setup_dataset()

        print(f"\n{'=' * 60}")
        print(f"  READY — collecting for hold {self.hold_id} ({hold_name})")
        print(f"{'=' * 60}")
        print("  SPACE  = start/stop recording")
        print("  g      = save as GOOD grasp")
        print("  b      = save as BAD grasp")
        print("  d      = discard episode")
        print("  i      = dataset info")
        print("  q      = quit")
        print(f"{'=' * 60}\n")

    def _setup_franka(self):
        if self.skip_franka:
            print("[1/4] Franka arm SKIPPED (--no-franka)")
            return
        print("[1/4] Subscribing to Franka arm state (read-only via ROS)...")
        try:
            import rospy
            from franka_interface_msgs.msg import RobotState
            if not rospy.core.is_initialized():
                rospy.init_node("data_collector", anonymous=True, disable_signals=True)
            self._arm_state_lock = threading.Lock()
            self._latest_robot_state = None

            def _robot_state_cb(msg):
                with self._arm_state_lock:
                    self._latest_robot_state = msg

            self._arm_sub = rospy.Subscriber(
                "/robot_state_publisher_node_1/robot_state",
                RobotState, _robot_state_cb, queue_size=1)
            self.fa = "ros_topic"

            # Wait briefly for first message
            import time
            for _ in range(20):
                with self._arm_state_lock:
                    if self._latest_robot_state is not None:
                        break
                time.sleep(0.1)

            with self._arm_state_lock:
                msg = self._latest_robot_state
            if msg is not None:
                q = np.array(msg.q, dtype=np.float32)
                ee = np.array(msg.O_T_EE, dtype=np.float64).reshape(4, 4, order='F')
                print(f"  OK — joints: ({len(q)},), EE: "
                      f"[{ee[0,3]:.3f}, {ee[1,3]:.3f}, {ee[2,3]:.3f}]")
            else:
                print("  WARNING: No arm state received yet (is VR_Teleoperation running?)")
        except Exception as e:
            print(f"  WARNING: Franka ROS setup failed: {e}")
            self.fa = None

    def _setup_leap_hand(self):
        if self.skip_leap:
            print("[2/4] LEAP hand SKIPPED (--no-leap)")
            return
        print("[2/4] Initializing LEAP hand control + recording...")
        try:
            self.leap_recorder = LeapHandRecorder(port=LEAP_PORT)
            if self.leap_recorder.connected:
                pos = self.leap_recorder.get_joint_positions()
                print(f"  OK — 16 motors, sample pos[0]: {pos[0]:.3f} rad")
                self.leap_recorder.start()
            else:
                print("  WARNING: LEAP hand connection failed.")
                self.leap_recorder = None
        except Exception as e:
            print(f"  WARNING: LEAP hand setup failed: {e}")
            self.leap_recorder = None

    def _setup_cameras(self):
        print(f"[3/4] Starting cameras {self.cam_numbers}...")
        self.cameras = vis.ThreadedCameras(
            cam_numbers=self.cam_numbers,
            image_height=CAMERA_RAW_H, image_width=CAMERA_RAW_W,
            get_point_cloud=False, get_verts=False,
        )
        frames = self.cameras.get_next_frames()
        for i, (color, depth, _, _) in enumerate(frames):
            has_depth = depth is not None and depth.size > 0
            print(f"  Camera {self.cam_numbers[i]}: {color.shape} OK"
                  f" (depth: {'yes' if has_depth else 'no'})")

    def _setup_dataset(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        zarr_path = os.path.join(self.dataset_dir, f"{self.task_name}.zarr")
        print(f"[4/4] Dataset: {zarr_path}")
        self.dataset = ZarrDatasetWriter(
            zarr_path=zarr_path, cam_names=self.cam_names,
            img_height=self.img_h, img_width=self.img_w,
            state_dim=STATE_DIM, action_dim=ACTION_DIM,
            store_depth=self.store_depth,
        )
        info = self.dataset.get_summary()
        if info["total_timesteps"] > 0:
            print(f"  Resuming: {info['num_episodes']} episodes, "
                  f"{info['total_timesteps']} timesteps")
        else:
            print("  New dataset created.")

    # -------------------------------------------------------------------
    # State reading
    # -------------------------------------------------------------------
    def _read_arm_state(self):
        if self.fa is not None and hasattr(self, '_arm_state_lock'):
            with self._arm_state_lock:
                msg = self._latest_robot_state
            if msg is not None:
                try:
                    joints = np.array(msg.q, dtype=np.float32)
                    # O_T_EE is column-major 4x4 homogeneous transform
                    T = np.array(msg.O_T_EE, dtype=np.float64).reshape(4, 4, order='F')
                    ee_pos = T[:3, 3].astype(np.float32)
                    # Extract quaternion from rotation matrix
                    from autolab_core import RigidTransform
                    rt = RigidTransform(rotation=T[:3, :3], translation=T[:3, 3])
                    ee_quat = rt.quaternion.astype(np.float32)
                    return joints, ee_pos, ee_quat
                except Exception:
                    pass
        return (np.zeros(7, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.array([1, 0, 0, 0], dtype=np.float32))

    def _read_hand_state(self):
        if self.leap_recorder is not None:
            return self.leap_recorder.get_joint_positions()
        return np.zeros(16, dtype=np.float32)

    def _read_camera_frames(self):
        raw_frames = self.cameras.get_frames()
        images = {}
        depths = {}
        for i, cam_name in enumerate(self.cam_names):
            color = raw_frames[i][0]
            images[cam_name] = resize_image(color, self.img_h, self.img_w)
            if self.store_depth:
                depth = raw_frames[i][1]
                if depth is not None and depth.size > 0:
                    depths[cam_name] = cv2.resize(
                        depth, (self.img_w, self.img_h),
                        interpolation=cv2.INTER_NEAREST).astype(np.uint16)
                else:
                    depths[cam_name] = np.zeros(
                        (self.img_h, self.img_w), dtype=np.uint16)
        return images, depths, raw_frames

    # -------------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------------
    def _show_preview_window(self, raw_frames):
        previews = []
        for i, frame_tuple in enumerate(raw_frames):
            color = frame_tuple[0]
            small = cv2.resize(color, (320, 180))
            cv2.putText(small, f"Cam {self.cam_numbers[i]}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            previews.append(small)

        if not previews:
            canvas = np.zeros((180, 320, 3), dtype=np.uint8)
        elif len(previews) <= 2:
            canvas = np.hstack(previews)
        else:
            ncols = 2
            while len(previews) % ncols != 0:
                previews.append(np.zeros_like(previews[0]))
            rows = [np.hstack(previews[i:i + ncols])
                    for i in range(0, len(previews), ncols)]
            canvas = np.vstack(rows)

        # Status bar
        hold_name = HOLD_NAMES.get(self.hold_id, str(self.hold_id))
        ep_len = len(self.episode_buf) if self.episode_buf else 0

        if self.recording:
            status = f"REC  hold={hold_name}  steps={ep_len}"
            color = (0, 0, 255)
        else:
            status = f"IDLE  hold={hold_name}"
            color = (0, 200, 0)
            if self.episode_buf is not None and ep_len > 0:
                status += f"  [{ep_len} steps — press g/b/d]"

        cv2.putText(canvas, status, (5, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        ep_count = self.dataset.num_episodes
        cv2.putText(canvas, f"Saved: {ep_count}",
                    (canvas.shape[1] - 120, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("Data Collection", canvas)

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------
    def run(self):
        self.running = True
        self.recording = False
        self.episode_buf = None
        dt = 1.0 / self.control_freq

        signal.signal(signal.SIGINT, self._signal_handler)

        if self.show_preview:
            cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
            nrows = (len(self.cam_numbers) + 1) // 2
            cv2.resizeWindow("Data Collection", 640, 360 * max(1, nrows))
        else:
            self._start_stdin_listener()

        print("Waiting for first camera frame...")
        self.cameras.get_next_frames()
        print("Cameras streaming. Ready!\n")

        while self.running:
            loop_start = time.time()

            arm_joints, ee_pos, ee_quat = self._read_arm_state()
            hand_joints = self._read_hand_state()
            images, depths, raw_frames = self._read_camera_frames()
            now = time.time()

            current_state = build_state_vector(arm_joints, ee_pos, ee_quat, hand_joints)

            if self.recording and self.episode_buf is not None:
                self.episode_buf.add_timestep(
                    state=current_state,
                    action=current_state.copy(),  # placeholder; finalize_actions() shifts these
                    images_dict=images,
                    timestamp=now,
                    depths_dict=depths if self.store_depth else None,
                )

            if self.show_preview:
                self._show_preview_window(raw_frames)
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)
            else:
                self._poll_stdin()

            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._cleanup()

    # -------------------------------------------------------------------
    # Stdin fallback (headless mode)
    # -------------------------------------------------------------------
    def _start_stdin_listener(self):
        self._stdin_key = None
        self._stdin_lock = threading.Lock()

        def _reader():
            try:
                import tty, termios
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
            except (ImportError, Exception):
                print("  (No interactive terminal — use Ctrl+C)")
                return
            try:
                tty.setcbreak(fd)
                while self.running:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    with self._stdin_lock:
                        self._stdin_key = ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        threading.Thread(target=_reader, daemon=True).start()

    def _poll_stdin(self):
        key = None
        with self._stdin_lock:
            if self._stdin_key is not None:
                key = self._stdin_key
                self._stdin_key = None
        if key is not None:
            self._handle_key(ord(key) if len(key) == 1 else -1)

    # -------------------------------------------------------------------
    # Keyboard handling
    # -------------------------------------------------------------------
    def _handle_key(self, key):
        if key == ord(" "):
            if not self.recording:
                self._start_episode()
            else:
                self._stop_episode()
        elif key == ord("g"):
            self._save_episode(quality=1)
        elif key == ord("b"):
            self._save_episode(quality=0)
        elif key == ord("d"):
            self._discard_episode()
        elif key == ord("i"):
            self._print_info()
        elif key == ord("q"):
            print("\nQuitting...")
            if self.recording:
                self._stop_episode()
            if self.episode_buf is not None and len(self.episode_buf) > 0:
                self._save_episode(quality=1)
            self.running = False

    def _start_episode(self):
        self.episode_buf = EpisodeBuffer(self.cam_names, store_depth=self.store_depth)
        self.recording = True
        hold_name = HOLD_NAMES.get(self.hold_id, str(self.hold_id))
        print(f">>> REC (ep {self.dataset.num_episodes + 1}, hold={hold_name})")

    def _stop_episode(self):
        self.recording = False
        if self.episode_buf is not None:
            secs = len(self.episode_buf) / max(self.control_freq, 1)
            print(f"<<< STOP — {len(self.episode_buf)} steps ({secs:.1f}s)")
            print("    g = save GOOD | b = save BAD | d = discard")

    def _save_episode(self, quality=1):
        if self.episode_buf is None or len(self.episode_buf) == 0:
            print("Nothing to save.")
            return
        self.dataset.append_episode(
            self.episode_buf,
            hold_id=self.hold_id,
            quality=quality,
        )
        self.episode_buf = None

    def _discard_episode(self):
        if self.episode_buf is None or len(self.episode_buf) == 0:
            print("Nothing to discard.")
            return
        n = len(self.episode_buf)
        self.episode_buf = None
        self.recording = False
        print(f"Discarded ({n} steps)")

    def _print_info(self):
        info = self.dataset.get_summary()
        print("\n--- Dataset Info ---")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()

    def _signal_handler(self, sig, frame):
        self._interrupt_count = getattr(self, '_interrupt_count', 0) + 1
        if self._interrupt_count >= 2:
            print("\nForce quit!")
            os._exit(1)
        print("\nInterrupt — shutting down... (Ctrl+C again to force)")
        if self.recording:
            self._stop_episode()
        self.running = False

    def _cleanup(self):
        print("Cleaning up...")
        try:
            self.cameras.record = False
            if self.cameras._thread.is_alive():
                self.cameras._thread.join(timeout=3.0)
        except Exception:
            pass
        if self.leap_recorder is not None:
            try:
                self.leap_recorder.stop()
            except Exception:
                pass
        if self.show_preview:
            cv2.destroyAllWindows()
        self._print_info()
        print("Done.")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Data collection for climbing hold grasps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Hold IDs:
{chr(10).join(f'  {k}: {v}' for k, v in HOLD_NAMES.items())}

Example:
  python3 collect_data.py --hold 0                  # edge_A
  python3 collect_data.py --hold 3 --task pinch_v2  # pinch hold, custom task name
  python3 collect_data.py --no-franka --no-leap     # cameras-only dry run
""")
    parser.add_argument("--hold", type=int, default=0,
                        help="Hold ID (default: 0)")
    parser.add_argument("--task", default=DEFAULT_TASK_NAME,
                        help="Task/dataset name (default: %(default)s)")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR,
                        help="Dataset directory (default: %(default)s)")
    parser.add_argument("--cameras", nargs="+", type=int, default=CAMERA_NUMBERS,
                        help="Camera numbers (default: %(default)s)")
    parser.add_argument("--freq", type=int, default=CONTROL_FREQ,
                        help="Recording Hz (default: %(default)s)")
    parser.add_argument("--img-h", type=int, default=IMG_SAVE_H,
                        help="Image height (default: %(default)s)")
    parser.add_argument("--img-w", type=int, default=IMG_SAVE_W,
                        help="Image width (default: %(default)s)")
    parser.add_argument("--no-franka", action="store_true")
    parser.add_argument("--no-leap", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--rgbd", action="store_true",
                        help="Also store depth images alongside RGB")
    args = parser.parse_args()

    collector = DataCollector(
        hold_id=args.hold,
        task_name=args.task,
        dataset_dir=args.dataset_dir,
        cam_numbers=args.cameras,
        skip_franka=args.no_franka,
        skip_leap=args.no_leap,
        control_freq=args.freq,
        img_h=args.img_h,
        img_w=args.img_w,
        show_preview=not args.no_preview,
        store_depth=args.rgbd,
    )
    collector.setup()
    collector.run()


if __name__ == "__main__":
    main()
