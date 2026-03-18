# Lessons Learned

<!-- Format: [date] | what went wrong | rule to prevent it -->

[2026-03-16] | State dim was 30 (arm7 + ee_pos3 + ee_quat4 + hand16) but should be 23 (arm7 + hand16) | ee_pos/ee_quat are redundant with arm joints via FK — always check if components are truly independent before including in state

[2026-03-16] | robomail.vision.ThreadedCameras stores per-camera objects as self.cameras.cameras (list) | Always inspect the source of library classes before writing wrapper code — check inspect.getsource(cls)

[2026-03-16] | torch is not in PATH by default on robot machine | Always source ~/franka/bin/activate before running Python on robot machine

[2026-03-16] | point cloud mapping to episodes: np.searchsorted(ep_ends, t, side='right') gives episode index for timestep t, but must .clip(max=len(ep_ends)-1) to avoid out-of-bounds | When mapping timesteps to episodes, always add clip guard against the last ep_ends boundary

[2026-03-16] | robomail CameraClass.get_cam_intrinsics() reads from stale calibration YAML files (wrong resolution, fx/fy off by 2-3x, duplicate values across cameras) | Always read intrinsics live from pyrealsense2 pipeline: profile.get_stream(rs2.stream.depth).as_video_stream_profile().get_intrinsics()

[2026-03-16] | fuse_multi_camera_points was called on 1.48M uncropped points, making FPS hang for minutes | Always pre-random-sample to ~20k before FPS when N >> n_points; quality loss is negligible at 1024 output

[2026-03-16] | DEFAULT_WORKSPACE_BOUNDS z_max=0.40 was a placeholder. Measured calibration: table Z in [-0.02, 0.006], rock hold Z up to 0.073m, scale ~1:1 (cm→m world-Z). Set z_max=0.30 to cover a 25cm jug with margin | Before first data collection, always run check_workspace.py with empty table then with object to verify bounds

[2026-03-16] | Suggested running leap_pip_dip_teleop.py as a third terminal during data collection — this is WRONG | Data collection is always exactly two terminals: VR_Teleoperation_Minimum.py (Franka + finger forwarding) and collect_data.py (LEAP motor control + recording). collect_data.py wraps LeapPipDipTeleop internally. leap_pip_dip_teleop.py is only for teleop WITHOUT recording. See CLAUDE.md LEARNED section for full details.

[2026-03-18] | DEFAULT_WORKSPACE_BOUNDS z_min=-0.02 caused 95% of 1024 FPS points to land on the flat table surface (Z ∈ [-0.02, 0.006]), leaving only ~50 pts on the hold — robot did the same thing regardless of hold position | Set z_min=0.006 (at the table surface). This captures the full hold geometry including the base without including the flat table. z_min=0.008 works but clips the very base of the hold; 0.006 is the empirically verified correct value. Table tops out at Z=0.006. Training data collected with old z_min is unrecoverable (no raw depth stored) and must be recollected.

[2026-03-18] | Pilot model (50 jug eps, z_min=-0.02) ignored the point cloud entirely — same arm motion regardless of hold position or zero-PC input. Root cause: 95% table noise meant the network learned to ignore PC. Model was trained on bad data and the checkpoint (checkpoints/pc_large/best.pt) is invalid. | When retraining: verify PC quality with check_pc_sensitivity.py BEFORE collecting any episodes. Centroids must shift by centimeters across positions; Z centroid must be > 0.01 m.

[2026-03-18] | Tried to create a second FrankaArm in collect_data.py (Terminal 2) and call stop_skill() to park the arm during PC capture — this permanently kills GotoPoseLive's live skill in Terminal 1 because self.initialize stays False after an external skill kill, with no recovery path | NEVER create a second FrankaArm or call stop_skill() from outside the process that owns the live skill. Use file-based IPC instead: GotoPoseLive checks for /tmp/franka_park_request and handles parking itself, then re-initializes its own skill.
