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

[2026-03-20] | CoRL paper novelty: Dexonomy (RSS 2025, arXiv:2504.18829) also conditions on GRASP taxonomy from point clouds with a dexterous hand — reviewers will immediately compare | The differentiator is execution policy vs pose generation. Dexonomy outputs static poses handed to a motion planner; our system learns continuous visuomotor trajectories via imitation learning. Have a crisp 2-sentence version of this ready. See RESEARCH_PLAN.md for full novelty gap table.

[2026-03-20] | CoRL paper: the with/without grasp type conditioning ablation is load-bearing — if conditioning doesn't help, the main technical claim collapses | Run this ablation before assuming the paper's contribution holds. `--no-grasp-conditioning` flag does not yet exist in train.py and must be added. If conditioning doesn't help, reframe as benchmark paper (climbing holds benchmark is novel regardless) or find where it does help (e.g. cross-category generalization on held-out test_edge).

[2026-03-25] | Arrow key handling added to leap_pip_dip_teleop.py keyboard thread did nothing during data collection — collect_data.py owns stdin via tty.setcbreak, so the second thread never received input | Keyboard input during data collection must go through collect_data.py's stdin reader and _handle_key. Arrow keys are escape sequences (\x1b[A/B) — use select() with 0.05s timeout after detecting \x1b to read the remaining 2 bytes as one atomic sequence. cv2.waitKeyEx() (not waitKey() & 0xFF) must be used to pass full arrow key codes through the cv2 path (Linux: up=65362, down=65364, left=65361, right=65363).

[2026-03-25] | Thumb MCP_Flex keyboard offset had no effect on the physical motor despite the offset variable updating — root cause: mcp_flex_post_scale_offset_rad_per_finger[3]=-1.3 pushes allegro[13] to -1.3 rad, which converts to 1.84 rad in LEAP coords, below the hardware clip minimum of 2.67 (sim_min=-0.47 + π). Every small offset was still within the clipped zone. | When adding a keyboard offset to a joint that has a large post-scale offset, check whether the base value already violates angle_safety_clip limits. Fix by initializing thumb_offsets[1]=+0.9 to start just above the clip boundary. Need ~0.83 rad of offset to clear the clip from -1.3 baseline.

[2026-03-26] | Unity HandController.cs already sends 28 values (16 joint angles + 12 thumb quaternions for base/mid/tip bones) but Python was silently stripping the last 12 with `gripper_values[:16]` — the raw thumb bone orientation data was never used | Before adding new retargeting logic, always audit what data the upstream sender is already providing. Check process_udp_data for any truncation. The 12 thumb quaternion values (bas/mid/tip, xyzw each) are available at UDP payload indices 16-27.

[2026-03-26] | BeaVR's thumb IK pipeline uses OVR XRHand 3D joint positions directly (26 joints × xyz), while our Unity app uses the older OVRPlugin.HandState.BoneRotations API (local quaternions). The two APIs give different data formats — positions vs local rotations — so BeaVR's FK assumptions don't transfer directly. | When adapting IK code from a reference that uses a different hand-tracking API, document the FK reconstruction needed: OVRPlugin BoneRotations[3,4,5] are local quaternions that must be composed (cumulative product) and walked along THUMB_BONE_AXIS to get 3D positions. Key tunable constants: THUMB_BASE_POS, THUMB_BONE_AXIS, THUMB_BONE_LENS in leap_thumb_ik_test.py.

[2026-03-26] | beavr-bot-reference was cloned as a reference repo into the project root — without gitignore it would have been tracked and pushed | Reference/comparison repos must always be added to .gitignore immediately after cloning. Added `beavr-bot-reference/` to .gitignore in the same session.

[2026-04-01] | Gave user `--hold 0` for pinch data collection — this is WRONG. `--hold` is the physical hold identity, not a session counter. `--hold 0` = edge_A. For pinch holds use `--hold 3`. | Always look up HOLD_NAMES in collect_data.py before giving a `--hold` value. The mapping: 0=edge_A, 1=edge_B, 2=sloper, 3=pinch, 4=test_edge. Match `--hold` to the actual physical hold being used.
