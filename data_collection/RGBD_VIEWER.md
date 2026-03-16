## Live RGBD Viewer for RealSense Cameras

### Purpose

This document describes the `view_rgbd_live.py` script and the **RGB + depth (RGBD)** observations it provides from the Intel RealSense D415/D455 cameras in this project. It is intended for downstream agents that understand the data collection and training pipeline, so they can reason about how to use RGBD information to improve grasp localization (e.g., when the robot hand goes to the correct hold type but the wrong physical location).

The viewer is **read-only**: it does not write datasets, but it exposes exactly what the RealSense sensors can provide at runtime.

---

### How the viewer works

- **Script**: `view_rgbd_live.py` (repo root).
- **Cameras**: Intel RealSense D415/D455, accessed via `robomail.vision.ThreadedCameras`.
- For each configured camera ID in `CAMERA_NUMBERS = [2, 3, 4, 5]`, the script continuously reads:
  - An RGB frame: `(H, W, 3)` `uint8`.
  - A depth frame: `(H, W)` `uint16` in **millimeters** (dense per-pixel depth map).
- For each camera, the script constructs a **2×1 tile**:
  - **Top half (RGB)**: 320×180 RGB image, with a label like `Cam 2`.
  - **Bottom half (Depth)**: 320×180 depth colormap, with a label like `Depth 2`.
    - Depth values are converted from mm → meters via a fixed scale of 0.001.
    - Values are normalized to \([0, 1]\) over a configurable max range (default 1 m).
    - A JET colormap is applied (blue = far, red = near).
- All per-camera tiles are arranged into a **grid** (2 columns by default) and shown in a single OpenCV window:
  - Window title: `RGB + Depth (all cameras)`.
  - Status text at the bottom: `Live RGB+Depth per camera  |  Press 'q' or ESC to quit`.

This is purely a visualization of the **same underlying RGBD signals** that can be used for:
- State augmentation (e.g., appending a 6-D hold pose).
- Depth-based segmentation / pose estimation.
- Debugging misalignment between perceived hold location and executed grasp.

---

### Conceptual “screenshot” of the viewer

Below is a **textual approximation** of what the viewer looks like when running with 4 cameras (2, 3, 4, 5). Each `[RGB X]` block is a 320×180 camera image; each `[DEPTH X]` block is the corresponding depth colormap.

```text
+---------------------------------------------------------------+
| [ Cam 2 RGB 320x180 ]   |   [ Cam 3 RGB 320x180 ]            |
|   rock wall + hold      |     rock wall + hold               |
|   label: "Cam 2"        |     label: "Cam 3"                 |
+---------------------------------------------------------------+
| [ Depth 2 colormap ]    |   [ Depth 3 colormap ]             |
|   blue=far, red=near    |     blue=far, red=near             |
|   label: "Depth 2"      |     label: "Depth 3"               |
+===============================================================+
| [ Cam 4 RGB 320x180 ]   |   [ Cam 5 RGB 320x180 ]            |
|   rock wall + hold      |     rock wall + hold               |
|   label: "Cam 4"        |     label: "Cam 5"                 |
+---------------------------------------------------------------+
| [ Depth 4 colormap ]    |   [ Depth 5 colormap ]             |
|   blue=far, red=near    |     blue=far, red=near             |
|   label: "Depth 4"      |     label: "Depth 5"               |
+---------------------------------------------------------------+
Status bar (overlaid at bottom):
  "Live RGB+Depth per camera  |  Press 'q' or ESC to quit"
```

This layout is what an OpenCV `imshow` window would display; the actual images are real RGB and depth data from the live RealSense sensors.

If a literal screenshot PNG is available (for this worktree, `rgbd_screenshot.png`
captured via `capture_rgbd_screenshot.py`), it can be embedded directly:

```markdown
![Example RGB+Depth live view](rgbd_screenshot.png)
```

---

### Key code paths (for other agents)

- **Camera acquisition** (identical pattern to `collect_data.py` / `evaluate.py`):
  - Construct threaded cameras:

    ```python
    cameras = vis.ThreadedCameras(
        cam_numbers=CAMERA_NUMBERS,
        image_height=CAMERA_RAW_H,
        image_width=CAMERA_RAW_W,
        get_point_cloud=False,
        get_verts=False,
    )
    frames = cameras.get_next_frames()  # warm-up
    ```

  - Each `frames[i]` is `(color, depth, _, _)` where:
    - `color`: `(H, W, 3)` `uint8`, RGB image.
    - `depth`: `(H, W)` `uint16`, depth in mm from the RealSense.

- **Depth colormap generation** (from `view_rgbd_live.py`):

    ```python
    DEPTH_SCALE = 0.001  # mm -> meters

    def colorize_depth(depth, max_depth_m=1.0):
        if depth is None or depth.size == 0:
            return np.zeros((CAMERA_RAW_H, CAMERA_RAW_W, 3), dtype=np.uint8)

        depth_m = depth.astype(np.float32) * DEPTH_SCALE
        depth_norm = np.clip(depth_m / max_depth_m, 0.0, 1.0)
        depth_8u = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        return depth_color
    ```

- **Per-camera tile construction**:

    ```python
    rgb_small = cv2.resize(color, (320, 180))
    cv2.putText(rgb_small, f"Cam {cam_id}", (5, 15), ...)

    depth_color = colorize_depth(depth, max_depth_m=1.0)
    depth_small = cv2.resize(depth_color, (320, 180))
    cv2.putText(depth_small, f"Depth {cam_id}", (5, 15), ...)

    tile = np.vstack([rgb_small, depth_small])  # shape ~ (360, 320, 3)
    ```

- **Global canvas tiling**:

    ```python
    tiles = [...]  # one tile per camera
    ncols = 2
    while len(tiles) % ncols != 0:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + ncols]) for i in range(0, len(tiles), ncols)]
    canvas = np.vstack(rows)

    status = "Live RGB+Depth per camera  |  Press 'q' or ESC to quit"
    cv2.putText(canvas, status, (5, canvas.shape[0] - 8), ...)
    cv2.imshow("RGB + Depth (all cameras)", canvas)
    ```

Any agent familiar with the training/evaluation pipeline can treat this viewer as:
- A **ground truth check** on what the RealSense depth actually looks like.
- A template for extracting more informative depth-based features (e.g., hold surface geometry, distance transforms, multi-camera fusion) to combat systematic spatial offsets in grasp execution.

---

### How this helps with the “wrong spot” issue

The observed failure mode is: the robot hand chooses the **correct hold type** but moves to the **wrong physical location** (off by a few inches).

This viewer is useful for:

- **Verifying depth quality and coverage**:
  - Ensuring the hold and wall are well represented in the depth map for each camera.
  - Detecting holes, saturation, or range clipping that could bias pose estimation.

- **Checking camera configuration consistency**:
  - Confirming that camera viewpoints, intrinsics, and mounting are consistent between **data collection** and **evaluation**.
  - Cross-checking that the pixel region used by any hold detector (e.g., `hold_detector.py`) matches where the hold actually appears in RGB and depth.

- **Designing better state representations**:
  - Using the dense depth maps to compute richer hold descriptors (e.g., centroid, normal, curvature, local height map).
  - Feeding per-episode or per-frame 3D hold pose into the state vector (36-D or higher), so the policy has explicit spatial grounding rather than inferring everything from RGB alone.

By understanding exactly what the RealSense sensors provide at the pixel level, downstream policies can be redesigned or retrained to minimize systematic pose errors and improve grasp precision.

