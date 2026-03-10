"""
Zarr-based episode storage for diffusion policy training data.

Dataset layout:
    dataset.zarr/
        data/
            state     (N, state_dim)      float32   # arm(7) + ee_pos(3) + ee_quat(4) + hand(16) = 30
            action    (N, action_dim)     float32   # same dims as state
            timestamps (N,)              float64
            img/
                cam2  (N, H, W, 3)        uint8
                cam3  (N, H, W, 3)        uint8
                cam4  (N, H, W, 3)        uint8
                cam5  (N, H, W, 3)        uint8
        meta/
            episode_ends   (num_episodes,)   int64
            hold_id        (num_episodes,)   int64    # which hold (0-3)
            quality        (num_episodes,)   int64    # 1=good, 0=bad
            grasp_type     (num_episodes,)   |S32     # optional string label

N = total timesteps across ALL episodes.
episode_ends[i] = cumulative index where episode i ends.
"""

import os
import numpy as np
import zarr
import cv2


class EpisodeBuffer:
    """Accumulates timesteps for a single episode in memory."""

    def __init__(self, cam_names, store_depth=False):
        self.cam_names = cam_names
        self.store_depth = store_depth
        self.states = []
        self.actions = []
        self.images = {name: [] for name in cam_names}
        self.depths = {name: [] for name in cam_names} if store_depth else {}
        self.timestamps = []

    def add_timestep(self, state, action, images_dict, timestamp, depths_dict=None):
        self.states.append(state.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        for name in self.cam_names:
            self.images[name].append(images_dict[name])
        if self.store_depth and depths_dict is not None:
            for name in self.cam_names:
                self.depths[name].append(depths_dict[name])
        self.timestamps.append(timestamp)

    def __len__(self):
        return len(self.states)

    def finalize_actions(self):
        """Shift actions forward by one timestep so action[t] = state[t+1].
        The last action is duplicated from the second-to-last."""
        if len(self.states) < 2:
            return
        self.actions = self.states[1:] + [self.states[-1]]

    def get_arrays(self):
        result = {
            "states": np.stack(self.states),
            "actions": np.stack(self.actions),
            "images": {name: np.stack(imgs) for name, imgs in self.images.items()},
            "timestamps": np.array(self.timestamps),
        }
        if self.store_depth and self.depths:
            result["depths"] = {name: np.stack(d) for name, d in self.depths.items()}
        return result

    def clear(self):
        self.states.clear()
        self.actions.clear()
        for name in self.cam_names:
            self.images[name].clear()
        for name in self.depths:
            self.depths[name].clear()
        self.timestamps.clear()


class ZarrDatasetWriter:
    """
    Writes episodes into a zarr store in the diffusion_policy format.
    Supports per-episode metadata (hold_id, quality, grasp_type).
    """

    def __init__(self, zarr_path, cam_names, img_height=224, img_width=224,
                 state_dim=30, action_dim=30, store_depth=False):
        self.zarr_path = zarr_path
        self.cam_names = cam_names
        self.img_height = img_height
        self.img_width = img_width
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.store_depth = store_depth

        self.root = zarr.open(zarr_path, mode="a")
        self._ensure_structure()

    def _ensure_structure(self):
        data = self.root.require_group("data")
        meta = self.root.require_group("meta")

        chunk_t = 256

        if "state" not in data:
            data.create_dataset(
                "state", shape=(0, self.state_dim),
                chunks=(chunk_t, self.state_dim),
                dtype=np.float32,
                compressor=zarr.Blosc(cname="lz4", clevel=5),
            )
        if "action" not in data:
            data.create_dataset(
                "action", shape=(0, self.action_dim),
                chunks=(chunk_t, self.action_dim),
                dtype=np.float32,
                compressor=zarr.Blosc(cname="lz4", clevel=5),
            )
        if "timestamps" not in data:
            data.create_dataset(
                "timestamps", shape=(0,),
                chunks=(chunk_t,), dtype=np.float64,
            )

        img_group = data.require_group("img")
        for cam_name in self.cam_names:
            if cam_name not in img_group:
                img_group.create_dataset(
                    cam_name,
                    shape=(0, self.img_height, self.img_width, 3),
                    chunks=(1, self.img_height, self.img_width, 3),
                    dtype=np.uint8,
                    compressor=zarr.Blosc(cname="lz4", clevel=5),
                )

        if self.store_depth:
            depth_group = data.require_group("depth")
            for cam_name in self.cam_names:
                if cam_name not in depth_group:
                    depth_group.create_dataset(
                        cam_name,
                        shape=(0, self.img_height, self.img_width),
                        chunks=(1, self.img_height, self.img_width),
                        dtype=np.uint16,
                        compressor=zarr.Blosc(cname="lz4", clevel=5),
                    )

        if "episode_ends" not in meta:
            meta.create_dataset("episode_ends", shape=(0,), chunks=(256,), dtype=np.int64)
        if "hold_id" not in meta:
            meta.create_dataset("hold_id", shape=(0,), chunks=(256,), dtype=np.int64)
        if "quality" not in meta:
            meta.create_dataset("quality", shape=(0,), chunks=(256,), dtype=np.int64)
        if "grasp_type" not in meta:
            meta.create_dataset("grasp_type", shape=(0,), chunks=(256,), dtype="<U32")

    @property
    def num_episodes(self):
        return self.root["meta/episode_ends"].shape[0]

    @property
    def total_timesteps(self):
        return self.root["data/state"].shape[0]

    def append_episode(self, episode_buffer, hold_id=0, quality=1, grasp_type=""):
        """Flush one EpisodeBuffer + metadata into the zarr store."""
        if len(episode_buffer) == 0:
            print("Warning: empty episode, skipping.")
            return

        episode_buffer.finalize_actions()
        arrays = episode_buffer.get_arrays()
        n = len(episode_buffer)

        self.root["data/state"].append(arrays["states"])
        self.root["data/action"].append(arrays["actions"])
        self.root["data/timestamps"].append(arrays["timestamps"])

        for cam_name in self.cam_names:
            self.root[f"data/img/{cam_name}"].append(arrays["images"][cam_name])

        if self.store_depth and "depths" in arrays:
            for cam_name in self.cam_names:
                self.root[f"data/depth/{cam_name}"].append(arrays["depths"][cam_name])

        new_end = self.total_timesteps
        self.root["meta/episode_ends"].append(np.array([new_end], dtype=np.int64))
        self.root["meta/hold_id"].append(np.array([hold_id], dtype=np.int64))
        self.root["meta/quality"].append(np.array([quality], dtype=np.int64))
        self.root["meta/grasp_type"].append(np.array([grasp_type]))

        quality_str = "GOOD" if quality == 1 else "BAD"
        print(f"Saved episode {self.num_episodes}: {n} steps, "
              f"hold={hold_id}, quality={quality_str}, "
              f"({self.total_timesteps} total across {self.num_episodes} episodes)")

    def get_summary(self):
        summary = {
            "zarr_path": self.zarr_path,
            "num_episodes": self.num_episodes,
            "total_timesteps": self.total_timesteps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cameras": self.cam_names,
            "img_size": (self.img_height, self.img_width),
        }
        if self.num_episodes > 0:
            holds = self.root["meta/hold_id"][:]
            quality = self.root["meta/quality"][:]
            summary["good_episodes"] = int(np.sum(quality == 1))
            summary["bad_episodes"] = int(np.sum(quality == 0))
            for h in np.unique(holds):
                mask = holds == h
                n_good = int(np.sum(quality[mask] == 1))
                summary[f"hold_{h}"] = f"{int(np.sum(mask))} eps ({n_good} good)"
        return summary


def resize_image(img, target_h, target_w):
    """Resize a BGR image to (target_h, target_w) using area interpolation."""
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
