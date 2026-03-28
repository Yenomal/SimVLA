"""
RMBench HDF5 Data Handler

Expected episode layout:
- observation/head_camera/rgb: [T] encoded RGB frames
- observation/left_camera/rgb: [T] encoded RGB frames
- observation/right_camera/rgb: [T] encoded RGB frames
- joint_action/vector: [T, 14] qpos vector for the default aloha-agilex setup

This baseline uses:
- proprio/state: current joint_action/vector
- action targets: future absolute joint_action/vector values
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterable, List

import numpy as np
import torch

from .base import DomainHandler
from ..utils import decode_image_from_bytes, open_h5


class RMBenchHDF5Handler(DomainHandler):
    """
    RMBench episode handler for absolute joint-space prediction.

    The current baseline assumes the default aloha-agilex embodiment, whose
    joint vector is 14-dimensional:
      [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]
    """

    dataset_name = "rmbench_hdf5"

    DEFAULT_OBSERVATION_KEYS = [
        "observation/head_camera/rgb",
        "observation/left_camera/rgb",
        "observation/right_camera/rgb",
    ]

    def __init__(self, meta: dict, num_views: int = 3) -> None:
        super().__init__(meta, num_views)
        self.items: List[dict] = list(meta.get("datalist", []))
        self.observation_keys: List[str] = list(
            meta.get("observation_key", self.DEFAULT_OBSERVATION_KEYS)
        )

    def _load_instructions(self, item: dict) -> List[str]:
        """Load episode-level language instructions."""
        candidates: List[str] = []
        instruction_path = item.get("instruction_path")

        if instruction_path and os.path.exists(instruction_path):
            try:
                with open(instruction_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for key in ("seen", "unseen"):
                    vals = data.get(key, [])
                    if isinstance(vals, list):
                        candidates.extend(str(v) for v in vals if str(v).strip())
            except Exception:
                pass

        if not candidates:
            task_name = str(item.get("task", "")).strip()
            if task_name:
                candidates.append(task_name.replace("_", " "))
            else:
                candidates.append("robot manipulation")

        # Preserve order while dropping duplicates.
        return list(dict.fromkeys(candidates))

    def _read_image_streams(self, f) -> List[np.ndarray]:
        """Read configured image streams from HDF5."""
        image_streams: List[np.ndarray] = []
        for key in self.observation_keys:
            try:
                image_streams.append(f[key][()])
            except KeyError:
                continue
        return image_streams

    @staticmethod
    def _get_joint_chunk(joint_vector: np.ndarray, start_idx: int, num_actions: int) -> np.ndarray:
        """
        Build [num_actions+1, D] absolute trajectory from the current state.

        Row 0 is the current joint vector.
        Rows 1..N are future joint targets, padded with the last frame if needed.
        """
        T, D = joint_vector.shape
        chunk = np.zeros((num_actions + 1, D), dtype=np.float32)
        chunk[0] = joint_vector[start_idx]
        for step in range(num_actions):
            src = min(start_idx + 1 + step, T - 1)
            chunk[step + 1] = joint_vector[src]
        return chunk

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 10,
        training: bool = True,
        image_aug=None,
        action_mode: str = "rmbench_joint",
        lang_aug_map: dict | None = None,
        **kwargs,
    ) -> Iterable[dict]:
        """
        Iterate over samples in one RMBench episode.

        The current implementation is intentionally simple:
        - current observation only
        - absolute future joint targets
        - episode-level instruction sampling
        """
        if action_mode.lower() != "rmbench_joint":
            raise ValueError(
                f"RMBenchHDF5Handler only supports action_mode='rmbench_joint', got '{action_mode}'."
            )

        item = self.items[traj_idx]
        episode_path = item["path"]
        instructions = self._load_instructions(item)

        with open_h5(episode_path) as f:
            try:
                joint_ds = f["joint_action/vector"]
            except KeyError:
                raise KeyError(f"Missing 'joint_action/vector' in {episode_path}")

            joint_vector = np.asarray(joint_ds, dtype=np.float32)
            image_streams = self._read_image_streams(f)

            if not image_streams:
                raise KeyError(f"No configured image streams found in {episode_path}")

            lengths = [len(joint_vector)] + [len(stream) for stream in image_streams]
            T = min(lengths)
            if T < 2:
                return

            joint_vector = joint_vector[:T]
            image_streams = [stream[:T] for stream in image_streams]

        indices = list(range(T - 1))
        if training:
            random.shuffle(indices)

        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[: min(len(image_streams), self.num_views)] = True

        for idx in indices:
            instruction = random.choice(instructions) if training else instructions[0]
            imgs = []

            for stream in image_streams[: self.num_views]:
                img = decode_image_from_bytes(stream[idx])
                if image_aug:
                    img = image_aug(img)
                elif not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                imgs.append(img)

            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            image_input = torch.stack(imgs, dim=0)
            abs_trajectory = self._get_joint_chunk(joint_vector, idx, num_actions)

            yield {
                "language_instruction": instruction,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.tensor(abs_trajectory, dtype=torch.float32),
            }


__all__ = ["RMBenchHDF5Handler"]
