#!/usr/bin/env python
"""
Compute RMBench normalization statistics for SimVLA.

This baseline uses absolute joint vectors as both state and action targets:
  state_t   = joint_action/vector[t]
  action_t  = joint_action/vector[t+1]

Example:
  python compute_rmbench_norm_stats.py \
      --data_dir ../RMBench/data/data \
      --task_config demo_clean \
      --tasks battery_try blocks_ranking_try \
      --split train \
      --train_episodes_per_task 40 \
      --eval_episodes_per_task 10 \
      --output ./norm_stats/rmbench_joint_norm.json
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
from tqdm import tqdm


class RunningStats:
    """Compute running mean/std with sampled quantiles."""

    def __init__(self, dim: int):
        self.dim = dim
        self._count = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._mean_sq = np.zeros(dim, dtype=np.float64)
        self._min = np.full(dim, np.inf, dtype=np.float64)
        self._max = np.full(dim, -np.inf, dtype=np.float64)
        self._samples: List[np.ndarray] = []
        self._max_samples = 100000

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, self.dim)
        n = batch.shape[0]
        if n == 0:
            return

        self._min = np.minimum(self._min, batch.min(axis=0))
        self._max = np.maximum(self._max, batch.max(axis=0))

        sample_count = min(100, n)
        if len(self._samples) * 100 < self._max_samples:
            sample_idx = np.random.choice(n, sample_count, replace=False)
            self._samples.append(batch[sample_idx])

        batch_mean = batch.mean(axis=0)
        batch_mean_sq = (batch ** 2).mean(axis=0)

        total = self._count + n
        self._mean = (self._mean * self._count + batch_mean * n) / total
        self._mean_sq = (self._mean_sq * self._count + batch_mean_sq * n) / total
        self._count = total

    def get(self) -> Dict[str, np.ndarray]:
        if self._count < 2:
            raise ValueError("Need at least two samples to compute stats")

        variance = self._mean_sq - self._mean ** 2
        std = np.sqrt(np.maximum(variance, 0.0))

        samples = np.concatenate(self._samples, axis=0) if self._samples else np.zeros((1, self.dim))
        q01 = np.percentile(samples, 1, axis=0)
        q99 = np.percentile(samples, 99, axis=0)

        return {
            "mean": self._mean.astype(np.float32),
            "std": std.astype(np.float32),
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
            "min": self._min.astype(np.float32),
            "max": self._max.astype(np.float32),
            "count": int(self._count),
        }


def list_tasks(data_dir: str) -> List[str]:
    """List task directories under the RMBench data root."""
    root = Path(data_dir)
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def select_episode_paths(
    episode_paths: List[str],
    split: str,
    train_episodes_per_task: int,
    eval_episodes_per_task: int,
) -> List[str]:
    """Select a deterministic train/eval/all split from sorted episode paths."""
    if split == "all":
        return episode_paths

    train_count = max(0, int(train_episodes_per_task))
    eval_count = max(0, int(eval_episodes_per_task))

    if split == "train":
        return episode_paths[:train_count]

    if split == "eval":
        start = min(train_count, len(episode_paths))
        end = min(start + eval_count, len(episode_paths))
        return episode_paths[start:end]

    raise ValueError(f"Unsupported split: {split}")


def compute_rmbench_norm_stats(
    data_dir: str,
    task_config: str = "demo_clean",
    tasks: Optional[List[str]] = None,
    split: str = "train",
    train_episodes_per_task: int = 40,
    eval_episodes_per_task: int = 10,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute normalization statistics from RMBench HDF5 episodes.
    """
    if tasks is None or len(tasks) == 0:
        tasks = list_tasks(data_dir)

    state_stats = RunningStats(dim=14)
    action_stats = RunningStats(dim=14)

    total_episodes = 0
    total_steps = 0

    print("Computing RMBench normalization statistics")
    print(f"  Data directory: {data_dir}")
    print(f"  Task config: {task_config}")
    print(f"  Split: {split}")
    print(f"  Train episodes/task: {train_episodes_per_task}")
    print(f"  Eval episodes/task: {eval_episodes_per_task}")
    print(f"  Tasks: {tasks}")

    for task_name in tasks:
        data_root = Path(data_dir) / task_name / task_config / "data"
        if not data_root.exists():
            print(f"Warning: skipping missing task data dir: {data_root}")
            continue

        episode_paths = sorted(glob.glob(str(data_root / "episode*.hdf5")))
        selected_episode_paths = select_episode_paths(
            episode_paths,
            split=split,
            train_episodes_per_task=train_episodes_per_task,
            eval_episodes_per_task=eval_episodes_per_task,
        )
        print(f"  {task_name}: selected {len(selected_episode_paths)}/{len(episode_paths)} episodes")

        for episode_path in tqdm(selected_episode_paths, desc=task_name):
            try:
                with h5py.File(episode_path, "r") as f:
                    joint_vector = np.asarray(f["joint_action/vector"], dtype=np.float32)
            except Exception as exc:
                print(f"Warning: failed to read {episode_path}: {exc}")
                continue

            if joint_vector.ndim != 2 or joint_vector.shape[-1] != 14 or joint_vector.shape[0] < 2:
                print(f"Warning: skipping malformed joint vector in {episode_path}, shape={joint_vector.shape}")
                continue

            state_stats.update(joint_vector[:-1])
            action_stats.update(joint_vector[1:])
            total_episodes += 1
            total_steps += joint_vector.shape[0] - 1

    state = state_stats.get()
    actions = action_stats.get()

    labels = [
        "left_joint_0",
        "left_joint_1",
        "left_joint_2",
        "left_joint_3",
        "left_joint_4",
        "left_joint_5",
        "left_gripper",
        "right_joint_0",
        "right_joint_1",
        "right_joint_2",
        "right_joint_3",
        "right_joint_4",
        "right_joint_5",
        "right_gripper",
    ]

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "norm_stats": {
                "state": {
                    "mean": state["mean"].tolist(),
                    "std": state["std"].tolist(),
                    "q01": state["q01"].tolist(),
                    "q99": state["q99"].tolist(),
                },
                "actions": {
                    "mean": actions["mean"].tolist(),
                    "std": actions["std"].tolist(),
                    "q01": actions["q01"].tolist(),
                    "q99": actions["q99"].tolist(),
                },
            },
            "metadata": {
                "data_dir": data_dir,
                "task_config": task_config,
                "split": split,
                "train_episodes_per_task": int(train_episodes_per_task),
                "eval_episodes_per_task": int(eval_episodes_per_task),
                "tasks": tasks,
                "num_episodes": total_episodes,
                "num_steps": total_steps,
                "state_dim": 14,
                "action_dim": 14,
                "labels": labels,
            },
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {output}")

    return {"state": state, "actions": actions}


def main():
    parser = argparse.ArgumentParser(description="Compute RMBench normalization statistics")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="RMBench data root, e.g. ../RMBench/data/data",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="demo_clean",
        help="Task config directory under each task, default demo_clean",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Optional task names. If omitted, scan all tasks under data_dir.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["all", "train", "eval"],
        help="Which deterministic split to use when computing stats",
    )
    parser.add_argument(
        "--train_episodes_per_task",
        type=int,
        default=40,
        help="How many sorted episodes per task go to the train split",
    )
    parser.add_argument(
        "--eval_episodes_per_task",
        type=int,
        default=10,
        help="How many sorted episodes per task go to the eval split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./norm_stats/rmbench_joint_norm.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    compute_rmbench_norm_stats(
        data_dir=args.data_dir,
        task_config=args.task_config,
        tasks=args.tasks,
        split=args.split,
        train_episodes_per_task=args.train_episodes_per_task,
        eval_episodes_per_task=args.eval_episodes_per_task,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
