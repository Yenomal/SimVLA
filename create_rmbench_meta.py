#!/usr/bin/env python
"""
Create RMBench training metadata for SimVLA.

Expected directory layout:
  <data_dir>/<task_name>/<task_config>/
      data/episode0.hdf5
      instructions/episode0.json

Example:
  python create_rmbench_meta.py \
      --data_dir ../RMBench/data/data \
      --task_config demo_clean \
      --tasks battery_try blocks_ranking_try \
      --split train \
      --train_episodes_per_task 40 \
      --eval_episodes_per_task 10 \
      --output ./datasets/metas/rmbench_train.json
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


EPISODE_RE = re.compile(r"episode(\d+)\.hdf5$")


def list_tasks(data_dir: str) -> List[str]:
    """List task directories under the RMBench data root."""
    root = Path(data_dir)
    tasks = [p.name for p in root.iterdir() if p.is_dir()]
    return sorted(tasks)


def parse_episode_id(path: str) -> int:
    """Extract episode id from episodeX.hdf5."""
    match = EPISODE_RE.search(os.path.basename(path))
    if match is None:
        raise ValueError(f"Unsupported episode filename: {path}")
    return int(match.group(1))


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


def create_rmbench_meta(
    data_dir: str,
    task_config: str = "demo_clean",
    tasks: Optional[List[str]] = None,
    split: str = "all",
    train_episodes_per_task: int = 40,
    eval_episodes_per_task: int = 10,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Create RMBench metadata JSON consumable by SimVLA dataloaders.
    """
    if tasks is None or len(tasks) == 0:
        tasks = list_tasks(data_dir)

    datalist = []
    task_stats = {}
    total_episodes = 0

    print(f"Scanning RMBench dataset: {data_dir}")
    print(f"Task config: {task_config}")
    print(f"Split: {split}")
    print(f"Train episodes/task: {train_episodes_per_task}")
    print(f"Eval episodes/task: {eval_episodes_per_task}")

    for task_name in tasks:
        task_root = Path(data_dir) / task_name / task_config
        data_root = task_root / "data"
        instruction_root = task_root / "instructions"

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
        episode_count = 0

        for episode_path in selected_episode_paths:
            episode_id = parse_episode_id(episode_path)
            instruction_path = instruction_root / f"episode{episode_id}.json"

            datalist.append(
                {
                    "path": episode_path,
                    "task": task_name,
                    "task_config": task_config,
                    "episode_id": episode_id,
                    "instruction_path": str(instruction_path) if instruction_path.exists() else None,
                }
            )
            episode_count += 1

        task_stats[task_name] = {"num_episodes": episode_count}
        total_episodes += episode_count
        print(f"  {task_name}: selected {episode_count}/{len(episode_paths)} episodes")

    meta = {
        "dataset_name": "rmbench_hdf5",
        "data_dir": data_dir,
        "task_config": task_config,
        "split": split,
        "train_episodes_per_task": int(train_episodes_per_task),
        "eval_episodes_per_task": int(eval_episodes_per_task),
        "tasks": list(task_stats.keys()),
        "task_stats": task_stats,
        "num_episodes": total_episodes,
        "datalist": datalist,
        "observation_key": [
            "observation/head_camera/rgb",
            "observation/left_camera/rgb",
            "observation/right_camera/rgb",
        ],
        "state_key": "joint_action/vector",
        "instruction_source": "instructions/episode*.json",
        "num_views": 3,
        "state_dim": 14,
        "action_dim": 14,
    }

    print(f"Found {total_episodes} episodes across {len(task_stats)} tasks.")

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {output}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Create RMBench training metadata")
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
        default="all",
        choices=["all", "train", "eval"],
        help="Which deterministic split to export",
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
        default="./datasets/metas/rmbench_train.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    create_rmbench_meta(
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
