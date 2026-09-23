#!/usr/bin/env python3
"""Dump CPU-generated PriorGT cognitive maps from commit 121c369 datasets."""

import gzip
import json
from pathlib import Path

import numpy as np

from habitat_extensions.task import VLNCEDatasetV2
from vlnce_baselines.models.etp_prior_gt.map_utils import (
    build_cognitive_map_for_episode,
    cognitive_map_to_tensors,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr"
OUTPUT_DIR = REPO_ROOT / "reports/online_maps_121c369"


def load_episodes(path: Path):
    """Use Habitat's VLNCE episode deserializer, as evaluation does."""
    dataset = VLNCEDatasetV2()
    with gzip.open(path, "rt") as file:
        dataset.from_json(file.read(), scenes_dir="data/scene_datasets/")
    return dataset.episodes


def dump(split: str, episodes) -> list:
    records = []
    for episode in episodes:
        tensors = cognitive_map_to_tensors(
            build_cognitive_map_for_episode(episode, radius_m=1.5)
        )
        filename = f"R2R_{split}_{episode.episode_id}.npz"
        np.savez_compressed(
            OUTPUT_DIR / filename,
            **{name: tensor.cpu().numpy() for name, tensor in tensors.items()},
        )
        records.append(
            {
                "split": split,
                "episode_id": str(episode.episode_id),
                "scene_id": episode.scene_id,
                "start_position_world": list(episode.start_position),
                "start_rotation": list(episode.start_rotation),
                "file": filename,
            }
        )
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_episodes = load_episodes(DATA_ROOT / "val_unseen/val_unseen.json.gz")[:50]
    train_episodes = load_episodes(DATA_ROOT / "train/train_90.json.gz")[:5]
    records = dump("val_unseen", val_episodes) + dump("train_90", train_episodes)
    with (OUTPUT_DIR / "index.json").open("w") as file:
        json.dump(records, file, indent=2)
    print(f"wrote {len(records)} maps to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
