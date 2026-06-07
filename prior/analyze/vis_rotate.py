"""Visualize right-angle tensor rotation for a cached cognitive map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from prior import DATA_DIR
from prior._coords import grid_to_meters
from prior.grid_map import CognitiveGridMap
from vlnce_baselines.models.etp_prior_gt.map_utils import (
    cognitive_map_to_tensors,
    rotate_cognitive_map_tensors_by_right_angle,
)

DEFAULT_INPUT_NPZ = DATA_DIR / "cognitive_maps" / "1LXtFkjw3qL" / "R2R_train_85.npz"
DEFAULT_OUTPUT_PNG = DATA_DIR / "samples" / "R2R_train_85_rot.png"


def _trajectory_keypoints_from_grid_tensor(
    trajectory_keypoints: torch.Tensor,
) -> list[tuple[float, float]]:
    path = []
    for row, col in trajectory_keypoints.detach().cpu().tolist():
        x, z = grid_to_meters(float(row), float(col))
        path.append((float(x), float(z)))
    return path


def _direction_from_tensor(start_direction_vector: torch.Tensor) -> tuple[float, float]:
    vector = start_direction_vector.detach().cpu()
    if vector.shape != (2,):
        raise ValueError(
            f"start_direction_vector must have shape (2,), got {tuple(vector.shape)}"
        )
    return (float(vector[0]), float(vector[1]))


def rotated_cognitive_map(
    cognitive_map: CognitiveGridMap, turns: int
) -> CognitiveGridMap:
    tensors = cognitive_map_to_tensors(cognitive_map)
    rotated_tensors = rotate_cognitive_map_tensors_by_right_angle(tensors, turns)

    rotated = CognitiveGridMap()
    rotated.grid = rotated_tensors["grid"].detach().cpu().numpy().astype(np.float32)
    rotated.range_y = list(cognitive_map.range_y)
    rotated.trajectory_keypoints = _trajectory_keypoints_from_grid_tensor(
        rotated_tensors["trajectory_keypoints"]
    )
    rotated.start_direction_vector = _direction_from_tensor(
        rotated_tensors["start_direction_vector"]
    )
    return rotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--turns", type=int, default=1, choices=[0, 1, 2, 3])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cognitive_map = CognitiveGridMap.load(args.input)
    rotated_cognitive_map(cognitive_map, args.turns).visualize(args.output)


if __name__ == "__main__":
    main()
