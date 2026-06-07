from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Optional

import torch
from prior import DATA_DIR
from prior._coords import meters_to_grid
from prior.constants import (
    COLS,
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    ROWS,
)
from prior.directions import start_rotation_to_direction_vector
from prior.bbox import SceneSemanticBoxes
from prior.grid_map import CognitiveGridMap
from prior.trajectory import TRAJECTORY_KEYPOINT_COUNT

NUM_MAP_CATEGORIES = len(MAPPED_OBJECT_NAMES) + len(MAPPED_REGION_NAMES)
VLNCE_COGNITIVE_MAP_DIR = DATA_DIR / "cognitive_maps"
ETP_R1_COGNITIVE_MAP_DIR = DATA_DIR / "cognitive_maps_etp_r1"

SIZE = ROWS
"""Number of rows and cols in the grid map."""

if ROWS != COLS:
    raise ValueError(f"PriorGT map encoder requires square maps, got {ROWS}x{COLS}")


def _scene_key(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(scene_id))[0]


def _trajectory_keypoints_to_grid_tensor(
    cognitive_map: CognitiveGridMap,
) -> torch.Tensor:
    if len(cognitive_map.trajectory_keypoints) != TRAJECTORY_KEYPOINT_COUNT:
        raise ValueError(
            "CognitiveGridMap.trajectory_keypoints must contain exactly "
            f"{TRAJECTORY_KEYPOINT_COUNT} points"
        )
    trajectory_keypoints = torch.zeros(
        TRAJECTORY_KEYPOINT_COUNT, 2, dtype=torch.float32
    )
    for idx, position in enumerate(cognitive_map.trajectory_keypoints):
        row, col = meters_to_grid(float(position[0]), float(position[1]))
        trajectory_keypoints[idx] = torch.tensor([row, col], dtype=torch.float32)
    return trajectory_keypoints


def _start_position_tensor(cognitive_map: CognitiveGridMap) -> torch.Tensor:
    if not cognitive_map.trajectory_keypoints:
        raise ValueError("CognitiveGridMap.trajectory_keypoints is empty")
    start_x, start_z = cognitive_map.trajectory_keypoints[0]
    return torch.tensor(
        meters_to_grid(float(start_x), float(start_z)),
        dtype=torch.float32,
    )


def cognitive_map_to_tensors(cognitive_map: CognitiveGridMap):
    return {
        "grid": torch.from_numpy(cognitive_map.grid),
        "trajectory_keypoints": _trajectory_keypoints_to_grid_tensor(cognitive_map),
        "start_direction_vector": torch.tensor(
            cognitive_map.start_direction_vector, dtype=torch.float32
        ),
        "start_position": _start_position_tensor(cognitive_map),
    }


def cognitive_map_cache_path(
    scene_id: str,
    cache_id: str,
    cache_dir: Optional[Path] = None,
) -> Path:
    if cache_dir is None:
        cache_dir = VLNCE_COGNITIVE_MAP_DIR
    return cache_dir / _scene_key(scene_id) / f"{cache_id}.npz"


def load_cached_cognitive_map(
    scene_id: str,
    cache_id: str,
    cache_dir: Optional[Path] = None,
) -> CognitiveGridMap:
    cache_path = cognitive_map_cache_path(scene_id, cache_id, cache_dir)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing cached cognitive map: {cache_path}")
    return CognitiveGridMap.load(cache_path)


def cached_cognitive_map_to_tensors(
    scene_id: str,
    cache_id: str,
    cache_dir: Optional[Path] = None,
    random_rotation_augmentation: bool = False,
) -> Dict[str, torch.Tensor]:
    tensors = cognitive_map_to_tensors(
        load_cached_cognitive_map(scene_id, cache_id, cache_dir)
    )
    if random_rotation_augmentation:
        tensors = rotate_cognitive_map_tensors_by_right_angle(
            tensors,
            random.randrange(4),
        )
    return tensors


def _rotate_grid_points_by_right_angle(
    points: torch.Tensor,
    turns: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    row = points[..., 0]
    col = points[..., 1]
    turns %= 4
    if turns == 0:
        return points
    if turns == 1:
        return torch.stack((col.new_tensor(float(cols - 1)) - col, row), dim=-1)
    if turns == 2:
        return torch.stack(
            (
                row.new_tensor(float(rows - 1)) - row,
                col.new_tensor(float(cols - 1)) - col,
            ),
            dim=-1,
        )
    return torch.stack((col, row.new_tensor(float(rows - 1)) - row), dim=-1)


def _rotate_trajectory_keypoints_by_right_angle(
    points: torch.Tensor,
    turns: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    nonzero = torch.any(points != 0, dim=-1)
    has_nonzero_at_or_after = torch.flip(
        torch.cumsum(
            torch.flip(nonzero.to(torch.int64), dims=(-1,)),
            dim=-1,
        ),
        dims=(-1,),
    )
    padding = (has_nonzero_at_or_after == 0).unsqueeze(-1)
    rotated = _rotate_grid_points_by_right_angle(points, turns, rows, cols)
    return torch.where(padding, torch.zeros_like(rotated), rotated)


def _rotate_direction_by_right_angle(vector: torch.Tensor, turns: int) -> torch.Tensor:
    sin_value = vector[..., 0]
    cos_value = vector[..., 1]
    turns %= 4
    if turns == 0:
        return vector
    if turns == 1:
        return torch.stack((cos_value, -sin_value), dim=-1)
    if turns == 2:
        return torch.stack((-sin_value, -cos_value), dim=-1)
    return torch.stack((-cos_value, sin_value), dim=-1)


def rotate_cognitive_map_tensors_by_right_angle(
    tensors: Dict[str, torch.Tensor], turns: int
) -> Dict[str, torch.Tensor]:
    turns %= 4
    if turns == 0:
        return dict(tensors)

    grid = torch.rot90(tensors["grid"], turns, dims=(-2, -1)).contiguous()
    rows = int(tensors["grid"].shape[-2])
    cols = int(tensors["grid"].shape[-1])
    rotated = dict(tensors)
    rotated["grid"] = grid
    rotated["trajectory_keypoints"] = _rotate_trajectory_keypoints_by_right_angle(
        tensors["trajectory_keypoints"], turns, rows, cols
    )
    rotated["start_position"] = _rotate_grid_points_by_right_angle(
        tensors["start_position"], turns, rows, cols
    )
    rotated["start_direction_vector"] = _rotate_direction_by_right_angle(
        tensors["start_direction_vector"], turns
    )
    return rotated


def start_metadata_to_tensors(scene_id: str, start_position, start_rotation):
    """Build inference-safe metadata from episode start state."""
    scene_key = _scene_key(scene_id)
    relevant_boxes = SceneSemanticBoxes.from_scene_id(scene_key).relevant_to(
        "",
        [start_position, start_position],
        start_rotation_to_direction_vector(start_rotation),
    )
    start_x, start_z = relevant_boxes.trajectory_keypoints[0]
    start_grid_position = meters_to_grid(float(start_x), float(start_z))
    return {
        "start_direction_vector": torch.tensor(
            start_rotation_to_direction_vector(start_rotation),
            dtype=torch.float32,
        ),
        "start_position": torch.tensor(start_grid_position, dtype=torch.float32),
    }


def start_metadata_for_episode(episode):
    return start_metadata_to_tensors(
        episode.scene_id,
        episode.start_position,
        episode.start_rotation,
    )
