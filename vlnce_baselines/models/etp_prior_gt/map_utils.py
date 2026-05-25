from __future__ import annotations

import os
import random
from typing import List

import torch
from prior.constants import (
    COLS,
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    REFERENCE_PATH_LENGTH as _REFERENCE_PATH_LENGTH,
    ROWS,
)
from prior.directions import start_rotation_to_direction_vector
from prior.bbox import SceneSemanticBoxes
from prior.grid_map import CognitiveGridMap

NUM_MAP_CATEGORIES = len(MAPPED_OBJECT_NAMES) + len(MAPPED_REGION_NAMES)
REFERENCE_PATH_LENGTH = _REFERENCE_PATH_LENGTH

SIZE = ROWS
"""Number of rows and cols in the grid map."""

if ROWS != COLS:
    raise ValueError(f"PriorGT map encoder requires square maps, got {ROWS}x{COLS}")


def _scene_key(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(scene_id))[0]


def _instruction_text(episode) -> str:
    instruction = episode.instruction
    return getattr(instruction, "instruction_text", instruction)


def _reference_path_to_grid_tensor(cognitive_map: CognitiveGridMap) -> torch.Tensor:
    reference_path = torch.zeros(REFERENCE_PATH_LENGTH, 2, dtype=torch.float32)
    for idx, position in enumerate(
        cognitive_map.reference_path[:REFERENCE_PATH_LENGTH]
    ):
        row, col = cognitive_map.world_to_grid(float(position[0]), float(position[1]))
        reference_path[idx] = torch.tensor([row, col], dtype=torch.float32)
    return reference_path


def _start_position_tensor(cognitive_map: CognitiveGridMap) -> torch.Tensor:
    if not cognitive_map.reference_path:
        raise ValueError("CognitiveGridMap.reference_path is empty")
    start_x, start_z = cognitive_map.reference_path[0]
    return torch.tensor(
        cognitive_map.world_to_grid(float(start_x), float(start_z)),
        dtype=torch.float32,
    )


def cognitive_map_to_tensors(cognitive_map: CognitiveGridMap):
    return {
        "grid": torch.from_numpy(cognitive_map.grid),
        "reference_paths": _reference_path_to_grid_tensor(cognitive_map),
        "start_direction_vector": torch.tensor(
            cognitive_map.start_direction_vector, dtype=torch.float32
        ),
        "start_position": _start_position_tensor(cognitive_map),
    }


def build_cognitive_map(
    scene_id: str,
    instruction: str,
    reference_path: List[List[float]],
    start_direction_vector,
    rotation_augmentation: int | None = None,
) -> CognitiveGridMap:
    relevant_boxes = SceneSemanticBoxes.from_scene_id(_scene_key(scene_id)).relevant_to(
        instruction,
        reference_path,
        start_direction_vector=start_direction_vector,
    )
    if rotation_augmentation is not None:
        relevant_boxes = relevant_boxes.rotate_by_right_angle(rotation_augmentation)
    return relevant_boxes.to_cognitive_map()


def start_metadata_to_tensors(scene_id: str, start_position, start_rotation):
    """Build inference-safe start metadata without using the reference path."""
    scene_key = _scene_key(scene_id)
    relevant_boxes = SceneSemanticBoxes.from_scene_id(scene_key).relevant_to(
        "",
        [start_position],
        start_rotation_to_direction_vector(start_rotation),
    )
    start_level = relevant_boxes.level
    start_x, start_z = relevant_boxes.reference_path[0]
    start_grid_position = start_level.world_to_grid(float(start_x), float(start_z))
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


def build_cognitive_map_for_episode(
    episode,
    random_rotation_augmentation: bool = False,
) -> CognitiveGridMap:
    return build_cognitive_map(
        episode.scene_id,
        _instruction_text(episode),
        episode.reference_path,
        start_rotation_to_direction_vector(episode.start_rotation),
        rotation_augmentation=(
            random.randrange(4) if random_rotation_augmentation else None
        ),
    )


def build_cognitive_map_for_annotation(
    annotation,
    connectivity_dir=None,
) -> CognitiveGridMap:
    positions = (
        annotation.positions()
        if connectivity_dir is None
        else annotation.positions(connectivity_dir)
    )
    return build_cognitive_map(
        annotation.scan,
        annotation.instruction,
        positions,
        annotation.start_direction_vector,
    )
