import os
from typing import List

import torch
from prior.constants import (
    COLS,
    DIRECTION_VECTOR_CNT as _DIRECTION_VECTOR_CNT,
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    ROWS,
)
from prior.directions import start_rotation_to_direction_vector
from prior.bbox import SceneSemanticBoxes
from prior.grid_map import CognitiveGridMap

NUM_MAP_CATEGORIES = len(MAPPED_OBJECT_NAMES) + len(MAPPED_REGION_NAMES)
DIRECTION_VECTOR_CNT = _DIRECTION_VECTOR_CNT

SIZE = ROWS
"""Number of rows and cols in the grid map."""

if ROWS != COLS:
    raise ValueError(f"PriorGT map encoder requires square maps, got {ROWS}x{COLS}")


def _scene_key(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(scene_id))[0]


def _instruction_text(episode) -> str:
    instruction = episode.instruction
    return getattr(instruction, "instruction_text", instruction)


def cognitive_map_to_tensors(cognitive_map: CognitiveGridMap):
    return {
        "grid": torch.from_numpy(cognitive_map.grid),
        "direction_vectors": torch.tensor(
            cognitive_map.direction_vectors, dtype=torch.float32
        ),
        "start_direction_vector": torch.tensor(
            cognitive_map.start_direction_vector, dtype=torch.float32
        ),
        "start_position": torch.tensor(cognitive_map.positions[0], dtype=torch.float32),
    }


def build_cognitive_map(
    scene_id: str,
    instruction: str,
    reference_path: List[List[float]],
    start_direction_vector,
) -> CognitiveGridMap:
    return SceneSemanticBoxes.from_scene_id(_scene_key(scene_id)).to_cognitive_map(
        instruction,
        reference_path,
        start_direction_vector=start_direction_vector,
    )


def start_metadata_to_tensors(scene_id: str, start_position, start_rotation):
    """Build inference-safe start metadata without using the reference path."""
    scene_key = _scene_key(scene_id)
    _, start_level = SceneSemanticBoxes.from_scene_id(
        scene_key
    ).first_encountered_level(
        [start_position],
    )
    start_x, _, start_z = start_position
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


def build_cognitive_map_for_episode(episode) -> CognitiveGridMap:
    return build_cognitive_map(
        episode.scene_id,
        _instruction_text(episode),
        episode.reference_path,
        start_rotation_to_direction_vector(episode.start_rotation),
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
