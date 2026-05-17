import os
from typing import List

import torch
from prior.constants import (
    COLS,
    DIRECTION_VECTOR_CNT,
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    ROWS,
)
from prior.directions import start_rotation_to_direction_vector
from prior.grid_map import CognitiveGridMap, GroundTruthGridMap

NUM_MAP_CATEGORIES = len(MAPPED_OBJECT_NAMES) + len(MAPPED_REGION_NAMES)

SIZE = ROWS
"""Number of rows and cols in the grid map."""

if ROWS != COLS:
    raise ValueError(f"PriorGT map encoder requires square maps, got {ROWS}x{COLS}")

def _scene_key(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(scene_id))[0]


def _instruction_text(episode) -> str:
    instruction = episode.instruction
    return getattr(instruction, "instruction_text", instruction)


def _contains_y(gt_map: GroundTruthGridMap, y: float) -> bool:
    lower, upper = gt_map.range_y
    if lower is not None and y < lower:
        return False
    if upper is not None and y >= upper:
        return False
    return True


def _first_encountered_level(
    gt_maps: List[GroundTruthGridMap],
    positions: List[List[float]],
) -> GroundTruthGridMap:
    for position in positions:
        y = float(position[1])
        for gt_map in gt_maps:
            if _contains_y(gt_map, y):
                return gt_map
    if len(gt_maps) == 0:
        raise ValueError("GroundTruthGridMap.from_scene_id returned no levels")
    return gt_maps[0]


def cognitive_map_to_tensors(cognitive_map: CognitiveGridMap):
    return {
        "grid": torch.from_numpy(cognitive_map.grid),
        "direction_vectors": torch.tensor(cognitive_map.direction_vectors, dtype=torch.float32),
        "start_direction_vector": torch.tensor(
            cognitive_map.start_direction_vector, dtype=torch.float32
        ),
        "start_position": torch.tensor(cognitive_map.positions[0], dtype=torch.float32),
    }


def build_cognitive_map(
    scene_id: str,
    instruction: str,
    positions: List[List[float]],
    start_direction_vector,
) -> CognitiveGridMap:
    gt_map = _first_encountered_level(
        GroundTruthGridMap.from_scene_id(_scene_key(scene_id)),
        positions,
    )
    return gt_map.to_cognitive_map(
        instruction,
        positions,
        start_direction_vector=start_direction_vector,
    )


def build_cognitive_map_for_episode(episode) -> CognitiveGridMap:
    return build_cognitive_map(
        episode.scene_id,
        _instruction_text(episode),
        episode.reference_path,
        start_rotation_to_direction_vector(episode.start_rotation),
    )
