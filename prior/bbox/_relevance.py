"""Instruction/path relevance filtering for semantic boxes."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Optional, Set, Tuple

from prior.directions import DirectionVector
from prior.trajectory import select_trajectory_keypoints

from ..constants import CELL_SIZE, MAX_DISTANCE_CELLS
from ._geometry import _empty_level_boxes, _is_position_in_level, _near_any_position
from ._types import (
    LevelSemanticBoxes,
    Point2D,
    RelevantSemanticBoxes,
    SceneSemanticBoxes,
)


def _first_encountered_level(
    scene: SceneSemanticBoxes,
    ground_truth_trajectory: List[List[float]],
) -> tuple[int, LevelSemanticBoxes]:
    if not scene.levels:
        raise ValueError("SceneSemanticBoxes contains no levels")
    if not ground_truth_trajectory:
        raise ValueError("ground_truth_trajectory is empty")

    for position in ground_truth_trajectory:
        for level_idx, level in enumerate(scene.levels):
            if _is_position_in_level(position, level.range_y):
                return level_idx, level

    ranges = [level.range_y for level in scene.levels]
    raise ValueError(
        f"ground_truth_trajectory does not intersect any semantic level range: {ranges}"
    )


def _extract_relevant_semantic_boxes(
    scene: SceneSemanticBoxes,
    instruction: str,
    ground_truth_trajectory: List[List[float]],
    start_direction_vector: DirectionVector,
    max_distance: float = MAX_DISTANCE_CELLS * CELL_SIZE,
    category_extractor: Optional[Callable[[str], Tuple[Set[int], Set[int]]]] = None,
) -> RelevantSemanticBoxes:
    """Return selected-level boxes near the path, marking categories in text."""
    if category_extractor is None:
        from prior.grid_map._cognitive import extract_categories

        category_extractor = extract_categories

    level_idx, level = _first_encountered_level(scene, ground_truth_trajectory)
    origin = scene._level_origins[level_idx]
    mentioned_objects, mentioned_regions = category_extractor(instruction)

    relevant_level = _empty_level_boxes()
    relevant_level.range_y = list(level.range_y)
    level_points = [
        _local_point(position, origin)
        for position in ground_truth_trajectory
        if _is_position_in_level(position, level.range_y)
    ]
    trajectory_keypoints = select_trajectory_keypoints(level_points)

    for category_idx, boxes in enumerate(level.objects):
        mentioned = category_idx in mentioned_objects
        relevant_level.objects[category_idx] = [
            replace(box, mentioned=mentioned)
            for box in boxes
            if _near_any_position(box, level_points, max_distance)
        ]

    for category_idx, boxes in enumerate(level.regions):
        mentioned = category_idx in mentioned_regions
        relevant_level.regions[category_idx] = [
            replace(box, mentioned=mentioned)
            for box in boxes
            if _near_any_position(box, level_points, max_distance)
        ]

    return RelevantSemanticBoxes(
        level_idx=level_idx,
        level=relevant_level,
        instruction=instruction,
        ground_truth_trajectory=level_points,
        trajectory_keypoints=trajectory_keypoints,
        start_direction_vector=start_direction_vector,
    )


def _local_point(position: List[float], origin: Point2D) -> Point2D:
    return (float(position[0]) - origin[0], float(position[2]) - origin[1])
