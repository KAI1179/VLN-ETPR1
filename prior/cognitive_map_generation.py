"""Shared cognitive-map generation helpers."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from prior.bbox import LevelSemanticBoxes, RelevantSemanticBoxes, SceneSemanticBoxes
from prior.bbox._rasterize import _rasterize_level_semantic_boxes
from prior.bbox._relevance import _first_touched_level_points
from prior.constants import (
    CELL_SIZE,
    COLS,
    MAX_DISTANCE_CELLS,
    OBJECT_CATEGORIES,
    REGION_CATEGORIES,
    ROWS,
)
from prior.directions import DirectionVector, world_delta_to_direction_vector
from prior.grid_map import CognitiveGridMap
from prior.grid_map._cognitive import extract_categories
from prior.trajectory import (
    TRAJECTORY_KEYPOINT_COUNT,
    TRAJECTORY_KEYPOINT_SIMILARITY_THRESHOLD,
    WorldTrajectory3D,
    select_trajectory_keypoints,
)

DEFAULT_RADIUS_M = MAX_DISTANCE_CELLS * CELL_SIZE
MapSource = Literal["bbox", "legacy"]
MetadataSchema = Literal["path5", "direction5"]


def _radius_label(radius_m: float) -> str:
    return f"{radius_m:g}".replace(".", "p")


def map_cache_namespace(
    map_source: MapSource,
    radius_m: float,
    metadata_schema: MetadataSchema = "path5",
) -> str:
    return f"gt.{map_source}.r{_radius_label(radius_m)}.{metadata_schema}.v1"


def cache_root(output_dir: Path, namespace: str) -> Path:
    return output_dir / namespace


def cache_paths(
    output_dir: Path,
    namespace: str,
    scene_id: str,
    episode_key: str,
) -> tuple[Path, Path]:
    root = cache_root(output_dir, namespace)
    return (
        root / "boxes" / scene_id / f"{episode_key}.npz",
        root / "raster" / scene_id / f"{episode_key}.npz",
    )


def build_cognitive_map(
    scene_boxes: SceneSemanticBoxes,
    instruction: str,
    ground_truth_trajectory: WorldTrajectory3D,
    start_direction_vector: DirectionVector,
    map_source: MapSource,
    radius_m: float,
) -> tuple[RelevantSemanticBoxes, CognitiveGridMap]:
    relevant_boxes = scene_boxes.relevant_to(
        instruction,
        ground_truth_trajectory,
        start_direction_vector,
        max_distance=radius_m,
    )
    if map_source == "bbox":
        return relevant_boxes, relevant_boxes.to_cognitive_map()
    return (
        relevant_boxes,
        legacy_cognitive_map(
            scene_boxes,
            instruction,
            ground_truth_trajectory,
            start_direction_vector,
            radius_m,
        ),
    )


def _direction_vector_between(
    prev_position: tuple[float, float, float],
    next_position: tuple[float, float, float],
) -> Optional[DirectionVector]:
    dx = float(next_position[0]) - float(prev_position[0])
    dz = float(next_position[2]) - float(prev_position[2])
    if dx == 0.0 and dz == 0.0:
        return None
    return world_delta_to_direction_vector(dx, dz)


def trajectory_direction_vectors(
    positions: WorldTrajectory3D,
) -> list[DirectionVector]:
    directions: list[DirectionVector] = []
    for index in range(len(positions) - 1):
        direction = _direction_vector_between(positions[index], positions[index + 1])
        if direction is None:
            continue
        if directions:
            prev = directions[-1]
            similarity = direction[0] * prev[0] + direction[1] * prev[1]
            if similarity >= TRAJECTORY_KEYPOINT_SIMILARITY_THRESHOLD:
                continue
        directions.append(direction)
        if len(directions) >= TRAJECTORY_KEYPOINT_COUNT:
            break
    return directions + [(0.0, 0.0)] * (TRAJECTORY_KEYPOINT_COUNT - len(directions))


def save_cognitive_map(
    cognitive_map: CognitiveGridMap,
    save_path: Path,
    metadata_schema: MetadataSchema,
    ground_truth_trajectory: WorldTrajectory3D,
) -> None:
    if metadata_schema == "path5":
        cognitive_map.save(save_path)
        return
    if not ground_truth_trajectory:
        raise ValueError("ground_truth_trajectory is required for direction5 metadata")
    if not cognitive_map.trajectory_keypoints:
        raise ValueError("trajectory_keypoints are required for direction5 metadata")
    np.savez_compressed(
        save_path,
        grid=cognitive_map.grid,
        range_y=np.asarray(cognitive_map.range_y, dtype=object),
        direction_vectors=np.asarray(
            trajectory_direction_vectors(ground_truth_trajectory),
            dtype=np.float32,
        ),
        start_direction_vector=np.asarray(
            cognitive_map.start_direction_vector,
            dtype=np.float32,
        ),
        start_position=np.asarray(
            cognitive_map.trajectory_keypoints[0],
            dtype=np.float32,
        ),
    )


def _all_mentioned_level(level: LevelSemanticBoxes) -> LevelSemanticBoxes:
    return LevelSemanticBoxes(
        objects=[
            [replace(box, mentioned=True) for box in boxes]
            for boxes in level.objects
        ],
        regions=[
            [replace(box, mentioned=True) for box in boxes]
            for boxes in level.regions
        ],
        range_y=list(level.range_y),
    )


def _copy_legacy_path_neighborhood(
    source_grid: np.ndarray,
    cognitive_map: CognitiveGridMap,
    row: int,
    col: int,
    radius_cells: int,
    mentioned_objects: set[int],
    mentioned_regions: set[int],
) -> None:
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return

    up = max(0, row - radius_cells)
    left = max(0, col - radius_cells)
    down = min(ROWS - 1, row + radius_cells)
    right = min(COLS - 1, col + radius_cells)

    for layer_idx in range(OBJECT_CATEGORIES + REGION_CATEGORIES):
        source_slice = source_grid[layer_idx, up : down + 1, left : right + 1]
        if layer_idx < OBJECT_CATEGORIES:
            mentioned = layer_idx in mentioned_objects
        else:
            mentioned = (layer_idx - OBJECT_CATEGORIES) in mentioned_regions
        values = source_slice if mentioned else source_slice * 0.6
        target = cognitive_map.grid[layer_idx, up : down + 1, left : right + 1]
        np.maximum(target, values, out=target)


def legacy_cognitive_map(
    scene_boxes: SceneSemanticBoxes,
    instruction: str,
    ground_truth_trajectory: WorldTrajectory3D,
    start_direction_vector: DirectionVector,
    radius_m: float,
) -> CognitiveGridMap:
    _, level, level_points = _first_touched_level_points(
        scene_boxes,
        ground_truth_trajectory,
    )
    full_grid = np.zeros(
        (OBJECT_CATEGORIES + REGION_CATEGORIES, ROWS, COLS),
        dtype=np.float32,
    )
    _rasterize_level_semantic_boxes(_all_mentioned_level(level), full_grid)

    cognitive_map = CognitiveGridMap()
    cognitive_map.range_y = list(level.range_y)
    cognitive_map.trajectory_keypoints = select_trajectory_keypoints(level_points)
    cognitive_map.start_direction_vector = start_direction_vector

    mentioned_objects, mentioned_regions = extract_categories(instruction)
    radius_cells = max(0, int(round(radius_m / CELL_SIZE)))
    for x, z in level_points:
        row = math.floor(x / CELL_SIZE)
        col = math.floor(z / CELL_SIZE)
        _copy_legacy_path_neighborhood(
            full_grid,
            cognitive_map,
            row,
            col,
            radius_cells,
            mentioned_objects,
            mentioned_regions,
        )
    return cognitive_map
