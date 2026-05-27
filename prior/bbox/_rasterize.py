"""Rasterization from semantic boxes into cognitive grid layers."""

from __future__ import annotations

import math

import numpy as np

from .._coords import grid_to_meters
from ..constants import CELL_SIZE, COLS, OBJECT_CATEGORIES, ROWS
from ._geometry import _point_to_obb_distance
from ._types import AABB2D, IRRELEVANT_MULTIPLIER, LevelSemanticBoxes, OBB2D


def _grid_center_index_range(
    min_coord: float,
    max_coord: float,
    limit: int,
) -> tuple[int, int] | None:
    start = math.ceil((min_coord - CELL_SIZE / 2.0) / CELL_SIZE)
    end = math.floor((max_coord - CELL_SIZE / 2.0) / CELL_SIZE)
    if end < 0 or start >= limit:
        return None
    return max(0, start), min(limit - 1, end)


def _rasterize_aabb(
    box: AABB2D,
    level: LevelSemanticBoxes,
    layer_idx: int,
    grid: np.ndarray,
    confidence: float,
) -> None:
    row_range = _grid_center_index_range(box.min[0], box.max[0], ROWS)
    col_range = _grid_center_index_range(box.min[1], box.max[1], COLS)
    if row_range is None or col_range is None:
        return
    row_start, row_end = row_range
    col_start, col_end = col_range
    grid[layer_idx, row_start : row_end + 1, col_start : col_end + 1] = np.maximum(
        grid[layer_idx, row_start : row_end + 1, col_start : col_end + 1],
        confidence,
    )


def _obb_bounds(box: OBB2D) -> tuple[float, float, float, float]:
    cos_theta = math.cos(box.rotation)
    sin_theta = math.sin(box.rotation)
    corners = []
    for sign_x in (-1.0, 1.0):
        for sign_z in (-1.0, 1.0):
            local_x = sign_x * box.half_extents[0]
            local_z = sign_z * box.half_extents[1]
            corners.append(
                (
                    box.center[0] + local_x * cos_theta - local_z * sin_theta,
                    box.center[1] + local_x * sin_theta + local_z * cos_theta,
                )
            )
    xs = [corner[0] for corner in corners]
    zs = [corner[1] for corner in corners]
    return (
        min(xs),
        max(xs),
        min(zs),
        max(zs),
    )


def _rasterize_obb(
    box: OBB2D,
    level: LevelSemanticBoxes,
    layer_idx: int,
    grid: np.ndarray,
    confidence: float,
) -> None:
    min_x, max_x, min_z, max_z = _obb_bounds(box)
    row_range = _grid_center_index_range(min_x, max_x, ROWS)
    col_range = _grid_center_index_range(min_z, max_z, COLS)
    if row_range is None or col_range is None:
        return

    row_start, row_end = row_range
    col_start, col_end = col_range
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            point = grid_to_meters(row, col)
            if _point_to_obb_distance(point, box) <= 1e-6:
                grid[layer_idx, row, col] = max(grid[layer_idx, row, col], confidence)


def _rasterize_level_semantic_boxes(
    level: LevelSemanticBoxes,
    grid: np.ndarray,
) -> None:
    for category_idx, boxes in enumerate(level.objects):
        for box in boxes:
            confidence = 1.0 if box.mentioned else IRRELEVANT_MULTIPLIER
            _rasterize_obb(box, level, category_idx, grid, confidence)

    for category_idx, boxes in enumerate(level.regions):
        layer_idx = OBJECT_CATEGORIES + category_idx
        for box in boxes:
            confidence = 1.0 if box.mentioned else IRRELEVANT_MULTIPLIER
            _rasterize_aabb(box, level, layer_idx, grid, confidence)
