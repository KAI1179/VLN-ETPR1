"""Module for constructing grid maps from MP3D dataset. Should not be used directly; use provided class methods instead."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, cast
from functools import lru_cache

from habitat_sim.scene import (
    Mp3dObjectCategory,
    Mp3dRegionCategory,
    SemanticLevel,
    SemanticScene,
)
from magnum import Vector3

from prior import MP3D_DIR

from ..constants import (
    CELL_SIZE,
    OBJECT_CATEGORIES,
    HABITAT_MP3D_ROTATION_VECTOR,
    COLS,
    OBJECT_MAPPING,
    ROWS,
    REGION_MAPPING,
)

if TYPE_CHECKING:
    from . import GroundTruthGridMap


def _grid_center_index_range(
    min_coord: float,
    max_coord: float,
    offset: float,
    limit: int,
) -> tuple[int, int] | None:
    """Return inclusive index range whose cell centers fall within [min_coord, max_coord]."""
    start = math.ceil((min_coord - offset - CELL_SIZE / 2.0) / CELL_SIZE)
    end = math.floor((max_coord - offset - CELL_SIZE / 2.0) / CELL_SIZE)
    if end < 0 or start >= limit:
        return None
    return max(0, start), min(limit - 1, end)


@lru_cache(maxsize=100)
def construct_grid_maps_from_scene_id(scene_id: str) -> List[GroundTruthGridMap]:
    """Constructs grid maps from the given MP3D scene ID. The return value is cached to improve performance, but not copied, so DO NOT MUTATE it."""
    # Load scene
    scene_path = str(MP3D_DIR / scene_id / f"{scene_id}.house")
    semantic_scene = SemanticScene()
    SemanticScene.load_mp3d_house(
        scene_path, semantic_scene, HABITAT_MP3D_ROTATION_VECTOR
    )
    return construct_grid_maps_from_scene(semantic_scene)


def construct_grid_maps_from_scene(
    semantic_scene: SemanticScene,
) -> List[GroundTruthGridMap]:
    """Constructs grid maps from the given semantic scene.

    Each returned map covers one semantic level.  The ``range_y`` attribute of
    every map is set so that, when building a cognitive map, only waypoints that
    physically lie on that level contribute to it.

    Floor boundaries are derived from ``region.aabb.min.y``, which maps to the
    floor-surface Y after the MP3D coordinate rotation and is available even
    though ``region.floor_height`` / ``region.extrusion_height`` are not
    populated for the ``.house`` format.

    Algorithm
    ---------
    1. For each level, compute ``floor_y = min(region.aabb.min.y)`` across all
       its regions.  This is the lowest floor surface in that level and is
       always ≤ any waypoint Y on that level.
    2. Sort levels by ``floor_y`` (ascending) so that level ordering is
       consistent regardless of dataset ordering.
    3. Assign Y boundaries:
       - Level 0 (bottom):  ``range_y = [None,       floor_y[1])``
       - Level i (middle):  ``range_y = [floor_y[i], floor_y[i+1])``
       - Level n (top):     ``range_y = [floor_y[n], None)``
       - Single level:      ``range_y = [None,       None)``
    """
    if not semantic_scene.levels:
        return []

    # Compute the representative floor Y for each level.
    # region.aabb.min.y reliably gives the floor surface height because the
    # MP3D .house loader sets the AABB (but NOT floor_height/extrusion_height).
    # level.aabb.min.y is unreliable for some scenes (identical across levels).
    def _level_floor_y(level) -> float:
        if level.regions:
            return min(region.aabb.min.y for region in level.regions)
        # Fallback: use the level AABB when no regions exist
        return float(level.aabb.min.y)

    # Build (floor_y, semantic_level) pairs and sort bottom-to-top
    level_pairs = sorted(
        ((float(_level_floor_y(lvl)), lvl) for lvl in semantic_scene.levels),
        key=lambda pair: pair[0],
    )

    grid_maps: List[GroundTruthGridMap] = []
    floor_ys = [fy for fy, _ in level_pairs]

    for i, (floor_y, semantic_level) in enumerate(level_pairs):
        grid_map = construct_grid_map_from_level(semantic_level)

        # range_y lower bound: None for the bottom level (accept everything
        # below the next floor), otherwise this level's own floor_y.
        lower = None if i == 0 else floor_ys[i]

        # range_y upper bound: next level's floor_y, or None for the top level.
        upper = floor_ys[i + 1] if i + 1 < len(floor_ys) else None

        grid_map.range_y = [lower, upper]
        grid_maps.append(grid_map)

    return grid_maps


def construct_grid_map_from_level(
    semantic_level: SemanticLevel,
) -> GroundTruthGridMap:
    """Constructs a grid map from the given semantic level."""
    # Import at runtime to avoid circular import
    from . import GroundTruthGridMap

    gt_grid_map = GroundTruthGridMap()

    # Calculate the offset to transform world coordinates to grid coordinates
    # The level's AABB defines the world space bounds
    level_aabb = semantic_level.aabb
    offset_x = level_aabb.min.x
    offset_z = level_aabb.min.z
    gt_grid_map.offset_x = offset_x
    gt_grid_map.offset_z = offset_z

    # Pre-compute cell-center world coordinates once for this level.
    x_coords_list = [offset_x + (r + 0.5) * CELL_SIZE for r in range(ROWS)]
    z_coords_list = [offset_z + (c + 0.5) * CELL_SIZE for c in range(COLS)]

    # Read regions
    for region in semantic_level.regions:
        assert isinstance(
            region.category, Mp3dRegionCategory
        ), "Region category is not Mp3dRegionCategory"
        category = cast(Mp3dRegionCategory, region.category)
        mapped_category = REGION_MAPPING[category.index()]

        # Set region value to 1 of mapped category in grid map.
        aabb_min = region.aabb.min
        aabb_max = region.aabb.max

        # Fill the exact index window whose cell centers lie in the region AABB.
        row_range = _grid_center_index_range(aabb_min.x, aabb_max.x, offset_x, ROWS)
        col_range = _grid_center_index_range(aabb_min.z, aabb_max.z, offset_z, COLS)
        if row_range is not None and col_range is not None:
            row_start, row_end = row_range
            col_start, col_end = col_range
            gt_grid_map.grid[
                OBJECT_CATEGORIES + mapped_category,
                row_start : row_end + 1,
                col_start : col_end + 1,
            ] = 1

        # Read objects
        # NOTE: We do not access semantic_level.objects, since its always empty. Instead, semantic_scene.objects and region.objects work correctly.
        for obj in region.objects:
            assert isinstance(
                obj.category, Mp3dObjectCategory
            ), "Object category is not Mp3dObjectCategory"
            category = cast(Mp3dObjectCategory, obj.category)
            mapped_category = OBJECT_MAPPING[category.index()]

            # Set object value to 1 of mapped category in grid map.
            aabb_min = obj.aabb.min
            aabb_max = obj.aabb.max

            row_range = _grid_center_index_range(aabb_min.x, aabb_max.x, offset_x, ROWS)
            col_range = _grid_center_index_range(aabb_min.z, aabb_max.z, offset_z, COLS)
            if row_range is None or col_range is None:
                continue

            row_start, row_end = row_range
            col_start, col_end = col_range

            # Use the middle Y coordinate of the object's AABB for testing
            test_y = (aabb_min.y + aabb_max.y) / 2.0

            # For all grid cells that overlap with the object's AABB, check OBB containment
            for row in range(row_start, row_end + 1):
                x = x_coords_list[row]
                for col in range(col_start, col_end + 1):
                    z = z_coords_list[col]

                    # Check if cell center is within the object's OBB (more accurate than AABB)
                    point = Vector3(x, test_y, z)
                    if obj.obb.contains(point, 0.0):
                        gt_grid_map.grid[mapped_category, row, col] = 1

    return gt_grid_map
