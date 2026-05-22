"""2D semantic bounding boxes from MP3D scenes."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Callable, List, Optional, Set, Tuple, cast

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
    HABITAT_MP3D_ROTATION_VECTOR,
    MAX_DISTANCE_CELLS,
    OBJECT_CATEGORIES,
    OBJECT_MAPPING,
    REGION_CATEGORIES,
    REGION_MAPPING,
)

Point2D = Tuple[float, float]
Axis2D = Tuple[float, float]


@dataclass(frozen=True)
class OBB2D:
    """2D oriented bounding box projected onto world X/Z axes."""

    id: str
    center: Point2D
    half_extents: Point2D
    axes: Tuple[Axis2D, Axis2D]
    mentioned: bool = False


@dataclass(frozen=True)
class AABB2D:
    """2D axis-aligned bounding box projected onto world X/Z axes."""

    id: str
    min: Point2D
    max: Point2D
    mentioned: bool = False


ObjectOBB2Ds = List[List[OBB2D]]
"""List indexed by mapped object category id, containing object OBBs of that type."""

RegionAABB2Ds = List[List[AABB2D]]
"""List indexed by mapped region category id, containing region AABBs of that type."""


@dataclass
class LevelBoundingBoxes:
    """2D semantic boxes for one MP3D semantic level."""

    objects: ObjectOBB2Ds
    regions: RegionAABB2Ds
    range_y: List[Optional[float]]


def _aabb_min(aabb) -> Vector3:
    """Return AABB minimum corner."""
    return Vector3(aabb.center - (aabb.sizes / 2.0))


def _aabb_max(aabb) -> Vector3:
    """Return AABB maximum corner."""
    return Vector3(aabb.center + (aabb.sizes / 2.0))


def _xz(point) -> Point2D:
    """Project a 3D point/vector to 2D world X/Z coordinates."""
    return (float(point[0]), float(point[2]))


def _normalized_xz(vector) -> Axis2D:
    x, z = _xz(vector)
    norm = math.hypot(x, z)
    if norm == 0.0:
        return (0.0, 0.0)
    return (x / norm, z / norm)


def _obb_axes_2d(obb) -> Tuple[Axis2D, Axis2D]:
    """Return projected local X and local Z axes for a 3D OBB."""
    local_to_world = obb.local_to_world
    if hasattr(local_to_world, "transform_vector"):
        x_axis = local_to_world.transform_vector(Vector3(1.0, 0.0, 0.0))
        z_axis = local_to_world.transform_vector(Vector3(0.0, 0.0, 1.0))
        return (_normalized_xz(x_axis), _normalized_xz(z_axis))

    # Habitat exposes OBB matrices as ndarray in some versions.  Treat columns
    # 0 and 2 as local X/Z basis vectors in world coordinates.
    x_axis = (local_to_world[0][0], local_to_world[1][0], local_to_world[2][0])
    z_axis = (local_to_world[0][2], local_to_world[1][2], local_to_world[2][2])
    return (_normalized_xz(x_axis), _normalized_xz(z_axis))


def _empty_level_boxes() -> LevelBoundingBoxes:
    return LevelBoundingBoxes(
        objects=[[] for _ in range(OBJECT_CATEGORIES)],
        regions=[[] for _ in range(REGION_CATEGORIES)],
        range_y=[None, None],
    )


def _point_to_aabb_distance(point: Point2D, box: AABB2D) -> float:
    """Return shortest 2D distance from a point to an axis-aligned box."""
    px, pz = point
    dx = max(box.min[0] - px, 0.0, px - box.max[0])
    dz = max(box.min[1] - pz, 0.0, pz - box.max[1])
    return math.hypot(dx, dz)


def _point_to_obb_distance(point: Point2D, box: OBB2D) -> float:
    """Return shortest 2D distance from a point to an oriented box."""
    dx = point[0] - box.center[0]
    dz = point[1] - box.center[1]
    axis_x, axis_z = box.axes

    local_x = dx * axis_x[0] + dz * axis_x[1]
    local_z = dx * axis_z[0] + dz * axis_z[1]
    clamped_x = min(max(local_x, -box.half_extents[0]), box.half_extents[0])
    clamped_z = min(max(local_z, -box.half_extents[1]), box.half_extents[1])

    closest_x = box.center[0] + clamped_x * axis_x[0] + clamped_z * axis_z[0]
    closest_z = box.center[1] + clamped_x * axis_x[1] + clamped_z * axis_z[1]
    return math.hypot(point[0] - closest_x, point[1] - closest_z)


def _is_position_in_level(
    position: List[float], range_y: List[Optional[float]]
) -> bool:
    y = position[1]
    if range_y[0] is not None and y < range_y[0]:
        return False
    if range_y[1] is not None and y >= range_y[1]:
        return False
    return True


def _near_any_position(
    box,
    points: List[Point2D],
    max_distance: float,
) -> bool:
    if isinstance(box, OBB2D):
        distance_func = _point_to_obb_distance
    else:
        distance_func = _point_to_aabb_distance
    return any(distance_func(point, box) <= max_distance for point in points)


def extract_relevant_bounding_boxes(
    levels: List[LevelBoundingBoxes],
    instruction: str,
    positions: List[List[float]],
    max_distance: float = MAX_DISTANCE_CELLS * CELL_SIZE,
    category_extractor: Optional[Callable[[str], Tuple[Set[int], Set[int]]]] = None,
) -> List[LevelBoundingBoxes]:
    """Return boxes close to the trajectory, marking categories mentioned in text."""
    if category_extractor is None:
        from prior.grid_map._cognitive import extract_categories

        category_extractor = extract_categories

    mentioned_objects, mentioned_regions = category_extractor(instruction)
    relevant_levels: List[LevelBoundingBoxes] = []

    for level in levels:
        relevant_level = _empty_level_boxes()
        relevant_level.range_y = list(level.range_y)
        level_points = [
            (float(position[0]), float(position[2]))
            for position in positions
            if _is_position_in_level(position, level.range_y)
        ]
        if not level_points:
            relevant_levels.append(relevant_level)
            continue

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

        relevant_levels.append(relevant_level)

    return relevant_levels


@lru_cache(maxsize=100)
def construct_bounding_boxes_from_scene_id(scene_id: str) -> List[LevelBoundingBoxes]:
    """Construct level-wise 2D bounding boxes from the given MP3D scene ID.

    The cached return value is not copied, so callers should prefer
    ``bounding_boxes_from_scene_id`` unless they will not mutate the result.
    """
    scene_path = str(MP3D_DIR / scene_id / f"{scene_id}.house")
    semantic_scene = SemanticScene()
    SemanticScene.load_mp3d_house(
        scene_path, semantic_scene, cast(Any, HABITAT_MP3D_ROTATION_VECTOR)
    )
    return construct_bounding_boxes_from_scene(semantic_scene)


def bounding_boxes_from_scene_id(scene_id: str) -> List[LevelBoundingBoxes]:
    """Return a mutable-safe copy of 2D boxes for each level in a scene."""
    return deepcopy(construct_bounding_boxes_from_scene_id(scene_id))


def construct_bounding_boxes_from_scene(
    semantic_scene: SemanticScene,
) -> List[LevelBoundingBoxes]:
    """Construct level-wise 2D semantic boxes from a loaded semantic scene."""
    if not semantic_scene.levels:
        return []

    def _level_floor_y(level) -> float:
        if level.regions:
            return min(_aabb_min(region.aabb).y for region in level.regions)
        return float(_aabb_min(level.aabb).y)

    level_pairs = sorted(
        ((float(_level_floor_y(lvl)), lvl) for lvl in semantic_scene.levels),
        key=lambda pair: pair[0],
    )

    floor_ys = [fy for fy, _ in level_pairs]
    level_boxes: List[LevelBoundingBoxes] = []

    for i, (_, semantic_level) in enumerate(level_pairs):
        boxes = construct_bounding_boxes_from_level(semantic_level)
        boxes.range_y = [
            None if i == 0 else floor_ys[i],
            floor_ys[i + 1] if i + 1 < len(floor_ys) else None,
        ]
        level_boxes.append(boxes)

    return level_boxes


def construct_bounding_boxes_from_level(
    semantic_level: SemanticLevel,
) -> LevelBoundingBoxes:
    """Construct 2D semantic boxes from one semantic level."""
    boxes = _empty_level_boxes()

    for region in semantic_level.regions:
        assert isinstance(region.category, Mp3dRegionCategory), (
            "Region category is not Mp3dRegionCategory"
        )
        mapped_region = REGION_MAPPING[region.category.index()]
        aabb_min = _aabb_min(region.aabb)
        aabb_max = _aabb_max(region.aabb)
        boxes.regions[mapped_region].append(
            AABB2D(id=region.id, min=_xz(aabb_min), max=_xz(aabb_max))
        )

        # semantic_level.objects is always empty for MP3D .house scenes; region
        # objects are populated and match the grid-map construction path.
        for obj in region.objects:
            assert isinstance(obj.category, Mp3dObjectCategory), (
                "Object category is not Mp3dObjectCategory"
            )
            mapped_object = OBJECT_MAPPING[obj.category.index()]
            obb = obj.obb
            boxes.objects[mapped_object].append(
                OBB2D(
                    id=obj.id,
                    center=_xz(obb.center),
                    half_extents=(
                        float(obb.half_extents[0]),
                        float(obb.half_extents[2]),
                    ),
                    axes=_obb_axes_2d(obb),
                )
            )

    return boxes


__all__ = [
    "AABB2D",
    "LevelBoundingBoxes",
    "OBB2D",
    "bounding_boxes_from_scene_id",
    "construct_bounding_boxes_from_level",
    "construct_bounding_boxes_from_scene",
    "construct_bounding_boxes_from_scene_id",
    "extract_relevant_bounding_boxes",
]
