"""Geometry helpers for semantic bounding boxes."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from magnum import Vector3

from ..constants import OBJECT_CATEGORIES, REGION_CATEGORIES
from ._types import AABB2D, Axis2D, LevelSemanticBoxes, OBB2D, Point2D


def _aabb_min(aabb) -> Vector3:
    """Return AABB minimum corner."""
    return Vector3(aabb.center - (aabb.sizes / 2.0))


def _aabb_max(aabb) -> Vector3:
    """Return AABB maximum corner."""
    return Vector3(aabb.center + (aabb.sizes / 2.0))


def _xz(point) -> Point2D:
    """Project a 3D point/vector to its X/Z components."""
    return (float(point[0]), float(point[2]))


def _normalized_xz(vector) -> Axis2D:
    x, z = _xz(vector)
    norm = math.hypot(x, z)
    if norm == 0.0:
        return (1.0, 0.0)
    return (x / norm, z / norm)


def _obb_rotation_2d(obb) -> float:
    """Return projected local X-axis angle."""
    local_to_world = obb.local_to_world
    if hasattr(local_to_world, "transform_vector"):
        x_axis = _normalized_xz(local_to_world.transform_vector(Vector3(1.0, 0.0, 0.0)))
    else:
        # Habitat exposes OBB matrices as ndarray in some versions. Treat
        # column 0 as the local X basis vector in world coordinates.
        x_axis = _normalized_xz(
            (local_to_world[0][0], local_to_world[1][0], local_to_world[2][0])
        )
    return math.atan2(x_axis[1], x_axis[0])


def _empty_level_boxes() -> LevelSemanticBoxes:
    return LevelSemanticBoxes(
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
    """Return shortest 2D distance from a point to a rotated object box."""
    dx = point[0] - box.center[0]
    dz = point[1] - box.center[1]
    cos_theta = math.cos(box.rotation)
    sin_theta = math.sin(box.rotation)

    local_x = dx * cos_theta + dz * sin_theta
    local_z = -dx * sin_theta + dz * cos_theta
    clamped_x = min(max(local_x, -box.half_extents[0]), box.half_extents[0])
    clamped_z = min(max(local_z, -box.half_extents[1]), box.half_extents[1])

    closest_x = box.center[0] + clamped_x * cos_theta - clamped_z * sin_theta
    closest_z = box.center[1] + clamped_x * sin_theta + clamped_z * cos_theta
    return math.hypot(point[0] - closest_x, point[1] - closest_z)


def _is_position_in_level(
    position: Sequence[float], range_y: List[Optional[float]]
) -> bool:
    y = position[1]
    if range_y[0] is not None and y < range_y[0]:
        return False
    if range_y[1] is not None and y >= range_y[1]:
        return False
    return True


def _near_any_position(
    box: AABB2D | OBB2D,
    points: List[Point2D],
    max_distance: float,
) -> bool:
    if isinstance(box, OBB2D):
        return any(
            _point_to_obb_distance(point, box) <= max_distance for point in points
        )
    return any(_point_to_aabb_distance(point, box) <= max_distance for point in points)
