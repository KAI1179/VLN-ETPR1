"""Direction vector helpers using the project visualization convention."""

from __future__ import annotations

from math import cos, sin
from typing import Sequence, Tuple

from magnum import Quaternion, Vector2, Vector3


DirectionVector = Tuple[float, float]
FRONT = Vector3(0.0, 0.0, -1.0)


def normalize_direction_vector(
    sin_value: float,
    cos_value: float,
) -> DirectionVector:
    """Normalize a direction vector represented as (sin, cos)."""
    direction = Vector2(sin_value, cos_value).normalized()
    return float(direction.x), float(direction.y)


def world_delta_to_direction_vector(dx: float, dz: float) -> DirectionVector:
    """Convert world-space x/z delta to normalized (sin, cos)."""
    # Grid rows increase with world x and grid cols increase with world z.
    # For visualization convention, "up" and "right" mean decreasing row/col.
    return normalize_direction_vector(-dz, -dx)


def heading_to_direction_vector(heading: float) -> DirectionVector:
    """Convert ETP-R1/Matterport heading to normalized (sin, cos)."""
    return cos(heading), -sin(heading)


def start_rotation_to_direction_vector(
    start_rotation: Sequence[float],
) -> DirectionVector:
    """Convert Habitat start_rotation quaternion coefficients to (sin, cos)."""

    quaternion = Quaternion(
        Vector3(
            float(start_rotation[0]),
            float(start_rotation[1]),
            float(start_rotation[2]),
        ),
        float(start_rotation[3]),
    ).normalized()
    forward = quaternion.transform_vector(FRONT)
    return world_delta_to_direction_vector(float(forward[0]), float(forward[2]))


__all__ = [
    "DirectionVector",
    "heading_to_direction_vector",
    "normalize_direction_vector",
    "start_rotation_to_direction_vector",
    "world_delta_to_direction_vector",
]
