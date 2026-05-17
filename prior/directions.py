"""Direction vector helpers using the project visualization convention."""

from __future__ import annotations

from math import cos, hypot, sin
from typing import Sequence, Tuple
import numpy as np
from habitat_sim.geo import FRONT
from habitat_sim.utils.common import quat_from_coeffs, quat_rotate_vector


DirectionVector = Tuple[float, float]


def normalize_direction_vector(
    sin_value: float,
    cos_value: float,
) -> DirectionVector:
    """Normalize a direction vector represented as (sin, cos)."""
    norm = hypot(sin_value, cos_value)
    return sin_value / norm, cos_value / norm


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

    quaternion = quat_from_coeffs(np.asarray(start_rotation, dtype=float))
    forward = quat_rotate_vector(quaternion, np.asarray(FRONT, dtype=float))
    return world_delta_to_direction_vector(float(forward[0]), float(forward[2]))


__all__ = [
    "DirectionVector",
    "heading_to_direction_vector",
    "normalize_direction_vector",
    "start_rotation_to_direction_vector",
    "world_delta_to_direction_vector",
]
