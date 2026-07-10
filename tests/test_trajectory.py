from math import cos, radians, sin

import pytest

from prior.trajectory import (
    TRAJECTORY_KEYPOINT_COUNT,
    select_trajectory_keypoints,
)


def test_select_trajectory_keypoints_keeps_start_abrupt_turn_and_final():
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
    ]

    assert select_trajectory_keypoints(points) == [
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]


def test_select_trajectory_keypoints_straight_path_keeps_start_and_final():
    assert select_trajectory_keypoints([(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)]) == [
        (0.0, 0.0),
        (3.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]


def test_select_trajectory_keypoints_detects_cumulative_gradual_turn():
    directions = [
        (cos(radians(angle)), sin(radians(angle))) for angle in (0.0, 25.0, 50.0)
    ]
    points = [(0.0, 0.0)]
    for dx, dz in directions:
        x, z = points[-1]
        points.append((x + dx, z + dz))

    assert select_trajectory_keypoints(points) == [
        points[0],
        points[2],
        points[3],
        (0.0, 0.0),
        (0.0, 0.0),
    ]


def test_select_trajectory_keypoints_truncates_turns_but_keeps_final():
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (3.0, 2.0),
        (3.0, 3.0),
    ]

    assert select_trajectory_keypoints(points) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (3.0, 3.0),
    ]


def test_select_trajectory_keypoints_rejects_too_short_path():
    with pytest.raises(ValueError, match="at least 2 selected-level points"):
        select_trajectory_keypoints([(0.0, 0.0)])


def test_keypoint_count_is_five():
    assert TRAJECTORY_KEYPOINT_COUNT == 5
