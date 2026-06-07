"""Ground-truth trajectory helpers."""

from math import hypot
from typing import List, Optional, Sequence, Tuple


Point2D = Tuple[float, float]

TRAJECTORY_KEYPOINT_COUNT = 5
TRAJECTORY_KEYPOINT_SIMILARITY_THRESHOLD = 0.8


def _segment_direction(
    start: Point2D, end: Point2D
) -> Optional[Point2D]:
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    norm = hypot(dx, dz)
    if norm == 0.0:
        return None
    return dx / norm, dz / norm


def select_trajectory_keypoints(
    points: Sequence[Point2D],
) -> List[Point2D]:
    """Keep start, abrupt turns, and final point; pad to fixed length."""
    if len(points) < 2:
        raise ValueError(
            "ground_truth_trajectory must contain at least 2 selected-level "
            "points for trajectory_keypoints"
        )

    keypoints = [(float(points[0][0]), float(points[0][1]))]
    baseline_direction = None

    for index in range(len(points) - 1):
        direction = _segment_direction(points[index], points[index + 1])
        if direction is None:
            continue

        if baseline_direction is None:
            baseline_direction = direction
        else:
            similarity = (
                direction[0] * baseline_direction[0]
                + direction[1] * baseline_direction[1]
            )
            if similarity < TRAJECTORY_KEYPOINT_SIMILARITY_THRESHOLD:
                turn_point = (
                    float(points[index][0]),
                    float(points[index][1]),
                )
                if turn_point != keypoints[-1]:
                    keypoints.append(turn_point)
                    baseline_direction = direction

        if len(keypoints) >= TRAJECTORY_KEYPOINT_COUNT - 1:
            break

    final_point = (float(points[-1][0]), float(points[-1][1]))
    if final_point != keypoints[-1]:
        keypoints.append(final_point)

    return keypoints + [(0.0, 0.0)] * (
        TRAJECTORY_KEYPOINT_COUNT - len(keypoints)
    )
