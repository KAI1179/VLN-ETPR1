from __future__ import annotations

import numpy as np
import pytest

from prior.analyze.llm_grid_registration import (
    SpatialBounds,
    WarpedGrid,
    direction_cosine,
    rotate_direction_vectors,
    score_warped_grid,
    select_best_angle,
    warp_grid_about_pivot,
)
from prior.directions import world_delta_to_direction_vector


def test_identity_preserves_coordinates_and_support() -> None:
    """Breaks if the identity warp shifts cells or loses category support."""
    grid = np.zeros((2, 4, 5), dtype=np.bool_)
    grid[0, 0, 1] = True
    grid[1, 3, 4] = True

    warped = warp_grid_about_pivot(grid, (1.2, 2.7), 0.0)

    assert warped.bounds == SpatialBounds(0, 4, 0, 5)
    assert np.array_equal(warped.grid, grid)
    assert warped.input_support == 2
    assert warped.total_support == 2
    assert warped.grid is not grid


@pytest.mark.parametrize(
    ("grid", "pivot", "angle_degrees"),
    [
        (np.zeros((4, 5), dtype=np.bool_), (1.0, 2.0), 0.0),
        (np.zeros((1, 4, 5), dtype=np.int64), (1.0, 2.0), 0.0),
        (np.zeros((1, 4, 5), dtype=np.bool_), (np.inf, 2.0), 0.0),
        (np.zeros((1, 4, 5), dtype=np.bool_), (1.0, np.nan), 0.0),
        (np.zeros((1, 4, 5), dtype=np.bool_), (1.0, 2.0), np.inf),
    ],
)
def test_warp_rejects_invalid_grid_pivot_and_angle(
    grid: np.ndarray,
    pivot: tuple[float, float],
    angle_degrees: float,
) -> None:
    """Breaks if invalid geometric input is accepted silently."""
    with pytest.raises(ValueError):
        warp_grid_about_pivot(grid, pivot, angle_degrees)


def test_fractional_pivot_warp_keeps_padded_outside_cell() -> None:
    """Breaks if bounds omit a fractional-pivot rotation's outside support."""
    grid = np.zeros((1, 5, 5), dtype=np.bool_)
    grid[0, 2, 3] = True

    warped = warp_grid_about_pivot(grid, (1.2, 2.2), 90.0)

    assert warped.bounds.row_min <= -1 < warped.bounds.row_max
    assert warped.grid[0, -1 - warped.bounds.row_min, 3 - warped.bounds.col_min]


@pytest.mark.parametrize(
    ("angle_degrees", "expected_bounds"),
    [
        (90.0, SpatialBounds(-5, 0, 0, 5)),
        (180.0, SpatialBounds(-5, 0, -5, 0)),
        (270.0, SpatialBounds(0, 5, -5, 0)),
    ],
)
def test_cardinal_rotation_bounds_are_exact(
    angle_degrees: float, expected_bounds: SpatialBounds
) -> None:
    """Breaks if trigonometric roundoff expands cardinal output bounds."""
    grid = np.zeros((1, 5, 5), dtype=np.bool_)

    warped = warp_grid_about_pivot(grid, (0.0, 0.0), angle_degrees)

    assert warped.bounds == expected_bounds


def test_scoring_keeps_fractional_pivot_support_outside_target_frame() -> None:
    """Breaks if score crops outside support before counting false positives."""
    grid = np.zeros((1, 5, 5), dtype=np.bool_)
    grid[0, 2, 3] = True
    warped = warp_grid_about_pivot(grid, (1.2, 2.2), 90.0)

    score_against_empty = score_warped_grid(warped, np.zeros((1, 5, 5), dtype=np.bool_))
    target = np.zeros((1, 5, 5), dtype=np.bool_)
    target[0, 2, 3] = True
    score_against_target = score_warped_grid(warped, target)

    assert score_against_empty.out_of_frame_support == 1
    assert score_against_empty.in_frame_support == 0
    assert score_against_empty.intersection == 0
    assert score_against_empty.union == 1
    assert score_against_target.intersection == 0
    assert score_against_target.union == 2


def test_channels_share_fractional_pivot_bounds_and_mapping() -> None:
    """Breaks if categories are transformed with separate spatial geometry."""
    grid = np.zeros((2, 5, 5), dtype=np.bool_)
    grid[:, 2, 3] = True

    warped = warp_grid_about_pivot(grid, (1.2, 2.2), 90.0)

    assert warped.grid.shape[1:] == (warped.bounds.rows, warped.bounds.cols)
    assert np.array_equal(warped.grid[0], warped.grid[1])
    assert warped.grid[:, -1 - warped.bounds.row_min, 3 - warped.bounds.col_min].tolist() == [
        True,
        True,
    ]


def test_score_counts_in_frame_true_positive_and_out_of_frame_false_positive() -> None:
    """Breaks if off-frame predicted support is omitted from the union."""
    predicted = np.zeros((1, 2, 2), dtype=np.bool_)
    predicted[0, 0, 0] = True
    predicted[0, 1, 1] = True
    warped = WarpedGrid(predicted, SpatialBounds(-1, 1, 0, 2), input_support=2)
    target = np.zeros((1, 1, 2), dtype=np.bool_)
    target[0, 0, 1] = True

    score = score_warped_grid(warped, target)

    assert score.intersection == 1
    assert score.union == 2
    assert score.predicted_support == 2
    assert score.target_support == 1
    assert score.in_frame_support == 1
    assert score.out_of_frame_support == 1
    assert score.iou == 0.5


def test_score_empty_prediction_against_nonempty_target() -> None:
    """Breaks if an empty prediction changes target support or union."""
    target = np.zeros((1, 2, 2), dtype=np.bool_)
    target[0, 1, 0] = True
    warped = WarpedGrid(
        np.zeros_like(target), SpatialBounds(0, 2, 0, 2), input_support=0
    )

    score = score_warped_grid(warped, target)

    assert score.intersection == 0
    assert score.union == 1
    assert score.predicted_support == 0
    assert score.target_support == 1
    assert score.in_frame_support == 0
    assert score.out_of_frame_support == 0
    assert score.iou == 0.0


def test_score_empty_prediction_and_target_has_zero_iou() -> None:
    """Breaks if empty-versus-empty returns an undefined or nonzero IoU."""
    target = np.zeros((1, 2, 2), dtype=np.bool_)
    warped = WarpedGrid(target.copy(), SpatialBounds(0, 2, 0, 2), input_support=0)

    score = score_warped_grid(warped, target)

    assert score.intersection == 0
    assert score.union == 0
    assert score.iou == 0.0


def test_score_counts_identical_occupancy_once_per_category() -> None:
    """Breaks if category-aware support collapses duplicate spatial occupancy."""
    target = np.zeros((2, 2, 2), dtype=np.bool_)
    target[:, 1, 1] = True
    warped = WarpedGrid(target.copy(), SpatialBounds(0, 2, 0, 2), input_support=2)

    score = score_warped_grid(warped, target)

    assert score.intersection == 2
    assert score.union == 2
    assert score.predicted_support == 2
    assert score.target_support == 2


def test_rotate_direction_vectors_uses_physical_grid_angle_in_reflected_basis() -> None:
    """Breaks if a physical grid rotation uses the same numeric stored-basis angle."""
    source = world_delta_to_direction_vector(1.0, 0.0)
    expected = world_delta_to_direction_vector(0.0, 1.0)
    rotated = rotate_direction_vectors(
        np.asarray([source, (0.0, 0.0)], dtype=np.float32), 90.0
    )

    assert rotated.dtype == np.float32
    assert rotated[0] == pytest.approx(expected, abs=1e-6)
    assert rotated[1] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_direction_cosine_averages_only_nonzero_vector_pairs() -> None:
    """Breaks if zero vectors enter the mean cosine or no pairs yield nonzero."""
    predicted = np.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    target = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, -1.0]], dtype=np.float32)

    assert direction_cosine(predicted, target) == pytest.approx(0.0)
    assert direction_cosine(
        np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32)
    ) == 0.0


def test_select_best_angle_keeps_first_angle_when_all_scores_tie() -> None:
    """Breaks if equal-IoU candidates replace the caller's earlier angle."""
    grid = np.zeros((1, 2, 2), dtype=np.bool_)
    target = np.zeros_like(grid)

    best, scores = select_best_angle(
        grid, target, (0.5, 0.5), (0.0, 90.0, 180.0, 270.0)
    )

    assert best.angle_degrees == 0.0
    assert tuple(score.angle_degrees for score in scores) == (0.0, 90.0, 180.0, 270.0)
    assert all(score.raster.iou == 0.0 for score in scores)
