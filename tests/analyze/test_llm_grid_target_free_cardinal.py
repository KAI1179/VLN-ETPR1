from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

from prior.analyze.d2026_07_28 import llm_grid_target_free_cardinal as target_free
from prior.analyze.llm_grid_registration import (
    WarpedGrid,
    warp_grid_about_pivot,
)


def test_quantize_heading_uses_clockwise_display_frame_and_declared_ties() -> None:
    """Breaks if heading uses the wrong display axis or tie ordering."""
    assert target_free.quantize_heading(np.asarray([0.0, 1.0], np.float32)) == 0.0
    assert target_free.quantize_heading(np.asarray([1.0, 0.0], np.float32)) == 90.0
    boundary = np.asarray([2**-0.5, 2**-0.5], np.float32)
    assert target_free.quantize_heading(boundary) == 0.0


@pytest.mark.parametrize(
    "bad",
    ([0.0, 0.0], [float("nan"), 1.0], [0.5, 0.0]),
)
def test_selector_input_rejects_invalid_start_direction(bad: list[float]) -> None:
    """Breaks if corrupt runtime heading metadata reaches the selector."""
    with pytest.raises(ValueError, match="start_direction"):
        target_free.SelectorInput(
            schema_valid=True,
            start_direction=np.asarray(bad, np.float32),
            predicted_grid=np.zeros((37, 50, 50), dtype=np.bool_),
            predicted_directions=np.zeros((5, 2), dtype=np.float32),
        )


def valid_selector_input(
    *,
    schema_valid: bool = True,
    start_direction: NDArray[np.float32] | None = None,
    predicted_directions: NDArray[np.float32] | None = None,
) -> target_free.SelectorInput:
    """Build a valid selector input with only the tested runtime fields changed."""
    return target_free.SelectorInput(
        schema_valid=schema_valid,
        start_direction=(
            np.asarray([0.0, 1.0], dtype=np.float32)
            if start_direction is None
            else start_direction
        ),
        predicted_grid=np.zeros((37, 50, 50), dtype=np.bool_),
        predicted_directions=(
            np.zeros((5, 2), dtype=np.float32)
            if predicted_directions is None
            else predicted_directions
        ),
    )


@pytest.mark.parametrize(
    ("mapping", "expected"),
    (
        (target_free.HeadingMapping(+1, 0.0), 90.0),
        (target_free.HeadingMapping(+1, 90.0), 180.0),
        (target_free.HeadingMapping(+1, 180.0), 270.0),
        (target_free.HeadingMapping(+1, 270.0), 0.0),
        (target_free.HeadingMapping(-1, 0.0), 270.0),
        (target_free.HeadingMapping(-1, 90.0), 0.0),
        (target_free.HeadingMapping(-1, 180.0), 90.0),
        (target_free.HeadingMapping(-1, 270.0), 180.0),
    ),
)
def test_heading_mapping_emits_each_declared_cardinal_correction(
    mapping: target_free.HeadingMapping, expected: float
) -> None:
    """Breaks if any frozen sign-and-offset mapping changes its physical angle."""
    assert mapping.angle_for(np.asarray([1.0, 0.0], dtype=np.float32)) == expected


@pytest.mark.parametrize(
    ("sign", "offset"),
    ((0, 0.0), (+1, 45.0), (True, 0.0)),
)
def test_heading_mapping_rejects_non_declared_identity_behavior(
    sign: int, offset: float
) -> None:
    """Breaks if invalid mappings silently become an identity correction."""
    with pytest.raises(ValueError):
        target_free.HeadingMapping(sign, offset)


def test_direct_selector_empty_prediction_is_identity_with_zero_margin() -> None:
    """Breaks if an empty prediction uses a directional tie as a correction."""
    selector_input = valid_selector_input(
        schema_valid=False,
        predicted_directions=np.zeros((5, 2), np.float32),
    )
    assert target_free.direct_heading_assignment(selector_input) == (0.0, 0.0)


def test_direct_selector_invalid_prediction_is_identity_with_zero_margin() -> None:
    """Breaks if schema-invalid predictions participate in direction selection."""
    selector_input = valid_selector_input(
        schema_valid=False,
        predicted_directions=np.asarray(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    assert target_free.direct_heading_assignment(selector_input) == (0.0, 0.0)


def test_direct_selector_uses_positive_angle_vector_sign_and_margin() -> None:
    """Breaks if physical positive angles rotate stored vectors in the wrong sign."""
    selector_input = valid_selector_input(
        start_direction=np.asarray([0.0, 1.0], dtype=np.float32),
        predicted_directions=np.asarray(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    angle, margin = target_free.direct_heading_assignment(selector_input)
    assert angle == 270.0
    assert margin == pytest.approx(1.0)


def test_direct_selector_retains_first_declared_angle_on_cosine_tie() -> None:
    """Breaks if equal cosine candidates replace the first physical angle."""
    diagonal = np.asarray([2**-0.5, 2**-0.5], dtype=np.float32)
    selector_input = valid_selector_input(
        start_direction=diagonal,
        predicted_directions=np.asarray(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    angle, margin = target_free.direct_heading_assignment(selector_input)
    assert angle == 0.0
    assert margin == pytest.approx(0.0, abs=1e-6)


def single_corner_cell_grid() -> NDArray[np.bool_]:
    grid = np.zeros((37, 50, 50), dtype=np.bool_)
    grid[0, 0, 0] = True
    return grid


def cardinal_warps(
    grid: NDArray[np.bool_], pivot: tuple[float, float]
) -> tuple[WarpedGrid, ...]:
    return tuple(warp_grid_about_pivot(grid, pivot, angle) for angle in target_free.ANGLE_ORDER)


def empty_target() -> NDArray[np.bool_]:
    return np.zeros((37, 50, 50), dtype=np.bool_)


def test_uniform_soft_score_keeps_out_of_frame_false_positive_mass() -> None:
    """Breaks if the common canvas clips padded cells before soft scoring."""
    hypotheses = cardinal_warps(single_corner_cell_grid(), pivot=(1.0, 1.0))
    score = target_free.uniform_soft_score(hypotheses, empty_target())
    assert any(
        hypothesis.bounds.row_min < 0 or hypothesis.bounds.col_min < 0
        for hypothesis in hypotheses
    )
    assert score.predicted_mass > 0.0
    assert score.intersection_mass == 0.0
    assert score.iou == 0.0


def test_uniform_soft_score_calculates_exact_soft_statistics() -> None:
    """Breaks if averaging, mass accounting, or soft precision metrics are wrong."""
    target = empty_target()
    target[0, 10, 10] = True
    hit = target.copy()
    miss = empty_target()
    miss[0, 10, 11] = True
    hypotheses = tuple(
        WarpedGrid(grid, warp_grid_about_pivot(grid, (25.0, 25.0), 0.0).bounds, 1)
        for grid in (hit, miss, empty_target(), empty_target())
    )

    score = target_free.uniform_soft_score(hypotheses, target)

    assert score.intersection_mass == pytest.approx(0.25)
    assert score.union_mass == pytest.approx(1.25)
    assert score.predicted_mass == pytest.approx(0.5)
    assert score.target_mass == pytest.approx(1.0)
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.25)
    assert score.f1 == pytest.approx(1.0 / 3.0)
    assert score.iou == pytest.approx(0.2)


def test_uniform_soft_score_returns_zero_metrics_for_empty_union() -> None:
    """Breaks if empty soft rasters produce undefined or nonzero metrics."""
    empty = empty_target()
    hypotheses = tuple(
        WarpedGrid(
            empty.copy(), warp_grid_about_pivot(empty, (25.0, 25.0), 0.0).bounds, 0
        )
        for _ in target_free.ANGLE_ORDER
    )

    score = target_free.uniform_soft_score(hypotheses, empty)

    assert score == target_free.SoftRasterScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
