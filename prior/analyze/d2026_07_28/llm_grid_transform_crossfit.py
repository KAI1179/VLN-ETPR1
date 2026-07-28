"""Typed object/region cross-fit scoring for LLM-Grid transform controls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from prior.analyze.llm_grid_registration import (
    RasterScore,
    SpatialBounds,
    WarpedGrid,
    score_warped_grid,
    warp_grid_about_pivot,
)


_CHANNEL_COUNT = 37
_OBJECT_CHANNEL_COUNT = 27


def _require_nonempty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _require_iou(value: float, name: str) -> float:
    result = _require_finite_real(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


def _require_boolean_grid(
    grid: NDArray[np.bool_], name: str
) -> NDArray[np.bool_]:
    if not isinstance(grid, np.ndarray) or grid.ndim != 3:
        raise ValueError(f"{name} must have shape (37, rows, cols)")
    if grid.dtype != np.bool_:
        raise ValueError(f"{name} must have boolean dtype")
    if grid.shape[0] != _CHANNEL_COUNT:
        raise ValueError(f"{name} must have {_CHANNEL_COUNT} channels")
    return grid


def _require_pivot(pivot: tuple[float, float], name: str) -> tuple[float, float]:
    if len(pivot) != 2:
        raise ValueError(f"{name} must contain row and column coordinates")
    return (
        _require_finite_real(pivot[0], f"{name} row"),
        _require_finite_real(pivot[1], f"{name} column"),
    )


def _is_close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))


class PivotMode(str, Enum):
    TRUE_START = "true_start"
    MAP_CENTER = "map_center"
    SHUFFLED_START = "shuffled_start"


class Direction(str, Enum):
    OBJECT_TO_REGION = "object_to_region"
    REGION_TO_OBJECT = "region_to_object"


@dataclass(frozen=True)
class EpisodeCase:
    split: str
    scene_id: str
    example_id: str
    schema_valid: bool
    predicted_grid: NDArray[np.bool_]
    target_grid: NDArray[np.bool_]
    true_start_pivot: tuple[float, float]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.split, "split")
        _require_nonempty_string(self.scene_id, "scene_id")
        _require_nonempty_string(self.example_id, "example_id")
        if not isinstance(self.schema_valid, bool):
            raise ValueError("schema_valid must be a boolean")
        predicted = _require_boolean_grid(self.predicted_grid, "predicted_grid")
        target = _require_boolean_grid(self.target_grid, "target_grid")
        if predicted.shape[1:] != target.shape[1:]:
            raise ValueError("predicted_grid and target_grid must have the same spatial shape")
        _require_pivot(self.true_start_pivot, "true_start_pivot")


@dataclass(frozen=True)
class FamilyAngleScore:
    angle_degrees: float
    bounds: SpatialBounds
    object_input_support: int
    region_input_support: int
    object_score: RasterScore
    region_score: RasterScore

    def __post_init__(self) -> None:
        _require_finite_real(self.angle_degrees, "angle_degrees")
        if not isinstance(self.bounds, SpatialBounds):
            raise ValueError("bounds must be a SpatialBounds")
        _require_nonnegative_int(self.object_input_support, "object_input_support")
        _require_nonnegative_int(self.region_input_support, "region_input_support")
        if not isinstance(self.object_score, RasterScore):
            raise ValueError("object_score must be a RasterScore")
        if not isinstance(self.region_score, RasterScore):
            raise ValueError("region_score must be a RasterScore")


@dataclass(frozen=True)
class CrossFitResult:
    direction: Direction
    selected_angle_degrees: float
    second_angle_degrees: float
    selector_identity_iou: float
    selector_selected_iou: float
    selector_margin: float
    heldout_identity_iou: float
    heldout_selected_iou: float
    delta_iou: float
    selector_predicted_support_empty: bool
    selector_target_support_empty: bool
    selector_union_empty: bool
    heldout_predicted_support_empty: bool
    heldout_target_support_empty: bool
    heldout_union_empty: bool

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            raise ValueError("direction must be a Direction")
        _require_finite_real(self.selected_angle_degrees, "selected_angle_degrees")
        _require_finite_real(self.second_angle_degrees, "second_angle_degrees")
        _require_iou(self.selector_identity_iou, "selector_identity_iou")
        selector_selected = _require_iou(
            self.selector_selected_iou, "selector_selected_iou"
        )
        margin = _require_iou(self.selector_margin, "selector_margin")
        heldout_identity = _require_iou(
            self.heldout_identity_iou, "heldout_identity_iou"
        )
        heldout_selected = _require_iou(
            self.heldout_selected_iou, "heldout_selected_iou"
        )
        delta = _require_finite_real(self.delta_iou, "delta_iou")
        if not -1.0 <= delta <= 1.0:
            raise ValueError("delta_iou must be within [-1, 1]")
        if not _is_close(margin, selector_selected - self._second_iou()):
            raise ValueError("selector_margin must equal selected minus second IoU")
        if not _is_close(delta, heldout_selected - heldout_identity):
            raise ValueError("delta_iou must equal heldout selected minus identity IoU")
        flags = (
            (self.selector_predicted_support_empty, "selector_predicted_support_empty"),
            (self.selector_target_support_empty, "selector_target_support_empty"),
            (self.selector_union_empty, "selector_union_empty"),
            (self.heldout_predicted_support_empty, "heldout_predicted_support_empty"),
            (self.heldout_target_support_empty, "heldout_target_support_empty"),
            (self.heldout_union_empty, "heldout_union_empty"),
        )
        for flag, name in flags:
            if not isinstance(flag, bool):
                raise ValueError(f"{name} must be a boolean")

    def _second_iou(self) -> float:
        return self.selector_selected_iou - self.selector_margin


def score_angle_families(
    predicted_grid: NDArray[np.bool_],
    target_grid: NDArray[np.bool_],
    pivot: tuple[float, float],
    angles: tuple[float, ...],
) -> tuple[FamilyAngleScore, ...]:
    """Score one shared warp per angle against object and region target families."""
    predicted = _require_boolean_grid(predicted_grid, "predicted_grid")
    target = _require_boolean_grid(target_grid, "target_grid")
    if predicted.shape[1:] != target.shape[1:]:
        raise ValueError("predicted_grid and target_grid must have the same spatial shape")
    checked_pivot = _require_pivot(pivot, "pivot")
    object_input_support = int(np.count_nonzero(predicted[:_OBJECT_CHANNEL_COUNT]))
    region_input_support = int(np.count_nonzero(predicted[_OBJECT_CHANNEL_COUNT:]))

    scores: list[FamilyAngleScore] = []
    for angle in angles:
        checked_angle = _require_finite_real(angle, "angle_degrees")
        warped = warp_grid_about_pivot(predicted, checked_pivot, checked_angle)
        object_warp = WarpedGrid(
            grid=warped.grid[:_OBJECT_CHANNEL_COUNT],
            bounds=warped.bounds,
            input_support=object_input_support,
        )
        region_warp = WarpedGrid(
            grid=warped.grid[_OBJECT_CHANNEL_COUNT:],
            bounds=warped.bounds,
            input_support=region_input_support,
        )
        scores.append(
            FamilyAngleScore(
                angle_degrees=checked_angle,
                bounds=warped.bounds,
                object_input_support=object_input_support,
                region_input_support=region_input_support,
                object_score=score_warped_grid(
                    object_warp, target[:_OBJECT_CHANNEL_COUNT]
                ),
                region_score=score_warped_grid(
                    region_warp, target[_OBJECT_CHANNEL_COUNT:]
                ),
            )
        )
    return tuple(scores)


def crossfit_scores(
    scores: Sequence[FamilyAngleScore],
    direction: Direction,
) -> CrossFitResult:
    """Select a rotation in one family and score it only in the held-out family."""
    if not isinstance(direction, Direction):
        raise ValueError("direction must be a Direction")
    rows = tuple(scores)
    if len(rows) < 2:
        raise ValueError("scores must contain at least two angle rows")
    if not all(isinstance(row, FamilyAngleScore) for row in rows):
        raise ValueError("scores must contain only FamilyAngleScore rows")

    angles = tuple(row.angle_degrees for row in rows)
    if len(set(angles)) != len(angles):
        raise ValueError("angle_degrees must be unique")
    try:
        identity_index = angles.index(0.0)
    except ValueError as error:
        raise ValueError("scores must contain identity angle 0.0") from error

    if direction is Direction.OBJECT_TO_REGION:
        selector_scores = tuple(row.object_score for row in rows)
        heldout_scores = tuple(row.region_score for row in rows)
    else:
        selector_scores = tuple(row.region_score for row in rows)
        heldout_scores = tuple(row.object_score for row in rows)

    ordered_indices = sorted(
        range(len(rows)), key=lambda index: selector_scores[index].iou, reverse=True
    )
    selected_index, second_index = ordered_indices[:2]
    selector_identity = selector_scores[identity_index]
    selector_selected = selector_scores[selected_index]
    selector_second = selector_scores[second_index]
    heldout_identity = heldout_scores[identity_index]
    heldout_selected = heldout_scores[selected_index]

    return CrossFitResult(
        direction=direction,
        selected_angle_degrees=rows[selected_index].angle_degrees,
        second_angle_degrees=rows[second_index].angle_degrees,
        selector_identity_iou=selector_identity.iou,
        selector_selected_iou=selector_selected.iou,
        selector_margin=selector_selected.iou - selector_second.iou,
        heldout_identity_iou=heldout_identity.iou,
        heldout_selected_iou=heldout_selected.iou,
        delta_iou=heldout_selected.iou - heldout_identity.iou,
        selector_predicted_support_empty=selector_identity.predicted_support == 0,
        selector_target_support_empty=selector_identity.target_support == 0,
        selector_union_empty=selector_identity.union == 0,
        heldout_predicted_support_empty=heldout_identity.predicted_support == 0,
        heldout_target_support_empty=heldout_identity.target_support == 0,
        heldout_union_empty=heldout_identity.union == 0,
    )
