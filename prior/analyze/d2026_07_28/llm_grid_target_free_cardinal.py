"""Typed target-free cardinal selector and soft-raster aggregation core."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from prior.analyze.llm_grid_registration import WarpedGrid, rotate_direction_vectors


ANGLE_ORDER = (0.0, 90.0, 180.0, 270.0)
_DIRECTION_TOLERANCE = 1e-4
_HEADING_TIE_TOLERANCE = 1e-6


def _require_boolean(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _require_float32_array(
    value: NDArray[np.float32], shape: tuple[int, ...], name: str
) -> NDArray[np.float32]:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != np.float32:
        raise ValueError(f"{name} must have float32 dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    result.setflags(write=False)
    return result


def _require_boolean_grid(value: NDArray[np.bool_]) -> NDArray[np.bool_]:
    if not isinstance(value, np.ndarray) or value.shape != (37, 50, 50):
        raise ValueError("predicted_grid must have shape (37, 50, 50)")
    if value.dtype != np.bool_:
        raise ValueError("predicted_grid must have boolean dtype")
    result = np.array(value, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


def _validate_start_direction(start_direction: NDArray[np.float32]) -> None:
    if not isinstance(start_direction, np.ndarray) or start_direction.shape != (2,):
        raise ValueError("start_direction must have shape (2,)")
    if start_direction.dtype != np.float32:
        raise ValueError("start_direction must have float32 dtype")
    if not np.isfinite(start_direction).all():
        raise ValueError("start_direction must contain only finite values")
    norm = float(np.linalg.norm(start_direction))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=_DIRECTION_TOLERANCE):
        raise ValueError("start_direction must have unit norm")


def _validate_predicted_directions(directions: NDArray[np.float32]) -> None:
    norms = np.linalg.norm(directions, axis=1)
    valid = (norms == 0.0) | np.isclose(
        norms, 1.0, rtol=0.0, atol=_DIRECTION_TOLERANCE
    )
    if not bool(np.all(valid)):
        raise ValueError("predicted_directions rows must be zero or unit length")


@dataclass(frozen=True, order=True)
class EpisodeKey:
    scene_id: str
    example_id: str


@dataclass(frozen=True)
class SelectorInput:
    schema_valid: bool
    start_direction: NDArray[np.float32]
    predicted_grid: NDArray[np.bool_]
    predicted_directions: NDArray[np.float32]

    def __post_init__(self) -> None:
        _require_boolean(self.schema_valid, "schema_valid")
        _validate_start_direction(self.start_direction)
        directions = _require_float32_array(
            self.predicted_directions, (5, 2), "predicted_directions"
        )
        _validate_predicted_directions(directions)
        start_direction = _require_float32_array(
            self.start_direction, (2,), "start_direction"
        )
        object.__setattr__(self, "start_direction", start_direction)
        object.__setattr__(self, "predicted_grid", _require_boolean_grid(self.predicted_grid))
        object.__setattr__(self, "predicted_directions", directions)


@dataclass(frozen=True)
class HeadingMapping:
    sign: int
    offset_degrees: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.sign, bool)
            or not isinstance(self.sign, Integral)
            or self.sign not in (-1, 1)
        ):
            raise ValueError("sign must be -1 or +1")
        if (
            isinstance(self.offset_degrees, bool)
            or not isinstance(self.offset_degrees, Real)
            or float(self.offset_degrees) not in ANGLE_ORDER
        ):
            raise ValueError("offset_degrees must be a declared cardinal angle")

    def angle_for(self, start_direction: NDArray[np.float32]) -> float:
        heading_bin = quantize_heading(start_direction)
        return float((self.sign * heading_bin + self.offset_degrees) % 360.0)


@dataclass(frozen=True)
class SelectorAssignment:
    key: EpisodeKey
    schema_valid: bool
    heading_bin_degrees: float
    primary_angle_degrees: float
    global_angle_degrees: float
    direct_angle_degrees: float
    direct_margin: float


@dataclass(frozen=True)
class SoftRasterScore:
    intersection_mass: float
    union_mass: float
    predicted_mass: float
    target_mass: float
    precision: float
    recall: float
    f1: float
    iou: float


def quantize_heading(start_direction: NDArray[np.float32]) -> float:
    """Quantize a display-frame unit heading to the declared cardinal order."""
    _validate_start_direction(start_direction)
    heading = float(
        np.degrees(np.arctan2(float(start_direction[0]), float(start_direction[1])))
        % 360.0
    )
    best_angle = ANGLE_ORDER[0]
    best_distance = 360.0
    for angle in ANGLE_ORDER:
        distance = abs((angle - heading + 180.0) % 360.0 - 180.0)
        if distance < best_distance - _HEADING_TIE_TOLERANCE:
            best_angle = angle
            best_distance = distance
    return best_angle


def direct_heading_assignment(selector_input: SelectorInput) -> tuple[float, float]:
    """Choose the cardinal rotation aligning the first predicted direction to start."""
    if not isinstance(selector_input, SelectorInput):
        raise ValueError("selector_input must be a SelectorInput")
    if not selector_input.schema_valid:
        return 0.0, 0.0

    norms = np.linalg.norm(selector_input.predicted_directions, axis=1)
    nonzero_indices = np.flatnonzero(norms > 0.0)
    if nonzero_indices.size == 0:
        return 0.0, 0.0
    first_vector = selector_input.predicted_directions[nonzero_indices[0] : nonzero_indices[0] + 1]

    scores: list[float] = []
    for angle in ANGLE_ORDER:
        rotated = rotate_direction_vectors(first_vector, angle)
        cosine = float(
            np.clip(np.dot(rotated[0], selector_input.start_direction), -1.0, 1.0)
        )
        scores.append(cosine)
    best_index = 0
    for index, score in enumerate(scores[1:], start=1):
        if score > scores[best_index]:
            best_index = index
    second_score = max(
        score for index, score in enumerate(scores) if index != best_index
    )
    return ANGLE_ORDER[best_index], scores[best_index] - second_score


def _require_target_grid(target: NDArray[np.bool_]) -> None:
    if not isinstance(target, np.ndarray) or target.shape != (37, 50, 50):
        raise ValueError("target must have shape (37, 50, 50)")
    if target.dtype != np.bool_:
        raise ValueError("target must have boolean dtype")


def uniform_soft_score(
    hypotheses: tuple[WarpedGrid, ...], target: NDArray[np.bool_]
) -> SoftRasterScore:
    """Average four padded hypotheses and calculate exact soft raster metrics."""
    if len(hypotheses) != len(ANGLE_ORDER):
        raise ValueError("hypotheses must contain one grid per declared angle")
    _require_target_grid(target)
    if not all(isinstance(hypothesis, WarpedGrid) for hypothesis in hypotheses):
        raise ValueError("hypotheses must contain WarpedGrid values")
    if any(hypothesis.grid.shape[0] != target.shape[0] for hypothesis in hypotheses):
        raise ValueError("hypotheses and target must have the same channel count")

    row_min = min(0, *(hypothesis.bounds.row_min for hypothesis in hypotheses))
    row_max = max(50, *(hypothesis.bounds.row_max for hypothesis in hypotheses))
    col_min = min(0, *(hypothesis.bounds.col_min for hypothesis in hypotheses))
    col_max = max(50, *(hypothesis.bounds.col_max for hypothesis in hypotheses))
    prediction = np.zeros(
        (target.shape[0], row_max - row_min, col_max - col_min), dtype=np.float64
    )
    for hypothesis in hypotheses:
        row_start = hypothesis.bounds.row_min - row_min
        row_end = hypothesis.bounds.row_max - row_min
        col_start = hypothesis.bounds.col_min - col_min
        col_end = hypothesis.bounds.col_max - col_min
        prediction[:, row_start:row_end, col_start:col_end] += hypothesis.grid
    prediction /= len(hypotheses)

    target_canvas = np.zeros_like(prediction)
    target_canvas[:, -row_min : 50 - row_min, -col_min : 50 - col_min] = target
    intersection_mass = float(np.minimum(prediction, target_canvas).sum())
    predicted_mass = float(prediction.sum())
    target_mass = float(target_canvas.sum())
    union_mass = predicted_mass + target_mass - intersection_mass
    precision = 0.0 if predicted_mass == 0.0 else intersection_mass / predicted_mass
    recall = 0.0 if target_mass == 0.0 else intersection_mass / target_mass
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    iou = 0.0 if union_mass == 0.0 else intersection_mass / union_mass
    return SoftRasterScore(
        intersection_mass=intersection_mass,
        union_mass=union_mass,
        predicted_mass=predicted_mass,
        target_mass=target_mass,
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
    )
