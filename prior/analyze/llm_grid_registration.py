from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray


def _require_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    _require_int(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


@dataclass(frozen=True)
class SpatialBounds:
    row_min: int
    row_max: int
    col_min: int
    col_max: int

    def __post_init__(self) -> None:
        _require_int(self.row_min, "row_min")
        _require_int(self.row_max, "row_max")
        _require_int(self.col_min, "col_min")
        _require_int(self.col_max, "col_max")
        if self.row_min > self.row_max:
            raise ValueError("row_min must not exceed row_max")
        if self.col_min > self.col_max:
            raise ValueError("col_min must not exceed col_max")

    @property
    def rows(self) -> int:
        return self.row_max - self.row_min

    @property
    def cols(self) -> int:
        return self.col_max - self.col_min


@dataclass(frozen=True)
class WarpedGrid:
    grid: NDArray[np.bool_]
    bounds: SpatialBounds
    input_support: int

    def __post_init__(self) -> None:
        if self.grid.ndim != 3:
            raise ValueError("grid must have shape (channels, rows, cols)")
        if self.grid.dtype != np.bool_:
            raise ValueError("grid must have boolean dtype")
        if self.grid.shape[1:] != (self.bounds.rows, self.bounds.cols):
            raise ValueError("grid shape must match bounds")
        _require_nonnegative_int(self.input_support, "input_support")

    @property
    def total_support(self) -> int:
        return int(np.count_nonzero(self.grid))


@dataclass(frozen=True)
class RasterScore:
    intersection: int
    union: int
    predicted_support: int
    target_support: int
    in_frame_support: int
    out_of_frame_support: int

    def __post_init__(self) -> None:
        fields = (
            (self.intersection, "intersection"),
            (self.union, "union"),
            (self.predicted_support, "predicted_support"),
            (self.target_support, "target_support"),
            (self.in_frame_support, "in_frame_support"),
            (self.out_of_frame_support, "out_of_frame_support"),
        )
        for value, name in fields:
            _require_nonnegative_int(value, name)
        if self.intersection > self.predicted_support:
            raise ValueError("intersection cannot exceed predicted_support")
        if self.intersection > self.target_support:
            raise ValueError("intersection cannot exceed target_support")
        if self.in_frame_support + self.out_of_frame_support != self.predicted_support:
            raise ValueError("frame supports must partition predicted_support")
        if self.union != self.predicted_support + self.target_support - self.intersection:
            raise ValueError("union must equal predicted + target - intersection")

    @property
    def iou(self) -> float:
        return 0.0 if self.union == 0 else self.intersection / self.union


@dataclass(frozen=True)
class AngleScore:
    angle_degrees: float
    raster: RasterScore
    warped_support_ratio: float

    def __post_init__(self) -> None:
        _require_finite_real(self.angle_degrees, "angle_degrees")
        if not isinstance(self.raster, RasterScore):
            raise ValueError("raster must be a RasterScore")
        if _require_finite_real(self.warped_support_ratio, "warped_support_ratio") < 0:
            raise ValueError("warped_support_ratio must be non-negative")


def _validate_grid(grid: NDArray[np.bool_]) -> None:
    if grid.ndim != 3:
        raise ValueError("grid must have shape (channels, rows, cols)")
    if grid.dtype != np.bool_:
        raise ValueError("grid must have boolean dtype")


def _validate_pivot(pivot: tuple[float, float]) -> tuple[float, float]:
    if len(pivot) != 2:
        raise ValueError("pivot must contain row and column coordinates")
    return (
        _require_finite_real(pivot[0], "pivot row"),
        _require_finite_real(pivot[1], "pivot column"),
    )


def _rotation_coefficients(angle_degrees: float) -> tuple[float, float]:
    normalized = angle_degrees % 360.0
    if normalized == 0.0:
        return 1.0, 0.0
    if normalized == 90.0:
        return 0.0, 1.0
    if normalized == 180.0:
        return -1.0, 0.0
    if normalized == 270.0:
        return 0.0, -1.0
    angle_radians = np.deg2rad(angle_degrees)
    return float(np.cos(angle_radians)), float(np.sin(angle_radians))


def warp_grid_about_pivot(
    grid: NDArray[np.bool_], pivot: tuple[float, float], angle_degrees: float
) -> WarpedGrid:
    """Nearest-neighbor warp of every category grid around one spatial pivot."""
    _validate_grid(grid)
    pivot_row, pivot_col = _validate_pivot(pivot)
    angle = _require_finite_real(angle_degrees, "angle_degrees")

    _, rows, cols = grid.shape
    input_support = int(np.count_nonzero(grid))
    if angle == 0.0:
        return WarpedGrid(
            grid=grid.copy(),
            bounds=SpatialBounds(0, rows, 0, cols),
            input_support=input_support,
        )

    cosine, sine = _rotation_coefficients(angle)
    corners = np.asarray(
        ((0.0, 0.0), (0.0, cols), (rows, 0.0), (rows, cols)), dtype=np.float64
    )
    corner_rows = corners[:, 0] - pivot_row
    corner_cols = corners[:, 1] - pivot_col
    rotated_rows = pivot_row + cosine * corner_rows - sine * corner_cols
    rotated_cols = pivot_col + sine * corner_rows + cosine * corner_cols
    bounds = SpatialBounds(
        int(np.floor(rotated_rows.min())),
        int(np.ceil(rotated_rows.max())),
        int(np.floor(rotated_cols.min())),
        int(np.ceil(rotated_cols.max())),
    )
    output_rows, output_cols = np.meshgrid(
        np.arange(bounds.row_min, bounds.row_max, dtype=np.float64) + 0.5,
        np.arange(bounds.col_min, bounds.col_max, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    output_row_offsets = output_rows - pivot_row
    output_col_offsets = output_cols - pivot_col
    source_rows = pivot_row + cosine * output_row_offsets + sine * output_col_offsets
    source_cols = pivot_col - sine * output_row_offsets + cosine * output_col_offsets
    valid = (
        (source_rows >= 0.0)
        & (source_rows < rows)
        & (source_cols >= 0.0)
        & (source_cols < cols)
    )
    result = np.zeros((grid.shape[0], bounds.rows, bounds.cols), dtype=np.bool_)
    result[:, valid] = grid[
        :, np.floor(source_rows[valid]).astype(np.intp), np.floor(source_cols[valid]).astype(np.intp)
    ]
    return WarpedGrid(
        grid=result,
        bounds=bounds,
        input_support=input_support,
    )


def score_warped_grid(warped: WarpedGrid, target: NDArray[np.bool_]) -> RasterScore:
    """Score a padded warp against a target frame without discarding outside support."""
    if not isinstance(warped, WarpedGrid):
        raise ValueError("warped must be a WarpedGrid")
    _validate_grid(target)
    if warped.grid.shape[0] != target.shape[0]:
        raise ValueError("warped and target must have the same channel count")

    target_rows, target_cols = target.shape[1:]
    row_start = max(warped.bounds.row_min, 0)
    row_end = min(warped.bounds.row_max, target_rows)
    col_start = max(warped.bounds.col_min, 0)
    col_end = min(warped.bounds.col_max, target_cols)
    intersection = 0
    in_frame_support = 0
    if row_start < row_end and col_start < col_end:
        predicted_overlap = warped.grid[
            :,
            row_start - warped.bounds.row_min : row_end - warped.bounds.row_min,
            col_start - warped.bounds.col_min : col_end - warped.bounds.col_min,
        ]
        target_overlap = target[:, row_start:row_end, col_start:col_end]
        intersection = int(np.count_nonzero(predicted_overlap & target_overlap))
        in_frame_support = int(np.count_nonzero(predicted_overlap))
    predicted_support = warped.total_support
    return RasterScore(
        intersection=intersection,
        union=predicted_support + int(np.count_nonzero(target)) - intersection,
        predicted_support=predicted_support,
        target_support=int(np.count_nonzero(target)),
        in_frame_support=in_frame_support,
        out_of_frame_support=predicted_support - in_frame_support,
    )


def _validate_direction_vectors(vectors: NDArray[np.float32], name: str) -> None:
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError(f"{name} must have shape (count, 2)")
    if not np.issubdtype(vectors.dtype, np.floating):
        raise ValueError(f"{name} must have a floating dtype")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name} must contain only finite values")


def rotate_direction_vectors(
    vectors: NDArray[np.float32], angle_degrees: float
) -> NDArray[np.float32]:
    """Apply a signed physical grid rotation to stored direction vectors."""
    _validate_direction_vectors(vectors, "vectors")
    angle = _require_finite_real(angle_degrees, "angle_degrees")
    cosine, sine = _rotation_coefficients(-angle)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float32)
    return np.asarray(np.asarray(vectors, dtype=np.float32) @ rotation.T, dtype=np.float32)


def direction_cosine(
    predicted: NDArray[np.float32], target: NDArray[np.float32]
) -> float:
    """Mean cosine over only direction pairs for which both vectors are nonzero."""
    _validate_direction_vectors(predicted, "predicted")
    _validate_direction_vectors(target, "target")
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have the same shape")
    predicted_norms = np.linalg.norm(predicted, axis=1)
    target_norms = np.linalg.norm(target, axis=1)
    eligible = (predicted_norms > 0.0) & (target_norms > 0.0)
    if not np.any(eligible):
        return 0.0
    cosines = (
        np.sum(predicted[eligible] * target[eligible], axis=1)
        / (predicted_norms[eligible] * target_norms[eligible])
    )
    return float(np.mean(np.clip(cosines, -1.0, 1.0)))


def select_best_angle(
    grid: NDArray[np.bool_],
    target: NDArray[np.bool_],
    pivot: tuple[float, float],
    angles: tuple[float, ...],
) -> tuple[AngleScore, tuple[AngleScore, ...]]:
    """Score angles in caller order, retaining the first angle on an IoU tie."""
    if not angles:
        raise ValueError("angles must not be empty")
    scores: list[AngleScore] = []
    best: AngleScore | None = None
    for angle in angles:
        warped = warp_grid_about_pivot(grid, pivot, angle)
        raster = score_warped_grid(warped, target)
        support_ratio = (
            0.0
            if warped.input_support == 0
            else warped.total_support / warped.input_support
        )
        score = AngleScore(angle, raster, support_ratio)
        scores.append(score)
        if best is None or score.raster.iou > best.raster.iou:
            best = score
    if best is None:
        raise RuntimeError("non-empty angles must produce a best score")
    return best, tuple(scores)
