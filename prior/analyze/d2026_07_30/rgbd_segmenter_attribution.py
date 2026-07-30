"""Stored-artifact attribution for the sealed ESANet object-grid result."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import Enum, IntEnum
from pathlib import Path
from typing import Optional, Sequence, Tuple, cast

import numpy as np
from tap import Tap

from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    PRIMARY_CATEGORY_INDICES,
    RAW_INDEX_SHA256,
    ObservationStatus,
    TrustedCohort,
    _require_projection_labels,
    run_p53_validation_subprocess,
    score_observation,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    AcceptedCandidatePackage,
    accept_candidate_package,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    RawFrameArrays,
    strict_read_bytes,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_esanet_benchmark import _p53_launch
from prior.analyze.d2026_07_30.rgbd_semantic_projector import (
    _CANDIDATE_MANIFEST_SHA256,
    _CANDIDATE_ROOT,
    _load_raw_observations,
    _mapping_authority,
    _repository_root,
    project_mapped_labels_semantic_only,
)
from vlnce_baselines.models.etp_llm.llm_grid_oracle_cache import (
    GRID_CELL_SIZE_M,
    GRID_SIZE,
    OracleSensorFrame,
    _world_hits,
)

_GRID_SHAPE = (27, GRID_SIZE, GRID_SIZE)
_P6_PAIR_TREE_SHA256 = (
    "7e7be242e6d46d0eadbdac450bb9b8e277e8fb238f95be3845ef322d6d7b3118"
)
_P6_REPORT = (
    "data/rgbd_segmenter_benchmark/experiments/"
    "esanet-r34-nbt1d-scenenet-semantic-projector-timing-v1/report.json"
)
_P6_REPORT_SHA256 = (
    "9509aa76d6d2adf0aebde08c6ad6fc825d11ea0d5ad118ce41303d3a50508d1a"
)

__all__ = (
    "CategoryCellEffect",
    "CategoryCellRow",
    "CategoryCellStatus",
    "DepthBin",
    "ContributorConservationReport",
    "GeometryLeakageReport",
    "PixelContributorLedger",
    "SemanticProjectionGeometry",
    "main",
    "run_contributor_conservation",
    "run_geometry_leakage_proof",
)


class DepthBin(IntEnum):
    INVALID = 0
    UP_TO_1 = 1
    UP_TO_2 = 2
    UP_TO_4 = 3
    UP_TO_6 = 4
    UP_TO_10 = 5
    SATURATED = 6


class CategoryCellStatus(str, Enum):
    TP = "TP"
    FP = "FP"
    FN = "FN"


class CategoryCellEffect(str, Enum):
    CLEAN_TP = "clean_tp"
    RESCUED_TP = "rescued_tp"
    AMPLIFIED_FP = "amplified_fp"
    FN_UNANIMOUS_ABSTENTION = "fn_unanimous_abstention"
    FN_UNANIMOUS_WRONG_CLASS = "fn_unanimous_wrong_class"
    FN_MIXED = "fn_mixed"


@dataclass(frozen=True)
class CategoryCellRow:
    category: int
    row: int
    column: int
    status: CategoryCellStatus
    effect: CategoryCellEffect
    gt_pixel_support: int
    predicted_pixel_support: int
    correct_pixel_support: int
    gt_view_support: int
    predicted_view_support: int


class SegmenterAttributionArgs(Tap):
    """The frozen P7 stored-artifact runner has no configurable inputs."""


@dataclass(frozen=True)
class GeometryLeakageReport:
    schema_version: int
    raw_index_sha256: str
    candidate_manifest_sha256: str
    candidate_tree_sha256: str
    p53_attestation_sha256: str
    benchmark_attestation_sha256: str
    p6_report_sha256: str
    p6_pair_tree_sha256: str
    observation_count: int
    comparison_count: int
    output_count: int
    output_tree_sha256: str

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                asdict(self),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")


@dataclass(frozen=True)
class ContributorConservationReport:
    schema_version: int
    observation_count: int
    view_count: int
    published_metric_match_count: int
    primary_tp_cell_count: int
    primary_fp_cell_count: int
    primary_fn_cell_count: int
    clean_tp_cell_count: int
    rescued_tp_cell_count: int
    amplified_fp_cell_count: int
    fn_unanimous_abstention_cell_count: int
    fn_unanimous_wrong_class_cell_count: int
    fn_mixed_cell_count: int
    grid_and_cell_tree_sha256: str

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                asdict(self),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")


def _owned_array(
    value: np.ndarray,
    *,
    label: str,
    dtype: np.dtype,
    shape: Tuple[int, ...],
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{label} has wrong projection schema")
    owned = np.array(value, dtype=dtype, order="C", copy=True)
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True)
class SemanticProjectionGeometry:
    """Only the geometry required to project semantic endpoint labels."""

    depth_m: np.ndarray
    sensor_positions: np.ndarray
    sensor_rotations_xyzw: np.ndarray
    sensor_hfov_degrees: np.ndarray
    target_origin_xz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "depth_m",
            _owned_array(
                self.depth_m,
                label="depth m",
                dtype=np.dtype("<f4"),
                shape=(12, 256, 256),
            ),
        )
        object.__setattr__(
            self,
            "sensor_positions",
            _owned_array(
                self.sensor_positions,
                label="sensor positions",
                dtype=np.dtype("<f8"),
                shape=(12, 3),
            ),
        )
        object.__setattr__(
            self,
            "sensor_rotations_xyzw",
            _owned_array(
                self.sensor_rotations_xyzw,
                label="sensor rotations xyzw",
                dtype=np.dtype("<f8"),
                shape=(12, 4),
            ),
        )
        object.__setattr__(
            self,
            "sensor_hfov_degrees",
            _owned_array(
                self.sensor_hfov_degrees,
                label="sensor hfov degrees",
                dtype=np.dtype("<f8"),
                shape=(),
            ),
        )
        object.__setattr__(
            self,
            "target_origin_xz",
            _owned_array(
                self.target_origin_xz,
                label="target origin xz",
                dtype=np.dtype("<f8"),
                shape=(2,),
            ),
        )
        if np.any(self.depth_m < 0) or np.any(self.depth_m > 10):
            raise ValueError("depth m has invalid projection range")
        if self.sensor_hfov_degrees.item() != 90:
            raise ValueError("sensor hfov degrees must equal 90")
        if not np.all(
            np.abs(np.linalg.norm(self.sensor_rotations_xyzw, axis=1) - 1) <= 5e-8
        ):
            raise ValueError("sensor rotations xyzw must be unit quaternions")

    @classmethod
    def from_raw_frame_arrays(
        cls,
        arrays: RawFrameArrays,
    ) -> "SemanticProjectionGeometry":
        if not isinstance(arrays, RawFrameArrays):
            raise TypeError("RawFrameArrays are required")
        return cls(
            depth_m=arrays.depth_m,
            sensor_positions=arrays.sensor_positions,
            sensor_rotations_xyzw=arrays.sensor_rotations_xyzw,
            sensor_hfov_degrees=arrays.sensor_hfov_degrees,
            target_origin_xz=arrays.target_origin_xz,
        )

    def project_mapped_labels(self, mapped_labels: np.ndarray) -> np.ndarray:
        if not isinstance(self, SemanticProjectionGeometry):
            raise TypeError("SemanticProjectionGeometry is required")
        _require_projection_labels(
            mapped_labels,
            allow_other=False,
            label="mapped labels",
        )
        return self._project_labels(mapped_labels)

    def project_oracle_labels(self, object_labels: np.ndarray) -> np.ndarray:
        if not isinstance(self, SemanticProjectionGeometry):
            raise TypeError("SemanticProjectionGeometry is required")
        _require_projection_labels(
            object_labels,
            allow_other=True,
            label="oracle object labels",
        )
        return self._project_labels(object_labels)

    def _project_labels(self, object_labels: np.ndarray) -> np.ndarray:
        semantic_grid = np.zeros(_GRID_SHAPE, dtype=np.bool_)
        target_origin = self.target_origin_xz
        for index in range(12):
            frame = OracleSensorFrame(
                depth_m=self.depth_m[index],
                object_categories=object_labels[index],
                region_categories=np.full((256, 256), -1, dtype="<i2"),
                sensor_position=cast(
                    Tuple[float, float, float],
                    tuple(self.sensor_positions[index]),
                ),
                sensor_rotation=cast(
                    Tuple[float, float, float, float],
                    tuple(self.sensor_rotations_xyzw[index]),
                ),
                hfov_degrees=float(self.sensor_hfov_degrees.item()),
            )
            world_points, objects, _, _ = _world_hits(frame)
            if world_points.size == 0:
                continue
            rows = np.floor(
                (world_points[:, 0] - target_origin[0]) / GRID_CELL_SIZE_M
            ).astype(np.int64)
            columns = np.floor(
                (world_points[:, 2] - target_origin[1]) / GRID_CELL_SIZE_M
            ).astype(np.int64)
            inside = (
                (0 <= rows)
                & (rows < GRID_SIZE)
                & (0 <= columns)
                & (columns < GRID_SIZE)
            )
            hits = inside & (objects > 0)
            semantic_grid[objects[hits], rows[hits], columns[hits]] = True
        return semantic_grid


@dataclass(frozen=True)
class PixelContributorLedger:
    """Columnar per-pixel evidence for one sealed observation."""

    ordinal: int
    observation_id: str
    scene_id: str
    sensor_yaw_degrees: np.ndarray
    depth_m: np.ndarray
    depth_bin: np.ndarray
    gt_labels: np.ndarray
    source_labels: np.ndarray
    mapped_labels: np.ndarray
    world_xz: np.ndarray
    target_cell_rc: np.ndarray
    endpoint_valid: np.ndarray
    in_bounds: np.ndarray

    def __post_init__(self) -> None:
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or not isinstance(self.observation_id, str)
            or not self.observation_id
            or not isinstance(self.scene_id, str)
            or not self.scene_id
        ):
            raise ValueError("contributor ledger identity is invalid")
        schemas = (
            (
                "sensor yaw degrees",
                self.sensor_yaw_degrees,
                np.dtype("<i2"),
                (12,),
            ),
            ("depth m", self.depth_m, np.dtype("<f4"), (12, 256, 256)),
            ("depth bin", self.depth_bin, np.dtype(np.uint8), (12, 256, 256)),
            ("GT labels", self.gt_labels, np.dtype("<i2"), (12, 256, 256)),
            (
                "source labels",
                self.source_labels,
                np.dtype("<i2"),
                (12, 256, 256),
            ),
            (
                "mapped labels",
                self.mapped_labels,
                np.dtype("<i2"),
                (12, 256, 256),
            ),
            ("world xz", self.world_xz, np.dtype("<f8"), (12, 256, 256, 2)),
            (
                "target cell rc",
                self.target_cell_rc,
                np.dtype("<i4"),
                (12, 256, 256, 2),
            ),
            (
                "endpoint valid",
                self.endpoint_valid,
                np.dtype(np.bool_),
                (12, 256, 256),
            ),
            (
                "in bounds",
                self.in_bounds,
                np.dtype(np.bool_),
                (12, 256, 256),
            ),
        )
        for label, value, dtype, shape in schemas:
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != dtype
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"{label} has wrong contributor schema")
            owned = np.array(value, dtype=dtype, order="C", copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, label.replace(" ", "_").replace("GT", "gt"), owned)
        if (
            not np.isfinite(self.depth_m).all()
            or not np.isfinite(self.world_xz).all()
            or np.any(self.depth_m < 0)
            or np.any(self.depth_m > 10)
            or np.any(self.depth_bin > DepthBin.SATURATED)
            or np.any(self.gt_labels < -1)
            or np.any(self.gt_labels > 26)
            or np.any(self.source_labels < 0)
            or np.any(self.source_labels >= 40)
            or np.any(self.mapped_labels < -1)
            or np.any(self.mapped_labels > 26)
            or np.any(self.mapped_labels == 16)
        ):
            raise ValueError("contributor ledger values are invalid")

    @staticmethod
    def depth_bins(depth_m: np.ndarray) -> np.ndarray:
        if (
            not isinstance(depth_m, np.ndarray)
            or not np.issubdtype(depth_m.dtype, np.floating)
            or not np.isfinite(depth_m).all()
            or np.any(depth_m < 0)
            or np.any(depth_m > 10)
        ):
            raise ValueError("depth m is invalid")
        bins = np.select(
            (
                depth_m == 0,
                depth_m <= 1,
                depth_m <= 2,
                depth_m <= 4,
                depth_m <= 6,
                depth_m < 10,
                depth_m == 10,
            ),
            tuple(int(value) for value in DepthBin),
        ).astype(np.uint8)
        return np.ascontiguousarray(bins)

    @classmethod
    def build(
        cls,
        *,
        ordinal: int,
        observation_id: str,
        scene_id: str,
        sensor_yaw_degrees: np.ndarray,
        geometry: SemanticProjectionGeometry,
        gt_labels: np.ndarray,
        source_labels: np.ndarray,
        mapped_labels: np.ndarray,
    ) -> "PixelContributorLedger":
        if not isinstance(geometry, SemanticProjectionGeometry):
            raise TypeError("SemanticProjectionGeometry is required")
        _require_projection_labels(
            gt_labels,
            allow_other=True,
            label="oracle object labels",
        )
        _require_projection_labels(
            mapped_labels,
            allow_other=False,
            label="mapped labels",
        )
        if (
            not isinstance(source_labels, np.ndarray)
            or source_labels.dtype != np.dtype("<i2")
            or source_labels.shape != (12, 256, 256)
            or not source_labels.flags.c_contiguous
            or np.any(source_labels < 0)
            or np.any(source_labels >= 40)
        ):
            raise ValueError("source labels have wrong contributor schema")
        if (
            not isinstance(sensor_yaw_degrees, np.ndarray)
            or sensor_yaw_degrees.dtype != np.dtype("<i2")
            or sensor_yaw_degrees.shape != (12,)
            or not sensor_yaw_degrees.flags.c_contiguous
        ):
            raise ValueError("sensor yaw degrees have wrong contributor schema")

        world_xz = np.zeros((12, 256, 256, 2), dtype="<f8")
        target_cell_rc = np.zeros((12, 256, 256, 2), dtype="<i4")
        endpoint_valid = np.zeros((12, 256, 256), dtype=np.bool_)
        in_bounds = np.zeros((12, 256, 256), dtype=np.bool_)
        for view in range(12):
            frame = OracleSensorFrame(
                depth_m=geometry.depth_m[view],
                object_categories=gt_labels[view],
                region_categories=np.full((256, 256), -1, dtype="<i2"),
                sensor_position=cast(
                    Tuple[float, float, float],
                    tuple(geometry.sensor_positions[view]),
                ),
                sensor_rotation=cast(
                    Tuple[float, float, float, float],
                    tuple(geometry.sensor_rotations_xyzw[view]),
                ),
                hfov_degrees=float(geometry.sensor_hfov_degrees.item()),
            )
            world_points, _, _, saturated = _world_hits(frame)
            pixel_rows, pixel_columns = np.nonzero(geometry.depth_m[view] > 0)
            if len(pixel_rows) != len(world_points):
                raise ValueError("world-hit pixel identities do not conserve")
            if not len(pixel_rows):
                continue
            rows = np.floor(
                (world_points[:, 0] - geometry.target_origin_xz[0])
                / GRID_CELL_SIZE_M
            ).astype(np.int64)
            columns = np.floor(
                (world_points[:, 2] - geometry.target_origin_xz[1])
                / GRID_CELL_SIZE_M
            ).astype(np.int64)
            if (
                np.any(rows < np.iinfo(np.int32).min)
                or np.any(rows > np.iinfo(np.int32).max)
                or np.any(columns < np.iinfo(np.int32).min)
                or np.any(columns > np.iinfo(np.int32).max)
            ):
                raise ValueError("target cell coordinate exceeds int32")
            world_xz[view, pixel_rows, pixel_columns] = world_points[:, (0, 2)]
            target_cell_rc[view, pixel_rows, pixel_columns, 0] = rows
            target_cell_rc[view, pixel_rows, pixel_columns, 1] = columns
            endpoint_valid[view, pixel_rows, pixel_columns] = ~saturated
            in_bounds[view, pixel_rows, pixel_columns] = (
                (0 <= rows)
                & (rows < GRID_SIZE)
                & (0 <= columns)
                & (columns < GRID_SIZE)
            )
        return cls(
            ordinal=ordinal,
            observation_id=observation_id,
            scene_id=scene_id,
            sensor_yaw_degrees=sensor_yaw_degrees,
            depth_m=geometry.depth_m,
            depth_bin=cls.depth_bins(geometry.depth_m),
            gt_labels=gt_labels,
            source_labels=source_labels,
            mapped_labels=mapped_labels,
            world_xz=world_xz,
            target_cell_rc=target_cell_rc,
            endpoint_valid=endpoint_valid,
            in_bounds=in_bounds,
        )

    def _grid_for(self, labels: np.ndarray) -> np.ndarray:
        grid = np.zeros(_GRID_SHAPE, dtype=np.bool_)
        contributes = self.endpoint_valid & self.in_bounds & (labels > 0)
        grid[
            labels[contributes],
            self.target_cell_rc[..., 0][contributes],
            self.target_cell_rc[..., 1][contributes],
        ] = True
        return grid

    def rebuild_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._grid_for(self.gt_labels), self._grid_for(self.mapped_labels)

    def category_cells(
        self,
        *,
        target: np.ndarray,
        prediction: np.ndarray,
    ) -> Tuple[CategoryCellRow, ...]:
        _require_exact_grid(target, target, label="target grid")
        _require_exact_grid(prediction, prediction, label="prediction grid")
        rows: list[CategoryCellRow] = []
        flat_projectable = (self.endpoint_valid & self.in_bounds).ravel()
        flat_cells = (
            self.target_cell_rc[..., 0].ravel().astype(np.int64) * GRID_SIZE
            + self.target_cell_rc[..., 1].ravel()
        )
        flat_gt = self.gt_labels.ravel()
        flat_mapped = self.mapped_labels.ravel()
        primary_lookup = np.zeros(_GRID_SHAPE[0], dtype=np.bool_)
        primary_lookup[np.asarray(PRIMARY_CATEGORY_INDICES)] = True

        def grouped_contributors(
            labels: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            selected = flat_projectable & (labels > 0)
            selected &= primary_lookup[np.maximum(labels, 0)]
            indices = np.flatnonzero(selected)
            keys = labels[indices].astype(np.int64) * GRID_SIZE**2
            keys += flat_cells[indices]
            order = np.argsort(keys, kind="stable")
            return keys[order], indices[order]

        gt_keys, gt_indices = grouped_contributors(flat_gt)
        predicted_keys, predicted_indices = grouped_contributors(flat_mapped)

        def contributors_for(
            keys: np.ndarray,
            indices: np.ndarray,
            key: int,
        ) -> np.ndarray:
            lower = int(np.searchsorted(keys, key, side="left"))
            upper = int(np.searchsorted(keys, key, side="right"))
            return indices[lower:upper]

        for category in PRIMARY_CATEGORY_INDICES:
            union = target[category] | prediction[category]
            for row, column in np.argwhere(union):
                key = (
                    int(category) * GRID_SIZE**2
                    + int(row) * GRID_SIZE
                    + int(column)
                )
                gt = contributors_for(gt_keys, gt_indices, key)
                predicted = contributors_for(
                    predicted_keys,
                    predicted_indices,
                    key,
                )
                correct = np.intersect1d(gt, predicted, assume_unique=True)
                target_present = bool(target[category, row, column])
                prediction_present = bool(prediction[category, row, column])
                if target_present and prediction_present:
                    status = CategoryCellStatus.TP
                    effect = (
                        CategoryCellEffect.RESCUED_TP
                        if np.any(flat_mapped[gt] != category)
                        or np.any(flat_gt[predicted] != category)
                        else CategoryCellEffect.CLEAN_TP
                    )
                elif prediction_present:
                    status = CategoryCellStatus.FP
                    effect = CategoryCellEffect.AMPLIFIED_FP
                else:
                    status = CategoryCellStatus.FN
                    mapped_on_gt = flat_mapped[gt]
                    if mapped_on_gt.size == 0:
                        raise ValueError("FN cell has no GT contributor")
                    if np.all(mapped_on_gt == -1):
                        effect = CategoryCellEffect.FN_UNANIMOUS_ABSTENTION
                    elif (
                        np.all(mapped_on_gt == mapped_on_gt[0])
                        and mapped_on_gt[0] >= 0
                        and mapped_on_gt[0] != category
                    ):
                        effect = CategoryCellEffect.FN_UNANIMOUS_WRONG_CLASS
                    else:
                        effect = CategoryCellEffect.FN_MIXED
                rows.append(
                    CategoryCellRow(
                        category=category,
                        row=int(row),
                        column=int(column),
                        status=status,
                        effect=effect,
                        gt_pixel_support=int(gt.size),
                        predicted_pixel_support=int(predicted.size),
                        correct_pixel_support=int(correct.size),
                        gt_view_support=int(
                            np.unique(gt // (256 * 256)).size
                        ),
                        predicted_view_support=int(
                            np.unique(predicted // (256 * 256)).size
                        ),
                    )
                )
        return tuple(rows)


_POISON_FIELDS = (
    "object_categories",
    "region_categories",
    "ego_observed_mask",
    "ego_free_mask",
    "target_observed_mask",
    "target_free_mask",
)
_GEOMETRY_FIELDS = (
    "depth_m",
    "sensor_positions",
    "sensor_rotations_xyzw",
    "sensor_hfov_degrees",
    "target_origin_xz",
)


def _poison_array(value: np.ndarray) -> np.ndarray:
    if value.dtype == np.dtype(np.bool_):
        poisoned = np.ascontiguousarray(~value)
    elif value.dtype == np.dtype("<i2"):
        poisoned = np.ascontiguousarray(np.where(value == -1, 0, -1).astype("<i2"))
    else:
        raise ValueError("unsupported GT-only poison dtype")
    if poisoned.shape != value.shape or np.array_equal(poisoned, value):
        raise ValueError("GT-only poison did not change the source array")
    return poisoned


def _poison_variants(arrays: RawFrameArrays) -> Tuple[Tuple[str, RawFrameArrays], ...]:
    variants = tuple(
        (
            field,
            replace(arrays, **{field: _poison_array(getattr(arrays, field))}),
        )
        for field in _POISON_FIELDS
    )
    joint_values = {
        field: _poison_array(getattr(arrays, field)) for field in _POISON_FIELDS
    }
    return variants + (("joint", replace(arrays, **joint_values)),)


def _require_geometry_unchanged(
    expected: RawFrameArrays,
    actual: RawFrameArrays,
    *,
    label: str,
) -> None:
    for field in _GEOMETRY_FIELDS:
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if (
            expected_value.dtype != actual_value.dtype
            or expected_value.shape != actual_value.shape
            or expected_value.tobytes(order="C") != actual_value.tobytes(order="C")
        ):
            raise ValueError(f"{label} changed projection geometry field {field}")


def _require_exact_grid(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    label: str,
) -> str:
    if (
        expected.dtype != np.dtype(np.bool_)
        or actual.dtype != expected.dtype
        or expected.shape != _GRID_SHAPE
        or actual.shape != expected.shape
        or not expected.flags.c_contiguous
        or not actual.flags.c_contiguous
        or not np.array_equal(actual, expected)
        or actual.tobytes(order="C") != expected.tobytes(order="C")
    ):
        raise ValueError(f"{label} differs from accepted semantic projection")
    return hashlib.sha256(actual.tobytes(order="C")).hexdigest()


def _observation_leakage_lines(
    ordinal: int,
    arrays: RawFrameArrays,
    mapped_labels: np.ndarray,
) -> Tuple[bytes, ...]:
    baseline = project_mapped_labels_semantic_only(mapped_labels, arrays)
    lines = [
        f"{ordinal:02d} current-pristine "
        f"{hashlib.sha256(baseline.tobytes(order='C')).hexdigest()}\n".encode("ascii")
    ]
    variants = _poison_variants(arrays)
    for name, poisoned in variants:
        _require_geometry_unchanged(arrays, poisoned, label=name)
        digest = _require_exact_grid(
            baseline,
            project_mapped_labels_semantic_only(mapped_labels, poisoned),
            label=f"observation {ordinal} current-{name}",
        )
        lines.append(f"{ordinal:02d} current-{name} {digest}\n".encode("ascii"))

    pristine_geometry = SemanticProjectionGeometry.from_raw_frame_arrays(arrays)
    digest = _require_exact_grid(
        baseline,
        pristine_geometry.project_mapped_labels(mapped_labels),
        label=f"observation {ordinal} typed-pristine",
    )
    lines.append(f"{ordinal:02d} typed-pristine {digest}\n".encode("ascii"))
    for name, poisoned in variants:
        digest = _require_exact_grid(
            baseline,
            SemanticProjectionGeometry.from_raw_frame_arrays(
                poisoned
            ).project_mapped_labels(mapped_labels),
            label=f"observation {ordinal} typed-{name}",
        )
        lines.append(f"{ordinal:02d} typed-{name} {digest}\n".encode("ascii"))
    return tuple(lines)


def _load_accepted_inputs(
    root: Path,
) -> Tuple[
    Tuple[RawFrameArrays, ...],
    TrustedCohort,
    AcceptedCandidatePackage,
]:
    raw_values, cohort = _load_raw_observations(root)
    accepted = accept_candidate_package(
        root / _CANDIDATE_ROOT,
        expected_manifest_sha256=_CANDIDATE_MANIFEST_SHA256,
        trusted_cohort=cohort,
        mapping_authority=_mapping_authority(root),
    )
    if (
        len(raw_values) != 50
        or len(accepted.observations) != 50
        or len(accepted.predictions) != 50
    ):
        raise ValueError("P7 requires exactly 50 aligned accepted observations")
    for identity, observation in zip(cohort.identities, accepted.observations):
        if (
            identity
            != (
                observation.ordinal,
                observation.observation_id,
                observation.scene_id,
            )
            or observation.status is not ObservationStatus.PASS
            or observation.failure_code is not None
        ):
            raise ValueError("P7 raw and accepted candidate identities differ")
    return raw_values, cohort, accepted


def run_geometry_leakage_proof() -> GeometryLeakageReport:
    """Run the frozen four-route GT-erasure proof over all accepted predictions."""

    root = _repository_root()
    strict_read_bytes(
        root / _P6_REPORT,
        _P6_REPORT_SHA256,
        "sealed P6.1 projector report",
    )
    p53_attestation = run_p53_validation_subprocess(_p53_launch(root))
    raw_values, _, accepted = _load_accepted_inputs(root)
    if accepted.manifest.summary.p53_attestation != p53_attestation:
        raise ValueError("P7.0 fresh P5.3 attestation differs from candidate evidence")

    lines = tuple(
        line
        for ordinal, (arrays, prediction) in enumerate(
            zip(raw_values, accepted.predictions)
        )
        for line in _observation_leakage_lines(
            ordinal,
            arrays,
            prediction.prediction.mapped_labels,
        )
    )
    output_count = len(lines)
    if output_count != 800:
        raise ValueError("P7.1 requires exactly 16 proof outputs per observation")
    return GeometryLeakageReport(
        schema_version=1,
        raw_index_sha256=RAW_INDEX_SHA256,
        candidate_manifest_sha256=_CANDIDATE_MANIFEST_SHA256,
        candidate_tree_sha256=accepted.validation.tree_sha256,
        p53_attestation_sha256=p53_attestation.sha256,
        benchmark_attestation_sha256=(
            accepted.manifest.summary.benchmark_attestation.sha256
        ),
        p6_report_sha256=_P6_REPORT_SHA256,
        p6_pair_tree_sha256=_P6_PAIR_TREE_SHA256,
        observation_count=50,
        comparison_count=750,
        output_count=output_count,
        output_tree_sha256=hashlib.sha256(b"".join(lines)).hexdigest(),
    )


def run_contributor_conservation() -> ContributorConservationReport:
    """Rebuild every grid from per-pixel contributors and conserve published counts."""

    root = _repository_root()
    raw_values, _, accepted = _load_accepted_inputs(root)
    tree_lines: list[bytes] = []
    rows_by_status = {status: 0 for status in CategoryCellStatus}
    rows_by_effect = {effect: 0 for effect in CategoryCellEffect}
    for arrays, observation, accepted_prediction in zip(
        raw_values,
        accepted.observations,
        accepted.predictions,
    ):
        geometry = SemanticProjectionGeometry.from_raw_frame_arrays(arrays)
        ledger = PixelContributorLedger.build(
            ordinal=observation.ordinal,
            observation_id=observation.observation_id,
            scene_id=observation.scene_id,
            sensor_yaw_degrees=arrays.sensor_yaw_degrees,
            geometry=geometry,
            gt_labels=arrays.object_categories,
            source_labels=accepted_prediction.prediction.source_labels,
            mapped_labels=accepted_prediction.prediction.mapped_labels,
        )
        target, prediction = ledger.rebuild_grids()
        _require_exact_grid(
            project_mapped_labels_semantic_only(
                accepted_prediction.prediction.mapped_labels,
                arrays,
            ),
            prediction,
            label=f"observation {observation.ordinal} prediction contributors",
        )
        _require_exact_grid(
            geometry.project_oracle_labels(arrays.object_categories),
            target,
            label=f"observation {observation.ordinal} target contributors",
        )
        metrics = score_observation(
            ordinal=observation.ordinal,
            scene_id=observation.scene_id,
            prediction=prediction,
            target=target,
        )
        if metrics != observation.metrics:
            raise ValueError(
                f"observation {observation.ordinal} contributor metrics differ"
            )
        tree_lines.append(
            (
                f"{observation.ordinal:02d} grids "
                f"{hashlib.sha256(target.tobytes(order='C')).hexdigest()} "
                f"{hashlib.sha256(prediction.tobytes(order='C')).hexdigest()}\n"
            ).encode("ascii")
        )
        category_cells = ledger.category_cells(
            target=target,
            prediction=prediction,
        )
        status_counts = {
            status: sum(row.status is status for row in category_cells)
            for status in CategoryCellStatus
        }
        if (
            status_counts[CategoryCellStatus.TP] != metrics.primary.counts.tp
            or status_counts[CategoryCellStatus.FP] != metrics.primary.counts.fp
            or status_counts[CategoryCellStatus.FN] != metrics.primary.counts.fn
        ):
            raise ValueError(
                f"observation {observation.ordinal} category cells do not conserve"
            )
        for row in category_cells:
            rows_by_status[row.status] += 1
            rows_by_effect[row.effect] += 1
            tree_lines.append(
                (
                    f"{observation.ordinal:02d} cell {row.category:02d} "
                    f"{row.row:02d} {row.column:02d} {row.status.value} "
                    f"{row.effect.value} {row.gt_pixel_support} "
                    f"{row.predicted_pixel_support} {row.correct_pixel_support} "
                    f"{row.gt_view_support} {row.predicted_view_support}\n"
                ).encode("ascii")
            )

    tp = rows_by_status[CategoryCellStatus.TP]
    fp = rows_by_status[CategoryCellStatus.FP]
    fn = rows_by_status[CategoryCellStatus.FN]
    if (tp, fp, fn) != (1222, 3961, 4898):
        raise ValueError("P7.2 pooled primary counts differ from the published result")
    if (
        rows_by_effect[CategoryCellEffect.CLEAN_TP]
        + rows_by_effect[CategoryCellEffect.RESCUED_TP]
        != tp
        or rows_by_effect[CategoryCellEffect.AMPLIFIED_FP] != fp
        or rows_by_effect[CategoryCellEffect.FN_UNANIMOUS_ABSTENTION]
        + rows_by_effect[CategoryCellEffect.FN_UNANIMOUS_WRONG_CLASS]
        + rows_by_effect[CategoryCellEffect.FN_MIXED]
        != fn
    ):
        raise ValueError("P7.2 category-cell effects do not partition published counts")
    return ContributorConservationReport(
        schema_version=1,
        observation_count=50,
        view_count=600,
        published_metric_match_count=50,
        primary_tp_cell_count=tp,
        primary_fp_cell_count=fp,
        primary_fn_cell_count=fn,
        clean_tp_cell_count=rows_by_effect[CategoryCellEffect.CLEAN_TP],
        rescued_tp_cell_count=rows_by_effect[CategoryCellEffect.RESCUED_TP],
        amplified_fp_cell_count=rows_by_effect[CategoryCellEffect.AMPLIFIED_FP],
        fn_unanimous_abstention_cell_count=rows_by_effect[
            CategoryCellEffect.FN_UNANIMOUS_ABSTENTION
        ],
        fn_unanimous_wrong_class_cell_count=rows_by_effect[
            CategoryCellEffect.FN_UNANIMOUS_WRONG_CLASS
        ],
        fn_mixed_cell_count=rows_by_effect[CategoryCellEffect.FN_MIXED],
        grid_and_cell_tree_sha256=hashlib.sha256(b"".join(tree_lines)).hexdigest(),
    )


def main(argv: Optional[Sequence[str]] = None) -> GeometryLeakageReport:
    SegmenterAttributionArgs(underscores_to_dashes=True).parse_args(argv)
    report = run_geometry_leakage_proof()
    print(report.canonical_bytes().decode("utf-8"), end="")
    return report


if __name__ == "__main__":
    main()
