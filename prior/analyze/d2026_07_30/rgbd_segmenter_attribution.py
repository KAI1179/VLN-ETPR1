"""Stored-artifact attribution for the sealed ESANet object-grid result."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import Enum, IntEnum
from typing import Optional, Sequence, Tuple, cast

import numpy as np
from tap import Tap

from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    PRIMARY_CATEGORY_INDICES,
    RAW_INDEX_SHA256,
    ObservationStatus,
    _require_projection_labels,
    run_p53_validation_subprocess,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
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
    "GeometryLeakageReport",
    "PixelContributorLedger",
    "SemanticProjectionGeometry",
    "main",
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
        cell_rows = self.target_cell_rc[..., 0]
        cell_columns = self.target_cell_rc[..., 1]
        projectable = self.endpoint_valid & self.in_bounds
        for category in PRIMARY_CATEGORY_INDICES:
            union = target[category] | prediction[category]
            for row, column in np.argwhere(union):
                at_cell = (
                    projectable & (cell_rows == row) & (cell_columns == column)
                )
                gt = at_cell & (self.gt_labels == category)
                predicted = at_cell & (self.mapped_labels == category)
                correct = gt & predicted
                target_present = bool(target[category, row, column])
                prediction_present = bool(prediction[category, row, column])
                if target_present and prediction_present:
                    status = CategoryCellStatus.TP
                    missed = gt & (self.mapped_labels != category)
                    wrong = predicted & (self.gt_labels != category)
                    effect = (
                        CategoryCellEffect.RESCUED_TP
                        if np.any(missed) or np.any(wrong)
                        else CategoryCellEffect.CLEAN_TP
                    )
                elif prediction_present:
                    status = CategoryCellStatus.FP
                    effect = CategoryCellEffect.AMPLIFIED_FP
                else:
                    status = CategoryCellStatus.FN
                    mapped_on_gt = self.mapped_labels[gt]
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
                        gt_pixel_support=int(np.count_nonzero(gt)),
                        predicted_pixel_support=int(np.count_nonzero(predicted)),
                        correct_pixel_support=int(np.count_nonzero(correct)),
                        gt_view_support=int(np.count_nonzero(np.any(gt, axis=(1, 2)))),
                        predicted_view_support=int(
                            np.count_nonzero(np.any(predicted, axis=(1, 2)))
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


def run_geometry_leakage_proof() -> GeometryLeakageReport:
    """Run the frozen four-route GT-erasure proof over all accepted predictions."""

    root = _repository_root()
    strict_read_bytes(
        root / _P6_REPORT,
        _P6_REPORT_SHA256,
        "sealed P6.1 projector report",
    )
    p53_attestation = run_p53_validation_subprocess(_p53_launch(root))
    raw_values, cohort = _load_raw_observations(root)
    accepted = accept_candidate_package(
        root / _CANDIDATE_ROOT,
        expected_manifest_sha256=_CANDIDATE_MANIFEST_SHA256,
        trusted_cohort=cohort,
        mapping_authority=_mapping_authority(root),
    )
    if len(raw_values) != 50 or len(accepted.predictions) != 50:
        raise ValueError("P7.1 requires exactly 50 aligned accepted observations")
    if accepted.manifest.summary.p53_attestation != p53_attestation:
        raise ValueError("P7.0 fresh P5.3 attestation differs from candidate evidence")
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
            raise ValueError("P7.0 raw and accepted candidate identities differ")

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


def main(argv: Optional[Sequence[str]] = None) -> GeometryLeakageReport:
    SegmenterAttributionArgs(underscores_to_dashes=True).parse_args(argv)
    report = run_geometry_leakage_proof()
    print(report.canonical_bytes().decode("utf-8"), end="")
    return report


if __name__ == "__main__":
    main()
