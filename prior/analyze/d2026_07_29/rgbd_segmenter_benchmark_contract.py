"""Typed, fail-closed contracts for the frozen RGB-D segmenter benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
from importlib import metadata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Optional, Protocol, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as functional

from prior.analyze.d2026_07_29 import rgbd_segmenter_raw_frame_package
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    FileRecord,
    IndexRow,
    RawFrameArrays,
    parse_index_bytes,
    parse_raw_frame_npz_bytes,
    validate_raw_frame_directory,
)
from vlnce_baselines.models.etp_llm.llm_grid_oracle_cache import (
    OracleSensorFrame,
    project_oracle_frames,
)

RAW_FRAME_ROOT = Path("data/rgbd_segmenter_benchmark/r2r-val-unseen-50-raw-v1")
RAW_PRODUCER_COMMIT = "923d44bc53ced1f4685e62fd8086a01056d513c5"
RAW_MANIFEST_SHA256 = "7aa278dd3b511bbf38aa647e3294a7fb9476b35bf7a2eb505a5bf3d2cf6c1c87"
RAW_INDEX_SHA256 = "38a68369f72654271648589c0d19508831a3b484473fe382dadeee457db04d17"
RAW_MANIFEST_BYTE_LENGTH = 56_531
RAW_INDEX_BYTE_LENGTH = 243_340
RAW_GPU_UUID = "GPU-9224ffcb-5f88-2ece-ae3f-fca02cf43533"
RAW_VALIDATOR_SOURCE_SHA256 = (
    "eb7fcbd29e6f0f3ed56f69b8e40af20b25ba9769567174bc843da7e54043c50a"
)
DEFAULT_NYU40_MAPPING_PATH = Path(__file__).with_name(
    "rgbd_segmenter_nyu40_mapping.json"
)
NYU40_MAPPING_SHA256 = (
    "133ebb300e5dc01f6176e4eb01a48569c3eec635a90fea2b8a602b53e50bc395"
)
BOOTSTRAP_MATRIX_SHA256 = (
    "a89573633a8efd5dac5ffe03f292491a8aba5202ff2951f7ac186f07f2c07f99"
)
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SCENE_COUNT = 11
MIN_COVERED_CATEGORIES = 14
MIN_SUPPORT_COVERAGE = 0.8
MIN_MEAN_IOU = 0.15
MIN_MEAN_F1 = 0.25
MAX_LATENCY_P95_SECONDS = 1.0
MAX_ABSOLUTE_RESERVED_BYTES = 16 * 1024**3
_MAX_RAW_FILE_BYTES = 64 * 1024 * 1024
_MAX_TREE_ENTRIES = 64
_MAX_TREE_DEPTH = 3
_HASH = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_CANONICAL_NAMES = (
    "void",
    "chair",
    "door",
    "table",
    "cushion",
    "sofa",
    "bed",
    "plant",
    "sink",
    "toilet",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "appliances",
    "structure",
    "other",
    "free-space",
    "picture",
    "cabinet",
    "chest_of_drawers",
    "stool",
    "towel",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes",
)
PRIMARY_CATEGORY_INDICES = (*range(1, 15), *range(18, 27))
_NYU40_SOURCE_NAMES = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refridgerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
)


class MappingKind(str, Enum):
    DIRECT = "direct"
    SYNONYM = "synonym"
    MANY_TO_ONE = "many_to_one"
    DIAGNOSTIC = "diagnostic"
    IGNORED = "ignored"
    UNMAPPED = "unmapped"


@dataclass(frozen=True)
class MappingEntry:
    source_index: int
    source_name: str
    kind: MappingKind
    canonical_index: int | None
    canonical_name: str | None

    def __post_init__(self) -> None:
        if type(self.source_index) is not int or self.source_index < 0:
            raise ValueError("source index must be a nonnegative integer")
        if not isinstance(self.source_name, str) or not self.source_name:
            raise ValueError("source name must be non-empty")
        if not isinstance(self.kind, MappingKind):
            raise ValueError("mapping kind must be a MappingKind")
        if self.kind in {MappingKind.IGNORED, MappingKind.UNMAPPED}:
            if self.canonical_index is not None or self.canonical_name is not None:
                raise ValueError("ignored mapping must abstain")
            return
        if (
            type(self.canonical_index) is not int
            or not 0 <= self.canonical_index < len(_CANONICAL_NAMES)
            or self.canonical_name != _CANONICAL_NAMES[self.canonical_index]
        ):
            raise ValueError("mapping target is not a canonical category")
        if self.canonical_index == 16:
            raise ValueError("mapping to canonical other is forbidden")
        if self.kind is MappingKind.DIAGNOSTIC and self.canonical_index not in {
            15,
            17,
        }:
            raise ValueError("diagnostic mapping must target structure or free-space")
        if self.kind is not MappingKind.DIAGNOSTIC and self.canonical_index in {
            0,
            15,
            17,
        }:
            raise ValueError("deployable mapping cannot target diagnostic categories")


@dataclass(frozen=True)
class SegmenterInput:
    """The only observation data visible to a candidate adapter."""

    rgb: torch.Tensor
    depth_m: torch.Tensor

    def __post_init__(self) -> None:
        _require_tensor(
            self.rgb,
            label="RGB",
            dtype=torch.uint8,
            shape=(12, 256, 256, 3),
            pinned=True,
        )
        _require_tensor(
            self.depth_m,
            label="depth",
            dtype=torch.float32,
            shape=(12, 256, 256),
            pinned=True,
        )
        if not bool(torch.isfinite(self.depth_m).all()):
            raise ValueError("depth must be finite")
        if bool((self.depth_m < 0).any()) or bool((self.depth_m > 10).any()):
            raise ValueError("depth must be in [0, 10] metres")


@dataclass(frozen=True)
class SpatialTransform:
    raw_height: int
    raw_width: int
    resized_height: int
    resized_width: int
    model_height: int
    model_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    def __post_init__(self) -> None:
        values = (
            self.raw_height,
            self.raw_width,
            self.resized_height,
            self.resized_width,
            self.model_height,
            self.model_width,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(
                "spatial dimensions and padding must be nonnegative integers"
            )
        if (
            min(
                self.raw_height,
                self.raw_width,
                self.resized_height,
                self.resized_width,
                self.model_height,
                self.model_width,
            )
            < 1
        ):
            raise ValueError("spatial dimensions must be positive")
        if (
            self.resized_height + self.pad_top + self.pad_bottom != self.model_height
            or self.resized_width + self.pad_left + self.pad_right != self.model_width
        ):
            raise ValueError("spatial padding does not produce the model size")
        if self.pad_bottom - self.pad_top not in {0, 1} or (
            self.pad_right - self.pad_left not in {0, 1}
        ):
            raise ValueError(
                "spatial padding must be symmetric with odd extra bottom/right"
            )
        expected = _spatial_values(
            self.raw_height, self.raw_width, self.model_height, self.model_width
        )
        if (
            self.resized_height,
            self.resized_width,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
        ) != (*expected,):
            raise ValueError(
                "spatial transform differs from the frozen resize contract"
            )

    @classmethod
    def from_sizes(
        cls,
        *,
        raw_height: int,
        raw_width: int,
        model_height: int,
        model_width: int,
    ) -> "SpatialTransform":
        if any(
            type(value) is not int or value < 1
            for value in (raw_height, raw_width, model_height, model_width)
        ):
            raise ValueError("spatial dimensions must be positive integers")
        (
            resized_height,
            resized_width,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        ) = _spatial_values(raw_height, raw_width, model_height, model_width)
        return cls(
            raw_height=raw_height,
            raw_width=raw_width,
            resized_height=resized_height,
            resized_width=resized_width,
            model_height=model_height,
            model_width=model_width,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
        )


def _spatial_values(
    raw_height: int, raw_width: int, model_height: int, model_width: int
) -> Tuple[int, int, int, int, int, int]:
    scale = min(model_height / raw_height, model_width / raw_width)
    resized_height = min(model_height, max(1, math.floor(raw_height * scale + 0.5)))
    resized_width = min(model_width, max(1, math.floor(raw_width * scale + 0.5)))
    height_remainder = model_height - resized_height
    width_remainder = model_width - resized_width
    return (
        resized_height,
        resized_width,
        height_remainder // 2,
        height_remainder - height_remainder // 2,
        width_remainder // 2,
        width_remainder - width_remainder // 2,
    )


@dataclass(frozen=True)
class PreparedHostBatch:
    tensors: Tuple[torch.Tensor, ...]
    spatial_transform: SpatialTransform

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, tuple) or not self.tensors:
            raise ValueError("prepared host batch must contain tensors")
        for tensor in self.tensors:
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise ValueError("prepared host tensors must be CPU tensors")


@dataclass(frozen=True)
class DeviceBatch:
    tensors: Tuple[torch.Tensor, ...]
    spatial_transform: SpatialTransform
    _producer_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._producer_token is not _DEVICE_BATCH_PRODUCER_TOKEN:
            raise ValueError("device batches must be produced by the benchmark runner")
        if not isinstance(self.tensors, tuple) or not self.tensors:
            raise ValueError("device batch must contain tensors")
        devices = {tensor.device for tensor in self.tensors}
        if len(devices) != 1:
            raise ValueError("device batch tensors must share one device")


_DEVICE_BATCH_PRODUCER_TOKEN = object()


def transfer_prepared_host_batch(
    batch: PreparedHostBatch,
    *,
    device: torch.device,
    non_blocking: bool,
) -> DeviceBatch:
    """The sole runner-owned conversion from prepared host to device batch."""

    if not isinstance(batch, PreparedHostBatch) or not isinstance(device, torch.device):
        raise ValueError("device transfer arguments are invalid")
    if type(non_blocking) is not bool:
        raise ValueError("non_blocking must be a bool")
    return DeviceBatch(
        tensors=tuple(
            tensor.to(device=device, non_blocking=non_blocking)
            for tensor in batch.tensors
        ),
        spatial_transform=batch.spatial_transform,
        _producer_token=_DEVICE_BATCH_PRODUCER_TOKEN,
    )


class SegmenterAdapter(Protocol):
    def preprocess_host(self, value: SegmenterInput) -> PreparedHostBatch: ...

    def infer(self, value: DeviceBatch) -> torch.Tensor: ...


class LicenseStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class CandidateCommitment:
    candidate_id: str
    synthetic: bool
    repository_url: str
    revision: str
    checkpoint_id: str
    checkpoint_url: str
    checkpoint_sha256: str
    source_dataset: str
    code_license_status: LicenseStatus
    weight_license_status: LicenseStatus
    environment_lock_path: str
    environment_lock_sha256: str
    rgb_units: str
    depth_units: str
    batch_views: int
    source_vocabulary: Tuple[str, ...]
    mapping_sha256: str
    rgb_interpolation: str
    rgb_coordinate_semantics: str
    depth_interpolation: str
    depth_coordinate_semantics: str
    normalization: str
    invalid_depth_policy: str
    rgb_padding_value: str
    depth_padding_value: str
    precision_mode: str

    def __post_init__(self) -> None:
        string_values = (
            self.candidate_id,
            self.repository_url,
            self.revision,
            self.checkpoint_id,
            self.checkpoint_url,
            self.checkpoint_sha256,
            self.source_dataset,
            self.environment_lock_path,
            self.environment_lock_sha256,
            self.mapping_sha256,
            self.rgb_units,
            self.depth_units,
            self.rgb_interpolation,
            self.rgb_coordinate_semantics,
            self.depth_interpolation,
            self.depth_coordinate_semantics,
            self.normalization,
            self.invalid_depth_policy,
            self.rgb_padding_value,
            self.depth_padding_value,
            self.precision_mode,
        )
        if any(not isinstance(value, str) or not value for value in string_values):
            raise ValueError("candidate commitment strings must be non-empty")
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", self.candidate_id) is None:
            raise ValueError("candidate ID is unsafe")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic flag must be a bool")
        if not isinstance(self.code_license_status, LicenseStatus) or not isinstance(
            self.weight_license_status, LicenseStatus
        ):
            raise ValueError("candidate license statuses are invalid")
        if self.batch_views != 12 or type(self.batch_views) is not int:
            raise ValueError("candidate batch commitment must be exactly twelve views")
        if (
            self.rgb_units != "uint8[0,255]"
            or self.depth_units != "float32-metres[0,10]"
        ):
            raise ValueError("candidate input units differ from the benchmark")
        if (
            not isinstance(self.source_vocabulary, tuple)
            or len(self.source_vocabulary) < 1
            or any(
                not isinstance(value, str) or not value
                for value in self.source_vocabulary
            )
        ):
            raise ValueError("candidate identity and source vocabulary are required")
        if len(set(self.source_vocabulary)) != len(self.source_vocabulary):
            raise ValueError("source vocabulary contains duplicates")
        if (
            _HASH.fullmatch(self.mapping_sha256) is None
            or _HASH.fullmatch(self.environment_lock_sha256) is None
        ):
            raise ValueError("candidate mapping or environment lock hash is invalid")
        lock_path = PurePosixPath(self.environment_lock_path)
        if (
            lock_path.is_absolute()
            or not lock_path.parts
            or any(part in {"", ".", ".."} for part in lock_path.parts)
            or "\x00" in self.environment_lock_path
            or "\\" in self.environment_lock_path
            or self.environment_lock_path == "NOT_APPLICABLE"
        ):
            raise ValueError("candidate environment lock path is unsafe")
        if self.synthetic:
            not_applicable = (
                self.repository_url,
                self.revision,
                self.checkpoint_id,
                self.checkpoint_url,
                self.checkpoint_sha256,
                self.source_dataset,
            )
            if any(value != "NOT_APPLICABLE" for value in not_applicable) or (
                self.code_license_status is not LicenseStatus.NOT_APPLICABLE
                or self.weight_license_status is not LicenseStatus.NOT_APPLICABLE
            ):
                raise ValueError("synthetic source and license fields must all be N/A")
        else:
            real_required = (
                self.repository_url,
                self.revision,
                self.checkpoint_id,
                self.checkpoint_url,
                self.checkpoint_sha256,
                self.source_dataset,
                self.environment_lock_path,
                self.environment_lock_sha256,
                self.rgb_interpolation,
                self.rgb_coordinate_semantics,
                self.depth_interpolation,
                self.depth_coordinate_semantics,
                self.normalization,
                self.invalid_depth_policy,
                self.rgb_padding_value,
                self.depth_padding_value,
                self.precision_mode,
            )
            if (
                any(value == "NOT_APPLICABLE" for value in real_required)
                or not self.repository_url.startswith("https://")
                or not self.checkpoint_url.startswith("https://")
                or _COMMIT.fullmatch(self.revision) is None
                or _HASH.fullmatch(self.checkpoint_sha256) is None
                or self.code_license_status is LicenseStatus.NOT_APPLICABLE
                or self.weight_license_status is LicenseStatus.NOT_APPLICABLE
            ):
                raise ValueError("real candidate source commitments are incomplete")


@dataclass(frozen=True)
class Prediction:
    source_labels: np.ndarray
    mapped_labels: np.ndarray

    def __post_init__(self) -> None:
        for label, value in (
            ("source", self.source_labels),
            ("mapped", self.mapped_labels),
        ):
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<i2")
                or value.shape != (12, 256, 256)
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"{label} labels have wrong schema")
        if np.any(self.source_labels < -1):
            raise ValueError("source labels must be -1 or nonnegative")
        if np.any(self.mapped_labels < -1) or np.any(self.mapped_labels > 26):
            raise ValueError("mapped labels must be in [-1, 26]")
        if np.any(self.mapped_labels == 16):
            raise ValueError("mapped labels cannot contain canonical other")


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0 for value in (self.tp, self.fp, self.fn)
        ):
            raise ValueError("confusion counts must be nonnegative integers")

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        if not isinstance(other, ConfusionCounts):
            return NotImplemented
        return ConfusionCounts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )


@dataclass(frozen=True)
class MetricEndpoint:
    counts: ConfusionCounts
    precision: Optional[float]
    recall: Optional[float]
    iou: Optional[float]
    f1: Optional[float]
    eligible: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.counts, ConfusionCounts)
            or type(self.eligible) is not bool
        ):
            raise ValueError("metric endpoint schema is invalid")
        expected = _endpoint_values(self.counts)
        if (
            self.precision,
            self.recall,
            self.iou,
            self.f1,
            self.eligible,
        ) != expected:
            raise ValueError("metric endpoint differs from its integer counts")

    @classmethod
    def from_counts(cls, counts: ConfusionCounts) -> "MetricEndpoint":
        precision, recall, iou, f1, eligible = _endpoint_values(counts)
        return cls(
            counts=counts,
            precision=precision,
            recall=recall,
            iou=iou,
            f1=f1,
            eligible=eligible,
        )


def _endpoint_values(
    counts: ConfusionCounts,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], bool]:
    precision_denominator = counts.tp + counts.fp
    recall_denominator = counts.tp + counts.fn
    union = counts.tp + counts.fp + counts.fn
    f1_denominator = 2 * counts.tp + counts.fp + counts.fn
    return (
        counts.tp / precision_denominator if precision_denominator else None,
        counts.tp / recall_denominator if recall_denominator else None,
        counts.tp / union if union else None,
        2 * counts.tp / f1_denominator if f1_denominator else None,
        union > 0,
    )


@dataclass(frozen=True)
class ObservationMetrics:
    ordinal: int
    scene_id: str
    primary: MetricEndpoint
    all_27: MetricEndpoint
    per_category: Tuple[MetricEndpoint, ...]

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 50:
            raise ValueError("metric ordinal must be in [0, 49]")
        if (
            not isinstance(self.scene_id, str)
            or not self.scene_id
            or "\x00" in self.scene_id
            or not isinstance(self.primary, MetricEndpoint)
            or not isinstance(self.all_27, MetricEndpoint)
            or not isinstance(self.per_category, tuple)
            or len(self.per_category) != 27
            or any(
                not isinstance(endpoint, MetricEndpoint)
                for endpoint in self.per_category
            )
        ):
            raise ValueError("observation metrics schema is invalid")
        all_counts = ConfusionCounts(0, 0, 0)
        primary_counts = ConfusionCounts(0, 0, 0)
        for index, endpoint in enumerate(self.per_category):
            all_counts += endpoint.counts
            if index in PRIMARY_CATEGORY_INDICES:
                primary_counts += endpoint.counts
        if self.all_27.counts != all_counts or self.primary.counts != primary_counts:
            raise ValueError("observation aggregate counts differ from categories")


@dataclass(frozen=True)
class MetricAggregate:
    observation_count: int
    eligible_observation_count: int
    empty_both_count: int
    target_empty_prediction_nonempty_count: int
    target_nonempty_prediction_empty_count: int
    mean_iou: Optional[float]
    mean_f1: Optional[float]
    pooled: MetricEndpoint
    scene_macro_iou: Optional[float]
    scene_macro_f1: Optional[float]

    def __post_init__(self) -> None:
        if (
            type(self.observation_count) is not int
            or self.observation_count < 1
            or type(self.eligible_observation_count) is not int
            or not 0 <= self.eligible_observation_count <= self.observation_count
            or type(self.empty_both_count) is not int
            or self.empty_both_count
            != self.observation_count - self.eligible_observation_count
            or type(self.target_empty_prediction_nonempty_count) is not int
            or not 0
            <= self.target_empty_prediction_nonempty_count
            <= self.observation_count
            or type(self.target_nonempty_prediction_empty_count) is not int
            or not 0
            <= self.target_nonempty_prediction_empty_count
            <= self.observation_count
            or not isinstance(self.pooled, MetricEndpoint)
        ):
            raise ValueError("metric aggregate schema is invalid")
        if (
            self.target_empty_prediction_nonempty_count
            + self.target_nonempty_prediction_empty_count
            > self.eligible_observation_count
            or self.pooled.eligible != (self.eligible_observation_count > 0)
            or (self.mean_iou is None) != (self.eligible_observation_count == 0)
            or (self.mean_f1 is None) != (self.eligible_observation_count == 0)
            or (self.scene_macro_iou is None) != (self.scene_macro_f1 is None)
        ):
            raise ValueError("metric aggregate eligibility is inconsistent")
        for value in (
            self.mean_iou,
            self.mean_f1,
            self.scene_macro_iou,
            self.scene_macro_f1,
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("aggregate metric must be null or in [0, 1]")


@dataclass(frozen=True)
class BenchmarkMetricSummary:
    primary: MetricAggregate
    all_27: MetricAggregate
    per_category: Tuple[MetricAggregate, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.primary, MetricAggregate)
            or not isinstance(self.all_27, MetricAggregate)
            or not isinstance(self.per_category, tuple)
            or len(self.per_category) != 27
            or any(
                not isinstance(aggregate, MetricAggregate)
                for aggregate in self.per_category
            )
        ):
            raise ValueError("benchmark metric summary schema is invalid")


class ObservationStatus(str, Enum):
    PASS = "PASS"
    FAILED = "FAILED"


class ObservationFailureCode(str, Enum):
    PREPROCESS_FAILURE = "PREPROCESS_FAILURE"
    INFERENCE_FAILURE = "INFERENCE_FAILURE"
    OUTPUT_SCHEMA_FAILURE = "OUTPUT_SCHEMA_FAILURE"


class AdapterObservationError(RuntimeError):
    def __init__(self, code: ObservationFailureCode) -> None:
        if not isinstance(code, ObservationFailureCode):
            raise ValueError("adapter failure code is invalid")
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True)
class ObservationResult:
    ordinal: int
    status: ObservationStatus
    failure_code: Optional[ObservationFailureCode]
    prediction: Prediction

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 50:
            raise ValueError("observation ordinal must be in [0, 49]")
        _require_outcome(self.status, self.failure_code)
        if self.status is ObservationStatus.FAILED and (
            np.any(self.prediction.source_labels != -1)
            or np.any(self.prediction.mapped_labels != -1)
        ):
            raise ValueError("failed observation prediction must be empty")


@dataclass(frozen=True)
class TimingSample:
    pass_index: int
    ordinal: int
    end_to_end: float
    source_labels_sha256: str
    mapped_labels_sha256: str
    status: ObservationStatus
    failure_code: Optional[ObservationFailureCode]

    def __post_init__(self) -> None:
        if (
            type(self.pass_index) is not int
            or self.pass_index not in {1, 2}
            or type(self.ordinal) is not int
            or not (0 <= self.ordinal < 50)
        ):
            raise ValueError("timing sample pass or ordinal is invalid")
        if (
            isinstance(self.end_to_end, bool)
            or not isinstance(self.end_to_end, (int, float))
            or not math.isfinite(self.end_to_end)
            or self.end_to_end < 0
        ):
            raise ValueError("timing sample must be finite and nonnegative")
        if (
            not isinstance(self.source_labels_sha256, str)
            or not isinstance(self.mapped_labels_sha256, str)
            or _HASH.fullmatch(self.source_labels_sha256) is None
            or _HASH.fullmatch(self.mapped_labels_sha256) is None
        ):
            raise ValueError("timing output hash is invalid")
        _require_outcome(self.status, self.failure_code)


@dataclass(frozen=True)
class ResourceMeasurement:
    baseline_allocated_bytes: int
    peak_allocated_bytes: int
    baseline_reserved_bytes: int
    peak_reserved_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.baseline_allocated_bytes,
            self.peak_allocated_bytes,
            self.baseline_reserved_bytes,
            self.peak_reserved_bytes,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("resource byte counts must be nonnegative integers")
        if (
            self.peak_allocated_bytes < self.baseline_allocated_bytes
            or self.peak_reserved_bytes < self.baseline_reserved_bytes
            or self.baseline_allocated_bytes > self.baseline_reserved_bytes
            or self.peak_allocated_bytes > self.peak_reserved_bytes
        ):
            raise ValueError("resource measurements are internally inconsistent")


@dataclass(frozen=True)
class TrustedCohort:
    identities: Tuple[Tuple[int, str, str], ...]

    def __post_init__(self) -> None:
        if (
            type(self.identities) is not tuple
            or len(self.identities) != 50
            or any(
                type(identity) is not tuple
                or len(identity) != 3
                or type(identity[0]) is not int
                or identity[0] != ordinal
                or not isinstance(identity[1], str)
                or not identity[1]
                or "\x00" in identity[1]
                or not isinstance(identity[2], str)
                or not identity[2]
                or "\x00" in identity[2]
                for ordinal, identity in enumerate(self.identities)
            )
            or len({identity[1] for identity in self.identities}) != 50
        ):
            raise ValueError("trusted cohort must contain exact ordered identities")
        scenes = tuple(sorted({identity[2] for identity in self.identities}))
        if len(scenes) != BOOTSTRAP_SCENE_COUNT:
            raise ValueError("trusted cohort must contain exactly 11 scenes")

    @property
    def scenes(self) -> Tuple[str, ...]:
        return tuple(sorted({identity[2] for identity in self.identities}))


@dataclass(frozen=True)
class EndpointRow:
    ordinal: int
    observation_id: str
    scene_id: str
    endpoint: Optional[float]
    status: ObservationStatus = ObservationStatus.PASS
    failure_code: Optional[ObservationFailureCode] = None

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 50:
            raise ValueError("endpoint ordinal must be in [0, 49]")
        if (
            not isinstance(self.observation_id, str)
            or not self.observation_id
            or "\x00" in self.observation_id
            or not isinstance(self.scene_id, str)
            or not self.scene_id
            or "\x00" in self.scene_id
        ):
            raise ValueError("endpoint identity is invalid")
        if self.endpoint is not None and (
            isinstance(self.endpoint, bool)
            or not isinstance(self.endpoint, (int, float))
            or not math.isfinite(self.endpoint)
            or not 0 <= self.endpoint <= 1
        ):
            raise ValueError("endpoint must be null or finite in [0, 1]")
        _require_outcome(self.status, self.failure_code)


@dataclass(frozen=True)
class RobustnessEstimate:
    point_estimate: float
    interval_low: float
    interval_high: float
    leave_one_scene_out_min: float
    leave_one_scene_out_max: float
    replicate_count: int = BOOTSTRAP_REPLICATES

    def __post_init__(self) -> None:
        _validate_estimate(
            self.point_estimate,
            self.interval_low,
            self.interval_high,
            self.leave_one_scene_out_min,
            self.leave_one_scene_out_max,
            self.replicate_count,
            lower_bound=0.0,
            upper_bound=1.0,
        )


@dataclass(frozen=True)
class ContrastEstimate:
    point_estimate: float
    interval_low: float
    interval_high: float
    leave_one_scene_out_min: float
    leave_one_scene_out_max: float
    replicate_count: int = BOOTSTRAP_REPLICATES

    def __post_init__(self) -> None:
        _validate_estimate(
            self.point_estimate,
            self.interval_low,
            self.interval_high,
            self.leave_one_scene_out_min,
            self.leave_one_scene_out_max,
            self.replicate_count,
            lower_bound=-1.0,
            upper_bound=1.0,
        )


def _validate_estimate(
    point_estimate: float,
    interval_low: float,
    interval_high: float,
    leave_one_scene_out_min: float,
    leave_one_scene_out_max: float,
    replicate_count: int,
    *,
    lower_bound: float,
    upper_bound: float,
) -> None:
    if type(replicate_count) is not int or replicate_count != BOOTSTRAP_REPLICATES:
        raise ValueError("estimate replicate count is invalid")
    numbers = (
        point_estimate,
        interval_low,
        interval_high,
        leave_one_scene_out_min,
        leave_one_scene_out_max,
    )
    if any(
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(number)
        or not lower_bound <= number <= upper_bound
        for number in numbers
    ):
        raise ValueError("estimate values are invalid")
    if numbers[1] > numbers[2] or numbers[3] > numbers[4]:
        raise ValueError("estimate bounds are reversed")


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class StaticCoverage:
    covered_category_count: int
    covered_support_count: int
    total_support_count: int

    def __post_init__(self) -> None:
        if (
            type(self.covered_category_count) is not int
            or not 0 <= self.covered_category_count <= len(PRIMARY_CATEGORY_INDICES)
            or type(self.covered_support_count) is not int
            or type(self.total_support_count) is not int
            or self.total_support_count < 1
            or not 0 <= self.covered_support_count <= self.total_support_count
        ):
            raise ValueError("static coverage counts are invalid")

    @property
    def support_ratio(self) -> float:
        return self.covered_support_count / self.total_support_count


@dataclass(frozen=True)
class ProvenanceChecks:
    clean_experiment_commit: bool
    complete_command: bool
    raw_package_pinned: bool
    cohort_pinned: bool
    candidate_revision_pinned: bool
    checkpoint_hash_pinned: bool
    environment_pinned: bool
    mapping_pinned: bool
    projector_pinned: bool
    preprocessing_pinned: bool
    precision_pinned: bool
    permission_evidence_pinned: bool

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in vars(self).values()):
            raise ValueError("provenance checks must be booleans")

    @property
    def passed(self) -> bool:
        return all(vars(self).values())


@dataclass(frozen=True)
class CandidateGateResult:
    coverage: GateStatus
    complete_rows: GateStatus
    quality: GateStatus
    latency: GateStatus
    resource: GateStatus
    license: GateStatus
    provenance: GateStatus
    overall: GateStatus

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if any(not isinstance(value, GateStatus) for value in values):
            raise ValueError("candidate gate status is invalid")
        expected = (
            GateStatus.PASS
            if all(value is GateStatus.PASS for value in values[:-1])
            else GateStatus.FAIL
        )
        if self.overall is not expected:
            raise ValueError("candidate overall gate is inconsistent")


@dataclass(frozen=True)
class LatencySummary:
    timing_comparable: bool
    unit: str
    sample_count: int
    p95_seconds: Optional[float]

    def __post_init__(self) -> None:
        if (
            type(self.timing_comparable) is not bool
            or not isinstance(self.unit, str)
            or not self.unit
            or type(self.sample_count) is not int
            or self.sample_count < 0
        ):
            raise ValueError("latency summary schema is invalid")
        if self.timing_comparable:
            if (
                self.unit != "seconds"
                or self.sample_count != 100
                or isinstance(self.p95_seconds, bool)
                or not isinstance(self.p95_seconds, (int, float))
                or not math.isfinite(self.p95_seconds)
                or self.p95_seconds < 0
            ):
                raise ValueError("comparable latency summary is invalid")
        elif self.p95_seconds is not None:
            raise ValueError("non-comparable latency cannot claim a P95")


def _require_outcome(
    status: ObservationStatus,
    failure_code: Optional[ObservationFailureCode],
) -> None:
    if not isinstance(status, ObservationStatus):
        raise ValueError("observation status is invalid")
    if (status is ObservationStatus.PASS) != (failure_code is None):
        raise ValueError("observation status and failure code are inconsistent")


def _require_tensor(
    value: object,
    *,
    label: str,
    dtype: torch.dtype,
    shape: Tuple[int, ...],
    pinned: bool,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{label} must be a tensor")
    if value.dtype is not dtype or tuple(value.shape) != shape:
        raise ValueError(f"{label} tensor has wrong dtype or shape")
    if value.device.type != "cpu":
        raise ValueError(f"{label} tensor must be on CPU")
    if pinned and not value.is_pinned():
        raise ValueError(f"{label} tensor must use pinned memory")
    if not value.is_contiguous():
        raise ValueError(f"{label} tensor must be C-contiguous")
    return value


def validate_source_logits(
    logits: torch.Tensor, *, source_class_count: int
) -> torch.Tensor:
    """Require finite floating logits with the exact batch/vocabulary axes."""

    if type(source_class_count) is not int or source_class_count < 1:
        raise ValueError("source class count must be a positive integer")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 4
        or tuple(logits.shape[:2]) != (12, source_class_count)
        or logits.shape[2] < 1
        or logits.shape[3] < 1
    ):
        raise ValueError("source logits have wrong shape")
    if not logits.is_floating_point():
        raise ValueError("source logits must be floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("source logits must be finite")
    return logits


def map_source_labels(
    labels: torch.Tensor, mapping: Sequence[MappingEntry]
) -> torch.Tensor:
    """Map hard source labels after argmax; ignored labels become ``-1``."""

    if (
        not isinstance(labels, torch.Tensor)
        or tuple(labels.shape) != (12, 256, 256)
        or labels.dtype
        not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    ):
        raise ValueError("source labels have wrong dtype or shape")
    if len(mapping) < 1 or tuple(entry.source_index for entry in mapping) != tuple(
        range(len(mapping))
    ):
        raise ValueError("mapping source indices are invalid")
    if bool((labels < 0).any()) or bool((labels >= len(mapping)).any()):
        raise ValueError("source labels are outside the mapping range")
    lookup = torch.tensor(
        [
            -1 if entry.canonical_index is None else entry.canonical_index
            for entry in mapping
        ],
        dtype=torch.int16,
        device=labels.device,
    )
    return lookup[labels.to(dtype=torch.int64)]


def restore_source_labels(
    logits: torch.Tensor,
    *,
    spatial_transform: SpatialTransform,
    source_class_count: int,
) -> torch.Tensor:
    """Restore source logits to shared pixels before lowest-index argmax."""

    if not isinstance(spatial_transform, SpatialTransform) or (
        spatial_transform.raw_height,
        spatial_transform.raw_width,
    ) != (256, 256):
        raise ValueError("spatial transform must restore the shared 256x256 grid")
    validate_source_logits(logits, source_class_count=source_class_count)
    restored = functional.interpolate(
        logits,
        size=(spatial_transform.model_height, spatial_transform.model_width),
        mode="bilinear",
        align_corners=False,
    )
    restored = restored[
        :,
        :,
        spatial_transform.pad_top : (
            spatial_transform.model_height - spatial_transform.pad_bottom
        ),
        spatial_transform.pad_left : (
            spatial_transform.model_width - spatial_transform.pad_right
        ),
    ]
    if tuple(restored.shape[2:]) != (
        spatial_transform.resized_height,
        spatial_transform.resized_width,
    ):
        raise ValueError("spatial unpadding produced the wrong resized shape")
    restored = functional.interpolate(
        restored,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )
    return torch.argmax(restored, dim=1).to(dtype=torch.int16)


def project_mapped_labels(
    mapped_labels: np.ndarray,
    arrays: RawFrameArrays,
) -> np.ndarray:
    """Project twelve mapped views and return only the 27 object channels."""

    if (
        not isinstance(mapped_labels, np.ndarray)
        or mapped_labels.dtype != np.dtype("<i2")
        or mapped_labels.shape != (12, 256, 256)
        or not mapped_labels.flags.c_contiguous
        or np.any(mapped_labels < -1)
        or np.any(mapped_labels > 26)
        or np.any(mapped_labels == 16)
    ):
        raise ValueError("mapped labels have wrong schema or range")
    frames = tuple(
        OracleSensorFrame(
            depth_m=arrays.depth_m[index],
            object_categories=mapped_labels[index],
            region_categories=np.full((256, 256), -1, dtype="<i2"),
            sensor_position=cast(
                Tuple[float, float, float],
                tuple(arrays.sensor_positions[index]),
            ),
            sensor_rotation=cast(
                Tuple[float, float, float, float],
                tuple(arrays.sensor_rotations_xyzw[index]),
            ),
            hfov_degrees=float(arrays.sensor_hfov_degrees.item()),
        )
        for index in range(12)
    )
    evidence = project_oracle_frames(
        frames,
        start_position=tuple(float(value) for value in arrays.start_position),
        start_rotation=tuple(float(value) for value in arrays.start_rotation_xyzw),
        target_origin_xz=tuple(float(value) for value in arrays.target_origin_xz),
    )
    semantic_grid = evidence.target_semantic_grid
    if (
        not isinstance(semantic_grid, np.ndarray)
        or semantic_grid.dtype != np.dtype(np.bool_)
        or semantic_grid.shape != (37, 50, 50)
        or not semantic_grid.flags.c_contiguous
    ):
        raise ValueError("projector returned the wrong object-grid schema")
    return np.ascontiguousarray(semantic_grid[:27])


def _confusion_counts(
    prediction: np.ndarray,
    target: np.ndarray,
) -> ConfusionCounts:
    return ConfusionCounts(
        tp=int(np.count_nonzero(prediction & target)),
        fp=int(np.count_nonzero(prediction & ~target)),
        fn=int(np.count_nonzero(~prediction & target)),
    )


def _require_object_grid(value: object, label: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.bool_)
        or value.shape != (27, 50, 50)
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"{label} must be a C-contiguous bool[27,50,50] grid")
    return value


def score_observation(
    *,
    ordinal: int,
    scene_id: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> ObservationMetrics:
    """Compute joint-primary, all-27, and descriptive category counts."""

    predicted = _require_object_grid(prediction, "prediction")
    expected = _require_object_grid(target, "target")
    primary_prediction = predicted[np.asarray(PRIMARY_CATEGORY_INDICES)]
    primary_target = expected[np.asarray(PRIMARY_CATEGORY_INDICES)]
    return ObservationMetrics(
        ordinal=ordinal,
        scene_id=scene_id,
        primary=MetricEndpoint.from_counts(
            _confusion_counts(primary_prediction, primary_target)
        ),
        all_27=MetricEndpoint.from_counts(_confusion_counts(predicted, expected)),
        per_category=tuple(
            MetricEndpoint.from_counts(
                _confusion_counts(predicted[index], expected[index])
            )
            for index in range(27)
        ),
    )


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _aggregate_endpoints(
    rows: Sequence[ObservationMetrics],
    endpoints: Sequence[MetricEndpoint],
    expected_scenes: Tuple[str, ...],
) -> MetricAggregate:
    eligible = tuple(endpoint for endpoint in endpoints if endpoint.eligible)
    pooled_counts = ConfusionCounts(0, 0, 0)
    for endpoint in endpoints:
        pooled_counts += endpoint.counts
    scene_ious: list[float] = []
    scene_f1s: list[float] = []
    complete_scene_macro = True
    for scene in expected_scenes:
        scene_endpoints = tuple(
            endpoint
            for row, endpoint in zip(rows, endpoints)
            if row.scene_id == scene and endpoint.eligible
        )
        if not scene_endpoints:
            complete_scene_macro = False
            continue
        scene_ious.append(
            sum(cast(float, endpoint.iou) for endpoint in scene_endpoints)
            / len(scene_endpoints)
        )
        scene_f1s.append(
            sum(cast(float, endpoint.f1) for endpoint in scene_endpoints)
            / len(scene_endpoints)
        )
    return MetricAggregate(
        observation_count=len(rows),
        eligible_observation_count=len(eligible),
        empty_both_count=len(rows) - len(eligible),
        target_empty_prediction_nonempty_count=sum(
            endpoint.counts.tp + endpoint.counts.fn == 0
            and endpoint.counts.tp + endpoint.counts.fp > 0
            for endpoint in endpoints
        ),
        target_nonempty_prediction_empty_count=sum(
            endpoint.counts.tp + endpoint.counts.fn > 0
            and endpoint.counts.tp + endpoint.counts.fp == 0
            for endpoint in endpoints
        ),
        mean_iou=_mean(tuple(cast(float, endpoint.iou) for endpoint in eligible)),
        mean_f1=_mean(tuple(cast(float, endpoint.f1) for endpoint in eligible)),
        pooled=MetricEndpoint.from_counts(pooled_counts),
        scene_macro_iou=_mean(scene_ious) if complete_scene_macro else None,
        scene_macro_f1=_mean(scene_f1s) if complete_scene_macro else None,
    )


def aggregate_observation_metrics(
    rows: Sequence[ObservationMetrics],
    *,
    expected_scenes: Sequence[str],
) -> BenchmarkMetricSummary:
    """Aggregate frozen observation endpoints without dropping empty scenes."""

    observations = tuple(rows)
    scenes = tuple(expected_scenes)
    if (
        not observations
        or any(not isinstance(row, ObservationMetrics) for row in observations)
        or len({row.ordinal for row in observations}) != len(observations)
    ):
        raise ValueError("metric rows must be non-empty with unique ordinals")
    if (
        not scenes
        or scenes != tuple(sorted(scenes))
        or len(set(scenes)) != len(scenes)
        or any(not isinstance(scene, str) or not scene for scene in scenes)
        or {row.scene_id for row in observations} != set(scenes)
    ):
        raise ValueError("expected scenes must exactly match rows in lexical order")
    return BenchmarkMetricSummary(
        primary=_aggregate_endpoints(
            observations,
            tuple(row.primary for row in observations),
            scenes,
        ),
        all_27=_aggregate_endpoints(
            observations,
            tuple(row.all_27 for row in observations),
            scenes,
        ),
        per_category=tuple(
            _aggregate_endpoints(
                observations,
                tuple(row.per_category[index] for row in observations),
                scenes,
            )
            for index in range(27)
        ),
    )


def linear_quantile(values: Iterable[float], probability: float) -> float:
    """Return the frozen linearly interpolated order statistic."""

    samples = tuple(values)
    if (
        not samples
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            for value in samples
        )
        or isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("linear quantile inputs are invalid")
    ordered = sorted(float(value) for value in samples)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def scene_bootstrap_matrix() -> np.ndarray:
    """Return the immutable, hash-pinned scene-composition matrix."""

    matrix = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0,
        BOOTSTRAP_SCENE_COUNT,
        size=(BOOTSTRAP_REPLICATES, BOOTSTRAP_SCENE_COUNT),
        endpoint=False,
    )
    matrix = np.ascontiguousarray(matrix, dtype="<i8")
    _require_bootstrap_matrix(matrix)
    matrix.flags.writeable = False
    return matrix


def _require_bootstrap_matrix(matrix: np.ndarray) -> np.ndarray:
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype != np.dtype("<i8")
        or matrix.shape != (BOOTSTRAP_REPLICATES, BOOTSTRAP_SCENE_COUNT)
        or not matrix.flags.c_contiguous
        or np.any(matrix < 0)
        or np.any(matrix >= BOOTSTRAP_SCENE_COUNT)
        or hashlib.sha256(matrix.tobytes(order="C")).hexdigest()
        != BOOTSTRAP_MATRIX_SHA256
    ):
        raise ValueError("bootstrap matrix differs from the frozen contract")
    return matrix


def _validate_endpoint_rows(
    rows: Sequence[EndpointRow],
    trusted_cohort: TrustedCohort,
) -> Tuple[Tuple[EndpointRow, ...], Tuple[str, ...]]:
    values = tuple(rows)
    if not isinstance(trusted_cohort, TrustedCohort):
        raise ValueError("trusted cohort must be TrustedCohort")
    identities = tuple(
        (row.ordinal, row.observation_id, row.scene_id)
        for row in values
        if isinstance(row, EndpointRow)
    )
    if (
        len(values) != 50
        or any(not isinstance(row, EndpointRow) for row in values)
        or identities != trusted_cohort.identities
    ):
        raise ValueError("endpoint identities differ from the trusted cohort")
    return values, trusted_cohort.scenes


def _scene_sums_and_counts(
    rows: Sequence[EndpointRow],
    scenes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(BOOTSTRAP_SCENE_COUNT, dtype=np.float64)
    counts = np.zeros(BOOTSTRAP_SCENE_COUNT, dtype="<i8")
    scene_indices = {scene: index for index, scene in enumerate(scenes)}
    for row in rows:
        if row.endpoint is not None:
            index = scene_indices[row.scene_id]
            sums[index] += row.endpoint
            counts[index] += 1
    return sums, counts


def _resampled_means(
    sums: np.ndarray,
    counts: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    denominators = counts[matrix].sum(axis=1)
    if np.any(denominators == 0):
        raise ValueError("bootstrap composition has no eligible endpoint")
    return sums[matrix].sum(axis=1) / denominators


def _leave_one_scene_out(
    sums: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    denominators = counts.sum() - counts
    if np.any(denominators == 0):
        raise ValueError("leave-one-scene-out composition has no eligible endpoint")
    return (sums.sum() - sums) / denominators


def estimate_scene_robustness(
    rows: Sequence[EndpointRow],
    *,
    trusted_cohort: TrustedCohort,
    matrix: Optional[np.ndarray] = None,
) -> RobustnessEstimate:
    """Estimate fixed-candidate scene-composition robustness."""

    values, scenes = _validate_endpoint_rows(rows, trusted_cohort)
    draws = _require_bootstrap_matrix(
        scene_bootstrap_matrix() if matrix is None else matrix
    )
    sums, counts = _scene_sums_and_counts(values, scenes)
    if counts.sum() == 0:
        raise ValueError("candidate has no eligible endpoint")
    replicates = _resampled_means(sums, counts, draws)
    leave_one_out = _leave_one_scene_out(sums, counts)
    return RobustnessEstimate(
        point_estimate=float(sums.sum() / counts.sum()),
        interval_low=linear_quantile(replicates, 0.025),
        interval_high=linear_quantile(replicates, 0.975),
        leave_one_scene_out_min=float(leave_one_out.min()),
        leave_one_scene_out_max=float(leave_one_out.max()),
    )


def estimate_scene_contrast(
    rows_a: Sequence[EndpointRow],
    rows_b: Sequence[EndpointRow],
    *,
    trusted_cohort: TrustedCohort,
    matrix: Optional[np.ndarray] = None,
) -> ContrastEstimate:
    """Estimate A-minus-B using separate eligibility on shared scene draws."""

    candidate_a, scenes = _validate_endpoint_rows(rows_a, trusted_cohort)
    candidate_b, scenes_b = _validate_endpoint_rows(rows_b, trusted_cohort)
    if scenes_b != scenes:
        raise ValueError("candidate endpoint scenes differ")
    draws = _require_bootstrap_matrix(
        scene_bootstrap_matrix() if matrix is None else matrix
    )
    sums_a, counts_a = _scene_sums_and_counts(candidate_a, scenes)
    sums_b, counts_b = _scene_sums_and_counts(candidate_b, scenes)
    if counts_a.sum() == 0 or counts_b.sum() == 0:
        raise ValueError("contrast candidate has no eligible endpoint")
    replicates = _resampled_means(sums_a, counts_a, draws) - _resampled_means(
        sums_b, counts_b, draws
    )
    leave_one_out = _leave_one_scene_out(sums_a, counts_a) - _leave_one_scene_out(
        sums_b, counts_b
    )
    return ContrastEstimate(
        point_estimate=float(
            sums_a.sum() / counts_a.sum() - sums_b.sum() / counts_b.sum()
        ),
        interval_low=linear_quantile(replicates, 0.025),
        interval_high=linear_quantile(replicates, 0.975),
        leave_one_scene_out_min=float(leave_one_out.min()),
        leave_one_scene_out_max=float(leave_one_out.max()),
    )


def evaluate_candidate_gates(
    *,
    synthetic: bool,
    coverage: StaticCoverage,
    trusted_cohort: TrustedCohort,
    rows: Sequence[EndpointRow],
    mean_iou: float,
    mean_f1: float,
    latency: LatencySummary,
    resource: ResourceMeasurement,
    code_license_status: LicenseStatus,
    weight_license_status: LicenseStatus,
    provenance: ProvenanceChecks,
) -> CandidateGateResult:
    """Apply only the frozen pure per-candidate minimum gates."""

    if type(synthetic) is not bool:
        raise ValueError("synthetic flag must be a bool")
    if synthetic:
        raise ValueError("synthetic runs do not enter candidate gates")
    if not isinstance(coverage, StaticCoverage):
        raise ValueError("coverage must be StaticCoverage")
    if not isinstance(trusted_cohort, TrustedCohort):
        raise ValueError("trusted cohort must be TrustedCohort")
    observations = tuple(rows)
    if any(not isinstance(row, EndpointRow) for row in observations):
        raise ValueError("gate rows must be EndpointRow values")
    if (
        isinstance(mean_iou, bool)
        or not isinstance(mean_iou, (int, float))
        or not math.isfinite(mean_iou)
        or not 0 <= mean_iou <= 1
        or isinstance(mean_f1, bool)
        or not isinstance(mean_f1, (int, float))
        or not math.isfinite(mean_f1)
        or not 0 <= mean_f1 <= 1
    ):
        raise ValueError("quality endpoints must be finite in [0, 1]")
    if (
        not isinstance(latency, LatencySummary)
        or not isinstance(resource, ResourceMeasurement)
        or not isinstance(code_license_status, LicenseStatus)
        or not isinstance(weight_license_status, LicenseStatus)
        or not isinstance(provenance, ProvenanceChecks)
    ):
        raise ValueError("candidate gate typed input is invalid")

    coverage_status = _gate(
        coverage.covered_category_count >= MIN_COVERED_CATEGORIES
        and coverage.support_ratio >= MIN_SUPPORT_COVERAGE
    )
    complete_status = _gate(
        tuple((row.ordinal, row.observation_id, row.scene_id) for row in observations)
        == trusted_cohort.identities
    )
    quality_status = _gate(
        coverage_status is GateStatus.PASS
        and complete_status is GateStatus.PASS
        and mean_iou >= MIN_MEAN_IOU
        and mean_f1 >= MIN_MEAN_F1
    )
    statuses = (
        coverage_status,
        complete_status,
        quality_status,
        _gate(
            latency.timing_comparable
            and cast(float, latency.p95_seconds) <= MAX_LATENCY_P95_SECONDS
        ),
        _gate(resource.peak_reserved_bytes <= MAX_ABSOLUTE_RESERVED_BYTES),
        _gate(
            code_license_status is LicenseStatus.PASS
            and weight_license_status is LicenseStatus.PASS
        ),
        _gate(provenance.passed),
    )
    return CandidateGateResult(
        coverage=statuses[0],
        complete_rows=statuses[1],
        quality=statuses[2],
        latency=statuses[3],
        resource=statuses[4],
        license=statuses[5],
        provenance=statuses[6],
        overall=_gate(all(status is GateStatus.PASS for status in statuses)),
    )


def _gate(value: bool) -> GateStatus:
    return GateStatus.PASS if value else GateStatus.FAIL


def _strict_json(data: bytes, label: str) -> object:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=lambda pairs: _unique_object(pairs, label),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error


def _unique_object(
    pairs: Sequence[Tuple[str, object]], label: str
) -> Mapping[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"{label} has duplicate key {key!r}")
        value[key] = item
    return value


def load_nyu40_mapping(
    path: Path = DEFAULT_NYU40_MAPPING_PATH,
) -> Tuple[MappingEntry, ...]:
    """Load and fully validate the frozen 40-class mapping."""

    if not isinstance(path, Path):
        raise ValueError("mapping path must be a Path")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != NYU40_MAPPING_SHA256:
        raise ValueError("NYU40 mapping differs from the frozen file hash")
    raw = _strict_json(data, "NYU40 mapping")
    if not isinstance(raw, list) or len(raw) != 40:
        raise ValueError("NYU40 mapping must contain exactly 40 rows")
    entries: list[MappingEntry] = []
    keys = {
        "canonical_index",
        "canonical_name",
        "kind",
        "source_index",
        "source_name",
    }
    for expected_index, value in enumerate(raw):
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("NYU40 mapping row has wrong keys")
        row = cast(Mapping[str, object], value)
        try:
            kind = MappingKind(_string(row["kind"], "mapping kind"))
        except (TypeError, ValueError) as error:
            raise ValueError("NYU40 mapping kind is invalid") from error
        entry = MappingEntry(
            source_index=cast(int, row["source_index"]),
            source_name=cast(str, row["source_name"]),
            kind=kind,
            canonical_index=cast(Optional[int], row["canonical_index"]),
            canonical_name=cast(Optional[str], row["canonical_name"]),
        )
        if entry.source_index != expected_index:
            raise ValueError("NYU40 source indices must be ordered 0 through 39")
        entries.append(entry)
    if len({entry.source_name for entry in entries}) != 40:
        raise ValueError("NYU40 source names must be unique")
    if tuple(entry.source_name for entry in entries) != _NYU40_SOURCE_NAMES:
        raise ValueError("NYU40 source vocabulary differs from the frozen order")
    return tuple(entries)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class P53ValidationAttestation:
    python_executable: str
    environment_sha256: str
    gpu_device_id: int
    gpu_uuid: str
    producer_git_commit: str
    raw_root: str
    raw_manifest_sha256: str
    raw_index_sha256: str
    validator_source_sha256: str
    status: str = "PASS"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or not isinstance(self.status, str)
            or self.status != "PASS"
            or not isinstance(self.python_executable, str)
            or not isinstance(self.raw_root, str)
            or not isinstance(self.gpu_uuid, str)
            or not isinstance(self.producer_git_commit, str)
            or not Path(self.python_executable).is_absolute()
            or not Path(self.raw_root).is_absolute()
            or type(self.gpu_device_id) is not int
            or self.gpu_device_id != 0
            or self.gpu_uuid != RAW_GPU_UUID
            or _COMMIT.fullmatch(self.producer_git_commit) is None
            or self.validator_source_sha256 != RAW_VALIDATOR_SOURCE_SHA256
        ):
            raise ValueError("P5.3 validation attestation identity is invalid")
        for value in (
            self.environment_sha256,
            self.raw_manifest_sha256,
            self.raw_index_sha256,
            self.validator_source_sha256,
        ):
            if not isinstance(value, str) or _HASH.fullmatch(value) is None:
                raise ValueError("P5.3 validation attestation hash is invalid")

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({
            "environment_sha256": self.environment_sha256,
            "gpu_device_id": self.gpu_device_id,
            "gpu_uuid": self.gpu_uuid,
            "producer_git_commit": self.producer_git_commit,
            "python_executable": self.python_executable,
            "raw_index_sha256": self.raw_index_sha256,
            "raw_manifest_sha256": self.raw_manifest_sha256,
            "raw_root": self.raw_root,
            "schema_version": self.schema_version,
            "status": self.status,
            "validator_source_sha256": self.validator_source_sha256,
        })

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def parse(cls, data: bytes) -> "P53ValidationAttestation":
        raw = _strict_json(data, "P5.3 validation attestation")
        expected = {
            "environment_sha256",
            "gpu_device_id",
            "gpu_uuid",
            "producer_git_commit",
            "python_executable",
            "raw_index_sha256",
            "raw_manifest_sha256",
            "raw_root",
            "schema_version",
            "status",
            "validator_source_sha256",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("P5.3 validation attestation has wrong keys")
        row = cast(Mapping[str, object], raw)
        value = cls(
            python_executable=_string(row["python_executable"], "python executable"),
            environment_sha256=_string(
                row["environment_sha256"], "environment SHA-256"
            ),
            gpu_device_id=_integer(row["gpu_device_id"], "GPU device ID"),
            gpu_uuid=_string(row["gpu_uuid"], "GPU UUID"),
            producer_git_commit=_string(
                row["producer_git_commit"], "producer Git commit"
            ),
            raw_root=_string(row["raw_root"], "raw root"),
            raw_manifest_sha256=_string(
                row["raw_manifest_sha256"], "raw manifest SHA-256"
            ),
            raw_index_sha256=_string(row["raw_index_sha256"], "raw index SHA-256"),
            validator_source_sha256=_string(
                row["validator_source_sha256"], "validator source SHA-256"
            ),
            status=_string(row["status"], "status"),
            schema_version=_integer(row["schema_version"], "schema version"),
        )
        if data != value.canonical_bytes():
            raise ValueError("P5.3 validation attestation is noncanonical")
        return value


@dataclass(frozen=True)
class BenchmarkEnvironmentAttestation:
    python_executable: str
    environment_sha256: str
    visible_device_count: int
    gpu_name: str
    gpu_uuid: str
    timing_comparable: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.python_executable, str)
            or not isinstance(self.environment_sha256, str)
            or not isinstance(self.gpu_name, str)
            or not isinstance(self.gpu_uuid, str)
            or not Path(self.python_executable).is_absolute()
            or _HASH.fullmatch(self.environment_sha256) is None
            or type(self.visible_device_count) is not int
            or self.visible_device_count < 0
            or type(self.timing_comparable) is not bool
        ):
            raise ValueError("benchmark environment attestation is invalid")
        if self.timing_comparable and (
            self.visible_device_count != 1
            or self.gpu_name != "NVIDIA GeForce RTX 3090"
            or not self.gpu_uuid.startswith("GPU-")
        ):
            raise ValueError(
                "comparable benchmark requires exactly one recorded RTX 3090"
            )
        if not self.timing_comparable and (
            self.gpu_name != "NOT_APPLICABLE" or self.gpu_uuid != "NOT_APPLICABLE"
        ):
            raise ValueError("non-comparable benchmark cannot claim a GPU")

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({
            "environment_sha256": self.environment_sha256,
            "gpu_name": self.gpu_name,
            "gpu_uuid": self.gpu_uuid,
            "python_executable": self.python_executable,
            "timing_comparable": self.timing_comparable,
            "visible_device_count": self.visible_device_count,
        })

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def require_current_process(
        self, *, gpu_inspector: Optional["VisibleGpuInspector"] = None
    ) -> None:
        if self.python_executable != str(Path(sys.executable).resolve(strict=True)):
            raise ValueError("benchmark interpreter differs from attestation")
        if self.environment_sha256 != capture_environment_sha256():
            raise ValueError("benchmark environment differs from attestation")
        if self.visible_device_count != torch.cuda.device_count():
            raise ValueError("benchmark visible device count differs from attestation")
        if self.timing_comparable:
            inspector = inspect_visible_gpu if gpu_inspector is None else gpu_inspector
            gpu_name, gpu_uuid = inspector()
            if (gpu_name, gpu_uuid) != (self.gpu_name, self.gpu_uuid):
                raise ValueError(
                    "benchmark visible GPU identity differs from attestation"
                )


class VisibleGpuInspector(Protocol):
    def __call__(self) -> Tuple[str, str]: ...


def inspect_visible_gpu() -> Tuple[str, str]:
    """Resolve the sole CUDA-visible physical GPU through NVIDIA's inventory."""

    if torch.cuda.device_count() != 1:
        raise ValueError("GPU inspection requires exactly one CUDA-visible device")
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        check=False,
        timeout=10.0,
    )
    if completed.returncode != 0 or completed.stderr:
        raise ValueError("unable to inspect the visible GPU")
    rows = [
        tuple(part.strip() for part in line.split(",", 2))
        for line in completed.stdout.decode("utf-8").splitlines()
        if line.strip()
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or re.fullmatch(r"GPU-[A-Za-z0-9-]+", visible) is None:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be one complete GPU UUID, not an index"
        )
    matches = [row for row in rows if row[2] == visible]
    if len(matches) != 1:
        raise ValueError("visible GPU UUID is absent from NVIDIA inventory")
    selected = matches[0]
    return selected[1], selected[2]


@dataclass(frozen=True)
class P53ValidatorLaunch:
    python_executable: Path
    expected_environment_sha256: str
    raw_root: Path
    timeout_seconds: float

    def __post_init__(self) -> None:
        if (
            not self.python_executable.is_absolute()
            or not self.raw_root.is_absolute()
            or not isinstance(self.expected_environment_sha256, str)
            or _HASH.fullmatch(self.expected_environment_sha256) is None
            or isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("P5.3 validator launch is invalid")
        if (
            self.python_executable.resolve(strict=True) != self.python_executable
            or self.raw_root.resolve(strict=True) != self.raw_root
            or not self.python_executable.is_file()
            or stat.S_ISLNK(os.lstat(self.python_executable).st_mode)
            or stat.S_ISLNK(os.lstat(self.raw_root).st_mode)
        ):
            raise ValueError("P5.3 validator paths must be canonical and non-symlink")


def capture_environment_sha256() -> str:
    """Hash the interpreter and complete installed-distribution environment."""

    distributions = sorted(
        (
            distribution.metadata["Name"].lower().replace("_", "-"),
            distribution.version,
        )
        for distribution in metadata.distributions()
        if distribution.metadata["Name"] is not None
    )
    return hashlib.sha256(
        canonical_json_bytes({
            "distributions": distributions,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
        })
    ).hexdigest()


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def emit_p53_validation_attestation(raw_root: Path) -> bytes:
    """Run the public P5.3 validator and return one canonical PASS record."""

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("P5.3 validation requires CUDA_VISIBLE_DEVICES absent")
    root = raw_root.resolve(strict=True)
    validate_raw_frame_directory(root, expected_git_commit=RAW_PRODUCER_COMMIT)
    source_path = rgbd_segmenter_raw_frame_package.__file__
    if source_path is None:
        raise ValueError("P5.3 validator module has no source path")
    source = Path(source_path).resolve(strict=True)
    return P53ValidationAttestation(
        python_executable=str(Path(sys.executable).resolve(strict=True)),
        environment_sha256=capture_environment_sha256(),
        gpu_device_id=0,
        gpu_uuid=RAW_GPU_UUID,
        producer_git_commit=RAW_PRODUCER_COMMIT,
        raw_root=str(root),
        raw_manifest_sha256=RAW_MANIFEST_SHA256,
        raw_index_sha256=RAW_INDEX_SHA256,
        validator_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    ).canonical_bytes()


_ATTEST_EXPRESSION = (
    "import pathlib,sys;"
    "from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract "
    "import emit_p53_validation_attestation;"
    "sys.stdout.buffer.write(emit_p53_validation_attestation(pathlib.Path(sys.argv[1])))"
)


class SubprocessRunner(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *,
        env: Mapping[str, str],
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]: ...


def run_p53_validation_subprocess(
    launch: P53ValidatorLaunch,
    *,
    runner: SubprocessRunner = subprocess.run,
) -> P53ValidationAttestation:
    """Launch and independently verify the path-pinned P5.3 validator."""

    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        completed = runner(
            (
                str(launch.python_executable),
                "-c",
                _ATTEST_EXPRESSION,
                str(launch.raw_root),
            ),
            env=environment,
            capture_output=True,
            check=False,
            timeout=launch.timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("P5.3 validator timed out") from error
    if completed.returncode != 0:
        raise ValueError("P5.3 validator did not exit successfully")
    if completed.stderr:
        raise ValueError("P5.3 validator emitted unexpected stderr")
    attestation = P53ValidationAttestation.parse(completed.stdout)
    if (
        attestation.python_executable
        != str(launch.python_executable.resolve(strict=True))
        or attestation.environment_sha256 != launch.expected_environment_sha256
        or attestation.raw_root != str(launch.raw_root.resolve(strict=True))
        or attestation.producer_git_commit != RAW_PRODUCER_COMMIT
        or attestation.raw_manifest_sha256 != RAW_MANIFEST_SHA256
        or attestation.raw_index_sha256 != RAW_INDEX_SHA256
        or attestation.validator_source_sha256 != RAW_VALIDATOR_SOURCE_SHA256
    ):
        raise ValueError("P5.3 validation attestation differs from launcher pins")
    return attestation


@dataclass(frozen=True)
class ValidatedRawObservation:
    row: IndexRow
    segmenter_input: SegmenterInput
    audit_arrays: RawFrameArrays


def _fingerprint(value: os.stat_result) -> Tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_at(
    root_descriptor: int,
    relative: str,
    *,
    expected: FileRecord,
) -> Tuple[bytes, Tuple[int, ...]]:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or len(path.parts) > _MAX_TREE_DEPTH
        or any(part in {"", ".", ".."} for part in path.parts)
        or type(expected.byte_length) is not int
        or not 0 <= expected.byte_length <= _MAX_RAW_FILE_BYTES
        or _HASH.fullmatch(expected.sha256) is None
    ):
        raise ValueError("raw package path is unsafe")
    descriptor = os.dup(root_descriptor)
    try:
        for part in path.parts[:-1]:
            before_child = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            if _fingerprint(before_child) != _fingerprint(os.fstat(child)):
                os.close(child)
                raise ValueError("raw package directory changed while opened")
            os.close(descriptor)
            descriptor = child
        leaf = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=descriptor,
        )
        try:
            before = os.fstat(leaf)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError(
                    "raw package entry must be a singly linked regular file"
                )
            if before.st_size != expected.byte_length:
                raise ValueError("raw package entry has wrong byte length")
            chunks: list[bytes] = []
            remaining = expected.byte_length
            while remaining:
                chunk = os.read(leaf, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("raw package entry ended before its commitment")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(leaf, 1):
                raise ValueError("raw package entry exceeds its commitment")
            after = os.fstat(leaf)
        finally:
            os.close(leaf)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if _fingerprint(before) != _fingerprint(after):
        raise ValueError("raw package entry changed while read")
    if (
        len(data) != before.st_size
        or hashlib.sha256(data).hexdigest() != expected.sha256
    ):
        raise ValueError("raw package entry differs from its external hash")
    return data, _fingerprint(before)


def _walk_metadata(
    descriptor: int,
    relative: str = "",
    *,
    depth: int = 0,
    entry_count: Optional[list[int]] = None,
) -> Tuple[Mapping[str, Tuple[int, ...]], Mapping[str, Tuple[int, ...]]]:
    if depth > _MAX_TREE_DEPTH:
        raise ValueError("raw package tree exceeds the depth bound")
    counter = [0] if entry_count is None else entry_count
    directories: dict[str, Tuple[int, ...]] = {}
    files: dict[str, Tuple[int, ...]] = {}
    names: list[str] = []
    with os.scandir(descriptor) as iterator:
        for entry in iterator:
            counter[0] += 1
            if counter[0] > _MAX_TREE_ENTRIES:
                raise ValueError("raw package tree exceeds the entry bound")
            names.append(entry.name)
    for name in sorted(names):
        if name in {"", ".", ".."} or "/" in name:
            raise ValueError("raw package contains an unsafe entry")
        child_relative = f"{relative}/{name}" if relative else name
        metadata_value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        identity = _fingerprint(metadata_value)
        if stat.S_ISDIR(metadata_value.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                if identity != _fingerprint(os.fstat(child)):
                    raise ValueError("raw package directory changed while opened")
                directories[child_relative] = identity
                child_directories, child_files = _walk_metadata(
                    child,
                    child_relative,
                    depth=depth + 1,
                    entry_count=counter,
                )
                directories.update(child_directories)
                files.update(child_files)
            finally:
                os.close(child)
        elif stat.S_ISREG(metadata_value.st_mode) and metadata_value.st_nlink == 1:
            files[child_relative] = identity
        else:
            raise ValueError("raw package contains a non-regular entry")
    return directories, files


def _pinned_cpu_tensor(array: np.ndarray) -> torch.Tensor:
    tensor = torch.empty(
        tuple(array.shape),
        dtype=torch.from_numpy(np.empty((), dtype=array.dtype)).dtype,
        pin_memory=True,
    )
    tensor.copy_(torch.from_numpy(array))
    return tensor


def _open_absolute_root(
    path: Path,
) -> Tuple[int, int, Mapping[str, Tuple[int, ...]]]:
    if not path.is_absolute() or path.anchor != "/":
        raise ValueError("raw package root must be an absolute POSIX path")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    identities: dict[str, Tuple[int, ...]] = {}
    prefix = Path("/")
    try:
        for part in path.parts[1:-1]:
            before = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            if _fingerprint(before) != _fingerprint(os.fstat(child)):
                os.close(child)
                raise ValueError("raw package ancestor changed while opened")
            os.close(descriptor)
            descriptor = child
            prefix /= part
            identities[str(prefix)] = _fingerprint(before)
        root_before = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        root_descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        if _fingerprint(root_before) != _fingerprint(os.fstat(root_descriptor)):
            os.close(root_descriptor)
            raise ValueError("raw package root changed while opened")
        identities[str(path)] = _fingerprint(root_before)
        return descriptor, root_descriptor, identities
    except BaseException:
        os.close(descriptor)
        raise


def iter_validated_raw_observations(
    root: Path,
    *,
    p53_attestation: P53ValidationAttestation,
    expected_p53_attestation_sha256: str,
    benchmark_attestation: BenchmarkEnvironmentAttestation,
    expected_benchmark_attestation_sha256: str,
    gpu_inspector: Optional[VisibleGpuInspector] = None,
) -> Tuple[ValidatedRawObservation, ...]:
    """Return all rows only after complete descriptor-bound validation."""

    if (
        _HASH.fullmatch(expected_p53_attestation_sha256) is None
        or p53_attestation.sha256 != expected_p53_attestation_sha256
        or p53_attestation.producer_git_commit != RAW_PRODUCER_COMMIT
        or p53_attestation.raw_manifest_sha256 != RAW_MANIFEST_SHA256
        or p53_attestation.raw_index_sha256 != RAW_INDEX_SHA256
    ):
        raise ValueError("P5.3 attestation does not match the trusted launcher hash")
    if (
        _HASH.fullmatch(expected_benchmark_attestation_sha256) is None
        or benchmark_attestation.sha256 != expected_benchmark_attestation_sha256
    ):
        raise ValueError("benchmark attestation differs from the launcher hash")
    benchmark_attestation.require_current_process(gpu_inspector=gpu_inspector)
    lexical = root if root.is_absolute() else Path.cwd() / root
    lexical_metadata = os.lstat(lexical)
    if stat.S_ISLNK(lexical_metadata.st_mode):
        raise ValueError("raw root must not be a symlink")
    absolute = lexical.resolve(strict=True)
    if Path(os.path.abspath(lexical)) != absolute:
        raise ValueError("raw root path contains a symlink")
    if str(absolute) != p53_attestation.raw_root:
        raise ValueError("raw root differs from P5.3 validation attestation")
    parent, root_descriptor, path_identities = _open_absolute_root(absolute)
    try:
        observations: list[ValidatedRawObservation] = []
        root_identity = _fingerprint(os.fstat(root_descriptor))
        manifest_bytes, manifest_identity = _read_at(
            root_descriptor,
            "manifest.json",
            expected=FileRecord(RAW_MANIFEST_BYTE_LENGTH, RAW_MANIFEST_SHA256),
        )
        manifest_value = _strict_json(manifest_bytes, "raw manifest")
        if not isinstance(manifest_value, dict):
            raise ValueError("raw manifest must be an object")
        manifest = cast(Mapping[str, object], manifest_value)
        collection = manifest.get("collection")
        files = manifest.get("files")
        collection_map = (
            cast(Mapping[str, object], collection)
            if isinstance(collection, dict)
            else None
        )
        files_map = (
            cast(Mapping[str, object], files) if isinstance(files, dict) else None
        )
        index_value = files_map.get("index") if files_map is not None else None
        index_map = (
            cast(Mapping[str, object], index_value)
            if isinstance(index_value, dict)
            else None
        )
        if (
            collection_map is None
            or collection_map.get("git_commit") != RAW_PRODUCER_COMMIT
            or index_map is None
            or index_map.get("sha256") != RAW_INDEX_SHA256
        ):
            raise ValueError("raw manifest differs from frozen benchmark pins")
        index_bytes, index_identity = _read_at(
            root_descriptor,
            "index.jsonl",
            expected=FileRecord(RAW_INDEX_BYTE_LENGTH, RAW_INDEX_SHA256),
        )
        rows = parse_index_bytes(index_bytes)
        expected_files = {
            "manifest.json",
            "index.jsonl",
            *(row.artifact for row in rows),
        }
        expected_directories = {
            "observations",
            *(str(PurePosixPath(row.artifact).parent) for row in rows),
        }
        directories, tree_files = _walk_metadata(root_descriptor)
        if (
            set(tree_files) != expected_files
            or set(directories) != expected_directories
        ):
            raise ValueError("raw package has missing, extra, or reordered entries")
        if (
            tree_files["manifest.json"] != manifest_identity
            or tree_files["index.jsonl"] != index_identity
        ):
            raise ValueError("raw package metadata changed before streaming")
        accepted: dict[str, Tuple[int, ...]] = {
            "manifest.json": manifest_identity,
            "index.jsonl": index_identity,
        }
        for row in rows:
            data, identity = _read_at(root_descriptor, row.artifact, expected=row.npz)
            if identity != tree_files[row.artifact]:
                raise ValueError("raw package artifact changed after tree capture")
            parsed = parse_raw_frame_npz_bytes(
                data, expected_members=row.members, expected_npz=row.npz
            )
            accepted[row.artifact] = identity
            observations.append(
                ValidatedRawObservation(
                    row=row,
                    segmenter_input=SegmenterInput(
                        rgb=_pinned_cpu_tensor(parsed.arrays.rgb),
                        depth_m=_pinned_cpu_tensor(parsed.arrays.depth_m),
                    ),
                    audit_arrays=parsed.arrays,
                )
            )
        final_parent, final_root_descriptor, final_path_identities = (
            _open_absolute_root(absolute)
        )
        try:
            if final_path_identities != path_identities:
                raise ValueError("raw package ancestor or root binding changed")
            final_directories, final_files = _walk_metadata(final_root_descriptor)
            if final_directories != directories or final_files != accepted:
                raise ValueError("raw package tree changed during streaming")
            for relative, expected in (
                (
                    "manifest.json",
                    FileRecord(RAW_MANIFEST_BYTE_LENGTH, RAW_MANIFEST_SHA256),
                ),
                (
                    "index.jsonl",
                    FileRecord(RAW_INDEX_BYTE_LENGTH, RAW_INDEX_SHA256),
                ),
                *((row.artifact, row.npz) for row in rows),
            ):
                _, identity = _read_at(
                    final_root_descriptor, relative, expected=expected
                )
                if identity != accepted[relative]:
                    raise ValueError(
                        "raw package content changed during final recapture"
                    )
            if _fingerprint(os.fstat(final_root_descriptor)) != root_identity:
                raise ValueError("raw package root path binding changed")
        finally:
            os.close(final_root_descriptor)
            os.close(final_parent)
        return tuple(observations)
    finally:
        os.close(root_descriptor)
        os.close(parent)
