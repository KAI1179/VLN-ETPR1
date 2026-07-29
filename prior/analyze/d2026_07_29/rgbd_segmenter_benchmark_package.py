"""Canonical package codecs for the frozen RGB-D segmenter benchmark."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import subprocess
import zipfile
from collections.abc import Iterator
from dataclasses import InitVar, asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import (
    Callable,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch

from prior import constants as _constants_module
from prior.analyze.d2026_07_29 import (
    rgbd_segmenter_benchmark_contract as _benchmark_contract_module,
)
from prior.analyze.d2026_07_29 import (
    rgbd_segmenter_raw_frame_package as _raw_package_module,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    BOOTSTRAP_MATRIX_SHA256,
    NYU40_MAPPING_SHA256,
    RAW_INDEX_SHA256,
    RAW_INDEX_BYTE_LENGTH,
    RAW_MANIFEST_SHA256,
    RAW_PRODUCER_COMMIT,
    RAW_VALIDATOR_SOURCE_SHA256,
    BenchmarkEnvironmentAttestation,
    BenchmarkMetricSummary,
    CandidateCommitment,
    CandidateGateResult,
    ComponentTimings,
    ConfusionCounts,
    EndpointRow,
    GateStatus,
    LatencySummary,
    LicenseStatus,
    MappingEntry,
    MetricAggregate,
    MetricEndpoint,
    ObservationFailureCode,
    ObservationMetrics,
    ObservationStatus,
    P53ValidationAttestation,
    P53ValidatorLaunch,
    Prediction,
    ProvenanceChecks,
    ResourceMeasurement,
    RobustnessEstimate,
    StaticCoverage,
    TimingSample,
    TimingStage,
    TimingProtocol,
    TrustedCohort,
    ValidatedRawObservation,
    aggregate_observation_metrics,
    canonical_json_bytes,
    compute_static_coverage,
    estimate_scene_robustness,
    evaluate_candidate_gates,
    iter_validated_raw_observations,
    logical_label_sha256,
    load_nyu40_mapping,
    map_source_labels,
    project_mapped_labels,
    project_oracle_target_labels,
    run_p53_validation_subprocess,
    score_observation,
    summarize_latency,
)
from vlnce_baselines.models.etp_llm import (
    llm_grid_oracle_cache as _projector_module,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    IndexRow,
    RawFrameArrays,
    parse_index_bytes,
)

_HASH = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}")
_IDENTITY = re.compile(r"[0-9a-f]{20}")
_SCENE = re.compile(r"[A-Za-z0-9_-]+")
_MAX_PREDICTION_BYTES = 4 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_OBSERVATIONS_BYTES = 16 * 1024 * 1024
_MAX_TIMINGS_BYTES = 8 * 1024 * 1024
_MAX_PACKAGE_ENTRIES = 80
_MAX_PACKAGE_DEPTH = 3
_READ_CHUNK_BYTES = 1024 * 1024
_LABEL_SHAPE = (12, 256, 256)
_LABEL_DTYPE = "<i2"
_LABEL_ARRAY_BYTES = 12 * 256 * 256 * 2
_MEMBER_NAMES = ("mapped_labels", "source_labels")
_ZIP_EOCD = struct.Struct("<4s4H2LH")
_ZIP_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_ACCEPTANCE_TOKEN = object()
_VALIDATION_TOKEN = object()
_COHORT_JSONL_SHA256 = (
    "89ae70f3e489fa702c66110f9bd9e7666ba0e16a9fbc3ac20aaa91a95adef0ce"
)
_SELECTION_SHA256 = "32a7adddf32291f059eb1045e63e077a693ca64d6fdf6e7418b54dd5d3644cb2"
_RAW_PAYLOAD_TREE_SHA256 = (
    "6b9c48460493bf8f50aeed408173c95210426723fb0c845324e87cfd4dad1c46"
)
_MAX_TRUSTED_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
_GIT_INSPECTION_TIMEOUT_SECONDS = 10.0
_MAX_LOCAL_GIT_CONFIG_BYTES = 1024 * 1024
_MAX_GIT_TREE_BYTES = 16 * 1024 * 1024
_MAX_GIT_IGNORE_BYTES = 16 * 1024 * 1024
_MAX_GIT_METADATA_FILE_BYTES = 1024 * 1024
_TRUSTED_GIT_PATH = "/usr/bin/git"
_RAW_PACKAGE_SOURCE_PATH = (
    "prior/analyze/d2026_07_29/rgbd_segmenter_raw_frame_package.py"
)


def _expected_npy_length() -> int:
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(_LABEL_DTYPE)),
            "fortran_order": False,
            "shape": _LABEL_SHAPE,
        },
    )
    return len(header.getvalue()) + _LABEL_ARRAY_BYTES


_LABEL_NPY_BYTES = _expected_npy_length()


@dataclass(frozen=True)
class FileRecord:
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        _plain_int(self.byte_length, "file byte length")
        _hash(self.sha256, "file SHA-256")


@dataclass(frozen=True)
class FileTreeAggregate:
    file_count: int
    total_byte_length: int
    tree_sha256: str

    def __post_init__(self) -> None:
        _plain_int(self.file_count, "tree file count", 1)
        _plain_int(self.total_byte_length, "tree total byte length", 1)
        _hash(self.tree_sha256, "tree SHA-256")


def file_tree_aggregate(
    records: Iterable[Tuple[str, FileRecord]],
) -> FileTreeAggregate:
    """Hash canonical lexical path/file records without delimiter ambiguity."""

    materialized = tuple(records)
    if not materialized:
        raise ValueError("file tree aggregate requires at least one file")
    for path, record in materialized:
        _relative_path(path)
        if not isinstance(record, FileRecord):
            raise ValueError("file tree aggregate requires typed file records")
    paths = tuple(path for path, _ in materialized)
    if len(set(paths)) != len(paths):
        raise ValueError("file tree aggregate paths must be unique")
    ordered = tuple(sorted(materialized))
    canonical = canonical_json_bytes([
        {
            "byte_length": record.byte_length,
            "path": path,
            "sha256": record.sha256,
        }
        for path, record in ordered
    ])
    return FileTreeAggregate(
        file_count=len(ordered),
        total_byte_length=sum(record.byte_length for _, record in ordered),
        tree_sha256=hashlib.sha256(canonical).hexdigest(),
    )


@dataclass(frozen=True)
class PredictionMember:
    array_byte_length: int
    array_sha256: str
    dtype: str
    npy_byte_length: int
    npy_sha256: str
    shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.array_byte_length != _LABEL_ARRAY_BYTES
            or self.dtype != _LABEL_DTYPE
            or self.npy_byte_length != _LABEL_NPY_BYTES
            or self.shape != _LABEL_SHAPE
        ):
            raise ValueError("prediction member contradicts the frozen schema")
        _hash(self.array_sha256, "prediction array SHA-256")
        _hash(self.npy_sha256, "prediction NPY SHA-256")


@dataclass(frozen=True)
class PredictionArtifact:
    data: bytes
    file: FileRecord
    members: Mapping[str, PredictionMember]

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise ValueError("prediction artifact data must be bytes")
        if self.file != _file_record(self.data):
            raise ValueError("prediction artifact file record is inconsistent")
        _require_member_schema(self.members)
        object.__setattr__(self, "members", _typed_mapping_copy(self.members))


@dataclass(frozen=True)
class ParsedPrediction:
    prediction: Prediction
    file: FileRecord
    members: Mapping[str, PredictionMember]

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, Prediction):
            raise ValueError("parsed prediction must contain a Prediction")
        _require_member_schema(self.members)
        object.__setattr__(self, "members", _typed_mapping_copy(self.members))


@dataclass(frozen=True)
class CandidateMappingAuthority:
    source_vocabulary: Tuple[str, ...]
    mapping: Tuple[MappingEntry, ...]
    mapping_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.source_vocabulary) is not tuple
            or not self.source_vocabulary
            or any(
                not isinstance(name, str) or not name for name in self.source_vocabulary
            )
            or len(set(self.source_vocabulary)) != len(self.source_vocabulary)
            or type(self.mapping) is not tuple
            or len(self.mapping) != len(self.source_vocabulary)
            or any(
                not isinstance(entry, MappingEntry)
                or entry.source_index != index
                or entry.source_name != self.source_vocabulary[index]
                for index, entry in enumerate(self.mapping)
            )
        ):
            raise ValueError("candidate mapping authority is invalid")
        _hash(self.mapping_sha256, "candidate mapping authority SHA-256")


@dataclass(frozen=True)
class TrustedArtifactAuthority:
    root_role: str
    path: str
    file: FileRecord

    def __post_init__(self) -> None:
        if self.root_role not in {
            "benchmark_repository",
            "candidate_repository",
        }:
            raise ValueError("trusted artifact root role is invalid")
        _relative_path(self.path)
        if not isinstance(self.file, FileRecord):
            raise ValueError("trusted artifact file record is invalid")


@dataclass(frozen=True)
class RealProvenanceAuthority:
    candidate_repository_root: Path
    checkpoint_root: Path
    checkpoint_path: str
    checkpoint_file: FileRecord
    code_permission: TrustedArtifactAuthority
    weight_permission: TrustedArtifactAuthority

    def __post_init__(self) -> None:
        _absolute_authority_root(
            self.candidate_repository_root,
            "candidate repository root",
        )
        _absolute_authority_root(self.checkpoint_root, "checkpoint root")
        _relative_path(self.checkpoint_path)
        if (
            not isinstance(self.checkpoint_file, FileRecord)
            or not isinstance(self.code_permission, TrustedArtifactAuthority)
            or not isinstance(self.weight_permission, TrustedArtifactAuthority)
        ):
            raise ValueError("real provenance artifact authority is invalid")


@dataclass(frozen=True)
class CandidateValidationAuthority:
    expected_manifest_sha256: str
    p53_launch: P53ValidatorLaunch
    benchmark_attestation: BenchmarkEnvironmentAttestation
    expected_benchmark_attestation_sha256: str
    expected_command: Tuple[str, ...]
    benchmark_repository_root: Path
    expected_producer_commit: str
    candidate: CandidateCommitment
    mapping: CandidateMappingAuthority
    adapter_path: str
    environment_lock: TrustedArtifactAuthority
    real: Optional[RealProvenanceAuthority] = None

    def __post_init__(self) -> None:
        _hash(self.expected_manifest_sha256, "expected candidate manifest SHA-256")
        _hash(
            self.expected_benchmark_attestation_sha256,
            "expected benchmark attestation SHA-256",
        )
        if (
            not isinstance(self.p53_launch, P53ValidatorLaunch)
            or not isinstance(
                self.benchmark_attestation, BenchmarkEnvironmentAttestation
            )
            or self.benchmark_attestation.sha256
            != self.expected_benchmark_attestation_sha256
            or type(self.expected_command) is not tuple
            or not self.expected_command
            or any(
                not isinstance(value, str) or not value
                for value in self.expected_command
            )
            or not isinstance(self.candidate, CandidateCommitment)
            or not isinstance(self.mapping, CandidateMappingAuthority)
            or not isinstance(
                self.environment_lock,
                TrustedArtifactAuthority,
            )
            or (
                self.real is not None
                and not isinstance(self.real, RealProvenanceAuthority)
            )
            or (self.real is not None) is not (not self.candidate.synthetic)
        ):
            raise ValueError("candidate validation authority is invalid")
        _absolute_authority_root(
            self.benchmark_repository_root,
            "benchmark repository root",
        )
        if (
            not isinstance(self.expected_producer_commit, str)
            or _COMMIT.fullmatch(self.expected_producer_commit) is None
        ):
            raise ValueError("expected producer commit is invalid")
        _relative_path(self.adapter_path)
        if (
            self.candidate.source_vocabulary != self.mapping.source_vocabulary
            or self.candidate.mapping_sha256 != self.mapping.mapping_sha256
            or self.candidate.environment_lock_path != self.environment_lock.path
            or self.candidate.environment_lock_sha256
            != self.environment_lock.file.sha256
            or (
                self.candidate.synthetic
                and self.environment_lock.root_role != "benchmark_repository"
            )
        ):
            raise ValueError("candidate authorities differ from commitment")


@dataclass(frozen=True)
class GitRepositoryState:
    commit: str
    clean: bool
    origin_url: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.commit, str)
            or _COMMIT.fullmatch(self.commit) is None
            or type(self.clean) is not bool
            or (
                self.origin_url is not None
                and (not isinstance(self.origin_url, str) or not self.origin_url)
            )
        ):
            raise ValueError("Git repository state is invalid")


@dataclass(frozen=True)
class AcceptedPrediction:
    path: str
    prediction: Prediction
    file: FileRecord
    members: Mapping[str, PredictionMember]

    def __post_init__(self) -> None:
        _relative_path(self.path)
        if not isinstance(self.prediction, Prediction):
            raise ValueError("accepted prediction must contain typed labels")
        if not isinstance(self.file, FileRecord):
            raise ValueError("accepted prediction file record is invalid")
        _require_member_schema(self.members)
        immutable_prediction = _immutable_prediction(self.prediction)
        canonical_artifact = encode_prediction_npz(immutable_prediction)
        if self.file != canonical_artifact.file or dict(self.members) != dict(
            canonical_artifact.members
        ):
            raise ValueError("accepted prediction metadata is not canonical")
        object.__setattr__(self, "prediction", immutable_prediction)
        object.__setattr__(self, "members", _typed_mapping_copy(self.members))


@dataclass(frozen=True)
class PackageValidationRecord:
    file_count: int
    total_byte_length: int
    tree_sha256: str
    manifest: FileRecord

    def __post_init__(self) -> None:
        if self.file_count != 53:
            raise ValueError("successful package must contain exactly 53 files")
        _plain_int(self.total_byte_length, "package total byte length", 1)
        _hash(self.tree_sha256, "package tree SHA-256")
        if not isinstance(self.manifest, FileRecord):
            raise ValueError("package manifest record is invalid")


@dataclass(frozen=True)
class AcceptedCandidatePackage:
    manifest: "SuccessfulManifest"
    observations: Tuple["ObservationPackageRow", ...]
    timings: Tuple["TimingPackageRow", ...]
    predictions: Tuple[AcceptedPrediction, ...]
    validation: PackageValidationRecord
    _acceptance_token: InitVar[Optional[object]] = None

    def __post_init__(self, _acceptance_token: Optional[object]) -> None:
        if _acceptance_token is not _ACCEPTANCE_TOKEN:
            raise ValueError(
                "accepted packages must be created by the acceptance factory"
            )
        if not isinstance(
            self.manifest, (RealSuccessfulManifest, SyntheticSuccessfulManifest)
        ):
            raise ValueError("accepted package manifest is invalid")
        if (
            type(self.observations) is not tuple
            or len(self.observations) != 50
            or any(
                not isinstance(value, ObservationPackageRow) or value.ordinal != ordinal
                for ordinal, value in enumerate(self.observations)
            )
            or type(self.timings) is not tuple
            or len(self.timings) != 100
            or any(not isinstance(value, TimingPackageRow) for value in self.timings)
            or type(self.predictions) is not tuple
            or len(self.predictions) != 50
            or any(
                not isinstance(value, AcceptedPrediction) for value in self.predictions
            )
            or not isinstance(self.validation, PackageValidationRecord)
        ):
            raise ValueError("accepted package payload is incomplete")
        if any(
            prediction.path != observation.prediction_path
            or prediction.file != observation.prediction
            or dict(prediction.members) != dict(observation.prediction_members)
            for observation, prediction in zip(self.observations, self.predictions)
        ):
            raise ValueError("accepted predictions do not bind observation order")
        _bind_timings(self.timings, self.observations, self.predictions)
        observations_record = _file_record(
            canonical_observations_bytes(self.observations)
        )
        timings_record = _file_record(canonical_timings_bytes(self.timings))
        if observations_record != _manifest_file_record(
            self.manifest, "observations"
        ) or timings_record != _manifest_file_record(self.manifest, "timings"):
            raise ValueError("accepted package payload records differ from manifest")
        manifest_record = _file_record(self.manifest.canonical_bytes())
        complete_records = (
            ("manifest.json", manifest_record),
            ("observations.jsonl", observations_record),
            ("timings.jsonl", timings_record),
        ) + tuple((prediction.path, prediction.file) for prediction in self.predictions)
        if self.validation != _package_validation_record(
            complete_records, manifest_record
        ):
            raise ValueError("accepted package validation record is inconsistent")


def _hash(data: object, label: str) -> str:
    if not isinstance(data, str) or _HASH.fullmatch(data) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return data


def _plain_int(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _file_record(data: bytes) -> FileRecord:
    return FileRecord(len(data), hashlib.sha256(data).hexdigest())


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, array, version=(1, 0), allow_pickle=False)
    return output.getvalue()


def _member(array: np.ndarray, npy: bytes) -> PredictionMember:
    logical = array.tobytes(order="C")
    return PredictionMember(
        array_byte_length=len(logical),
        array_sha256=hashlib.sha256(logical).hexdigest(),
        dtype=array.dtype.str,
        npy_byte_length=len(npy),
        npy_sha256=hashlib.sha256(npy).hexdigest(),
        shape=array.shape,
    )


def _require_member_schema(value: Mapping[str, PredictionMember]) -> None:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != _MEMBER_NAMES:
        raise ValueError("prediction members have an invalid schema")
    if any(not isinstance(value[name], PredictionMember) for name in _MEMBER_NAMES):
        raise ValueError("prediction members must be typed metadata")


def _typed_mapping_copy(
    value: Mapping[str, PredictionMember],
) -> Mapping[str, PredictionMember]:
    return _FrozenMapping[PredictionMember](
        tuple((name, value[name]) for name in _MEMBER_NAMES)
    )


def encode_prediction_npz(prediction: Prediction) -> PredictionArtifact:
    """Encode the two exact label arrays as a deterministic stored NPZ."""

    if not isinstance(prediction, Prediction):
        raise ValueError("prediction must be a Prediction")
    output = io.BytesIO()
    members: dict[str, PredictionMember] = {}
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in _MEMBER_NAMES:
            array = getattr(prediction, name)
            npy = _npy_bytes(array)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.comment = b""
            info.extra = b""
            archive.writestr(info, npy)
            members[name] = _member(array, npy)
    data = output.getvalue()
    return PredictionArtifact(data=data, file=_file_record(data), members=members)


def _parse_npy(name: str, payload: bytes) -> np.ndarray:
    stream = io.BytesIO(payload)
    try:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError(f"{name} must use NPY v1.0")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    except (EOFError, ValueError) as error:
        raise ValueError(f"{name} has an invalid NPY header") from error
    if dtype.hasobject:
        raise ValueError(f"{name} object dtype is forbidden")
    if fortran_order or dtype.str != _LABEL_DTYPE or shape != _LABEL_SHAPE:
        raise ValueError(f"{name} has wrong dtype, shape, or array order")
    raw = stream.read()
    if len(raw) != _LABEL_ARRAY_BYTES:
        raise ValueError(f"{name} has truncated or trailing array bytes")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _require_zip_end(data: bytes) -> None:
    offset = data.rfind(b"PK\x05\x06")
    if offset < 0 or offset + _ZIP_EOCD.size != len(data):
        raise ValueError("prediction ZIP end record is missing or has trailing bytes")
    values = _ZIP_EOCD.unpack_from(data, offset)
    (
        signature,
        disk,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = values
    if (
        signature != b"PK\x05\x06"
        or disk != 0
        or central_disk != 0
        or disk_entries != 2
        or total_entries != 2
        or central_offset + central_size != offset
        or comment_length != 0
    ):
        raise ValueError("prediction ZIP end metadata is invalid")


def _require_zip_info(data: bytes, info: zipfile.ZipInfo, expected_name: str) -> None:
    try:
        values = _ZIP_LOCAL_HEADER.unpack_from(data, info.header_offset)
    except struct.error as error:
        raise ValueError("prediction ZIP local header is truncated") from error
    (
        signature,
        extract_version,
        flags,
        compression,
        timestamp,
        date,
        crc,
        compressed_size,
        file_size,
        name_length,
        extra_length,
    ) = values
    name_start = info.header_offset + _ZIP_LOCAL_HEADER.size
    name_end = name_start + name_length
    try:
        local_name = data[name_start:name_end].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("prediction ZIP member name must be ASCII") from error
    expected_archive_name = f"{expected_name}.npy"
    if (
        signature != b"PK\x03\x04"
        or extract_version != 20
        or flags != 0
        or compression != zipfile.ZIP_STORED
        or timestamp != 0
        or date != 33
        or crc != info.CRC
        or compressed_size != info.compress_size
        or file_size != info.file_size
        or local_name != expected_archive_name
        or info.filename != expected_archive_name
        or extra_length != 0
        or info.compress_type != zipfile.ZIP_STORED
        or info.date_time != (1980, 1, 1, 0, 0, 0)
        or info.create_system != 3
        or info.external_attr != (stat.S_IFREG | 0o600) << 16
        or info.comment
        or info.extra
    ):
        raise ValueError("prediction ZIP member metadata is invalid")


def parse_prediction_npz_bytes(
    data: bytes,
    *,
    expected_file: FileRecord | None = None,
    expected_members: Mapping[str, PredictionMember] | None = None,
) -> ParsedPrediction:
    """Strict-read one prediction buffer and require its unique encoding."""

    if not isinstance(data, bytes) or not data or len(data) > _MAX_PREDICTION_BYTES:
        raise ValueError("prediction NPZ byte length is invalid")
    accepted_file = _file_record(data)
    if expected_file is not None and accepted_file != expected_file:
        raise ValueError("prediction NPZ file record mismatch")
    _require_zip_end(data)
    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
            if archive.comment:
                raise ValueError("prediction ZIP comment is forbidden")
            infos = archive.infolist()
            if tuple(info.filename for info in infos) != tuple(
                f"{name}.npy" for name in _MEMBER_NAMES
            ):
                raise ValueError("prediction ZIP members or order are invalid")
            arrays: dict[str, np.ndarray] = {}
            for name, info in zip(_MEMBER_NAMES, infos):
                _require_zip_info(data, info, name)
                if info.file_size != _LABEL_NPY_BYTES:
                    raise ValueError("prediction NPY member byte length is invalid")
                arrays[name] = _parse_npy(name, archive.read(info))
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError("prediction NPZ is invalid") from error
    prediction = Prediction(
        source_labels=arrays["source_labels"],
        mapped_labels=arrays["mapped_labels"],
    )
    canonical = encode_prediction_npz(prediction)
    if canonical.data != data:
        raise ValueError("prediction NPZ is not canonically encoded")
    if expected_members is not None:
        _require_member_schema(expected_members)
        if dict(canonical.members) != dict(expected_members):
            raise ValueError("prediction member metadata mismatch")
    return ParsedPrediction(
        prediction=prediction,
        file=accepted_file,
        members=canonical.members,
    )


def _endpoint_json(value: MetricEndpoint) -> Mapping[str, object]:
    return {
        "counts": asdict(value.counts),
        "eligible": value.eligible,
        "f1": value.f1,
        "iou": value.iou,
        "precision": value.precision,
        "recall": value.recall,
    }


def _parse_endpoint(value: object, label: str) -> MetricEndpoint:
    raw = _exact_object(
        value,
        ("counts", "eligible", "f1", "iou", "precision", "recall"),
        label,
    )
    counts = _exact_object(raw["counts"], ("fn", "fp", "tp"), f"{label}.counts")
    endpoint = MetricEndpoint.from_counts(
        ConfusionCounts(
            tp=_plain_int(counts["tp"], f"{label}.tp"),
            fp=_plain_int(counts["fp"], f"{label}.fp"),
            fn=_plain_int(counts["fn"], f"{label}.fn"),
        )
    )
    if _endpoint_json(endpoint) != raw:
        raise ValueError(f"{label} derived values are inconsistent")
    return endpoint


@dataclass(frozen=True)
class ObservationPackageRow:
    ordinal: int
    observation_id: str
    scene_id: str
    status: ObservationStatus
    failure_code: ObservationFailureCode | None
    prediction_path: str
    prediction: FileRecord
    prediction_members: Mapping[str, PredictionMember]
    metrics: ObservationMetrics

    def __post_init__(self) -> None:
        _plain_int(self.ordinal, "observation ordinal")
        if not 0 <= self.ordinal < 50:
            raise ValueError("observation ordinal must be in [0, 49]")
        if (
            not isinstance(self.observation_id, str)
            or _IDENTITY.fullmatch(self.observation_id) is None
        ):
            raise ValueError("observation ID is invalid")
        if (
            not isinstance(self.scene_id, str)
            or _SCENE.fullmatch(self.scene_id) is None
        ):
            raise ValueError("scene ID is invalid")
        if self.prediction_path != (
            f"predictions/{self.scene_id}/{self.ordinal:02d}-{self.observation_id}.npz"
        ):
            raise ValueError("prediction path is not identity-derived")
        _outcome(self.status, self.failure_code)
        if not isinstance(self.prediction, FileRecord):
            raise ValueError("prediction record must be a FileRecord")
        _require_member_schema(self.prediction_members)
        object.__setattr__(
            self, "prediction_members", _typed_mapping_copy(self.prediction_members)
        )
        if (
            not isinstance(self.metrics, ObservationMetrics)
            or self.metrics.ordinal != self.ordinal
            or self.metrics.scene_id != self.scene_id
        ):
            raise ValueError("observation metrics do not bind the row")


@dataclass(frozen=True)
class TimingPackageRow:
    sample: TimingSample

    def __post_init__(self) -> None:
        if not isinstance(self.sample, TimingSample):
            raise ValueError("timing row must contain a TimingSample")


def _member_json(value: PredictionMember) -> Mapping[str, object]:
    result = asdict(value)
    result["shape"] = list(value.shape)
    return result


def _observation_json(row: ObservationPackageRow) -> Mapping[str, object]:
    return {
        "failure_code": (None if row.failure_code is None else row.failure_code.value),
        "metrics": {
            "all_27": _endpoint_json(row.metrics.all_27),
            "per_category": [
                _endpoint_json(endpoint) for endpoint in row.metrics.per_category
            ],
            "primary": _endpoint_json(row.metrics.primary),
        },
        "observation_id": row.observation_id,
        "ordinal": row.ordinal,
        "prediction": asdict(row.prediction),
        "prediction_members": {
            name: _member_json(row.prediction_members[name]) for name in _MEMBER_NAMES
        },
        "prediction_path": row.prediction_path,
        "scene_id": row.scene_id,
        "status": row.status.value,
    }


def _components_json(value: ComponentTimings) -> Mapping[str, object]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _timing_json(row: TimingPackageRow) -> Mapping[str, object]:
    sample = row.sample
    return {
        "components": _components_json(sample.components),
        "end_to_end": sample.end_to_end,
        "failure_code": (
            None if sample.failure_code is None else sample.failure_code.value
        ),
        "mapped_labels_sha256": sample.mapped_labels_sha256,
        "ordinal": sample.ordinal,
        "pass_index": sample.pass_index,
        "sequence_index": sample.sequence_index,
        "source_labels_sha256": sample.source_labels_sha256,
        "status": sample.status.value,
    }


def canonical_observations_bytes(rows: Iterable[ObservationPackageRow]) -> bytes:
    materialized = tuple(rows)
    _validate_observation_sequence(materialized)
    return b"".join(
        canonical_json_bytes(_observation_json(row)) for row in materialized
    )


def canonical_timings_bytes(rows: Iterable[TimingPackageRow]) -> bytes:
    materialized = tuple(rows)
    _validate_timing_sequence(materialized)
    return b"".join(canonical_json_bytes(_timing_json(row)) for row in materialized)


def _strict_json_lines(data: bytes, label: str) -> Tuple[Mapping[str, object], ...]:
    if not isinstance(data, bytes) or not data.endswith(b"\n"):
        raise ValueError(f"{label} must be bytes with a trailing newline")
    try:
        data.decode("utf-8")
        values = tuple(
            json.loads(
                line,
                object_pairs_hook=lambda pairs: _unique_object(pairs, label),
                parse_constant=lambda value: _reject_constant(value, label),
            )
            for line in data.splitlines()
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSONL") from error
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"{label} rows must be JSON objects")
    return cast(Tuple[Mapping[str, object], ...], values)


def _prediction_member_from_json(value: object, label: str) -> PredictionMember:
    raw = _exact_object(
        value,
        (
            "array_byte_length",
            "array_sha256",
            "dtype",
            "npy_byte_length",
            "npy_sha256",
            "shape",
        ),
        label,
    )
    shape = raw["shape"]
    if not isinstance(shape, list):
        raise ValueError(f"{label}.shape must be a list")
    return PredictionMember(
        array_byte_length=cast(int, raw["array_byte_length"]),
        array_sha256=cast(str, raw["array_sha256"]),
        dtype=cast(str, raw["dtype"]),
        npy_byte_length=cast(int, raw["npy_byte_length"]),
        npy_sha256=cast(str, raw["npy_sha256"]),
        shape=tuple(cast(Sequence[int], shape)),
    )


def _observation_from_json(value: object) -> ObservationPackageRow:
    raw = _exact_object(
        value,
        (
            "failure_code",
            "metrics",
            "observation_id",
            "ordinal",
            "prediction",
            "prediction_members",
            "prediction_path",
            "scene_id",
            "status",
        ),
        "observation row",
    )
    prediction = _exact_object(
        raw["prediction"], ("byte_length", "sha256"), "observation prediction"
    )
    members = _exact_object(
        raw["prediction_members"], _MEMBER_NAMES, "observation prediction members"
    )
    metrics = _exact_object(
        raw["metrics"], ("all_27", "per_category", "primary"), "observation metrics"
    )
    per_category = metrics["per_category"]
    if not isinstance(per_category, list) or len(per_category) != 27:
        raise ValueError("observation per-category metrics must have 27 rows")
    ordinal = cast(int, raw["ordinal"])
    scene = cast(str, raw["scene_id"])
    failure_raw = raw["failure_code"]
    try:
        status = ObservationStatus(cast(str, raw["status"]))
        failure = (
            None
            if failure_raw is None
            else ObservationFailureCode(cast(str, failure_raw))
        )
    except (TypeError, ValueError) as error:
        raise ValueError("observation outcome is invalid") from error
    return ObservationPackageRow(
        ordinal=ordinal,
        observation_id=cast(str, raw["observation_id"]),
        scene_id=scene,
        status=status,
        failure_code=failure,
        prediction_path=cast(str, raw["prediction_path"]),
        prediction=FileRecord(
            byte_length=cast(int, prediction["byte_length"]),
            sha256=cast(str, prediction["sha256"]),
        ),
        prediction_members={
            name: _prediction_member_from_json(members[name], f"members.{name}")
            for name in _MEMBER_NAMES
        },
        metrics=ObservationMetrics(
            ordinal=ordinal,
            scene_id=scene,
            primary=_parse_endpoint(metrics["primary"], "metrics.primary"),
            all_27=_parse_endpoint(metrics["all_27"], "metrics.all_27"),
            per_category=tuple(
                _parse_endpoint(item, f"metrics.per_category[{index}]")
                for index, item in enumerate(per_category)
            ),
        ),
    )


def parse_observations_bytes(data: bytes) -> Tuple[ObservationPackageRow, ...]:
    rows = tuple(
        _observation_from_json(value)
        for value in _strict_json_lines(data, "observations")
    )
    _validate_observation_sequence(rows)
    if canonical_observations_bytes(rows) != data:
        raise ValueError("observations are not canonical JSONL")
    return rows


def _finite_optional(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite, nonnegative, or null")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite, nonnegative, or null")
    return result


def _timing_from_json(value: object) -> TimingPackageRow:
    raw = _exact_object(
        value,
        (
            "components",
            "end_to_end",
            "failure_code",
            "mapped_labels_sha256",
            "ordinal",
            "pass_index",
            "sequence_index",
            "source_labels_sha256",
            "status",
        ),
        "timing row",
    )
    components = _exact_object(
        raw["components"],
        ("d2h", "device_postprocess", "h2d", "inference", "preprocess", "projection"),
        "timing components",
    )
    failure_raw = raw["failure_code"]
    try:
        status = ObservationStatus(cast(str, raw["status"]))
        failure = (
            None
            if failure_raw is None
            else ObservationFailureCode(cast(str, failure_raw))
        )
    except (TypeError, ValueError) as error:
        raise ValueError("timing outcome is invalid") from error
    return TimingPackageRow(
        TimingSample(
            sequence_index=cast(int, raw["sequence_index"]),
            pass_index=cast(int, raw["pass_index"]),
            ordinal=cast(int, raw["ordinal"]),
            end_to_end=cast(float, raw["end_to_end"]),
            components=ComponentTimings(
                preprocess=_finite_optional(components["preprocess"], "preprocess"),
                h2d=_finite_optional(components["h2d"], "h2d"),
                inference=_finite_optional(components["inference"], "inference"),
                device_postprocess=_finite_optional(
                    components["device_postprocess"], "device_postprocess"
                ),
                d2h=_finite_optional(components["d2h"], "d2h"),
                projection=_finite_optional(components["projection"], "projection"),
            ),
            source_labels_sha256=cast(str, raw["source_labels_sha256"]),
            mapped_labels_sha256=cast(str, raw["mapped_labels_sha256"]),
            status=status,
            failure_code=failure,
        )
    )


def parse_timings_bytes(data: bytes) -> Tuple[TimingPackageRow, ...]:
    rows = tuple(
        _timing_from_json(value) for value in _strict_json_lines(data, "timings")
    )
    _validate_timing_sequence(rows)
    if canonical_timings_bytes(rows) != data:
        raise ValueError("timings are not canonical JSONL")
    return rows


def _validate_observation_sequence(rows: Sequence[ObservationPackageRow]) -> None:
    if len(rows) != 50 or any(row.ordinal != index for index, row in enumerate(rows)):
        raise ValueError("observations must contain ordered ordinals 0 through 49")
    if len({row.observation_id for row in rows}) != 50:
        raise ValueError("observation IDs must be unique")
    if len({row.prediction_path for row in rows}) != 50:
        raise ValueError("prediction paths must be unique")


def _validate_timing_sequence(rows: Sequence[TimingPackageRow]) -> None:
    if len(rows) != 100:
        raise ValueError("timings must contain exactly 100 rows")
    if any(row.sample.sequence_index != index + 20 for index, row in enumerate(rows)):
        raise ValueError("timings must follow the exact measured schedule")


class RunKind(str, Enum):
    REAL = "candidate"
    SYNTHETIC = "synthetic-contract"


@dataclass(frozen=True)
class StatisticsProtocolSummary:
    numpy_version: str
    bootstrap_matrix_sha256: str
    quantile_rule: str
    replicate_count: int
    scene_count: int
    seed: int

    def __post_init__(self) -> None:
        if (
            self.numpy_version != np.__version__
            or self.bootstrap_matrix_sha256 != BOOTSTRAP_MATRIX_SHA256
            or self.quantile_rule != "linear-h=(n-1)*p"
            or self.replicate_count != 10_000
            or self.scene_count != 11
            or self.seed != 20260728
        ):
            raise ValueError("statistics protocol differs from the frozen runtime")


@dataclass(frozen=True)
class RobustnessSummary:
    iou: RobustnessEstimate
    f1: RobustnessEstimate

    def __post_init__(self) -> None:
        if not isinstance(self.iou, RobustnessEstimate) or not isinstance(
            self.f1, RobustnessEstimate
        ):
            raise ValueError("robustness summary must contain typed estimates")


@dataclass(frozen=True)
class ColdStartSummary:
    model_construction_seconds: float
    checkpoint_byte_read_seconds: float
    checkpoint_load_seconds: float

    def __post_init__(self) -> None:
        if any(
            not _nonnegative_finite(value)
            for value in (
                self.model_construction_seconds,
                self.checkpoint_byte_read_seconds,
                self.checkpoint_load_seconds,
            )
        ):
            raise ValueError("cold-start measurements must be finite and nonnegative")


@dataclass(frozen=True)
class RealManifestSummary:
    cold_start: ColdStartSummary
    latency: LatencySummary
    resource: ResourceMeasurement
    gates: CandidateGateResult
    cuda_evidence: ArchivalCudaEvidence

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cold_start, ColdStartSummary)
            or not isinstance(self.latency, LatencySummary)
            or not isinstance(self.resource, ResourceMeasurement)
            or not isinstance(self.gates, CandidateGateResult)
            or not isinstance(self.cuda_evidence, ArchivalCudaEvidence)
        ):
            raise ValueError("real manifest summary is invalid")


@dataclass(frozen=True)
class ManifestTypedSummary:
    run_kind: RunKind
    candidate: CandidateCommitment
    p53_attestation: P53ValidationAttestation
    benchmark_attestation: BenchmarkEnvironmentAttestation
    coverage: StaticCoverage
    timing_protocol: TimingProtocol
    statistics_protocol: StatisticsProtocolSummary
    metrics: BenchmarkMetricSummary
    robustness: RobustnessSummary
    real: Optional[RealManifestSummary]

    def __post_init__(self) -> None:
        expected_real = self.run_kind is RunKind.REAL
        if (
            not isinstance(self.run_kind, RunKind)
            or not isinstance(self.candidate, CandidateCommitment)
            or not isinstance(self.p53_attestation, P53ValidationAttestation)
            or not isinstance(
                self.benchmark_attestation, BenchmarkEnvironmentAttestation
            )
            or not isinstance(self.coverage, StaticCoverage)
            or not isinstance(self.timing_protocol, TimingProtocol)
            or not isinstance(self.statistics_protocol, StatisticsProtocolSummary)
            or not isinstance(self.metrics, BenchmarkMetricSummary)
            or not isinstance(self.robustness, RobustnessSummary)
            or (self.real is not None) is not expected_real
            or (
                self.real is not None and not isinstance(self.real, RealManifestSummary)
            )
        ):
            raise ValueError("typed manifest summary is invalid")


_Value = TypeVar("_Value")


class _FrozenMapping(Mapping[str, _Value], Generic[_Value]):
    def __init__(self, items: Tuple[Tuple[str, _Value], ...]) -> None:
        self._items = items
        self._dict = dict(items)

    def __getitem__(self, key: str) -> _Value:
        return self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        items: list[Tuple[str, object]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object key must be a string")
            items.append((key, _freeze_json(item)))
        return _FrozenMapping(tuple(items))
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ArchivalCudaDeviceEvidence:
    gpu_name: str
    gpu_uuid: str
    driver_version: str
    clock_policy: str
    persistence_mode: str
    power_limit_watts: float
    temperature_celsius: float
    compute_pids: Tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.gpu_name != "NVIDIA GeForce RTX 3090"
            or not isinstance(self.gpu_uuid, str)
            or not self.gpu_uuid.startswith("GPU-")
            or any(
                not isinstance(value, str) or not value
                for value in (
                    self.driver_version,
                    self.clock_policy,
                    self.persistence_mode,
                )
            )
            or not _positive_finite(self.power_limit_watts)
            or isinstance(self.temperature_celsius, bool)
            or not isinstance(self.temperature_celsius, (int, float))
            or not math.isfinite(self.temperature_celsius)
            or not 30 <= self.temperature_celsius <= 80
            or type(self.compute_pids) is not tuple
            or any(type(pid) is not int or pid < 1 for pid in self.compute_pids)
            or len(set(self.compute_pids)) != len(self.compute_pids)
        ):
            raise ValueError("archival CUDA device evidence is invalid")


@dataclass(frozen=True)
class ArchivalCudaEvidence:
    before: ArchivalCudaDeviceEvidence
    after: ArchivalCudaDeviceEvidence
    historical_pid: int
    pytorch_cuda_alloc_conf: None

    def __post_init__(self) -> None:
        invariant = (
            "gpu_name",
            "gpu_uuid",
            "driver_version",
            "clock_policy",
            "persistence_mode",
            "power_limit_watts",
        )
        if (
            not isinstance(self.before, ArchivalCudaDeviceEvidence)
            or not isinstance(self.after, ArchivalCudaDeviceEvidence)
            or type(self.historical_pid) is not int
            or self.historical_pid < 1
            or self.pytorch_cuda_alloc_conf is not None
            or tuple(getattr(self.before, name) for name in invariant)
            != tuple(getattr(self.after, name) for name in invariant)
            or any(
                set(snapshot.compute_pids) - {self.historical_pid}
                for snapshot in (self.before, self.after)
            )
        ):
            raise ValueError("archival CUDA evidence is invalid or drifted")

    def to_json(self) -> Mapping[str, object]:
        return {
            "after": _cuda_device_json(self.after),
            "before": _cuda_device_json(self.before),
            "historical_pid": self.historical_pid,
            "pytorch_cuda_alloc_conf": None,
        }

    @classmethod
    def from_json(cls, value: object) -> "ArchivalCudaEvidence":
        raw = _exact_object(
            value,
            (
                "after",
                "before",
                "historical_pid",
                "pytorch_cuda_alloc_conf",
            ),
            "archival CUDA evidence",
        )
        return cls(
            before=_cuda_device_from_json(raw["before"], "cuda.before"),
            after=_cuda_device_from_json(raw["after"], "cuda.after"),
            historical_pid=cast(int, raw["historical_pid"]),
            pytorch_cuda_alloc_conf=cast(None, raw["pytorch_cuda_alloc_conf"]),
        )


def _cuda_device_json(value: ArchivalCudaDeviceEvidence) -> Mapping[str, object]:
    return {
        "clock_policy": value.clock_policy,
        "compute_pids": list(value.compute_pids),
        "driver_version": value.driver_version,
        "gpu_name": value.gpu_name,
        "gpu_uuid": value.gpu_uuid,
        "persistence_mode": value.persistence_mode,
        "power_limit_watts": value.power_limit_watts,
        "temperature_celsius": value.temperature_celsius,
    }


def _cuda_device_from_json(value: object, label: str) -> ArchivalCudaDeviceEvidence:
    raw = _exact_object(
        value,
        (
            "clock_policy",
            "compute_pids",
            "driver_version",
            "gpu_name",
            "gpu_uuid",
            "persistence_mode",
            "power_limit_watts",
            "temperature_celsius",
        ),
        label,
    )
    compute_pids = raw["compute_pids"]
    if not isinstance(compute_pids, list):
        raise ValueError(f"{label}.compute_pids must be a list")
    return ArchivalCudaDeviceEvidence(
        gpu_name=cast(str, raw["gpu_name"]),
        gpu_uuid=cast(str, raw["gpu_uuid"]),
        driver_version=cast(str, raw["driver_version"]),
        clock_policy=cast(str, raw["clock_policy"]),
        persistence_mode=cast(str, raw["persistence_mode"]),
        power_limit_watts=cast(float, raw["power_limit_watts"]),
        temperature_celsius=cast(float, raw["temperature_celsius"]),
        compute_pids=tuple(cast(Sequence[int], compute_pids)),
    )


_COMMON_MANIFEST_KEYS = (
    "attestations",
    "candidate_commitment",
    "candidate_id",
    "command",
    "coverage",
    "files",
    "metric_summary",
    "producer_commit_prefix",
    "producer_git_commit",
    "raw_inputs",
    "robustness",
    "run_kind",
    "run_status",
    "schema_version",
    "source_hashes",
    "statistics_protocol",
    "timing_protocol",
)
_REAL_MANIFEST_KEYS = _COMMON_MANIFEST_KEYS + (
    "candidate_status",
    "cold_start",
    "cuda_evidence",
    "gates",
    "latency",
    "permission_evidence",
    "resource",
)
_SYNTHETIC_MANIFEST_KEYS = _COMMON_MANIFEST_KEYS + ("decision", "latency_claim")
_SECTION_KEYS: Mapping[str, Tuple[str, ...]] = {
    "attestations": ("benchmark", "p53_validator"),
    "candidate_commitment": (
        "batch_views",
        "candidate_id",
        "checkpoint_id",
        "checkpoint_sha256",
        "checkpoint_url",
        "code_license_status",
        "depth_coordinate_semantics",
        "depth_interpolation",
        "depth_padding_value",
        "depth_units",
        "environment_lock_path",
        "environment_lock_sha256",
        "invalid_depth_policy",
        "mapping_sha256",
        "normalization",
        "precision_mode",
        "repository_url",
        "revision",
        "rgb_coordinate_semantics",
        "rgb_interpolation",
        "rgb_padding_value",
        "rgb_units",
        "source_dataset",
        "source_vocabulary",
        "synthetic",
        "weight_license_status",
    ),
    "coverage": (
        "covered_category_count",
        "covered_support_count",
        "support_ratio",
        "total_support_count",
    ),
    "files": ("observations", "predictions", "timings"),
    "metric_summary": ("all_27", "per_category", "primary"),
    "raw_inputs": (
        "cohort_jsonl_sha256",
        "raw_index_sha256",
        "raw_manifest_sha256",
        "raw_payload_tree_sha256",
        "raw_producer_git_commit",
        "selection_sha256",
    ),
    "robustness": ("f1", "iou"),
    "statistics_protocol": (
        "bootstrap_matrix_sha256",
        "numpy_version",
        "quantile_rule",
        "replicate_count",
        "scene_count",
        "seed",
    ),
    "timing_protocol": (
        "backend",
        "comparable",
        "measured_pass_count",
        "measured_sample_count",
        "unit",
        "warmup_count",
    ),
    "cold_start": (
        "checkpoint_byte_read_seconds",
        "checkpoint_load_seconds",
        "model_construction_seconds",
    ),
    "gates": (
        "complete_rows",
        "coverage",
        "latency",
        "license",
        "overall",
        "provenance",
        "quality",
        "resource",
    ),
    "latency": (
        "p50_seconds",
        "p95_seconds",
        "sample_count",
        "total_seconds",
        "views_per_second",
    ),
    "resource": (
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ),
    "permission_evidence": ("code", "weights"),
}
_SOURCE_HASH_KEYS = (
    "adapter",
    "constants",
    "contract",
    "mapping",
    "package",
    "projector",
)
_FIXED_SOURCE_PATHS: Mapping[str, str] = {
    "constants": "prior/constants.py",
    "contract": ("prior/analyze/d2026_07_29/rgbd_segmenter_benchmark_contract.py"),
    "mapping": "prior/analyze/d2026_07_29/rgbd_segmenter_nyu40_mapping.json",
    "package": "prior/analyze/d2026_07_29/rgbd_segmenter_benchmark_package.py",
    "projector": ("vlnce_baselines/models/etp_llm/llm_grid_oracle_cache.py"),
}
_ABORTED_MANIFEST_KEYS = (
    "attestations",
    "candidate_commitment",
    "candidate_id",
    "candidate_status",
    "command",
    "cuda_evidence",
    "failure_code",
    "failure_stage",
    "permission_evidence",
    "producer_commit_prefix",
    "producer_git_commit",
    "raw_inputs",
    "resource_gate",
    "resource_snapshot",
    "run_kind",
    "run_status",
    "schema_version",
    "source_hashes",
)


@dataclass(frozen=True)
class RealSuccessfulManifest:
    fields: Mapping[str, object]
    summary: ManifestTypedSummary = field(init=False, repr=False)

    def __post_init__(self) -> None:
        summary = _validate_manifest(self.fields, RunKind.REAL)
        object.__setattr__(
            self, "fields", cast(Mapping[str, object], _freeze_json(dict(self.fields)))
        )
        object.__setattr__(self, "summary", summary)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(_thaw_json(self.fields))


@dataclass(frozen=True)
class SyntheticSuccessfulManifest:
    fields: Mapping[str, object]
    summary: ManifestTypedSummary = field(init=False, repr=False)

    def __post_init__(self) -> None:
        summary = _validate_manifest(self.fields, RunKind.SYNTHETIC)
        object.__setattr__(
            self, "fields", cast(Mapping[str, object], _freeze_json(dict(self.fields)))
        )
        object.__setattr__(self, "summary", summary)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(_thaw_json(self.fields))


SuccessfulManifest = Union[RealSuccessfulManifest, SyntheticSuccessfulManifest]


@dataclass(frozen=True)
class CandidateValidationRecord:
    schema_version: int
    validation_status: str
    candidate_status: str
    run_kind: RunKind
    candidate_id: str
    producer_git_commit: str
    p53_attestation_sha256: str
    benchmark_attestation_sha256: str
    synthetic_semantic_scores_exposed: bool
    observation_count: int
    package: PackageValidationRecord
    _validation_token: InitVar[Optional[object]] = None

    def __post_init__(self, _validation_token: Optional[object]) -> None:
        if _validation_token is not _VALIDATION_TOKEN:
            raise ValueError("validation records must be created by the validator")
        if (
            self.schema_version != 1
            or self.validation_status != "PASS"
            or not isinstance(self.run_kind, RunKind)
            or not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or not isinstance(self.package, PackageValidationRecord)
            or self.synthetic_semantic_scores_exposed is not False
            or self.observation_count != 50
            or not isinstance(self.producer_git_commit, str)
            or _COMMIT.fullmatch(self.producer_git_commit) is None
        ):
            raise ValueError("candidate validation record is invalid")
        expected_candidate_statuses = (
            {"NOT_APPLICABLE"}
            if self.run_kind is RunKind.SYNTHETIC
            else {"PASS", "FAIL"}
        )
        if self.candidate_status not in expected_candidate_statuses:
            raise ValueError("candidate status is invalid for validation run kind")
        _hash(self.p53_attestation_sha256, "validation P5.3 attestation SHA-256")
        _hash(
            self.benchmark_attestation_sha256,
            "validation benchmark attestation SHA-256",
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes({
            "benchmark_attestation_sha256": self.benchmark_attestation_sha256,
            "candidate_id": self.candidate_id,
            "candidate_status": self.candidate_status,
            "observation_count": self.observation_count,
            "p53_attestation_sha256": self.p53_attestation_sha256,
            "package": {
                "file_count": self.package.file_count,
                "manifest": {
                    "byte_length": self.package.manifest.byte_length,
                    "sha256": self.package.manifest.sha256,
                },
                "total_byte_length": self.package.total_byte_length,
                "tree_sha256": self.package.tree_sha256,
            },
            "producer_git_commit": self.producer_git_commit,
            "run_kind": self.run_kind.value,
            "schema_version": self.schema_version,
            "validation_status": self.validation_status,
            "synthetic_semantic_scores_exposed": (
                self.synthetic_semantic_scores_exposed
            ),
        })


@dataclass(frozen=True)
class ValidatedCandidatePackage:
    validation: PackageValidationRecord
    record: CandidateValidationRecord
    _validation_token: InitVar[Optional[object]] = None

    def __post_init__(self, _validation_token: Optional[object]) -> None:
        if _validation_token is not _VALIDATION_TOKEN:
            raise ValueError("validated packages must be created by the validator")
        if (
            not isinstance(self.validation, PackageValidationRecord)
            or not isinstance(self.record, CandidateValidationRecord)
            or self.record.package != self.validation
        ):
            raise ValueError("validated package result is inconsistent")


@dataclass(frozen=True)
class AbortedOomManifest:
    fields: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_aborted_manifest(self.fields)
        object.__setattr__(
            self,
            "fields",
            cast(Mapping[str, object], _freeze_json(dict(self.fields))),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(_thaw_json(self.fields))


def _validate_aborted_manifest(value: Mapping[str, object]) -> None:
    raw = _exact_object(value, _ABORTED_MANIFEST_KEYS, "aborted OOM manifest")
    _validate_json_tree(raw, "aborted OOM manifest")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["run_kind"] != RunKind.REAL.value
        or raw["run_status"] != "aborted"
        or raw["candidate_status"] != "FAIL_RESOURCE"
        or raw["resource_gate"] != "FAIL"
        or raw["failure_code"] != "CUDA_OUT_OF_MEMORY"
    ):
        raise ValueError("aborted OOM manifest literals are invalid")
    candidate_id = raw["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("aborted candidate ID is invalid")
    commitment = _candidate_from_json(raw["candidate_commitment"])
    if commitment.synthetic or commitment.candidate_id != candidate_id:
        raise ValueError("aborted OOM candidate commitment is invalid")
    commit = raw["producer_git_commit"]
    if (
        not isinstance(commit, str)
        or _COMMIT.fullmatch(commit) is None
        or raw["producer_commit_prefix"] != commit[:12]
    ):
        raise ValueError("aborted OOM producer commit is invalid")
    command = raw["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError("aborted OOM command is invalid")
    try:
        TimingStage(cast(str, raw["failure_stage"]))
    except (TypeError, ValueError) as error:
        raise ValueError("aborted OOM failure stage is invalid") from error
    _validate_source_hashes(raw["source_hashes"])
    _validate_raw_inputs(raw["raw_inputs"])
    _validate_attestations(raw["attestations"], comparable=True)
    _validate_permission_evidence(raw["permission_evidence"])
    ArchivalCudaEvidence.from_json(raw["cuda_evidence"])
    snapshot = _exact_object(
        raw["resource_snapshot"],
        (
            "baseline_allocated_bytes",
            "baseline_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ),
        "aborted resource snapshot",
    )
    baseline_allocated = _plain_int(
        snapshot["baseline_allocated_bytes"], "baseline allocated bytes"
    )
    baseline_reserved = _plain_int(
        snapshot["baseline_reserved_bytes"], "baseline reserved bytes"
    )
    if baseline_allocated > baseline_reserved:
        raise ValueError("aborted resource baseline is inconsistent")
    for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        peak_value = snapshot[name]
        if peak_value is not None:
            _plain_int(peak_value, f"aborted {name}")
    peak_allocated = snapshot["peak_allocated_bytes"]
    peak_reserved = snapshot["peak_reserved_bytes"]
    if peak_allocated is not None and cast(int, peak_allocated) < baseline_allocated:
        raise ValueError("aborted allocated peak is below its baseline")
    if peak_reserved is not None and cast(int, peak_reserved) < baseline_reserved:
        raise ValueError("aborted reserved peak is below its baseline")
    if (
        peak_allocated is not None
        and peak_reserved is not None
        and cast(int, peak_allocated) > cast(int, peak_reserved)
    ):
        raise ValueError("aborted resource peaks are inconsistent")
    _validate_json_tree(raw, "aborted OOM manifest")


def parse_aborted_oom_manifest_bytes(data: bytes) -> AbortedOomManifest:
    manifest = AbortedOomManifest(_strict_json_object(data, "aborted OOM manifest"))
    if manifest.canonical_bytes() != data:
        raise ValueError("aborted OOM manifest is not canonical JSON")
    return manifest


def _validate_manifest(
    value: Mapping[str, object], kind: RunKind
) -> ManifestTypedSummary:
    expected = _REAL_MANIFEST_KEYS if kind is RunKind.REAL else _SYNTHETIC_MANIFEST_KEYS
    raw = _exact_object(value, expected, "successful manifest")
    _validate_json_tree(raw, "successful manifest")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["run_kind"] != kind.value
        or raw["run_status"] != "success"
    ):
        raise ValueError("successful manifest literals are invalid")
    candidate_id = raw["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate ID must be non-empty")
    commit = raw["producer_git_commit"]
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("producer commit must be a lowercase 40-character commit")
    if raw["producer_commit_prefix"] != commit[:12]:
        raise ValueError(
            "producer commit prefix must be exactly the first 12 characters"
        )
    command = raw["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError("command must be a non-empty string list")
    for section, keys in _SECTION_KEYS.items():
        if section in raw:
            _exact_object(raw[section], keys, f"manifest.{section}")
    _validate_source_hashes(raw["source_hashes"])
    candidate = cast(Mapping[str, object], raw["candidate_commitment"])
    if candidate["candidate_id"] != candidate_id or candidate["synthetic"] is not (
        kind is RunKind.SYNTHETIC
    ):
        raise ValueError("candidate commitment does not bind manifest identity")
    timing = cast(Mapping[str, object], raw["timing_protocol"])
    if (
        timing["warmup_count"] != 20
        or timing["measured_pass_count"] != 2
        or timing["measured_sample_count"] != 100
    ):
        raise ValueError("timing protocol schedule has drifted")
    if kind is RunKind.SYNTHETIC:
        if (
            raw["decision"] != "NOT_APPLICABLE"
            or raw["latency_claim"] is not False
            or timing["backend"] != "deterministic-fake"
            or timing["unit"] != "synthetic-tick"
            or timing["comparable"] is not False
        ):
            raise ValueError("synthetic manifest non-claim contract is invalid")
    else:
        if timing["comparable"] is not True:
            raise ValueError("real successful manifest timing must be comparable")
    summary = _validate_manifest_sections(raw, kind)
    _validate_json_tree(raw, "successful manifest")
    return summary


def _validate_manifest_sections(
    raw: Mapping[str, object], kind: RunKind
) -> ManifestTypedSummary:
    candidate = _candidate_from_json(raw["candidate_commitment"])
    _validate_raw_inputs(raw["raw_inputs"])
    p53, benchmark = _validate_attestations(
        raw["attestations"], comparable=kind is RunKind.REAL
    )
    coverage = _validate_coverage(raw["coverage"])
    timing = _validate_timing_protocol(raw["timing_protocol"], kind)
    statistics = _validate_statistics_protocol(raw["statistics_protocol"])
    _validate_files(raw["files"])
    metrics = _validate_metric_summary(raw["metric_summary"])
    robustness = _validate_robustness(raw["robustness"])
    real = None
    if kind is RunKind.REAL:
        _validate_permission_evidence(raw["permission_evidence"])
        real = _validate_real_sections(raw)
    return ManifestTypedSummary(
        run_kind=kind,
        candidate=candidate,
        p53_attestation=p53,
        benchmark_attestation=benchmark,
        coverage=coverage,
        timing_protocol=timing,
        statistics_protocol=statistics,
        metrics=metrics,
        robustness=robustness,
        real=real,
    )


def _candidate_from_json(value: object) -> CandidateCommitment:
    raw = _exact_object(
        value, _SECTION_KEYS["candidate_commitment"], "candidate commitment"
    )
    vocabulary = raw["source_vocabulary"]
    if not isinstance(vocabulary, list):
        raise ValueError("candidate source vocabulary must be a list")
    try:
        return CandidateCommitment(
            candidate_id=cast(str, raw["candidate_id"]),
            synthetic=cast(bool, raw["synthetic"]),
            repository_url=cast(str, raw["repository_url"]),
            revision=cast(str, raw["revision"]),
            checkpoint_id=cast(str, raw["checkpoint_id"]),
            checkpoint_url=cast(str, raw["checkpoint_url"]),
            checkpoint_sha256=cast(str, raw["checkpoint_sha256"]),
            source_dataset=cast(str, raw["source_dataset"]),
            code_license_status=LicenseStatus(cast(str, raw["code_license_status"])),
            weight_license_status=LicenseStatus(
                cast(str, raw["weight_license_status"])
            ),
            environment_lock_path=cast(str, raw["environment_lock_path"]),
            environment_lock_sha256=cast(str, raw["environment_lock_sha256"]),
            rgb_units=cast(str, raw["rgb_units"]),
            depth_units=cast(str, raw["depth_units"]),
            batch_views=cast(int, raw["batch_views"]),
            source_vocabulary=tuple(cast(Sequence[str], vocabulary)),
            mapping_sha256=cast(str, raw["mapping_sha256"]),
            rgb_interpolation=cast(str, raw["rgb_interpolation"]),
            rgb_coordinate_semantics=cast(str, raw["rgb_coordinate_semantics"]),
            depth_interpolation=cast(str, raw["depth_interpolation"]),
            depth_coordinate_semantics=cast(str, raw["depth_coordinate_semantics"]),
            normalization=cast(str, raw["normalization"]),
            invalid_depth_policy=cast(str, raw["invalid_depth_policy"]),
            rgb_padding_value=cast(str, raw["rgb_padding_value"]),
            depth_padding_value=cast(str, raw["depth_padding_value"]),
            precision_mode=cast(str, raw["precision_mode"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("candidate commitment is invalid") from error


def _validate_raw_inputs(value: object) -> None:
    raw = _exact_object(value, _SECTION_KEYS["raw_inputs"], "manifest.raw_inputs")
    for name in (
        "cohort_jsonl_sha256",
        "raw_index_sha256",
        "raw_manifest_sha256",
        "raw_payload_tree_sha256",
        "selection_sha256",
    ):
        _hash(raw[name], f"raw_inputs.{name}")
    commit = raw["raw_producer_git_commit"]
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("raw producer commit is invalid")


def _validate_source_hashes(value: object) -> None:
    raw = _exact_object(value, _SOURCE_HASH_KEYS, "manifest.source_hashes")
    for role in _SOURCE_HASH_KEYS:
        record = _exact_object(raw[role], ("path", "sha256"), f"source_hashes.{role}")
        path = record["path"]
        if not isinstance(path, str):
            raise ValueError(f"source_hashes.{role}.path is invalid")
        _relative_path(path)
        expected_path = _FIXED_SOURCE_PATHS.get(role)
        if expected_path is not None and path != expected_path:
            raise ValueError(f"source_hashes.{role}.path differs from frozen role")
        _hash(record["sha256"], f"source_hashes.{role}.sha256")


def _validate_permission_evidence(value: object) -> None:
    raw = _exact_object(
        value, _SECTION_KEYS["permission_evidence"], "permission_evidence"
    )
    for role in ("code", "weights"):
        record = _exact_object(
            raw[role],
            ("byte_length", "path", "root_role", "sha256"),
            f"permission_evidence.{role}",
        )
        if record["root_role"] not in {
            "benchmark_repository",
            "candidate_repository",
        }:
            raise ValueError(f"permission_evidence.{role}.root_role is invalid")
        path = record["path"]
        if not isinstance(path, str):
            raise ValueError(f"permission_evidence.{role}.path is invalid")
        _relative_path(path)
        _plain_int(
            record["byte_length"],
            f"permission_evidence.{role}.byte_length",
            1,
        )
        _hash(record["sha256"], f"permission_evidence.{role}.sha256")


def _validate_attestations(
    value: object, *, comparable: bool
) -> Tuple[P53ValidationAttestation, BenchmarkEnvironmentAttestation]:
    raw = _exact_object(value, _SECTION_KEYS["attestations"], "manifest.attestations")
    p53 = _embedded_attestation(raw["p53_validator"], "P5.3 attestation")
    p53_attestation = P53ValidationAttestation.parse(canonical_json_bytes(p53))
    benchmark = _embedded_attestation(
        raw["benchmark"], "benchmark environment attestation"
    )
    benchmark_raw = _exact_object(
        benchmark,
        (
            "environment_sha256",
            "gpu_name",
            "gpu_uuid",
            "python_executable",
            "timing_comparable",
            "visible_device_count",
        ),
        "benchmark environment attestation value",
    )
    try:
        attestation = BenchmarkEnvironmentAttestation(
            python_executable=cast(str, benchmark_raw["python_executable"]),
            environment_sha256=cast(str, benchmark_raw["environment_sha256"]),
            visible_device_count=cast(int, benchmark_raw["visible_device_count"]),
            gpu_name=cast(str, benchmark_raw["gpu_name"]),
            gpu_uuid=cast(str, benchmark_raw["gpu_uuid"]),
            timing_comparable=cast(bool, benchmark_raw["timing_comparable"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("benchmark environment attestation is invalid") from error
    if attestation.timing_comparable is not comparable:
        raise ValueError("benchmark attestation timing claim differs from run kind")
    return p53_attestation, attestation


def _embedded_attestation(value: object, label: str) -> Mapping[str, object]:
    raw = _exact_object(value, ("sha256", "value"), label)
    accepted = _exact_object(
        raw["value"], tuple(cast(Mapping[str, object], raw["value"])), f"{label}.value"
    )
    expected = hashlib.sha256(canonical_json_bytes(accepted)).hexdigest()
    if raw["sha256"] != expected:
        raise ValueError(f"{label} hash mismatch")
    return accepted


def _validate_coverage(value: object) -> StaticCoverage:
    raw = _exact_object(value, _SECTION_KEYS["coverage"], "manifest.coverage")
    coverage = StaticCoverage(
        covered_category_count=cast(int, raw["covered_category_count"]),
        covered_support_count=cast(int, raw["covered_support_count"]),
        total_support_count=cast(int, raw["total_support_count"]),
    )
    if raw["support_ratio"] != coverage.support_ratio:
        raise ValueError("coverage support ratio is inconsistent")
    return coverage


def _validate_timing_protocol(value: object, kind: RunKind) -> TimingProtocol:
    raw = _exact_object(
        value, _SECTION_KEYS["timing_protocol"], "manifest.timing_protocol"
    )
    if (
        not isinstance(raw["backend"], str)
        or not raw["backend"]
        or not isinstance(raw["unit"], str)
        or not raw["unit"]
        or type(raw["comparable"]) is not bool
    ):
        raise ValueError("timing protocol types are invalid")
    if kind is RunKind.REAL and (
        raw["backend"] != "official-cuda" or raw["unit"] != "seconds"
    ):
        raise ValueError("real timing protocol must be official CUDA seconds")
    return TimingProtocol(
        backend=cast(str, raw["backend"]),
        unit=cast(str, raw["unit"]),
        comparable=raw["comparable"],
    )


def _validate_statistics_protocol(value: object) -> StatisticsProtocolSummary:
    raw = _exact_object(
        value,
        _SECTION_KEYS["statistics_protocol"],
        "manifest.statistics_protocol",
    )
    if raw["bootstrap_matrix_sha256"] != BOOTSTRAP_MATRIX_SHA256:
        raise ValueError("bootstrap matrix SHA-256 differs from the frozen contract")
    try:
        return StatisticsProtocolSummary(
            numpy_version=cast(str, raw["numpy_version"]),
            bootstrap_matrix_sha256=cast(str, raw["bootstrap_matrix_sha256"]),
            quantile_rule=cast(str, raw["quantile_rule"]),
            replicate_count=cast(int, raw["replicate_count"]),
            scene_count=cast(int, raw["scene_count"]),
            seed=cast(int, raw["seed"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("statistics protocol has drifted") from error


def _validate_files(value: object) -> None:
    raw = _exact_object(value, _SECTION_KEYS["files"], "manifest.files")
    for name, count in (("observations", 50), ("timings", 100)):
        record = _exact_object(
            raw[name], ("byte_length", "row_count", "sha256"), f"files.{name}"
        )
        FileRecord(
            byte_length=cast(int, record["byte_length"]),
            sha256=cast(str, record["sha256"]),
        )
        if record["row_count"] != count:
            raise ValueError(f"files.{name} row count is invalid")
    predictions = _exact_object(
        raw["predictions"],
        ("file_count", "total_byte_length", "tree_sha256"),
        "files.predictions",
    )
    if predictions["file_count"] != 50:
        raise ValueError("prediction tree must contain exactly 50 files")
    _plain_int(predictions["total_byte_length"], "prediction tree byte length", 1)
    _hash(predictions["tree_sha256"], "prediction tree SHA-256")


_AGGREGATE_KEYS = (
    "eligible_observation_count",
    "empty_both_count",
    "mean_f1",
    "mean_iou",
    "observation_count",
    "pooled",
    "scene_macro_f1",
    "scene_macro_iou",
    "target_empty_prediction_nonempty_count",
    "target_nonempty_prediction_empty_count",
)


def _validate_metric_summary(value: object) -> BenchmarkMetricSummary:
    raw = _exact_object(
        value, _SECTION_KEYS["metric_summary"], "manifest.metric_summary"
    )
    primary = _metric_aggregate(raw["primary"], "metric_summary.primary")
    all_27 = _metric_aggregate(raw["all_27"], "metric_summary.all_27")
    categories = raw["per_category"]
    if not isinstance(categories, list) or len(categories) != 27:
        raise ValueError("metric summary must contain 27 category aggregates")
    per_category = tuple(
        _metric_aggregate(item, f"metric_summary.per_category[{index}]")
        for index, item in enumerate(categories)
    )
    return BenchmarkMetricSummary(
        primary=primary,
        all_27=all_27,
        per_category=per_category,
    )


def _metric_aggregate(value: object, label: str) -> MetricAggregate:
    raw = _exact_object(value, _AGGREGATE_KEYS, label)
    endpoint = _parse_endpoint(raw["pooled"], f"{label}.pooled")
    return MetricAggregate(
        observation_count=cast(int, raw["observation_count"]),
        eligible_observation_count=cast(int, raw["eligible_observation_count"]),
        empty_both_count=cast(int, raw["empty_both_count"]),
        target_empty_prediction_nonempty_count=cast(
            int, raw["target_empty_prediction_nonempty_count"]
        ),
        target_nonempty_prediction_empty_count=cast(
            int, raw["target_nonempty_prediction_empty_count"]
        ),
        mean_iou=cast(Optional[float], raw["mean_iou"]),
        mean_f1=cast(Optional[float], raw["mean_f1"]),
        pooled=endpoint,
        scene_macro_iou=cast(Optional[float], raw["scene_macro_iou"]),
        scene_macro_f1=cast(Optional[float], raw["scene_macro_f1"]),
    )


_ESTIMATE_KEYS = (
    "interval_high",
    "interval_low",
    "leave_one_scene_out_max",
    "leave_one_scene_out_min",
    "point_estimate",
    "replicate_count",
)


def _validate_robustness(value: object) -> RobustnessSummary:
    raw = _exact_object(value, _SECTION_KEYS["robustness"], "manifest.robustness")
    estimates: dict[str, RobustnessEstimate] = {}
    for name in ("iou", "f1"):
        estimate = _exact_object(
            raw[name], _ESTIMATE_KEYS, f"manifest.robustness.{name}"
        )
        estimates[name] = RobustnessEstimate(
            point_estimate=cast(float, estimate["point_estimate"]),
            interval_low=cast(float, estimate["interval_low"]),
            interval_high=cast(float, estimate["interval_high"]),
            leave_one_scene_out_min=cast(float, estimate["leave_one_scene_out_min"]),
            leave_one_scene_out_max=cast(float, estimate["leave_one_scene_out_max"]),
            replicate_count=cast(int, estimate["replicate_count"]),
        )
    return RobustnessSummary(iou=estimates["iou"], f1=estimates["f1"])


def _validate_real_sections(raw: Mapping[str, object]) -> RealManifestSummary:
    cold = cast(Mapping[str, object], raw["cold_start"])
    cold_start = ColdStartSummary(
        model_construction_seconds=cast(float, cold["model_construction_seconds"]),
        checkpoint_byte_read_seconds=cast(float, cold["checkpoint_byte_read_seconds"]),
        checkpoint_load_seconds=cast(float, cold["checkpoint_load_seconds"]),
    )
    latency = cast(Mapping[str, object], raw["latency"])
    latency_summary = LatencySummary(
        timing_comparable=True,
        unit="seconds",
        sample_count=cast(int, latency["sample_count"]),
        p50_seconds=cast(float, latency["p50_seconds"]),
        p95_seconds=cast(float, latency["p95_seconds"]),
        total_seconds=cast(float, latency["total_seconds"]),
        views_per_second=cast(float, latency["views_per_second"]),
    )
    resource = cast(Mapping[str, object], raw["resource"])
    resource_measurement = ResourceMeasurement(
        baseline_allocated_bytes=cast(int, resource["baseline_allocated_bytes"]),
        peak_allocated_bytes=cast(int, resource["peak_allocated_bytes"]),
        baseline_reserved_bytes=cast(int, resource["baseline_reserved_bytes"]),
        peak_reserved_bytes=cast(int, resource["peak_reserved_bytes"]),
    )
    gates = cast(Mapping[str, object], raw["gates"])
    try:
        gate_result = CandidateGateResult(
            coverage=GateStatus(cast(str, gates["coverage"])),
            complete_rows=GateStatus(cast(str, gates["complete_rows"])),
            quality=GateStatus(cast(str, gates["quality"])),
            latency=GateStatus(cast(str, gates["latency"])),
            resource=GateStatus(cast(str, gates["resource"])),
            license=GateStatus(cast(str, gates["license"])),
            provenance=GateStatus(cast(str, gates["provenance"])),
            overall=GateStatus(cast(str, gates["overall"])),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("candidate gates are invalid") from error
    if raw["candidate_status"] != gate_result.overall.value:
        raise ValueError("candidate status differs from overall gate")
    return RealManifestSummary(
        cold_start=cold_start,
        latency=latency_summary,
        resource=resource_measurement,
        gates=gate_result,
        cuda_evidence=ArchivalCudaEvidence.from_json(raw["cuda_evidence"]),
    )


def parse_successful_manifest_bytes(data: bytes) -> SuccessfulManifest:
    raw = _strict_json_object(data, "successful manifest")
    kind_raw = raw.get("run_kind")
    try:
        kind = RunKind(cast(str, kind_raw))
    except (TypeError, ValueError) as error:
        raise ValueError("successful manifest run kind is invalid") from error
    manifest: SuccessfulManifest
    if kind is RunKind.REAL:
        manifest = RealSuccessfulManifest(raw)
    else:
        manifest = SyntheticSuccessfulManifest(raw)
    if manifest.canonical_bytes() != data:
        raise ValueError("successful manifest is not canonical JSON")
    return manifest


def _strict_json_object(data: bytes, label: str) -> Mapping[str, object]:
    if not isinstance(data, bytes):
        raise ValueError(f"{label} must be bytes")
    try:
        data.decode("utf-8")
        value = json.loads(
            data,
            object_pairs_hook=lambda pairs: _unique_object(pairs, label),
            parse_constant=lambda item: _reject_constant(item, label),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _unique_object(
    pairs: Sequence[Tuple[str, object]], label: str
) -> Mapping[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{label} has duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str, label: str) -> object:
    raise ValueError(f"{label} contains non-finite number {value}")


def _exact_object(
    value: object, expected: Sequence[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} has an invalid schema")
    return cast(Mapping[str, object], value)


def _validate_json_tree(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_tree(item, label)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} contains a non-string key")
        for item in value.values():
            _validate_json_tree(item, label)
        return
    raise ValueError(f"{label} contains a non-JSON value")


def _nonnegative_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _positive_finite(value: object) -> bool:
    return _nonnegative_finite(value) and cast(float, value) > 0


def _outcome(
    status: ObservationStatus, failure_code: ObservationFailureCode | None
) -> None:
    if not isinstance(status, ObservationStatus):
        raise ValueError("status must be an ObservationStatus")
    if (status is ObservationStatus.PASS) != (failure_code is None):
        raise ValueError("failure code must be absent exactly for PASS")
    if failure_code is not None and not isinstance(
        failure_code, ObservationFailureCode
    ):
        raise ValueError("failure code is invalid")


@dataclass(frozen=True)
class _Fingerprint:
    device: int
    inode: int
    mode: int
    link_count: int
    user: int
    group: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _DirectoryIdentity:
    device: int
    inode: int
    mode: int


@dataclass(frozen=True)
class _TreeSnapshot:
    root: _Fingerprint
    directories: Mapping[str, _Fingerprint]
    files: Mapping[str, _Fingerprint]


class _DescriptorCleanupError(RuntimeError):
    def __init__(self, primary: BaseException, cleanup: BaseException) -> None:
        super().__init__(
            "candidate package validation failed and descriptor cleanup also failed"
        )
        self.primary = primary
        self.cleanup = cleanup


def _fingerprint(value: os.stat_result) -> _Fingerprint:
    return _Fingerprint(
        device=value.st_dev,
        inode=value.st_ino,
        mode=value.st_mode,
        link_count=value.st_nlink,
        user=value.st_uid,
        group=value.st_gid,
        size=value.st_size,
        modified_ns=value.st_mtime_ns,
        changed_ns=value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> _DirectoryIdentity:
    return _DirectoryIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        mode=value.st_mode,
    )


def _relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError("package path must be normalized printable relative POSIX")


def _absolute_authority_root(value: Path, label: str) -> None:
    if (
        not isinstance(value, Path)
        or not value.is_absolute()
        or value == Path(value.anchor)
        or Path(os.path.abspath(value)) != value
    ):
        raise ValueError(f"{label} must be a normalized absolute non-root path")


def _open_anchored_root(
    root: Path,
) -> Tuple[
    int,
    Tuple[int, ...],
    Tuple[Tuple[int, str, _DirectoryIdentity], ...],
]:
    if not isinstance(root, Path) or not root.is_absolute() or root == Path("/"):
        raise ValueError("candidate package root must be an absolute non-root Path")
    if any(part in {"", ".", ".."} for part in root.parts[1:]):
        raise ValueError("candidate package root is not normalized")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptors: list[int] = []
    bindings: list[Tuple[int, str, _DirectoryIdentity]] = []
    try:
        descriptor = os.open("/", flags)
        descriptors.append(descriptor)
        for component in root.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("candidate package path contains a non-directory")
            descriptors.append(child)
            bindings.append((descriptor, component, _directory_identity(metadata)))
            descriptor = child
        return descriptor, tuple(descriptors), tuple(bindings)
    except BaseException:
        _close_descriptors(tuple(reversed(descriptors)))
        raise


def _close_descriptors(descriptors: Sequence[int]) -> None:
    failures: list[OSError] = []
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as error:
            failures.append(error)
    if failures:
        raise RuntimeError(
            "failed to close candidate package descriptors"
        ) from failures[0]


def _recapture_ancestor_bindings(
    bindings: Sequence[Tuple[int, str, _DirectoryIdentity]],
) -> None:
    for parent, name, expected in bindings:
        try:
            metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except OSError as error:
            raise ValueError("candidate package ancestor binding changed") from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _directory_identity(metadata) != expected
        ):
            raise ValueError("candidate package ancestor binding changed")


def _trusted_file_record(root: Path, path: str) -> FileRecord:
    """Descriptor-read one authority-rooted regular file without symlink traversal."""

    _relative_path(path)
    try:
        root_descriptor, descriptors, bindings = _open_anchored_root(root)
    except OSError as error:
        raise ValueError("trusted artifact root is unsafe") from error
    opened: list[int] = []
    directories: list[Tuple[int, _Fingerprint]] = [
        (root_descriptor, _fingerprint(os.fstat(root_descriptor)))
    ]
    name_bindings: list[Tuple[int, str, _Fingerprint]] = []
    try:
        current = root_descriptor
        parts = PurePosixPath(path).parts
        for part in parts[:-1]:
            descriptor = os.open(
                part,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
                dir_fd=current,
            )
            opened.append(descriptor)
            fingerprint = _fingerprint(os.fstat(descriptor))
            name_bindings.append((current, part, fingerprint))
            current = descriptor
            directories.append((descriptor, fingerprint))
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=current,
        )
        opened.append(descriptor)
        before = _fingerprint(os.fstat(descriptor))
        name_bindings.append((current, parts[-1], before))
        if (
            not stat.S_ISREG(before.mode)
            or before.link_count != 1
            or not 0 < before.size <= _MAX_TRUSTED_ARTIFACT_BYTES
        ):
            raise ValueError(f"trusted artifact is not a bounded unique file: {path}")
        digest = hashlib.sha256()
        byte_length = 0
        remaining = before.size
        while remaining:
            chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                raise ValueError(f"trusted artifact was truncated: {path}")
            digest.update(chunk)
            byte_length += len(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"trusted artifact grew while reading: {path}")
        if _fingerprint(os.fstat(descriptor)) != before or any(
            _fingerprint(os.fstat(directory)) != expected
            for directory, expected in directories
        ):
            raise ValueError(f"trusted artifact changed while reading: {path}")
        for parent, name, expected in name_bindings:
            try:
                recaptured = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"trusted artifact binding changed: {path}") from error
            if _fingerprint(recaptured) != expected:
                raise ValueError(f"trusted artifact binding changed: {path}")
        _recapture_ancestor_bindings(bindings)
        return FileRecord(byte_length=byte_length, sha256=digest.hexdigest())
    except OSError as error:
        raise ValueError(f"trusted artifact is unsafe: {path}") from error
    finally:
        _close_descriptors(tuple(reversed(opened)) + tuple(reversed(descriptors)))


def _trusted_cohort_from_raw_index(
    root: Path,
    *,
    p53_attestation: P53ValidationAttestation,
) -> TrustedCohort:
    """Read only sealed target-free index identities before package acceptance."""

    if (
        str(root) != p53_attestation.raw_root
        or p53_attestation.raw_index_sha256 != RAW_INDEX_SHA256
        or p53_attestation.raw_manifest_sha256 != RAW_MANIFEST_SHA256
        or p53_attestation.producer_git_commit != RAW_PRODUCER_COMMIT
    ):
        raise ValueError("raw index authority differs from P5.3 attestation")
    try:
        root_descriptor, descriptors, bindings = _open_anchored_root(root)
    except OSError as error:
        raise ValueError("raw index root is unsafe") from error
    opened: list[int] = []
    try:
        root_before = _fingerprint(os.fstat(root_descriptor))
        descriptor = os.open(
            "index.jsonl",
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=root_descriptor,
        )
        opened.append(descriptor)
        before = _fingerprint(os.fstat(descriptor))
        if (
            not stat.S_ISREG(before.mode)
            or before.link_count != 1
            or before.size != RAW_INDEX_BYTE_LENGTH
        ):
            raise ValueError("raw index is not the exact sealed regular file")
        chunks: list[bytes] = []
        remaining = before.size
        while remaining:
            chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                raise ValueError("raw index was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("raw index grew while reading")
        data = b"".join(chunks)
        if (
            _fingerprint(os.fstat(descriptor)) != before
            or _fingerprint(os.fstat(root_descriptor)) != root_before
            or hashlib.sha256(data).hexdigest() != RAW_INDEX_SHA256
        ):
            raise ValueError("raw index changed or differs from frozen authority")
        recaptured = os.stat(
            "index.jsonl",
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if _fingerprint(recaptured) != before:
            raise ValueError("raw index name binding changed")
        _recapture_ancestor_bindings(bindings)
        rows = parse_index_bytes(data)
        return TrustedCohort(
            tuple((row.ordinal, row.observation_id, row.scene_id) for row in rows)
        )
    except OSError as error:
        raise ValueError("raw index is unsafe") from error
    finally:
        _close_descriptors(tuple(reversed(opened)) + tuple(reversed(descriptors)))


def _origin_from_safe_local_git_config(data: bytes) -> Optional[str]:
    if (
        not isinstance(data, bytes)
        or len(data) > _MAX_LOCAL_GIT_CONFIG_BYTES
        or (data and not data.endswith(b"\x00"))
    ):
        raise ValueError("local Git config output is invalid")
    unsafe_exact = {
        "core.attributesfile",
        "core.fsmonitor",
        "core.hookspath",
        "core.worktree",
        "diff.external",
        "extensions.objectformat",
        "extensions.partialclone",
        "extensions.worktreeconfig",
    }
    origins: list[str] = []
    for entry in data.split(b"\x00")[:-1]:
        if entry.count(b"\n") < 1:
            raise ValueError("local Git config entry is invalid")
        key_bytes, value_bytes = entry.split(b"\n", 1)
        try:
            key = key_bytes.decode("utf-8").lower()
            value = value_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("local Git config is not valid UTF-8") from error
        if (
            not key
            or key in unsafe_exact
            or key.startswith(("filter.", "include.", "includeif."))
        ):
            raise ValueError(f"unsafe local Git config key: {key}")
        if key == "remote.origin.url":
            if not value or any(
                ord(character) < 0x20 or ord(character) == 0x7F for character in value
            ):
                raise ValueError("Git origin URL is invalid")
            origins.append(value)
    if len(origins) > 1:
        raise ValueError("local Git config has ambiguous origin URLs")
    return origins[0] if origins else None


def _read_git_metadata_file(descriptor: int, label: str) -> bytes:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size < 0
        or metadata.st_size > _MAX_GIT_METADATA_FILE_BYTES
    ):
        raise ValueError(f"{label} is not a bounded unique regular file")
    before = _fingerprint(metadata)
    data = bytearray()
    remaining = metadata.st_size
    while remaining:
        chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
        if not chunk:
            raise ValueError(f"{label} was truncated")
        data.extend(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1) or _fingerprint(os.fstat(descriptor)) != before:
        raise ValueError(f"{label} changed while reading")
    return bytes(data)


@dataclass(frozen=True)
class _GitTreeEntry:
    mode: str
    object_id: str

    def __post_init__(self) -> None:
        if self.mode not in {"100644", "100755", "120000"}:
            raise ValueError("unsupported Git tree file mode")
        if _GIT_OBJECT_ID.fullmatch(self.object_id) is None:
            raise ValueError("invalid Git tree object ID")


def _parse_git_tree(data: bytes) -> Mapping[str, _GitTreeEntry]:
    if (
        not isinstance(data, bytes)
        or len(data) > _MAX_GIT_TREE_BYTES
        or (data and not data.endswith(b"\x00"))
    ):
        raise ValueError("Git tree output is invalid")
    parsed: dict[str, _GitTreeEntry] = {}
    for entry in data.split(b"\x00")[:-1]:
        try:
            header, raw_path = entry.split(b"\t", 1)
            mode, object_type, object_id = header.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("Git tree entry is invalid") from error
        _relative_path(path)
        if path == ".git" or path.startswith(".git/") or path in parsed:
            raise ValueError("Git tree path is unsafe or duplicated")
        if any(
            path.startswith(f"{existing}/") or existing.startswith(f"{path}/")
            for existing in parsed
        ):
            raise ValueError("Git tree paths have conflicting prefixes")
        if mode == "160000" or object_type == "commit":
            raise ValueError("Git submodules are unsupported")
        if object_type != "blob":
            raise ValueError("Git tree contains an unsupported object type")
        parsed[path] = _GitTreeEntry(mode=mode, object_id=object_id)
    return _FrozenMapping(tuple(sorted(parsed.items())))


def _git_blob_digest(length: int, chunks: Iterable[bytes]) -> str:
    digest = hashlib.new("sha1", usedforsecurity=False)
    digest.update(f"blob {length}\0".encode("ascii"))
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _GitWorktreeCapture:
    root: _Fingerprint
    entries: Mapping[str, _GitTreeEntry]
    directories: Mapping[str, _Fingerprint]
    bindings: Mapping[str, _Fingerprint]
    extra_paths: Tuple[str, ...]


def _capture_git_worktree(
    root_descriptor: int,
    *,
    git_directory_fingerprint: _Fingerprint,
) -> _GitWorktreeCapture:
    captured: dict[str, _GitTreeEntry] = {}
    directories: dict[str, _Fingerprint] = {}
    bindings: dict[str, _Fingerprint] = {}
    extra_paths: list[str] = []
    root_before = _fingerprint(os.fstat(root_descriptor))

    def walk(descriptor: int, relative: str) -> None:
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise ValueError("Git worktree directory cannot be listed") from error
        for name in names:
            if relative == "" and name == ".git":
                if (
                    _fingerprint(
                        os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    )
                    != git_directory_fingerprint
                ):
                    raise ValueError("Git metadata directory binding changed")
                continue
            path = f"{relative}/{name}" if relative else name
            _relative_path(path)
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise ValueError(
                    f"Git worktree path cannot be inspected: {path}"
                ) from error
            before = _fingerprint(metadata)
            if stat.S_ISDIR(metadata.st_mode):
                directories[path] = before
                try:
                    child = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                except OSError as error:
                    raise ValueError(
                        f"Git worktree directory is unsafe: {path}"
                    ) from error
                try:
                    if _fingerprint(os.fstat(child)) != before:
                        raise ValueError(f"Git worktree directory changed: {path}")
                    walk(child, path)
                    if (
                        _fingerprint(os.fstat(child)) != before
                        or _fingerprint(
                            os.stat(
                                name,
                                dir_fd=descriptor,
                                follow_symlinks=False,
                            )
                        )
                        != before
                    ):
                        raise ValueError(f"Git worktree directory changed: {path}")
                finally:
                    os.close(child)
                continue
            if stat.S_ISREG(metadata.st_mode):
                if (
                    metadata.st_nlink != 1
                    or metadata.st_size < 0
                    or metadata.st_size > _MAX_TRUSTED_ARTIFACT_BYTES
                ):
                    raise ValueError(f"Git worktree file is unsafe: {path}")
                try:
                    file_descriptor = os.open(
                        name,
                        os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                except OSError as error:
                    raise ValueError(f"Git worktree file is unsafe: {path}") from error
                try:
                    if _fingerprint(os.fstat(file_descriptor)) != before:
                        raise ValueError(f"Git worktree file changed: {path}")
                    remaining = metadata.st_size

                    def chunks() -> Iterator[bytes]:
                        nonlocal remaining
                        while remaining:
                            chunk = os.read(
                                file_descriptor,
                                min(remaining, _READ_CHUNK_BYTES),
                            )
                            if not chunk:
                                raise ValueError(
                                    f"Git worktree file was truncated: {path}"
                                )
                            remaining -= len(chunk)
                            yield chunk
                        if os.read(file_descriptor, 1):
                            raise ValueError(
                                f"Git worktree file grew while reading: {path}"
                            )

                    object_id = _git_blob_digest(metadata.st_size, chunks())
                    if (
                        _fingerprint(os.fstat(file_descriptor)) != before
                        or _fingerprint(
                            os.stat(
                                name,
                                dir_fd=descriptor,
                                follow_symlinks=False,
                            )
                        )
                        != before
                    ):
                        raise ValueError(f"Git worktree file changed: {path}")
                finally:
                    os.close(file_descriptor)
                mode = "100755" if metadata.st_mode & stat.S_IXUSR else "100644"
            elif stat.S_ISLNK(metadata.st_mode):
                try:
                    target = os.readlink(name, dir_fd=descriptor)
                    recaptured = os.stat(
                        name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise ValueError(
                        f"Git worktree symlink is unsafe: {path}"
                    ) from error
                if _fingerprint(recaptured) != before:
                    raise ValueError(f"Git worktree symlink changed: {path}")
                target_bytes = os.fsencode(target)
                object_id = _git_blob_digest(len(target_bytes), (target_bytes,))
                mode = "120000"
            else:
                raise ValueError(f"Git worktree entry type is unsupported: {path}")
            captured[path] = _GitTreeEntry(mode=mode, object_id=object_id)
            bindings[path] = before
            extra_paths.append(path)

    walk(root_descriptor, "")
    if _fingerprint(os.fstat(root_descriptor)) != root_before:
        raise ValueError("Git worktree root changed during capture")
    return _GitWorktreeCapture(
        root=root_before,
        entries=_FrozenMapping(tuple(sorted(captured.items()))),
        directories=_FrozenMapping(tuple(sorted(directories.items()))),
        bindings=_FrozenMapping(tuple(sorted(bindings.items()))),
        extra_paths=tuple(sorted(extra_paths)),
    )


def _parse_git_check_ignore(data: bytes) -> Mapping[str, Tuple[str, str]]:
    if (
        not isinstance(data, bytes)
        or len(data) > _MAX_GIT_IGNORE_BYTES
        or (data and not data.endswith(b"\x00"))
    ):
        raise ValueError("Git check-ignore output is invalid")
    fields = data.split(b"\x00")[:-1]
    if len(fields) % 4:
        raise ValueError("Git check-ignore verbose output is malformed")
    matches: dict[str, Tuple[str, str]] = {}
    for index in range(0, len(fields), 4):
        try:
            source = fields[index].decode("utf-8")
            line_number = fields[index + 1].decode("ascii")
            pattern = fields[index + 2].decode("utf-8")
            path = fields[index + 3].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Git check-ignore output has invalid encoding") from error
        _relative_path(source)
        _relative_path(path)
        if (
            not line_number.isdigit()
            or int(line_number) < 1
            or not pattern
            or pattern.startswith("!")
            or path in matches
        ):
            raise ValueError("Git check-ignore output is ambiguous")
        matches[path] = (source, pattern)
    return _FrozenMapping(tuple(sorted(matches.items())))


def _inspect_git_repository(root: Path) -> GitRepositoryState:
    _absolute_authority_root(root, "Git repository root")
    try:
        root_descriptor, descriptors, bindings = _open_anchored_root(root)
    except OSError as error:
        raise ValueError("Git repository root is unsafe") from error
    git_descriptor: Optional[int] = None
    git_directory_descriptor: Optional[int] = None
    metadata_descriptors: list[int] = []
    metadata_files: list[Tuple[int, str, int, _Fingerprint, bytes]] = []
    metadata_directories: list[Tuple[int, str, int, _Fingerprint]] = []
    try:
        root_before = _fingerprint(os.fstat(root_descriptor))
        try:
            git_directory_descriptor = os.open(
                ".git",
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=root_descriptor,
            )
        except OSError as error:
            raise ValueError(
                "Git repository must contain an anchored .git directory"
            ) from error
        git_directory_before = _fingerprint(os.fstat(git_directory_descriptor))
        if not stat.S_ISDIR(git_directory_before.mode):
            raise ValueError("Git metadata is not a directory")

        def open_metadata_directory(parent: int, name: str, label: str) -> int:
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=parent,
                )
            except OSError as error:
                raise ValueError(f"{label} is unsafe or missing") from error
            metadata_descriptors.append(descriptor)
            fingerprint = _fingerprint(os.fstat(descriptor))
            if not stat.S_ISDIR(fingerprint.mode):
                raise ValueError(f"{label} is not a directory")
            if (
                _fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False))
                != fingerprint
            ):
                raise ValueError(f"{label} binding changed")
            metadata_directories.append((parent, name, descriptor, fingerprint))
            return descriptor

        def open_metadata_file(parent: int, name: str, label: str) -> bytes:
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=parent,
                )
            except OSError as error:
                raise ValueError(f"{label} is unsafe or missing") from error
            metadata_descriptors.append(descriptor)
            fingerprint = _fingerprint(os.fstat(descriptor))
            data = _read_git_metadata_file(descriptor, label)
            if (
                _fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False))
                != fingerprint
            ):
                raise ValueError(f"{label} binding changed")
            metadata_files.append((parent, name, descriptor, fingerprint, data))
            return data

        head_bytes = open_metadata_file(git_directory_descriptor, "HEAD", "Git HEAD")
        if not head_bytes or not head_bytes.endswith(b"\n"):
            raise ValueError("Git HEAD is invalid")
        head_value = head_bytes[:-1]
        if head_value.startswith(b"ref: "):
            try:
                head_reference = head_value[5:].decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError("Git HEAD reference is invalid") from error
            _relative_path(head_reference)
            if not head_reference.startswith("refs/heads/"):
                raise ValueError(
                    "only loose local branch HEAD references are supported"
                )
            reference_parent = git_directory_descriptor
            reference_parts = PurePosixPath(head_reference).parts
            for index, part in enumerate(reference_parts[:-1]):
                reference_parent = open_metadata_directory(
                    reference_parent,
                    part,
                    f"Git HEAD reference directory {index}",
                )
            reference_bytes = open_metadata_file(
                reference_parent,
                reference_parts[-1],
                "Git HEAD loose reference",
            )
            if not reference_bytes.endswith(b"\n"):
                raise ValueError("Git HEAD loose reference is invalid")
            revision_bytes = reference_bytes[:-1]
        else:
            revision_bytes = head_value
        try:
            revision = revision_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("Git HEAD object ID has invalid encoding") from error
        if _GIT_OBJECT_ID.fullmatch(revision) is None:
            raise ValueError("Git HEAD object ID is invalid")

        config_bytes = open_metadata_file(
            git_directory_descriptor,
            "config",
            "Git local config",
        )
        if not config_bytes:
            raise ValueError("Git local config is empty")
        config_descriptor = metadata_files[-1][2]
        objects_descriptor = open_metadata_directory(
            git_directory_descriptor,
            "objects",
            "Git objects directory",
        )
        objects_info_descriptor = open_metadata_directory(
            objects_descriptor,
            "info",
            "Git objects info directory",
        )
        try:
            os.stat(
                "alternates",
                dir_fd=objects_info_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError("Git object alternates are unsupported")
        info_descriptor = open_metadata_directory(
            git_directory_descriptor,
            "info",
            "Git info directory",
        )
        open_metadata_file(
            info_descriptor,
            "exclude",
            "Git info exclude",
        )
        try:
            git_descriptor = os.open(
                _TRUSTED_GIT_PATH,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
        except OSError as error:
            raise ValueError("trusted Git executable cannot be opened") from error
        git_before = _fingerprint(os.fstat(git_descriptor))
        if (
            not stat.S_ISREG(git_before.mode)
            or git_before.user != 0
            or git_before.mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise ValueError("Git executable is not trusted")
        repository_fd_path = f"/proc/self/fd/{root_descriptor}"
        git_directory_fd_path = f"/proc/self/fd/{git_directory_descriptor}"
        git_prefix = (
            _TRUSTED_GIT_PATH,
            "--no-pager",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "diff.external=",
            "-c",
            "core.excludesFile=/dev/null",
            "-c",
            "core.ignoreCase=false",
            f"--git-dir={git_directory_fd_path}",
            f"--work-tree={repository_fd_path}",
        )
        git_environment = {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
        }
        try:
            config = subprocess.run(
                (
                    _TRUSTED_GIT_PATH,
                    "--no-pager",
                    "config",
                    f"--file=/proc/self/fd/{config_descriptor}",
                    "--no-includes",
                    "--null",
                    "--list",
                ),
                check=True,
                capture_output=True,
                env=git_environment,
                pass_fds=(config_descriptor,),
                timeout=_GIT_INSPECTION_TIMEOUT_SECONDS,
            )
            if config.stderr:
                raise ValueError("Git config inspection emitted unexpected stderr")
            origin_url = _origin_from_safe_local_git_config(config.stdout)
            passed_descriptors = (
                root_descriptor,
                git_directory_descriptor,
                objects_descriptor,
            )
            tree_result = subprocess.run(
                git_prefix
                + (
                    "ls-tree",
                    "-r",
                    "-z",
                    "--full-tree",
                    revision,
                ),
                check=True,
                capture_output=True,
                env=git_environment,
                pass_fds=passed_descriptors,
                timeout=_GIT_INSPECTION_TIMEOUT_SECONDS,
            )
            tree = _parse_git_tree(tree_result.stdout)
            worktree = _capture_git_worktree(
                root_descriptor,
                git_directory_fingerprint=git_directory_before,
            )
            tracked_ignore_paths: set[str] = set()
            for ignore_path, entry in tree.items():
                if PurePosixPath(ignore_path).name != ".gitignore":
                    continue
                if entry.mode not in {"100644", "100755"}:
                    raise ValueError("tracked .gitignore must be a regular file")
                parent = root_descriptor
                parts = PurePosixPath(ignore_path).parts
                for index, part in enumerate(parts[:-1]):
                    parent = open_metadata_directory(
                        parent,
                        part,
                        f"tracked .gitignore ancestor {index}",
                    )
                ignore_bytes = open_metadata_file(
                    parent,
                    parts[-1],
                    "tracked .gitignore",
                )
                if (
                    _git_blob_digest(len(ignore_bytes), (ignore_bytes,))
                    != entry.object_id
                ):
                    raise ValueError("tracked .gitignore differs from Git tree")
                tracked_ignore_paths.add(ignore_path)
            extra_paths = tuple(
                path for path in worktree.extra_paths if path not in tree
            )
            ignore_result = subprocess.run(
                git_prefix
                + (
                    "check-ignore",
                    "--no-index",
                    "-v",
                    "-z",
                    "--stdin",
                ),
                check=False,
                capture_output=True,
                env=git_environment,
                input=b"".join(os.fsencode(path) + b"\x00" for path in extra_paths),
                pass_fds=passed_descriptors,
                timeout=_GIT_INSPECTION_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError("Git repository inspection failed") from error
        if any(
            result.stderr
            for result in (
                tree_result,
                ignore_result,
            )
        ):
            raise ValueError("Git inspection emitted unexpected stderr")
        if ignore_result.returncode not in {0, 1}:
            raise ValueError("Git check-ignore failed")
        final_worktree = _capture_git_worktree(
            root_descriptor,
            git_directory_fingerprint=git_directory_before,
        )
        if final_worktree != worktree:
            raise ValueError("Git worktree changed during inspection")
        ignored = _parse_git_check_ignore(ignore_result.stdout)
        if set(ignored) != set(extra_paths) or any(
            source not in tracked_ignore_paths for source, _ in ignored.values()
        ):
            extras_clean = False
        else:
            extras_clean = True
        tracked_clean = all(
            worktree.entries.get(path) == entry for path, entry in tree.items()
        )
        for parent, name, descriptor, expected, expected_bytes in metadata_files:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if (
                _fingerprint(os.fstat(descriptor)) != expected
                or _read_git_metadata_file(descriptor, name) != expected_bytes
                or _fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False))
                != expected
            ):
                raise ValueError("Git metadata file changed during inspection")
        for parent, name, descriptor, expected in metadata_directories:
            if (
                _fingerprint(os.fstat(descriptor)) != expected
                or _fingerprint(os.stat(name, dir_fd=parent, follow_symlinks=False))
                != expected
            ):
                raise ValueError("Git metadata directory changed during inspection")
        try:
            os.stat(
                "alternates",
                dir_fd=objects_info_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError("Git object alternates changed during inspection")
        if _fingerprint(os.fstat(root_descriptor)) != root_before:
            raise ValueError("Git repository root changed during inspection")
        if _fingerprint(os.fstat(git_directory_descriptor)) != git_directory_before:
            raise ValueError("Git metadata directory changed during inspection")
        try:
            git_directory_path_after = _fingerprint(
                os.stat(".git", dir_fd=root_descriptor, follow_symlinks=False)
            )
        except OSError as error:
            raise ValueError("Git metadata directory binding changed") from error
        if git_directory_path_after != git_directory_before:
            raise ValueError("Git metadata directory binding changed")
        try:
            git_path_after = _fingerprint(
                os.stat(_TRUSTED_GIT_PATH, follow_symlinks=False)
            )
        except OSError as error:
            raise ValueError("trusted Git executable binding changed") from error
        if (
            _fingerprint(os.fstat(git_descriptor)) != git_before
            or git_path_after != git_before
        ):
            raise ValueError("trusted Git executable binding changed")
        _recapture_ancestor_bindings(bindings)
        return GitRepositoryState(
            commit=revision,
            clean=tracked_clean and extras_clean,
            origin_url=origin_url,
        )
    finally:
        git_descriptors = (() if git_descriptor is None else (git_descriptor,)) + (
            () if git_directory_descriptor is None else (git_directory_descriptor,)
        )
        _close_descriptors(
            git_descriptors
            + tuple(reversed(metadata_descriptors))
            + tuple(reversed(descriptors))
        )


def _capture_tree(root_descriptor: int) -> _TreeSnapshot:
    root_metadata = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("candidate package root must remain a directory")
    directories: dict[str, _Fingerprint] = {}
    files: dict[str, _Fingerprint] = {}
    count = [0]

    def walk(descriptor: int, relative: str, depth: int) -> None:
        if depth >= _MAX_PACKAGE_DEPTH:
            raise ValueError("candidate package tree exceeds maximum depth")
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise ValueError("candidate package directory cannot be listed") from error
        if not names:
            raise ValueError("candidate package contains an empty directory")
        for name in names:
            if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                raise ValueError("candidate package contains an invalid entry name")
            path = f"{relative}/{name}" if relative else name
            _relative_path(path)
            count[0] += 1
            if count[0] > _MAX_PACKAGE_ENTRIES:
                raise ValueError("candidate package contains too many entries")
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise ValueError(
                    "candidate package entry cannot be inspected"
                ) from error
            fingerprint = _fingerprint(metadata)
            if stat.S_ISDIR(metadata.st_mode):
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
                try:
                    child = os.open(name, flags, dir_fd=descriptor)
                except OSError as error:
                    raise ValueError("candidate package directory is unsafe") from error
                try:
                    if _fingerprint(os.fstat(child)) != fingerprint:
                        raise ValueError("candidate package directory changed")
                    directories[path] = fingerprint
                    walk(child, path, depth + 1)
                    if _fingerprint(os.fstat(child)) != fingerprint:
                        raise ValueError("candidate package directory changed")
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise ValueError("candidate package hard-linked file is forbidden")
                files[path] = fingerprint
            else:
                raise ValueError("candidate package entry type is forbidden")

    walk(root_descriptor, "", 0)
    return _TreeSnapshot(
        root=_fingerprint(root_metadata),
        directories=_FrozenMapping(tuple(sorted(directories.items()))),
        files=_FrozenMapping(tuple(sorted(files.items()))),
    )


def _expected_tree(
    trusted_cohort: TrustedCohort,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if not isinstance(trusted_cohort, TrustedCohort):
        raise ValueError("trusted cohort authority is required")
    directories = ("predictions",) + tuple(
        f"predictions/{scene}" for scene in trusted_cohort.scenes
    )
    for path in directories:
        _relative_path(path)
    predictions = tuple(
        f"predictions/{scene}/{ordinal:02d}-{observation_id}.npz"
        for ordinal, observation_id, scene in trusted_cohort.identities
    )
    for path in predictions:
        _relative_path(path)
    files = ("manifest.json", "observations.jsonl", "timings.jsonl") + predictions
    return tuple(sorted(directories)), tuple(sorted(files))


def _require_exact_tree(
    snapshot: _TreeSnapshot,
    expected_directories: Sequence[str],
    expected_files: Sequence[str],
) -> None:
    if tuple(sorted(snapshot.directories)) != tuple(expected_directories):
        raise ValueError("candidate package directories differ from the exact tree")
    if tuple(sorted(snapshot.files)) != tuple(expected_files):
        raise ValueError("candidate package files differ from the exact tree")


def _max_file_bytes(path: str) -> int:
    if path == "manifest.json":
        return _MAX_MANIFEST_BYTES
    if path == "observations.jsonl":
        return _MAX_OBSERVATIONS_BYTES
    if path == "timings.jsonl":
        return _MAX_TIMINGS_BYTES
    if path.startswith("predictions/") and path.endswith(".npz"):
        return _MAX_PREDICTION_BYTES
    raise ValueError("candidate package path is outside the frozen tree")


def _read_file_at(
    root_descriptor: int,
    path: str,
    *,
    expected_fingerprint: _Fingerprint,
    expected_directories: Mapping[str, _Fingerprint],
    expected_record: FileRecord | None = None,
) -> Tuple[bytes, FileRecord]:
    _relative_path(path)
    parts = PurePosixPath(path).parts
    opened: list[int] = []
    opened_directories: list[Tuple[int, _Fingerprint]] = []
    descriptor = root_descriptor
    relative_directory = ""
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        for component in parts[:-1]:
            descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            opened.append(descriptor)
            relative_directory = (
                f"{relative_directory}/{component}" if relative_directory else component
            )
            expected_directory = expected_directories.get(relative_directory)
            if (
                expected_directory is None
                or _fingerprint(os.fstat(descriptor)) != expected_directory
            ):
                raise ValueError(
                    f"candidate package directory changed: {relative_directory}"
                )
            opened_directories.append((descriptor, expected_directory))
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        opened.append(file_descriptor)
    except BaseException as error:
        _close_descriptors(tuple(reversed(opened)))
        raise ValueError(f"candidate package file is unsafe: {path}") from error
    try:
        before = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _fingerprint(before) != expected_fingerprint
            or before.st_size < 1
            or before.st_size > _max_file_bytes(path)
        ):
            raise ValueError(f"candidate package file metadata is invalid: {path}")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(file_descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError(f"candidate package file is truncated: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(file_descriptor, 1):
            raise ValueError(f"candidate package file grew while reading: {path}")
        after = os.fstat(file_descriptor)
        if _fingerprint(after) != expected_fingerprint:
            raise ValueError(f"candidate package file changed while reading: {path}")
        if any(
            _fingerprint(os.fstat(directory)) != expected
            for directory, expected in opened_directories
        ):
            raise ValueError(
                f"candidate package directory changed while reading: {path}"
            )
    finally:
        _close_descriptors(tuple(reversed(opened)))
    data = b"".join(chunks)
    record = _file_record(data)
    if expected_record is not None and record != expected_record:
        raise ValueError(f"candidate package file record mismatch: {path}")
    return data, record


def _manifest_file_record(manifest: SuccessfulManifest, name: str) -> FileRecord:
    fields = manifest.fields
    files = _mapping(fields["files"], "manifest.files")
    record = _mapping(files[name], f"manifest.files.{name}")
    return FileRecord(
        byte_length=cast(int, record["byte_length"]),
        sha256=cast(str, record["sha256"]),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _bind_manifest_authority(
    manifest: SuccessfulManifest,
    authority: CandidateMappingAuthority,
) -> None:
    fields = _mapping(manifest.fields, "manifest")
    commitment_raw = _mapping(
        fields["candidate_commitment"], "manifest.candidate_commitment"
    )
    vocabulary = commitment_raw["source_vocabulary"]
    if (
        not isinstance(vocabulary, tuple)
        or tuple(vocabulary) != authority.source_vocabulary
        or commitment_raw["mapping_sha256"] != authority.mapping_sha256
    ):
        raise ValueError("manifest candidate mapping authority mismatch")


def _accept_predictions(
    root_descriptor: int,
    snapshot: _TreeSnapshot,
    observations: Sequence[ObservationPackageRow],
    authority: CandidateMappingAuthority,
) -> Tuple[AcceptedPrediction, ...]:
    accepted: list[AcceptedPrediction] = []
    source_class_count = len(authority.source_vocabulary)
    for row in observations:
        data, record = _read_file_at(
            root_descriptor,
            row.prediction_path,
            expected_fingerprint=snapshot.files[row.prediction_path],
            expected_directories=snapshot.directories,
            expected_record=row.prediction,
        )
        parsed = parse_prediction_npz_bytes(
            data,
            expected_file=row.prediction,
            expected_members=row.prediction_members,
        )
        source = parsed.prediction.source_labels
        mapped = parsed.prediction.mapped_labels
        if row.status is ObservationStatus.PASS:
            if np.any(source < 0) or np.any(source >= source_class_count):
                raise ValueError("PASS prediction source labels are out of range")
            expected_mapped = (
                map_source_labels(
                    torch.from_numpy(source.copy()),
                    authority.mapping,
                )
                .cpu()
                .numpy()
                .astype("<i2", copy=False)
            )
            if not np.array_equal(mapped, expected_mapped):
                raise ValueError("mapped prediction differs from authoritative mapping")
        elif not (np.all(source == -1) and np.all(mapped == -1)):
            raise ValueError("FAILED prediction arrays must both be all -1")
        accepted.append(
            AcceptedPrediction(
                path=row.prediction_path,
                prediction=parsed.prediction,
                file=record,
                members=parsed.members,
            )
        )
    return tuple(accepted)


def _immutable_prediction(prediction: Prediction) -> Prediction:
    source = np.frombuffer(
        prediction.source_labels.tobytes(order="C"), dtype="<i2"
    ).reshape(_LABEL_SHAPE)
    mapped = np.frombuffer(
        prediction.mapped_labels.tobytes(order="C"), dtype="<i2"
    ).reshape(_LABEL_SHAPE)
    if source.flags.writeable or mapped.flags.writeable:
        raise RuntimeError("immutable prediction construction failed")
    return Prediction(source_labels=source, mapped_labels=mapped)


def _bind_observations(
    rows: Sequence[ObservationPackageRow], trusted_cohort: TrustedCohort
) -> None:
    if len(rows) != 50:
        raise ValueError("candidate package must contain 50 observation rows")
    for row, identity in zip(rows, trusted_cohort.identities):
        ordinal, observation_id, scene_id = identity
        if (
            row.ordinal,
            row.observation_id,
            row.scene_id,
        ) != (ordinal, observation_id, scene_id):
            raise ValueError("observation row differs from trusted cohort authority")


def _bind_timings(
    rows: Sequence[TimingPackageRow],
    observations: Sequence[ObservationPackageRow],
    predictions: Sequence[AcceptedPrediction],
) -> None:
    if len(rows) != 100:
        raise ValueError("candidate package must contain 100 timing rows")
    for timing in rows:
        sample = timing.sample
        observation = observations[sample.ordinal]
        prediction = predictions[sample.ordinal].prediction
        if (
            sample.status is not observation.status
            or sample.failure_code is not observation.failure_code
            or sample.source_labels_sha256
            != logical_label_sha256(prediction.source_labels)
            or sample.mapped_labels_sha256
            != logical_label_sha256(prediction.mapped_labels)
        ):
            raise ValueError("timing row differs from canonical observation output")


def _bind_manifest_payloads(
    manifest: SuccessfulManifest,
    observations_record: FileRecord,
    timings_record: FileRecord,
    prediction_records: Sequence[Tuple[str, FileRecord]],
) -> None:
    if _manifest_file_record(manifest, "observations") != observations_record:
        raise ValueError("manifest observations record mismatch")
    if _manifest_file_record(manifest, "timings") != timings_record:
        raise ValueError("manifest timings record mismatch")
    fields = _mapping(manifest.fields, "manifest")
    files = _mapping(fields["files"], "manifest.files")
    predictions = _mapping(files["predictions"], "manifest.files.predictions")
    aggregate = file_tree_aggregate(prediction_records)
    if (
        predictions["file_count"] != aggregate.file_count
        or predictions["total_byte_length"] != aggregate.total_byte_length
        or predictions["tree_sha256"] != aggregate.tree_sha256
    ):
        raise ValueError("manifest prediction tree aggregate mismatch")


def _package_validation_record(
    records: Sequence[Tuple[str, FileRecord]], manifest: FileRecord
) -> PackageValidationRecord:
    aggregate = file_tree_aggregate(records)
    return PackageValidationRecord(
        file_count=aggregate.file_count,
        total_byte_length=aggregate.total_byte_length,
        tree_sha256=aggregate.tree_sha256,
        manifest=manifest,
    )


def accept_candidate_package(
    root: Path,
    *,
    expected_manifest_sha256: str,
    trusted_cohort: TrustedCohort,
    mapping_authority: CandidateMappingAuthority,
) -> AcceptedCandidatePackage:
    """Accept every untrusted candidate payload without reading oracle targets."""

    _hash(expected_manifest_sha256, "external candidate manifest SHA-256")
    if not isinstance(mapping_authority, CandidateMappingAuthority):
        raise ValueError("candidate mapping authority is required")
    expected_directories, expected_files = _expected_tree(trusted_cohort)
    root_descriptor, descriptors, bindings = _open_anchored_root(root)
    primary: BaseException | None = None
    try:
        snapshot = _capture_tree(root_descriptor)
        _require_exact_tree(snapshot, expected_directories, expected_files)
        manifest_data, manifest_record = _read_file_at(
            root_descriptor,
            "manifest.json",
            expected_fingerprint=snapshot.files["manifest.json"],
            expected_directories=snapshot.directories,
            expected_record=FileRecord(
                snapshot.files["manifest.json"].size,
                expected_manifest_sha256,
            ),
        )
        manifest = parse_successful_manifest_bytes(manifest_data)
        _bind_manifest_authority(manifest, mapping_authority)
        observations_data, observations_record = _read_file_at(
            root_descriptor,
            "observations.jsonl",
            expected_fingerprint=snapshot.files["observations.jsonl"],
            expected_directories=snapshot.directories,
            expected_record=_manifest_file_record(manifest, "observations"),
        )
        timings_data, timings_record = _read_file_at(
            root_descriptor,
            "timings.jsonl",
            expected_fingerprint=snapshot.files["timings.jsonl"],
            expected_directories=snapshot.directories,
            expected_record=_manifest_file_record(manifest, "timings"),
        )
        observations = parse_observations_bytes(observations_data)
        _bind_observations(observations, trusted_cohort)
        timings = parse_timings_bytes(timings_data)
        accepted_predictions = _accept_predictions(
            root_descriptor,
            snapshot,
            observations,
            mapping_authority,
        )
        _bind_timings(timings, observations, accepted_predictions)
        prediction_records = tuple(
            (value.path, value.file) for value in accepted_predictions
        )
        _bind_manifest_payloads(
            manifest,
            observations_record,
            timings_record,
            prediction_records,
        )
        complete_records = (
            ("manifest.json", manifest_record),
            ("observations.jsonl", observations_record),
            ("timings.jsonl", timings_record),
        ) + prediction_records
        recaptured = _capture_tree(root_descriptor)
        if recaptured != snapshot:
            raise ValueError("candidate package tree changed during validation")
        for path, record in complete_records:
            _read_file_at(
                root_descriptor,
                path,
                expected_fingerprint=recaptured.files[path],
                expected_directories=recaptured.directories,
                expected_record=record,
            )
        if _capture_tree(root_descriptor) != recaptured:
            raise ValueError("candidate package tree changed during final recapture")
        _recapture_ancestor_bindings(bindings)
        return AcceptedCandidatePackage(
            manifest=manifest,
            observations=observations,
            timings=timings,
            predictions=accepted_predictions,
            validation=_package_validation_record(complete_records, manifest_record),
            _acceptance_token=_ACCEPTANCE_TOKEN,
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            _close_descriptors(tuple(reversed(descriptors)))
        except BaseException as cleanup:
            if primary is None:
                raise
            raise _DescriptorCleanupError(primary, cleanup) from primary


@dataclass(frozen=True)
class _ScientificRecomputation:
    candidate: CandidateCommitment
    run_kind: RunKind
    metrics: BenchmarkMetricSummary
    coverage: StaticCoverage
    robustness: RobustnessSummary
    latency: Optional[LatencySummary]
    provenance: Optional[ProvenanceChecks]
    gates: Optional[CandidateGateResult]

    def __post_init__(self) -> None:
        real = self.run_kind is RunKind.REAL
        if (
            not isinstance(self.candidate, CandidateCommitment)
            or not isinstance(self.run_kind, RunKind)
            or not isinstance(self.metrics, BenchmarkMetricSummary)
            or not isinstance(self.coverage, StaticCoverage)
            or not isinstance(self.robustness, RobustnessSummary)
            or (self.latency is not None) is not real
            or (self.provenance is not None) is not real
            or (self.gates is not None) is not real
        ):
            raise ValueError("scientific recomputation is incomplete")


def _endpoint_rows(
    observations: Sequence[ObservationPackageRow],
    metrics: Sequence[ObservationMetrics],
    *,
    endpoint: Callable[[ObservationMetrics], Optional[float]],
) -> Tuple[EndpointRow, ...]:
    return tuple(
        EndpointRow(
            ordinal=row.ordinal,
            observation_id=row.observation_id,
            scene_id=row.scene_id,
            endpoint=endpoint(metric),
            status=row.status,
            failure_code=row.failure_code,
        )
        for row, metric in zip(observations, metrics)
    )


def _recompute_accepted_candidate_science(
    accepted: AcceptedCandidatePackage,
    *,
    raw_observations: Sequence[ValidatedRawObservation],
    trusted_cohort: TrustedCohort,
    mapping_authority: CandidateMappingAuthority,
    provenance: Optional[ProvenanceChecks] = None,
    oracle_projector: Optional[Callable[[RawFrameArrays], np.ndarray]] = None,
    prediction_projector: Optional[
        Callable[[np.ndarray, RawFrameArrays], np.ndarray]
    ] = None,
) -> _ScientificRecomputation:
    """Recompute claims without minting an authority-bearing PASS record."""

    if not isinstance(accepted, AcceptedCandidatePackage):
        raise ValueError("scientific validation requires an accepted package")
    if not isinstance(trusted_cohort, TrustedCohort):
        raise ValueError("scientific validation requires a trusted cohort")
    if not isinstance(mapping_authority, CandidateMappingAuthority):
        raise ValueError("scientific validation requires mapping authority")
    sealed_oracle_projector = (
        project_oracle_target_labels if oracle_projector is None else oracle_projector
    )
    sealed_prediction_projector = (
        project_mapped_labels if prediction_projector is None else prediction_projector
    )
    raw = tuple(raw_observations)
    if len(raw) != 50 or any(
        not isinstance(value, ValidatedRawObservation)
        or not isinstance(value.row, IndexRow)
        or not isinstance(value.audit_arrays, RawFrameArrays)
        for value in raw
    ):
        raise ValueError("scientific validation requires 50 typed raw observations")
    raw_identities = tuple(
        (value.row.ordinal, value.row.observation_id, value.row.scene_id)
        for value in raw
    )
    accepted_identities = tuple(
        (value.ordinal, value.observation_id, value.scene_id)
        for value in accepted.observations
    )
    if raw_identities != trusted_cohort.identities:
        raise ValueError("raw observations differ from trusted cohort authority")
    if accepted_identities != trusted_cohort.identities:
        raise ValueError("accepted observations differ from trusted cohort authority")

    manifest_summary = accepted.manifest.summary
    candidate = manifest_summary.candidate
    if (
        candidate.source_vocabulary != mapping_authority.source_vocabulary
        or candidate.mapping_sha256 != mapping_authority.mapping_sha256
    ):
        raise ValueError("candidate differs from scientific mapping authority")

    recomputed_rows: list[ObservationMetrics] = []
    targets: list[np.ndarray] = []
    for raw_value, row, accepted_prediction in zip(
        raw, accepted.observations, accepted.predictions
    ):
        target = sealed_oracle_projector(raw_value.audit_arrays)
        projected = sealed_prediction_projector(
            accepted_prediction.prediction.mapped_labels,
            raw_value.audit_arrays,
        )
        metrics = score_observation(
            ordinal=row.ordinal,
            scene_id=row.scene_id,
            prediction=projected,
            target=target,
        )
        if metrics != row.metrics:
            raise ValueError(
                f"observation {row.ordinal} metrics differ from recomputation"
            )
        targets.append(target)
        recomputed_rows.append(metrics)

    metric_rows = tuple(recomputed_rows)
    metric_summary = aggregate_observation_metrics(
        metric_rows,
        expected_scenes=trusted_cohort.scenes,
    )
    if metric_summary != manifest_summary.metrics:
        raise ValueError("manifest metric summary differs from recomputation")
    coverage = compute_static_coverage(mapping_authority.mapping, targets)
    if coverage != manifest_summary.coverage:
        raise ValueError("manifest static coverage differs from recomputation")

    iou_rows = _endpoint_rows(
        accepted.observations,
        metric_rows,
        endpoint=lambda value: value.primary.iou,
    )
    f1_rows = _endpoint_rows(
        accepted.observations,
        metric_rows,
        endpoint=lambda value: value.primary.f1,
    )
    robustness = RobustnessSummary(
        iou=estimate_scene_robustness(iou_rows, trusted_cohort=trusted_cohort),
        f1=estimate_scene_robustness(f1_rows, trusted_cohort=trusted_cohort),
    )
    if robustness != manifest_summary.robustness:
        raise ValueError("manifest robustness differs from recomputation")

    latency: Optional[LatencySummary] = None
    gates: Optional[CandidateGateResult] = None
    if manifest_summary.run_kind is RunKind.REAL:
        if provenance is None or manifest_summary.real is None:
            raise ValueError("real scientific validation requires provenance")
        latency = summarize_latency(tuple(row.sample for row in accepted.timings))
        if latency != manifest_summary.real.latency:
            raise ValueError("manifest latency differs from recomputation")
        mean_iou = metric_summary.primary.mean_iou
        mean_f1 = metric_summary.primary.mean_f1
        if mean_iou is None or mean_f1 is None:
            raise ValueError("real candidate has no eligible quality endpoints")
        gates = evaluate_candidate_gates(
            synthetic=False,
            coverage=coverage,
            trusted_cohort=trusted_cohort,
            rows=iou_rows,
            mean_iou=mean_iou,
            mean_f1=mean_f1,
            latency=latency,
            resource=manifest_summary.real.resource,
            code_license_status=candidate.code_license_status,
            weight_license_status=candidate.weight_license_status,
            provenance=provenance,
        )
        if gates != manifest_summary.real.gates:
            raise ValueError("manifest gates differ from recomputation")
    elif provenance is not None:
        raise ValueError("synthetic scientific validation forbids provenance claims")

    return _ScientificRecomputation(
        candidate=candidate,
        run_kind=manifest_summary.run_kind,
        metrics=metric_summary,
        coverage=coverage,
        robustness=robustness,
        latency=latency,
        provenance=provenance,
        gates=gates,
    )


def _require_git_state(
    actual: GitRepositoryState,
    *,
    expected_commit: str,
    expected_origin_url: Optional[str] = None,
    label: str,
) -> None:
    if (
        not actual.clean
        or actual.commit != expected_commit
        or (
            expected_origin_url is not None and actual.origin_url != expected_origin_url
        )
    ):
        raise ValueError(f"{label} Git state differs from authority")


def _require_runtime_source_bindings(
    authority: CandidateValidationAuthority,
) -> None:
    runtime_sources = {
        "constants": (_constants_module.__file__, _FIXED_SOURCE_PATHS["constants"]),
        "contract": _benchmark_contract_module.__file__,
        "package": __file__,
        "projector": _projector_module.__file__,
        "raw_package": (_raw_package_module.__file__, _RAW_PACKAGE_SOURCE_PATH),
    }
    for role, value in runtime_sources.items():
        if isinstance(value, tuple):
            module_path, relative_path = value
        else:
            module_path = value
            relative_path = _FIXED_SOURCE_PATHS[role]
        if not isinstance(module_path, str):
            raise ValueError(f"runtime {role} module has no source path")
        expected = authority.benchmark_repository_root / relative_path
        try:
            actual = Path(module_path).resolve(strict=True)
            expected_resolved = expected.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"runtime {role} source cannot be resolved") from error
        if expected_resolved != expected or actual != expected:
            raise ValueError(
                f"runtime {role} source is outside benchmark authority root"
            )
    raw_source = _trusted_file_record(
        authority.benchmark_repository_root,
        _RAW_PACKAGE_SOURCE_PATH,
    )
    if raw_source.sha256 != RAW_VALIDATOR_SOURCE_SHA256:
        raise ValueError("runtime raw package source differs from frozen validator")


def _require_manifest_authority(
    accepted: AcceptedCandidatePackage,
    *,
    authority: CandidateValidationAuthority,
    p53_attestation: P53ValidationAttestation,
) -> None:
    summary = accepted.manifest.summary
    fields = _mapping(accepted.manifest.fields, "manifest")
    if (
        summary.p53_attestation != p53_attestation
        or summary.benchmark_attestation != authority.benchmark_attestation
        or summary.candidate != authority.candidate
        or fields["candidate_id"] != authority.candidate.candidate_id
        or fields["producer_git_commit"] != authority.expected_producer_commit
        or fields["command"] != authority.expected_command
    ):
        raise ValueError("candidate manifest differs from validation authority")
    raw_inputs = _mapping(fields["raw_inputs"], "manifest.raw_inputs")
    if raw_inputs != {
        "cohort_jsonl_sha256": _COHORT_JSONL_SHA256,
        "raw_index_sha256": RAW_INDEX_SHA256,
        "raw_manifest_sha256": RAW_MANIFEST_SHA256,
        "raw_payload_tree_sha256": _RAW_PAYLOAD_TREE_SHA256,
        "raw_producer_git_commit": RAW_PRODUCER_COMMIT,
        "selection_sha256": _SELECTION_SHA256,
    }:
        raise ValueError("candidate manifest raw inputs differ from frozen authority")


def _capture_source_records(
    manifest: SuccessfulManifest,
    *,
    authority: CandidateValidationAuthority,
) -> Mapping[str, FileRecord]:
    raw = _mapping(manifest.fields["source_hashes"], "manifest.source_hashes")
    captured: dict[str, FileRecord] = {}
    for role in _SOURCE_HASH_KEYS:
        value = _mapping(raw[role], f"manifest.source_hashes.{role}")
        path = cast(str, value["path"])
        if role == "adapter" and path != authority.adapter_path:
            raise ValueError("candidate adapter path differs from authority")
        record = _trusted_file_record(authority.benchmark_repository_root, path)
        if record.sha256 != value["sha256"]:
            raise ValueError(f"candidate source hash differs from bytes: {role}")
        captured[role] = record
    mapping_path = cast(
        str,
        _mapping(raw["mapping"], "manifest.source_hashes.mapping")["path"],
    )
    mapping_record = captured["mapping"]
    if (
        mapping_record.sha256 != NYU40_MAPPING_SHA256
        or authority.mapping.mapping_sha256 != NYU40_MAPPING_SHA256
    ):
        raise ValueError("mapping source differs from frozen NYU40 authority")
    loaded_mapping = load_nyu40_mapping(
        authority.benchmark_repository_root / mapping_path
    )
    if (
        loaded_mapping != authority.mapping.mapping
        or tuple(entry.source_name for entry in loaded_mapping)
        != authority.mapping.source_vocabulary
    ):
        raise ValueError("mapping source contents differ from mapping authority")
    return _FrozenMapping(tuple(sorted(captured.items())))


def _authority_root(
    authority: CandidateValidationAuthority,
    root_role: str,
) -> Path:
    if root_role == "benchmark_repository":
        return authority.benchmark_repository_root
    if root_role == "candidate_repository" and authority.real is not None:
        return authority.real.candidate_repository_root
    raise ValueError("trusted artifact root role is unavailable")


def _require_permission_record(
    raw: Mapping[str, object],
    *,
    expected: TrustedArtifactAuthority,
    authority: CandidateValidationAuthority,
    label: str,
) -> FileRecord:
    if raw != {
        "byte_length": expected.file.byte_length,
        "path": expected.path,
        "root_role": expected.root_role,
        "sha256": expected.file.sha256,
    }:
        raise ValueError(f"{label} permission evidence differs from authority")
    actual = _trusted_file_record(
        _authority_root(authority, expected.root_role),
        expected.path,
    )
    if actual != expected.file:
        raise ValueError(f"{label} permission evidence differs from bytes")
    return actual


@dataclass(frozen=True)
class _RealProvenanceCapture:
    checkpoint: FileRecord
    code_permission: FileRecord
    weight_permission: FileRecord


def _capture_real_provenance(
    accepted: AcceptedCandidatePackage,
    *,
    authority: CandidateValidationAuthority,
) -> _RealProvenanceCapture:
    real = authority.real
    summary = accepted.manifest.summary
    if real is None or summary.real is None:
        raise ValueError("real provenance authority is unavailable")
    candidate = authority.candidate
    checkpoint = _trusted_file_record(real.checkpoint_root, real.checkpoint_path)
    if (
        checkpoint != real.checkpoint_file
        or checkpoint.sha256 != candidate.checkpoint_sha256
    ):
        raise ValueError("candidate checkpoint differs from authority")
    permissions = _mapping(
        accepted.manifest.fields["permission_evidence"],
        "manifest.permission_evidence",
    )
    code_permission = _require_permission_record(
        _mapping(permissions["code"], "manifest.permission_evidence.code"),
        expected=real.code_permission,
        authority=authority,
        label="code",
    )
    weight_permission = _require_permission_record(
        _mapping(permissions["weights"], "manifest.permission_evidence.weights"),
        expected=real.weight_permission,
        authority=authority,
        label="weight",
    )
    benchmark = authority.benchmark_attestation
    cuda = summary.real.cuda_evidence
    if any(
        (snapshot.gpu_name, snapshot.gpu_uuid)
        != (benchmark.gpu_name, benchmark.gpu_uuid)
        for snapshot in (cuda.before, cuda.after)
    ):
        raise ValueError("archival CUDA evidence differs from benchmark authority")
    return _RealProvenanceCapture(
        checkpoint=checkpoint,
        code_permission=code_permission,
        weight_permission=weight_permission,
    )


def _capture_environment_lock(
    authority: CandidateValidationAuthority,
) -> FileRecord:
    expected = authority.environment_lock
    actual = _trusted_file_record(
        _authority_root(authority, expected.root_role),
        expected.path,
    )
    if (
        actual != expected.file
        or expected.path != authority.candidate.environment_lock_path
        or actual.sha256 != authority.candidate.environment_lock_sha256
    ):
        raise ValueError("candidate environment lock differs from authority")
    return actual


def _passed_provenance_checks() -> ProvenanceChecks:
    return ProvenanceChecks(
        clean_experiment_commit=True,
        complete_command=True,
        raw_package_pinned=True,
        cohort_pinned=True,
        candidate_revision_pinned=True,
        checkpoint_hash_pinned=True,
        environment_pinned=True,
        mapping_pinned=True,
        projector_pinned=True,
        preprocessing_pinned=True,
        precision_pinned=True,
        permission_evidence_pinned=True,
    )


def validate_candidate_package(
    root: Path,
    *,
    authority: CandidateValidationAuthority,
) -> ValidatedCandidatePackage:
    """Validate one package and mint an endpoint-free, factory-only PASS record."""

    if not isinstance(authority, CandidateValidationAuthority):
        raise ValueError("candidate validation authority is required")
    _require_runtime_source_bindings(authority)
    benchmark_before = _inspect_git_repository(authority.benchmark_repository_root)
    _require_git_state(
        benchmark_before,
        expected_commit=authority.expected_producer_commit,
        label="benchmark repository",
    )
    candidate_before: Optional[GitRepositoryState] = None
    if authority.real is not None:
        candidate_before = _inspect_git_repository(
            authority.real.candidate_repository_root
        )
        _require_git_state(
            candidate_before,
            expected_commit=authority.candidate.revision,
            expected_origin_url=authority.candidate.repository_url,
            label="candidate repository",
        )

    p53_attestation = run_p53_validation_subprocess(authority.p53_launch)
    trusted_cohort = _trusted_cohort_from_raw_index(
        authority.p53_launch.raw_root,
        p53_attestation=p53_attestation,
    )
    accepted = accept_candidate_package(
        root,
        expected_manifest_sha256=authority.expected_manifest_sha256,
        trusted_cohort=trusted_cohort,
        mapping_authority=authority.mapping,
    )
    _require_manifest_authority(
        accepted,
        authority=authority,
        p53_attestation=p53_attestation,
    )
    source_before = _capture_source_records(
        accepted.manifest,
        authority=authority,
    )
    environment_before = _capture_environment_lock(authority)

    provenance: Optional[ProvenanceChecks] = None
    real_before: Optional[_RealProvenanceCapture] = None
    if authority.real is not None:
        real_before = _capture_real_provenance(accepted, authority=authority)
        provenance = _passed_provenance_checks()
    raw_observations = iter_validated_raw_observations(
        authority.p53_launch.raw_root,
        p53_attestation=p53_attestation,
        expected_p53_attestation_sha256=p53_attestation.sha256,
        benchmark_attestation=authority.benchmark_attestation,
        expected_benchmark_attestation_sha256=(
            authority.expected_benchmark_attestation_sha256
        ),
    )
    _recompute_accepted_candidate_science(
        accepted,
        raw_observations=raw_observations,
        trusted_cohort=trusted_cohort,
        mapping_authority=authority.mapping,
        provenance=provenance,
    )

    final_p53_attestation = run_p53_validation_subprocess(authority.p53_launch)
    if final_p53_attestation != p53_attestation:
        raise ValueError("P5.3 attestation changed during validation")
    if (
        _trusted_cohort_from_raw_index(
            authority.p53_launch.raw_root,
            p53_attestation=final_p53_attestation,
        )
        != trusted_cohort
    ):
        raise ValueError("raw cohort changed during validation")
    final_accepted = accept_candidate_package(
        root,
        expected_manifest_sha256=authority.expected_manifest_sha256,
        trusted_cohort=trusted_cohort,
        mapping_authority=authority.mapping,
    )
    if final_accepted.validation != accepted.validation:
        raise ValueError("candidate package changed during validation")
    if _capture_source_records(accepted.manifest, authority=authority) != source_before:
        raise ValueError("candidate sources changed during validation")
    if _capture_environment_lock(authority) != environment_before:
        raise ValueError("candidate environment changed during validation")
    if authority.real is not None:
        if _capture_real_provenance(accepted, authority=authority) != real_before:
            raise ValueError("candidate provenance changed during validation")
        candidate_after = _inspect_git_repository(
            authority.real.candidate_repository_root
        )
        _require_git_state(
            candidate_after,
            expected_commit=authority.candidate.revision,
            expected_origin_url=authority.candidate.repository_url,
            label="candidate repository",
        )
        if candidate_after != candidate_before:
            raise ValueError("candidate Git state changed during validation")
    benchmark_after = _inspect_git_repository(authority.benchmark_repository_root)
    _require_git_state(
        benchmark_after,
        expected_commit=authority.expected_producer_commit,
        label="benchmark repository",
    )
    if benchmark_after != benchmark_before:
        raise ValueError("benchmark Git state changed during validation")
    authority.benchmark_attestation.require_current_process()

    summary = accepted.manifest.summary
    record = CandidateValidationRecord(
        schema_version=1,
        validation_status="PASS",
        candidate_status=(
            "NOT_APPLICABLE"
            if summary.real is None
            else summary.real.gates.overall.value
        ),
        run_kind=summary.run_kind,
        candidate_id=summary.candidate.candidate_id,
        producer_git_commit=authority.expected_producer_commit,
        p53_attestation_sha256=p53_attestation.sha256,
        benchmark_attestation_sha256=(authority.expected_benchmark_attestation_sha256),
        synthetic_semantic_scores_exposed=False,
        observation_count=50,
        package=accepted.validation,
        _validation_token=_VALIDATION_TOKEN,
    )
    return ValidatedCandidatePackage(
        validation=accepted.validation,
        record=record,
        _validation_token=_VALIDATION_TOKEN,
    )
