"""Pinned source binding and private scene snapshots for the RGB-D experiment."""

from __future__ import annotations

import hashlib
import csv
import ctypes
import errno
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import groupby
from multiprocessing.connection import Connection
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import (
    Callable,
    List,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import torch
from tap import Tap

from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    FileRecord,
    IndexRow,
    RawFrameArrays,
    SourceRecord,
    _MAX_ARTIFACT_BYTES,
    _fingerprint,
    _close_many,
    _require_closed,
    _open_package_root,
    _parse_manifest_bytes,
    _PackageSnapshot,
    _EntryFingerprint,
    _read_manifest_at,
    _strict_read_at,
    _recapture_package_descriptor,
    _require_root_path_binding,
    _validate_raw_frame_descriptor_structural,
    canonical_index_bytes,
    encode_raw_frame_npz,
    parse_raw_frame_npz_bytes,
    strict_read_bytes,
    tree_aggregate,
)
from prior.constants import (
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    OBJECT_MAPPING,
    REGION_MAPPING,
)
from vlnce_baselines.models.etp_llm.llm_grid_oracle_cache import (
    OracleSensorFrame,
    project_oracle_frames,
)

COHORT_ROOT = Path("data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1")
ORACLE_ARTIFACT_ROOT = Path("data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen")
RAW_SPLIT_PATH = Path(
    "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz"
)
SCENE_DATASET_ROOT = Path("data/scene_datasets/mp3d")
FINAL_PACKAGE_ROOT = Path("data/rgbd_segmenter_benchmark/r2r-val-unseen-50-raw-v1")
STAGING_PACKAGE_ROOT = Path(
    "data/rgbd_segmenter_benchmark/.r2r-val-unseen-50-raw-v1.staging"
)
SNAPSHOT_PACKAGE_ROOT = Path(
    "data/rgbd_segmenter_benchmark/.r2r-val-unseen-50-raw-v1.scene-assets"
)
SMOKE_PACKAGE_ROOT = Path(
    "data/rgbd_segmenter_benchmark/.r2r-val-unseen-50-raw-v1-smoke"
)
_COHORT_MANIFEST_SHA256 = (
    "d71f04f102d80df3799e5fea76162147ad76c060c82ccbca88813d8b14a0b191"
)
_COHORT_SHA256 = "89ae70f3e489fa702c66110f9bd9e7666ba0e16a9fbc3ac20aaa91a95adef0ce"
_SELECTION_SHA256 = "32a7adddf32291f059eb1045e63e077a693ca64d6fdf6e7418b54dd5d3644cb2"
_EVIDENCE_MANIFEST_SHA256 = (
    "be2d25890c01234dd1bb8191af6c1d87121330991299287b635205f2ebf0327a"
)
_EVIDENCE_INDEX_SHA256 = (
    "0acc9d18aea6d369db4b0a73f8ab3eebbe8fab51257b34614d1a7c407443acb9"
)
_RAW_SPLIT_SHA256 = "6140b46759fe332ee96aa849d4bb64e1c1829b8f65acee127355040a6ee23484"
_SEALING_COMMIT = "80780e5a8a3736dc666b8dd861c00ce55f4685d8"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OBSERVATION_ID = re.compile(r"[0-9a-f]{20}")
_SCENE_ID = re.compile(r"[A-Za-z0-9_-]+")
_EXAMPLE_ID = re.compile(r"R2R_val_unseen_(?:0|[1-9][0-9]*)")
_ORACLE_NAMES = (
    "schema_version",
    "grid_scale",
    "cell_size_m",
    "ego_semantic_grid",
    "ego_observed_mask",
    "ego_free_mask",
    "target_semantic_grid",
    "target_observed_mask",
    "target_free_mask",
    "start_position",
    "start_direction",
)


@dataclass(frozen=True)
class CollectionObservation:
    """Pre-render metadata coupled from the sealed cohort and evidence index."""

    artifact_path: Path
    artifact_sha256: str
    cohort_row_sha256: str
    example_ids: Tuple[str, ...]
    observation_id: str
    scene_id: str
    start_position: Tuple[float, float, float]
    start_rotation: Tuple[float, float, float, float]


@dataclass(frozen=True)
class CollectionInputs:
    """The complete fixed observation cohort, without loaded oracle arrays."""

    observations: Tuple[CollectionObservation, ...]
    scenes: Tuple[str, ...]


@dataclass(frozen=True)
class PinnedOracle:
    """The strict 11-member oracle artifact parsed from its accepted bytes."""

    sha256: str
    ego_semantic_grid: np.ndarray
    ego_observed_mask: np.ndarray
    ego_free_mask: np.ndarray
    target_semantic_grid: np.ndarray
    target_observed_mask: np.ndarray
    target_free_mask: np.ndarray
    start_position: Tuple[float, float]
    start_direction: Tuple[float, float]


@dataclass(frozen=True)
class SnapshotFile:
    """One immutable copied asset, identified only by its private snapshot path."""

    path: Path
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class SceneBundle:
    """The simulator-facing, process-private scene asset commitment."""

    scene_id: str
    files: Mapping[str, SnapshotFile]


@dataclass(frozen=True, order=True)
class InstalledDistribution:
    name: str
    version: str


@dataclass(frozen=True)
class EnvironmentCapture:
    python_version: str
    python_implementation: str
    platform: str
    numpy_version: str
    zlib_version: str
    habitat_version: str
    habitat_sim_version: str
    cuda_runtime_version: str
    nvidia_driver_version: str
    gpu_name: str
    gpu_uuid: str
    gpu_device_id: int
    installed_distributions: Tuple[InstalledDistribution, ...]


@dataclass(frozen=True)
class ManifestSources:
    collector_source: SourceRecord
    package_source: SourceRecord
    asset_roles: SourceRecord
    cohort_manifest: SourceRecord
    cohort_jsonl: SourceRecord
    evidence_manifest: SourceRecord
    evidence_index: SourceRecord
    raw_split: SourceRecord
    projector_source: SourceRecord
    mapping_source: SourceRecord


@dataclass(frozen=True)
class _SourceCaptureState:
    sources: ManifestSources
    fingerprints: Mapping[str, Tuple[int, ...]]


@dataclass
class _OwnedDirectory:
    path: Path
    name: str
    parent_descriptor: int
    descriptor: int
    fingerprint: _EntryFingerprint


@dataclass(frozen=True)
class _CollectedAttempt:
    manifest: bytes
    structural_snapshot: _PackageSnapshot


@dataclass(frozen=True)
class _SceneSnapshotState:
    directory: _EntryFingerprint
    files: Mapping[str, _EntryFingerprint]


class _OperationCleanupError(RuntimeError):
    def __init__(
        self,
        label: str,
        primary: BaseException,
        failures: Sequence[BaseException],
    ) -> None:
        self.primary = primary
        self.failures = tuple(failures)
        super().__init__(
            f"{label}: " + "; ".join(str(error) for error in self.failures)
        )


def _same_directory_identity(
    fingerprint: _EntryFingerprint,
    metadata: os.stat_result,
) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_dev == fingerprint.st_dev
        and metadata.st_ino == fingerprint.st_ino
    )


@dataclass(frozen=True)
class SceneAssetFile:
    path: str
    required: bool
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or path.as_posix() != self.path
        ):
            raise ValueError("scene asset path is not normalized relative POSIX")
        if (
            type(self.required) is not bool
            or type(self.byte_length) is not int
            or self.byte_length < 0
            or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValueError("scene asset metadata is invalid")


@dataclass(frozen=True)
class SceneAssetCommitment:
    bundle_sha256: str
    files: Mapping[str, SceneAssetFile]

    @classmethod
    def from_files(
        cls, files: Mapping[str, SceneAssetFile]
    ) -> "SceneAssetCommitment":
        if set(files) != {"glb", "house", "navmesh", "semantic_ply"}:
            raise ValueError("scene asset commitment requires exactly four roles")
        expected_required = {
            "glb": True,
            "house": True,
            "navmesh": False,
            "semantic_ply": True,
        }
        if any(
            files[role].required is not required
            for role, required in expected_required.items()
        ):
            raise ValueError("scene asset required roles differ from classification")
        serialized = {
            role: asdict(files[role])
            for role in sorted(files)
        }
        return cls(
            bundle_sha256=hashlib.sha256(
                json.dumps(
                    serialized, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            files=MappingProxyType(dict(files)),
        )


def _json_object(data: bytes, label: str) -> Mapping[str, object]:
    def no_duplicates(pairs: Sequence[Tuple[str, object]]) -> Mapping[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} has duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has invalid fields")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_scene_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _SCENE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _require_observation_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _OBSERVATION_ID.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _finite_vector(value: object, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} has invalid shape")
    if any(type(item) is not float or not math.isfinite(item) for item in value):
        raise ValueError(f"{label} must contain finite JSON floats")
    return tuple(cast(float, item) for item in value)


def _parse_cohort(
    manifest_bytes: bytes, cohort_bytes: bytes
) -> tuple[
    Mapping[str, object],
    tuple[Mapping[str, object], ...],
    tuple[str, ...],
]:
    manifest = _json_object(manifest_bytes, "cohort manifest")
    _require_keys(
        manifest,
        {
            "cohort_id",
            "files",
            "git_commit",
            "population",
            "schema_version",
            "selection",
            "source",
        },
        "cohort manifest",
    )
    if (
        manifest["cohort_id"] != "r2r-val-unseen-50-v1"
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["git_commit"] != _SEALING_COMMIT
        or not isinstance(manifest["files"], dict)
        or not isinstance(manifest["selection"], dict)
    ):
        raise ValueError("cohort manifest contract drift")
    files = cast(Mapping[str, object], manifest["files"])
    selection = cast(Mapping[str, object], manifest["selection"])
    if (
        set(files) != {"cohort.jsonl"}
        or not isinstance(files["cohort.jsonl"], dict)
        or cast(Mapping[str, object], files["cohort.jsonl"]).get("sha256")
        != _COHORT_SHA256
        or selection.get("selection_sha256") != _SELECTION_SHA256
        or type(selection.get("selected_observation_count")) is not int
        or selection.get("selected_observation_count") != 50
        or type(selection.get("selected_scene_count")) is not int
        or selection.get("selected_scene_count") != 11
    ):
        raise ValueError("cohort manifest commitments drift")
    if not cohort_bytes.endswith(b"\n"):
        raise ValueError("cohort JSONL must end with newline")
    rows = []
    row_hashes = []
    for number, line in enumerate(cohort_bytes.splitlines(), start=1):
        row = _json_object(line, f"cohort row {number}")
        _require_keys(
            row,
            {
                "artifact_sha256",
                "example_ids",
                "observation_id",
                "scene_id",
                "start_position",
                "start_rotation",
            },
            f"cohort row {number}",
        )
        if json.dumps(row, sort_keys=True, separators=(",", ":")).encode() != line:
            raise ValueError(f"cohort row {number} is not canonical")
        _require_sha256(row["artifact_sha256"], f"cohort row {number} artifact SHA-256")
        _require_observation_id(
            row["observation_id"], f"cohort row {number} observation ID"
        )
        _require_scene_id(row["scene_id"], f"cohort row {number} scene ID")
        aliases = row["example_ids"]
        if (
            not isinstance(aliases, list)
            or not aliases
            or any(
                not isinstance(alias, str) or _EXAMPLE_ID.fullmatch(alias) is None
                for alias in aliases
            )
            or aliases != sorted(set(aliases))
        ):
            raise ValueError(f"cohort row {number} aliases are invalid")
        _finite_vector(row["start_position"], 3, f"cohort row {number} start_position")
        rotation = _finite_vector(
            row["start_rotation"], 4, f"cohort row {number} start_rotation"
        )
        if not math.isclose(
            math.sqrt(sum(item * item for item in rotation)), 1.0, abs_tol=1e-12
        ):
            raise ValueError(f"cohort row {number} start_rotation is not unit")
        rows.append(row)
        row_hashes.append(hashlib.sha256(line).hexdigest())
    if len(rows) != 50 or len({cast(str, row["observation_id"]) for row in rows}) != 50:
        raise ValueError("cohort observation population drift")
    scenes = sorted({cast(str, row["scene_id"]) for row in rows})
    if len(scenes) != 11:
        raise ValueError("cohort scene population drift")
    return manifest, tuple(rows), tuple(row_hashes)


def _parse_evidence_metadata(
    manifest_bytes: bytes, index_bytes: bytes
) -> Mapping[str, Mapping[str, object]]:
    manifest = _json_object(manifest_bytes, "evidence manifest")
    required = {
        "schema_version",
        "evidence_key",
        "dataset",
        "split",
        "semantic_shape",
        "mask_shape",
        "grid_scale",
        "cell_size_m",
        "ego_frame",
        "target_frame",
        "example_count",
        "observation_count",
        "index_sha256",
        "provenance",
    }
    _require_keys(manifest, required, "evidence manifest")
    expected = {
        "schema_version": 1,
        "evidence_key": "oracle-t0-v1",
        "dataset": "R2R",
        "split": "val_unseen",
        "semantic_shape": [37, 50, 50],
        "mask_shape": [50, 50],
        "grid_scale": 2,
        "cell_size_m": 1.0,
        "ego_frame": "start-centered heading-normalized",
        "target_frame": "level-local world-aligned",
        "index_sha256": _EVIDENCE_INDEX_SHA256,
    }
    if any(
        type(manifest[key]) is not type(value) or manifest[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("evidence manifest contract drift")
    if not index_bytes.endswith(b"\n"):
        raise ValueError("evidence index must end with newline")
    records: dict[str, Mapping[str, object]] = {}
    for number, line in enumerate(index_bytes.splitlines(), start=1):
        row = _json_object(line, f"evidence index row {number}")
        _require_keys(
            row,
            {"artifact_sha256", "example_id", "observation_id", "scene_id"},
            f"evidence index row {number}",
        )
        if json.dumps(row, sort_keys=True, separators=(",", ":")).encode() != line:
            raise ValueError(f"evidence index row {number} is not canonical")
        alias = row["example_id"]
        if not isinstance(alias, str) or _EXAMPLE_ID.fullmatch(alias) is None:
            raise ValueError(f"evidence index row {number} alias is invalid")
        _require_sha256(
            row["artifact_sha256"], f"evidence index row {number} artifact SHA-256"
        )
        _require_observation_id(
            row["observation_id"], f"evidence index row {number} observation ID"
        )
        _require_scene_id(row["scene_id"], f"evidence index row {number} scene ID")
        if alias in records:
            raise ValueError("evidence index aliases are not unique")
        records[alias] = row
    if list(records) != sorted(records) or manifest["example_count"] != len(records):
        raise ValueError("evidence index population drift")
    if manifest["observation_count"] != len({
        row["observation_id"] for row in records.values()
    }):
        raise ValueError("evidence index observations drift")
    return MappingProxyType(records)


def load_collection_inputs() -> CollectionInputs:
    """Bind the sealed metadata; oracle arrays remain unopened until rendering ends."""

    cohort_manifest = strict_read_bytes(
        COHORT_ROOT / "manifest.json", _COHORT_MANIFEST_SHA256, "cohort manifest"
    )
    cohort_rows = strict_read_bytes(
        COHORT_ROOT / "cohort.jsonl", _COHORT_SHA256, "cohort JSONL"
    )
    evidence_manifest = strict_read_bytes(
        ORACLE_ARTIFACT_ROOT / "manifest.json",
        _EVIDENCE_MANIFEST_SHA256,
        "evidence manifest",
    )
    evidence_index = strict_read_bytes(
        ORACLE_ARTIFACT_ROOT / "index.jsonl", _EVIDENCE_INDEX_SHA256, "evidence index"
    )
    strict_read_bytes(RAW_SPLIT_PATH, _RAW_SPLIT_SHA256, "raw split")
    _, cohort, cohort_row_hashes = _parse_cohort(cohort_manifest, cohort_rows)
    evidence = _parse_evidence_metadata(evidence_manifest, evidence_index)
    observations = []
    for row, cohort_row_sha256 in zip(cohort, cohort_row_hashes):
        aliases = tuple(cast(List[str], row["example_ids"]))
        records = tuple(evidence[alias] for alias in aliases)
        observation_id = cast(str, row["observation_id"])
        scene_id = cast(str, row["scene_id"])
        artifact_sha256 = cast(str, row["artifact_sha256"])
        if any(
            record["observation_id"] != observation_id
            or record["scene_id"] != scene_id
            or record["artifact_sha256"] != artifact_sha256
            for record in records
        ):
            raise ValueError("cohort/evidence binding drift")
        observations.append(
            CollectionObservation(
                artifact_path=Path("observations") / scene_id / f"{observation_id}.npz",
                artifact_sha256=artifact_sha256,
                cohort_row_sha256=cohort_row_sha256,
                example_ids=aliases,
                observation_id=observation_id,
                scene_id=scene_id,
                start_position=cast(
                    Tuple[float, float, float],
                    _finite_vector(row["start_position"], 3, "start_position"),
                ),
                start_rotation=cast(
                    Tuple[float, float, float, float],
                    _finite_vector(row["start_rotation"], 4, "start_rotation"),
                ),
            )
        )
    scenes = tuple(sorted({item.scene_id for item in observations}))
    return CollectionInputs(observations=tuple(observations), scenes=scenes)


def _oracle_array(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    if value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"{label} has invalid dtype or shape")
    value.setflags(write=False)
    return value


def _oracle_scalar(
    value: np.ndarray, dtype: np.dtype, expected: float, label: str
) -> None:
    if value.dtype != dtype or value.shape != () or value.item() != expected:
        raise ValueError(f"{label} has invalid value")


def _oracle_vector(value: np.ndarray, label: str, *, unit: bool) -> Tuple[float, float]:
    if value.dtype != np.float32 or value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError(f"{label} has invalid dtype, shape, or values")
    result = float(value[0]), float(value[1])
    if unit and not math.isclose(math.hypot(*result), 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"{label} must be unit")
    return result


def load_pinned_oracle_after_render(observation: CollectionObservation) -> PinnedOracle:
    """Open one pinned oracle once, then parse that exact accepted byte buffer."""

    if not isinstance(observation, CollectionObservation):
        raise ValueError("oracle load requires a collection observation")
    _require_scene_id(observation.scene_id, "oracle scene ID")
    _require_observation_id(observation.observation_id, "oracle observation ID")
    relative = PurePosixPath(observation.artifact_path.as_posix())
    expected_relative = PurePosixPath(
        "observations", observation.scene_id, f"{observation.observation_id}.npz"
    )
    if relative != expected_relative:
        raise ValueError("oracle artifact path escapes the evidence root")
    data = strict_read_bytes(
        ORACLE_ARTIFACT_ROOT / Path(relative),
        observation.artifact_sha256,
        "oracle artifact",
    )
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.namelist() != [f"{name}.npy" for name in _ORACLE_NAMES]:
                raise ValueError("oracle artifact has invalid ordered members")
        with np.load(io.BytesIO(data), allow_pickle=False) as artifact:
            if tuple(artifact.files) != _ORACLE_NAMES:
                raise ValueError("oracle artifact has invalid members")
            _oracle_scalar(
                artifact["schema_version"], np.dtype(np.int64), 1, "schema_version"
            )
            _oracle_scalar(artifact["grid_scale"], np.dtype(np.int64), 2, "grid_scale")
            _oracle_scalar(
                artifact["cell_size_m"], np.dtype(np.float32), 1.0, "cell_size_m"
            )
            ego_semantic = _oracle_array(
                artifact["ego_semantic_grid"], (37, 50, 50), "ego semantic grid"
            )
            ego_observed = _oracle_array(
                artifact["ego_observed_mask"], (50, 50), "ego observed mask"
            )
            ego_free = _oracle_array(
                artifact["ego_free_mask"], (50, 50), "ego free mask"
            )
            target_semantic = _oracle_array(
                artifact["target_semantic_grid"], (37, 50, 50), "target semantic grid"
            )
            target_observed = _oracle_array(
                artifact["target_observed_mask"], (50, 50), "target observed mask"
            )
            target_free = _oracle_array(
                artifact["target_free_mask"], (50, 50), "target free mask"
            )
            start_position = _oracle_vector(
                artifact["start_position"], "start_position", unit=False
            )
            start_direction = _oracle_vector(
                artifact["start_direction"], "start_direction", unit=True
            )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("oracle artifact violates the 11-member contract") from error
    for prefix, semantic, observed, free in (
        ("ego", ego_semantic, ego_observed, ego_free),
        ("target", target_semantic, target_observed, target_free),
    ):
        if np.any(semantic.any(axis=0) & ~observed) or np.any(free & ~observed):
            raise ValueError(f"{prefix} oracle evidence exceeds observation")
    return PinnedOracle(
        sha256=hashlib.sha256(data).hexdigest(),
        ego_semantic_grid=ego_semantic,
        ego_observed_mask=ego_observed,
        ego_free_mask=ego_free,
        target_semantic_grid=target_semantic,
        target_observed_mask=target_observed,
        target_free_mask=target_free,
        start_position=start_position,
        start_direction=start_direction,
    )


def _open_real_directory(path: Path, label: str) -> int:
    absolute = path if path.is_absolute() else Path.cwd() / path
    if any(part in {"", ".", ".."} for part in absolute.parts[1:]):
        raise ValueError(f"{label} path escape")
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in absolute.parts[1:]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            previous = descriptor
            descriptor = child
            _require_closed(None, previous)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a directory")
        return descriptor
    except OSError as error:
        converted = ValueError(f"{label} must be a real directory")
        _require_closed(converted, descriptor)
        raise converted from error
    except BaseException as error:
        _require_closed(error, descriptor)
        raise


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _is_equal_or_descendant(path: Path, root: Path) -> bool:
    try:
        _absolute_path(path).relative_to(_absolute_path(root))
    except ValueError:
        return False
    return True


def _require_private_snapshot_root(path: Path, descriptor: int) -> None:
    if any(
        _is_equal_or_descendant(path, forbidden)
        for forbidden in (FINAL_PACKAGE_ROOT, SMOKE_PACKAGE_ROOT)
    ):
        raise ValueError("private snapshot root cannot be package staging")
    state = os.fstat(descriptor)
    if state.st_uid != os.geteuid() or stat.S_IMODE(state.st_mode) != 0o700:
        raise ValueError("private snapshot root must be process-private mode 0700")


def _copy_asset(
    source_directory: int, destination_directory: int, name: str
) -> SnapshotFile:
    source_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW
    destination_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        source = os.open(name, source_flags, dir_fd=source_directory)
    except OSError as error:
        raise ValueError(f"scene asset must be a regular file: {name}") from error
    destination = -1
    created = False
    primary = None
    try:
        before = os.fstat(source)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"scene asset must be a regular file: {name}")
        destination = os.open(
            name, destination_flags, 0o444, dir_fd=destination_directory
        )
        created = True
        os.fchmod(destination, 0o444)
        source_hash = hashlib.sha256()
        destination_hash = hashlib.sha256()
        byte_length = 0
        successful_writes = 0
        while True:
            chunk = os.read(source, 1024 * 1024)
            if not chunk:
                break
            source_hash.update(chunk)
            written = 0
            while written < len(chunk):
                count = os.write(destination, chunk[written:])
                if count <= 0:
                    raise OSError("short asset write")
                destination_hash.update(chunk[written : written + count])
                written += count
                successful_writes += count
            byte_length += len(chunk)
        after = os.fstat(source)
        destination_state = os.fstat(destination)
        fingerprint = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(before, field) != getattr(after, field) for field in fingerprint
        ):
            raise ValueError(f"scene asset changed while copied: {name}")
        if (
            byte_length != before.st_size
            or successful_writes != before.st_size
            or destination_state.st_size != before.st_size
            or source_hash.digest() != destination_hash.digest()
        ):
            raise ValueError(f"scene asset copy verification failed: {name}")
        os.fsync(destination)
        os.fsync(destination_directory)
        return SnapshotFile(
            path=Path(name), byte_length=byte_length, sha256=source_hash.hexdigest()
        )
    except BaseException as caught:
        primary = caught
        if created:
            try:
                os.unlink(name, dir_fd=destination_directory)
                os.fsync(destination_directory)
            except OSError as cleanup_error:
                combined = _OperationCleanupError(
                    f"asset cleanup failed: {name}",
                    caught,
                    (cleanup_error,),
                )
                primary = combined
                raise combined from caught
        raise
    finally:
        descriptors = (source,) if destination < 0 else (destination, source)
        _require_closed(primary, *descriptors)


def _snapshot_scene_bundle_at(
    scene_id: str,
    private_sibling: Path,
    private_root: int,
) -> SceneBundle:
    scene_id = _require_scene_id(scene_id, "scene_id")
    if not isinstance(private_sibling, Path):
        raise ValueError("private snapshot root is invalid")
    if any(
        _is_equal_or_descendant(private_sibling, forbidden)
        for forbidden in (FINAL_PACKAGE_ROOT, SMOKE_PACKAGE_ROOT)
    ):
        raise ValueError("private snapshot root cannot be package staging")
    source_root = _open_real_directory(SCENE_DATASET_ROOT, "scene dataset root")
    scene_source = -1
    scene_destination = -1
    scene_created = False
    copied_names: list[str] = []
    primary = None
    try:
        _require_private_snapshot_root(private_sibling, private_root)
        try:
            scene_source = os.open(
                scene_id,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=source_root,
            )
            if not stat.S_ISDIR(os.fstat(scene_source).st_mode):
                raise ValueError("scene source must be a directory")
            os.mkdir(scene_id, 0o700, dir_fd=private_root)
            scene_created = True
            os.fsync(private_root)
            scene_destination = os.open(
                scene_id,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=private_root,
            )
        except OSError as error:
            raise FileExistsError(
                f"private scene snapshot cannot be created: {scene_id}"
            ) from error
        names = {
            "glb": f"{scene_id}.glb",
            "house": f"{scene_id}.house",
            "semantic_ply": f"{scene_id}_semantic.ply",
            "navmesh": f"{scene_id}.navmesh",
        }
        files = {}
        for role, name in names.items():
            copied = _copy_asset(scene_source, scene_destination, name)
            copied_names.append(name)
            files[role] = SnapshotFile(
                path=private_sibling / scene_id / copied.path,
                byte_length=copied.byte_length,
                sha256=copied.sha256,
            )
        os.fsync(scene_destination)
        os.fsync(private_root)
        return SceneBundle(scene_id=scene_id, files=MappingProxyType(files))
    except BaseException as caught:
        primary = caught
        if scene_created:
            cleanup_errors = []
            for name in reversed(copied_names):
                try:
                    os.unlink(name, dir_fd=scene_destination)
                except OSError as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            try:
                if scene_destination >= 0:
                    os.fsync(scene_destination)
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                os.rmdir(scene_id, dir_fd=private_root)
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                os.fsync(private_root)
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                combined = _OperationCleanupError(
                    f"snapshot cleanup failed: {scene_id}",
                    caught,
                    cleanup_errors,
                )
                primary = combined
                raise combined from caught
        raise
    finally:
        descriptors = tuple(
            descriptor
            for descriptor in (
                scene_destination,
                scene_source,
                private_root,
                source_root,
            )
            if descriptor >= 0
        )
        _require_closed(primary, *descriptors)


def snapshot_scene_bundle(scene_id: str, private_sibling: Path) -> SceneBundle:
    """Copy exactly the four immutable simulator assets to a private snapshot."""

    if not isinstance(private_sibling, Path):
        raise ValueError("private snapshot root is invalid")
    if any(
        _is_equal_or_descendant(private_sibling, forbidden)
        for forbidden in (FINAL_PACKAGE_ROOT, SMOKE_PACKAGE_ROOT)
    ):
        raise ValueError("private snapshot root cannot be package staging")
    private_root = _open_real_directory(private_sibling, "private snapshot root")
    return _snapshot_scene_bundle_at(scene_id, private_sibling, private_root)


def _snapshot_owned_scene_bundle(
    scene_id: str,
    snapshots: _OwnedDirectory,
) -> SceneBundle:
    _require_owned_directory_binding(snapshots)
    bundle = _snapshot_scene_bundle_at(
        scene_id,
        snapshots.path,
        os.dup(snapshots.descriptor),
    )
    _require_owned_directory_binding(snapshots)
    return bundle


class _QuaternionComponents(Protocol):
    x: float
    y: float
    z: float
    w: float


class _SensorState(Protocol):
    position: Sequence[float]
    rotation: object


class _AgentState(Protocol):
    position: Sequence[float]
    rotation: object
    sensor_states: Mapping[str, _SensorState]


class _SemanticCategory(Protocol):
    def index(self, mapping: str = ...) -> int: ...


class _AABB(Protocol):
    center: Sequence[float]
    sizes: Sequence[float]


class _SemanticRegion(Protocol):
    category: _SemanticCategory | None
    aabb: _AABB


class _SemanticObject(Protocol):
    id: str
    category: _SemanticCategory | None
    region: _SemanticRegion | None


class _SemanticLevel(Protocol):
    aabb: _AABB
    regions: Sequence[_SemanticRegion]


class _SemanticScene(Protocol):
    objects: Sequence[_SemanticObject | None]
    levels: Sequence[_SemanticLevel]


class _Simulator(Protocol):
    def get_observations_at(
        self,
        *,
        position: list[float],
        rotation: list[float],
        keep_agent_at_new_pose: bool,
    ) -> Mapping[str, object]: ...

    def get_agent_state(self) -> _AgentState: ...

    def semantic_annotations(self) -> _SemanticScene: ...

    def close(self) -> None: ...


class _Closable(Protocol):
    def close(self) -> None: ...


class _ResolvedSensorConfig(Protocol):
    TYPE: str
    HEIGHT: int
    HFOV: float
    MAX_DEPTH: float
    MIN_DEPTH: float
    NORMALIZE_DEPTH: bool
    ORIENTATION: Sequence[float]
    POSITION: Sequence[float]
    UUID: str
    WIDTH: int


class _ResolvedAgentConfig(Protocol):
    SENSORS: Sequence[str]


class _ResolvedSimulatorConfig(Protocol):
    AGENT_0: _ResolvedAgentConfig


class _ResolvedHabitatConfig(Protocol):
    SIMULATOR: _ResolvedSimulatorConfig


_SENSOR_YAWS = tuple(range(0, 360, 30))
_SENSOR_KINDS = ("RGB", "DEPTH", "SEMANTIC")
_SENSOR_POSITION = np.asarray([0.0, 1.25, 0.0], dtype="<f8")
_RAW_QUATERNION_NORM_ABS_TOLERANCE = float(np.finfo(np.float32).eps)


def _sensor_name(kind: str, yaw_degrees: int) -> str:
    return f"{kind}_{yaw_degrees:03d}"


def _require_scene_bundle(bundle: SceneBundle) -> None:
    if not isinstance(bundle, SceneBundle):
        raise ValueError("simulator construction requires a scene snapshot bundle")
    expected = {
        "glb": f"{bundle.scene_id}.glb",
        "house": f"{bundle.scene_id}.house",
        "semantic_ply": f"{bundle.scene_id}_semantic.ply",
        "navmesh": f"{bundle.scene_id}.navmesh",
    }
    if set(bundle.files) != set(expected):
        raise ValueError("scene snapshot bundle must contain exactly four assets")
    parents = {record.path.parent for record in bundle.files.values()}
    if len(parents) != 1 or any(
        bundle.files[role].path.name != name for role, name in expected.items()
    ):
        raise ValueError("scene snapshot bundle paths are not canonical")


def _sensor_config() -> Mapping[str, object]:
    sensor_types = {
        "depth": "DEPTH",
        "rgb": "COLOR",
        "semantic": "SEMANTIC",
    }
    specs = []
    for kind in ("depth", "rgb", "semantic"):
        for yaw_degrees in _SENSOR_YAWS:
            is_depth = kind == "depth"
            specs.append(
                {
                    "habitat_sensor_subtype": "PINHOLE",
                    "habitat_sensor_type": sensor_types[kind],
                    "hfov_degrees": 90.0,
                    "max_depth_m": 10.0 if is_depth else None,
                    "min_depth_m": 0.0 if is_depth else None,
                    "modality": kind,
                    "normalize_depth": False if is_depth else None,
                    "orientation": [0.0, math.radians(yaw_degrees), 0.0],
                    "position": [0.0, 1.25, 0.0],
                    "resolution": [256, 256],
                    "uuid": _sensor_name(kind, yaw_degrees),
                }
            )
    specs.sort(key=lambda item: cast(str, item["uuid"]))
    return {
        "camera_pose_authority": "returned-depth-sensor-state",
        "depth_units": "metres",
        "height": 256,
        "hfov_degrees": 90.0,
        "max_depth_m": 10.0,
        "min_depth_m": 0.0,
        "normalize_depth": False,
        "orientation_rule": "[0,radians(yaw_degrees),0]",
        "position": [0.0, 1.25, 0.0],
        "resolved_specs": specs,
        "rgb_channel_order": "RGB",
        "views": 12,
        "width": 256,
        "yaw_degrees": list(_SENSOR_YAWS),
    }


def _require_live_sensor_config(config: object) -> None:
    import habitat_sim

    sensor_spec = cast(Callable[[], object], getattr(habitat_sim, "SensorSpec"))()
    sensor_subtype = getattr(habitat_sim, "SensorSubType")
    if (
        getattr(sensor_spec, "sensor_subtype")
        != getattr(sensor_subtype, "PINHOLE")
    ):
        raise ValueError("live Habitat sensor subtype differs from commitment")
    simulator = cast(_ResolvedHabitatConfig, config).SIMULATOR
    resolved = []
    type_names = {
        "HabitatSimDepthSensor": ("depth", "DEPTH"),
        "HabitatSimRGBSensor": ("rgb", "COLOR"),
        "HabitatSimSemanticSensor": ("semantic", "SEMANTIC"),
    }
    for name in simulator.AGENT_0.SENSORS:
        sensor = cast(_ResolvedSensorConfig, getattr(simulator, name))
        try:
            modality, sensor_type = type_names[sensor.TYPE]
        except KeyError as error:
            raise ValueError("live Habitat sensor type differs from commitment") from error
        if not name.startswith(f"{modality.upper()}_"):
            raise ValueError("live Habitat sensor name and type disagree")
        is_depth = modality == "depth"
        resolved.append(
            {
                "habitat_sensor_subtype": "PINHOLE",
                "habitat_sensor_type": sensor_type,
                "hfov_degrees": sensor.HFOV,
                "max_depth_m": sensor.MAX_DEPTH if is_depth else None,
                "min_depth_m": sensor.MIN_DEPTH if is_depth else None,
                "modality": modality,
                "normalize_depth": sensor.NORMALIZE_DEPTH if is_depth else None,
                "orientation": list(sensor.ORIENTATION),
                "position": list(sensor.POSITION),
                "resolution": [sensor.HEIGHT, sensor.WIDTH],
                "uuid": sensor.UUID,
            }
        )
    resolved.sort(key=lambda item: cast(str, item["uuid"]))
    if resolved != _sensor_config()["resolved_specs"]:
        raise ValueError("live Habitat sensor configuration differs from commitment")


def _require_live_simulator_sensor_specs(simulator: object) -> None:
    try:
        sim_config = getattr(simulator, "sim_config")
        agent = getattr(sim_config, "agents")[0]
        specifications = getattr(agent, "sensor_specifications")
        sensors = getattr(getattr(simulator, "sensor_suite"), "sensors")
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("live Habitat simulator sensor specs are unavailable") from error
    expected = {
        cast(str, record["uuid"]): record
        for record in cast(
            Sequence[Mapping[str, object]],
            _sensor_config()["resolved_specs"],
        )
    }
    if len(specifications) != len(expected):
        raise ValueError("live Habitat simulator sensor specs differ")
    seen = set()
    for specification in specifications:
        try:
            uuid = getattr(specification, "uuid")
            record = expected[uuid]
            resolution = np.asarray(getattr(specification, "resolution"))
            position = np.asarray(getattr(specification, "position"))
            orientation = np.asarray(getattr(specification, "orientation"))
            hfov = float(getattr(specification, "parameters")["hfov"])
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError("live Habitat simulator sensor specs differ") from error
        if (
            uuid in seen
            or getattr(getattr(specification, "sensor_type"), "name")
            != record["habitat_sensor_type"]
            or getattr(getattr(specification, "sensor_subtype"), "name")
            != record["habitat_sensor_subtype"]
            or resolution.dtype != np.int32
            or not np.array_equal(
                resolution, np.asarray(record["resolution"], dtype=np.int32)
            )
            or position.dtype != np.float32
            or not np.array_equal(
                position, np.asarray(record["position"], dtype=np.float32)
            )
            or orientation.dtype != np.float32
            or not np.array_equal(
                orientation, np.asarray(record["orientation"], dtype=np.float32)
            )
            or hfov != record["hfov_degrees"]
        ):
            raise ValueError("live Habitat simulator sensor specs differ")
        if record["modality"] == "depth":
            try:
                depth = sensors[uuid].config
            except (AttributeError, KeyError, TypeError) as error:
                raise ValueError(
                    "live Habitat simulator depth settings differ"
                ) from error
            if (
                depth.MIN_DEPTH != record["min_depth_m"]
                or depth.MAX_DEPTH != record["max_depth_m"]
                or depth.NORMALIZE_DEPTH is not record["normalize_depth"]
            ):
                raise ValueError("live Habitat simulator depth settings differ")
        seen.add(uuid)
    if seen != set(expected):
        raise ValueError("live Habitat simulator sensor specs differ")


def build_scene_simulator(bundle: SceneBundle) -> _Simulator:
    """Build the fixed 36-sensor Habitat simulator from private snapshot paths."""

    from habitat import get_config
    from habitat.sims import make_sim

    _require_scene_bundle(bundle)
    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = str(bundle.files["glb"].path)
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
    config.SIMULATOR.AGENT_0.SENSORS = []
    templates = {
        kind: deepcopy(getattr(config.SIMULATOR, f"{kind}_SENSOR"))
        for kind in _SENSOR_KINDS
    }
    for kind, template in templates.items():
        template.WIDTH = 256
        template.HEIGHT = 256
        template.HFOV = 90.0
        template.POSITION = list(_SENSOR_POSITION)
        if kind == "DEPTH":
            template.MIN_DEPTH = 0.0
            template.MAX_DEPTH = 10.0
            template.NORMALIZE_DEPTH = False
    for yaw_degrees in _SENSOR_YAWS:
        for kind in _SENSOR_KINDS:
            name = _sensor_name(kind, yaw_degrees)
            sensor = deepcopy(templates[kind])
            sensor.UUID = name.lower()
            sensor.ORIENTATION = [0.0, math.radians(yaw_degrees), 0.0]
            setattr(config.SIMULATOR, name, sensor)
            config.SIMULATOR.AGENT_0.SENSORS.append(name)
    _require_live_sensor_config(config)
    config.freeze()
    simulator = cast(
        _Simulator,
        make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR),
    )
    try:
        _require_live_simulator_sensor_specs(simulator)
    except BaseException:
        simulator.close()
        raise
    return simulator


def _position(value: object, label: str) -> np.ndarray:
    result = np.asarray(value, dtype="<f8")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} position must be finite float64[3]")
    return result


def _quaternion_components(value: object, label: str) -> np.ndarray:
    raw: np.ndarray
    if isinstance(value, np.ndarray):
        if value.dtype != np.float64:
            raise ValueError(f"{label} quaternion must be float64")
        raw = value
    elif isinstance(value, (list, tuple)):
        raw = np.asarray(value, dtype="<f8")
    else:
        quaternion = cast(_QuaternionComponents, value)
        raw = np.asarray(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
            dtype="<f8",
        )
    if raw.shape != (4,) or not np.isfinite(raw).all():
        raise ValueError(f"{label} quaternion must be finite float64 XYZW")
    components = struct.unpack("<4d", raw.tobytes())
    norm = math.sqrt(sum(component**2 for component in components))
    if not math.isclose(
        norm,
        1.0,
        rel_tol=0.0,
        abs_tol=_RAW_QUATERNION_NORM_ABS_TOLERANCE,
    ):
        raise ValueError(f"{label} quaternion norm is outside tolerance")
    return np.asarray([component / norm for component in components], dtype="<f8")


def _semantic_object_map(
    semantic_scene: _SemanticScene,
) -> Mapping[int, _SemanticObject]:
    result = {}
    for semantic_object in semantic_scene.objects:
        if semantic_object is None:
            continue
        try:
            semantic_id = int(semantic_object.id.rsplit("_", 1)[-1])
        except (AttributeError, ValueError) as error:
            raise ValueError("semantic object ID must end in an integer suffix") from error
        if semantic_id in result:
            raise ValueError(f"duplicate semantic object ID {semantic_id}")
        result[semantic_id] = semantic_object
    return result


def _map_semantic_ids(
    semantic_ids: np.ndarray,
    objects: Mapping[int, _SemanticObject],
) -> Tuple[np.ndarray, np.ndarray]:
    object_categories = np.full(semantic_ids.shape, -1, dtype="<i2")
    region_categories = np.full(semantic_ids.shape, -1, dtype="<i2")
    for semantic_id in np.unique(semantic_ids):
        mask = semantic_ids == semantic_id
        semantic_object = objects.get(int(semantic_id))
        if semantic_object is None or semantic_object.category is None:
            continue
        raw_object = int(semantic_object.category.index(mapping="mpcat40"))
        if raw_object == 0:
            mapped_object = 0
        elif 0 < raw_object < len(OBJECT_MAPPING):
            mapped_object = OBJECT_MAPPING[raw_object]
        else:
            mapped_object = MAPPED_OBJECT_NAMES.index("other")
        object_categories[mask] = mapped_object
        region = semantic_object.region
        if region is None or region.category is None:
            continue
        raw_region = int(region.category.index())
        if not 0 <= raw_region < len(REGION_MAPPING):
            raise ValueError(f"invalid MP3D region category {raw_region}")
        region_categories[mask] = REGION_MAPPING[raw_region]
    return object_categories, region_categories


def _aabb_minimum(aabb: _AABB) -> np.ndarray:
    center = np.asarray(aabb.center, dtype="<f4")
    sizes = np.asarray(aabb.sizes, dtype="<f4")
    minimum = center - sizes / np.float32(2.0)
    if (
        center.shape != (3,)
        or sizes.shape != (3,)
        or not np.isfinite(minimum).all()
    ):
        raise ValueError("semantic AABB must contain finite 3-vectors")
    return minimum.astype("<f8")


def _level_floor_y(level: _SemanticLevel) -> float:
    if level.regions:
        return min(float(_aabb_minimum(region.aabb)[1]) for region in level.regions)
    return float(_aabb_minimum(level.aabb)[1])


def _target_origin_at_start(
    semantic_scene: _SemanticScene, start_y: float
) -> Tuple[float, float]:
    if not math.isfinite(start_y):
        raise ValueError("start Y must be finite")
    if not semantic_scene.levels:
        raise ValueError("MP3D semantic scene contains no levels")
    levels = sorted(semantic_scene.levels, key=_level_floor_y)
    selected = levels[0]
    for level in levels:
        if _level_floor_y(level) <= start_y:
            selected = level
        else:
            break
    minimum = _aabb_minimum(selected.aabb)
    return float(minimum[0]), float(minimum[2])


def _observation_array(
    observations: Mapping[str, object], name: str
) -> np.ndarray:
    if name not in observations or not isinstance(observations[name], np.ndarray):
        raise ValueError(f"{name} must be an ndarray")
    return cast(np.ndarray, observations[name])


def _frames_from_arrays(arrays: RawFrameArrays) -> Tuple[OracleSensorFrame, ...]:
    return tuple(
        OracleSensorFrame(
            depth_m=arrays.depth_m[index],
            object_categories=arrays.object_categories[index],
            region_categories=arrays.region_categories[index],
            sensor_position=cast(
                Tuple[float, float, float], tuple(arrays.sensor_positions[index])
            ),
            sensor_rotation=cast(
                Tuple[float, float, float, float],
                tuple(arrays.sensor_rotations_xyzw[index]),
            ),
        )
        for index in range(12)
    )


def render_raw_frame_artifact(
    simulator: object,
    observation: CollectionObservation,
) -> RawFrameArrays:
    """Render and validate one sealed observation into the frozen raw arrays."""

    habitat_simulator = cast(_Simulator, simulator)
    observations = habitat_simulator.get_observations_at(
        position=list(observation.start_position),
        rotation=list(observation.start_rotation),
        keep_agent_at_new_pose=True,
    )
    agent_state = habitat_simulator.get_agent_state()
    semantic_scene = habitat_simulator.semantic_annotations()
    sealed_position = np.asarray(observation.start_position, dtype="<f8")
    if not np.array_equal(_position(agent_state.position, "agent"), sealed_position):
        raise ValueError("returned agent position differs from sealed position")
    agent_rotation = _quaternion_components(agent_state.rotation, "agent")
    sealed_rotation = _quaternion_components(
        np.asarray(observation.start_rotation, dtype="<f8"),
        "sealed",
    )
    if 1.0 - abs(float(np.dot(agent_rotation, sealed_rotation))) > 1e-12:
        raise ValueError("returned agent rotation differs from sealed rotation")

    semantic_objects = _semantic_object_map(semantic_scene)
    rgb_frames = []
    depth_frames = []
    object_frames = []
    region_frames = []
    sensor_positions = []
    sensor_rotations = []
    for yaw_degrees in _SENSOR_YAWS:
        names = {
            kind: _sensor_name(kind.lower(), yaw_degrees) for kind in _SENSOR_KINDS
        }
        rgb = _observation_array(observations, names["RGB"])
        if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3):
            raise ValueError(
                f"{names['RGB']} must be exact uint8[256,256,3] RGB"
            )
        depth = _observation_array(observations, names["DEPTH"])
        if depth.dtype != np.float32:
            raise ValueError(f"{names['DEPTH']} must be metric float32")
        if depth.shape == (256, 256, 1):
            depth = depth[..., 0]
        if (
            depth.shape != (256, 256)
            or not np.isfinite(depth).all()
            or np.any(depth < 0.0)
            or np.any(depth > 10.0)
        ):
            raise ValueError(f"{names['DEPTH']} has invalid shape or metric range")
        semantic = _observation_array(observations, names["SEMANTIC"])
        if (
            semantic.shape != (256, 256)
            or not np.issubdtype(semantic.dtype, np.integer)
        ):
            raise ValueError(f"{names['SEMANTIC']} must be integer[256,256]")
        object_categories, region_categories = _map_semantic_ids(
            semantic, semantic_objects
        )
        positions = []
        rotations = []
        for kind in _SENSOR_KINDS:
            state = agent_state.sensor_states[names[kind]]
            positions.append(_position(state.position, names[kind]))
            rotations.append(_quaternion_components(state.rotation, names[kind]))
        if any(
            not np.array_equal(positions[0], position) for position in positions[1:]
        ) or any(
            not np.array_equal(rotations[0], rotation) for rotation in rotations[1:]
        ):
            raise ValueError(f"sensor extrinsics differ at yaw {yaw_degrees}")
        rgb_frames.append(rgb)
        depth_frames.append(depth)
        object_frames.append(object_categories)
        region_frames.append(region_categories)
        sensor_positions.append(positions[1])
        sensor_rotations.append(rotations[1])

    target_origin = np.asarray(
        _target_origin_at_start(semantic_scene, observation.start_position[1]),
        dtype="<f8",
    )
    arrays_without_masks = {
        "schema_version": np.asarray(1, dtype="<i8"),
        "rgb": np.ascontiguousarray(np.stack(rgb_frames)),
        "depth_m": np.ascontiguousarray(np.stack(depth_frames)),
        "object_categories": np.ascontiguousarray(np.stack(object_frames)),
        "region_categories": np.ascontiguousarray(np.stack(region_frames)),
        "sensor_positions": np.ascontiguousarray(
            np.stack(sensor_positions).astype("<f8", copy=False)
        ),
        "sensor_rotations_xyzw": np.ascontiguousarray(
            np.stack(sensor_rotations).astype("<f8", copy=False)
        ),
        "sensor_yaw_degrees": np.arange(0, 360, 30, dtype="<i2"),
        "sensor_hfov_degrees": np.asarray(90.0, dtype="<f8"),
        "sensor_position_relative": _SENSOR_POSITION.copy(),
        "start_position": sealed_position.copy(),
        "start_rotation_xyzw": np.asarray(observation.start_rotation, dtype="<f8"),
        "target_origin_xz": target_origin,
    }
    provisional = RawFrameArrays(
        **arrays_without_masks,
        ego_observed_mask=np.zeros((50, 50), dtype=np.bool_),
        ego_free_mask=np.zeros((50, 50), dtype=np.bool_),
        target_observed_mask=np.zeros((50, 50), dtype=np.bool_),
        target_free_mask=np.zeros((50, 50), dtype=np.bool_),
    )
    evidence = project_oracle_frames(
        _frames_from_arrays(provisional),
        start_position=tuple(float(value) for value in provisional.start_position),
        start_rotation=tuple(
            float(value) for value in provisional.start_rotation_xyzw
        ),
        target_origin_xz=tuple(
            float(value) for value in provisional.target_origin_xz
        ),
    )
    return RawFrameArrays(
        **arrays_without_masks,
        ego_observed_mask=np.ascontiguousarray(evidence.ego_observed_mask),
        ego_free_mask=np.ascontiguousarray(evidence.ego_free_mask),
        target_observed_mask=np.ascontiguousarray(evidence.target_observed_mask),
        target_free_mask=np.ascontiguousarray(evidence.target_free_mask),
    )


def replay_and_require_exact(
    arrays: RawFrameArrays,
    observation: CollectionObservation,
    oracle: PinnedOracle,
) -> None:
    """Replay stored public frames and require exact equality with the oracle."""

    if not np.array_equal(
        arrays.start_position, np.asarray(observation.start_position, dtype="<f8")
    ) or not np.array_equal(
        arrays.start_rotation_xyzw,
        np.asarray(observation.start_rotation, dtype="<f8"),
    ):
        raise ValueError("stored start pose differs from sealed metadata")
    evidence = project_oracle_frames(
        _frames_from_arrays(arrays),
        start_position=tuple(float(value) for value in arrays.start_position),
        start_rotation=tuple(float(value) for value in arrays.start_rotation_xyzw),
        target_origin_xz=tuple(float(value) for value in arrays.target_origin_xz),
    )
    for field in (
        "ego_observed_mask",
        "ego_free_mask",
        "target_observed_mask",
        "target_free_mask",
    ):
        if not np.array_equal(
            getattr(arrays, field), getattr(evidence, field)
        ) or not np.array_equal(getattr(arrays, field), getattr(oracle, field)):
            raise ValueError(f"{field} differs from stored replay commitment")
    for field in (
        "ego_semantic_grid",
        "ego_observed_mask",
        "ego_free_mask",
        "target_semantic_grid",
        "target_observed_mask",
        "target_free_mask",
    ):
        if not np.array_equal(getattr(evidence, field), getattr(oracle, field)):
            raise ValueError(f"{field} differs from pinned oracle")
    replayed_position = np.asarray(evidence.start_position, dtype=np.float32)
    replayed_direction = np.asarray(evidence.start_direction, dtype=np.float32)
    if not np.array_equal(
        replayed_position, np.asarray(oracle.start_position, dtype=np.float32)
    ):
        raise ValueError("replayed target-local start_position differs")
    if not np.array_equal(
        replayed_direction, np.asarray(oracle.start_direction, dtype=np.float32)
    ):
        raise ValueError("replayed start_direction differs")


_ASSET_ROLE_VALUE = {
    "algorithm": "single-role-omission-first-row-per-scene-v1",
    "auxiliary": ["navmesh"],
    "required": ["glb", "house", "semantic_ply"],
    "schema_version": 1,
}
_ASSET_ROLE_BYTES = (
    json.dumps(_ASSET_ROLE_VALUE, indent=2, sort_keys=True).encode("utf-8") + b"\n"
)
_SOURCE_PATHS = {
    "collector_source": "prior/analyze/d2026_07_29/rgbd_segmenter_raw_frames.py",
    "package_source": (
        "prior/analyze/d2026_07_29/rgbd_segmenter_raw_frame_package.py"
    ),
    "asset_roles": (
        "prior/analyze/d2026_07_29/rgbd_segmenter_asset_roles.json"
    ),
    "cohort_manifest": (
        "data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1/manifest.json"
    ),
    "cohort_jsonl": (
        "data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1/cohort.jsonl"
    ),
    "evidence_manifest": (
        "data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen/manifest.json"
    ),
    "evidence_index": (
        "data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen/index.jsonl"
    ),
    "raw_split": (
        "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/"
        "val_unseen.json.gz"
    ),
    "projector_source": (
        "vlnce_baselines/models/etp_llm/llm_grid_oracle_cache.py"
    ),
    "mapping_source": "prior/constants.py",
}


def _source_json(source: SourceRecord) -> Mapping[str, object]:
    return {
        "path": source.path,
        "byte_length": source.byte_length,
        "sha256": source.sha256,
    }


def _distribution_name(name: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    if not normalized:
        raise ValueError("installed distribution name is empty")
    return normalized


def _require_unmasked_physical_gpu_zero() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be absent to bind logical GPU 0 "
            "to physical GPU 0"
        )


def _require_little_endian() -> None:
    if sys.byteorder != "little":
        raise RuntimeError("official raw-frame publication requires little-endian")


def capture_environment() -> EnvironmentCapture:
    """Capture the exact software and fixed GPU-0 identity."""

    _require_unmasked_physical_gpu_zero()
    distributions = tuple(
        sorted(
            {
                InstalledDistribution(
                    name=_distribution_name(distribution.metadata["Name"]),
                    version=distribution.version,
                )
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"] is not None
            }
        )
    )
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("PyTorch has no CUDA runtime version")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,name,uuid",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.reader(completed.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError("nvidia-smi returned an invalid GPU-0 identity")
    driver, gpu_name, gpu_uuid = (field.strip() for field in rows[0])
    if not driver or not gpu_name or not gpu_uuid.startswith("GPU-"):
        raise RuntimeError("nvidia-smi returned an incomplete GPU-0 identity")
    return EnvironmentCapture(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        numpy_version=np.__version__,
        zlib_version=zlib.ZLIB_RUNTIME_VERSION,
        habitat_version=importlib.metadata.version("habitat"),
        habitat_sim_version=importlib.metadata.version("habitat-sim"),
        cuda_runtime_version=cuda_version,
        nvidia_driver_version=driver,
        gpu_name=gpu_name,
        gpu_uuid=gpu_uuid,
        gpu_device_id=0,
        installed_distributions=distributions,
    )


def _environment_json(environment: EnvironmentCapture) -> Mapping[str, object]:
    if (
        environment.gpu_device_id != 0
        or tuple(sorted(set(environment.installed_distributions)))
        != environment.installed_distributions
    ):
        raise ValueError("environment GPU or distributions have drifted")
    distributions = [
        asdict(distribution) for distribution in environment.installed_distributions
    ]
    distributions_bytes = json.dumps(
        distributions, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "cuda_runtime_version": environment.cuda_runtime_version,
        "gpu_device_id": environment.gpu_device_id,
        "gpu_name": environment.gpu_name,
        "gpu_uuid": environment.gpu_uuid,
        "habitat_sim_version": environment.habitat_sim_version,
        "habitat_version": environment.habitat_version,
        "installed_distributions": distributions,
        "installed_distributions_sha256": hashlib.sha256(
            distributions_bytes
        ).hexdigest(),
        "numpy_version": environment.numpy_version,
        "nvidia_driver_version": environment.nvidia_driver_version,
        "platform": environment.platform,
        "python_implementation": environment.python_implementation,
        "python_version": environment.python_version,
        "zlib_version": environment.zlib_version,
    }


def _mapping_sha256() -> str:
    mapping = {
        "MAPPED_OBJECT_NAMES": MAPPED_OBJECT_NAMES,
        "MAPPED_REGION_NAMES": MAPPED_REGION_NAMES,
        "OBJECT_MAPPING": OBJECT_MAPPING,
        "REGION_MAPPING": REGION_MAPPING,
    }
    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_manifest_sources(sources: ManifestSources) -> None:
    for name, expected_path in _SOURCE_PATHS.items():
        source = getattr(sources, name)
        if source.path != expected_path:
            raise ValueError(f"{name} path differs from the fixed source")
    if sources.asset_roles.data != _ASSET_ROLE_BYTES:
        raise ValueError("asset-role source content differs from frozen classification")


def build_manifest(
    *,
    git_commit: str,
    sources: ManifestSources,
    scene_assets: Mapping[str, SceneAssetCommitment],
    environment: EnvironmentCapture,
    index_bytes: bytes,
    rows: Sequence[IndexRow],
) -> bytes:
    """Build canonical manifest bytes from already accepted attempt commitments."""

    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ValueError("collection Git commit must be a lowercase full hash")
    _require_manifest_sources(sources)
    materialized_rows = tuple(rows)
    if canonical_index_bytes(materialized_rows) != index_bytes:
        raise ValueError("index bytes differ from supplied rows")
    if tuple(sorted(scene_assets)) != tuple(scene_assets) or len(scene_assets) != 11:
        raise ValueError("scene assets must contain 11 lexical scenes")
    if tuple(sorted({row.scene_id for row in materialized_rows})) != tuple(
        scene_assets
    ):
        raise ValueError("index scenes differ from scene asset commitments")
    artifact_records = [(row.artifact, row.npz) for row in materialized_rows]
    index_record = FileRecord(
        byte_length=len(index_bytes),
        sha256=hashlib.sha256(index_bytes).hexdigest(),
    )
    artifact_tree = tree_aggregate(artifact_records)
    payload_tree = tree_aggregate(
        (*artifact_records, ("index.jsonl", index_record))
    )
    sensor_config = _sensor_config()
    sensor_config_bytes = json.dumps(
        sensor_config, sort_keys=True, separators=(",", ":")
    ).encode()
    scenes = {}
    for scene_id, commitment in scene_assets.items():
        if _SCENE_ID.fullmatch(scene_id) is None:
            raise ValueError("manifest scene ID is invalid")
        files = {
            role: asdict(commitment.files[role])
            for role in sorted(commitment.files)
        }
        expected = SceneAssetCommitment.from_files(commitment.files)
        if commitment.bundle_sha256 != expected.bundle_sha256:
            raise ValueError("scene bundle hash differs from its files")
        expected_names = {
            "glb": f"{scene_id}/{scene_id}.glb",
            "house": f"{scene_id}/{scene_id}.house",
            "navmesh": f"{scene_id}/{scene_id}.navmesh",
            "semantic_ply": f"{scene_id}/{scene_id}_semantic.ply",
        }
        if any(
            commitment.files[role].path != path
            for role, path in expected_names.items()
        ):
            raise ValueError("scene asset source paths differ from canonical names")
        scenes[scene_id] = {
            "bundle_sha256": commitment.bundle_sha256,
            "files": files,
        }
    manifest = {
        "collection": {
            "asset_roles": _source_json(sources.asset_roles),
            "collector_source": _source_json(sources.collector_source),
            "command": (
                "python -m prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames"
            ),
            "git_commit": git_commit,
            "gpu_device_id": 0,
            "package_source": _source_json(sources.package_source),
        },
        "collection_id": "r2r-val-unseen-50-raw-v1",
        "cohort": {
            "cohort_id": "r2r-val-unseen-50-v1",
            "cohort_jsonl": _source_json(sources.cohort_jsonl),
            "directory": COHORT_ROOT.as_posix(),
            "manifest": _source_json(sources.cohort_manifest),
            "observation_count": 50,
            "scene_count": 11,
            "sealing_git_commit": _SEALING_COMMIT,
            "selection_sha256": _SELECTION_SHA256,
        },
        "environment": _environment_json(environment),
        "evidence": {
            "dataset": "R2R",
            "evidence_key": "oracle-t0-v1",
            "index": _source_json(sources.evidence_index),
            "manifest": _source_json(sources.evidence_manifest),
            "mapping_sha256": _mapping_sha256(),
            "mapping_source": _source_json(sources.mapping_source),
            "projector_source": _source_json(sources.projector_source),
            "raw_split": _source_json(sources.raw_split),
            "root": ORACLE_ARTIFACT_ROOT.as_posix(),
            "split": "val_unseen",
        },
        "files": {
            "artifacts": asdict(artifact_tree),
            "index": {
                "byte_length": index_record.byte_length,
                "path": "index.jsonl",
                "row_count": 50,
                "sha256": index_record.sha256,
            },
            "payload": asdict(payload_tree),
        },
        "replay": {
            "array_equal_fields": [
                "ego_free_mask",
                "ego_observed_mask",
                "ego_semantic_grid",
                "target_free_mask",
                "target_observed_mask",
                "target_semantic_grid",
            ],
            "metadata_comparison": (
                "cast-replayed-values-to-float32-then-array-equal"
            ),
            "observation_count": 50,
            "passed_count": 50,
            "scene_count": 11,
        },
        "scene_assets": {
            "auxiliary_roles": ["navmesh"],
            "required_roles": ["glb", "house", "semantic_ply"],
            "role_classification_algorithm": (
                "single-role-omission-first-row-per-scene-v1"
            ),
            "scenes": scenes,
            "source_root": SCENE_DATASET_ROOT.as_posix(),
        },
        "schema_version": 1,
        "sensor": {
            "config": sensor_config,
            "config_sha256": hashlib.sha256(sensor_config_bytes).hexdigest(),
        },
    }
    return json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _read_source_capture(path: str) -> tuple[SourceRecord, tuple[int, ...]]:
    source_path = PurePosixPath(path)
    if source_path.is_absolute() or not source_path.parts or any(
        part in ("", ".", "..") for part in source_path.parts
    ):
        raise ValueError(f"source path must be canonical and relative: {path}")
    directory = os.open(
        ".",
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY,
    )
    outer_error = None
    try:
        for part in source_path.parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            previous = directory
            directory = child
            _require_closed(None, previous)
        leaf = source_path.parts[-1]
        before = os.stat(leaf, dir_fd=directory, follow_symlinks=False)
        descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        read_error = None
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise ValueError(f"source changed before read: {path}")
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            opened_after = os.fstat(descriptor)
        except BaseException as caught:
            read_error = caught
            raise
        finally:
            _require_closed(read_error, descriptor)
        after = os.stat(leaf, dir_fd=directory, follow_symlinks=False)
    except OSError as error:
        converted = ValueError(f"source must be a stable regular file: {path}")
        outer_error = converted
        raise converted from error
    except BaseException as error:
        outer_error = error
        raise
    finally:
        _require_closed(outer_error, directory)
    data = b"".join(chunks)

    def fingerprint(value: os.stat_result) -> tuple[int, ...]:
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

    if (
        len(data) != opened.st_size
        or fingerprint(before) != fingerprint(opened)
        or fingerprint(opened) != fingerprint(opened_after)
        or fingerprint(opened_after) != fingerprint(after)
    ):
        raise ValueError(f"source changed while read: {path}")
    return SourceRecord(path=path, data=data), fingerprint(opened_after)


def _read_source(path: str) -> SourceRecord:
    return _read_source_capture(path)[0]


def _capture_source_state() -> _SourceCaptureState:
    asset_role_path = Path(_SOURCE_PATHS["asset_roles"])
    if not asset_role_path.exists():
        raise FileNotFoundError(
            "tracked rgbd_segmenter_asset_roles.json is required for full collection"
        )
    captures = {
        name: _read_source_capture(path) for name, path in _SOURCE_PATHS.items()
    }
    expected_hashes = {
        "cohort_manifest": _COHORT_MANIFEST_SHA256,
        "cohort_jsonl": _COHORT_SHA256,
        "evidence_manifest": _EVIDENCE_MANIFEST_SHA256,
        "evidence_index": _EVIDENCE_INDEX_SHA256,
        "raw_split": _RAW_SPLIT_SHA256,
    }
    for name, expected_hash in expected_hashes.items():
        if captures[name][0].sha256 != expected_hash:
            raise ValueError(f"{name} differs from its frozen source commitment")
    sources = ManifestSources(
        collector_source=captures["collector_source"][0],
        package_source=captures["package_source"][0],
        asset_roles=captures["asset_roles"][0],
        cohort_manifest=captures["cohort_manifest"][0],
        cohort_jsonl=captures["cohort_jsonl"][0],
        evidence_manifest=captures["evidence_manifest"][0],
        evidence_index=captures["evidence_index"][0],
        raw_split=captures["raw_split"][0],
        projector_source=captures["projector_source"][0],
        mapping_source=captures["mapping_source"][0],
    )
    _require_manifest_sources(sources)
    return _SourceCaptureState(
        sources=sources,
        fingerprints=MappingProxyType(
            {name: capture[1] for name, capture in captures.items()}
        ),
    )


def _capture_sources() -> ManifestSources:
    return _capture_source_state().sources


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", result) is None:
        raise RuntimeError("Git HEAD is not a full lowercase commit")
    return result


def _require_clean_git() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        check=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError("Git worktree and index must be clean")
    return _git_head()


def _capture_attempt_state() -> tuple[str, ManifestSources, EnvironmentCapture]:
    return _git_head(), _capture_sources(), capture_environment()


def _require_bundle_unchanged(bundle: SceneBundle) -> None:
    names = {
        "glb": f"{bundle.scene_id}.glb",
        "house": f"{bundle.scene_id}.house",
        "navmesh": f"{bundle.scene_id}.navmesh",
        "semantic_ply": f"{bundle.scene_id}_semantic.ply",
    }
    for role, record in bundle.files.items():
        strict_read_bytes(record.path, record.sha256, f"{role} snapshot")
        strict_read_bytes(
            SCENE_DATASET_ROOT / bundle.scene_id / names[role],
            record.sha256,
            f"{role} original",
        )


def _require_owned_directory_binding(owned: _OwnedDirectory) -> _EntryFingerprint:
    accepted = _fingerprint(os.fstat(owned.descriptor))
    current = os.stat(
        owned.name,
        dir_fd=owned.parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(accepted.st_mode)
        or accepted.st_dev != owned.fingerprint.st_dev
        or accepted.st_ino != owned.fingerprint.st_ino
        or not _same_directory_identity(owned.fingerprint, current)
    ):
        raise RuntimeError(f"owned directory binding changed: {owned.name}")
    return accepted


def _require_owned_bundle_unchanged(
    snapshots: _OwnedDirectory,
    bundle: SceneBundle,
    expected_state: _SceneSnapshotState | None = None,
) -> _SceneSnapshotState:
    names = {
        "glb": f"{bundle.scene_id}.glb",
        "house": f"{bundle.scene_id}.house",
        "navmesh": f"{bundle.scene_id}.navmesh",
        "semantic_ply": f"{bundle.scene_id}_semantic.ply",
    }
    scene_descriptor = os.open(
        bundle.scene_id,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=snapshots.descriptor,
    )
    error = None
    try:
        directory = _fingerprint(os.fstat(scene_descriptor))
        parent_entry = _fingerprint(
            os.stat(
                bundle.scene_id,
                dir_fd=snapshots.descriptor,
                follow_symlinks=False,
            )
        )
        if directory != parent_entry or not stat.S_ISDIR(directory.st_mode):
            raise RuntimeError("snapshot scene directory binding changed")
        if set(os.listdir(scene_descriptor)) != set(names.values()):
            raise RuntimeError("snapshot scene entries changed")
        fingerprints = {}
        for role, record in bundle.files.items():
            accepted, fingerprint = _strict_read_at(
                snapshots.descriptor,
                f"{bundle.scene_id}/{names[role]}",
                expected=FileRecord(record.byte_length, record.sha256),
                label=f"{role} snapshot",
                max_bytes=record.byte_length,
            )
            if len(accepted) != record.byte_length:
                raise ValueError(f"{role} snapshot byte length changed")
            fingerprints[role] = fingerprint
            strict_read_bytes(
                SCENE_DATASET_ROOT / bundle.scene_id / names[role],
                record.sha256,
                f"{role} original",
            )
        state = _SceneSnapshotState(
            directory=directory,
            files=MappingProxyType(fingerprints),
        )
        if expected_state is not None and state != expected_state:
            raise RuntimeError("snapshot scene metadata changed during Habitat lifetime")
        return state
    except BaseException as caught:
        error = caught
        raise
    finally:
        _require_closed(error, scene_descriptor)


def _close_and_require_bundle_unchanged(
    simulator: _Closable,
    bundle: SceneBundle,
) -> None:
    try:
        simulator.close()
    finally:
        _require_bundle_unchanged(bundle)


def _scene_commitment(bundle: SceneBundle) -> SceneAssetCommitment:
    required = {"glb", "house", "semantic_ply"}
    files = {
        role: SceneAssetFile(
            path=f"{bundle.scene_id}/{record.path.name}",
            required=role in required,
            byte_length=record.byte_length,
            sha256=record.sha256,
        )
        for role, record in bundle.files.items()
    }
    return SceneAssetCommitment.from_files(files)


def _require_original_asset_commitments(
    scene_assets: Mapping[str, SceneAssetCommitment],
) -> None:
    for scene_id, commitment in scene_assets.items():
        for role, record in commitment.files.items():
            relative = PurePosixPath(record.path)
            if relative.parts[0] != scene_id:
                raise ValueError("scene asset commitment path differs from scene")
            accepted = strict_read_bytes(
                SCENE_DATASET_ROOT / Path(relative),
                record.sha256,
                f"{role} committed original",
            )
            if len(accepted) != record.byte_length:
                raise ValueError(f"{role} committed original byte length differs")


def _manifest_scene_commitments(
    manifest: Mapping[str, object],
) -> Mapping[str, SceneAssetCommitment]:
    scene_assets = cast(Mapping[str, object], manifest["scene_assets"])
    raw_scenes = cast(Mapping[str, object], scene_assets["scenes"])
    commitments = {}
    for scene_id, raw_commitment in raw_scenes.items():
        commitment = cast(Mapping[str, object], raw_commitment)
        raw_files = cast(Mapping[str, object], commitment["files"])
        files = {
            role: SceneAssetFile(
                path=cast(str, cast(Mapping[str, object], value)["path"]),
                required=cast(bool, cast(Mapping[str, object], value)["required"]),
                byte_length=cast(
                    int, cast(Mapping[str, object], value)["byte_length"]
                ),
                sha256=cast(str, cast(Mapping[str, object], value)["sha256"]),
            )
            for role, value in raw_files.items()
        }
        accepted = SceneAssetCommitment.from_files(files)
        if accepted.bundle_sha256 != commitment["bundle_sha256"]:
            raise ValueError("scene bundle commitment differs from its files")
        commitments[scene_id] = accepted
    return commitments


def _require_external_sources(
    manifest: Mapping[str, object],
    sources: ManifestSources,
) -> None:
    collection = cast(Mapping[str, object], manifest["collection"])
    evidence = cast(Mapping[str, object], manifest["evidence"])
    expected = {
        "collector_source": _source_json(sources.collector_source),
        "package_source": _source_json(sources.package_source),
        "asset_roles": _source_json(sources.asset_roles),
    }
    for name, record in expected.items():
        if collection[name] != record:
            raise ValueError(f"manifest {name} differs from current source")
    evidence_expected = {
        "manifest": _source_json(sources.evidence_manifest),
        "index": _source_json(sources.evidence_index),
        "raw_split": _source_json(sources.raw_split),
        "projector_source": _source_json(sources.projector_source),
        "mapping_source": _source_json(sources.mapping_source),
    }
    for name, record in evidence_expected.items():
        if evidence[name] != record:
            raise ValueError(f"manifest evidence {name} differs from current source")
    if evidence["mapping_sha256"] != _mapping_sha256():
        raise ValueError("manifest mapping commitment differs from current mapping")
    cohort = cast(Mapping[str, object], manifest["cohort"])
    if (
        cohort["manifest"] != _source_json(sources.cohort_manifest)
        or cohort["cohort_jsonl"] != _source_json(sources.cohort_jsonl)
    ):
        raise ValueError("manifest cohort commitment differs from current cohort")


def _validate_external_raw_frame_directory(
    root: Path,
    *,
    expected_git_commit: str,
    expected_root_identity: tuple[int, int] | None = None,
) -> None:
    """Reconstruct and replay every external commitment without mutation."""

    global _FRESH_VALIDATION_IDENTITY
    if expected_root_identity is None and _FRESH_VALIDATION_IDENTITY is not None:
        expected_root_identity = _FRESH_VALIDATION_IDENTITY
        _FRESH_VALIDATION_IDENTITY = None
    parent_descriptor, root_descriptor, root_name = _open_package_root(root)
    try:
        opened_root = _fingerprint(os.fstat(root_descriptor))
        if expected_root_identity is not None and (
            opened_root.st_dev,
            opened_root.st_ino,
        ) != expected_root_identity:
            raise ValueError("raw-frame root identity differs from publisher")
        manifest_bytes, manifest_fingerprint = _read_manifest_at(root_descriptor)
        manifest = _parse_manifest_bytes(manifest_bytes)
        collection = cast(Mapping[str, object], manifest["collection"])
        if collection["git_commit"] != expected_git_commit:
            raise ValueError("manifest producer commit differs from expected commit")
        _require_unmasked_physical_gpu_zero()
        pre_input_source_state = _capture_source_state()
        inputs = load_collection_inputs()
        initial_source_state = _capture_source_state()
        if (
            initial_source_state.sources != pre_input_source_state.sources
            or initial_source_state.fingerprints
            != pre_input_source_state.fingerprints
        ):
            raise ValueError(
                "collection sources changed while loading sealed collection inputs"
            )
        initial_sources = initial_source_state.sources
        _require_external_sources(manifest, initial_sources)
        initial_environment = capture_environment()
        if _environment_json(initial_environment) != manifest["environment"]:
            raise ValueError("manifest environment differs from current environment")
        scene_assets = _manifest_scene_commitments(manifest)
        if tuple(scene_assets) != inputs.scenes:
            raise ValueError("manifest scene set differs from sealed cohort")
        _require_original_asset_commitments(scene_assets)
        ordinal = 0

        def replay_row(row: IndexRow, arrays: RawFrameArrays) -> None:
            nonlocal ordinal
            observation = inputs.observations[ordinal]
            expected_artifact = (
                f"observations/{observation.scene_id}/"
                f"{ordinal:02d}-{observation.observation_id}.npz"
            )
            if (
                row.ordinal != ordinal
                or row.observation_id != observation.observation_id
                or row.scene_id != observation.scene_id
                or row.oracle_artifact_sha256 != observation.artifact_sha256
                or row.cohort_row_sha256 != observation.cohort_row_sha256
                or row.artifact != expected_artifact
            ):
                raise ValueError(
                    "index row differs from independently sealed observation"
                )
            oracle = load_pinned_oracle_after_render(observation)
            replay_and_require_exact(arrays, observation, oracle)
            ordinal += 1

        snapshot = _validate_raw_frame_descriptor_structural(
            root_descriptor,
            expected_manifest=manifest_bytes,
            on_row=replay_row,
            expected_root_identity=expected_root_identity,
            accepted_manifest=(manifest_bytes, manifest_fingerprint),
        )
        if snapshot.root != opened_root:
            raise ValueError("raw-frame root changed during external validation")
        if ordinal != 50:
            raise ValueError("external replay did not accept all 50 observations")
        _require_original_asset_commitments(scene_assets)
        final_source_state = _capture_source_state()
        final_sources = final_source_state.sources
        _require_external_sources(manifest, final_sources)
        if (
            final_sources != initial_sources
            or final_source_state.fingerprints
            != initial_source_state.fingerprints
        ):
            raise ValueError("collection sources changed during external validation")
        final_environment = capture_environment()
        if (
            final_environment != initial_environment
            or _environment_json(final_environment) != manifest["environment"]
        ):
            raise ValueError("environment or GPU changed during external validation")
        _recapture_package_descriptor(root_descriptor, snapshot)
        _require_root_path_binding(parent_descriptor, root_name, snapshot.root)
    except BaseException as error:
        _require_closed(error, root_descriptor, parent_descriptor)
        raise
    _require_closed(None, root_descriptor, parent_descriptor)


def _absolute_lexical_path(path: Path) -> Path:
    if not isinstance(path, Path) or any(part == ".." for part in path.parts):
        raise ValueError("collection roots contain a lexical escape")
    return Path(os.path.abspath(path))


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _require_private_collection_roots(
    attempt_root: Path,
    snapshot_root: Path,
) -> None:
    attempt = _absolute_lexical_path(attempt_root)
    snapshots = _absolute_lexical_path(snapshot_root)
    forbidden = tuple(
        _absolute_lexical_path(root)
        for root in (FINAL_PACKAGE_ROOT, SMOKE_PACKAGE_ROOT)
    )
    if any(
        _is_within(candidate, root)
        for candidate in (attempt, snapshots)
        for root in forbidden
    ):
        raise ValueError("collection root is inside a forbidden package root")
    if _is_within(attempt, snapshots) or _is_within(snapshots, attempt):
        raise ValueError("collection attempt and snapshot roots overlap")
    resolved_attempt = attempt.resolve(strict=False)
    resolved_snapshots = snapshots.resolve(strict=False)
    resolved_forbidden = tuple(root.resolve(strict=False) for root in forbidden)
    if any(
        _is_within(candidate, root)
        for candidate in (resolved_attempt, resolved_snapshots)
        for root in resolved_forbidden
    ):
        raise ValueError("collection root resolves inside a forbidden package root")
    if _is_within(
        resolved_attempt, resolved_snapshots
    ) or _is_within(resolved_snapshots, resolved_attempt):
        raise ValueError("resolved collection roots overlap")

    for candidate in (attempt, snapshots):
        current = Path(candidate.anchor)
        for part in candidate.parts[1:-1]:
            current /= part
            try:
                metadata = os.stat(current, follow_symlinks=False)
            except FileNotFoundError:
                break
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("collection root has a symlinked ancestor")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("collection root ancestor is not a directory")
    if attempt_root.exists() or snapshot_root.exists():
        raise FileExistsError("collection attempt and snapshot roots must be absent")


def _open_or_create_directory_at(
    root_descriptor: int,
    parts: Sequence[str],
) -> int:
    descriptor = os.dup(root_descriptor)
    error = None
    try:
        for part in parts:
            if not part or part in {".", ".."} or "/" in part:
                raise ValueError("package directory path is not canonical")
            try:
                os.mkdir(part, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            previous = descriptor
            descriptor = child
            _require_closed(None, previous)
        return descriptor
    except BaseException as caught:
        error = caught
        _require_closed(error, descriptor)
        raise


def _write_exclusive_at(
    root_descriptor: int,
    relative: PurePosixPath,
    data: bytes,
) -> tuple[_EntryFingerprint, FileRecord]:
    directory = _open_or_create_directory_at(
        root_descriptor, relative.parts[:-1]
    )
    descriptor = -1
    error = None
    try:
        descriptor = os.open(
            relative.parts[-1],
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("new package entry is not a regular file")
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("package write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        accepted = _fingerprint(os.fstat(descriptor))
        record = FileRecord(
            byte_length=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
    except BaseException as caught:
        error = caught
        raise
    finally:
        descriptors = (directory,) if descriptor < 0 else (descriptor, directory)
        _require_closed(error, *descriptors)
    return accepted, record


def _fsync_directories_postorder(
    descriptor: int,
) -> tuple[_EntryFingerprint, Mapping[str, _EntryFingerprint]]:
    directories: dict[str, _EntryFingerprint] = {}
    for name in sorted(os.listdir(descriptor)):
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            error = None
            try:
                if _fingerprint(metadata) != _fingerprint(os.fstat(child)):
                    raise RuntimeError(
                        "package directory changed during durability walk"
                    )
                child_root, descendants = _fsync_directories_postorder(child)
                after = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    child_root != _fingerprint(metadata)
                    or child_root != _fingerprint(after)
                ):
                    raise RuntimeError(
                        "package directory changed during durability walk"
                    )
                directories[name] = child_root
                directories.update(
                    {
                        f"{name}/{relative}": fingerprint
                        for relative, fingerprint in descendants.items()
                    }
                )
            except BaseException as caught:
                error = caught
                raise
            finally:
                _require_closed(error, child)
        elif not stat.S_ISREG(metadata.st_mode):
            raise ValueError("package durability walk found a non-regular entry")
    before_fsync = _fingerprint(os.fstat(descriptor))
    os.fsync(descriptor)
    after_fsync = _fingerprint(os.fstat(descriptor))
    if after_fsync != before_fsync:
        raise RuntimeError("package directory changed during durability fsync")
    return after_fsync, MappingProxyType(directories)


def collect_attempt(
    attempt: _OwnedDirectory,
    snapshots: _OwnedDirectory,
) -> _CollectedAttempt:
    """Collect all sealed rows into one unpublished private attempt."""

    if (
        not _same_directory_identity(
            attempt.fingerprint, os.fstat(attempt.descriptor)
        )
        or not _same_directory_identity(
            snapshots.fingerprint, os.fstat(snapshots.descriptor)
        )
    ):
        raise RuntimeError("publisher-owned collection roots changed")
    initial_state = _capture_attempt_state()
    inputs = load_collection_inputs()
    grouped = [
        (scene_id, tuple(observations))
        for scene_id, observations in groupby(
            inputs.observations, key=lambda item: item.scene_id
        )
    ]
    if tuple(scene for scene, _ in grouped) != inputs.scenes:
        raise ValueError("sealed observations are not grouped in lexical scene order")
    rows = []
    scene_assets = {}
    written_files: dict[str, tuple[_EntryFingerprint, FileRecord]] = {}
    try:
        ordinal = 0
        for scene_id, observations in grouped:
            _require_owned_directory_binding(snapshots)
            bundle = _snapshot_owned_scene_bundle(scene_id, snapshots)
            quiescent_snapshot = _require_owned_directory_binding(snapshots)
            quiescent_scene = _require_owned_bundle_unchanged(snapshots, bundle)
            scene_assets[scene_id] = _scene_commitment(bundle)
            simulator = None
            scene_error = None
            try:
                if _require_owned_directory_binding(snapshots) != quiescent_snapshot:
                    raise RuntimeError("snapshot root changed before Habitat build")
                simulator = build_scene_simulator(bundle)
                _require_owned_bundle_unchanged(
                    snapshots, bundle, quiescent_scene
                )
                if _require_owned_directory_binding(snapshots) != quiescent_snapshot:
                    raise RuntimeError("snapshot root changed during Habitat build")
                for observation in observations:
                    arrays = render_raw_frame_artifact(simulator, observation)
                    encoded = encode_raw_frame_npz(arrays)
                    relative = (
                        Path("observations")
                        / scene_id
                        / f"{ordinal:02d}-{observation.observation_id}.npz"
                    )
                    written_files[relative.as_posix()] = _write_exclusive_at(
                        attempt.descriptor,
                        PurePosixPath(relative.as_posix()),
                        encoded.data,
                    )
                    npz_record = written_files[relative.as_posix()][1]
                    accepted, _ = _strict_read_at(
                        attempt.descriptor,
                        relative.as_posix(),
                        expected=npz_record,
                        label="raw-frame artifact",
                        max_bytes=_MAX_ARTIFACT_BYTES,
                    )
                    parsed = parse_raw_frame_npz_bytes(
                        accepted,
                        expected_members=encoded.members,
                        expected_npz=npz_record,
                    )
                    oracle = load_pinned_oracle_after_render(observation)
                    replay_and_require_exact(parsed.arrays, observation, oracle)
                    rows.append(
                        IndexRow(
                            artifact=relative.as_posix(),
                            cohort_row_sha256=observation.cohort_row_sha256,
                            members=parsed.members,
                            npz=npz_record,
                            observation_id=observation.observation_id,
                            oracle_artifact_sha256=observation.artifact_sha256,
                            ordinal=ordinal,
                            scene_id=scene_id,
                        )
                    )
                    ordinal += 1
            except BaseException as caught:
                scene_error = caught
            secondary_failures = []
            if simulator is not None:
                try:
                    simulator.close()
                except BaseException as caught:
                    secondary_failures.append(caught)
            try:
                _require_owned_bundle_unchanged(
                    snapshots, bundle, quiescent_scene
                )
            except BaseException as caught:
                secondary_failures.append(caught)
            try:
                current_snapshot = _require_owned_directory_binding(snapshots)
                if current_snapshot != quiescent_snapshot:
                    raise RuntimeError(
                        "snapshot root changed during Habitat lifetime"
                    )
            except BaseException as caught:
                secondary_failures.append(caught)
            if secondary_failures:
                detail = "; ".join(str(error) for error in secondary_failures)
                raise RuntimeError(
                    f"scene finalization failed: {detail}"
                ) from scene_error
            if scene_error is not None:
                raise scene_error
        if len(rows) != 50 or _capture_attempt_state() != initial_state:
            raise ValueError("collection commit, sources, environment, or GPU changed")
        _require_original_asset_commitments(scene_assets)
        _remove_owned_root(snapshots)
        index_bytes = canonical_index_bytes(rows)
        written_files["index.jsonl"] = _write_exclusive_at(
            attempt.descriptor,
            PurePosixPath("index.jsonl"),
            index_bytes,
        )
        git_commit, sources, environment = initial_state
        manifest = build_manifest(
            git_commit=git_commit,
            sources=sources,
            scene_assets=scene_assets,
            environment=environment,
            index_bytes=index_bytes,
            rows=rows,
        )
        written_files["manifest.json"] = _write_exclusive_at(
            attempt.descriptor,
            PurePosixPath("manifest.json"),
            manifest,
        )
        durable_root, durable_directories = _fsync_directories_postorder(
            attempt.descriptor
        )
        structural_snapshot = _validate_raw_frame_descriptor_structural(
            attempt.descriptor,
            expected_manifest=manifest,
            expected_root_identity=(
                attempt.fingerprint.st_dev,
                attempt.fingerprint.st_ino,
            ),
            expected_written_files=written_files,
            expected_durable_root=durable_root,
            expected_durable_directories=durable_directories,
        )
        if _capture_attempt_state() != initial_state:
            raise ValueError("collection commit, sources, environment, or GPU changed")
        return _CollectedAttempt(manifest, structural_snapshot)
    except BaseException:
        raise


def _smoke_report(observation: CollectionObservation) -> bytes:
    return (
        json.dumps(
            {
                "control": "PASS",
                "observation_id": observation.observation_id,
                "replay": "PASS",
                "scene_id": observation.scene_id,
                "schema_version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _cleanup_smoke_paths(
    staging: Path | None,
    snapshots: Path | None,
) -> None:
    first_error = None
    for path in (staging, snapshots):
        if path is None:
            continue
        try:
            shutil.rmtree(path)
        except OSError as error:
            if first_error is None:
                first_error = error
    try:
        SMOKE_PACKAGE_ROOT.rmdir()
    except OSError:
        pass
    if first_error is not None:
        raise RuntimeError("smoke cleanup failed") from first_error


def run_first_row_smoke() -> bytes:
    """Render, strictly reload, replay, report, and remove one private smoke."""

    inputs = load_collection_inputs()
    observation = inputs.observations[0]
    staging = None
    snapshots = None
    try:
        SMOKE_PACKAGE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f"{os.getpid()}-", dir=SMOKE_PACKAGE_ROOT)
        )
        staging.chmod(0o700)
        snapshots = Path(
            tempfile.mkdtemp(
                prefix=".r2r-val-unseen-50-raw-v1-snapshots-",
                dir=SMOKE_PACKAGE_ROOT.parent,
            )
        )
        snapshots.chmod(0o700)
        bundle = snapshot_scene_bundle(observation.scene_id, snapshots)
        simulator = build_scene_simulator(bundle)
        try:
            arrays = render_raw_frame_artifact(simulator, observation)
            encoded = encode_raw_frame_npz(arrays)
            artifact = staging / f"{observation.observation_id}.npz"
            artifact.write_bytes(encoded.data)
            accepted = strict_read_bytes(
                artifact,
                hashlib.sha256(encoded.data).hexdigest(),
                "smoke raw artifact",
            )
            parsed = parse_raw_frame_npz_bytes(
                accepted,
                expected_members=encoded.members,
            )
            oracle = load_pinned_oracle_after_render(observation)
            replay_and_require_exact(parsed.arrays, observation, oracle)
        finally:
            simulator.close()
        report = _smoke_report(observation)
        sys.stdout.buffer.write(report)
        sys.stdout.buffer.flush()
        return report
    finally:
        _cleanup_smoke_paths(staging, snapshots)


def _require_variant_assets(
    bundle: SceneBundle,
    omitted_role: str | None,
) -> None:
    names = {
        "glb": f"{bundle.scene_id}.glb",
        "house": f"{bundle.scene_id}.house",
        "navmesh": f"{bundle.scene_id}.navmesh",
        "semantic_ply": f"{bundle.scene_id}_semantic.ply",
    }
    for role, record in bundle.files.items():
        if role == omitted_role:
            if record.path.exists():
                raise ValueError("omitted snapshot asset unexpectedly exists")
        else:
            strict_read_bytes(record.path, record.sha256, f"{role} smoke snapshot")
        strict_read_bytes(
            SCENE_DATASET_ROOT / bundle.scene_id / names[role],
            record.sha256,
            f"{role} smoke original",
        )


def _render_worker(
    connection: Connection,
    scene_id: str,
    serialized_files: Tuple[Tuple[str, str, int, str], ...],
    observation: CollectionObservation,
) -> None:
    bundle = SceneBundle(
        scene_id=scene_id,
        files=MappingProxyType(
            {
                role: SnapshotFile(
                    path=Path(path),
                    byte_length=byte_length,
                    sha256=sha256,
                )
                for role, path, byte_length, sha256 in serialized_files
            }
        ),
    )
    simulator = None
    classification_failure = False
    try:
        try:
            simulator = build_scene_simulator(bundle)
            arrays = render_raw_frame_artifact(simulator, observation)
        except Exception:
            classification_failure = True
        finally:
            if simulator is not None:
                simulator.close()
        if classification_failure:
            connection.send(("classification_failure", None))
        else:
            connection.send(("success", arrays))
    except BaseException as error:
        connection.send(
            (
                "infrastructure_error",
                f"{type(error).__name__}: {error}",
            )
        )
    finally:
        connection.close()


def _render_in_spawned_process(
    bundle: SceneBundle,
    observation: CollectionObservation,
    omitted_role: str | None,
    *,
    worker: Callable[..., None] = _render_worker,
) -> RawFrameArrays | None:
    serialized_files = tuple(
        (
            role,
            record.path.as_posix(),
            record.byte_length,
            record.sha256,
        )
        for role, record in sorted(bundle.files.items())
    )
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=worker,
        args=(send, bundle.scene_id, serialized_files, observation),
    )
    try:
        process.start()
    except BaseException:
        receive.close()
        send.close()
        raise
    send.close()
    message = None
    try:
        try:
            message = receive.recv()
        except EOFError:
            pass
    finally:
        receive.close()
        process.join()
    if process.exitcode != 0 or message is None:
        if omitted_role is not None:
            return None
        raise RuntimeError("all-assets control render subprocess terminated")
    if (
        not isinstance(message, tuple)
        or len(message) != 2
        or message[0]
        not in {"classification_failure", "infrastructure_error", "success"}
    ):
        raise RuntimeError("render subprocess returned an invalid result")
    status, payload = message
    if status == "classification_failure":
        if omitted_role is not None:
            return None
        raise RuntimeError("all-assets control render failed")
    if status == "infrastructure_error":
        raise RuntimeError(f"render subprocess infrastructure failure: {payload}")
    if not isinstance(payload, RawFrameArrays):
        raise RuntimeError("render subprocess returned invalid arrays")
    return payload


def _run_smoke_variant(
    observation: CollectionObservation,
    staging: Path,
    snapshots: Path,
    omitted_role: str | None,
) -> bool:
    variant_name = omitted_role if omitted_role is not None else "all"
    private_assets = snapshots / f"{observation.scene_id}-{variant_name}"
    private_assets.mkdir(mode=0o700)
    bundle = snapshot_scene_bundle(observation.scene_id, private_assets)
    if omitted_role is not None:
        bundle.files[omitted_role].path.unlink()
    try:
        arrays = _render_in_spawned_process(bundle, observation, omitted_role)
        if arrays is None:
            return False
        encoded = encode_raw_frame_npz(arrays)
        artifact = staging / f"{observation.scene_id}-{variant_name}.npz"
        artifact.write_bytes(encoded.data)
        record = FileRecord(
            byte_length=len(encoded.data),
            sha256=hashlib.sha256(encoded.data).hexdigest(),
        )
        accepted = strict_read_bytes(
            artifact, record.sha256, "first-per-scene smoke artifact"
        )
        parsed = parse_raw_frame_npz_bytes(
            accepted,
            expected_members=encoded.members,
            expected_npz=record,
        )
        oracle = load_pinned_oracle_after_render(observation)
        try:
            replay_and_require_exact(parsed.arrays, observation, oracle)
        except Exception:
            if omitted_role is None:
                raise
            return False
    finally:
        _require_variant_assets(bundle, omitted_role)
    return True


def run_first_per_scene_smoke() -> bytes:
    """Classify scene asset roles through 55 disposable first-row variants."""

    inputs = load_collection_inputs()
    first_by_scene = {}
    for observation in inputs.observations:
        first_by_scene.setdefault(observation.scene_id, observation)
    if tuple(first_by_scene) != inputs.scenes or len(first_by_scene) != 11:
        raise ValueError("first-per-scene smoke selection differs from sealed scenes")
    staging = None
    snapshots = None
    try:
        SMOKE_PACKAGE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f"{os.getpid()}-per-scene-", dir=SMOKE_PACKAGE_ROOT)
        )
        staging.chmod(0o700)
        snapshots = Path(
            tempfile.mkdtemp(
                prefix=".r2r-val-unseen-50-raw-v1-role-snapshots-",
                dir=SMOKE_PACKAGE_ROOT.parent,
            )
        )
        snapshots.chmod(0o700)
        omission_passes = {
            role: []
            for role in ("glb", "house", "navmesh", "semantic_ply")
        }
        for observation in first_by_scene.values():
            if not _run_smoke_variant(observation, staging, snapshots, None):
                raise ValueError("all-assets control did not pass")
            for role in omission_passes:
                omission_passes[role].append(
                    _run_smoke_variant(
                        observation,
                        staging,
                        snapshots,
                        role,
                    )
                )
        classification = {
            "algorithm": "single-role-omission-first-row-per-scene-v1",
            "auxiliary": sorted(
                role for role, passes in omission_passes.items() if all(passes)
            ),
            "required": sorted(
                role for role, passes in omission_passes.items() if not all(passes)
            ),
            "schema_version": 1,
        }
        if classification != _ASSET_ROLE_VALUE:
            raise ValueError("live scene asset role classification differs from plan")
        report = (
            json.dumps(
                {
                    "classification": classification,
                    "control_passed_count": 11,
                    "scene_count": 11,
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        sys.stdout.buffer.write(report)
        sys.stdout.buffer.flush()
        return report
    finally:
        _cleanup_smoke_paths(staging, snapshots)


def _open_publication_parent() -> int:
    roots = (
        FINAL_PACKAGE_ROOT,
        STAGING_PACKAGE_ROOT,
        SNAPSHOT_PACKAGE_ROOT,
        SMOKE_PACKAGE_ROOT,
    )
    if any(
        not isinstance(root, Path)
        or any(part in {"", ".", ".."} for part in root.parts)
        for root in roots
    ):
        raise ValueError("fixed publication roots are not canonical")
    absolute_parents = {
        _absolute_lexical_path(root).parent for root in roots
    }
    if len(absolute_parents) != 1:
        raise ValueError("fixed publication roots must be siblings")
    parent = absolute_parents.pop()
    grandparent, descriptor, _ = _open_package_root(parent)
    try:
        _require_closed(None, grandparent)
    except BaseException as error:
        _require_closed(error, descriptor)
        raise
    return descriptor


def _require_absent_at(parent_descriptor: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(f"fixed publication root already exists: {name}")


def _remove_tree_contents(descriptor: int) -> None:
    for name in os.listdir(descriptor):
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            error = None
            try:
                child_fingerprint = _fingerprint(os.fstat(child))
                if _fingerprint(metadata) != child_fingerprint:
                    raise RuntimeError("owned cleanup entry changed while opening")
                _remove_tree_contents(child)
                current = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    not _same_directory_identity(child_fingerprint, os.fstat(child))
                    or not _same_directory_identity(child_fingerprint, current)
                ):
                    raise RuntimeError("owned cleanup directory was replaced")
                os.rmdir(name, dir_fd=descriptor)
            except BaseException as caught:
                error = caught
                raise
            finally:
                _require_closed(error, child)
        else:
            if stat.S_ISLNK(metadata.st_mode):
                current = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if _fingerprint(current) != _fingerprint(metadata):
                    raise RuntimeError("owned cleanup symbolic link was replaced")
                os.unlink(name, dir_fd=descriptor)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("owned cleanup found an unexpected entry type")
            child = os.open(
                name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            error = None
            try:
                child_fingerprint = _fingerprint(os.fstat(child))
                current = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    _fingerprint(metadata) != child_fingerprint
                    or _fingerprint(current) != child_fingerprint
                ):
                    raise RuntimeError("owned cleanup file was replaced")
                os.unlink(name, dir_fd=descriptor)
            except BaseException as caught:
                error = caught
                raise
            finally:
                _require_closed(error, child)


def _create_owned_directory(
    parent_descriptor: int,
    path: Path,
) -> _OwnedDirectory:
    os.mkdir(path.name, 0o700, dir_fd=parent_descriptor)
    descriptor = -1
    fingerprint = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        fingerprint = _fingerprint(os.fstat(descriptor))
        metadata = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _fingerprint(metadata) != fingerprint:
            raise RuntimeError("owned directory changed while being created")
    except BaseException as error:
        if fingerprint is None:
            failures = _close_many(descriptor) if descriptor >= 0 else ()
            detail = "; ".join(str(item) for item in failures)
            raise RuntimeError(
                "owned directory creation failed before its cleanup token "
                f"was trustworthy; root preserved{': ' + detail if detail else ''}"
            ) from error
        owned = _OwnedDirectory(
            path=path,
            name=path.name,
            parent_descriptor=parent_descriptor,
            descriptor=descriptor,
            fingerprint=fingerprint,
        )
        try:
            _remove_owned_root(owned)
        except BaseException as cleanup_error:
            raise RuntimeError(
                f"owned directory creation and cleanup failed: {cleanup_error}"
            ) from error
        raise
    return _OwnedDirectory(
        path=path,
        name=path.name,
        parent_descriptor=parent_descriptor,
        descriptor=descriptor,
        fingerprint=fingerprint,
    )


def _remove_owned_root(owned: _OwnedDirectory) -> None:
    if owned.descriptor < 0:
        return
    try:
        initial = os.stat(
            owned.name,
            dir_fd=owned.parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        descriptor = owned.descriptor
        owned.descriptor = -1
        _require_closed(None, descriptor)
        return
    error = None
    try:
        if not _same_directory_identity(
            owned.fingerprint, os.fstat(owned.descriptor)
        ):
            raise RuntimeError(f"owned cleanup descriptor changed: {owned.name}")
        if not _same_directory_identity(owned.fingerprint, initial):
            raise RuntimeError(f"owned cleanup root was replaced: {owned.name}")
        _remove_tree_contents(owned.descriptor)
        metadata = os.stat(
            owned.name,
            dir_fd=owned.parent_descriptor,
            follow_symlinks=False,
        )
        if not _same_directory_identity(owned.fingerprint, metadata):
            raise RuntimeError(f"owned cleanup root was replaced: {owned.name}")
        os.rmdir(owned.name, dir_fd=owned.parent_descriptor)
    except BaseException as caught:
        error = caught
    descriptor = owned.descriptor
    owned.descriptor = -1
    _require_closed(error, descriptor)
    if error is not None:
        raise error


def _rename_noreplace(
    parent_descriptor: int,
    source: str,
    destination: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = library.renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source),
        parent_descriptor,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError("final raw-frame package already exists")
        raise OSError(error_number, os.strerror(error_number))


_FRESH_VALIDATION_IDENTITY: tuple[int, int] | None = None


def _run_fresh_validation(
    root: str,
    expected_git_commit: str,
    device: str,
    inode: str,
) -> None:
    global _FRESH_VALIDATION_IDENTITY
    if _FRESH_VALIDATION_IDENTITY is not None:
        raise RuntimeError("fresh validation identity channel is already occupied")
    _FRESH_VALIDATION_IDENTITY = (int(device), int(inode))
    from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
        validate_raw_frame_directory,
    )

    validate_raw_frame_directory(
        Path(root),
        expected_git_commit=expected_git_commit,
    )
    if _FRESH_VALIDATION_IDENTITY is not None:
        raise RuntimeError("fresh validation did not consume its identity proof")


def _fresh_validate_subprocess(
    root: Path,
    expected_git_commit: str,
    identity: tuple[int, int],
) -> None:
    command = (
        "from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames "
        "import _run_fresh_validation as run;"
        "import sys;"
        "run(*sys.argv[1:])"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            root.as_posix(),
            expected_git_commit,
            str(identity[0]),
            str(identity[1]),
        ],
        check=True,
    )


class RawFrameArgs(Tap):
    smoke: Literal["none", "first-row", "first-per-scene"] = "none"


def publish_raw_frame_directory() -> bytes:
    """Build, independently validate, and atomically publish the fixed package."""

    parent_descriptor = _open_publication_parent()
    staging = None
    snapshots = None
    renamed = False
    publication_error = None
    staging_name = STAGING_PACKAGE_ROOT.name
    snapshot_name = SNAPSHOT_PACKAGE_ROOT.name
    final_name = FINAL_PACKAGE_ROOT.name
    smoke_name = SMOKE_PACKAGE_ROOT.name
    try:
        for name in (final_name, staging_name, snapshot_name, smoke_name):
            _require_absent_at(parent_descriptor, name)
        _require_little_endian()
        _require_unmasked_physical_gpu_zero()
        initial_head = _require_clean_git()
        initial_sources = _capture_source_state()
        initial_environment = capture_environment()
        staging = _create_owned_directory(
            parent_descriptor, STAGING_PACKAGE_ROOT
        )
        snapshots = _create_owned_directory(
            parent_descriptor, SNAPSHOT_PACKAGE_ROOT
        )
        collected = collect_attempt(staging, snapshots)
        manifest_bytes = collected.manifest
        staging_identity = (
            staging.fingerprint.st_dev,
            staging.fingerprint.st_ino,
        )
        _fresh_validate_subprocess(
            STAGING_PACKAGE_ROOT,
            initial_head,
            staging_identity,
        )
        if _require_clean_git() != initial_head:
            raise RuntimeError("Git HEAD changed during publication")
        final_sources = _capture_source_state()
        if final_sources != initial_sources:
            raise RuntimeError("sources changed during publication")
        final_environment = capture_environment()
        if final_environment != initial_environment:
            raise RuntimeError("environment or GPU changed during publication")
        manifest = _parse_manifest_bytes(manifest_bytes)
        _require_external_sources(manifest, final_sources.sources)
        if _environment_json(final_environment) != manifest["environment"]:
            raise RuntimeError("staging environment differs during publication")
        _require_original_asset_commitments(
            _manifest_scene_commitments(manifest)
        )
        _require_absent_at(parent_descriptor, final_name)
        _require_absent_at(parent_descriptor, smoke_name)
        _require_absent_at(parent_descriptor, snapshot_name)
        if not _same_directory_identity(
            staging.fingerprint, os.fstat(staging.descriptor)
        ):
            raise RuntimeError("staging root was replaced after validation")
        _recapture_package_descriptor(
            staging.descriptor,
            collected.structural_snapshot,
        )
        _require_root_path_binding(
            parent_descriptor,
            staging_name,
            collected.structural_snapshot.root,
        )
        _rename_noreplace(parent_descriptor, staging_name, final_name)
        renamed = True
        final_metadata = os.stat(
            final_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not _same_directory_identity(staging.fingerprint, final_metadata):
            raise RuntimeError("published final root identity differs from staging")
        os.fsync(parent_descriptor)
        descriptor = staging.descriptor
        staging.descriptor = -1
        _require_closed(None, descriptor)
        return manifest_bytes
    except BaseException as error:
        publication_error = error
        cleanup_failures = []
        if renamed and staging is not None and staging.descriptor >= 0:
            descriptor = staging.descriptor
            staging.descriptor = -1
            try:
                _require_closed(None, descriptor)
            except BaseException as cleanup_error:
                cleanup_failures.append(cleanup_error)
        elif not renamed:
            for owned in (staging, snapshots):
                if owned is None:
                    continue
                try:
                    _remove_owned_root(owned)
                except BaseException as cleanup_error:
                    cleanup_failures.append(cleanup_error)
        if cleanup_failures:
            combined = _OperationCleanupError(
                "publication failed and owned cleanup also failed",
                error,
                cleanup_failures,
            )
            publication_error = combined
            raise combined from error
        raise
    finally:
        _require_closed(publication_error, parent_descriptor)


def main(argv: Sequence[str] | None = None) -> bytes:
    args = RawFrameArgs().parse_args(argv)
    if args.smoke == "first-row":
        return run_first_row_smoke()
    if args.smoke == "first-per-scene":
        return run_first_per_scene_smoke()
    return publish_raw_frame_directory()


if __name__ == "__main__":
    main()
