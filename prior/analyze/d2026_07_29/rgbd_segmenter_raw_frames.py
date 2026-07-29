"""Pinned source binding and private scene snapshots for the RGB-D experiment."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import (
    List,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
from tap import Tap

from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    RawFrameArrays,
    encode_raw_frame_npz,
    parse_raw_frame_npz_bytes,
    strict_read_bytes,
)
from prior.constants import MAPPED_OBJECT_NAMES, OBJECT_MAPPING, REGION_MAPPING
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
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]:
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
    if len(rows) != 50 or len({cast(str, row["observation_id"]) for row in rows}) != 50:
        raise ValueError("cohort observation population drift")
    scenes = sorted({cast(str, row["scene_id"]) for row in rows})
    if len(scenes) != 11:
        raise ValueError("cohort scene population drift")
    return manifest, tuple(rows)


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
    cohort = _parse_cohort(cohort_manifest, cohort_rows)[1]
    evidence = _parse_evidence_metadata(evidence_manifest, evidence_index)
    observations = []
    for row in cohort:
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
            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a directory")
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise ValueError(f"{label} must be a real directory") from error
    except BaseException:
        os.close(descriptor)
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
    except BaseException:
        if created:
            try:
                os.unlink(name, dir_fd=destination_directory)
                os.fsync(destination_directory)
            except OSError as error:
                raise RuntimeError(f"asset cleanup failed: {name}") from error
        raise
    finally:
        if destination >= 0:
            os.close(destination)
        os.close(source)


def snapshot_scene_bundle(scene_id: str, private_sibling: Path) -> SceneBundle:
    """Copy exactly the four immutable simulator assets to a new private sibling."""

    scene_id = _require_scene_id(scene_id, "scene_id")
    if not isinstance(private_sibling, Path):
        raise ValueError("private snapshot root is invalid")
    if any(
        _is_equal_or_descendant(private_sibling, forbidden)
        for forbidden in (FINAL_PACKAGE_ROOT, SMOKE_PACKAGE_ROOT)
    ):
        raise ValueError("private snapshot root cannot be package staging")
    source_root = _open_real_directory(SCENE_DATASET_ROOT, "scene dataset root")
    private_root = _open_real_directory(private_sibling, "private snapshot root")
    scene_source = -1
    scene_destination = -1
    scene_created = False
    copied_names: list[str] = []
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
    except BaseException:
        if scene_created:
            try:
                for name in reversed(copied_names):
                    os.unlink(name, dir_fd=scene_destination)
                if scene_destination >= 0:
                    os.fsync(scene_destination)
                    os.close(scene_destination)
                    scene_destination = -1
                os.rmdir(scene_id, dir_fd=private_root)
                os.fsync(private_root)
            except OSError as error:
                raise RuntimeError(f"snapshot cleanup failed: {scene_id}") from error
        raise
    finally:
        if scene_destination >= 0:
            os.close(scene_destination)
        if scene_source >= 0:
            os.close(scene_source)
        os.close(private_root)
        os.close(source_root)


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
    config.freeze()
    return cast(
        _Simulator,
        make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR),
    )


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
    center = np.asarray(aabb.center, dtype="<f8")
    sizes = np.asarray(aabb.sizes, dtype="<f8")
    minimum = center - sizes / 2.0
    if (
        center.shape != (3,)
        or sizes.shape != (3,)
        or not np.isfinite(minimum).all()
    ):
        raise ValueError("semantic AABB must contain finite float64 vectors")
    return minimum


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


class RawFrameArgs(Tap):
    smoke: Literal["none", "first-row", "first-per-scene"] = "none"


def main(argv: Sequence[str] | None = None) -> bytes:
    args = RawFrameArgs().parse_args(argv)
    if args.smoke == "first-row":
        return run_first_row_smoke()
    if args.smoke == "first-per-scene":
        raise ValueError("first-per-scene smoke is not implemented yet")
    raise ValueError("full raw-frame collection is not implemented yet")


if __name__ == "__main__":
    main()
