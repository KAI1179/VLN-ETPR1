"""Typed target-free cardinal selector and soft-raster aggregation core."""

from __future__ import annotations

import csv
import ctypes
from dataclasses import asdict, dataclass, replace
from enum import Enum
import errno
import gzip
from hashlib import sha256
from io import BytesIO, StringIO
import json
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Dict, Optional, Sequence, Tuple, cast
from zipfile import BadZipFile, ZipFile

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tap import Tap

from prior import R2R_DIR
from prior.analyze.llm_grid_registration import (
    WarpedGrid,
    direction_cosine,
    rotate_direction_vectors,
    score_warped_grid,
    warp_grid_about_pivot,
)
from prior.constants import CELL_SIZE
from prior.llm_grid_samples import downsample_grid
from vlnce_baselines.models.etp_llm.llm_grid_train import (
    GRID_SCALE,
    LLMGridValidationError,
    load_llm_grid_examples,
    parse_grid_text,
)
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_prediction_path,
    llm_navigation_split_dir,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import cognitive_map_cache_path


ANGLE_ORDER = (0.0, 90.0, 180.0, 270.0)
_DIRECTION_TOLERANCE = 1e-4
_HEADING_TIE_TOLERANCE = 1e-6
_OBJECT_CHANNEL_COUNT = 27
_ASSIGNMENT_HEADER = (
    "scene_id",
    "example_id",
    "schema_valid",
    "heading_bin_degrees",
    "primary_angle_degrees",
    "global_angle_degrees",
    "direct_angle_degrees",
    "direct_margin",
)
DEVELOPMENT_MAPPING_SCORES_HEADER = (
    "candidate_kind",
    "mapping_sign",
    "heading_offset_degrees",
    "global_angle_degrees",
    "episode_count",
    "valid_count",
    "all_mean_iou",
    "object_mean_iou",
    "region_mean_iou",
    "selected",
)
TEST_ASSIGNMENTS_HEADER = _ASSIGNMENT_HEADER
TEST_ANGLE_SCORES_HEADER = (
    "scene_id",
    "example_id",
    "angle_degrees",
    "all_iou",
    "object_iou",
    "region_iou",
    "direction_cosine",
    "predicted_support",
    "target_support",
    "in_frame_support",
    "out_of_frame_support",
    "union",
)
TEST_EPISODE_SCORES_HEADER = (
    "scene_id",
    "example_id",
    "schema_valid",
    "heading_bin_degrees",
    "primary_angle_degrees",
    "global_angle_degrees",
    "direct_angle_degrees",
    "identity_iou",
    "primary_iou",
    "global_iou",
    "direct_iou",
    "random_expected_iou",
    "soft_identity_iou",
    "soft_aggregate_iou",
    "oracle_iou",
    "identity_object_iou",
    "primary_object_iou",
    "identity_region_iou",
    "primary_region_iou",
    "soft_identity_object_iou",
    "soft_aggregate_object_iou",
    "soft_identity_region_iou",
    "soft_aggregate_region_iou",
    "primary_predicted_support",
    "primary_target_support",
    "primary_in_frame_support",
    "primary_out_of_frame_support",
    "primary_union",
    "soft_prediction_mass",
    "soft_target_mass",
    "soft_intersection_mass",
    "soft_union_mass",
    "primary_direction_cosine",
    "direct_direction_cosine",
)
SELECTOR_LOCK_KEYS = (
    "schema_version",
    "chosen_mapping",
    "chosen_global_angle",
    "development",
    "protocol",
)
SUMMARY_KEYS = (
    "schema_version",
    "population",
    "means",
    "contrasts",
    "semantic_families",
    "cell_metrics",
    "support",
    "directions",
    "angles",
    "cost",
    "oracle_gain",
    "audit",
    "decision",
)
BOOTSTRAP_KEYS = (
    "schema_version",
    "seed",
    "repetitions",
    "scene_ids",
    "intervals",
    "leave_one_scene_out",
)
MANIFEST_KEYS = (
    "schema_version",
    "sources",
    "git_commit",
    "protocol",
    "population",
    "schemas",
    "artifacts",
)
_SCHEMA_VERSION = "llm-grid-target-free-cardinal-v3"
_SOURCE_SCHEMA_VERSION = "llm-grid-target-free-source-fingerprint-v3"
_DEFAULT_CACHE_DIR = Path("data/llm_navigation")
_DEFAULT_CACHE_MODEL_KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2"
_DEFAULT_COGNITIVE_MAP_NAMESPACE = "gt.legacy.r1p5.direction5.v1"
_DEFAULT_OUTPUT_DIR = Path(
    "outputs/llm_grid_analysis/target_free_cardinal_selector_r2r_epoch2"
)
_DEFAULT_BOOTSTRAP_SEED = 42
_DEFAULT_BOOTSTRAP_REPETITIONS = 10_000
_GATE_THRESHOLD = 0.01
_DEVELOPMENT_SOURCE_SHA256 = (
    "1d3eb25eb7583a2c6373f430119da25ef71472d2d065697e673e2f70c307d146"
)
_TEST_SOURCE_SHA256 = (
    "31e7c5f8d186e75222b12f1aa8862b16fa9904c75221a0fd55cfba61f93fa8ce"
)
_SOURCE_PHASE_CONTRACTS = {
    "development_all_sources": (
        3893,
        "7b0f8ce883a4929679ea51627726e97fc318f9785cb2794afb5908f002a860cc",
    ),
    "test_assignment_sources": (
        5520,
        "e6065ad984f05bd42950fa52855310bc7364421d424d4e29704e508bda447d10",
    ),
    "test_target_sources": (
        3678,
        "62307358663497a3bfb98220b4d2a875fe50fb0922ba020186cc6be6cbcf89ef",
    ),
}
_COMBINED_SOURCE_SHA256 = (
    "5e8702088364ee294ad0876486d403fe1b68283897bff2468369167c262bf942"
)


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

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must be non-empty")
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("example_id must be non-empty")


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

    def __post_init__(self) -> None:
        if not isinstance(self.key, EpisodeKey):
            raise ValueError("key must be an EpisodeKey")
        _require_boolean(self.schema_valid, "schema_valid")
        heading = _require_cardinal_angle(self.heading_bin_degrees, "heading_bin_degrees")
        primary = _require_cardinal_angle(self.primary_angle_degrees, "primary_angle_degrees")
        global_angle = _require_cardinal_angle(
            self.global_angle_degrees, "global_angle_degrees"
        )
        direct = _require_cardinal_angle(self.direct_angle_degrees, "direct_angle_degrees")
        margin = _require_finite_float(self.direct_margin, "direct_margin")
        if not 0.0 <= margin <= 2.0:
            raise ValueError("direct_margin must be within [0, 2]")
        object.__setattr__(self, "heading_bin_degrees", heading)
        object.__setattr__(self, "primary_angle_degrees", primary)
        object.__setattr__(self, "global_angle_degrees", global_angle)
        object.__setattr__(self, "direct_angle_degrees", direct)
        object.__setattr__(self, "direct_margin", margin)


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
    return _soft_score(hypotheses, target)


def _soft_score(
    hypotheses: tuple[WarpedGrid, ...], target: NDArray[np.bool_]
) -> SoftRasterScore:
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


def _family_soft_score(
    hypotheses: tuple[WarpedGrid, ...],
    target: NDArray[np.bool_],
    start: int,
    end: int,
) -> SoftRasterScore:
    family_hypotheses = tuple(
        WarpedGrid(
            hypothesis.grid[start:end],
            hypothesis.bounds,
            int(np.count_nonzero(hypothesis.grid[start:end])),
        )
        for hypothesis in hypotheses
    )
    return _soft_score(family_hypotheses, target[start:end])


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    return 0.0 if result == 0.0 else result


def _require_cardinal_angle(value: float, name: str) -> float:
    angle = _require_finite_float(value, name)
    if angle not in ANGLE_ORDER:
        raise ValueError(f"{name} must be a declared cardinal angle")
    return angle


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class PopulationContract:
    split: str
    episodes: int
    scenes: int
    valid: int
    invalid: int
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("split must be a non-empty string")
        episodes = _require_nonnegative_int(self.episodes, "episodes")
        scenes = _require_nonnegative_int(self.scenes, "scenes")
        valid = _require_nonnegative_int(self.valid, "valid")
        invalid = _require_nonnegative_int(self.invalid, "invalid")
        if episodes == 0 or scenes == 0:
            raise ValueError("episodes and scenes must be positive")
        if valid + invalid != episodes:
            raise ValueError("valid plus invalid must equal episodes")
        if (
            not isinstance(self.manifest_sha256, str)
            or len(self.manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.manifest_sha256)
        ):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "scenes", scenes)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "invalid", invalid)


VAL_SEEN_POPULATION = PopulationContract(
    "val_seen",
    778,
    53,
    770,
    8,
    "1d3eb25eb7583a2c6373f430119da25ef71472d2d065697e673e2f70c307d146",
)
VAL_UNSEEN_POPULATION = PopulationContract(
    "val_unseen",
    1839,
    11,
    1830,
    9,
    "31e7c5f8d186e75222b12f1aa8862b16fa9904c75221a0fd55cfba61f93fa8ce",
)
POPULATION_CONTRACTS = (VAL_SEEN_POPULATION, VAL_UNSEEN_POPULATION)


@dataclass(frozen=True)
class _CapturedSource:
    role: str
    path: str
    member: str
    sha256: str
    size_bytes: int
    data: bytes
    filesystem_path: Path


@dataclass(frozen=True)
class SourcePhaseArtifact:
    name: str
    sha256: str
    entry_count: int


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _validate_logical_source_path(logical_path: str) -> None:
    if (
        not isinstance(logical_path, str)
        or not logical_path
        or "\x00" in logical_path
        or "\\" in logical_path
    ):
        raise ValueError("source logical path is invalid")
    path = Path(logical_path)
    if path.is_absolute() or ".." in path.parts or path.parts[0] not in {
        "val_seen",
        "val_unseen",
    }:
        raise ValueError("source logical path must be split-relative")


def _read_regular_unsymlinked(path: Path) -> bytes:
    absolute = path.absolute()
    directory_fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    final_fd: int | None = None
    try:
        for index, part in enumerate(absolute.parts[1:]):
            final = index == len(absolute.parts) - 2
            flags = os.O_RDONLY | os.O_NOFOLLOW
            if not final:
                flags |= os.O_DIRECTORY
            next_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        final_fd = directory_fd
        directory_fd = -1
        if not stat.S_ISREG(os.fstat(final_fd).st_mode):
            raise ValueError(f"source path must be a regular file: {path}")
        with os.fdopen(final_fd, "rb") as stream:
            final_fd = None
            return stream.read()
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"source path contains a symlink or non-directory component: {path}"
            ) from error
        raise
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)
        if final_fd is not None:
            os.close(final_fd)


def _captured_source(
    data: bytes,
    role: str,
    logical_path: str,
    member: str = "",
    filesystem_path: Path = Path("."),
) -> _CapturedSource:
    _validate_logical_source_path(logical_path)
    if not isinstance(member, str) or "\x00" in member or "\\" in member:
        raise ValueError("source member is invalid")
    parts = Path(logical_path).parts
    split = parts[0]
    if role == "r2r_episode_source":
        valid = len(parts) == 2 and parts[1] == f"{split}.json.gz" and not member
    elif role == "r2r_ground_truth_source":
        valid = len(parts) == 2 and parts[1] == f"{split}_gt.json.gz" and not member
    elif role == "prediction_manifest":
        valid = len(parts) == 2 and parts[1] == "manifest.json" and not member
    elif role == "prediction_text":
        valid = len(parts) == 3 and parts[2].endswith(".txt") and not member
    elif role == "cognitive_map_assignment_member":
        valid = (
            len(parts) == 3
            and parts[2].endswith(".npz")
            and member in {"start_position.npy", "start_direction_vector.npy"}
        )
    elif role == "cognitive_map_target_member":
        valid = (
            len(parts) == 3
            and parts[2].endswith(".npz")
            and member in {"grid.npy", "direction_vectors.npy"}
        )
    else:
        valid = False
    if not valid:
        raise ValueError("source role, logical path, and member do not match")
    return _CapturedSource(
        role,
        logical_path,
        member,
        sha256(data).hexdigest(),
        len(data),
        data,
        filesystem_path,
    )


def _ordinary_source_entry(
    path: Path, role: str, logical_path: str
) -> _CapturedSource:
    data = _read_regular_unsymlinked(path)
    return _captured_source(data, role, logical_path, filesystem_path=path)


def _npz_member_sources(
    path: Path,
    role: str,
    logical_path: str,
    members: tuple[str, ...],
) -> tuple[_CapturedSource, ...]:
    if len(set(members)) != len(members):
        raise ValueError("requested NPZ members must be unique")
    archive_bytes = _read_regular_unsymlinked(path)
    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            names = tuple(info.filename for info in archive.infolist())
            result = []
            for member in members:
                if names.count(member) != 1:
                    raise ValueError(f"NPZ member {member!r} must occur exactly once")
                result.append(
                    _captured_source(
                        archive.read(member), role, logical_path, member, path
                    )
                )
    except BadZipFile as error:
        raise ValueError(f"invalid NPZ archive: {path}") from error
    return tuple(result)


def _source_entry_payload(source: _CapturedSource) -> dict[str, object]:
    return {
        "member": source.member,
        "path": source.path,
        "role": source.role,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
    }


def _source_phase(
    name: str, sources: Sequence[_CapturedSource]
) -> SourcePhaseArtifact:
    ordered = sorted(sources, key=lambda source: (source.role, source.path, source.member))
    if len({(source.role, source.path, source.member) for source in ordered}) != len(
        ordered
    ):
        raise ValueError("source phase contains a duplicate inventory entry")
    payload = {
        "entries": [_source_entry_payload(source) for source in ordered],
        "phase": name,
        "schema_version": _SOURCE_SCHEMA_VERSION,
    }
    return SourcePhaseArtifact(
        name, sha256(_canonical_json_bytes(payload)).hexdigest(), len(ordered)
    )


def _combined_sources_sha256(phases: Sequence[SourcePhaseArtifact]) -> str:
    expected_names = tuple(_SOURCE_PHASE_CONTRACTS)
    by_name = {phase.name: phase.sha256 for phase in phases}
    if tuple(by_name) != expected_names:
        raise ValueError("source phases must use the frozen phase order")
    return sha256(
        _canonical_json_bytes(
            {
                "phase_digests": by_name,
                "schema_version": _SOURCE_SCHEMA_VERSION,
            }
        )
    ).hexdigest()


def _load_npy_bytes(data: bytes, member: str) -> NDArray[np.generic]:
    try:
        value = np.load(BytesIO(data), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid NPY member {member!r}") from error
    if not isinstance(value, np.ndarray):
        raise ValueError(f"NPY member {member!r} must contain an array")
    return value


@dataclass(frozen=True)
class RuntimeEpisode:
    key: EpisodeKey
    split: str
    selector_input: SelectorInput
    start_pivot: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.key, EpisodeKey):
            raise ValueError("key must be an EpisodeKey")
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("split must be a non-empty string")
        if not isinstance(self.selector_input, SelectorInput):
            raise ValueError("selector_input must be a SelectorInput")
        if len(self.start_pivot) != 2:
            raise ValueError("start_pivot must contain exactly two coordinates")
        pivot = (
            _require_finite_float(self.start_pivot[0], "start_pivot row"),
            _require_finite_float(self.start_pivot[1], "start_pivot column"),
        )
        if not self.selector_input.schema_valid and (
            np.any(self.selector_input.predicted_grid)
            or np.any(self.selector_input.predicted_directions)
        ):
            raise ValueError("invalid selector_input must be empty")
        object.__setattr__(self, "start_pivot", pivot)


@dataclass(frozen=True)
class TargetEpisode:
    runtime: RuntimeEpisode
    target_grid: NDArray[np.bool_]
    target_directions: NDArray[np.float32]

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, RuntimeEpisode):
            raise ValueError("runtime must be a RuntimeEpisode")
        object.__setattr__(self, "target_grid", _require_boolean_grid(self.target_grid))
        object.__setattr__(
            self,
            "target_directions",
            _require_float32_array(self.target_directions, (5, 2), "target_directions"),
        )


@dataclass(frozen=True)
class DevelopmentCandidateScore:
    candidate_kind: str
    mapping_sign: int
    heading_offset_degrees: int
    global_angle_degrees: int
    episode_count: int
    valid_count: int
    all_mean_iou: float
    object_mean_iou: float
    region_mean_iou: float
    selected: bool

    def __post_init__(self) -> None:
        if self.candidate_kind not in ("mapping", "global"):
            raise ValueError("candidate_kind must be mapping or global")
        if self.candidate_kind == "mapping":
            HeadingMapping(self.mapping_sign, float(self.heading_offset_degrees))
            if self.global_angle_degrees != -1:
                raise ValueError("mapping global_angle_degrees must be -1")
        elif (
            self.mapping_sign != -1
            or self.heading_offset_degrees != -1
            or self.global_angle_degrees not in tuple(int(angle) for angle in ANGLE_ORDER)
        ):
            raise ValueError("global candidates must use mapping sentinels and cardinal angle")
        episodes = _require_nonnegative_int(self.episode_count, "episode_count")
        valid = _require_nonnegative_int(self.valid_count, "valid_count")
        if episodes == 0 or valid > episodes:
            raise ValueError("valid_count must be within a non-empty episode_count")
        scores = (
            _require_finite_float(self.all_mean_iou, "all_mean_iou"),
            _require_finite_float(self.object_mean_iou, "object_mean_iou"),
            _require_finite_float(self.region_mean_iou, "region_mean_iou"),
        )
        if any(not 0.0 <= score <= 1.0 for score in scores):
            raise ValueError("mean IoU values must be within [0, 1]")
        _require_boolean(self.selected, "selected")
        object.__setattr__(self, "episode_count", episodes)
        object.__setattr__(self, "valid_count", valid)
        object.__setattr__(self, "all_mean_iou", scores[0])
        object.__setattr__(self, "object_mean_iou", scores[1])
        object.__setattr__(self, "region_mean_iou", scores[2])


def _declared_mappings() -> tuple[HeadingMapping, ...]:
    return tuple(
        HeadingMapping(sign, angle) for sign in (+1, -1) for angle in ANGLE_ORDER
    )


@dataclass(frozen=True)
class SelectorLock:
    chosen_mapping: HeadingMapping
    chosen_global_angle: float
    development_scores: tuple[DevelopmentCandidateScore, ...]
    mapping_tied_candidate_count: int
    global_tied_candidate_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.chosen_mapping, HeadingMapping):
            raise ValueError("chosen_mapping must be a HeadingMapping")
        global_angle = _require_cardinal_angle(
            self.chosen_global_angle, "chosen_global_angle"
        )
        scores = self.development_scores
        if len(scores) != 12 or not all(
            isinstance(score, DevelopmentCandidateScore) for score in scores
        ):
            raise ValueError("development_scores must contain eight mappings and four globals")
        expected_mappings = _declared_mappings()
        mapping_scores = scores[:8]
        global_scores = scores[8:]
        if any(score.candidate_kind != "mapping" for score in mapping_scores) or any(
            score.candidate_kind != "global" for score in global_scores
        ):
            raise ValueError("development scores must use declared candidate order")
        if tuple(
            HeadingMapping(score.mapping_sign, float(score.heading_offset_degrees))
            for score in mapping_scores
        ) != expected_mappings or tuple(float(score.global_angle_degrees) for score in global_scores) != ANGLE_ORDER:
            raise ValueError("development scores must use declared candidate order")
        selected_mappings = tuple(score for score in mapping_scores if score.selected)
        selected_globals = tuple(score for score in global_scores if score.selected)
        if len(selected_mappings) != 1 or len(selected_globals) != 1:
            raise ValueError("development scores must select one mapping and one global angle")
        if HeadingMapping(
            selected_mappings[0].mapping_sign,
            float(selected_mappings[0].heading_offset_degrees),
        ) != self.chosen_mapping or float(selected_globals[0].global_angle_degrees) != global_angle:
            raise ValueError("selected development rows must match the lock")
        mapping_ties = _require_nonnegative_int(
            self.mapping_tied_candidate_count, "mapping_tied_candidate_count"
        )
        global_ties = _require_nonnegative_int(
            self.global_tied_candidate_count, "global_tied_candidate_count"
        )
        if mapping_ties == 0 or global_ties == 0:
            raise ValueError("tie counts must be positive")
        best_mapping = max(score.all_mean_iou for score in mapping_scores)
        best_global = max(score.all_mean_iou for score in global_scores)
        if mapping_ties != sum(
            score.all_mean_iou == best_mapping for score in mapping_scores
        ) or global_ties != sum(score.all_mean_iou == best_global for score in global_scores):
            raise ValueError("tie counts must match exact aggregate ties")
        object.__setattr__(self, "chosen_global_angle", global_angle)


def _start_pivot(start_position: NDArray[np.generic]) -> tuple[float, float]:
    if start_position.shape != (2,):
        raise ValueError("start_position must have shape (2,)")
    row = _require_finite_float(float(start_position[0]), "start_position row")
    column = _require_finite_float(float(start_position[1]), "start_position column")
    return row / (CELL_SIZE * GRID_SCALE), column / (CELL_SIZE * GRID_SCALE)


def _empty_selector_input(start_direction: NDArray[np.float32]) -> SelectorInput:
    return SelectorInput(
        schema_valid=False,
        start_direction=start_direction,
        predicted_grid=np.zeros((37, 50, 50), dtype=np.bool_),
        predicted_directions=np.zeros((5, 2), dtype=np.float32),
    )


def _validate_runtime_population(
    runtime_episodes: Sequence[RuntimeEpisode], contract: PopulationContract | None = None
) -> None:
    if not runtime_episodes:
        raise ValueError("runtime population must not be empty")
    keys: set[EpisodeKey] = set()
    scenes: set[str] = set()
    valid_count = 0
    split = runtime_episodes[0].split
    for episode in runtime_episodes:
        if not isinstance(episode, RuntimeEpisode):
            raise ValueError("runtime population must contain RuntimeEpisode values")
        if episode.split != split:
            raise ValueError("runtime population must contain one split")
        if episode.key in keys:
            raise ValueError("runtime population must have unique EpisodeKey values")
        keys.add(episode.key)
        scenes.add(episode.key.scene_id)
        valid_count += int(episode.selector_input.schema_valid)
    if contract is not None:
        if split != contract.split or len(runtime_episodes) != contract.episodes:
            raise ValueError("runtime population does not match its split contract")
        if len(scenes) != contract.scenes or valid_count != contract.valid:
            raise ValueError("runtime population does not match its split contract")
        if len(runtime_episodes) - valid_count != contract.invalid:
            raise ValueError("runtime population does not match its split contract")


def load_runtime_population(
    split: str,
    contract: PopulationContract,
    cache_dir: Path,
    cache_model_key: str,
    cognitive_map_namespace: str,
    quiet: bool,
) -> tuple[RuntimeEpisode, ...]:
    """Load only selector-eligible inputs and seal their fixed split contract."""
    if not isinstance(contract, PopulationContract) or split != contract.split:
        raise ValueError("split must match the PopulationContract")
    manifest_path = llm_navigation_split_dir(
        "R2R", split, cache_dir=cache_dir, model_key=cache_model_key
    ) / "manifest.json"
    if sha256(manifest_path.read_bytes()).hexdigest() != contract.manifest_sha256:
        raise ValueError("prediction manifest SHA-256 does not match the split contract")
    loaded = load_llm_grid_examples(
        (split,),
        quiet=quiet,
        cognitive_map_namespace=cognitive_map_namespace,
        datasets=("R2R",),
    )
    episodes: list[RuntimeEpisode] = []
    for example in loaded.examples:
        if example.dataset != "R2R" or example.split != split:
            raise ValueError("LLM-Grid loader returned a non-R2R requested-split example")
        with np.load(example.raster_path, allow_pickle=False) as raster:
            start_position = np.asarray(raster["start_position"])
            start_direction = np.asarray(
                raster["start_direction_vector"], dtype=np.float32
            )
        pivot = _start_pivot(start_position)
        prediction_path = llm_navigation_prediction_path(
            example.scene_id,
            example.example_id,
            "R2R",
            split,
            cache_dir=cache_dir,
            model_key=cache_model_key,
        )
        try:
            parsed = parse_grid_text(prediction_path.read_text(encoding="utf-8"))
            selector_input = SelectorInput(
                schema_valid=True,
                start_direction=start_direction,
                predicted_grid=parsed.grid > 0,
                predicted_directions=parsed.direction_vectors,
            )
        except (FileNotFoundError, UnicodeDecodeError, LLMGridValidationError):
            selector_input = _empty_selector_input(start_direction)
        episodes.append(
            RuntimeEpisode(
                key=EpisodeKey(example.scene_id, example.example_id),
                split=split,
                selector_input=selector_input,
                start_pivot=pivot,
            )
        )
    result = tuple(sorted(episodes, key=lambda episode: episode.key))
    _validate_runtime_population(result, contract)
    return result


def _load_target_arrays(path: Path) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    """Read the target-only cache arrays and apply the authoritative scale-two raster."""
    with np.load(path, allow_pickle=False) as raster:
        grid = np.asarray(raster["grid"], dtype=np.float32)
        directions = np.asarray(raster["direction_vectors"], dtype=np.float32)
    target_grid = np.asarray(downsample_grid(grid, GRID_SCALE) > 0, dtype=np.bool_)
    return _require_boolean_grid(target_grid), _require_float32_array(
        directions, (5, 2), "target_directions"
    )


def load_target_population(
    runtime_episodes: Sequence[RuntimeEpisode],
    cognitive_map_namespace: str,
    quiet: bool,
) -> tuple[TargetEpisode, ...]:
    """Join target-bearing rasters only after the runtime population is sealed."""
    _validate_runtime_population(runtime_episodes)
    split = runtime_episodes[0].split
    loaded = load_llm_grid_examples(
        (split,),
        quiet=quiet,
        cognitive_map_namespace=cognitive_map_namespace,
        datasets=("R2R",),
    )
    paths: dict[EpisodeKey, Path] = {}
    for example in loaded.examples:
        if example.dataset != "R2R" or example.split != split:
            raise ValueError("LLM-Grid loader returned a non-R2R requested-split example")
        key = EpisodeKey(example.scene_id, example.example_id)
        if key in paths:
            raise ValueError("target population must have unique EpisodeKey values")
        paths[key] = example.raster_path
    if set(paths) != {episode.key for episode in runtime_episodes}:
        raise ValueError("target population keys must exactly match runtime population keys")
    return tuple(
        TargetEpisode(episode, *_load_target_arrays(paths[episode.key]))
        for episode in sorted(runtime_episodes, key=lambda runtime: runtime.key)
    )


def _decode_json_source(source: _CapturedSource, *, gzipped: bool) -> dict[str, object]:
    try:
        raw = gzip.decompress(source.data) if gzipped else source.data
        value = json.loads(raw.decode("utf-8"))
    except (gzip.BadGzipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON source: {source.path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON source must contain an object: {source.path}")
    return cast(Dict[str, object], value)


def _episode_inventory(
    split: str,
) -> tuple[tuple[tuple[str, str], ...], tuple[_CapturedSource, ...]]:
    split_dir = R2R_DIR / split
    episode = _ordinary_source_entry(
        split_dir / f"{split}.json.gz",
        "r2r_episode_source",
        f"{split}/{split}.json.gz",
    )
    ground_truth = _ordinary_source_entry(
        split_dir / f"{split}_gt.json.gz",
        "r2r_ground_truth_source",
        f"{split}/{split}_gt.json.gz",
    )
    episode_payload = _decode_json_source(episode, gzipped=True)
    ground_truth_payload = _decode_json_source(ground_truth, gzipped=True)
    raw_episodes = episode_payload.get("episodes")
    if not isinstance(raw_episodes, list):
        raise ValueError("R2R episode source must contain an episodes list")
    records: list[tuple[str, str]] = []
    for raw_value in raw_episodes:
        if not isinstance(raw_value, dict):
            raise ValueError("R2R episode entries must be objects")
        value = cast(Dict[str, object], raw_value)
        raw_instruction = value.get("instruction")
        if not isinstance(raw_instruction, dict):
            raise ValueError("R2R instruction must be an object")
        instruction = cast(Dict[str, object], raw_instruction)
        language = instruction.get("language", "en-US")
        if not isinstance(language, str) or not language.startswith("en-"):
            continue
        episode_id = value.get("episode_id")
        scene_path = value.get("scene_id")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int):
            raise ValueError("R2R episode_id must be an integer")
        if not isinstance(scene_path, str) or not scene_path:
            raise ValueError("R2R scene_id must be a non-empty string")
        ground_truth_entry = ground_truth_payload.get(str(episode_id))
        if not isinstance(ground_truth_entry, dict) or "locations" not in ground_truth_entry:
            raise ValueError(f"missing R2R ground truth for episode {episode_id}")
        records.append((Path(scene_path).stem, f"R2R_{split}_{episode_id}"))
    if len(set(records)) != len(records):
        raise ValueError("R2R source contains duplicate episode identities")
    return tuple(records), (episode, ground_truth)


def _prediction_source(
    split: str,
    scene_id: str,
    example_id: str,
    cache_dir: Path,
    cache_model_key: str,
) -> _CapturedSource:
    path = llm_navigation_prediction_path(
        scene_id,
        example_id,
        "R2R",
        split,
        cache_dir=cache_dir,
        model_key=cache_model_key,
    )
    return _ordinary_source_entry(
        path, "prediction_text", f"{split}/{scene_id}/{example_id}.txt"
    )


def _raster_path(
    scene_id: str, example_id: str, cognitive_map_namespace: str
) -> Path:
    return cognitive_map_cache_path(
        scene_id, example_id, namespace=cognitive_map_namespace
    )


def _selector_input_from_sources(
    prediction: _CapturedSource,
    start_position: _CapturedSource,
    start_direction: _CapturedSource,
) -> tuple[SelectorInput, tuple[float, float]]:
    position = _load_npy_bytes(start_position.data, start_position.member)
    direction = np.asarray(
        _load_npy_bytes(start_direction.data, start_direction.member), dtype=np.float32
    )
    pivot = _start_pivot(position)
    try:
        parsed = parse_grid_text(prediction.data.decode("utf-8"))
        selector_input = SelectorInput(
            True, direction, parsed.grid > 0, parsed.direction_vectors
        )
    except (UnicodeDecodeError, LLMGridValidationError):
        selector_input = _empty_selector_input(direction)
    return selector_input, pivot


def _load_runtime_population_coupled(
    split: str,
    contract: PopulationContract,
    cache_dir: Path,
    cache_model_key: str,
    cognitive_map_namespace: str,
    *,
    include_targets: bool,
) -> tuple[
    tuple[RuntimeEpisode, ...],
    tuple[TargetEpisode, ...],
    tuple[_CapturedSource, ...],
    SourcePhaseArtifact,
]:
    records, dataset_sources = _episode_inventory(split)
    manifest_path = llm_navigation_split_dir(
        "R2R", split, cache_dir=cache_dir, model_key=cache_model_key
    ) / "manifest.json"
    manifest = _ordinary_source_entry(
        manifest_path, "prediction_manifest", f"{split}/manifest.json"
    )
    if manifest.sha256 != contract.manifest_sha256:
        raise ValueError("prediction manifest SHA-256 does not match the split contract")
    sources: list[_CapturedSource] = [*dataset_sources, manifest]
    runtimes: list[RuntimeEpisode] = []
    targets: list[TargetEpisode] = []
    for scene_id, example_id in records:
        prediction = _prediction_source(
            split, scene_id, example_id, cache_dir, cache_model_key
        )
        logical_raster = f"{split}/{scene_id}/{example_id}.npz"
        assignment_members = _npz_member_sources(
            _raster_path(scene_id, example_id, cognitive_map_namespace),
            "cognitive_map_assignment_member",
            logical_raster,
            ("start_position.npy", "start_direction_vector.npy"),
        )
        selector_input, pivot = _selector_input_from_sources(
            prediction, assignment_members[0], assignment_members[1]
        )
        runtime = RuntimeEpisode(
            EpisodeKey(scene_id, example_id), split, selector_input, pivot
        )
        runtimes.append(runtime)
        sources.extend((prediction, *assignment_members))
        if include_targets:
            target_members = _npz_member_sources(
                _raster_path(scene_id, example_id, cognitive_map_namespace),
                "cognitive_map_target_member",
                logical_raster,
                ("grid.npy", "direction_vectors.npy"),
            )
            grid = np.asarray(
                _load_npy_bytes(target_members[0].data, target_members[0].member),
                dtype=np.float32,
            )
            directions = np.asarray(
                _load_npy_bytes(target_members[1].data, target_members[1].member),
                dtype=np.float32,
            )
            target_grid = np.asarray(
                downsample_grid(grid, GRID_SCALE) > 0, dtype=np.bool_
            )
            targets.append(TargetEpisode(runtime, target_grid, directions))
            sources.extend(target_members)
    ordered_runtimes = tuple(sorted(runtimes, key=lambda item: item.key))
    _validate_runtime_population(ordered_runtimes, contract)
    by_key = {target.runtime.key: target for target in targets}
    ordered_targets = (
        tuple(by_key[runtime.key] for runtime in ordered_runtimes)
        if include_targets
        else ()
    )
    phase_name = (
        "development_all_sources" if include_targets else "test_assignment_sources"
    )
    phase = _source_phase(phase_name, sources)
    return ordered_runtimes, ordered_targets, tuple(sources), phase


def _load_test_targets_coupled(
    runtimes: Sequence[RuntimeEpisode], cognitive_map_namespace: str
) -> tuple[tuple[TargetEpisode, ...], tuple[_CapturedSource, ...], SourcePhaseArtifact]:
    _validate_runtime_population(runtimes)
    sources: list[_CapturedSource] = []
    targets: list[TargetEpisode] = []
    for runtime in sorted(runtimes, key=lambda item: item.key):
        logical_raster = (
            f"{runtime.split}/{runtime.key.scene_id}/{runtime.key.example_id}.npz"
        )
        members = _npz_member_sources(
            _raster_path(
                runtime.key.scene_id,
                runtime.key.example_id,
                cognitive_map_namespace,
            ),
            "cognitive_map_target_member",
            logical_raster,
            ("grid.npy", "direction_vectors.npy"),
        )
        grid = np.asarray(
            _load_npy_bytes(members[0].data, members[0].member), dtype=np.float32
        )
        directions = np.asarray(
            _load_npy_bytes(members[1].data, members[1].member), dtype=np.float32
        )
        targets.append(
            TargetEpisode(
                runtime,
                np.asarray(downsample_grid(grid, GRID_SCALE) > 0, dtype=np.bool_),
                directions,
            )
        )
        sources.extend(members)
    phase = _source_phase("test_target_sources", sources)
    return tuple(targets), tuple(sources), phase


def _phase_matches_contract(phase: SourcePhaseArtifact) -> bool:
    expected_count, expected_sha256 = _SOURCE_PHASE_CONTRACTS[phase.name]
    return phase.entry_count == expected_count and phase.sha256 == expected_sha256


def _require_phase_contract(phase: SourcePhaseArtifact) -> None:
    if not _phase_matches_contract(phase):
        raise ValueError(f"{phase.name} does not match its frozen source commitment")


def _require_combined_sources(phases: Sequence[SourcePhaseArtifact]) -> str:
    combined = _combined_sources_sha256(phases)
    if combined != _COMBINED_SOURCE_SHA256:
        raise ValueError("combined sources do not match the frozen commitment")
    return combined


def _post_score_sources_verified(sources: Sequence[_CapturedSource]) -> bool:
    try:
        ordinary = tuple(source for source in sources if not source.member)
        for source in ordinary:
            current = _ordinary_source_entry(
                source.filesystem_path, source.role, source.path
            )
            if current.sha256 != source.sha256 or current.size_bytes != source.size_bytes:
                return False
        grouped: dict[tuple[Path, str, str], list[_CapturedSource]] = {}
        for source in sources:
            if source.member:
                grouped.setdefault(
                    (source.filesystem_path, source.role, source.path), []
                ).append(source)
        for (path, role, logical_path), expected in grouped.items():
            current = _npz_member_sources(
                path, role, logical_path, tuple(source.member for source in expected)
            )
            if tuple(
                (item.member, item.sha256, item.size_bytes) for item in current
            ) != tuple(
                (item.member, item.sha256, item.size_bytes) for item in expected
            ):
                return False
    except (FileNotFoundError, OSError, ValueError):
        return False
    return True


def _candidate_iou(
    target: TargetEpisode, angle: float
) -> tuple[float, float, float]:
    runtime = target.runtime
    if not runtime.selector_input.schema_valid:
        angle = 0.0
    warped = warp_grid_about_pivot(
        runtime.selector_input.predicted_grid, runtime.start_pivot, angle
    )
    all_score = score_warped_grid(warped, target.target_grid).iou

    def family_iou(start: int, end: int) -> float:
        family = warped.grid[start:end]
        family_warp = WarpedGrid(
            family,
            warped.bounds,
            int(np.count_nonzero(family)),
        )
        return score_warped_grid(family_warp, target.target_grid[start:end]).iou

    return all_score, family_iou(0, _OBJECT_CHANNEL_COUNT), family_iou(
        _OBJECT_CHANNEL_COUNT, 37
    )


def _candidate_score(
    candidate_kind: str,
    mapping: HeadingMapping | None,
    angle: float | None,
    development_targets: Sequence[TargetEpisode],
) -> DevelopmentCandidateScore:
    if candidate_kind == "mapping" and mapping is None:
        raise ValueError("mapping candidate requires a HeadingMapping")
    if candidate_kind == "global" and angle is None:
        raise ValueError("global candidate requires an angle")
    values: list[tuple[float, float, float]] = []
    valid_count = 0
    for target in development_targets:
        runtime = target.runtime
        valid_count += int(runtime.selector_input.schema_valid)
        selected_angle = (
            mapping.angle_for(runtime.selector_input.start_direction)
            if mapping is not None
            else angle
        )
        if selected_angle is None:
            raise RuntimeError("candidate must define an angle")
        values.append(_candidate_iou(target, selected_angle))
    means = tuple(float(np.mean([value[index] for value in values])) for index in range(3))
    if mapping is not None:
        return DevelopmentCandidateScore(
            "mapping",
            mapping.sign,
            int(mapping.offset_degrees),
            -1,
            len(values),
            valid_count,
            means[0],
            means[1],
            means[2],
            False,
        )
    if angle is None:
        raise RuntimeError("global candidate must define an angle")
    return DevelopmentCandidateScore(
        "global",
        -1,
        -1,
        int(angle),
        len(values),
        valid_count,
        means[0],
        means[1],
        means[2],
        False,
    )


def develop_selector(development_targets: Sequence[TargetEpisode]) -> SelectorLock:
    """Calibrate one mapping and one global angle on the complete development split."""
    if not development_targets or not all(
        isinstance(target, TargetEpisode) for target in development_targets
    ):
        raise ValueError("development_targets must contain TargetEpisode values")
    runtime_episodes = tuple(target.runtime for target in development_targets)
    _validate_runtime_population(runtime_episodes)
    if len({target.runtime.key for target in development_targets}) != len(development_targets):
        raise ValueError("development targets must have unique EpisodeKey values")
    mappings = _declared_mappings()
    mapping_scores = tuple(
        _candidate_score("mapping", mapping, None, development_targets)
        for mapping in mappings
    )
    global_scores = tuple(
        _candidate_score("global", None, angle, development_targets) for angle in ANGLE_ORDER
    )
    best_mapping_score = max(score.all_mean_iou for score in mapping_scores)
    best_global_score = max(score.all_mean_iou for score in global_scores)
    chosen_mapping_index = next(
        index
        for index, score in enumerate(mapping_scores)
        if score.all_mean_iou == best_mapping_score
    )
    chosen_global_index = next(
        index
        for index, score in enumerate(global_scores)
        if score.all_mean_iou == best_global_score
    )
    scores = tuple(
        replace(score, selected=index == chosen_mapping_index)
        for index, score in enumerate(mapping_scores)
    ) + tuple(
        replace(score, selected=index == chosen_global_index)
        for index, score in enumerate(global_scores)
    )
    return SelectorLock(
        mappings[chosen_mapping_index],
        ANGLE_ORDER[chosen_global_index],
        scores,
        sum(score.all_mean_iou == best_mapping_score for score in mapping_scores),
        sum(score.all_mean_iou == best_global_score for score in global_scores),
    )


def assign_population(
    runtime_episodes: Sequence[RuntimeEpisode], selector_lock: SelectorLock
) -> tuple[SelectorAssignment, ...]:
    """Seal target-free assignments from runtime selector inputs and a frozen lock."""
    _validate_runtime_population(runtime_episodes)
    if not isinstance(selector_lock, SelectorLock):
        raise ValueError("selector_lock must be a SelectorLock")
    assignments: list[SelectorAssignment] = []
    for runtime in sorted(runtime_episodes, key=lambda episode: episode.key):
        selector_input = runtime.selector_input
        heading = quantize_heading(selector_input.start_direction)
        if selector_input.schema_valid:
            primary = selector_lock.chosen_mapping.angle_for(selector_input.start_direction)
            global_angle = selector_lock.chosen_global_angle
            direct_angle, direct_margin = direct_heading_assignment(selector_input)
        else:
            primary = 0.0
            global_angle = 0.0
            direct_angle = 0.0
            direct_margin = 0.0
        assignments.append(
            SelectorAssignment(
                runtime.key,
                selector_input.schema_valid,
                heading,
                primary,
                global_angle,
                direct_angle,
                direct_margin,
            )
        )
    return tuple(assignments)


def _format_float(value: float) -> str:
    finite = _require_finite_float(value, "CSV float")
    return "0" if finite == 0.0 else format(finite, ".17g")


def assignment_csv_bytes(assignments: Sequence[SelectorAssignment]) -> bytes:
    """Return canonical, sorted target-free assignment evidence bytes."""
    if not all(isinstance(assignment, SelectorAssignment) for assignment in assignments):
        raise ValueError("assignments must contain SelectorAssignment values")
    ordered = tuple(sorted(assignments, key=lambda assignment: assignment.key))
    if len({assignment.key for assignment in ordered}) != len(ordered):
        raise ValueError("assignments must have unique EpisodeKey values")
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(_ASSIGNMENT_HEADER)
    for assignment in ordered:
        writer.writerow(
            (
                assignment.key.scene_id,
                assignment.key.example_id,
                "true" if assignment.schema_valid else "false",
                _format_float(assignment.heading_bin_degrees),
                _format_float(assignment.primary_angle_degrees),
                _format_float(assignment.global_angle_degrees),
                _format_float(assignment.direct_angle_degrees),
                _format_float(assignment.direct_margin),
            )
        )
    return stream.getvalue().encode("utf-8")


@dataclass(frozen=True)
class AngleScoreRow:
    key: EpisodeKey
    angle_degrees: float
    all_iou: float
    object_iou: float
    region_iou: float
    direction_cosine: float
    predicted_support: int
    target_support: int
    in_frame_support: int
    out_of_frame_support: int
    union: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, EpisodeKey):
            raise ValueError("key must be an EpisodeKey")
        object.__setattr__(
            self, "angle_degrees", _require_cardinal_angle(self.angle_degrees, "angle_degrees")
        )
        for name in ("all_iou", "object_iou", "region_iou"):
            value = _require_finite_float(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)
        cosine = _require_finite_float(self.direction_cosine, "direction_cosine")
        if not -1.0 <= cosine <= 1.0:
            raise ValueError("direction_cosine must be within [-1, 1]")
        object.__setattr__(self, "direction_cosine", cosine)
        supports = tuple(
            _require_nonnegative_int(getattr(self, name), name)
            for name in (
                "predicted_support",
                "target_support",
                "in_frame_support",
                "out_of_frame_support",
                "union",
            )
        )
        if supports[2] + supports[3] != supports[0]:
            raise ValueError("frame supports must partition predicted_support")
        for name, value in zip(
            (
                "predicted_support",
                "target_support",
                "in_frame_support",
                "out_of_frame_support",
                "union",
            ),
            supports,
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class EpisodeScoreRow:
    key: EpisodeKey
    schema_valid: bool
    heading_bin_degrees: float
    primary_angle_degrees: float
    global_angle_degrees: float
    direct_angle_degrees: float
    identity_iou: float
    primary_iou: float
    global_iou: float
    direct_iou: float
    random_expected_iou: float
    soft_identity_iou: float
    soft_aggregate_iou: float
    oracle_iou: float
    identity_object_iou: float
    primary_object_iou: float
    identity_region_iou: float
    primary_region_iou: float
    soft_identity_object_iou: float
    soft_aggregate_object_iou: float
    soft_identity_region_iou: float
    soft_aggregate_region_iou: float
    primary_predicted_support: int
    primary_target_support: int
    primary_in_frame_support: int
    primary_out_of_frame_support: int
    primary_union: int
    soft_prediction_mass: float
    soft_target_mass: float
    soft_intersection_mass: float
    soft_union_mass: float
    primary_direction_cosine: float
    direct_direction_cosine: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, EpisodeKey):
            raise ValueError("key must be an EpisodeKey")
        _require_boolean(self.schema_valid, "schema_valid")
        for name in (
            "heading_bin_degrees",
            "primary_angle_degrees",
            "global_angle_degrees",
            "direct_angle_degrees",
        ):
            object.__setattr__(self, name, _require_cardinal_angle(getattr(self, name), name))
        for name in (
            "identity_iou",
            "primary_iou",
            "global_iou",
            "direct_iou",
            "random_expected_iou",
            "soft_identity_iou",
            "soft_aggregate_iou",
            "oracle_iou",
            "identity_object_iou",
            "primary_object_iou",
            "identity_region_iou",
            "primary_region_iou",
            "soft_identity_object_iou",
            "soft_aggregate_object_iou",
            "soft_identity_region_iou",
            "soft_aggregate_region_iou",
        ):
            value = _require_finite_float(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)
        supports = tuple(
            _require_nonnegative_int(getattr(self, name), name)
            for name in (
                "primary_predicted_support",
                "primary_target_support",
                "primary_in_frame_support",
                "primary_out_of_frame_support",
                "primary_union",
            )
        )
        if supports[2] + supports[3] != supports[0]:
            raise ValueError("primary frame supports must partition predicted_support")
        for name, value in zip(
            (
                "primary_predicted_support",
                "primary_target_support",
                "primary_in_frame_support",
                "primary_out_of_frame_support",
                "primary_union",
            ),
            supports,
        ):
            object.__setattr__(self, name, value)
        masses = tuple(
            _require_finite_float(getattr(self, name), name)
            for name in (
                "soft_prediction_mass",
                "soft_target_mass",
                "soft_intersection_mass",
                "soft_union_mass",
            )
        )
        if any(value < 0.0 for value in masses) or masses[2] > masses[3]:
            raise ValueError("soft masses must be nonnegative with intersection at most union")
        for name, value in zip(
            (
                "soft_prediction_mass",
                "soft_target_mass",
                "soft_intersection_mass",
                "soft_union_mass",
            ),
            masses,
        ):
            object.__setattr__(self, name, value)
        for name in ("primary_direction_cosine", "direct_direction_cosine"):
            value = _require_finite_float(getattr(self, name), name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [-1, 1]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class OracleGainSummary:
    available: bool
    primary_fraction: float
    global_fraction: float
    direct_fraction: float
    soft_fraction: float

    def __post_init__(self) -> None:
        _require_boolean(self.available, "available")
        for name in (
            "primary_fraction",
            "global_fraction",
            "direct_fraction",
            "soft_fraction",
        ):
            object.__setattr__(self, name, _require_finite_float(getattr(self, name), name))
        if not self.available and any(
            getattr(self, name) != 0.0
            for name in (
                "primary_fraction",
                "global_fraction",
                "direct_fraction",
                "soft_fraction",
            )
        ):
            raise ValueError("unavailable oracle gain fractions must be zero")


def oracle_gain_summary(rows: Sequence[EpisodeScoreRow]) -> OracleGainSummary:
    """Return frozen aggregate oracle-gain recovery fractions for every candidate."""
    ordered = _validated_score_rows(rows)
    denominator = sum(row.oracle_iou - row.identity_iou for row in ordered)
    if denominator <= 0.0:
        return OracleGainSummary(False, 0.0, 0.0, 0.0, 0.0)
    return OracleGainSummary(
        True,
        sum(row.primary_iou - row.identity_iou for row in ordered) / denominator,
        sum(row.global_iou - row.identity_iou for row in ordered) / denominator,
        sum(row.direct_iou - row.identity_iou for row in ordered) / denominator,
        sum(row.soft_aggregate_iou - row.soft_identity_iou for row in ordered) / denominator,
    )


def _family_score(warped: WarpedGrid, target: NDArray[np.bool_], start: int, end: int) -> float:
    family = warped.grid[start:end]
    return score_warped_grid(
        WarpedGrid(family, warped.bounds, int(np.count_nonzero(family))), target[start:end]
    ).iou


def score_population(
    targets: Sequence[TargetEpisode], assignments: Sequence[SelectorAssignment]
) -> tuple[tuple[AngleScoreRow, ...], tuple[EpisodeScoreRow, ...]]:
    """Score sealed assignments using all four precomputed cardinal warps per episode."""
    if not targets or not all(isinstance(target, TargetEpisode) for target in targets):
        raise ValueError("targets must contain TargetEpisode values")
    if not all(isinstance(assignment, SelectorAssignment) for assignment in assignments):
        raise ValueError("assignments must contain SelectorAssignment values")
    target_by_key = {target.runtime.key: target for target in targets}
    assignment_by_key = {assignment.key: assignment for assignment in assignments}
    if len(target_by_key) != len(targets) or len(assignment_by_key) != len(assignments):
        raise ValueError("targets and assignments must have unique EpisodeKey values")
    if set(target_by_key) != set(assignment_by_key):
        raise ValueError("target and assignment keys must exactly match")

    angle_rows: list[AngleScoreRow] = []
    episode_rows: list[EpisodeScoreRow] = []
    for key in sorted(target_by_key):
        target = target_by_key[key]
        assignment = assignment_by_key[key]
        runtime = target.runtime
        if assignment.schema_valid != runtime.selector_input.schema_valid:
            raise ValueError("assignment schema_valid must match its target runtime")
        warps = tuple(
            warp_grid_about_pivot(runtime.selector_input.predicted_grid, runtime.start_pivot, angle)
            for angle in ANGLE_ORDER
        )
        rows_by_angle: dict[float, AngleScoreRow] = {}
        for angle, warped in zip(ANGLE_ORDER, warps):
            raster = score_warped_grid(warped, target.target_grid)
            row = AngleScoreRow(
                key,
                angle,
                raster.iou,
                _family_score(warped, target.target_grid, 0, _OBJECT_CHANNEL_COUNT),
                _family_score(warped, target.target_grid, _OBJECT_CHANNEL_COUNT, 37),
                direction_cosine(
                    rotate_direction_vectors(runtime.selector_input.predicted_directions, angle),
                    target.target_directions,
                ),
                raster.predicted_support,
                raster.target_support,
                raster.in_frame_support,
                raster.out_of_frame_support,
                raster.union,
            )
            rows_by_angle[angle] = row
            angle_rows.append(row)
        identity = rows_by_angle[0.0]
        primary = rows_by_angle[assignment.primary_angle_degrees]
        global_row = rows_by_angle[assignment.global_angle_degrees]
        direct = rows_by_angle[assignment.direct_angle_degrees]
        soft = uniform_soft_score(warps, target.target_grid)
        soft_identity_object = _family_soft_score(
            (warps[0],), target.target_grid, 0, _OBJECT_CHANNEL_COUNT
        )
        soft_aggregate_object = _family_soft_score(
            warps, target.target_grid, 0, _OBJECT_CHANNEL_COUNT
        )
        soft_identity_region = _family_soft_score(
            (warps[0],), target.target_grid, _OBJECT_CHANNEL_COUNT, 37
        )
        soft_aggregate_region = _family_soft_score(
            warps, target.target_grid, _OBJECT_CHANNEL_COUNT, 37
        )
        episode_rows.append(
            EpisodeScoreRow(
                key,
                assignment.schema_valid,
                assignment.heading_bin_degrees,
                assignment.primary_angle_degrees,
                assignment.global_angle_degrees,
                assignment.direct_angle_degrees,
                identity.all_iou,
                primary.all_iou,
                global_row.all_iou,
                direct.all_iou,
                float(np.mean([row.all_iou for row in rows_by_angle.values()])),
                identity.all_iou,
                soft.iou,
                max(row.all_iou for row in rows_by_angle.values()),
                identity.object_iou,
                primary.object_iou,
                identity.region_iou,
                primary.region_iou,
                soft_identity_object.iou,
                soft_aggregate_object.iou,
                soft_identity_region.iou,
                soft_aggregate_region.iou,
                primary.predicted_support,
                primary.target_support,
                primary.in_frame_support,
                primary.out_of_frame_support,
                primary.union,
                soft.predicted_mass,
                soft.target_mass,
                soft.intersection_mass,
                soft.union_mass,
                primary.direction_cosine,
                direct.direction_cosine,
            )
        )
    return tuple(angle_rows), tuple(episode_rows)


@dataclass(frozen=True)
class ContrastInterval:
    mean: float
    ci_lower: float
    ci_upper: float
    loso_min: float
    loso_max: float

    def __post_init__(self) -> None:
        for name in ("mean", "ci_lower", "ci_upper", "loso_min", "loso_max"):
            object.__setattr__(self, name, _require_finite_float(getattr(self, name), name))
        if self.ci_lower > self.ci_upper:
            raise ValueError("ci_lower must not exceed ci_upper")
        if self.loso_min > self.loso_max:
            raise ValueError("loso_min must not exceed loso_max")


@dataclass(frozen=True)
class BootstrapIntervals:
    primary_identity: ContrastInterval
    primary_global: ContrastInterval
    global_identity: ContrastInterval
    direct_identity: ContrastInterval
    soft_aggregate_identity: ContrastInterval

    def __post_init__(self) -> None:
        if not all(isinstance(value, ContrastInterval) for value in self.__dict__.values()):
            raise ValueError("bootstrap intervals must contain ContrastInterval values")


def _contrast_values(rows: Sequence[EpisodeScoreRow], contrast_name: str) -> tuple[float, ...]:
    attributes = {
        "primary_identity": ("primary_iou", "identity_iou"),
        "primary_global": ("primary_iou", "global_iou"),
        "global_identity": ("global_iou", "identity_iou"),
        "direct_identity": ("direct_iou", "identity_iou"),
        "soft_aggregate_identity": ("soft_aggregate_iou", "soft_identity_iou"),
    }
    if contrast_name not in attributes:
        raise ValueError("contrast_name must be a declared contrast")
    positive, baseline = attributes[contrast_name]
    return tuple(float(getattr(row, positive) - getattr(row, baseline)) for row in rows)


def _validated_score_rows(rows: Sequence[EpisodeScoreRow]) -> tuple[EpisodeScoreRow, ...]:
    if not rows or not all(isinstance(row, EpisodeScoreRow) for row in rows):
        raise ValueError("rows must contain EpisodeScoreRow values")
    ordered = tuple(sorted(rows, key=lambda row: row.key))
    if len({row.key for row in ordered}) != len(ordered):
        raise ValueError("rows must have unique EpisodeKey values")
    return ordered


def leave_one_scene_out(rows: Sequence[EpisodeScoreRow], contrast_name: str) -> tuple[float, float]:
    """Return the episode-macro contrast range after omitting each scene once."""
    ordered = _validated_score_rows(rows)
    values = _contrast_values(ordered, contrast_name)
    scene_ids = tuple(sorted({row.key.scene_id for row in ordered}))
    if len(scene_ids) < 2:
        raise ValueError("leave-one-scene-out requires at least two scenes")
    means = tuple(
        float(np.mean([value for row, value in zip(ordered, values) if row.key.scene_id != scene_id]))
        for scene_id in scene_ids
    )
    return min(means), max(means)


def paired_scene_bootstrap(
    rows: Sequence[EpisodeScoreRow], repetitions: int, seed: int
) -> BootstrapIntervals:
    """Estimate every frozen contrast from one shared paired scene bootstrap."""
    ordered = _validated_score_rows(rows)
    repetitions = _require_nonnegative_int(repetitions, "repetitions")
    if repetitions == 0:
        raise ValueError("repetitions must be positive")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    scene_ids = tuple(sorted({row.key.scene_id for row in ordered}))
    draws = np.random.default_rng(int(seed)).integers(
        0, len(scene_ids), size=(repetitions, len(scene_ids))
    )
    multiplicities = np.asarray(
        [np.count_nonzero(draws == index, axis=1) for index in range(len(scene_ids))],
        dtype=np.int64,
    ).T
    scene_counts = np.asarray(
        [sum(row.key.scene_id == scene_id for row in ordered) for scene_id in scene_ids],
        dtype=np.int64,
    )
    denominator = multiplicities @ scene_counts
    intervals: list[ContrastInterval] = []
    for name in (
        "primary_identity",
        "primary_global",
        "global_identity",
        "direct_identity",
        "soft_aggregate_identity",
    ):
        values = _contrast_values(ordered, name)
        scene_sums = np.asarray(
            [
                sum(value for row, value in zip(ordered, values) if row.key.scene_id == scene_id)
                for scene_id in scene_ids
            ],
            dtype=np.float64,
        )
        replicates = (multiplicities @ scene_sums) / denominator
        ci_lower, ci_upper = np.percentile(replicates, (2.5, 97.5))
        loso_min, loso_max = leave_one_scene_out(ordered, name)
        intervals.append(
            ContrastInterval(
                float(np.mean(values)), float(ci_lower), float(ci_upper), loso_min, loso_max
            )
        )
    return BootstrapIntervals(*intervals)


class DecisionLabel(str, Enum):
    SELECTOR_GO = "SELECTOR GO"
    FIXED_CORRECTION_GO = "FIXED-CORRECTION GO"
    PARTIAL = "PARTIAL"
    NO_GO = "NO GO"


@dataclass(frozen=True)
class DecisionResult:
    label: DecisionLabel
    selector_conditions: tuple[bool, ...]
    global_conditions: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, DecisionLabel):
            raise ValueError("label must be a DecisionLabel")
        if not all(isinstance(value, bool) for value in self.selector_conditions + self.global_conditions):
            raise ValueError("decision conditions must be booleans")


def classify_decision(
    selector_identity: ContrastInterval,
    selector_global: ContrastInterval,
    global_identity: ContrastInterval,
    selector_object_mean: float,
    selector_region_mean: float,
    global_object_mean: float,
    global_region_mean: float,
    audit_passed: bool,
) -> DecisionResult:
    """Classify the preregistered selector result in its frozen precedence order."""
    if not all(
        isinstance(interval, ContrastInterval)
        for interval in (selector_identity, selector_global, global_identity)
    ):
        raise ValueError("contrasts must be ContrastInterval values")
    _require_boolean(audit_passed, "audit_passed")
    selector_semantic = (
        _require_finite_float(selector_object_mean, "selector_object_mean") > 0.0
        and _require_finite_float(selector_region_mean, "selector_region_mean") > 0.0
    )
    global_semantic = (
        _require_finite_float(global_object_mean, "global_object_mean") > 0.0
        and _require_finite_float(global_region_mean, "global_region_mean") > 0.0
    )
    selector_conditions = (
        selector_identity.mean >= 0.01,
        selector_identity.ci_lower > 0.0,
        selector_semantic,
        selector_identity.loso_min > 0.0,
        selector_global.ci_lower > 0.0,
        audit_passed,
    )
    global_conditions = (
        global_identity.mean >= 0.01,
        global_identity.ci_lower > 0.0,
        global_semantic,
        global_identity.loso_min > 0.0,
        audit_passed,
    )
    if all(selector_conditions):
        label = DecisionLabel.SELECTOR_GO
    elif all(global_conditions):
        label = DecisionLabel.FIXED_CORRECTION_GO
    elif (
        selector_identity.mean >= 0.01 and selector_identity.ci_lower > 0.0
    ) or (global_identity.mean >= 0.01 and global_identity.ci_lower > 0.0):
        label = DecisionLabel.PARTIAL
    else:
        label = DecisionLabel.NO_GO
    return DecisionResult(label, selector_conditions, global_conditions)


@dataclass(frozen=True)
class PopulationSummary:
    episodes: int
    scenes: int
    valid: int
    invalid: int


@dataclass(frozen=True)
class PopulationArtifact:
    development: PopulationSummary
    test: PopulationSummary


@dataclass(frozen=True)
class CandidateMeans:
    all_iou: float
    object_iou: float
    region_iou: float


@dataclass(frozen=True)
class MeansSummary:
    identity: CandidateMeans
    primary: CandidateMeans
    global_correction: CandidateMeans
    direct: CandidateMeans
    random_expected: CandidateMeans
    soft_identity: CandidateMeans
    soft_aggregate: CandidateMeans
    oracle: CandidateMeans


@dataclass(frozen=True)
class ContrastSummary:
    mean: float
    ci_lower: float
    ci_upper: float
    loso_min: float
    loso_max: float


@dataclass(frozen=True)
class ContrastCollection:
    primary_identity: ContrastSummary
    primary_global: ContrastSummary
    global_identity: ContrastSummary
    direct_identity: ContrastSummary
    soft_aggregate_identity: ContrastSummary


@dataclass(frozen=True)
class SemanticFamilySummary:
    primary_object_delta: float
    primary_region_delta: float
    global_object_delta: float
    global_region_delta: float


@dataclass(frozen=True)
class CellMetricSummary:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class CellMetricsCollection:
    identity: CellMetricSummary
    primary: CellMetricSummary
    global_correction: CellMetricSummary
    direct: CellMetricSummary
    soft_identity: CellMetricSummary
    soft_aggregate: CellMetricSummary


@dataclass(frozen=True)
class SupportSummary:
    predicted_support: int
    target_support: int
    in_frame_support: int
    out_of_frame_support: int
    union: int
    soft_predicted_mass: float
    soft_target_mass: float
    soft_intersection_mass: float
    soft_union_mass: float
    invalid_count: int
    empty_prediction_count: int
    empty_target_count: int
    empty_union_count: int


@dataclass(frozen=True)
class DirectionSummary:
    identity_cosine: float
    primary_cosine: float
    global_cosine: float
    direct_cosine: float


@dataclass(frozen=True)
class AngleSummary:
    primary_counts: tuple[tuple[float, int], ...]
    global_counts: tuple[tuple[float, int], ...]
    direct_counts: tuple[tuple[float, int], ...]
    oracle_counts: tuple[tuple[float, int], ...]
    primary_oracle_agreement: float
    direct_oracle_agreement: float
    heading_angle_counts: tuple[tuple[float, float, int], ...]


@dataclass(frozen=True)
class CostSummary:
    primary_warps: int
    global_warps: int
    direct_warps: int
    soft_warps: int
    maximum_materialized_array_bytes: int


@dataclass(frozen=True)
class DecisionSummary:
    label: str
    selector_conditions: tuple[bool, ...]
    global_conditions: tuple[bool, ...]


@dataclass(frozen=True)
class AuditSummary:
    development_sources_verified: bool
    test_assignment_sources_verified: bool
    assignments_serialized_before_test_targets: bool
    assignment_sha256: str
    test_target_sources_verified: bool
    combined_sources_verified: bool
    post_score_sources_verified: bool
    development_contract_exact: bool
    test_contract_exact: bool
    passed: bool

    def __post_init__(self) -> None:
        checks = (
            self.development_sources_verified,
            self.test_assignment_sources_verified,
            self.assignments_serialized_before_test_targets,
            self.test_target_sources_verified,
            self.combined_sources_verified,
            self.post_score_sources_verified,
            self.development_contract_exact,
            self.test_contract_exact,
        )
        if not all(isinstance(value, bool) for value in checks + (self.passed,)):
            raise ValueError("audit checks must be booleans")
        if (
            len(self.assignment_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.assignment_sha256)
        ):
            raise ValueError("assignment_sha256 must be a lowercase SHA-256 digest")
        if self.passed != all(checks):
            raise ValueError("audit passed must equal the conjunction of audit checks")


@dataclass(frozen=True)
class ChosenMappingArtifact:
    sign: int
    offset_degrees: float
    development_mean_iou: float
    tied_candidate_count: int


@dataclass(frozen=True)
class ChosenGlobalAngleArtifact:
    angle_degrees: float
    development_mean_iou: float
    tied_candidate_count: int


@dataclass(frozen=True)
class DevelopmentArtifact:
    population: PopulationSummary
    candidate_scores: tuple[DevelopmentCandidateScore, ...]


@dataclass(frozen=True)
class LockProtocolArtifact:
    angle_order: tuple[float, ...]
    mapping_order: tuple[tuple[int, int], ...]
    heading_tie_order: tuple[float, ...]
    invalid_angle: float


@dataclass(frozen=True)
class SelectorLockArtifact:
    schema_version: str
    chosen_mapping: ChosenMappingArtifact
    chosen_global_angle: ChosenGlobalAngleArtifact
    development: DevelopmentArtifact
    protocol: LockProtocolArtifact


@dataclass(frozen=True)
class SummaryArtifact:
    schema_version: str
    population: PopulationArtifact
    means: MeansSummary
    contrasts: ContrastCollection
    semantic_families: SemanticFamilySummary
    cell_metrics: CellMetricsCollection
    support: SupportSummary
    directions: DirectionSummary
    angles: AngleSummary
    cost: CostSummary
    oracle_gain: OracleGainSummary
    audit: AuditSummary
    decision: DecisionSummary


@dataclass(frozen=True)
class LeaveOneSceneOutSummary:
    primary_identity: tuple[float, float]
    primary_global: tuple[float, float]
    global_identity: tuple[float, float]


@dataclass(frozen=True)
class BootstrapArtifact:
    schema_version: str
    seed: int
    repetitions: int
    scene_ids: tuple[str, ...]
    intervals: BootstrapIntervals
    leave_one_scene_out: LeaveOneSceneOutSummary


@dataclass(frozen=True)
class SourcesArtifact:
    development_all_sources: SourcePhaseArtifact
    test_assignment_sources: SourcePhaseArtifact
    test_target_sources: SourcePhaseArtifact
    combined_sha256: str


@dataclass(frozen=True)
class ProtocolArtifact:
    cache_model_key: str
    cognitive_map_namespace: str
    scale: int
    raster_shape: tuple[int, int, int]
    object_channels: tuple[int, int]
    region_channels: tuple[int, int]
    angle_order: tuple[float, ...]
    mapping_order: tuple[tuple[int, int], ...]
    heading_tie_order: tuple[float, ...]
    invalid_angle: float
    bootstrap_seed: int
    bootstrap_repetitions: int
    bootstrap_cluster: str
    bootstrap_weighting: str
    bootstrap_interval: str
    gate_threshold: float
    selector_input_schema: tuple[str, ...]
    coordinate_contract: tuple[str, ...]
    phase_order: tuple[str, ...]
    decision_gate_contract: tuple[str, ...]
    schema_version: str
    target_free_limitations: tuple[str, ...]


@dataclass(frozen=True)
class CsvHeadersArtifact:
    development_mapping_scores: tuple[str, ...]
    test_assignments: tuple[str, ...]
    test_angle_scores: tuple[str, ...]
    test_episode_scores: tuple[str, ...]


@dataclass(frozen=True)
class JsonTopLevelKeysArtifact:
    selector_lock: tuple[str, ...]
    summary: tuple[str, ...]
    bootstrap: tuple[str, ...]
    manifest: tuple[str, ...]


@dataclass(frozen=True)
class SchemaArtifact:
    csv_headers: CsvHeadersArtifact
    json_top_level_keys: JsonTopLevelKeysArtifact
    float_format: str
    row_order: str
    json_serialization: str


@dataclass(frozen=True)
class ArtifactEntry:
    name: str
    sha256: str
    size_bytes: int
    data_rows: int


@dataclass(frozen=True)
class ManifestArtifact:
    schema_version: str
    sources: SourcesArtifact
    git_commit: str
    protocol: ProtocolArtifact
    population: PopulationArtifact
    schemas: SchemaArtifact
    artifacts: tuple[ArtifactEntry, ...]


@dataclass(frozen=True)
class ManifestInputs:
    sources: SourcesArtifact
    git_commit: str
    protocol: ProtocolArtifact
    population: PopulationArtifact
    schemas: SchemaArtifact


@dataclass(frozen=True)
class ValidationInputs:
    development_contract: PopulationContract
    test_contract: PopulationContract
    cache_dir: Path
    cache_model_key: str
    cognitive_map_namespace: str
    expected_git_commit: str
    quiet: bool


@dataclass(frozen=True)
class ArtifactBundle:
    manifest_json: bytes
    development_mapping_scores_csv: bytes
    selector_lock_json: bytes
    test_assignments_csv: bytes
    test_angle_scores_csv: bytes
    test_episode_scores_csv: bytes
    summary_json: bytes
    bootstrap_json: bytes
    target_free_control_intervals_png: bytes

    def files(self) -> Dict[str, bytes]:
        return {
            "manifest.json": self.manifest_json,
            "development_mapping_scores.csv": self.development_mapping_scores_csv,
            "selector_lock.json": self.selector_lock_json,
            "test_assignments.csv": self.test_assignments_csv,
            "test_angle_scores.csv": self.test_angle_scores_csv,
            "test_episode_scores.csv": self.test_episode_scores_csv,
            "summary.json": self.summary_json,
            "bootstrap.json": self.bootstrap_json,
            "target_free_control_intervals.png": (
                self.target_free_control_intervals_png
            ),
        }


class TargetFreeArgs(Tap):
    """Fixed inputs for the target-free cardinal-selector experiment."""

    cache_dir: Path = _DEFAULT_CACHE_DIR
    cache_model_key: str = _DEFAULT_CACHE_MODEL_KEY
    cognitive_map_namespace: str = _DEFAULT_COGNITIVE_MAP_NAMESPACE
    output_dir: Path = _DEFAULT_OUTPUT_DIR
    angles: Tuple[float, ...] = ANGLE_ORDER
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED
    bootstrap_repetitions: int = _DEFAULT_BOOTSTRAP_REPETITIONS
    gate_threshold: float = _GATE_THRESHOLD
    quiet: bool = False


def _json_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value)


def _csv_cell(value: str | bool | int | float) -> str:
    if isinstance(value, str):
        if not value:
            raise ValueError("CSV strings must be non-empty")
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        return _format_float(float(value))
    raise ValueError("unsupported CSV cell type")


def _csv_bytes(
    header: tuple[str, ...], rows: Sequence[tuple[str | bool | int | float, ...]]
) -> bytes:
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        if len(row) != len(header):
            raise ValueError("CSV row width must match its header")
        writer.writerow(tuple(_csv_cell(value) for value in row))
    return stream.getvalue().encode("utf-8")


def _development_csv(scores: Sequence[DevelopmentCandidateScore]) -> bytes:
    return _csv_bytes(
        DEVELOPMENT_MAPPING_SCORES_HEADER,
        tuple(
            (
                score.candidate_kind,
                score.mapping_sign,
                score.heading_offset_degrees,
                score.global_angle_degrees,
                score.episode_count,
                score.valid_count,
                score.all_mean_iou,
                score.object_mean_iou,
                score.region_mean_iou,
                score.selected,
            )
            for score in scores
        ),
    )


def _angle_csv(rows: Sequence[AngleScoreRow]) -> bytes:
    return _csv_bytes(
        TEST_ANGLE_SCORES_HEADER,
        tuple(
            (
                row.key.scene_id,
                row.key.example_id,
                row.angle_degrees,
                row.all_iou,
                row.object_iou,
                row.region_iou,
                row.direction_cosine,
                row.predicted_support,
                row.target_support,
                row.in_frame_support,
                row.out_of_frame_support,
                row.union,
            )
            for row in rows
        ),
    )


def _episode_csv(rows: Sequence[EpisodeScoreRow]) -> bytes:
    return _csv_bytes(
        TEST_EPISODE_SCORES_HEADER,
        tuple(
            (
                row.key.scene_id,
                row.key.example_id,
                row.schema_valid,
                row.heading_bin_degrees,
                row.primary_angle_degrees,
                row.global_angle_degrees,
                row.direct_angle_degrees,
                row.identity_iou,
                row.primary_iou,
                row.global_iou,
                row.direct_iou,
                row.random_expected_iou,
                row.soft_identity_iou,
                row.soft_aggregate_iou,
                row.oracle_iou,
                row.identity_object_iou,
                row.primary_object_iou,
                row.identity_region_iou,
                row.primary_region_iou,
                row.soft_identity_object_iou,
                row.soft_aggregate_object_iou,
                row.soft_identity_region_iou,
                row.soft_aggregate_region_iou,
                row.primary_predicted_support,
                row.primary_target_support,
                row.primary_in_frame_support,
                row.primary_out_of_frame_support,
                row.primary_union,
                row.soft_prediction_mass,
                row.soft_target_mass,
                row.soft_intersection_mass,
                row.soft_union_mass,
                row.primary_direction_cosine,
                row.direct_direction_cosine,
            )
            for row in rows
        ),
    )


def _contrast_summary(interval: ContrastInterval) -> ContrastSummary:
    return ContrastSummary(
        interval.mean,
        interval.ci_lower,
        interval.ci_upper,
        interval.loso_min,
        interval.loso_max,
    )


def _selected_angle_rows(
    angle_rows: Sequence[AngleScoreRow],
    episode_rows: Sequence[EpisodeScoreRow],
    endpoint: str,
) -> tuple[AngleScoreRow, ...]:
    by_key_angle = {(row.key, row.angle_degrees): row for row in angle_rows}
    selected: list[AngleScoreRow] = []
    for episode in episode_rows:
        if endpoint == "identity":
            angle = 0.0
        elif endpoint == "primary":
            angle = episode.primary_angle_degrees
        elif endpoint == "global":
            angle = episode.global_angle_degrees
        elif endpoint == "direct":
            angle = episode.direct_angle_degrees
        else:
            raise ValueError("unknown endpoint")
        selected.append(by_key_angle[(episode.key, angle)])
    return tuple(selected)


def _candidate_means(rows: Sequence[AngleScoreRow]) -> CandidateMeans:
    return CandidateMeans(
        float(np.mean([row.all_iou for row in rows])),
        float(np.mean([row.object_iou for row in rows])),
        float(np.mean([row.region_iou for row in rows])),
    )


def _cell_metrics(rows: Sequence[AngleScoreRow]) -> CellMetricSummary:
    predicted = sum(row.predicted_support for row in rows)
    target = sum(row.target_support for row in rows)
    intersection = sum(
        row.predicted_support + row.target_support - row.union for row in rows
    )
    precision = 0.0 if predicted == 0 else intersection / predicted
    recall = 0.0 if target == 0 else intersection / target
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (
        precision + recall
    )
    return CellMetricSummary(precision, recall, f1)


def _soft_metrics(
    predicted: float, target: float, intersection: float
) -> CellMetricSummary:
    precision = 0.0 if predicted == 0.0 else intersection / predicted
    recall = 0.0 if target == 0.0 else intersection / target
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (
        precision + recall
    )
    return CellMetricSummary(precision, recall, f1)


def _angle_counts(values: Sequence[float]) -> tuple[tuple[float, int], ...]:
    return tuple((angle, sum(value == angle for value in values)) for angle in ANGLE_ORDER)


def _build_bootstrap_artifact(
    episode_rows: Sequence[EpisodeScoreRow], *, seed: int, repetitions: int
) -> BootstrapArtifact:
    intervals = paired_scene_bootstrap(episode_rows, repetitions, seed)
    return BootstrapArtifact(
        _SCHEMA_VERSION,
        seed,
        repetitions,
        tuple(sorted({row.key.scene_id for row in episode_rows})),
        intervals,
        LeaveOneSceneOutSummary(
            leave_one_scene_out(episode_rows, "primary_identity"),
            leave_one_scene_out(episode_rows, "primary_global"),
            leave_one_scene_out(episode_rows, "global_identity"),
        ),
    )


def _build_summary_artifact(
    angle_rows: Sequence[AngleScoreRow],
    episode_rows: Sequence[EpisodeScoreRow],
    population: PopulationArtifact,
    audit: AuditSummary,
) -> SummaryArtifact:
    if not isinstance(audit, AuditSummary):
        raise ValueError("audit must be an AuditSummary")
    ordered_episodes = _validated_score_rows(episode_rows)
    expected_angle_order = tuple(
        (episode.key, angle) for episode in ordered_episodes for angle in ANGLE_ORDER
    )
    if tuple((row.key, row.angle_degrees) for row in angle_rows) != expected_angle_order:
        raise ValueError("angle rows must use sorted episode and declared angle order")
    interval_values = paired_scene_bootstrap(
        ordered_episodes, _DEFAULT_BOOTSTRAP_REPETITIONS, _DEFAULT_BOOTSTRAP_SEED
    )
    contrasts = ContrastCollection(
        _contrast_summary(interval_values.primary_identity),
        _contrast_summary(interval_values.primary_global),
        _contrast_summary(interval_values.global_identity),
        _contrast_summary(interval_values.direct_identity),
        _contrast_summary(interval_values.soft_aggregate_identity),
    )
    identity_rows = _selected_angle_rows(angle_rows, ordered_episodes, "identity")
    primary_rows = _selected_angle_rows(angle_rows, ordered_episodes, "primary")
    global_rows = _selected_angle_rows(angle_rows, ordered_episodes, "global")
    direct_rows = _selected_angle_rows(angle_rows, ordered_episodes, "direct")
    identity_means = _candidate_means(identity_rows)
    primary_means = _candidate_means(primary_rows)
    global_means = _candidate_means(global_rows)
    direct_means = _candidate_means(direct_rows)
    random_means = _candidate_means(angle_rows)
    oracle_rows = tuple(
        max(
            (row for row in angle_rows if row.key == episode.key),
            key=lambda row: row.all_iou,
        )
        for episode in ordered_episodes
    )
    oracle_means = _candidate_means(oracle_rows)
    soft_all = float(np.mean([row.soft_aggregate_iou for row in ordered_episodes]))
    soft_identity_means = CandidateMeans(
        float(np.mean([row.soft_identity_iou for row in ordered_episodes])),
        float(np.mean([row.soft_identity_object_iou for row in ordered_episodes])),
        float(np.mean([row.soft_identity_region_iou for row in ordered_episodes])),
    )
    soft_aggregate_means = CandidateMeans(
        soft_all,
        float(np.mean([row.soft_aggregate_object_iou for row in ordered_episodes])),
        float(np.mean([row.soft_aggregate_region_iou for row in ordered_episodes])),
    )
    means = MeansSummary(
        identity_means,
        primary_means,
        global_means,
        direct_means,
        random_means,
        soft_identity_means,
        soft_aggregate_means,
        oracle_means,
    )
    primary_object_delta = primary_means.object_iou - identity_means.object_iou
    primary_region_delta = primary_means.region_iou - identity_means.region_iou
    global_object_delta = global_means.object_iou - identity_means.object_iou
    global_region_delta = global_means.region_iou - identity_means.region_iou
    decision = classify_decision(
        interval_values.primary_identity,
        interval_values.primary_global,
        interval_values.global_identity,
        primary_object_delta,
        primary_region_delta,
        global_object_delta,
        global_region_delta,
        audit.passed,
    )
    predicted_mass = sum(row.soft_prediction_mass for row in ordered_episodes)
    target_mass = sum(row.soft_target_mass for row in ordered_episodes)
    intersection_mass = sum(row.soft_intersection_mass for row in ordered_episodes)
    primary_angles = tuple(row.primary_angle_degrees for row in ordered_episodes)
    global_angles = tuple(row.global_angle_degrees for row in ordered_episodes)
    direct_angles = tuple(row.direct_angle_degrees for row in ordered_episodes)
    oracle_angles = tuple(row.angle_degrees for row in oracle_rows)
    return SummaryArtifact(
        _SCHEMA_VERSION,
        population,
        means,
        contrasts,
        SemanticFamilySummary(
            primary_object_delta,
            primary_region_delta,
            global_object_delta,
            global_region_delta,
        ),
        CellMetricsCollection(
            _cell_metrics(identity_rows),
            _cell_metrics(primary_rows),
            _cell_metrics(global_rows),
            _cell_metrics(direct_rows),
            _cell_metrics(identity_rows),
            _soft_metrics(predicted_mass, target_mass, intersection_mass),
        ),
        SupportSummary(
            sum(row.primary_predicted_support for row in ordered_episodes),
            sum(row.primary_target_support for row in ordered_episodes),
            sum(row.primary_in_frame_support for row in ordered_episodes),
            sum(row.primary_out_of_frame_support for row in ordered_episodes),
            sum(row.primary_union for row in ordered_episodes),
            predicted_mass,
            target_mass,
            intersection_mass,
            sum(row.soft_union_mass for row in ordered_episodes),
            sum(not row.schema_valid for row in ordered_episodes),
            sum(row.primary_predicted_support == 0 for row in ordered_episodes),
            sum(row.primary_target_support == 0 for row in ordered_episodes),
            sum(row.primary_union == 0 for row in ordered_episodes),
        ),
        DirectionSummary(
            float(np.mean([row.direction_cosine for row in identity_rows])),
            float(np.mean([row.direction_cosine for row in primary_rows])),
            float(np.mean([row.direction_cosine for row in global_rows])),
            float(np.mean([row.direction_cosine for row in direct_rows])),
        ),
        AngleSummary(
            _angle_counts(primary_angles),
            _angle_counts(global_angles),
            _angle_counts(direct_angles),
            _angle_counts(oracle_angles),
            float(np.mean([left == right for left, right in zip(primary_angles, oracle_angles)])),
            float(np.mean([left == right for left, right in zip(direct_angles, oracle_angles)])),
            tuple(
                (
                    heading,
                    angle,
                    sum(
                        row.heading_bin_degrees == heading
                        and row.primary_angle_degrees == angle
                        for row in ordered_episodes
                    ),
                )
                for heading in ANGLE_ORDER
                for angle in ANGLE_ORDER
            ),
        ),
        CostSummary(
            len(ordered_episodes),
            len(ordered_episodes),
            len(ordered_episodes) * len(ANGLE_ORDER),
            len(ordered_episodes) * len(ANGLE_ORDER),
            37 * 100 * 100 * np.dtype(np.float64).itemsize,
        ),
        oracle_gain_summary(ordered_episodes),
        audit,
        DecisionSummary(
            decision.label.value,
            decision.selector_conditions,
            decision.global_conditions,
        ),
    )


def _plot_bytes(summary: SummaryArtifact) -> bytes:
    labels = ("Primary−identity", "Primary−global", "Global−identity")
    values = (
        summary.contrasts.primary_identity,
        summary.contrasts.primary_global,
        summary.contrasts.global_identity,
    )
    figure, axis = plt.subplots(figsize=(7.0, 3.5))
    means = np.asarray([value.mean for value in values])
    lower = means - np.asarray([value.ci_lower for value in values])
    upper = np.asarray([value.ci_upper for value in values]) - means
    axis.errorbar(
        means,
        np.arange(len(labels)),
        xerr=np.vstack((lower, upper)),
        fmt="o",
        color="black",
        capsize=4,
    )
    axis.axvline(0.0, color="gray", linewidth=1)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Paired episode-macro IoU contrast")
    axis.set_title("Target-free cardinal selector controls")
    figure.tight_layout()
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=150, metadata={"Software": "ETP-R1"})
    plt.close(figure)
    return stream.getvalue()


def build_artifacts(
    selector_lock: SelectorLock,
    assignments: Sequence[SelectorAssignment],
    angle_rows: Sequence[AngleScoreRow],
    episode_rows: Sequence[EpisodeScoreRow],
    summary: SummaryArtifact,
    bootstrap: BootstrapArtifact,
    manifest_inputs: ManifestInputs,
) -> ArtifactBundle:
    """Build the complete canonical nine-file transaction in memory."""
    if summary.population != manifest_inputs.population:
        raise ValueError("summary and manifest populations must match")
    if summary.schema_version != _SCHEMA_VERSION or bootstrap.schema_version != _SCHEMA_VERSION:
        raise ValueError("artifact schema versions must match the frozen protocol")
    if manifest_inputs.protocol.schema_version != _SCHEMA_VERSION:
        raise ValueError("manifest protocol schema version must be frozen")
    if tuple(assignments) != tuple(sorted(assignments, key=lambda row: row.key)):
        raise ValueError("assignments must use canonical key order")
    expected_angle_order = tuple(
        (episode.key, angle) for episode in episode_rows for angle in ANGLE_ORDER
    )
    if tuple((row.key, row.angle_degrees) for row in angle_rows) != expected_angle_order:
        raise ValueError("angle rows must use canonical key and angle order")
    mapping_score = next(
        score
        for score in selector_lock.development_scores[:8]
        if score.selected
    )
    global_score = next(
        score
        for score in selector_lock.development_scores[8:]
        if score.selected
    )
    lock_artifact = SelectorLockArtifact(
        _SCHEMA_VERSION,
        ChosenMappingArtifact(
            selector_lock.chosen_mapping.sign,
            selector_lock.chosen_mapping.offset_degrees,
            mapping_score.all_mean_iou,
            selector_lock.mapping_tied_candidate_count,
        ),
        ChosenGlobalAngleArtifact(
            selector_lock.chosen_global_angle,
            global_score.all_mean_iou,
            selector_lock.global_tied_candidate_count,
        ),
        DevelopmentArtifact(
            manifest_inputs.population.development,
            selector_lock.development_scores,
        ),
        LockProtocolArtifact(
            manifest_inputs.protocol.angle_order,
            manifest_inputs.protocol.mapping_order,
            manifest_inputs.protocol.heading_tie_order,
            manifest_inputs.protocol.invalid_angle,
        ),
    )
    development_csv = _development_csv(selector_lock.development_scores)
    lock_json = _json_bytes(asdict(lock_artifact))
    assignments_csv = assignment_csv_bytes(assignments)
    angle_csv = _angle_csv(angle_rows)
    episode_csv = _episode_csv(episode_rows)
    summary_json = _json_bytes(asdict(summary))
    bootstrap_json = _json_bytes(asdict(bootstrap))
    plot_png = _plot_bytes(summary)
    payloads = (
        ("development_mapping_scores.csv", development_csv, len(selector_lock.development_scores)),
        ("selector_lock.json", lock_json, 1),
        ("test_assignments.csv", assignments_csv, len(assignments)),
        ("test_angle_scores.csv", angle_csv, len(angle_rows)),
        ("test_episode_scores.csv", episode_csv, len(episode_rows)),
        ("summary.json", summary_json, 1),
        ("bootstrap.json", bootstrap_json, 1),
        ("target_free_control_intervals.png", plot_png, 0),
    )
    entries = tuple(
        ArtifactEntry(name, sha256(payload).hexdigest(), len(payload), rows)
        for name, payload, rows in payloads
    )
    manifest = ManifestArtifact(
        _SCHEMA_VERSION,
        manifest_inputs.sources,
        manifest_inputs.git_commit,
        manifest_inputs.protocol,
        manifest_inputs.population,
        manifest_inputs.schemas,
        entries,
    )
    return ArtifactBundle(
        _json_bytes(asdict(manifest)),
        development_csv,
        lock_json,
        assignments_csv,
        angle_csv,
        episode_csv,
        summary_json,
        bootstrap_json,
        plot_png,
    )


def _protocol_artifact() -> ProtocolArtifact:
    return ProtocolArtifact(
        _DEFAULT_CACHE_MODEL_KEY,
        _DEFAULT_COGNITIVE_MAP_NAMESPACE,
        GRID_SCALE,
        (37, 50, 50),
        (0, _OBJECT_CHANNEL_COUNT),
        (_OBJECT_CHANNEL_COUNT, 37),
        ANGLE_ORDER,
        tuple(
            (mapping.sign, int(mapping.offset_degrees))
            for mapping in _declared_mappings()
        ),
        ANGLE_ORDER,
        0.0,
        _DEFAULT_BOOTSTRAP_SEED,
        _DEFAULT_BOOTSTRAP_REPETITIONS,
        "scene_id",
        "episode-macro scene multiplicity ratio-of-sums",
        "np.percentile[2.5,97.5]; numpy-1.24 linear",
        _GATE_THRESHOLD,
        (
            "schema_valid:bool",
            "start_direction:float32[2]",
            "predicted_grid:bool[37,50,50]",
            "predicted_directions:float32[5,2]",
        ),
        (
            "pivot=start_position/(CELL_SIZE*GRID_SCALE)",
            "grid_angle=physical-positive",
            "stored_direction_angle=-grid_angle",
            "warp=padded-nearest-no-crop",
        ),
        (
            "clean-head",
            "development-coupled-load-and-fingerprint",
            "development-lock",
            "test-assignment-coupled-load-and-fingerprint",
            "assignment-serialize-and-hash",
            "test-target-coupled-load-and-fingerprint",
            "score",
            "post-score-source-recheck",
            "audit-and-decision",
            "artifact-build",
            "atomic-no-replace-publish-and-validate",
        ),
        (
            "selector_mean>=0.01",
            "selector_ci_lower>0",
            "selector_object_delta>0",
            "selector_region_delta>0",
            "selector_loso_min>0",
            "selector_global_ci_lower>0",
            "global_mean>=0.01",
            "global_ci_lower>0",
            "global_object_delta>0",
            "global_region_delta>0",
            "global_loso_min>0",
            "audit_passed",
            "partial=(selector_mean>=0.01 and selector_ci_lower>0) or (global_mean>=0.01 and global_ci_lower>0)",
            "precedence=SELECTOR GO,FIXED-CORRECTION GO,PARTIAL,NO GO",
        ),
        _SCHEMA_VERSION,
        (
            "offline raster overlap is not navigation performance",
            "oracle and target-vector diagnostics are nondeployable",
            "val_unseen is a reused evaluation population, not a pristine held-out generalisation test",
        ),
    )


def _schema_artifact() -> SchemaArtifact:
    return SchemaArtifact(
        CsvHeadersArtifact(
            DEVELOPMENT_MAPPING_SCORES_HEADER,
            TEST_ASSIGNMENTS_HEADER,
            TEST_ANGLE_SCORES_HEADER,
            TEST_EPISODE_SCORES_HEADER,
        ),
        JsonTopLevelKeysArtifact(
            SELECTOR_LOCK_KEYS,
            SUMMARY_KEYS,
            BOOTSTRAP_KEYS,
            MANIFEST_KEYS,
        ),
        ".17g finite floats; negative zero normalized to 0",
        "UTF-8 lexicographic episode keys, then declared candidate/angle order",
        "UTF-8 sorted compact keys, no NaN, one trailing newline",
    )


def _json_object(payload: bytes, name: str) -> Dict[str, object]:
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise ValueError(f"{name} must contain a JSON object")
    result = cast(Dict[str, object], decoded)
    if _json_bytes(result) != payload:
        raise ValueError(f"{name} must use canonical JSON serialization")
    return result


def _require_object(value: object, name: str) -> Dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(Dict[str, object], value)


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _source_phase_artifact(value: object, name: str) -> SourcePhaseArtifact:
    payload = _require_object(value, name)
    if set(payload) != {"name", "sha256", "entry_count"}:
        raise ValueError(f"{name} has a mismatched key set")
    entry_count = payload["entry_count"]
    if isinstance(entry_count, bool) or not isinstance(entry_count, int):
        raise ValueError(f"{name} entry_count must be an integer")
    return SourcePhaseArtifact(
        _require_string(payload["name"], f"{name} name"),
        _require_string(payload["sha256"], f"{name} sha256"),
        entry_count,
    )


def _parse_csv_bytes(
    payload: bytes, header: tuple[str, ...], name: str
) -> tuple[tuple[str, ...], ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8 CSV") from error
    rows = tuple(tuple(row) for row in csv.reader(StringIO(text, newline="")))
    if not rows or rows[0] != header:
        raise ValueError(f"{name} has a mismatched exact header")
    text_fields = {"candidate_kind", "scene_id", "example_id"}
    boolean_fields = {"selected", "schema_valid"}
    for row in rows[1:]:
        if len(row) != len(header) or any(cell == "" for cell in row):
            raise ValueError(f"{name} contains an empty or wrong-width row")
        for field, cell in zip(header, row):
            if field in text_fields:
                continue
            if field in boolean_fields:
                if cell not in {"true", "false"}:
                    raise ValueError(f"{name} contains a non-canonical boolean")
                continue
            try:
                value = float(cell)
            except ValueError as error:
                raise ValueError(f"{name} contains a non-numeric cell") from error
            if not np.isfinite(value):
                raise ValueError(f"{name} contains a non-finite numeric cell")
    return rows


def _sources_from_manifest(manifest: Dict[str, object]) -> SourcesArtifact:
    sources = _require_object(manifest.get("sources"), "manifest sources")
    if set(sources) != {
        "development_all_sources",
        "test_assignment_sources",
        "test_target_sources",
        "combined_sha256",
    }:
        raise ValueError("manifest sources have a mismatched key set")
    expected = SourcesArtifact(
        SourcePhaseArtifact(
            "development_all_sources", _SOURCE_PHASE_CONTRACTS["development_all_sources"][1],
            _SOURCE_PHASE_CONTRACTS["development_all_sources"][0],
        ),
        SourcePhaseArtifact(
            "test_assignment_sources", _SOURCE_PHASE_CONTRACTS["test_assignment_sources"][1],
            _SOURCE_PHASE_CONTRACTS["test_assignment_sources"][0],
        ),
        SourcePhaseArtifact(
            "test_target_sources", _SOURCE_PHASE_CONTRACTS["test_target_sources"][1],
            _SOURCE_PHASE_CONTRACTS["test_target_sources"][0],
        ),
        _COMBINED_SOURCE_SHA256,
    )
    result = SourcesArtifact(
        _source_phase_artifact(
            sources["development_all_sources"], "development source phase"
        ),
        _source_phase_artifact(
            sources["test_assignment_sources"], "test assignment source phase"
        ),
        _source_phase_artifact(
            sources["test_target_sources"], "test target source phase"
        ),
        _require_string(sources["combined_sha256"], "combined source sha256"),
    )
    if result != expected:
        raise ValueError("manifest source phases must match frozen constants")
    return result


def _expected_artifacts(
    validation_inputs: ValidationInputs,
    sources: SourcesArtifact,
    git_commit: str,
) -> ArtifactBundle:
    development_runtime, development_targets, development_inputs, development_phase = (
        _load_runtime_population_coupled(
        "val_seen",
        validation_inputs.development_contract,
        validation_inputs.cache_dir,
        validation_inputs.cache_model_key,
        validation_inputs.cognitive_map_namespace,
        include_targets=True,
        )
    )
    _require_phase_contract(development_phase)
    selector_lock = develop_selector(development_targets)
    test_runtime, _, assignment_inputs, assignment_phase = (
        _load_runtime_population_coupled(
        "val_unseen",
        validation_inputs.test_contract,
        validation_inputs.cache_dir,
        validation_inputs.cache_model_key,
        validation_inputs.cognitive_map_namespace,
        include_targets=False,
        )
    )
    _require_phase_contract(assignment_phase)
    assignments = assign_population(test_runtime, selector_lock)
    sealed_assignment_bytes = assignment_csv_bytes(assignments)
    sealed_assignment_sha256 = sha256(sealed_assignment_bytes).hexdigest()
    test_targets, target_inputs, target_phase = _load_test_targets_coupled(
        test_runtime, validation_inputs.cognitive_map_namespace
    )
    _require_phase_contract(target_phase)
    combined = _require_combined_sources(
        (development_phase, assignment_phase, target_phase)
    )
    angle_rows, episode_rows = score_population(test_targets, assignments)
    recomputed_sources = SourcesArtifact(
        development_phase, assignment_phase, target_phase, combined
    )
    if recomputed_sources != sources:
        raise ValueError("source inventories do not match the artifact manifest")
    all_inputs = (*development_inputs, *assignment_inputs, *target_inputs)
    audit = AuditSummary(
        _phase_matches_contract(development_phase),
        _phase_matches_contract(assignment_phase),
        True,
        sealed_assignment_sha256,
        _phase_matches_contract(target_phase),
        combined == _COMBINED_SOURCE_SHA256,
        _post_score_sources_verified(all_inputs),
        len(development_runtime) == validation_inputs.development_contract.episodes,
        len(test_runtime) == validation_inputs.test_contract.episodes,
        True,
    )
    population = PopulationArtifact(
        PopulationSummary(
            validation_inputs.development_contract.episodes,
            validation_inputs.development_contract.scenes,
            validation_inputs.development_contract.valid,
            validation_inputs.development_contract.invalid,
        ),
        PopulationSummary(
            validation_inputs.test_contract.episodes,
            validation_inputs.test_contract.scenes,
            validation_inputs.test_contract.valid,
            validation_inputs.test_contract.invalid,
        ),
    )
    summary = _build_summary_artifact(angle_rows, episode_rows, population, audit)
    bootstrap = _build_bootstrap_artifact(
        episode_rows,
        seed=_DEFAULT_BOOTSTRAP_SEED,
        repetitions=_DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    return build_artifacts(
        selector_lock,
        assignments,
        angle_rows,
        episode_rows,
        summary,
        bootstrap,
        ManifestInputs(
            sources,
            git_commit,
            _protocol_artifact(),
            population,
            _schema_artifact(),
        ),
    )


def _validate_artifact_directory(
    output_dir: Path, validation_inputs: ValidationInputs
) -> None:
    if not output_dir.is_dir():
        raise ValueError(f"artifact directory does not exist: {output_dir}")
    expected_names = set(
        ArtifactBundle(b"", b"", b"", b"", b"", b"", b"", b"", b"").files()
    )
    if {path.name for path in output_dir.iterdir()} != expected_names:
        raise ValueError("artifact directory has a mismatched exact file set")
    actual = ArtifactBundle(
        (output_dir / "manifest.json").read_bytes(),
        (output_dir / "development_mapping_scores.csv").read_bytes(),
        (output_dir / "selector_lock.json").read_bytes(),
        (output_dir / "test_assignments.csv").read_bytes(),
        (output_dir / "test_angle_scores.csv").read_bytes(),
        (output_dir / "test_episode_scores.csv").read_bytes(),
        (output_dir / "summary.json").read_bytes(),
        (output_dir / "bootstrap.json").read_bytes(),
        (output_dir / "target_free_control_intervals.png").read_bytes(),
    )
    manifest = _json_object(actual.manifest_json, "manifest.json")
    if set(manifest) != set(MANIFEST_KEYS):
        raise ValueError("manifest.json has a mismatched exact key set")
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("manifest schema_version must match the frozen v3 constant")
    if _json_bytes(manifest.get("protocol")) != _json_bytes(
        asdict(_protocol_artifact())
    ):
        raise ValueError("manifest protocol must match frozen constants")
    if _json_bytes(manifest.get("schemas")) != _json_bytes(asdict(_schema_artifact())):
        raise ValueError("manifest schemas must match frozen constants")
    if (
        validation_inputs.cache_model_key != _DEFAULT_CACHE_MODEL_KEY
        or validation_inputs.cognitive_map_namespace
        != _DEFAULT_COGNITIVE_MAP_NAMESPACE
    ):
        raise ValueError("validation inputs must match frozen cache constants")
    sources = _sources_from_manifest(manifest)
    git_commit = _require_string(manifest.get("git_commit"), "manifest git_commit")
    if len(git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_commit
    ):
        raise ValueError("manifest git_commit must be a lowercase 40-digit hash")
    if git_commit != validation_inputs.expected_git_commit:
        raise ValueError("manifest git_commit does not match expected_git_commit")
    for name in ("selector_lock.json", "summary.json", "bootstrap.json"):
        _json_object(actual.files()[name], name)
    for name, header in (
        ("development_mapping_scores.csv", DEVELOPMENT_MAPPING_SCORES_HEADER),
        ("test_assignments.csv", TEST_ASSIGNMENTS_HEADER),
        ("test_angle_scores.csv", TEST_ANGLE_SCORES_HEADER),
        ("test_episode_scores.csv", TEST_EPISODE_SCORES_HEADER),
    ):
        _parse_csv_bytes(actual.files()[name], header, name)
    try:
        image = plt.imread(
            BytesIO(actual.target_free_control_intervals_png), format="png"
        )
    except Exception as error:
        raise ValueError("target_free_control_intervals.png must decode") from error
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("target_free_control_intervals.png must be a finite RGB image")
    expected = _expected_artifacts(validation_inputs, sources, git_commit)
    for name, payload in actual.files().items():
        if payload != expected.files()[name]:
            raise ValueError(f"{name} does not match independent semantic recomputation")


def validate_artifact_directory(
    output_dir: Path,
    *,
    expected_git_commit: str,
    expected_development_episodes: int,
    expected_development_scenes: int,
    expected_development_valid: int,
    expected_development_invalid: int,
    expected_test_episodes: int,
    expected_test_scenes: int,
    expected_test_valid: int,
    expected_test_invalid: int,
) -> None:
    """Independently regenerate and validate the complete artifact transaction."""
    manifest = _json_object((output_dir / "manifest.json").read_bytes(), "manifest.json")
    _sources_from_manifest(manifest)
    _validate_artifact_directory(
        output_dir,
        ValidationInputs(
            PopulationContract(
                "val_seen",
                expected_development_episodes,
                expected_development_scenes,
                expected_development_valid,
                expected_development_invalid,
                _DEVELOPMENT_SOURCE_SHA256,
            ),
            PopulationContract(
                "val_unseen",
                expected_test_episodes,
                expected_test_scenes,
                expected_test_valid,
                expected_test_invalid,
                _TEST_SOURCE_SHA256,
            ),
            _DEFAULT_CACHE_DIR,
            _DEFAULT_CACHE_MODEL_KEY,
            _DEFAULT_COGNITIVE_MAP_NAMESPACE,
            expected_git_commit,
            True,
        ),
    )


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically move a path without replacing a concurrent destination."""
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = library.renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    ) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number, os.strerror(error_number), str(destination)
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def publish_artifacts(
    output_dir: Path,
    artifacts: ArtifactBundle,
    validation_inputs: ValidationInputs,
) -> None:
    """Validate one temporary sibling and atomically publish it."""
    if output_dir.exists():
        raise FileExistsError(f"official output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        for name, payload in artifacts.files().items():
            (temporary / name).write_bytes(payload)
        _validate_artifact_directory(temporary, validation_inputs)
        _rename_noreplace(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_args(args: TargetFreeArgs) -> None:
    if args.cache_dir != _DEFAULT_CACHE_DIR:
        raise ValueError(f"cache_dir must be exactly {_DEFAULT_CACHE_DIR}")
    if args.cache_model_key != _DEFAULT_CACHE_MODEL_KEY:
        raise ValueError(f"cache_model_key must be exactly {_DEFAULT_CACHE_MODEL_KEY!r}")
    if args.cognitive_map_namespace != _DEFAULT_COGNITIVE_MAP_NAMESPACE:
        raise ValueError(
            "cognitive_map_namespace must be exactly "
            f"{_DEFAULT_COGNITIVE_MAP_NAMESPACE!r}"
        )
    if args.output_dir != _DEFAULT_OUTPUT_DIR:
        raise ValueError(f"output_dir must be exactly {_DEFAULT_OUTPUT_DIR}")
    if tuple(args.angles) != ANGLE_ORDER:
        raise ValueError(f"angles must be exactly {ANGLE_ORDER}")
    if args.bootstrap_seed != _DEFAULT_BOOTSTRAP_SEED:
        raise ValueError(f"bootstrap_seed must be exactly {_DEFAULT_BOOTSTRAP_SEED}")
    if args.bootstrap_repetitions != _DEFAULT_BOOTSTRAP_REPETITIONS:
        raise ValueError(
            "bootstrap_repetitions must be exactly "
            f"{_DEFAULT_BOOTSTRAP_REPETITIONS}"
        )
    if args.gate_threshold != _GATE_THRESHOLD:
        raise ValueError(f"gate_threshold must be exactly {_GATE_THRESHOLD}")


def _preflight_git_commit() -> str:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("fixed experiment requires a clean committed worktree")
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise ValueError("git rev-parse HEAD must return a full commit hash")
    return commit


def _run(args: TargetFreeArgs) -> None:
    _validate_args(args)
    if args.output_dir.exists():
        raise FileExistsError(f"official output already exists: {args.output_dir}")
    run_commit = _preflight_git_commit()
    development_runtime, development_targets, development_inputs, development_phase = (
        _load_runtime_population_coupled(
            "val_seen",
            VAL_SEEN_POPULATION,
            args.cache_dir,
            args.cache_model_key,
            args.cognitive_map_namespace,
            include_targets=True,
        )
    )
    _require_phase_contract(development_phase)
    selector_lock = develop_selector(development_targets)
    test_runtime, _, assignment_inputs, assignment_phase = (
        _load_runtime_population_coupled(
            "val_unseen",
            VAL_UNSEEN_POPULATION,
            args.cache_dir,
            args.cache_model_key,
            args.cognitive_map_namespace,
            include_targets=False,
        )
    )
    _require_phase_contract(assignment_phase)
    assignments = assign_population(test_runtime, selector_lock)
    assignment_sha256 = sha256(assignment_csv_bytes(assignments)).hexdigest()
    test_targets, target_inputs, target_phase = _load_test_targets_coupled(
        test_runtime, args.cognitive_map_namespace
    )
    _require_phase_contract(target_phase)
    combined_sha256 = _require_combined_sources(
        (development_phase, assignment_phase, target_phase)
    )
    angle_rows, episode_rows = score_population(test_targets, assignments)
    sources = SourcesArtifact(
        development_phase, assignment_phase, target_phase, combined_sha256
    )
    audit_checks = (
        _phase_matches_contract(development_phase),
        _phase_matches_contract(assignment_phase),
        True,
        _phase_matches_contract(target_phase),
        combined_sha256 == _COMBINED_SOURCE_SHA256,
        _post_score_sources_verified(
            (*development_inputs, *assignment_inputs, *target_inputs)
        ),
        len(development_runtime) == VAL_SEEN_POPULATION.episodes,
        len(test_runtime) == VAL_UNSEEN_POPULATION.episodes,
    )
    audit = AuditSummary(
        audit_checks[0],
        audit_checks[1],
        audit_checks[2],
        assignment_sha256,
        audit_checks[3],
        audit_checks[4],
        audit_checks[5],
        audit_checks[6],
        audit_checks[7],
        all(audit_checks),
    )
    population = PopulationArtifact(
        PopulationSummary(778, 53, 770, 8),
        PopulationSummary(1_839, 11, 1_830, 9),
    )
    summary = _build_summary_artifact(angle_rows, episode_rows, population, audit)
    bootstrap = _build_bootstrap_artifact(
        episode_rows,
        seed=args.bootstrap_seed,
        repetitions=args.bootstrap_repetitions,
    )
    artifacts = build_artifacts(
        selector_lock,
        assignments,
        angle_rows,
        episode_rows,
        summary,
        bootstrap,
        ManifestInputs(
            sources,
            run_commit,
            _protocol_artifact(),
            population,
            _schema_artifact(),
        ),
    )
    validation_inputs = ValidationInputs(
        VAL_SEEN_POPULATION,
        VAL_UNSEEN_POPULATION,
        args.cache_dir,
        args.cache_model_key,
        args.cognitive_map_namespace,
        run_commit,
        args.quiet,
    )
    publish_artifacts(args.output_dir, artifacts, validation_inputs)
    validate_artifact_directory(
        args.output_dir,
        expected_git_commit=run_commit,
        expected_development_episodes=778,
        expected_development_scenes=53,
        expected_development_valid=770,
        expected_development_invalid=8,
        expected_test_episodes=1_839,
        expected_test_scenes=11,
        expected_test_valid=1_830,
        expected_test_invalid=9,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = TargetFreeArgs(underscores_to_dashes=True).parse_args(argv)
    _run(args)
    if not args.quiet:
        print(f"Wrote target-free cardinal-selector artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
