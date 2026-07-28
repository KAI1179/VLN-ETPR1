"""Typed target-free cardinal selector and soft-raster aggregation core."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from io import StringIO
from numbers import Integral, Real
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

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
    selector_object_mean: float = 1.0,
    selector_region_mean: float = 1.0,
    global_object_mean: float = 1.0,
    global_region_mean: float = 1.0,
    audit_passed: bool = False,
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
