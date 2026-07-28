"""Typed object/region cross-fit scoring for LLM-Grid transform controls."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from io import BytesIO, StringIO
import json
from numbers import Integral, Real
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tap import Tap

from prior.analyze.llm_grid_registration import (
    RasterScore,
    SpatialBounds,
    WarpedGrid,
    score_warped_grid,
    warp_grid_about_pivot,
)
from prior.constants import CELL_SIZE
from vlnce_baselines.models.etp_llm.llm_grid_train import (
    GRID_SCALE,
    GRID_SHAPE,
    LLMGridDataset,
    LLMGridValidationError,
    load_llm_grid_examples,
    parse_grid_text,
)
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_prediction_path,
    llm_navigation_split_dir,
    validate_llm_navigation_manifest,
)


_CHANNEL_COUNT = 37
_OBJECT_CHANNEL_COUNT = 27
_GRID_SHAPE = GRID_SHAPE
_EXPECTED_EPISODES = 1_839
_EXPECTED_SCENES = 11
_EXPECTED_VALID_EPISODES = 1_830
_EXPECTED_INVALID_EPISODES = 9
_DEFAULT_ANGLES = (0.0, 90.0, 180.0, 270.0)
_DEFAULT_CACHE_MODEL_KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2"
_DEFAULT_COGNITIVE_MAP_NAMESPACE = "gt.legacy.r1p5.direction5.v1"
_DEFAULT_SHUFFLE_SEED = 43
_DEFAULT_BOOTSTRAP_REPETITIONS = 10_000
_DEFAULT_BOOTSTRAP_SEED = 42

_ANGLE_SCORE_HEADER = (
    "split",
    "scene_id",
    "example_id",
    "schema_valid",
    "pivot_mode",
    "pivot_row",
    "pivot_column",
    "donor_example_id",
    "angle_degrees",
    "bounds_row_min",
    "bounds_row_max",
    "bounds_col_min",
    "bounds_col_max",
    "object_input_support",
    "object_intersection",
    "object_union",
    "object_predicted_support",
    "object_target_support",
    "object_in_frame_support",
    "object_out_of_frame_support",
    "object_iou",
    "region_input_support",
    "region_intersection",
    "region_union",
    "region_predicted_support",
    "region_target_support",
    "region_in_frame_support",
    "region_out_of_frame_support",
    "region_iou",
)
_CROSSFIT_RESULT_HEADER = (
    "split",
    "scene_id",
    "example_id",
    "schema_valid",
    "pivot_mode",
    "pivot_row",
    "pivot_column",
    "donor_example_id",
    "direction",
    "selected_angle_degrees",
    "second_angle_degrees",
    "selector_identity_iou",
    "selector_selected_iou",
    "selector_second_iou",
    "selector_margin",
    "heldout_identity_iou",
    "heldout_selected_iou",
    "delta_iou",
    "selector_predicted_support_empty",
    "selector_target_support_empty",
    "selector_union_empty",
    "heldout_predicted_support_empty",
    "heldout_target_support_empty",
    "heldout_union_empty",
)
_PIVOT_ASSIGNMENT_HEADER = (
    "scene_id",
    "example_id",
    "donor_example_id",
    "true_pivot_row",
    "true_pivot_column",
    "assigned_pivot_row",
    "assigned_pivot_column",
)
_ROW_METADATA_FIELDS = (
    "split",
    "scene_id",
    "example_id",
    "schema_valid",
    "pivot_mode",
    "pivot_row",
    "pivot_column",
    "donor_example_id",
)


class CrossFitArgs(Tap):
    """Fixed inputs for the full R2R cross-fit registration control."""

    cache_dir: Path = Path("data/llm_navigation")
    cache_model_key: str = _DEFAULT_CACHE_MODEL_KEY
    cognitive_map_namespace: str = _DEFAULT_COGNITIVE_MAP_NAMESPACE
    output_dir: Path = Path(
        "outputs/llm_grid_analysis/"
        "start_centered_crossfit_controls_r2r_rxr_epoch2"
    )
    angles: Tuple[float, ...] = _DEFAULT_ANGLES
    shuffle_seed: int = _DEFAULT_SHUFFLE_SEED
    bootstrap_repetitions: int = _DEFAULT_BOOTSTRAP_REPETITIONS
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED
    quiet: bool = False


def _require_nonempty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _require_iou(value: float, name: str) -> float:
    result = _require_finite_real(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


def _require_boolean_grid(
    grid: NDArray[np.bool_], name: str
) -> NDArray[np.bool_]:
    if not isinstance(grid, np.ndarray) or grid.shape != _GRID_SHAPE:
        raise ValueError(f"{name} must have shape {_GRID_SHAPE}")
    if grid.dtype != np.bool_:
        raise ValueError(f"{name} must have boolean dtype")
    return grid


def _require_pivot(pivot: tuple[float, float], name: str) -> tuple[float, float]:
    if len(pivot) != 2:
        raise ValueError(f"{name} must contain row and column coordinates")
    return (
        _require_finite_real(pivot[0], f"{name} row"),
        _require_finite_real(pivot[1], f"{name} column"),
    )


def _is_close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))


class PivotMode(str, Enum):
    TRUE_START = "true_start"
    MAP_CENTER = "map_center"
    SHUFFLED_START = "shuffled_start"


class Direction(str, Enum):
    OBJECT_TO_REGION = "object_to_region"
    REGION_TO_OBJECT = "region_to_object"


class GateDecision(str, Enum):
    GO = "GO"
    PARTIAL_EVIDENCE = "partial_evidence"
    NO_GO = "NO_GO"


@dataclass(frozen=True)
class _GateConditions:
    true_start_symmetric_mean_at_least_0_01: bool
    true_start_symmetric_ci_lower_above_zero: bool
    both_true_start_directional_means_above_zero: bool
    true_start_minus_map_center_ci_lower_above_zero: bool
    true_start_minus_shuffled_ci_lower_above_zero: bool

    def __post_init__(self) -> None:
        for value in (
            self.true_start_symmetric_mean_at_least_0_01,
            self.true_start_symmetric_ci_lower_above_zero,
            self.both_true_start_directional_means_above_zero,
            self.true_start_minus_map_center_ci_lower_above_zero,
            self.true_start_minus_shuffled_ci_lower_above_zero,
        ):
            if not isinstance(value, bool):
                raise ValueError("gate conditions must be booleans")

    def payload(self) -> Dict[str, object]:
        return {
            "true_start_symmetric_mean_at_least_0_01": (
                self.true_start_symmetric_mean_at_least_0_01
            ),
            "true_start_symmetric_ci_lower_above_zero": (
                self.true_start_symmetric_ci_lower_above_zero
            ),
            "both_true_start_directional_means_above_zero": (
                self.both_true_start_directional_means_above_zero
            ),
            "true_start_minus_map_center_ci_lower_above_zero": (
                self.true_start_minus_map_center_ci_lower_above_zero
            ),
            "true_start_minus_shuffled_ci_lower_above_zero": (
                self.true_start_minus_shuffled_ci_lower_above_zero
            ),
        }


@dataclass(frozen=True)
class EndpointEstimate:
    mean: float
    ci_lower: float
    ci_upper: float
    leave_one_scene_out_min: float
    leave_one_scene_out_max: float

    def __post_init__(self) -> None:
        _require_finite_real(self.mean, "mean")
        ci_lower = _require_finite_real(self.ci_lower, "ci_lower")
        ci_upper = _require_finite_real(self.ci_upper, "ci_upper")
        loso_min = _require_finite_real(
            self.leave_one_scene_out_min, "leave_one_scene_out_min"
        )
        loso_max = _require_finite_real(
            self.leave_one_scene_out_max, "leave_one_scene_out_max"
        )
        if ci_lower > ci_upper:
            raise ValueError("ci_lower must not exceed ci_upper")
        if loso_min > loso_max:
            raise ValueError(
                "leave_one_scene_out_min must not exceed leave_one_scene_out_max"
            )


@dataclass(frozen=True)
class InferenceEstimates:
    true_start_object_to_region: EndpointEstimate
    true_start_region_to_object: EndpointEstimate
    true_start_symmetric: EndpointEstimate
    map_center_object_to_region: EndpointEstimate
    map_center_region_to_object: EndpointEstimate
    map_center_symmetric: EndpointEstimate
    shuffled_start_object_to_region: EndpointEstimate
    shuffled_start_region_to_object: EndpointEstimate
    shuffled_start_symmetric: EndpointEstimate
    true_start_minus_map_center: EndpointEstimate
    true_start_minus_shuffled: EndpointEstimate

    def __post_init__(self) -> None:
        if not all(isinstance(estimate, EndpointEstimate) for estimate in self.all()):
            raise ValueError("inference estimates must contain EndpointEstimate values")

    def all(self) -> Tuple[EndpointEstimate, ...]:
        return (
            self.true_start_object_to_region,
            self.true_start_region_to_object,
            self.true_start_symmetric,
            self.map_center_object_to_region,
            self.map_center_region_to_object,
            self.map_center_symmetric,
            self.shuffled_start_object_to_region,
            self.shuffled_start_region_to_object,
            self.shuffled_start_symmetric,
            self.true_start_minus_map_center,
            self.true_start_minus_shuffled,
        )


@dataclass(frozen=True)
class InferenceRanges:
    true_start_object_to_region: Tuple[float, float]
    true_start_region_to_object: Tuple[float, float]
    true_start_symmetric: Tuple[float, float]
    map_center_object_to_region: Tuple[float, float]
    map_center_region_to_object: Tuple[float, float]
    map_center_symmetric: Tuple[float, float]
    shuffled_start_object_to_region: Tuple[float, float]
    shuffled_start_region_to_object: Tuple[float, float]
    shuffled_start_symmetric: Tuple[float, float]
    true_start_minus_map_center: Tuple[float, float]
    true_start_minus_shuffled: Tuple[float, float]

    def __post_init__(self) -> None:
        for bounds in self.all():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError("leave-one-scene-out ranges must be two-item tuples")
            lower = _require_finite_real(bounds[0], "leave-one-scene-out lower")
            upper = _require_finite_real(bounds[1], "leave-one-scene-out upper")
            if lower > upper:
                raise ValueError("leave-one-scene-out lower must not exceed upper")

    def all(self) -> Tuple[Tuple[float, float], ...]:
        return (
            self.true_start_object_to_region,
            self.true_start_region_to_object,
            self.true_start_symmetric,
            self.map_center_object_to_region,
            self.map_center_region_to_object,
            self.map_center_symmetric,
            self.shuffled_start_object_to_region,
            self.shuffled_start_region_to_object,
            self.shuffled_start_symmetric,
            self.true_start_minus_map_center,
            self.true_start_minus_shuffled,
        )


@dataclass(frozen=True)
class PivotSummary:
    object_to_region_mean: float
    region_to_object_mean: float
    symmetric_mean: float
    object_to_region_angle_counts: Tuple[Tuple[float, int], ...]
    region_to_object_angle_counts: Tuple[Tuple[float, int], ...]
    object_to_region_empty_rates: Tuple[float, float, float, float, float, float]
    region_to_object_empty_rates: Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class EpisodeCase:
    split: str
    scene_id: str
    example_id: str
    schema_valid: bool
    predicted_grid: NDArray[np.bool_]
    target_grid: NDArray[np.bool_]
    true_start_pivot: tuple[float, float]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.split, "split")
        _require_nonempty_string(self.scene_id, "scene_id")
        _require_nonempty_string(self.example_id, "example_id")
        if not isinstance(self.schema_valid, bool):
            raise ValueError("schema_valid must be a boolean")
        predicted = _require_boolean_grid(self.predicted_grid, "predicted_grid")
        target = _require_boolean_grid(self.target_grid, "target_grid")
        if predicted.shape[1:] != target.shape[1:]:
            raise ValueError("predicted_grid and target_grid must have the same spatial shape")
        _require_pivot(self.true_start_pivot, "true_start_pivot")


def _validate_args(args: CrossFitArgs) -> None:
    """Reject every override that would change the frozen control experiment."""
    if args.cache_model_key != _DEFAULT_CACHE_MODEL_KEY:
        raise ValueError(f"cache_model_key must be exactly {_DEFAULT_CACHE_MODEL_KEY!r}")
    if args.cognitive_map_namespace != _DEFAULT_COGNITIVE_MAP_NAMESPACE:
        raise ValueError(
            "cognitive_map_namespace must be exactly "
            f"{_DEFAULT_COGNITIVE_MAP_NAMESPACE!r}"
        )
    if tuple(args.angles) != _DEFAULT_ANGLES:
        raise ValueError(f"angles must be exactly {_DEFAULT_ANGLES}")
    if args.shuffle_seed != _DEFAULT_SHUFFLE_SEED:
        raise ValueError(f"shuffle_seed must be exactly {_DEFAULT_SHUFFLE_SEED}")
    if args.bootstrap_repetitions != _DEFAULT_BOOTSTRAP_REPETITIONS:
        raise ValueError(
            "bootstrap_repetitions must be exactly "
            f"{_DEFAULT_BOOTSTRAP_REPETITIONS}"
        )
    if args.bootstrap_seed != _DEFAULT_BOOTSTRAP_SEED:
        raise ValueError(f"bootstrap_seed must be exactly {_DEFAULT_BOOTSTRAP_SEED}")
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    if (
        cache_dir == output_dir
        or cache_dir in output_dir.parents
        or output_dir in cache_dir.parents
    ):
        raise ValueError("output_dir and cache_dir must not overlap")


def _prediction_manifest_path(args: CrossFitArgs) -> Path:
    """Validate and return the manifest for the one permitted prediction cache."""
    _validate_args(args)
    split_dir = llm_navigation_split_dir(
        "R2R", "val_unseen", cache_dir=args.cache_dir, model_key=args.cache_model_key
    )
    validate_llm_navigation_manifest(
        split_dir,
        {
            "dataset": "R2R",
            "split": "val_unseen",
            "generator": "llm-grid",
            "scale": GRID_SCALE,
            "cache_model_key": args.cache_model_key,
        },
    )
    return split_dir / "manifest.json"


def _start_pivot(start_position: Sequence[float]) -> tuple[float, float]:
    """Convert the two level-local start coordinates to the scale-two grid."""
    if len(start_position) != 2:
        raise ValueError("start_position must contain exactly two coordinates")
    try:
        row_m = float(start_position[0])
        column_m = float(start_position[1])
    except (TypeError, ValueError) as error:
        raise ValueError("start_position coordinates must be finite") from error
    if not np.isfinite(row_m) or not np.isfinite(column_m):
        raise ValueError("start_position coordinates must be finite")
    return row_m / (CELL_SIZE * GRID_SCALE), column_m / (CELL_SIZE * GRID_SCALE)


def _validate_episode_population(cases: Sequence[EpisodeCase]) -> None:
    """Require the complete fixed R2R val_unseen population before scoring."""
    identities: Set[tuple[str, str]] = set()
    valid_count = 0
    invalid_count = 0
    scene_ids: Set[str] = set()
    for case in cases:
        if not isinstance(case, EpisodeCase):
            raise ValueError("population must contain EpisodeCase rows")
        if case.split != "val_unseen":
            raise ValueError("population must contain only val_unseen rows")
        identity = (case.scene_id, case.example_id)
        if identity in identities:
            raise ValueError("population must have unique scene/example identities")
        identities.add(identity)
        scene_ids.add(case.scene_id)
        if case.schema_valid:
            valid_count += 1
        else:
            invalid_count += 1
            if np.any(case.predicted_grid):
                raise ValueError("invalid predictions must use an empty grid")
    if len(cases) != _EXPECTED_EPISODES:
        raise ValueError(f"population must contain exactly {_EXPECTED_EPISODES} rows")
    if len(scene_ids) != _EXPECTED_SCENES:
        raise ValueError(f"population must contain exactly {_EXPECTED_SCENES} scenes")
    if valid_count != _EXPECTED_VALID_EPISODES:
        raise ValueError(
            f"population must contain exactly {_EXPECTED_VALID_EPISODES} valid rows"
        )
    if invalid_count != _EXPECTED_INVALID_EPISODES:
        raise ValueError(
            f"population must contain exactly {_EXPECTED_INVALID_EPISODES} invalid rows"
        )


def _load_episode_cases(
    args: CrossFitArgs,
) -> tuple[tuple[EpisodeCase, ...], Path]:
    """Load the fixed R2R population without repairing unexpected failures."""
    _validate_args(args)
    prediction_manifest_path = _prediction_manifest_path(args)
    loaded = load_llm_grid_examples(
        ("val_unseen",),
        quiet=args.quiet,
        cognitive_map_namespace=args.cognitive_map_namespace,
        datasets=("R2R",),
    )
    dataset = LLMGridDataset(loaded.examples, scale=GRID_SCALE)
    if len(dataset) != len(loaded.examples):
        raise ValueError("LLM-Grid dataset cardinality changed during cross-fit loading")
    cases: list[EpisodeCase] = []
    for example, item in zip(loaded.examples, dataset):
        if example.dataset != "R2R" or example.split != "val_unseen":
            raise ValueError("LLM-Grid loader returned a non-R2R val_unseen example")
        if item["example_id"] != example.example_id:
            raise ValueError("LLM-Grid dataset order changed during cross-fit loading")
        target_grid = item["target_grid"] > 0
        prediction_path = llm_navigation_prediction_path(
            example.scene_id,
            example.example_id,
            "R2R",
            "val_unseen",
            cache_dir=args.cache_dir,
            model_key=args.cache_model_key,
        )
        schema_valid = True
        predicted_grid = np.zeros(GRID_SHAPE, dtype=np.bool_)
        try:
            parsed = parse_grid_text(
                prediction_path.read_text(encoding="utf-8"), shape=GRID_SHAPE
            )
            predicted_grid = parsed.grid > 0
        except (FileNotFoundError, LLMGridValidationError):
            schema_valid = False
        cases.append(
            EpisodeCase(
                split="val_unseen",
                scene_id=example.scene_id,
                example_id=example.example_id,
                schema_valid=schema_valid,
                predicted_grid=predicted_grid,
                target_grid=target_grid,
                true_start_pivot=_start_pivot(item["start_position"]),
            )
        )
    result = tuple(cases)
    _validate_episode_population(result)
    return result, prediction_manifest_path


@dataclass(frozen=True)
class FamilyAngleScore:
    angle_degrees: float
    bounds: SpatialBounds
    object_input_support: int
    region_input_support: int
    object_score: RasterScore
    region_score: RasterScore

    def __post_init__(self) -> None:
        _require_finite_real(self.angle_degrees, "angle_degrees")
        if not isinstance(self.bounds, SpatialBounds):
            raise ValueError("bounds must be a SpatialBounds")
        _require_nonnegative_int(self.object_input_support, "object_input_support")
        _require_nonnegative_int(self.region_input_support, "region_input_support")
        if not isinstance(self.object_score, RasterScore):
            raise ValueError("object_score must be a RasterScore")
        if not isinstance(self.region_score, RasterScore):
            raise ValueError("region_score must be a RasterScore")


@dataclass(frozen=True)
class CrossFitResult:
    direction: Direction
    selected_angle_degrees: float
    second_angle_degrees: float
    selector_identity_iou: float
    selector_selected_iou: float
    selector_second_iou: float
    selector_margin: float
    heldout_identity_iou: float
    heldout_selected_iou: float
    delta_iou: float
    selector_predicted_support_empty: bool
    selector_target_support_empty: bool
    selector_union_empty: bool
    heldout_predicted_support_empty: bool
    heldout_target_support_empty: bool
    heldout_union_empty: bool

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            raise ValueError("direction must be a Direction")
        _require_finite_real(self.selected_angle_degrees, "selected_angle_degrees")
        _require_finite_real(self.second_angle_degrees, "second_angle_degrees")
        _require_iou(self.selector_identity_iou, "selector_identity_iou")
        selector_selected = _require_iou(
            self.selector_selected_iou, "selector_selected_iou"
        )
        selector_second = _require_iou(
            self.selector_second_iou, "selector_second_iou"
        )
        margin = _require_iou(self.selector_margin, "selector_margin")
        heldout_identity = _require_iou(
            self.heldout_identity_iou, "heldout_identity_iou"
        )
        heldout_selected = _require_iou(
            self.heldout_selected_iou, "heldout_selected_iou"
        )
        delta = _require_finite_real(self.delta_iou, "delta_iou")
        if not -1.0 <= delta <= 1.0:
            raise ValueError("delta_iou must be within [-1, 1]")
        if not _is_close(margin, selector_selected - selector_second):
            raise ValueError("selector_margin must equal selected minus second IoU")
        if not _is_close(delta, heldout_selected - heldout_identity):
            raise ValueError("delta_iou must equal heldout selected minus identity IoU")
        flags = (
            (self.selector_predicted_support_empty, "selector_predicted_support_empty"),
            (self.selector_target_support_empty, "selector_target_support_empty"),
            (self.selector_union_empty, "selector_union_empty"),
            (self.heldout_predicted_support_empty, "heldout_predicted_support_empty"),
            (self.heldout_target_support_empty, "heldout_target_support_empty"),
            (self.heldout_union_empty, "heldout_union_empty"),
        )
        for flag, name in flags:
            if not isinstance(flag, bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class PivotAssignment:
    scene_id: str
    example_id: str
    donor_example_id: str
    true_pivot: tuple[float, float]
    assigned_pivot: tuple[float, float]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.scene_id, "scene_id")
        _require_nonempty_string(self.example_id, "example_id")
        _require_nonempty_string(self.donor_example_id, "donor_example_id")
        _require_pivot(self.true_pivot, "true_pivot")
        _require_pivot(self.assigned_pivot, "assigned_pivot")
        if self.example_id == self.donor_example_id:
            raise ValueError("donor_example_id must differ from example_id")
        if self.true_pivot == self.assigned_pivot:
            raise ValueError("assigned_pivot must differ from true_pivot")


@dataclass(frozen=True)
class EpisodePivotResult:
    split: str
    scene_id: str
    example_id: str
    schema_valid: bool
    pivot_mode: PivotMode
    pivot: tuple[float, float]
    donor_example_id: Optional[str]
    angle_scores: tuple[FamilyAngleScore, ...]
    object_to_region: CrossFitResult
    region_to_object: CrossFitResult

    def __post_init__(self) -> None:
        _require_nonempty_string(self.split, "split")
        _require_nonempty_string(self.scene_id, "scene_id")
        _require_nonempty_string(self.example_id, "example_id")
        if not isinstance(self.schema_valid, bool):
            raise ValueError("schema_valid must be a boolean")
        if not isinstance(self.pivot_mode, PivotMode):
            raise ValueError("pivot_mode must be a PivotMode")
        _require_pivot(self.pivot, "pivot")
        if self.donor_example_id is not None:
            _require_nonempty_string(self.donor_example_id, "donor_example_id")
        if not isinstance(self.angle_scores, tuple) or not self.angle_scores:
            raise ValueError("angle_scores must be a nonempty tuple")
        if not all(isinstance(score, FamilyAngleScore) for score in self.angle_scores):
            raise ValueError("angle_scores must contain only FamilyAngleScore rows")
        if not isinstance(self.object_to_region, CrossFitResult):
            raise ValueError("object_to_region must be a CrossFitResult")
        if not isinstance(self.region_to_object, CrossFitResult):
            raise ValueError("region_to_object must be a CrossFitResult")
        if self.object_to_region.direction is not Direction.OBJECT_TO_REGION:
            raise ValueError("object_to_region must use object-to-region direction")
        if self.region_to_object.direction is not Direction.REGION_TO_OBJECT:
            raise ValueError("region_to_object must use region-to-object direction")

    @property
    def symmetric_delta(self) -> float:
        return 0.5 * (
            self.object_to_region.delta_iou + self.region_to_object.delta_iou
        )


def _identity_bytes(episode: EpisodeCase) -> bytes:
    return (episode.scene_id + "\0" + episode.example_id).encode("utf-8")


def _matching_error(scene_id: str) -> ValueError:
    return ValueError(
        f"scene {scene_id} has no valid shuffled-start perfect matching"
    )


def _build_scene_pivot_assignments(
    scene_id: str, episodes: Sequence[EpisodeCase]
) -> tuple[PivotAssignment, ...]:
    by_identity: Dict[bytes, EpisodeCase] = {}
    for episode in episodes:
        identity = _identity_bytes(episode)
        if identity in by_identity:
            raise ValueError("episodes must have unique scene/example identities")
        by_identity[identity] = episode

    receiver_ids = sorted(
        by_identity,
        key=lambda identity: (
            sha256(b"43\0receiver\0" + identity).digest(),
            identity,
        ),
    )
    donor_ids_by_receiver: Dict[bytes, tuple[bytes, ...]] = {}
    for receiver_id in receiver_ids:
        receiver = by_identity[receiver_id]
        eligible_donors = (
            donor_id
            for donor_id, donor in by_identity.items()
            if receiver_id != donor_id
            and receiver.true_start_pivot != donor.true_start_pivot
        )
        donor_ids_by_receiver[receiver_id] = tuple(
            sorted(
                eligible_donors,
                key=lambda donor_id: (
                    sha256(
                        b"43\0donor\0"
                        + receiver_id
                        + b"\0"
                        + donor_id
                    ).digest(),
                    donor_id,
                ),
            )
        )

    donor_to_receiver: Dict[bytes, bytes] = {}

    def assign(receiver_id: bytes, visited_donors: Set[bytes]) -> bool:
        for donor_id in donor_ids_by_receiver[receiver_id]:
            if donor_id in visited_donors:
                continue
            visited_donors.add(donor_id)
            occupied_receiver = donor_to_receiver.get(donor_id)
            if occupied_receiver is None or assign(occupied_receiver, visited_donors):
                donor_to_receiver[donor_id] = receiver_id
                return True
        return False

    for receiver_id in receiver_ids:
        if not assign(receiver_id, set()):
            raise _matching_error(scene_id)
    if len(donor_to_receiver) != len(episodes):
        raise _matching_error(scene_id)

    receiver_to_donor = {
        receiver_id: donor_id for donor_id, receiver_id in donor_to_receiver.items()
    }
    if set(receiver_to_donor) != set(by_identity):
        raise _matching_error(scene_id)

    assignments = []
    for receiver_id, donor_id in receiver_to_donor.items():
        receiver = by_identity[receiver_id]
        donor = by_identity[donor_id]
        assignments.append(
            PivotAssignment(
                scene_id=scene_id,
                example_id=receiver.example_id,
                donor_example_id=donor.example_id,
                true_pivot=receiver.true_start_pivot,
                assigned_pivot=donor.true_start_pivot,
            )
        )
    return tuple(assignments)


def build_pivot_assignments(
    episodes: Sequence[EpisodeCase],
) -> tuple[PivotAssignment, ...]:
    """Create one deterministic valid shuffled-start donor for every episode."""
    rows = tuple(episodes)
    if not all(isinstance(episode, EpisodeCase) for episode in rows):
        raise ValueError("episodes must contain only EpisodeCase rows")

    by_scene: Dict[str, list[EpisodeCase]] = {}
    identities: Set[bytes] = set()
    for episode in rows:
        identity = _identity_bytes(episode)
        if identity in identities:
            raise ValueError("episodes must have unique scene/example identities")
        identities.add(identity)
        by_scene.setdefault(episode.scene_id, []).append(episode)

    assignments = []
    for scene_id in sorted(by_scene, key=lambda value: value.encode("utf-8")):
        assignments.extend(_build_scene_pivot_assignments(scene_id, by_scene[scene_id]))

    result = tuple(
        sorted(
            assignments,
            key=lambda row: (
                row.scene_id.encode("utf-8"),
                row.example_id.encode("utf-8"),
            ),
        )
    )
    if len(result) != len(rows):
        raise ValueError("pivot assignments must contain one row per episode")
    for scene_id, scene_episodes in by_scene.items():
        scene_rows = tuple(row for row in result if row.scene_id == scene_id)
        donor_ids = {row.donor_example_id for row in scene_rows}
        expected_donor_ids = {episode.example_id for episode in scene_episodes}
        assigned_pivots = Counter(row.assigned_pivot for row in scene_rows)
        expected_pivots = Counter(
            episode.true_start_pivot for episode in scene_episodes
        )
        if (
            len(scene_rows) != len(scene_episodes)
            or donor_ids != expected_donor_ids
            or assigned_pivots != expected_pivots
            or any(
                row.example_id == row.donor_example_id
                or row.true_pivot == row.assigned_pivot
                for row in scene_rows
            )
        ):
            raise _matching_error(scene_id)
    return result


def pivot_for_mode(
    episode: EpisodeCase,
    assignment: PivotAssignment,
    pivot_mode: PivotMode,
) -> tuple[float, float]:
    """Return the single prespecified pivot for one control condition."""
    if not isinstance(episode, EpisodeCase):
        raise ValueError("episode must be an EpisodeCase")
    if not isinstance(assignment, PivotAssignment):
        raise ValueError("assignment must be a PivotAssignment")
    if not isinstance(pivot_mode, PivotMode):
        raise ValueError("pivot_mode must be a PivotMode")
    if (
        assignment.scene_id != episode.scene_id
        or assignment.example_id != episode.example_id
    ):
        raise ValueError("assignment identity does not match episode")
    if assignment.true_pivot != episode.true_start_pivot:
        raise ValueError("assignment true_pivot does not match episode")
    if pivot_mode is PivotMode.TRUE_START:
        return episode.true_start_pivot
    if pivot_mode is PivotMode.MAP_CENTER:
        return (25.0, 25.0)
    return assignment.assigned_pivot

def score_angle_families(
    predicted_grid: NDArray[np.bool_],
    target_grid: NDArray[np.bool_],
    pivot: tuple[float, float],
    angles: tuple[float, ...],
) -> tuple[FamilyAngleScore, ...]:
    """Score one shared warp per angle against object and region target families."""
    predicted = _require_boolean_grid(predicted_grid, "predicted_grid")
    target = _require_boolean_grid(target_grid, "target_grid")
    if predicted.shape[1:] != target.shape[1:]:
        raise ValueError("predicted_grid and target_grid must have the same spatial shape")
    checked_pivot = _require_pivot(pivot, "pivot")
    object_input_support = int(np.count_nonzero(predicted[:_OBJECT_CHANNEL_COUNT]))
    region_input_support = int(np.count_nonzero(predicted[_OBJECT_CHANNEL_COUNT:]))

    scores: list[FamilyAngleScore] = []
    for angle in angles:
        checked_angle = _require_finite_real(angle, "angle_degrees")
        warped = warp_grid_about_pivot(predicted, checked_pivot, checked_angle)
        object_warp = WarpedGrid(
            grid=warped.grid[:_OBJECT_CHANNEL_COUNT],
            bounds=warped.bounds,
            input_support=object_input_support,
        )
        region_warp = WarpedGrid(
            grid=warped.grid[_OBJECT_CHANNEL_COUNT:],
            bounds=warped.bounds,
            input_support=region_input_support,
        )
        scores.append(
            FamilyAngleScore(
                angle_degrees=checked_angle,
                bounds=warped.bounds,
                object_input_support=object_input_support,
                region_input_support=region_input_support,
                object_score=score_warped_grid(
                    object_warp, target[:_OBJECT_CHANNEL_COUNT]
                ),
                region_score=score_warped_grid(
                    region_warp, target[_OBJECT_CHANNEL_COUNT:]
                ),
            )
        )
    return tuple(scores)


def crossfit_scores(
    scores: Sequence[FamilyAngleScore],
    direction: Direction,
) -> CrossFitResult:
    """Select a rotation in one family and score it only in the held-out family."""
    if not isinstance(direction, Direction):
        raise ValueError("direction must be a Direction")
    rows = tuple(scores)
    if len(rows) < 2:
        raise ValueError("scores must contain at least two angle rows")
    if not all(isinstance(row, FamilyAngleScore) for row in rows):
        raise ValueError("scores must contain only FamilyAngleScore rows")

    angles = tuple(row.angle_degrees for row in rows)
    if len(set(angles)) != len(angles):
        raise ValueError("angle_degrees must be unique")
    try:
        identity_index = angles.index(0.0)
    except ValueError as error:
        raise ValueError("scores must contain identity angle 0.0") from error

    if direction is Direction.OBJECT_TO_REGION:
        selector_scores = tuple(row.object_score for row in rows)
        heldout_scores = tuple(row.region_score for row in rows)
    else:
        selector_scores = tuple(row.region_score for row in rows)
        heldout_scores = tuple(row.object_score for row in rows)

    ordered_indices = sorted(
        range(len(rows)), key=lambda index: selector_scores[index].iou, reverse=True
    )
    selected_index, second_index = ordered_indices[:2]
    selector_identity = selector_scores[identity_index]
    selector_selected = selector_scores[selected_index]
    selector_second = selector_scores[second_index]
    heldout_identity = heldout_scores[identity_index]
    heldout_selected = heldout_scores[selected_index]

    return CrossFitResult(
        direction=direction,
        selected_angle_degrees=rows[selected_index].angle_degrees,
        second_angle_degrees=rows[second_index].angle_degrees,
        selector_identity_iou=selector_identity.iou,
        selector_selected_iou=selector_selected.iou,
        selector_second_iou=selector_second.iou,
        selector_margin=selector_selected.iou - selector_second.iou,
        heldout_identity_iou=heldout_identity.iou,
        heldout_selected_iou=heldout_selected.iou,
        delta_iou=heldout_selected.iou - heldout_identity.iou,
        selector_predicted_support_empty=selector_identity.predicted_support == 0,
        selector_target_support_empty=selector_identity.target_support == 0,
        selector_union_empty=selector_identity.union == 0,
        heldout_predicted_support_empty=heldout_identity.predicted_support == 0,
        heldout_target_support_empty=heldout_identity.target_support == 0,
        heldout_union_empty=heldout_identity.union == 0,
    )


def evaluate_episode(
    episode: EpisodeCase,
    assignment: PivotAssignment,
    angles: tuple[float, ...],
) -> tuple[EpisodePivotResult, ...]:
    """Score one episode under every prespecified registration-pivot control."""
    if not isinstance(episode, EpisodeCase):
        raise ValueError("episode must be an EpisodeCase")
    if not isinstance(assignment, PivotAssignment):
        raise ValueError("assignment must be a PivotAssignment")
    if (
        assignment.scene_id != episode.scene_id
        or assignment.example_id != episode.example_id
    ):
        raise ValueError("assignment identity does not match episode")
    if assignment.true_pivot != episode.true_start_pivot:
        raise ValueError("assignment true_pivot does not match episode")
    if not isinstance(angles, tuple):
        raise ValueError("angles must be a tuple")

    results = []
    for pivot_mode in PivotMode:
        pivot = pivot_for_mode(episode, assignment, pivot_mode)
        angle_scores = score_angle_families(
            episode.predicted_grid,
            episode.target_grid,
            pivot,
            angles,
        )
        results.append(
            EpisodePivotResult(
                split=episode.split,
                scene_id=episode.scene_id,
                example_id=episode.example_id,
                schema_valid=episode.schema_valid,
                pivot_mode=pivot_mode,
                pivot=pivot,
                donor_example_id=(
                    assignment.donor_example_id
                    if pivot_mode is PivotMode.SHUFFLED_START
                    else None
                ),
                angle_scores=angle_scores,
                object_to_region=crossfit_scores(
                    angle_scores, Direction.OBJECT_TO_REGION
                ),
                region_to_object=crossfit_scores(
                    angle_scores, Direction.REGION_TO_OBJECT
                ),
            )
        )
    return tuple(results)


_PIVOT_MODES = tuple(PivotMode)
_ENDPOINT_FIELDS = (
    "true_start_object_to_region",
    "true_start_region_to_object",
    "true_start_symmetric",
    "map_center_object_to_region",
    "map_center_region_to_object",
    "map_center_symmetric",
    "shuffled_start_object_to_region",
    "shuffled_start_region_to_object",
    "shuffled_start_symmetric",
    "true_start_minus_map_center",
    "true_start_minus_shuffled",
)


def _endpoint_name(pivot_mode: PivotMode, suffix: str) -> str:
    return f"{pivot_mode.value}_{suffix}"


def _complete_pivot_results(
    results: Sequence[EpisodePivotResult],
) -> Dict[Tuple[str, str], Dict[PivotMode, EpisodePivotResult]]:
    rows = tuple(results)
    if not rows:
        raise ValueError("results must contain at least one EpisodePivotResult")

    by_identity: Dict[Tuple[str, str], Dict[PivotMode, EpisodePivotResult]] = {}
    for row in rows:
        if not isinstance(row, EpisodePivotResult):
            raise ValueError("results must contain only EpisodePivotResult rows")
        identity = (row.scene_id, row.example_id)
        pivot_rows = by_identity.setdefault(identity, {})
        if row.pivot_mode in pivot_rows:
            raise ValueError("results must not duplicate an episode pivot identity")
        pivot_rows[row.pivot_mode] = row

    required_modes = set(_PIVOT_MODES)
    for pivot_rows in by_identity.values():
        if set(pivot_rows) != required_modes:
            raise ValueError("results must contain a complete pivot triple per episode")
    return by_identity


def _prepared_endpoint_values(
    results: Sequence[EpisodePivotResult],
) -> Tuple[Tuple[str, ...], Dict[str, Dict[str, Tuple[float, ...]]]]:
    by_identity = _complete_pivot_results(results)
    scene_ids = tuple(
        sorted({scene_id for scene_id, _ in by_identity}, key=lambda value: value.encode("utf-8"))
    )
    if not scene_ids:
        raise ValueError("results must contain at least one scene")
    mutable_values: Dict[str, Dict[str, list[float]]] = {
        endpoint: {scene_id: [] for scene_id in scene_ids}
        for endpoint in _ENDPOINT_FIELDS
    }
    for scene_id, example_id in sorted(
        by_identity,
        key=lambda identity: (identity[0].encode("utf-8"), identity[1].encode("utf-8")),
    ):
        pivot_rows = by_identity[(scene_id, example_id)]
        for pivot_mode in _PIVOT_MODES:
            row = pivot_rows[pivot_mode]
            object_delta = _require_finite_real(
                row.object_to_region.delta_iou, "object_to_region delta_iou"
            )
            region_delta = _require_finite_real(
                row.region_to_object.delta_iou, "region_to_object delta_iou"
            )
            mutable_values[_endpoint_name(pivot_mode, "object_to_region")][
                scene_id
            ].append(object_delta)
            mutable_values[_endpoint_name(pivot_mode, "region_to_object")][
                scene_id
            ].append(region_delta)
            mutable_values[_endpoint_name(pivot_mode, "symmetric")][scene_id].append(
                0.5 * (object_delta + region_delta)
            )
        true_start = pivot_rows[PivotMode.TRUE_START].symmetric_delta
        map_center = pivot_rows[PivotMode.MAP_CENTER].symmetric_delta
        shuffled_start = pivot_rows[PivotMode.SHUFFLED_START].symmetric_delta
        mutable_values["true_start_minus_map_center"][scene_id].append(
            _require_finite_real(
                true_start - map_center, "true_start minus map_center symmetric delta"
            )
        )
        mutable_values["true_start_minus_shuffled"][scene_id].append(
            _require_finite_real(
                true_start - shuffled_start,
                "true_start minus shuffled_start symmetric delta",
            )
        )

    endpoint_values: Dict[str, Dict[str, Tuple[float, ...]]] = {}
    for endpoint, scene_values in mutable_values.items():
        endpoint_values[endpoint] = {
            scene_id: tuple(values) for scene_id, values in scene_values.items()
        }
    return scene_ids, endpoint_values


def scene_bootstrap_multiplicities(
    results: Sequence[EpisodePivotResult],
) -> Tuple[Tuple[str, ...], NDArray[np.int64]]:
    """Return the fixed shared scene-bootstrap multiplicity matrix."""
    scene_ids, _ = _prepared_endpoint_values(results)
    generator = np.random.default_rng(42)
    draws = generator.integers(0, len(scene_ids), size=(10_000, len(scene_ids)))
    multiplicities = np.zeros((10_000, len(scene_ids)), dtype=np.int64)
    for scene_index in range(len(scene_ids)):
        multiplicities[:, scene_index] = np.count_nonzero(
            draws == scene_index, axis=1
        )
    return scene_ids, multiplicities


def _bootstrap_episode_macro(
    values: Mapping[str, Sequence[float]],
    scene_ids: Sequence[str],
    multiplicities: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Bootstrap an episode-macro endpoint from fixed scene multiplicities."""
    ordered_scenes = tuple(scene_ids)
    if not ordered_scenes or len(set(ordered_scenes)) != len(ordered_scenes):
        raise ValueError("scene_ids must be a nonempty unique sequence")
    if set(values) != set(ordered_scenes):
        raise ValueError("values must contain exactly the declared scene keys")
    if (
        not isinstance(multiplicities, np.ndarray)
        or multiplicities.ndim != 2
        or multiplicities.shape[1] != len(ordered_scenes)
        or not np.issubdtype(multiplicities.dtype, np.integer)
    ):
        raise ValueError("multiplicities must be a two-dimensional integer matrix")
    if multiplicities.shape[0] == 0 or np.any(multiplicities < 0):
        raise ValueError("multiplicities must be nonnegative with at least one replicate")

    scene_sums = []
    scene_counts = []
    for scene_id in ordered_scenes:
        scene_values = tuple(values[scene_id])
        if not scene_values:
            raise ValueError("every bootstrap scene must contain at least one episode")
        checked_values = tuple(
            _require_finite_real(value, f"values for scene {scene_id}")
            for value in scene_values
        )
        scene_sums.append(sum(checked_values))
        scene_counts.append(len(checked_values))
    sums = np.asarray(scene_sums, dtype=np.float64)
    counts = np.asarray(scene_counts, dtype=np.int64)
    denominators = multiplicities @ counts
    if np.any(denominators == 0):
        raise ValueError("bootstrap multiplicities must give every replicate an episode")
    replicates = (multiplicities @ sums) / denominators
    if not np.isfinite(replicates).all():
        raise ValueError("bootstrap replicates must be finite")
    return replicates


def _mean(values: Sequence[float], name: str) -> float:
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    checked_values = tuple(_require_finite_real(value, name) for value in values)
    return float(sum(checked_values) / len(checked_values))


def _leave_one_scene_out_range(
    values: Mapping[str, Sequence[float]], scene_ids: Sequence[str]
) -> Tuple[float, float]:
    if len(scene_ids) < 2:
        raise ValueError("leave-one-scene-out requires at least two scenes")
    means = []
    for omitted_scene in scene_ids:
        remaining = tuple(
            value
            for scene_id in scene_ids
            if scene_id != omitted_scene
            for value in values[scene_id]
        )
        means.append(_mean(remaining, "leave-one-scene-out values"))
    return (float(min(means)), float(max(means)))


def _inference_ranges(
    values: Mapping[str, Tuple[float, float]],
) -> InferenceRanges:
    if set(values) != set(_ENDPOINT_FIELDS):
        raise ValueError("inference ranges must contain every endpoint")
    return InferenceRanges(
        true_start_object_to_region=values["true_start_object_to_region"],
        true_start_region_to_object=values["true_start_region_to_object"],
        true_start_symmetric=values["true_start_symmetric"],
        map_center_object_to_region=values["map_center_object_to_region"],
        map_center_region_to_object=values["map_center_region_to_object"],
        map_center_symmetric=values["map_center_symmetric"],
        shuffled_start_object_to_region=values["shuffled_start_object_to_region"],
        shuffled_start_region_to_object=values["shuffled_start_region_to_object"],
        shuffled_start_symmetric=values["shuffled_start_symmetric"],
        true_start_minus_map_center=values["true_start_minus_map_center"],
        true_start_minus_shuffled=values["true_start_minus_shuffled"],
    )


def _inference_estimates(
    values: Mapping[str, EndpointEstimate],
) -> InferenceEstimates:
    if set(values) != set(_ENDPOINT_FIELDS):
        raise ValueError("inference estimates must contain every endpoint")
    return InferenceEstimates(
        true_start_object_to_region=values["true_start_object_to_region"],
        true_start_region_to_object=values["true_start_region_to_object"],
        true_start_symmetric=values["true_start_symmetric"],
        map_center_object_to_region=values["map_center_object_to_region"],
        map_center_region_to_object=values["map_center_region_to_object"],
        map_center_symmetric=values["map_center_symmetric"],
        shuffled_start_object_to_region=values["shuffled_start_object_to_region"],
        shuffled_start_region_to_object=values["shuffled_start_region_to_object"],
        shuffled_start_symmetric=values["shuffled_start_symmetric"],
        true_start_minus_map_center=values["true_start_minus_map_center"],
        true_start_minus_shuffled=values["true_start_minus_shuffled"],
    )


def leave_one_scene_out_ranges(
    results: Sequence[EpisodePivotResult],
) -> InferenceRanges:
    """Return episode-macro minimum/maximum values after omitting each scene."""
    scene_ids, endpoint_values = _prepared_endpoint_values(results)
    ranges = {
        endpoint: _leave_one_scene_out_range(values, scene_ids)
        for endpoint, values in endpoint_values.items()
    }
    return _inference_ranges(ranges)


def bootstrap_results(
    results: Sequence[EpisodePivotResult],
) -> InferenceEstimates:
    """Estimate all endpoints from one shared paired scene-bootstrap matrix."""
    scene_ids, endpoint_values = _prepared_endpoint_values(results)
    bootstrap_scene_ids, multiplicities = scene_bootstrap_multiplicities(results)
    if bootstrap_scene_ids != scene_ids:
        raise ValueError("bootstrap scene order does not match endpoint scene order")
    ranges = {
        endpoint: _leave_one_scene_out_range(values, scene_ids)
        for endpoint, values in endpoint_values.items()
    }
    estimates: Dict[str, EndpointEstimate] = {}
    for endpoint in _ENDPOINT_FIELDS:
        values = endpoint_values[endpoint]
        all_values = tuple(
            value for scene_id in scene_ids for value in values[scene_id]
        )
        replicates = _bootstrap_episode_macro(values, scene_ids, multiplicities)
        ci_lower, ci_upper = np.percentile(replicates, (2.5, 97.5))
        leave_one_scene_out_min, leave_one_scene_out_max = ranges[endpoint]
        estimates[endpoint] = EndpointEstimate(
            mean=_mean(all_values, endpoint),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            leave_one_scene_out_min=leave_one_scene_out_min,
            leave_one_scene_out_max=leave_one_scene_out_max,
        )
    return _inference_estimates(estimates)


def _direction_summary(rows: Sequence[CrossFitResult]) -> Tuple[
    float,
    Tuple[Tuple[float, int], ...],
    Tuple[float, float, float, float, float, float],
]:
    if not rows:
        raise ValueError("direction summary requires at least one result")
    angle_counts = tuple(
        (angle, sum(row.selected_angle_degrees == angle for row in rows))
        for angle in sorted({row.selected_angle_degrees for row in rows})
    )
    empty_flags = tuple(
        (
            row.selector_predicted_support_empty,
            row.selector_target_support_empty,
            row.selector_union_empty,
            row.heldout_predicted_support_empty,
            row.heldout_target_support_empty,
            row.heldout_union_empty,
        )
        for row in rows
    )
    empty_rates = (
        _mean(tuple(float(flags[0]) for flags in empty_flags), "selector predicted"),
        _mean(tuple(float(flags[1]) for flags in empty_flags), "selector target"),
        _mean(tuple(float(flags[2]) for flags in empty_flags), "selector union"),
        _mean(tuple(float(flags[3]) for flags in empty_flags), "heldout predicted"),
        _mean(tuple(float(flags[4]) for flags in empty_flags), "heldout target"),
        _mean(tuple(float(flags[5]) for flags in empty_flags), "heldout union"),
    )
    return (
        _mean(tuple(row.delta_iou for row in rows), "directional deltas"),
        angle_counts,
        empty_rates,
    )


def summarize_results(
    results: Sequence[EpisodePivotResult],
) -> Dict[PivotMode, PivotSummary]:
    """Summarize retained rows by pivot without filtering invalid episodes."""
    by_identity = _complete_pivot_results(results)
    summary: Dict[PivotMode, PivotSummary] = {}
    for pivot_mode in _PIVOT_MODES:
        pivot_rows = tuple(
            pivot_rows[pivot_mode] for pivot_rows in by_identity.values()
        )
        object_mean, object_angles, object_empty_rates = _direction_summary(
            tuple(row.object_to_region for row in pivot_rows)
        )
        region_mean, region_angles, region_empty_rates = _direction_summary(
            tuple(row.region_to_object for row in pivot_rows)
        )
        summary[pivot_mode] = PivotSummary(
            object_to_region_mean=object_mean,
            region_to_object_mean=region_mean,
            symmetric_mean=0.5 * (object_mean + region_mean),
            object_to_region_angle_counts=object_angles,
            region_to_object_angle_counts=region_angles,
            object_to_region_empty_rates=object_empty_rates,
            region_to_object_empty_rates=region_empty_rates,
        )
    return summary


def classify_gate(
    *,
    true_start_symmetric: EndpointEstimate,
    true_start_object_to_region_mean: float,
    true_start_region_to_object_mean: float,
    start_minus_center: EndpointEstimate,
    start_minus_shuffled: EndpointEstimate,
) -> GateDecision:
    """Apply the frozen, ordered intersection-union decision gate literally."""
    if not isinstance(true_start_symmetric, EndpointEstimate):
        raise ValueError("true_start_symmetric must be an EndpointEstimate")
    if not isinstance(start_minus_center, EndpointEstimate):
        raise ValueError("start_minus_center must be an EndpointEstimate")
    if not isinstance(start_minus_shuffled, EndpointEstimate):
        raise ValueError("start_minus_shuffled must be an EndpointEstimate")
    object_delta = _require_finite_real(
        true_start_object_to_region_mean, "true_start_object_to_region_mean"
    )
    region_delta = _require_finite_real(
        true_start_region_to_object_mean, "true_start_region_to_object_mean"
    )
    symmetric_passes = (
        true_start_symmetric.mean >= 0.01
        and true_start_symmetric.ci_lower > 0.0
    )
    if not symmetric_passes:
        return GateDecision.NO_GO
    if (
        object_delta > 0.0
        and region_delta > 0.0
        and start_minus_center.ci_lower > 0.0
        and start_minus_shuffled.ci_lower > 0.0
    ):
        return GateDecision.GO
    return GateDecision.PARTIAL_EVIDENCE


def _gate_conditions(estimates: InferenceEstimates) -> _GateConditions:
    return _GateConditions(
        true_start_symmetric_mean_at_least_0_01=(
            estimates.true_start_symmetric.mean >= 0.01
        ),
        true_start_symmetric_ci_lower_above_zero=(
            estimates.true_start_symmetric.ci_lower > 0.0
        ),
        both_true_start_directional_means_above_zero=(
            estimates.true_start_object_to_region.mean > 0.0
            and estimates.true_start_region_to_object.mean > 0.0
        ),
        true_start_minus_map_center_ci_lower_above_zero=(
            estimates.true_start_minus_map_center.ci_lower > 0.0
        ),
        true_start_minus_shuffled_ci_lower_above_zero=(
            estimates.true_start_minus_shuffled.ci_lower > 0.0
        ),
    )


def _angle_score_rows(
    results: Sequence[EpisodePivotResult],
) -> tuple[Dict[str, object], ...]:
    rows: list[Dict[str, object]] = []
    for result in results:
        for angle_score in result.angle_scores:
            object_score = angle_score.object_score
            region_score = angle_score.region_score
            rows.append(
                {
                    "split": result.split,
                    "scene_id": result.scene_id,
                    "example_id": result.example_id,
                    "schema_valid": result.schema_valid,
                    "pivot_mode": result.pivot_mode.value,
                    "pivot_row": result.pivot[0],
                    "pivot_column": result.pivot[1],
                    "donor_example_id": result.donor_example_id or "",
                    "angle_degrees": angle_score.angle_degrees,
                    "bounds_row_min": angle_score.bounds.row_min,
                    "bounds_row_max": angle_score.bounds.row_max,
                    "bounds_col_min": angle_score.bounds.col_min,
                    "bounds_col_max": angle_score.bounds.col_max,
                    "object_input_support": angle_score.object_input_support,
                    "object_intersection": object_score.intersection,
                    "object_union": object_score.union,
                    "object_predicted_support": object_score.predicted_support,
                    "object_target_support": object_score.target_support,
                    "object_in_frame_support": object_score.in_frame_support,
                    "object_out_of_frame_support": object_score.out_of_frame_support,
                    "object_iou": object_score.iou,
                    "region_input_support": angle_score.region_input_support,
                    "region_intersection": region_score.intersection,
                    "region_union": region_score.union,
                    "region_predicted_support": region_score.predicted_support,
                    "region_target_support": region_score.target_support,
                    "region_in_frame_support": region_score.in_frame_support,
                    "region_out_of_frame_support": region_score.out_of_frame_support,
                    "region_iou": region_score.iou,
                }
            )
    return tuple(rows)


def _crossfit_result_rows(
    results: Sequence[EpisodePivotResult],
) -> tuple[Dict[str, object], ...]:
    rows: list[Dict[str, object]] = []
    for episode_result in results:
        for result in (
            episode_result.object_to_region,
            episode_result.region_to_object,
        ):
            rows.append(
                {
                    "split": episode_result.split,
                    "scene_id": episode_result.scene_id,
                    "example_id": episode_result.example_id,
                    "schema_valid": episode_result.schema_valid,
                    "pivot_mode": episode_result.pivot_mode.value,
                    "pivot_row": episode_result.pivot[0],
                    "pivot_column": episode_result.pivot[1],
                    "donor_example_id": episode_result.donor_example_id or "",
                    "direction": result.direction.value,
                    "selected_angle_degrees": result.selected_angle_degrees,
                    "second_angle_degrees": result.second_angle_degrees,
                    "selector_identity_iou": result.selector_identity_iou,
                    "selector_selected_iou": result.selector_selected_iou,
                    "selector_second_iou": result.selector_second_iou,
                    "selector_margin": result.selector_margin,
                    "heldout_identity_iou": result.heldout_identity_iou,
                    "heldout_selected_iou": result.heldout_selected_iou,
                    "delta_iou": result.delta_iou,
                    "selector_predicted_support_empty": (
                        result.selector_predicted_support_empty
                    ),
                    "selector_target_support_empty": (
                        result.selector_target_support_empty
                    ),
                    "selector_union_empty": result.selector_union_empty,
                    "heldout_predicted_support_empty": (
                        result.heldout_predicted_support_empty
                    ),
                    "heldout_target_support_empty": (
                        result.heldout_target_support_empty
                    ),
                    "heldout_union_empty": result.heldout_union_empty,
                }
            )
    return tuple(rows)


def _pivot_assignment_rows(
    assignments: Sequence[PivotAssignment],
) -> tuple[Dict[str, object], ...]:
    return tuple(
        {
            "scene_id": assignment.scene_id,
            "example_id": assignment.example_id,
            "donor_example_id": assignment.donor_example_id,
            "true_pivot_row": assignment.true_pivot[0],
            "true_pivot_column": assignment.true_pivot[1],
            "assigned_pivot_row": assignment.assigned_pivot[0],
            "assigned_pivot_column": assignment.assigned_pivot[1],
        }
        for assignment in assignments
    )


def _csv_bytes(
    header: tuple[str, ...],
    rows: Sequence[Mapping[str, object]],
) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    stream = StringIO()
    json.dump(payload, stream, sort_keys=True, indent=2)
    stream.write("\n")
    return stream.getvalue().encode("utf-8")


def _git_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    _require_nonempty_string(commit, "git commit")
    return commit


def _estimates_by_name(
    estimates: InferenceEstimates,
) -> Dict[str, EndpointEstimate]:
    return {
        "true_start_object_to_region": estimates.true_start_object_to_region,
        "true_start_region_to_object": estimates.true_start_region_to_object,
        "true_start_symmetric": estimates.true_start_symmetric,
        "map_center_object_to_region": estimates.map_center_object_to_region,
        "map_center_region_to_object": estimates.map_center_region_to_object,
        "map_center_symmetric": estimates.map_center_symmetric,
        "shuffled_start_object_to_region": (
            estimates.shuffled_start_object_to_region
        ),
        "shuffled_start_region_to_object": (
            estimates.shuffled_start_region_to_object
        ),
        "shuffled_start_symmetric": estimates.shuffled_start_symmetric,
        "true_start_minus_map_center": estimates.true_start_minus_map_center,
        "true_start_minus_shuffled": estimates.true_start_minus_shuffled,
    }


def _ranges_by_name(ranges: InferenceRanges) -> Dict[str, Tuple[float, float]]:
    return {
        "true_start_object_to_region": ranges.true_start_object_to_region,
        "true_start_region_to_object": ranges.true_start_region_to_object,
        "true_start_symmetric": ranges.true_start_symmetric,
        "map_center_object_to_region": ranges.map_center_object_to_region,
        "map_center_region_to_object": ranges.map_center_region_to_object,
        "map_center_symmetric": ranges.map_center_symmetric,
        "shuffled_start_object_to_region": (
            ranges.shuffled_start_object_to_region
        ),
        "shuffled_start_region_to_object": (
            ranges.shuffled_start_region_to_object
        ),
        "shuffled_start_symmetric": ranges.shuffled_start_symmetric,
        "true_start_minus_map_center": ranges.true_start_minus_map_center,
        "true_start_minus_shuffled": ranges.true_start_minus_shuffled,
    }


_EMPTY_RATE_NAMES = (
    "selector_predicted_support_empty",
    "selector_target_support_empty",
    "selector_union_empty",
    "heldout_predicted_support_empty",
    "heldout_target_support_empty",
    "heldout_union_empty",
)


def _direction_summary_payload(
    mean: float,
    angle_counts: Sequence[Tuple[float, int]],
    empty_rates: Sequence[float],
) -> Dict[str, object]:
    return {
        "mean_delta_iou": mean,
        "selected_angle_counts": [
            {"angle_degrees": angle, "episodes": count}
            for angle, count in angle_counts
        ],
        "empty_rates": {
            name: rate for name, rate in zip(_EMPTY_RATE_NAMES, empty_rates)
        },
    }


def _summary_payload(
    *,
    population: Mapping[str, object],
    summaries: Mapping[PivotMode, PivotSummary],
    ranges: InferenceRanges,
    estimates: InferenceEstimates,
    gate_conditions: _GateConditions,
    decision: GateDecision,
) -> Dict[str, object]:
    pivot_payload: Dict[str, object] = {}
    for pivot_mode in _PIVOT_MODES:
        summary = summaries[pivot_mode]
        pivot_payload[pivot_mode.value] = {
            Direction.OBJECT_TO_REGION.value: _direction_summary_payload(
                summary.object_to_region_mean,
                summary.object_to_region_angle_counts,
                summary.object_to_region_empty_rates,
            ),
            Direction.REGION_TO_OBJECT.value: _direction_summary_payload(
                summary.region_to_object_mean,
                summary.region_to_object_angle_counts,
                summary.region_to_object_empty_rates,
            ),
            "symmetric_mean_delta_iou": summary.symmetric_mean,
        }
    named_ranges = _ranges_by_name(ranges)
    range_payload = {
        name: {"minimum": bounds[0], "maximum": bounds[1]}
        for name, bounds in named_ranges.items()
    }
    return {
        "population": dict(population),
        "pivots": pivot_payload,
        "leave_one_scene_out_ranges": range_payload,
        "start_specific_contrasts": {
            "true_start_minus_map_center": {
                "mean_delta_iou": estimates.true_start_minus_map_center.mean
            },
            "true_start_minus_shuffled": {
                "mean_delta_iou": estimates.true_start_minus_shuffled.mean
            },
        },
        "gate": {
            "conditions": gate_conditions.payload(),
            "decision": decision.value,
        },
    }


def _bootstrap_payload(
    estimates: InferenceEstimates, scene_count: int
) -> Dict[str, object]:
    endpoints = {
        name: {
            "mean": estimate.mean,
            "ci_lower": estimate.ci_lower,
            "ci_upper": estimate.ci_upper,
            "leave_one_scene_out_min": estimate.leave_one_scene_out_min,
            "leave_one_scene_out_max": estimate.leave_one_scene_out_max,
        }
        for name, estimate in _estimates_by_name(estimates).items()
    }
    return {
        "contract": {
            "clusters": scene_count,
            "interval": "percentile",
            "pairing": "shared_scene_multiplicity_matrix",
            "percentiles": [2.5, 97.5],
            "repetitions": _DEFAULT_BOOTSTRAP_REPETITIONS,
            "sampling_unit": "scene",
            "seed": _DEFAULT_BOOTSTRAP_SEED,
            "weighting": "episode_macro",
        },
        "endpoints": endpoints,
    }


def _plot_bytes(estimates: InferenceEstimates) -> bytes:
    named = _estimates_by_name(estimates)
    endpoint_names = _ENDPOINT_FIELDS
    means = [named[name].mean for name in endpoint_names]
    lower_errors = [
        named[name].mean - named[name].ci_lower for name in endpoint_names
    ]
    upper_errors = [
        named[name].ci_upper - named[name].mean for name in endpoint_names
    ]
    labels = [name.replace("_", "\n") for name in endpoint_names]
    figure, axis = plt.subplots(figsize=(14, 6))
    positions = np.arange(len(endpoint_names))
    axis.errorbar(
        positions,
        means,
        yerr=np.asarray((lower_errors, upper_errors)),
        fmt="o",
        capsize=4,
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylabel("Held-out category-aware IoU delta")
    axis.set_title("ground-truth cross-fit diagnostic (not deployable performance)")
    figure.tight_layout()
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=160, metadata={"Date": None})
    plt.close(figure)
    return stream.getvalue()


@dataclass(frozen=True)
class _ArtifactBundle:
    manifest_json: bytes
    angle_scores_csv: bytes
    crossfit_results_csv: bytes
    pivot_assignments_csv: bytes
    summary_json: bytes
    bootstrap_json: bytes
    control_intervals_png: bytes

    def files(self) -> Dict[str, bytes]:
        return {
            "manifest.json": self.manifest_json,
            "angle_scores.csv": self.angle_scores_csv,
            "crossfit_results.csv": self.crossfit_results_csv,
            "pivot_assignments.csv": self.pivot_assignments_csv,
            "summary.json": self.summary_json,
            "bootstrap.json": self.bootstrap_json,
            "crossfit_control_intervals.png": self.control_intervals_png,
        }


_ArtifactFiles = Tuple[Tuple[str, bytes], ...]


def _capture_artifact_bundle(
    artifacts: _ArtifactBundle,
) -> Tuple[_ArtifactBundle, _ArtifactFiles]:
    captured = _ArtifactBundle(
        manifest_json=artifacts.manifest_json,
        angle_scores_csv=artifacts.angle_scores_csv,
        crossfit_results_csv=artifacts.crossfit_results_csv,
        pivot_assignments_csv=artifacts.pivot_assignments_csv,
        summary_json=artifacts.summary_json,
        bootstrap_json=artifacts.bootstrap_json,
        control_intervals_png=artifacts.control_intervals_png,
    )
    return captured, (
        ("manifest.json", captured.manifest_json),
        ("angle_scores.csv", captured.angle_scores_csv),
        ("crossfit_results.csv", captured.crossfit_results_csv),
        ("pivot_assignments.csv", captured.pivot_assignments_csv),
        ("summary.json", captured.summary_json),
        ("bootstrap.json", captured.bootstrap_json),
        (
            "crossfit_control_intervals.png",
            captured.control_intervals_png,
        ),
    )


def _write_artifacts(output_dir: Path, artifacts: _ArtifactBundle) -> None:
    """Validate, capture, and persist one seven-file transaction."""
    validated_files = _validate_artifact_bundle(artifacts)
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise ValueError(
            f"output directory must be empty or absent, got nonempty: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            dir=str(output_dir.parent),
            prefix=f".{output_dir.name}.tmp-",
        )
    )
    try:
        for filename, data in validated_files:
            (temporary_dir / filename).write_bytes(data)
        temporary_dir.replace(output_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def _validate_analysis(
    *,
    cases: Sequence[EpisodeCase],
    assignments: Sequence[PivotAssignment],
    results: Sequence[EpisodePivotResult],
    summaries: Mapping[PivotMode, PivotSummary],
    ranges: InferenceRanges,
    estimates: InferenceEstimates,
    gate_conditions: _GateConditions,
    decision: GateDecision,
    angle_rows: Sequence[Mapping[str, object]],
    crossfit_rows: Sequence[Mapping[str, object]],
    pivot_assignment_rows: Sequence[Mapping[str, object]],
    summary_payload: Mapping[str, object],
    bootstrap_payload: Mapping[str, object],
    expected_population: int,
    expected_scenes: int,
    expected_valid: int,
    expected_invalid: int,
    angles: tuple[float, ...],
) -> None:
    """Validate every scientific and relational invariant before serialization."""
    for value, name in (
        (expected_population, "expected_population"),
        (expected_scenes, "expected_scenes"),
        (expected_valid, "expected_valid"),
        (expected_invalid, "expected_invalid"),
    ):
        _require_nonnegative_int(value, name)
    if expected_valid + expected_invalid != expected_population:
        raise ValueError("valid and invalid counts must partition the population")
    if expected_scenes < 2:
        raise ValueError("analysis requires at least two scenes")
    if tuple(angles) != _DEFAULT_ANGLES:
        raise ValueError(f"angle membership must be exactly {_DEFAULT_ANGLES}")

    episode_rows = tuple(cases)
    if len(episode_rows) != expected_population:
        raise ValueError("episode row count does not match the population contract")
    identities: Set[Tuple[str, str]] = set()
    by_identity: Dict[Tuple[str, str], EpisodeCase] = {}
    for case in episode_rows:
        if not isinstance(case, EpisodeCase):
            raise ValueError("cases must contain only EpisodeCase rows")
        case.__post_init__()
        identity = (case.scene_id, case.example_id)
        if identity in identities:
            raise ValueError("cases must have unique scene/example identities")
        identities.add(identity)
        by_identity[identity] = case
        if case.split != "val_unseen":
            raise ValueError("cases must contain only val_unseen rows")
        if not case.schema_valid and np.any(case.predicted_grid):
            raise ValueError("invalid predictions must use an empty grid")
    if len({case.scene_id for case in episode_rows}) != expected_scenes:
        raise ValueError("scene count does not match the population contract")
    if sum(case.schema_valid for case in episode_rows) != expected_valid:
        raise ValueError("valid count does not match the population contract")
    if sum(not case.schema_valid for case in episode_rows) != expected_invalid:
        raise ValueError("invalid count does not match the population contract")

    assignment_rows = tuple(assignments)
    if len(assignment_rows) != expected_population:
        raise ValueError("pivot assignment row count must match the population")
    assignment_by_identity: Dict[Tuple[str, str], PivotAssignment] = {}
    for assignment in assignment_rows:
        if not isinstance(assignment, PivotAssignment):
            raise ValueError("assignments must contain only PivotAssignment rows")
        assignment.__post_init__()
        identity = (assignment.scene_id, assignment.example_id)
        if identity in assignment_by_identity:
            raise ValueError("pivot assignments must have unique receiver identities")
        case = by_identity.get(identity)
        if case is None:
            raise ValueError("pivot assignment identity is absent from cases")
        if assignment.true_pivot != case.true_start_pivot:
            raise ValueError("pivot assignment true pivot does not match its case")
        donor = by_identity.get((assignment.scene_id, assignment.donor_example_id))
        if donor is None:
            raise ValueError("pivot assignment donor must be in the receiver scene")
        if assignment.assigned_pivot != donor.true_start_pivot:
            raise ValueError("pivot assignment does not use the donor true pivot")
        assignment_by_identity[identity] = assignment
    if set(assignment_by_identity) != identities:
        raise ValueError("pivot assignments must cover every case identity")
    for scene_id in {case.scene_id for case in episode_rows}:
        scene_cases = tuple(case for case in episode_rows if case.scene_id == scene_id)
        scene_assignments = tuple(
            assignment
            for assignment in assignment_rows
            if assignment.scene_id == scene_id
        )
        if (
            Counter(row.donor_example_id for row in scene_assignments)
            != Counter(row.example_id for row in scene_cases)
            or Counter(row.assigned_pivot for row in scene_assignments)
            != Counter(row.true_start_pivot for row in scene_cases)
        ):
            raise ValueError("pivot assignments must preserve each scene donor/pivot multiset")

    result_rows = tuple(results)
    if len(result_rows) != expected_population * len(PivotMode):
        raise ValueError("pivot result row count does not match population times pivots")
    complete_results = _complete_pivot_results(result_rows)
    if set(complete_results) != identities:
        raise ValueError("pivot results must cover every case identity")
    expected_result_order = tuple(
        (case.scene_id, case.example_id, pivot_mode)
        for case in sorted(
            episode_rows,
            key=lambda row: (
                row.scene_id.encode("utf-8"),
                row.example_id.encode("utf-8"),
            ),
        )
        for pivot_mode in _PIVOT_MODES
    )
    actual_result_order = tuple(
        (row.scene_id, row.example_id, row.pivot_mode) for row in result_rows
    )
    if actual_result_order != expected_result_order:
        raise ValueError("pivot results must use stable identity/pivot row ordering")
    if any(
        tuple(score.angle_degrees for score in row.angle_scores) != angles
        for row in result_rows
    ):
        raise ValueError("angle membership or ordering is inconsistent")
    reevaluated_results = tuple(
        reevaluated
        for case in sorted(
            episode_rows,
            key=lambda row: (
                row.scene_id.encode("utf-8"),
                row.example_id.encode("utf-8"),
            ),
        )
        for reevaluated in evaluate_episode(
            case,
            assignment_by_identity[(case.scene_id, case.example_id)],
            angles,
        )
    )
    if result_rows != reevaluated_results:
        raise ValueError(
            "pivot bounds and raster scores must match evaluation from typed cases"
        )

    for row in result_rows:
        row.__post_init__()
        identity = (row.scene_id, row.example_id)
        case = by_identity[identity]
        assignment = assignment_by_identity[identity]
        if row.split != case.split or row.schema_valid != case.schema_valid:
            raise ValueError("pivot result identity metadata does not match its case")
        if row.pivot != pivot_for_mode(case, assignment, row.pivot_mode):
            raise ValueError("pivot result uses the wrong pivot coordinates")
        expected_donor = (
            assignment.donor_example_id
            if row.pivot_mode is PivotMode.SHUFFLED_START
            else None
        )
        if row.donor_example_id != expected_donor:
            raise ValueError("pivot result donor metadata is inconsistent")
        row_angles = tuple(score.angle_degrees for score in row.angle_scores)
        if row_angles != angles:
            raise ValueError("angle membership or ordering is inconsistent")
        object_input_support = int(
            np.count_nonzero(case.predicted_grid[:_OBJECT_CHANNEL_COUNT])
        )
        region_input_support = int(
            np.count_nonzero(case.predicted_grid[_OBJECT_CHANNEL_COUNT:])
        )
        object_target_support = int(
            np.count_nonzero(case.target_grid[:_OBJECT_CHANNEL_COUNT])
        )
        region_target_support = int(
            np.count_nonzero(case.target_grid[_OBJECT_CHANNEL_COUNT:])
        )
        for score in row.angle_scores:
            score.__post_init__()
            score.object_score.__post_init__()
            score.region_score.__post_init__()
            if (
                score.object_input_support != object_input_support
                or score.region_input_support != region_input_support
                or score.object_score.predicted_support != object_input_support
                or score.region_score.predicted_support != region_input_support
                or score.object_score.target_support != object_target_support
                or score.region_score.target_support != region_target_support
            ):
                raise ValueError("family support partitions are inconsistent")
            _require_iou(score.object_score.iou, "object IoU")
            _require_iou(score.region_score.iou, "region IoU")
        for crossfit_result in (row.object_to_region, row.region_to_object):
            crossfit_result.__post_init__()
            expected_crossfit = crossfit_scores(
                row.angle_scores, crossfit_result.direction
            )
            if crossfit_result != expected_crossfit:
                raise ValueError("cross-fit selection, margin, or delta is inconsistent")

    required_pivots = set(_PIVOT_MODES)
    if set(summaries) != required_pivots:
        raise ValueError("summaries must contain every pivot")
    for summary in summaries.values():
        if not isinstance(summary, PivotSummary):
            raise ValueError("summaries must contain only PivotSummary values")
    if dict(summaries) != summarize_results(result_rows):
        raise ValueError("summary inputs do not match pivot results")
    if not isinstance(ranges, InferenceRanges):
        raise ValueError("ranges must be an InferenceRanges")
    ranges.__post_init__()
    if ranges != leave_one_scene_out_ranges(result_rows):
        raise ValueError("leave-one-scene-out ranges do not match pivot results")
    if not isinstance(estimates, InferenceEstimates):
        raise ValueError("estimates must be an InferenceEstimates")
    estimates.__post_init__()
    for estimate in estimates.all():
        estimate.__post_init__()
    if estimates != bootstrap_results(result_rows):
        raise ValueError("bootstrap intervals or contrast arithmetic are inconsistent")
    if not isinstance(decision, GateDecision):
        raise ValueError("decision must be a GateDecision")
    if not isinstance(gate_conditions, _GateConditions):
        raise ValueError("gate_conditions must be a _GateConditions")
    gate_conditions.__post_init__()
    if gate_conditions != _gate_conditions(estimates):
        raise ValueError("gate condition inputs are inconsistent")
    true_start = summaries[PivotMode.TRUE_START]
    expected_decision = classify_gate(
        true_start_symmetric=estimates.true_start_symmetric,
        true_start_object_to_region_mean=true_start.object_to_region_mean,
        true_start_region_to_object_mean=true_start.region_to_object_mean,
        start_minus_center=estimates.true_start_minus_map_center,
        start_minus_shuffled=estimates.true_start_minus_shuffled,
    )
    if decision is not expected_decision:
        raise ValueError("gate inputs and final decision are inconsistent")

    if tuple(angle_rows) != _angle_score_rows(result_rows):
        raise ValueError("angle score output rows are inconsistent")
    if tuple(crossfit_rows) != _crossfit_result_rows(result_rows):
        raise ValueError("cross-fit output rows are inconsistent")
    if tuple(pivot_assignment_rows) != _pivot_assignment_rows(assignment_rows):
        raise ValueError("pivot assignment output rows are inconsistent")
    expected_population_payload: Dict[str, object] = {
        "episodes": expected_population,
        "scenes": expected_scenes,
        "schema_valid": expected_valid,
        "schema_invalid": expected_invalid,
    }
    if dict(summary_payload) != _summary_payload(
        population=expected_population_payload,
        summaries=summaries,
        ranges=ranges,
        estimates=estimates,
        gate_conditions=gate_conditions,
        decision=decision,
    ):
        raise ValueError("summary output payload is inconsistent")
    if dict(bootstrap_payload) != _bootstrap_payload(estimates, expected_scenes):
        raise ValueError("bootstrap output payload is inconsistent")


def _require_json_object(value: object, name: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    result: Dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        result[key] = item
    return result


def _manifest_population_count(
    population: Mapping[str, object], field: str
) -> int:
    value = population.get(field)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"manifest population {field} must be a non-negative integer")
    return int(value)


def _decode_json_object(data: bytes, name: str) -> Dict[str, object]:
    text = data.decode("utf-8")
    if not text.endswith("\n"):
        raise ValueError(f"{name} must end with a newline")
    payload: object = json.loads(text)
    result = _require_json_object(payload, name)
    expected_text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if text != expected_text:
        raise ValueError(f"{name} must use sorted deterministic JSON")
    return result


def _decode_csv_rows(
    data: bytes,
    name: str,
    expected_header: tuple[str, ...],
    expected_rows: int,
) -> tuple[Dict[str, str], ...]:
    stream = StringIO(data.decode("utf-8"), newline="")
    reader = csv.DictReader(stream)
    if tuple(reader.fieldnames or ()) != expected_header:
        raise ValueError(f"{name} has a mismatched header invariant")
    rows: list[Dict[str, str]] = []
    for row in reader:
        checked: Dict[str, str] = {}
        for key, value in row.items():
            if key is None or value is None:
                raise ValueError(f"{name} contains a malformed CSV row")
            checked[key] = value
        rows.append(checked)
    if len(rows) != expected_rows:
        raise ValueError(
            f"{name} has a mismatched row invariant: "
            f"expected {expected_rows}, got {len(rows)}"
        )
    return tuple(rows)


def _csv_int(row: Mapping[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error


def _csv_float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{field} must be a finite real number") from error
    return _require_finite_real(value, field)


def _csv_bool(row: Mapping[str, str], field: str) -> bool:
    try:
        value = row[field]
    except KeyError as error:
        raise ValueError(f"{field} must be a boolean") from error
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{field} must be a boolean")


def _raster_from_csv(row: Mapping[str, str], prefix: str) -> RasterScore:
    serialized_iou = _csv_float(row, f"{prefix}_iou")
    score = RasterScore(
        intersection=_csv_int(row, f"{prefix}_intersection"),
        union=_csv_int(row, f"{prefix}_union"),
        predicted_support=_csv_int(row, f"{prefix}_predicted_support"),
        target_support=_csv_int(row, f"{prefix}_target_support"),
        in_frame_support=_csv_int(row, f"{prefix}_in_frame_support"),
        out_of_frame_support=_csv_int(row, f"{prefix}_out_of_frame_support"),
    )
    if serialized_iou != score.iou:
        raise ValueError(
            f"{prefix}_iou does not match the reconstructed RasterScore.iou"
        )
    return score


def _crossfit_from_csv(row: Mapping[str, str]) -> CrossFitResult:
    try:
        direction = Direction(row["direction"])
    except (KeyError, ValueError) as error:
        raise ValueError("cross-fit direction membership is invalid") from error
    return CrossFitResult(
        direction=direction,
        selected_angle_degrees=_csv_float(row, "selected_angle_degrees"),
        second_angle_degrees=_csv_float(row, "second_angle_degrees"),
        selector_identity_iou=_csv_float(row, "selector_identity_iou"),
        selector_selected_iou=_csv_float(row, "selector_selected_iou"),
        selector_second_iou=_csv_float(row, "selector_second_iou"),
        selector_margin=_csv_float(row, "selector_margin"),
        heldout_identity_iou=_csv_float(row, "heldout_identity_iou"),
        heldout_selected_iou=_csv_float(row, "heldout_selected_iou"),
        delta_iou=_csv_float(row, "delta_iou"),
        selector_predicted_support_empty=_csv_bool(
            row, "selector_predicted_support_empty"
        ),
        selector_target_support_empty=_csv_bool(
            row, "selector_target_support_empty"
        ),
        selector_union_empty=_csv_bool(row, "selector_union_empty"),
        heldout_predicted_support_empty=_csv_bool(
            row, "heldout_predicted_support_empty"
        ),
        heldout_target_support_empty=_csv_bool(
            row, "heldout_target_support_empty"
        ),
        heldout_union_empty=_csv_bool(row, "heldout_union_empty"),
    )


def _artifact_models(
    angle_rows: Sequence[Mapping[str, str]],
    crossfit_rows: Sequence[Mapping[str, str]],
    assignment_rows: Sequence[Mapping[str, str]],
) -> tuple[tuple[PivotAssignment, ...], tuple[EpisodePivotResult, ...]]:
    assignments = tuple(
        PivotAssignment(
            scene_id=row["scene_id"],
            example_id=row["example_id"],
            donor_example_id=row["donor_example_id"],
            true_pivot=(
                _csv_float(row, "true_pivot_row"),
                _csv_float(row, "true_pivot_column"),
            ),
            assigned_pivot=(
                _csv_float(row, "assigned_pivot_row"),
                _csv_float(row, "assigned_pivot_column"),
            ),
        )
        for row in assignment_rows
    )
    angles_by_key: Dict[
        Tuple[str, str, PivotMode], tuple[FamilyAngleScore, ...]
    ] = {}
    metadata_by_key: Dict[Tuple[str, str, PivotMode], Mapping[str, str]] = {}
    for offset in range(0, len(angle_rows), len(_DEFAULT_ANGLES)):
        group = tuple(angle_rows[offset : offset + len(_DEFAULT_ANGLES)])
        first = group[0]
        try:
            pivot_mode = PivotMode(first["pivot_mode"])
        except (KeyError, ValueError) as error:
            raise ValueError("angle pivot membership is invalid") from error
        key = (first["scene_id"], first["example_id"], pivot_mode)
        if key in angles_by_key:
            raise ValueError("angle score identities must be unique")
        scores = tuple(
            FamilyAngleScore(
                angle_degrees=_csv_float(row, "angle_degrees"),
                bounds=SpatialBounds(
                    row_min=_csv_int(row, "bounds_row_min"),
                    row_max=_csv_int(row, "bounds_row_max"),
                    col_min=_csv_int(row, "bounds_col_min"),
                    col_max=_csv_int(row, "bounds_col_max"),
                ),
                object_input_support=_csv_int(row, "object_input_support"),
                region_input_support=_csv_int(row, "region_input_support"),
                object_score=_raster_from_csv(row, "object"),
                region_score=_raster_from_csv(row, "region"),
            )
            for row in group
        )
        if tuple(score.angle_degrees for score in scores) != _DEFAULT_ANGLES:
            raise ValueError("angle membership or ordering is inconsistent")
        if any(
            score.object_input_support != score.object_score.predicted_support
            or score.region_input_support != score.region_score.predicted_support
            for score in scores
        ):
            raise ValueError("serialized family support partitions are inconsistent")
        if any(
            tuple(row[field] for field in _ROW_METADATA_FIELDS)
            != tuple(first[field] for field in _ROW_METADATA_FIELDS)
            for row in group
        ):
            raise ValueError("angle score group metadata is inconsistent")
        angles_by_key[key] = scores
        metadata_by_key[key] = first

    crossfit_by_key: Dict[
        Tuple[str, str, PivotMode], tuple[CrossFitResult, CrossFitResult]
    ] = {}
    crossfit_metadata_by_key: Dict[
        Tuple[str, str, PivotMode], Mapping[str, str]
    ] = {}
    for offset in range(0, len(crossfit_rows), len(Direction)):
        group = tuple(crossfit_rows[offset : offset + len(Direction)])
        first = group[0]
        try:
            pivot_mode = PivotMode(first["pivot_mode"])
        except (KeyError, ValueError) as error:
            raise ValueError("cross-fit pivot membership is invalid") from error
        key = (first["scene_id"], first["example_id"], pivot_mode)
        if any(
            tuple(row[field] for field in _ROW_METADATA_FIELDS)
            != tuple(first[field] for field in _ROW_METADATA_FIELDS)
            for row in group
        ):
            raise ValueError("cross-fit result group metadata is inconsistent")
        directional = tuple(_crossfit_from_csv(row) for row in group)
        if tuple(result.direction for result in directional) != tuple(Direction):
            raise ValueError("cross-fit direction membership or ordering is invalid")
        crossfit_by_key[key] = (directional[0], directional[1])
        crossfit_metadata_by_key[key] = first

    if set(angles_by_key) != set(crossfit_by_key):
        raise ValueError("angle and cross-fit pivot identities must match")
    results = []
    for key, scores in angles_by_key.items():
        metadata = metadata_by_key[key]
        crossfit_metadata = crossfit_metadata_by_key[key]
        if any(
            metadata[field] != crossfit_metadata[field]
            for field in _ROW_METADATA_FIELDS
        ):
            raise ValueError("angle and cross-fit metadata are inconsistent")
        directional = crossfit_by_key[key]
        donor = metadata["donor_example_id"] or None
        result = EpisodePivotResult(
            split=metadata["split"],
            scene_id=key[0],
            example_id=key[1],
            schema_valid=_csv_bool(metadata, "schema_valid"),
            pivot_mode=key[2],
            pivot=(
                _csv_float(metadata, "pivot_row"),
                _csv_float(metadata, "pivot_column"),
            ),
            donor_example_id=donor,
            angle_scores=scores,
            object_to_region=directional[0],
            region_to_object=directional[1],
        )
        for directional_result in directional:
            if directional_result != crossfit_scores(
                scores, directional_result.direction
            ):
                raise ValueError("serialized cross-fit arithmetic is inconsistent")
        results.append(result)
    return assignments, tuple(results)


def _validate_artifact_snapshot(
    artifacts: _ArtifactBundle,
    manifest: Mapping[str, object],
    *,
    expected_population: int,
    expected_scenes: int,
    expected_valid: int,
    expected_invalid: int,
) -> None:
    for value, name in (
        (expected_population, "expected_population"),
        (expected_scenes, "expected_scenes"),
        (expected_valid, "expected_valid"),
        (expected_invalid, "expected_invalid"),
    ):
        _require_nonnegative_int(value, name)
    if expected_valid + expected_invalid != expected_population:
        raise ValueError("artifact status counts must partition the population")
    population: Dict[str, object] = {
        "episodes": expected_population,
        "scenes": expected_scenes,
        "schema_valid": expected_valid,
        "schema_invalid": expected_invalid,
    }
    decoded_manifest = _decode_json_object(
        artifacts.manifest_json, "manifest.json"
    )
    if decoded_manifest != dict(manifest):
        raise ValueError("manifest.json does not match the validated manifest")
    if decoded_manifest.get("population") != population:
        raise ValueError("manifest population contract is inconsistent")
    if decoded_manifest.get("oracle_only_limitation") != (
        "Angle selection uses ground-truth target raster cells and is not "
        "deployable performance."
    ):
        raise ValueError("manifest must state the oracle-only limitation")
    for field in (
        "cache_dir",
        "prediction_manifest_path",
        "prediction_manifest_sha256",
        "git_commit",
    ):
        value = decoded_manifest.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"manifest {field} must be a nonempty string")
    if decoded_manifest.get("cache_model_key") != _DEFAULT_CACHE_MODEL_KEY:
        raise ValueError("manifest cache model key provenance is inconsistent")
    if (
        decoded_manifest.get("cognitive_map_namespace")
        != _DEFAULT_COGNITIVE_MAP_NAMESPACE
    ):
        raise ValueError("manifest cognitive-map provenance is inconsistent")
    prediction_manifest_value = decoded_manifest["prediction_manifest_path"]
    prediction_sha_value = decoded_manifest["prediction_manifest_sha256"]
    if not isinstance(prediction_manifest_value, str) or not isinstance(
        prediction_sha_value, str
    ):
        raise ValueError("manifest prediction provenance must be strings")
    prediction_manifest_path = Path(prediction_manifest_value)
    if not prediction_manifest_path.is_file():
        raise ValueError("manifest prediction source file does not exist")
    if sha256(prediction_manifest_path.read_bytes()).hexdigest() != (
        prediction_sha_value
    ):
        raise ValueError("manifest prediction source SHA-256 is inconsistent")
    if decoded_manifest.get("dataset") != "R2R" or decoded_manifest.get(
        "split"
    ) != "val_unseen":
        raise ValueError("manifest dataset/split provenance is inconsistent")
    if (
        decoded_manifest.get("raster_shape") != list(_GRID_SHAPE)
        or decoded_manifest.get("grid_scale") != GRID_SCALE
        or decoded_manifest.get("cell_size_m") != CELL_SIZE * GRID_SCALE
        or decoded_manifest.get("angles_degrees") != list(_DEFAULT_ANGLES)
    ):
        raise ValueError("manifest raster/angle provenance is inconsistent")
    if (
        decoded_manifest.get("angle_tie_breaking")
        != "first declared angle, identity first"
        or decoded_manifest.get("aggregation_unit") != "episode"
        or decoded_manifest.get("invalid_prediction_policy")
        != (
            "retain missing or schema-invalid predictions as explicit empty "
            "predictions in the full denominator"
        )
    ):
        raise ValueError("manifest analysis protocol provenance is inconsistent")
    if decoded_manifest.get("channel_folds") != {
        "object": {"start_inclusive": 0, "stop_exclusive": 27},
        "region": {"start_inclusive": 27, "stop_exclusive": 37},
    }:
        raise ValueError("manifest channel fold provenance is inconsistent")
    if decoded_manifest.get("pivot_definitions") != {
        PivotMode.TRUE_START.value: "continuous scale-2 episode start coordinates",
        PivotMode.MAP_CENTER.value: [25.0, 25.0],
        PivotMode.SHUFFLED_START.value: (
            "within-scene one-to-one assigned true start"
        ),
    }:
        raise ValueError("manifest pivot provenance is inconsistent")
    shuffle = _require_json_object(decoded_manifest.get("shuffle"), "shuffle")
    if (
        shuffle.get("algorithm")
        != (
            "SHA-256-seeded receiver-ordered augmenting-path "
            "bipartite matching"
        )
        or
        shuffle.get("seed") != _DEFAULT_SHUFFLE_SEED
        or shuffle.get("preserves_scene_pivot_multiset") is not True
    ):
        raise ValueError("manifest shuffle provenance is inconsistent")
    if decoded_manifest.get("gate_thresholds") != {
        "true_start_symmetric_mean_minimum": 0.01,
        "true_start_symmetric_ci_lower_strictly_above": 0.0,
        "both_true_start_directional_means_strictly_above": 0.0,
        "true_start_minus_map_center_ci_lower_strictly_above": 0.0,
        "true_start_minus_shuffled_ci_lower_strictly_above": 0.0,
    }:
        raise ValueError("manifest gate provenance is inconsistent")

    hashes = _require_json_object(
        decoded_manifest.get("artifact_sha256"), "manifest artifact_sha256"
    )
    hashed_bytes = {
        "angle_scores.csv": artifacts.angle_scores_csv,
        "crossfit_results.csv": artifacts.crossfit_results_csv,
        "pivot_assignments.csv": artifacts.pivot_assignments_csv,
        "summary.json": artifacts.summary_json,
        "bootstrap.json": artifacts.bootstrap_json,
        "crossfit_control_intervals.png": artifacts.control_intervals_png,
    }
    if set(hashes) != set(hashed_bytes):
        raise ValueError("manifest artifact_sha256 has a mismatched file set")
    for filename, data in hashed_bytes.items():
        if hashes[filename] != sha256(data).hexdigest():
            raise ValueError(f"{filename} SHA-256 mismatch")
    if decoded_manifest.get("pivot_assignments_sha256") != hashes[
        "pivot_assignments.csv"
    ]:
        raise ValueError("pivot_assignments.csv SHA-256 provenance is inconsistent")
    if not artifacts.control_intervals_png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("crossfit_control_intervals.png is not a PNG artifact")
    try:
        decoded_plot = plt.imread(
            BytesIO(artifacts.control_intervals_png),
            format="png",
        )
    except (OSError, SyntaxError, ValueError) as error:
        raise ValueError(
            "crossfit_control_intervals.png must be a valid PNG image"
        ) from error
    if (
        decoded_plot.ndim < 2
        or decoded_plot.shape[0] <= 0
        or decoded_plot.shape[1] <= 0
        or not np.isfinite(decoded_plot.shape[:2]).all()
    ):
        raise ValueError(
            "crossfit_control_intervals.png must have nonempty finite dimensions"
        )

    angle_rows = _decode_csv_rows(
        artifacts.angle_scores_csv,
        "angle_scores.csv",
        _ANGLE_SCORE_HEADER,
        expected_population * len(PivotMode) * len(_DEFAULT_ANGLES),
    )
    crossfit_rows = _decode_csv_rows(
        artifacts.crossfit_results_csv,
        "crossfit_results.csv",
        _CROSSFIT_RESULT_HEADER,
        expected_population * len(PivotMode) * len(Direction),
    )
    assignment_rows = _decode_csv_rows(
        artifacts.pivot_assignments_csv,
        "pivot_assignments.csv",
        _PIVOT_ASSIGNMENT_HEADER,
        expected_population,
    )
    assignments, results = _artifact_models(
        angle_rows, crossfit_rows, assignment_rows
    )
    identities = {(row.scene_id, row.example_id) for row in assignments}
    if len(identities) != expected_population:
        raise ValueError("artifact identities must be unique")
    sorted_identities = tuple(
        sorted(
            identities,
            key=lambda identity: (
                identity[0].encode("utf-8"),
                identity[1].encode("utf-8"),
            ),
        )
    )
    if tuple((row.scene_id, row.example_id) for row in assignments) != (
        sorted_identities
    ):
        raise ValueError("artifact assignments must use stable UTF-8 identity order")
    if len({row.scene_id for row in assignments}) != expected_scenes:
        raise ValueError("artifact scene count is inconsistent")
    empty_grid = np.zeros(_GRID_SHAPE, dtype=np.bool_)
    expected_assignments = build_pivot_assignments(
        tuple(
            EpisodeCase(
                split="val_unseen",
                scene_id=row.scene_id,
                example_id=row.example_id,
                schema_valid=True,
                predicted_grid=empty_grid,
                target_grid=empty_grid,
                true_start_pivot=row.true_pivot,
            )
            for row in assignments
        )
    )
    if assignments != expected_assignments:
        raise ValueError(
            "artifact pivot assignments must equal the frozen SHA-256 "
            "seed-43 matching"
        )
    complete = _complete_pivot_results(results)
    if set(complete) != identities:
        raise ValueError("artifact pivot results must cover all assignments")
    expected_result_order = tuple(
        (scene_id, example_id, pivot_mode)
        for scene_id, example_id in sorted_identities
        for pivot_mode in _PIVOT_MODES
    )
    if tuple(
        (row.scene_id, row.example_id, row.pivot_mode) for row in results
    ) != expected_result_order:
        raise ValueError("artifact results must use stable identity/pivot order")
    if sum(rows[PivotMode.TRUE_START].schema_valid for rows in complete.values()) != (
        expected_valid
    ):
        raise ValueError("artifact valid status count is inconsistent")
    if sum(
        not rows[PivotMode.TRUE_START].schema_valid for rows in complete.values()
    ) != expected_invalid:
        raise ValueError("artifact invalid status count is inconsistent")
    assignment_by_identity = {
        (row.scene_id, row.example_id): row for row in assignments
    }
    for scene_id in {row.scene_id for row in assignments}:
        scene_assignments = tuple(
            row for row in assignments if row.scene_id == scene_id
        )
        if (
            Counter(row.donor_example_id for row in scene_assignments)
            != Counter(row.example_id for row in scene_assignments)
            or Counter(row.assigned_pivot for row in scene_assignments)
            != Counter(row.true_pivot for row in scene_assignments)
        ):
            raise ValueError(
                "artifact pivot assignments must preserve scene donor/pivot multisets"
            )
    for identity, pivot_rows in complete.items():
        assignment = assignment_by_identity[identity]
        true_start_valid = pivot_rows[PivotMode.TRUE_START].schema_valid
        for pivot_mode, row in pivot_rows.items():
            if row.split != "val_unseen" or row.schema_valid != true_start_valid:
                raise ValueError(
                    "artifact split/status must be consistent across every pivot"
                )
            expected_pivot = (
                assignment.true_pivot
                if pivot_mode is PivotMode.TRUE_START
                else (25.0, 25.0)
                if pivot_mode is PivotMode.MAP_CENTER
                else assignment.assigned_pivot
            )
            expected_donor = (
                assignment.donor_example_id
                if pivot_mode is PivotMode.SHUFFLED_START
                else None
            )
            if row.pivot != expected_pivot or row.donor_example_id != expected_donor:
                raise ValueError("artifact pivot or donor identity is inconsistent")
            if not true_start_valid:
                for angle_score in row.angle_scores:
                    for input_support, score in (
                        (
                            angle_score.object_input_support,
                            angle_score.object_score,
                        ),
                        (
                            angle_score.region_input_support,
                            angle_score.region_score,
                        ),
                    ):
                        if (
                            input_support != 0
                            or score.predicted_support != 0
                            or score.in_frame_support != 0
                            or score.out_of_frame_support != 0
                        ):
                            raise ValueError(
                                "schema-invalid identities must have empty "
                                "prediction support at every pivot and angle"
                            )
                for directional in (
                    row.object_to_region,
                    row.region_to_object,
                ):
                    if (
                        not directional.selector_predicted_support_empty
                        or not directional.heldout_predicted_support_empty
                    ):
                        raise ValueError(
                            "schema-invalid identities must have empty prediction "
                            "support and predicted-empty flags in every direction"
                        )

    summaries = summarize_results(results)
    ranges = leave_one_scene_out_ranges(results)
    estimates = bootstrap_results(results)
    true_start = summaries[PivotMode.TRUE_START]
    decision = classify_gate(
        true_start_symmetric=estimates.true_start_symmetric,
        true_start_object_to_region_mean=true_start.object_to_region_mean,
        true_start_region_to_object_mean=true_start.region_to_object_mean,
        start_minus_center=estimates.true_start_minus_map_center,
        start_minus_shuffled=estimates.true_start_minus_shuffled,
    )
    expected_summary = _summary_payload(
        population=population,
        summaries=summaries,
        ranges=ranges,
        estimates=estimates,
        gate_conditions=_gate_conditions(estimates),
        decision=decision,
    )
    if _decode_json_object(artifacts.summary_json, "summary.json") != expected_summary:
        raise ValueError("summary.json scientific invariants are inconsistent")
    if _decode_json_object(
        artifacts.bootstrap_json, "bootstrap.json"
    ) != _bootstrap_payload(estimates, expected_scenes):
        raise ValueError("bootstrap.json scientific invariants are inconsistent")
    expected_output_schema: Dict[str, object] = {
        "angle_scores.csv": {
            "columns": list(_ANGLE_SCORE_HEADER),
            "rows": len(angle_rows),
        },
        "crossfit_results.csv": {
            "columns": list(_CROSSFIT_RESULT_HEADER),
            "rows": len(crossfit_rows),
        },
        "pivot_assignments.csv": {
            "columns": list(_PIVOT_ASSIGNMENT_HEADER),
            "rows": len(assignment_rows),
        },
        "summary.json": {"format": "sorted_keys_indented_json"},
        "bootstrap.json": {"format": "sorted_keys_indented_json"},
        "crossfit_control_intervals.png": {
            "description": "ground-truth cross-fit diagnostic"
        },
    }
    if decoded_manifest.get("output_schema") != expected_output_schema:
        raise ValueError("manifest output schema is inconsistent")
    bootstrap_contract = _bootstrap_payload(estimates, expected_scenes)["contract"]
    if decoded_manifest.get("bootstrap") != bootstrap_contract:
        raise ValueError("manifest bootstrap contract is inconsistent")
    if decoded_manifest.get("gate_decision") != decision.value:
        raise ValueError("manifest gate decision is inconsistent")


def _validate_artifact_bundle_against_contract(
    artifacts: _ArtifactBundle,
    manifest: Mapping[str, object],
    *,
    expected_population: int,
    expected_scenes: int,
    expected_valid: int,
    expected_invalid: int,
) -> _ArtifactFiles:
    captured, files = _capture_artifact_bundle(artifacts)
    _validate_artifact_snapshot(
        captured,
        manifest,
        expected_population=expected_population,
        expected_scenes=expected_scenes,
        expected_valid=expected_valid,
        expected_invalid=expected_invalid,
    )
    return files


def _validate_artifact_bundle(
    artifacts: _ArtifactBundle,
) -> _ArtifactFiles:
    captured, files = _capture_artifact_bundle(artifacts)
    manifest = _decode_json_object(captured.manifest_json, "manifest.json")
    population = _require_json_object(
        manifest.get("population"), "manifest population"
    )
    _validate_artifact_snapshot(
        captured,
        manifest,
        expected_population=_manifest_population_count(population, "episodes"),
        expected_scenes=_manifest_population_count(population, "scenes"),
        expected_valid=_manifest_population_count(population, "schema_valid"),
        expected_invalid=_manifest_population_count(
            population, "schema_invalid"
        ),
    )
    return files


def validate_artifact_directory(
    output_dir: Path,
    *,
    expected_population: int,
    expected_scenes: int,
    expected_valid: int,
    expected_invalid: int,
) -> None:
    """Reconstruct and validate the complete fixed artifact transaction."""
    if not output_dir.is_dir():
        raise ValueError(f"artifact directory does not exist: {output_dir}")
    expected_names = set(
        _ArtifactBundle(
            manifest_json=b"",
            angle_scores_csv=b"",
            crossfit_results_csv=b"",
            pivot_assignments_csv=b"",
            summary_json=b"",
            bootstrap_json=b"",
            control_intervals_png=b"",
        ).files()
    )
    if {path.name for path in output_dir.iterdir()} != expected_names:
        raise ValueError("artifact directory has a mismatched fixed file set")
    artifacts = _ArtifactBundle(
        manifest_json=(output_dir / "manifest.json").read_bytes(),
        angle_scores_csv=(output_dir / "angle_scores.csv").read_bytes(),
        crossfit_results_csv=(output_dir / "crossfit_results.csv").read_bytes(),
        pivot_assignments_csv=(output_dir / "pivot_assignments.csv").read_bytes(),
        summary_json=(output_dir / "summary.json").read_bytes(),
        bootstrap_json=(output_dir / "bootstrap.json").read_bytes(),
        control_intervals_png=(
            output_dir / "crossfit_control_intervals.png"
        ).read_bytes(),
    )
    manifest = _decode_json_object(artifacts.manifest_json, "manifest.json")
    _validate_artifact_bundle_against_contract(
        artifacts,
        manifest,
        expected_population=expected_population,
        expected_scenes=expected_scenes,
        expected_valid=expected_valid,
        expected_invalid=expected_invalid,
    )


def _run_cases(
    args: CrossFitArgs,
    cases: Sequence[EpisodeCase],
    *,
    expected_population: int,
    expected_scenes: int,
    expected_valid: int,
    expected_invalid: int,
    prediction_manifest_path: Path,
) -> dict[str, object]:
    """Run the typed synthetic seam and create its fixed artifact transaction."""
    _validate_args(args)
    rows = tuple(cases)
    if len(rows) != expected_population:
        raise ValueError("population does not match the injected contract")
    if len({row.scene_id for row in rows}) != expected_scenes:
        raise ValueError("scene count does not match the injected contract")
    if sum(row.schema_valid for row in rows) != expected_valid:
        raise ValueError("valid count does not match the injected contract")
    if sum(not row.schema_valid for row in rows) != expected_invalid:
        raise ValueError("invalid count does not match the injected contract")
    if not prediction_manifest_path.is_file():
        raise FileNotFoundError(
            f"prediction_manifest_path must be a file: {prediction_manifest_path}"
        )

    ordered_cases = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.scene_id.encode("utf-8"),
                row.example_id.encode("utf-8"),
            ),
        )
    )
    assignments = build_pivot_assignments(ordered_cases)
    assignments_by_identity = {
        (row.scene_id, row.example_id): row for row in assignments
    }
    results = tuple(
        result
        for case in ordered_cases
        for result in evaluate_episode(
            case,
            assignments_by_identity[(case.scene_id, case.example_id)],
            tuple(args.angles),
        )
    )
    summaries = summarize_results(results)
    ranges = leave_one_scene_out_ranges(results)
    estimates = bootstrap_results(results)
    true_start_summary = summaries[PivotMode.TRUE_START]
    decision = classify_gate(
        true_start_symmetric=estimates.true_start_symmetric,
        true_start_object_to_region_mean=true_start_summary.object_to_region_mean,
        true_start_region_to_object_mean=true_start_summary.region_to_object_mean,
        start_minus_center=estimates.true_start_minus_map_center,
        start_minus_shuffled=estimates.true_start_minus_shuffled,
    )
    gate_conditions = _gate_conditions(estimates)
    population: Dict[str, object] = {
        "episodes": expected_population,
        "scenes": expected_scenes,
        "schema_valid": expected_valid,
        "schema_invalid": expected_invalid,
    }
    summary_payload = _summary_payload(
        population=population,
        summaries=summaries,
        ranges=ranges,
        estimates=estimates,
        gate_conditions=gate_conditions,
        decision=decision,
    )
    bootstrap_payload = _bootstrap_payload(estimates, expected_scenes)
    angle_rows = _angle_score_rows(results)
    crossfit_rows = _crossfit_result_rows(results)
    assignment_rows = _pivot_assignment_rows(assignments)
    _validate_analysis(
        cases=ordered_cases,
        assignments=assignments,
        results=results,
        summaries=summaries,
        ranges=ranges,
        estimates=estimates,
        gate_conditions=gate_conditions,
        decision=decision,
        angle_rows=angle_rows,
        crossfit_rows=crossfit_rows,
        pivot_assignment_rows=assignment_rows,
        summary_payload=summary_payload,
        bootstrap_payload=bootstrap_payload,
        expected_population=expected_population,
        expected_scenes=expected_scenes,
        expected_valid=expected_valid,
        expected_invalid=expected_invalid,
        angles=tuple(args.angles),
    )

    angle_scores_csv = _csv_bytes(_ANGLE_SCORE_HEADER, angle_rows)
    crossfit_results_csv = _csv_bytes(_CROSSFIT_RESULT_HEADER, crossfit_rows)
    pivot_assignments_csv = _csv_bytes(
        _PIVOT_ASSIGNMENT_HEADER, assignment_rows
    )
    summary_json = _json_bytes(summary_payload)
    bootstrap_json = _json_bytes(bootstrap_payload)
    control_intervals_png = _plot_bytes(estimates)
    artifact_sha256 = {
        "angle_scores.csv": sha256(angle_scores_csv).hexdigest(),
        "crossfit_results.csv": sha256(crossfit_results_csv).hexdigest(),
        "pivot_assignments.csv": sha256(pivot_assignments_csv).hexdigest(),
        "summary.json": sha256(summary_json).hexdigest(),
        "bootstrap.json": sha256(bootstrap_json).hexdigest(),
        "crossfit_control_intervals.png": sha256(
            control_intervals_png
        ).hexdigest(),
    }
    manifest: Dict[str, object] = {
        "dataset": "R2R",
        "split": "val_unseen",
        "cache_dir": str(args.cache_dir),
        "cache_model_key": args.cache_model_key,
        "cognitive_map_namespace": args.cognitive_map_namespace,
        "prediction_manifest_path": str(prediction_manifest_path),
        "prediction_manifest_sha256": sha256(
            prediction_manifest_path.read_bytes()
        ).hexdigest(),
        "git_commit": _git_commit(),
        "population": population,
        "raster_shape": list(_GRID_SHAPE),
        "grid_scale": GRID_SCALE,
        "cell_size_m": CELL_SIZE * GRID_SCALE,
        "angles_degrees": list(_DEFAULT_ANGLES),
        "angle_tie_breaking": "first declared angle, identity first",
        "aggregation_unit": "episode",
        "invalid_prediction_policy": (
            "retain missing or schema-invalid predictions as explicit empty "
            "predictions in the full denominator"
        ),
        "channel_folds": {
            "object": {"start_inclusive": 0, "stop_exclusive": 27},
            "region": {"start_inclusive": 27, "stop_exclusive": 37},
        },
        "pivot_definitions": {
            PivotMode.TRUE_START.value: (
                "continuous scale-2 episode start coordinates"
            ),
            PivotMode.MAP_CENTER.value: [25.0, 25.0],
            PivotMode.SHUFFLED_START.value: (
                "within-scene one-to-one assigned true start"
            ),
        },
        "shuffle": {
            "algorithm": (
                "SHA-256-seeded receiver-ordered augmenting-path "
                "bipartite matching"
            ),
            "seed": _DEFAULT_SHUFFLE_SEED,
            "preserves_scene_pivot_multiset": True,
        },
        "bootstrap": bootstrap_payload["contract"],
        "gate_thresholds": {
            "true_start_symmetric_mean_minimum": 0.01,
            "true_start_symmetric_ci_lower_strictly_above": 0.0,
            "both_true_start_directional_means_strictly_above": 0.0,
            "true_start_minus_map_center_ci_lower_strictly_above": 0.0,
            "true_start_minus_shuffled_ci_lower_strictly_above": 0.0,
        },
        "gate_decision": decision.value,
        "output_schema": {
            "angle_scores.csv": {
                "columns": list(_ANGLE_SCORE_HEADER),
                "rows": len(angle_rows),
            },
            "crossfit_results.csv": {
                "columns": list(_CROSSFIT_RESULT_HEADER),
                "rows": len(crossfit_rows),
            },
            "pivot_assignments.csv": {
                "columns": list(_PIVOT_ASSIGNMENT_HEADER),
                "rows": len(assignment_rows),
            },
            "summary.json": {"format": "sorted_keys_indented_json"},
            "bootstrap.json": {"format": "sorted_keys_indented_json"},
            "crossfit_control_intervals.png": {
                "description": "ground-truth cross-fit diagnostic"
            },
        },
        "artifact_sha256": artifact_sha256,
        "pivot_assignments_sha256": artifact_sha256["pivot_assignments.csv"],
        "oracle_only_limitation": (
            "Angle selection uses ground-truth target raster cells and is not "
            "deployable performance."
        ),
    }
    artifacts = _ArtifactBundle(
        manifest_json=_json_bytes(manifest),
        angle_scores_csv=angle_scores_csv,
        crossfit_results_csv=crossfit_results_csv,
        pivot_assignments_csv=pivot_assignments_csv,
        summary_json=summary_json,
        bootstrap_json=bootstrap_json,
        control_intervals_png=control_intervals_png,
    )
    _validate_artifact_bundle_against_contract(
        artifacts,
        manifest,
        expected_population=expected_population,
        expected_scenes=expected_scenes,
        expected_valid=expected_valid,
        expected_invalid=expected_invalid,
    )
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"output directory must be empty or absent, got nonempty: {output_dir}"
        )
    _write_artifacts(output_dir, artifacts)
    return {
        "output_dir": output_dir,
        "manifest": manifest,
        "summary": summary_payload,
        "bootstrap": bootstrap_payload,
        "decision": decision,
    }


def run_crossfit(args: CrossFitArgs) -> dict[str, object]:
    """Run only the complete frozen R2R val_unseen analysis."""
    episodes, prediction_manifest_path = _load_episode_cases(args)
    return _run_cases(
        args,
        episodes,
        expected_population=1_839,
        expected_scenes=11,
        expected_valid=1_830,
        expected_invalid=9,
        prediction_manifest_path=prediction_manifest_path,
    )


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    args = CrossFitArgs(underscores_to_dashes=True).parse_args(argv)
    result = run_crossfit(args)
    if not args.quiet:
        print(f"Wrote cross-fit artifacts to {result['output_dir']}")
    return result


if __name__ == "__main__":
    main()
