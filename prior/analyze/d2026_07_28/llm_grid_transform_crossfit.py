"""Typed object/region cross-fit scoring for LLM-Grid transform controls."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

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
