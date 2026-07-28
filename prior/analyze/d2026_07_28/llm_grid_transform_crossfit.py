"""Typed object/region cross-fit scoring for LLM-Grid transform controls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from numbers import Integral, Real
from typing import Dict, Optional, Sequence, Set

import numpy as np
from numpy.typing import NDArray

from prior.analyze.llm_grid_registration import (
    RasterScore,
    SpatialBounds,
    WarpedGrid,
    score_warped_grid,
    warp_grid_about_pivot,
)


_CHANNEL_COUNT = 37
_OBJECT_CHANNEL_COUNT = 27
_GRID_SHAPE = (_CHANNEL_COUNT, 50, 50)


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
        if (
            len(scene_rows) != len(scene_episodes)
            or donor_ids != expected_donor_ids
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
