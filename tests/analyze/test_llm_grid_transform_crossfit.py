from __future__ import annotations

import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

import numpy as np
from numpy.typing import NDArray
import pytest

from prior.analyze.d2026_07_28 import llm_grid_transform_crossfit as crossfit
from prior.analyze.d2026_07_28.llm_grid_transform_crossfit import (
    CrossFitArgs,
    CrossFitResult,
    Direction,
    EndpointEstimate,
    EpisodeCase,
    EpisodePivotResult,
    FamilyAngleScore,
    GateDecision,
    PivotAssignment,
    PivotMode,
    _bootstrap_episode_macro,
    _load_episode_cases,
    _prediction_manifest_path,
    _validate_args,
    build_pivot_assignments,
    bootstrap_results,
    classify_gate,
    crossfit_scores,
    evaluate_episode,
    leave_one_scene_out_ranges,
    pivot_for_mode,
    scene_bootstrap_multiplicities,
    score_angle_families,
    summarize_results,
)
from vlnce_baselines.models.etp_llm.llm_grid_train import (
    LLMGridExample,
)
from vlnce_baselines.models.etp_llm.sft import ExampleLoadResult, SourceLoadStats
from prior.analyze.llm_grid_registration import (
    RasterScore,
    SpatialBounds,
    WarpedGrid,
    warp_grid_about_pivot,
)


def family_score(
    angle: float, *, object_iou_tenths: int, region_iou_tenths: int
) -> FamilyAngleScore:
    bounds = SpatialBounds(0, 50, 0, 50)

    def raster(intersection: int) -> RasterScore:
        return RasterScore(
            intersection=intersection,
            union=10,
            predicted_support=intersection,
            target_support=10,
            in_frame_support=intersection,
            out_of_frame_support=0,
        )

    return FamilyAngleScore(
        angle_degrees=angle,
        bounds=bounds,
        object_input_support=object_iou_tenths,
        region_input_support=region_iou_tenths,
        object_score=raster(object_iou_tenths),
        region_score=raster(region_iou_tenths),
    )


@pytest.fixture
def canonical_grids() -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    predicted = np.zeros((37, 50, 50), dtype=np.bool_)
    return predicted, np.zeros_like(predicted)


def episode_case(
    example_id: str,
    pivot: tuple[float, float],
    *,
    scene_id: str = "scene-1",
    schema_valid: bool = True,
    predicted_grid: Optional[NDArray[np.bool_]] = None,
    target_grid: Optional[NDArray[np.bool_]] = None,
) -> EpisodeCase:
    predicted = (
        np.zeros((37, 50, 50), dtype=np.bool_)
        if predicted_grid is None
        else predicted_grid
    )
    target = np.zeros_like(predicted) if target_grid is None else target_grid
    return EpisodeCase(
        split="val_unseen",
        scene_id=scene_id,
        example_id=example_id,
        schema_valid=schema_valid,
        predicted_grid=predicted,
        target_grid=target,
        true_start_pivot=pivot,
    )


def four_episode_cases() -> tuple[EpisodeCase, ...]:
    return tuple(
        episode_case(f"episode-{name}", (float(index), float(index)))
        for index, name in enumerate(("a", "b", "c", "d"), start=1)
    )


def test_pivot_assignment_matches_frozen_sha256_fixture() -> None:
    """Breaks if the frozen digest ordering or DFS matching changes."""
    assignments = build_pivot_assignments(four_episode_cases())

    assert [(row.example_id, row.donor_example_id) for row in assignments] == [
        ("episode-a", "episode-b"),
        ("episode-b", "episode-c"),
        ("episode-c", "episode-d"),
        ("episode-d", "episode-a"),
    ]


def test_pivot_matching_is_input_order_independent_and_scene_local() -> None:
    """Breaks if matching consumes input order or crosses a scene boundary."""
    episodes = four_episode_cases() + (
        episode_case("episode-e", (5.0, 5.0), scene_id="scene-2"),
        episode_case("episode-f", (6.0, 6.0), scene_id="scene-2"),
    )

    assignments = build_pivot_assignments(episodes)

    assert assignments == build_pivot_assignments(tuple(reversed(episodes)))
    assert {row.example_id for row in assignments} == {row.example_id for row in episodes}
    assert len({(row.scene_id, row.donor_example_id) for row in assignments}) == len(
        assignments
    )
    assert all(
        row.example_id != row.donor_example_id
        and row.true_pivot != row.assigned_pivot
        for row in assignments
    )


def test_pivot_matching_rejects_duplicate_pivots_without_eligible_edges() -> None:
    """Breaks if equal-pivot donors remain eligible after matching setup."""
    episodes = (
        episode_case("episode-a", (1.0, 1.0)),
        episode_case("episode-b", (1.0, 1.0)),
        episode_case("episode-c", (2.0, 2.0)),
    )

    with pytest.raises(
        ValueError,
        match="scene scene-1 has no valid shuffled-start perfect matching",
    ):
        build_pivot_assignments(episodes)


def test_pivot_matching_excludes_equal_edges_with_solvable_duplicate_pivots() -> None:
    """Breaks if a solvable duplicate-start scene uses an equal-pivot donor."""
    episodes = (
        episode_case("episode-a", (1.0, 1.0)),
        episode_case("episode-b", (1.0, 1.0)),
        episode_case("episode-c", (2.0, 2.0)),
        episode_case("episode-d", (2.0, 2.0)),
    )

    assignments = build_pivot_assignments(episodes)

    assert len(assignments) == len(episodes)
    assert len({row.donor_example_id for row in assignments}) == len(episodes)
    assert all(row.true_pivot != row.assigned_pivot for row in assignments)
    assert sorted(row.assigned_pivot for row in assignments) == sorted(
        episode.true_start_pivot for episode in episodes
    )


def test_pivot_matching_rejects_postmatch_assigned_pivot_multiset_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if validation accepts donor IDs with a corrupted pivot multiset."""
    episodes = four_episode_cases()
    assignments = build_pivot_assignments(episodes)
    corrupted_assignments = tuple(
        replace(row, assigned_pivot=(9.0, 9.0))
        if row.example_id == "episode-a"
        else row
        for row in assignments
    )

    def corrupted_scene_match(
        scene_id: str, scene_episodes: tuple[EpisodeCase, ...]
    ) -> tuple[PivotAssignment, ...]:
        assert scene_id == "scene-1"
        assert tuple(scene_episodes) == episodes
        return corrupted_assignments

    monkeypatch.setattr(
        crossfit, "_build_scene_pivot_assignments", corrupted_scene_match
    )

    with pytest.raises(
        ValueError,
        match="scene scene-1 has no valid shuffled-start perfect matching",
    ):
        build_pivot_assignments(episodes)


@pytest.mark.parametrize(
    ("episodes", "scene_id"),
    (
        ((episode_case("only", (1.0, 1.0)),), "scene-1"),
        (
            (
                episode_case("episode-a", (1.0, 1.0)),
                episode_case("episode-b", (1.0, 1.0)),
            ),
            "scene-1",
        ),
    ),
)
def test_pivot_matching_rejects_impossible_scene(
    episodes: tuple[EpisodeCase, ...], scene_id: str
) -> None:
    """Breaks if an impossible within-scene derangement is silently accepted."""
    with pytest.raises(
        ValueError,
        match=rf"scene {scene_id} has no valid shuffled-start perfect matching",
    ):
        build_pivot_assignments(episodes)


def test_pivot_for_mode_returns_declared_control_pivots() -> None:
    """Breaks if a pivot mode substitutes a derived or wrong start coordinate."""
    episode = episode_case("episode-a", (4.0, 8.0))
    assignment = PivotAssignment(
        scene_id="scene-1",
        example_id="episode-a",
        donor_example_id="episode-b",
        true_pivot=(4.0, 8.0),
        assigned_pivot=(12.0, 16.0),
    )

    assert pivot_for_mode(episode, assignment, PivotMode.TRUE_START) == (4.0, 8.0)
    assert pivot_for_mode(episode, assignment, PivotMode.MAP_CENTER) == (25.0, 25.0)
    assert pivot_for_mode(episode, assignment, PivotMode.SHUFFLED_START) == (12.0, 16.0)


def test_evaluate_episode_true_start_beats_center_and_shuffled_pivots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if all three controls do not score their own pivot geometry."""
    predicted = np.zeros((37, 50, 50), dtype=np.bool_)
    predicted[0, 10, 11] = True
    predicted[27, 11, 10] = True
    true_pivot = (10.5, 10.5)
    warped = warp_grid_about_pivot(predicted, true_pivot, 90.0)
    target = np.zeros_like(predicted)
    row_start = max(warped.bounds.row_min, 0)
    row_end = min(warped.bounds.row_max, 50)
    col_start = max(warped.bounds.col_min, 0)
    col_end = min(warped.bounds.col_max, 50)
    target[:, row_start:row_end, col_start:col_end] = warped.grid[
        :,
        row_start - warped.bounds.row_min : row_end - warped.bounds.row_min,
        col_start - warped.bounds.col_min : col_end - warped.bounds.col_min,
    ]
    episode = episode_case(
        "episode-a",
        true_pivot,
        predicted_grid=predicted,
        target_grid=target,
    )
    assignment = PivotAssignment(
        scene_id="scene-1",
        example_id="episode-a",
        donor_example_id="episode-b",
        true_pivot=true_pivot,
        assigned_pivot=(2.5, 2.5),
    )
    calls: list[tuple[tuple[float, float], tuple[float, ...]]] = []
    real_score_angle_families = crossfit.score_angle_families

    def spy_score_angle_families(
        scored_prediction: NDArray[np.bool_],
        scored_target: NDArray[np.bool_],
        pivot: tuple[float, float],
        angles: tuple[float, ...],
    ) -> tuple[FamilyAngleScore, ...]:
        assert scored_prediction is predicted
        assert scored_target is target
        calls.append((pivot, angles))
        return real_score_angle_families(
            scored_prediction, scored_target, pivot, angles
        )

    monkeypatch.setattr(crossfit, "score_angle_families", spy_score_angle_families)

    results = evaluate_episode(episode, assignment, (0.0, 90.0))

    assert all(isinstance(result, EpisodePivotResult) for result in results)
    assert tuple(result.pivot_mode for result in results) == tuple(PivotMode)
    assert calls == [
        (true_pivot, (0.0, 90.0)),
        ((25.0, 25.0), (0.0, 90.0)),
        ((2.5, 2.5), (0.0, 90.0)),
    ]
    assert all(
        result.object_to_region
        == crossfit.crossfit_scores(result.angle_scores, Direction.OBJECT_TO_REGION)
        and result.region_to_object
        == crossfit.crossfit_scores(result.angle_scores, Direction.REGION_TO_OBJECT)
        for result in results
    )
    deltas = {result.pivot_mode: result.symmetric_delta for result in results}
    assert deltas[PivotMode.TRUE_START] > deltas[PivotMode.MAP_CENTER]
    assert deltas[PivotMode.TRUE_START] > deltas[PivotMode.SHUFFLED_START]


def test_invalid_episode_is_retained_at_every_pivot_with_zero_deltas() -> None:
    """Breaks if empty invalid predictions are dropped or acquire a false delta."""
    target = np.zeros((37, 50, 50), dtype=np.bool_)
    target[0, 10, 10] = True
    target[27, 12, 12] = True
    episode = episode_case(
        "episode-a",
        (4.0, 4.0),
        schema_valid=False,
        target_grid=target,
    )
    assignment = PivotAssignment(
        scene_id="scene-1",
        example_id="episode-a",
        donor_example_id="episode-b",
        true_pivot=(4.0, 4.0),
        assigned_pivot=(12.0, 12.0),
    )

    results = evaluate_episode(episode, assignment, (0.0, 90.0))

    assert len(results) == len(PivotMode)
    assert all(not result.schema_valid for result in results)
    assert all(result.object_to_region.delta_iou == 0.0 for result in results)
    assert all(result.region_to_object.delta_iou == 0.0 for result in results)


def test_evaluate_episode_rejects_assignment_for_another_identity() -> None:
    """Breaks if results can label one episode with another assignment's pivot."""
    episode = episode_case("episode-a", (4.0, 4.0))
    assignment = PivotAssignment(
        scene_id="scene-1",
        example_id="episode-b",
        donor_example_id="episode-a",
        true_pivot=(5.0, 5.0),
        assigned_pivot=(4.0, 4.0),
    )

    with pytest.raises(ValueError, match="assignment identity"):
        evaluate_episode(episode, assignment, (0.0, 90.0))


def test_crossfit_selection_uses_only_declared_family() -> None:
    """Breaks if held-out scores independently select their own best angle."""
    scores = (
        family_score(0.0, object_iou_tenths=2, region_iou_tenths=8),
        family_score(90.0, object_iou_tenths=9, region_iou_tenths=1),
        family_score(180.0, object_iou_tenths=1, region_iou_tenths=1),
        family_score(270.0, object_iou_tenths=1, region_iou_tenths=1),
    )

    object_to_region = crossfit_scores(scores, Direction.OBJECT_TO_REGION)
    region_to_object = crossfit_scores(scores, Direction.REGION_TO_OBJECT)

    assert object_to_region.selected_angle_degrees == 90.0
    assert object_to_region.selector_second_iou == 0.2
    assert object_to_region.heldout_selected_iou == 0.1
    assert object_to_region.delta_iou == pytest.approx(-0.7)
    assert region_to_object.selected_angle_degrees == 0.0
    assert region_to_object.heldout_selected_iou == 0.2
    assert region_to_object.delta_iou == 0.0


def test_crossfit_tie_retains_declared_identity_angle() -> None:
    """Breaks if a tie changes the caller-declared identity-angle selection."""
    scores = (
        family_score(0.0, object_iou_tenths=0, region_iou_tenths=0),
        family_score(90.0, object_iou_tenths=0, region_iou_tenths=0),
        family_score(180.0, object_iou_tenths=0, region_iou_tenths=0),
        family_score(270.0, object_iou_tenths=0, region_iou_tenths=0),
    )

    result = crossfit_scores(scores, Direction.OBJECT_TO_REGION)

    assert result.selected_angle_degrees == 0.0
    assert result.selector_margin == 0.0
    assert result.delta_iou == 0.0


def test_score_angle_families_preserves_family_support_and_out_of_frame_pixels(
    canonical_grids: tuple[NDArray[np.bool_], NDArray[np.bool_]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if family scoring rewarps slices or drops rotated false positives."""
    predicted, target = canonical_grids
    predicted[0, 10, 10] = True
    predicted[27, 12, 12] = True
    target[:] = predicted
    calls: list[tuple[NDArray[np.bool_], tuple[float, float], float]] = []
    real_warp = crossfit.warp_grid_about_pivot

    def spy_warp(
        grid: NDArray[np.bool_], pivot: tuple[float, float], angle: float
    ) -> WarpedGrid:
        calls.append((grid, pivot, angle))
        return real_warp(grid, pivot, angle)

    monkeypatch.setattr(crossfit, "warp_grid_about_pivot", spy_warp)

    scores = crossfit.score_angle_families(
        predicted,
        target,
        pivot=(0.5, 0.5),
        angles=(180.0, 0.0, 90.0),
    )

    assert tuple(score.angle_degrees for score in scores) == (180.0, 0.0, 90.0)
    assert scores[0].bounds == SpatialBounds(-49, 1, -49, 1)
    assert scores[1].bounds == SpatialBounds(0, 50, 0, 50)
    assert scores[0].object_input_support == 1
    assert scores[0].region_input_support == 1
    assert scores[0].object_score.out_of_frame_support == 1
    assert scores[0].region_score.out_of_frame_support == 1
    assert [call[2] for call in calls] == [180.0, 0.0, 90.0]
    assert all(call[0].shape == (37, 50, 50) for call in calls)


def test_score_angle_families_rejects_malformed_grid_or_pivot(
    canonical_grids: tuple[NDArray[np.bool_], NDArray[np.bool_]],
) -> None:
    """Breaks if malformed grids reach geometry or hidden coercion paths."""
    predicted, target = canonical_grids

    with pytest.raises(ValueError, match="boolean"):
        score_angle_families(
            predicted.astype(np.float32), target, (1.0, 1.0), (0.0,)
        )
    with pytest.raises(ValueError, match="37"):
        score_angle_families(
            predicted[:36], target[:36], (1.0, 1.0), (0.0,)
        )
    with pytest.raises(ValueError, match="shape"):
        score_angle_families(
            predicted, target[:, :49], (1.0, 1.0), (0.0,)
        )
    with pytest.raises(ValueError, match="shape"):
        score_angle_families(
            predicted[:, :, :49], target[:, :, :49], (1.0, 1.0), (0.0,)
        )
    with pytest.raises(ValueError, match="pivot"):
        score_angle_families(predicted, target, (np.nan, 1.0), (0.0,))


def test_crossfit_rejects_duplicate_or_missing_identity_angles() -> None:
    """Breaks if cross-fit output can lack an unambiguous baseline row."""
    with pytest.raises(ValueError, match="unique"):
        crossfit_scores(
            (
                family_score(0.0, object_iou_tenths=1, region_iou_tenths=1),
                family_score(0.0, object_iou_tenths=2, region_iou_tenths=2),
            ),
            Direction.OBJECT_TO_REGION,
        )
    with pytest.raises(ValueError, match="identity"):
        crossfit_scores(
            (
                family_score(90.0, object_iou_tenths=1, region_iou_tenths=1),
                family_score(180.0, object_iou_tenths=1, region_iou_tenths=1),
            ),
            Direction.OBJECT_TO_REGION,
        )


def test_episode_case_rejects_noncanonical_grid_or_empty_identity(
    canonical_grids: tuple[NDArray[np.bool_], NDArray[np.bool_]],
) -> None:
    """Breaks if records admit malformed data before grouped analysis."""
    predicted, target = canonical_grids
    assert PivotMode.TRUE_START.value == "true_start"
    valid = EpisodeCase(
        split="val_unseen",
        scene_id="scene",
        example_id="example",
        schema_valid=True,
        predicted_grid=predicted,
        target_grid=target,
        true_start_pivot=(1.0, 1.0),
    )
    assert valid.true_start_pivot == (1.0, 1.0)
    with pytest.raises(ValueError, match="split"):
        EpisodeCase(
            split="",
            scene_id="scene",
            example_id="example",
            schema_valid=True,
            predicted_grid=predicted,
            target_grid=target,
            true_start_pivot=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="shape"):
        EpisodeCase(
            split="val_unseen",
            scene_id="scene",
            example_id="example",
            schema_valid=True,
            predicted_grid=predicted[:, :, :49],
            target_grid=target[:, :, :49],
            true_start_pivot=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="boolean"):
        EpisodeCase(
            split="val_unseen",
            scene_id="scene",
            example_id="example",
            schema_valid=True,
            predicted_grid=predicted.astype(np.float32),
            target_grid=target,
            true_start_pivot=(1.0, 1.0),
        )


def test_crossfit_result_rejects_inconsistent_selector_margin() -> None:
    """Breaks if a serialized result can report a margin unrelated to its scores."""
    with pytest.raises(ValueError, match="selector_margin"):
        CrossFitResult(
            direction=Direction.OBJECT_TO_REGION,
            selected_angle_degrees=90.0,
            second_angle_degrees=0.0,
            selector_identity_iou=0.2,
            selector_selected_iou=0.9,
            selector_second_iou=0.2,
            selector_margin=0.6,
            heldout_identity_iou=0.8,
            heldout_selected_iou=0.1,
            delta_iou=-0.7,
            selector_predicted_support_empty=False,
            selector_target_support_empty=False,
            selector_union_empty=False,
            heldout_predicted_support_empty=False,
            heldout_target_support_empty=False,
            heldout_union_empty=False,
        )


def synthetic_crossfit_result(
    direction: Direction,
    delta: float,
    *,
    selected_angle: float = 90.0,
    empty_flags: Optional[tuple[bool, bool, bool, bool, bool, bool]] = None,
) -> CrossFitResult:
    """Builds a valid held-out result; breaks if test fixtures stop matching records."""
    flags = (False,) * 6 if empty_flags is None else empty_flags
    return CrossFitResult(
        direction=direction,
        selected_angle_degrees=selected_angle,
        second_angle_degrees=0.0,
        selector_identity_iou=0.1,
        selector_selected_iou=0.2,
        selector_second_iou=0.1,
        selector_margin=0.1,
        heldout_identity_iou=0.5,
        heldout_selected_iou=0.5 + delta,
        delta_iou=delta,
        selector_predicted_support_empty=flags[0],
        selector_target_support_empty=flags[1],
        selector_union_empty=flags[2],
        heldout_predicted_support_empty=flags[3],
        heldout_target_support_empty=flags[4],
        heldout_union_empty=flags[5],
    )


def synthetic_pivot_result(
    scene_id: str,
    example_id: str,
    pivot_mode: PivotMode,
    object_delta: float,
    region_delta: float,
    *,
    schema_valid: bool = True,
    object_empty_flags: Optional[tuple[bool, bool, bool, bool, bool, bool]] = None,
    region_empty_flags: Optional[tuple[bool, bool, bool, bool, bool, bool]] = None,
) -> EpisodePivotResult:
    """Builds one real typed pivot row with literal directional deltas."""
    return EpisodePivotResult(
        split="val_unseen",
        scene_id=scene_id,
        example_id=example_id,
        schema_valid=schema_valid,
        pivot_mode=pivot_mode,
        pivot=(1.0, 1.0),
        donor_example_id="donor" if pivot_mode is PivotMode.SHUFFLED_START else None,
        angle_scores=(family_score(0.0, object_iou_tenths=1, region_iou_tenths=1),),
        object_to_region=synthetic_crossfit_result(
            Direction.OBJECT_TO_REGION,
            object_delta,
            selected_angle=90.0,
            empty_flags=object_empty_flags,
        ),
        region_to_object=synthetic_crossfit_result(
            Direction.REGION_TO_OBJECT,
            region_delta,
            selected_angle=180.0,
            empty_flags=region_empty_flags,
        ),
    )


def object_empty_flag_pattern(index: int) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Returns six different retained-row rates for the object direction."""
    return (
        index < 0,
        index < 1,
        index < 2,
        index < 3,
        index < 4,
        index < 5,
    )


def region_empty_flag_pattern(index: int) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Returns the reverse six-rate pattern for the region direction."""
    return (
        index < 6,
        index < 5,
        index < 4,
        index < 3,
        index < 2,
        index < 1,
    )


def inference_rows() -> tuple[EpisodePivotResult, ...]:
    """Three unequal scenes retain an invalid row in every declared pivot."""
    rows: list[EpisodePivotResult] = []
    values = (
        ("A", "a", True, (0.10, -0.10), (0.05, 0.20), (0.15, 0.05)),
        ("B", "b1", True, (0.20, 0.00), (0.10, 0.10), (0.05, -0.10)),
        ("B", "b2", True, (0.30, 0.10), (0.15, 0.00), (0.10, 0.00)),
        ("C", "c1", True, (0.00, 0.20), (-0.05, -0.10), (0.20, 0.10)),
        ("C", "c2", True, (0.10, 0.10), (0.00, 0.00), (0.10, 0.20)),
        ("C", "c3", True, (0.20, 0.00), (0.05, 0.10), (0.00, 0.30)),
        ("C", "c4", False, (0.30, -0.10), (0.10, 0.20), (-0.10, 0.40)),
    )
    for index, (scene_id, example_id, schema_valid, true, center, shuffled) in enumerate(
        values
    ):
        object_empty_flags = object_empty_flag_pattern(index)
        region_empty_flags = region_empty_flag_pattern(index)
        rows.extend(
            (
                synthetic_pivot_result(
                    scene_id,
                    example_id,
                    PivotMode.TRUE_START,
                    *true,
                    schema_valid=schema_valid,
                    object_empty_flags=object_empty_flags,
                    region_empty_flags=region_empty_flags,
                ),
                synthetic_pivot_result(
                    scene_id,
                    example_id,
                    PivotMode.MAP_CENTER,
                    *center,
                    schema_valid=schema_valid,
                ),
                synthetic_pivot_result(
                    scene_id,
                    example_id,
                    PivotMode.SHUFFLED_START,
                    *shuffled,
                    schema_valid=schema_valid,
                ),
            )
        )
    return tuple(rows)


def test_bootstrap_replicates_preserve_episode_weighting() -> None:
    """Breaks if multiplicities average unequal scene means equally."""
    scene_ids = ("A", "B")
    multiplicities = np.asarray(((1, 1), (2, 0), (0, 2)), dtype=np.int64)
    values = {"A": (1.0,), "B": (0.0, 0.0, 0.0)}

    replicates = _bootstrap_episode_macro(values, scene_ids, multiplicities)

    np.testing.assert_allclose(replicates, (0.25, 1.0, 0.0))


def test_scene_bootstrap_multiplicities_are_seeded_and_scene_sorted() -> None:
    """Breaks if scene bootstrap consumes row order or an unfrozen random source."""
    rows = inference_rows()

    first_scenes, first = scene_bootstrap_multiplicities(rows)
    second_scenes, second = scene_bootstrap_multiplicities(tuple(reversed(rows)))

    assert first_scenes == ("A", "B", "C")
    assert second_scenes == first_scenes
    np.testing.assert_array_equal(second, first)
    assert first.shape == (10_000, 3)
    assert first.dtype == np.int64


def _fixed_multiplicities() -> NDArray[np.int64]:
    return np.asarray(((1, 1, 1), (3, 0, 0), (0, 2, 1), (0, 0, 3)), dtype=np.int64)


def test_all_pivot_endpoints_use_hand_calculated_means_and_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if any directional/symmetric endpoint is wired to the wrong pivot."""
    monkeypatch.setattr(
        crossfit,
        "scene_bootstrap_multiplicities",
        lambda _: (("A", "B", "C"), _fixed_multiplicities()),
    )

    estimates = bootstrap_results(inference_rows())

    expected = (
        (estimates.true_start_object_to_region, 0.1714285714, 0.10375, 0.1978571429),
        (estimates.true_start_region_to_object, 0.0285714286, -0.0903571429, 0.05),
        (estimates.true_start_symmetric, 0.1, 0.0075, 0.123125),
        (estimates.map_center_object_to_region, 0.0571428571, 0.026875, 0.0736607143),
        (estimates.map_center_region_to_object, 0.0714285714, 0.05, 0.1903571429),
        (estimates.map_center_symmetric, 0.0642857143, 0.039375, 0.1204464286),
        (estimates.shuffled_start_object_to_region, 0.0714285714, 0.0509375, 0.1441071429),
        (estimates.shuffled_start_region_to_object, 0.1357142857, 0.05375, 0.2414285714),
        (estimates.shuffled_start_symmetric, 0.1035714286, 0.08265625, 0.1465178571),
    )
    for estimate_value, mean, lower, upper in expected:
        assert estimate_value.mean == pytest.approx(mean)
        assert estimate_value.ci_lower == pytest.approx(lower)
        assert estimate_value.ci_upper == pytest.approx(upper)
    for estimate in estimates.all():
        assert np.isfinite(
            (
                estimate.mean,
                estimate.ci_lower,
                estimate.ci_upper,
                estimate.leave_one_scene_out_min,
                estimate.leave_one_scene_out_max,
            )
        ).all()


def test_all_endpoints_receive_the_same_bootstrap_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if an endpoint draws its own scene bootstrap matrix."""
    received: list[tuple[Mapping[str, tuple[float, ...]], NDArray[np.int64]]] = []
    real_bootstrap = crossfit._bootstrap_episode_macro

    def capture_bootstrap(
        values: Mapping[str, tuple[float, ...]],
        scene_ids: tuple[str, ...],
        multiplicities: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        received.append((values, multiplicities))
        return real_bootstrap(values, scene_ids, multiplicities)

    monkeypatch.setattr(crossfit, "_bootstrap_episode_macro", capture_bootstrap)

    bootstrap_results(inference_rows())

    assert len(received) == 11
    assert all(matrix is received[0][1] for _, matrix in received)
    expected_center = {"A": (-0.125,), "B": (0.0, 0.125), "C": (0.175, 0.1, 0.025, -0.05)}
    expected_shuffled = {"A": (-0.1,), "B": (0.125, 0.15), "C": (-0.05, -0.05, -0.05, -0.05)}
    for captured, expected in ((received[9][0], expected_center), (received[10][0], expected_shuffled)):
        assert tuple(captured) == ("A", "B", "C")
        for scene_id, values in expected.items():
            assert captured[scene_id] == pytest.approx(values)


def test_leave_one_scene_out_reports_only_episode_macro_range() -> None:
    """Breaks if LOSO averages remaining scenes equally rather than episodes."""
    ranges = leave_one_scene_out_ranges(inference_rows())

    assert ranges.true_start_symmetric == pytest.approx((0.08, 0.1166666667))
    assert ranges.true_start_minus_map_center == pytest.approx((0.0, 0.0625))


def test_summary_retains_invalid_rows_and_all_empty_rates() -> None:
    """Breaks if summary filtering removes invalid rows or collapses empty flags."""
    summary = summarize_results(inference_rows())
    true_start = summary[PivotMode.TRUE_START]

    assert true_start.object_to_region_mean == pytest.approx(0.1714285714)
    assert true_start.region_to_object_mean == pytest.approx(0.0285714286)
    assert true_start.symmetric_mean == pytest.approx(0.1)
    assert true_start.object_to_region_angle_counts == ((90.0, 7),)
    assert true_start.region_to_object_angle_counts == ((180.0, 7),)
    assert true_start.object_to_region_empty_rates == pytest.approx(
        (0.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0)
    )
    assert true_start.region_to_object_empty_rates == pytest.approx(
        (6.0 / 7.0, 5.0 / 7.0, 4.0 / 7.0, 3.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0)
    )


@pytest.mark.parametrize(
    ("values", "scene_ids", "multiplicities", "match"),
    (
        ({"A": (1.0,)}, ("A", "B"), np.ones((1, 2), dtype=np.int64), "keys"),
        ({"A": (), "B": (1.0,)}, ("A", "B"), np.ones((1, 2), dtype=np.int64), "scene"),
        ({"A": (np.nan,), "B": (1.0,)}, ("A", "B"), np.ones((1, 2), dtype=np.int64), "finite"),
        ({"A": (1.0,), "B": (1.0,)}, ("A", "B"), np.ones((1, 2), dtype=np.float64), "integer"),
        ({"A": (1.0,), "B": (1.0,)}, ("A", "B"), np.asarray(((1, -1),), dtype=np.int64), "nonnegative"),
        ({"A": (1.0,), "B": (1.0,)}, ("A", "B"), np.ones(2, dtype=np.int64), "two-dimensional"),
        ({"A": (1.0,), "B": (1.0,)}, ("A", "B"), np.zeros((1, 2), dtype=np.int64), "every replicate"),
    ),
)
def test_bootstrap_rejects_invalid_scene_values_and_multiplicities(
    values: Mapping[str, tuple[float, ...]],
    scene_ids: tuple[str, ...],
    multiplicities: np.ndarray,
    match: str,
) -> None:
    """Breaks if malformed bootstrap inputs silently alter the estimand."""
    with pytest.raises(ValueError, match=match):
        _bootstrap_episode_macro(values, scene_ids, multiplicities)


@pytest.mark.parametrize("case", ("duplicate", "incomplete"))
def test_inference_rejects_duplicate_or_incomplete_pivot_triples(case: str) -> None:
    """Breaks if inference aggregates duplicated or missing episode pivot rows."""
    rows = inference_rows()
    malformed = rows + (rows[0],) if case == "duplicate" else rows[:-1]

    with pytest.raises(ValueError, match="duplicate|complete pivot triple"):
        bootstrap_results(malformed)


def estimate(*, mean: float, lower: float) -> EndpointEstimate:
    """Builds a finite endpoint estimate with literal bounds for gate tests."""
    return EndpointEstimate(
        mean=mean,
        ci_lower=lower,
        ci_upper=lower + 0.1,
        leave_one_scene_out_min=mean,
        leave_one_scene_out_max=mean,
    )


@pytest.mark.parametrize(
    (
        "mean",
        "lower",
        "object_delta",
        "region_delta",
        "center_lower",
        "shuffle_lower",
        "expected",
    ),
    (
        (0.011, 0.001, 0.002, 0.020, 0.001, 0.001, GateDecision.GO),
        (
            0.011,
            0.001,
            -0.001,
            0.023,
            0.001,
            0.001,
            GateDecision.PARTIAL_EVIDENCE,
        ),
        (0.009, 0.001, 0.002, 0.016, 0.001, 0.001, GateDecision.NO_GO),
        (0.011, 0.000, 0.002, 0.020, 0.001, 0.001, GateDecision.NO_GO),
    ),
)
def test_gate_is_literal(
    mean: float,
    lower: float,
    object_delta: float,
    region_delta: float,
    center_lower: float,
    shuffle_lower: float,
    expected: GateDecision,
) -> None:
    """Breaks if any frozen gate threshold or decision ordering changes."""
    symmetric = estimate(mean=mean, lower=lower)
    center = estimate(mean=0.0, lower=center_lower)
    shuffle = estimate(mean=0.0, lower=shuffle_lower)

    assert classify_gate(
        true_start_symmetric=symmetric,
        true_start_object_to_region_mean=object_delta,
        true_start_region_to_object_mean=region_delta,
        start_minus_center=center,
        start_minus_shuffled=shuffle,
    ) is expected


def fixed_args(**overrides: object) -> CrossFitArgs:
    """Build the frozen loader arguments, allowing one deliberate mutation."""
    args = CrossFitArgs()
    for field, value in overrides.items():
        setattr(args, field, value)
    return args


def artifact_cases() -> tuple[EpisodeCase, ...]:
    """Small complete population spanning three bootstrap scenes."""
    predicted = np.zeros((37, 50, 50), dtype=np.bool_)
    target = np.zeros_like(predicted)
    predicted[0, 5, 6] = True
    predicted[27, 8, 9] = True
    target[0, 5, 6] = True
    target[27, 8, 9] = True
    return tuple(
        episode_case(
            f"episode-{scene_index}-{episode_index}",
            (float(episode_index + 1), float(scene_index + episode_index + 1)),
            scene_id=f"scene-{scene_index}",
            predicted_grid=predicted,
            target_grid=target,
        )
        for scene_index in range(3)
        for episode_index in range(2)
    )


def half_iou_artifact_cases() -> tuple[EpisodeCase, ...]:
    cases = []
    for case in artifact_cases():
        target_grid = case.target_grid.copy()
        target_grid[0, 10, 10] = True
        cases.append(replace(case, target_grid=target_grid))
    return tuple(cases)


def write_artifact_fixture(
    tmp_path: Path,
    cases: Optional[tuple[EpisodeCase, ...]] = None,
) -> tuple[Path, Path, tuple[EpisodeCase, ...]]:
    fixture_cases = artifact_cases() if cases is None else cases
    source_manifest = tmp_path / "prediction-manifest.json"
    source_manifest.write_text('{"source": "synthetic"}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    crossfit._run_cases(
        fixed_args(output_dir=output_dir),
        fixture_cases,
        expected_population=len(fixture_cases),
        expected_scenes=3,
        expected_valid=len(fixture_cases),
        expected_invalid=0,
        prediction_manifest_path=source_manifest,
    )
    return output_dir, source_manifest, fixture_cases


def artifact_bundle_from_directory(output_dir: Path) -> crossfit._ArtifactBundle:
    return crossfit._ArtifactBundle(
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


class StatefulArtifactBundle(crossfit._ArtifactBundle):
    files_calls = 0

    def files(self) -> Dict[str, bytes]:
        type(self).files_calls += 1
        if type(self).files_calls == 1:
            return super().files()
        return {"malicious.txt": b"unvalidated\n"}


def refresh_artifact_hash(output_dir: Path, filename: str) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest: Dict[str, object] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    artifact_hashes = checked_json_object(manifest["artifact_sha256"])
    artifact_hashes[filename] = sha256((output_dir / filename).read_bytes()).hexdigest()
    manifest["artifact_sha256"] = artifact_hashes
    if filename == "pivot_assignments.csv":
        manifest["pivot_assignments_sha256"] = artifact_hashes[filename]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def mutate_csv_field(
    output_dir: Path,
    filename: str,
    *,
    row_index: int,
    field: str,
    value: str,
) -> None:
    path = output_dir / filename
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        assert fieldnames is not None
        rows: list[Dict[str, str]] = []
        for row in reader:
            checked_row: Dict[str, str] = {}
            for key, cell in row.items():
                assert key is not None
                assert cell is not None
                checked_row[key] = cell
            rows.append(checked_row)
    rows[row_index][field] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    refresh_artifact_hash(output_dir, filename)


def checked_json_object(value: object) -> Dict[str, object]:
    assert isinstance(value, dict)
    result: Dict[str, object] = {}
    for key, item in value.items():
        assert isinstance(key, str)
        result[key] = item
    return result


def write_json_artifact(
    output_dir: Path,
    filename: str,
    payload: Mapping[str, object],
) -> None:
    (output_dir / filename).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refresh_artifact_hash(output_dir, filename)


@pytest.mark.parametrize("family", ["object", "region"])
def test_validator_rejects_refreshed_hash_iou_corruption(
    tmp_path: Path,
    family: str,
) -> None:
    output_dir, _, cases = write_artifact_fixture(tmp_path)
    mutate_csv_field(
        output_dir,
        "angle_scores.csv",
        row_index=0,
        field=f"{family}_iou",
        value="0.25",
    )

    with pytest.raises(ValueError, match=rf"{family}_iou"):
        crossfit.validate_artifact_directory(
            output_dir,
            expected_population=len(cases),
            expected_scenes=3,
            expected_valid=len(cases),
            expected_invalid=0,
        )


def test_validator_rejects_sub_tolerance_iou_corruption(
    tmp_path: Path,
) -> None:
    output_dir, _, cases = write_artifact_fixture(
        tmp_path,
        half_iou_artifact_cases(),
    )
    mutate_csv_field(
        output_dir,
        "angle_scores.csv",
        row_index=0,
        field="object_iou",
        value="0.5000000000005",
    )

    with pytest.raises(ValueError, match="object_iou"):
        crossfit.validate_artifact_directory(
            output_dir,
            expected_population=len(cases),
            expected_scenes=3,
            expected_valid=len(cases),
            expected_invalid=0,
        )


def test_artifact_writer_rejects_invalid_raw_bundle_before_output(
    tmp_path: Path,
) -> None:
    output_dir, _, _ = write_artifact_fixture(tmp_path)
    raw = artifact_bundle_from_directory(output_dir)
    invalid = replace(
        raw,
        angle_scores_csv=raw.angle_scores_csv + b"corrupt\n",
    )
    invalid_dir = tmp_path / "invalid"

    with pytest.raises(ValueError, match="angle_scores.csv.*SHA-256"):
        crossfit._write_artifacts(invalid_dir, invalid)
    assert not invalid_dir.exists()


def test_artifact_writer_writes_valid_raw_bundle(tmp_path: Path) -> None:
    output_dir, _, _ = write_artifact_fixture(tmp_path)
    raw = artifact_bundle_from_directory(output_dir)
    written_dir = tmp_path / "written"

    crossfit._write_artifacts(written_dir, raw)

    for filename, data in raw.files().items():
        assert (written_dir / filename).read_bytes() == data


def test_artifact_writer_captures_stateful_bundle_bytes_once(
    tmp_path: Path,
) -> None:
    output_dir, _, _ = write_artifact_fixture(tmp_path)
    raw = artifact_bundle_from_directory(output_dir)
    StatefulArtifactBundle.files_calls = 0
    stateful = StatefulArtifactBundle(
        manifest_json=raw.manifest_json,
        angle_scores_csv=raw.angle_scores_csv,
        crossfit_results_csv=raw.crossfit_results_csv,
        pivot_assignments_csv=raw.pivot_assignments_csv,
        summary_json=raw.summary_json,
        bootstrap_json=raw.bootstrap_json,
        control_intervals_png=raw.control_intervals_png,
    )
    written_dir = tmp_path / "stateful"

    crossfit._write_artifacts(written_dir, stateful)

    assert StatefulArtifactBundle.files_calls <= 1
    for filename, data in raw.files().items():
        assert (written_dir / filename).read_bytes() == data
    assert not (written_dir / "malicious.txt").exists()


@pytest.mark.parametrize(
    ("filename", "row_index", "field", "value"),
    (
        ("angle_scores.csv", 1, "split", "corrupted_split"),
        ("crossfit_results.csv", 1, "schema_valid", "False"),
    ),
)
def test_validator_checks_metadata_on_every_serialized_row(
    tmp_path: Path,
    filename: str,
    row_index: int,
    field: str,
    value: str,
) -> None:
    output_dir, _, cases = write_artifact_fixture(tmp_path)
    mutate_csv_field(
        output_dir,
        filename,
        row_index=row_index,
        field=field,
        value=value,
    )

    with pytest.raises(ValueError, match="metadata"):
        crossfit.validate_artifact_directory(
            output_dir,
            expected_population=len(cases),
            expected_scenes=3,
            expected_valid=len(cases),
            expected_invalid=0,
        )


def test_artifact_runner_writes_exact_output_set(tmp_path: Path) -> None:
    """Breaks if the validated transaction omits or adds an output artifact."""
    cases = artifact_cases()
    source_manifest = tmp_path / "prediction-manifest.json"
    source_manifest.write_text('{"source": "synthetic"}\n', encoding="utf-8")
    output_dir = tmp_path / "output"

    result = crossfit._run_cases(
        fixed_args(output_dir=output_dir),
        cases,
        expected_population=len(cases),
        expected_scenes=3,
        expected_valid=len(cases),
        expected_invalid=0,
        prediction_manifest_path=source_manifest,
    )

    assert result["output_dir"] == output_dir
    assert {path.name for path in output_dir.iterdir()} == {
        "manifest.json",
        "angle_scores.csv",
        "crossfit_results.csv",
        "pivot_assignments.csv",
        "summary.json",
        "bootstrap.json",
        "crossfit_control_intervals.png",
    }


def test_artifact_csvs_have_complete_literal_rows(tmp_path: Path) -> None:
    """Breaks if a score, support, identity, or control column is not exported."""
    cases = artifact_cases()
    source_manifest = tmp_path / "prediction-manifest.json"
    source_manifest.write_text('{"source": "synthetic"}\n', encoding="utf-8")
    output_dir = tmp_path / "output"

    crossfit._run_cases(
        fixed_args(output_dir=output_dir),
        cases,
        expected_population=len(cases),
        expected_scenes=3,
        expected_valid=len(cases),
        expected_invalid=0,
        prediction_manifest_path=source_manifest,
    )

    with (output_dir / "angle_scores.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        angle_rows = tuple(csv.DictReader(stream))
    assert len(angle_rows) == len(cases) * 3 * 4
    assert tuple(angle_rows[0]) == (
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

    with (output_dir / "crossfit_results.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        result_rows = tuple(csv.DictReader(stream))
    assert len(result_rows) == len(cases) * 3 * 2
    assert tuple(result_rows[0]) == (
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

    with (output_dir / "pivot_assignments.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        assignment_rows = tuple(csv.DictReader(stream))
    assert len(assignment_rows) == len(cases)
    assert tuple(assignment_rows[0]) == (
        "scene_id",
        "example_id",
        "donor_example_id",
        "true_pivot_row",
        "true_pivot_column",
        "assigned_pivot_row",
        "assigned_pivot_column",
    )


def test_artifact_json_is_complete_hashed_and_byte_stable(tmp_path: Path) -> None:
    """Breaks if provenance, inference, or deterministic serialization regresses."""
    cases = artifact_cases()
    source_manifest = tmp_path / "prediction-manifest.json"
    source_manifest.write_text('{"source": "synthetic"}\n', encoding="utf-8")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    for output_dir in (first_dir, second_dir):
        crossfit._run_cases(
            fixed_args(output_dir=output_dir),
            tuple(reversed(cases)),
            expected_population=len(cases),
            expected_scenes=3,
            expected_valid=len(cases),
            expected_invalid=0,
            prediction_manifest_path=source_manifest,
        )

    deterministic_names = (
        "manifest.json",
        "angle_scores.csv",
        "crossfit_results.csv",
        "pivot_assignments.csv",
        "summary.json",
        "bootstrap.json",
    )
    for filename in deterministic_names:
        assert (first_dir / filename).read_bytes() == (
            second_dir / filename
        ).read_bytes()
    for filename in ("manifest.json", "summary.json", "bootstrap.json"):
        text = (first_dir / filename).read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert text == json.dumps(
            json.loads(text), sort_keys=True, indent=2
        ) + "\n"

    manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prediction_manifest_path"] == str(source_manifest)
    assert manifest["prediction_manifest_sha256"] == sha256(
        source_manifest.read_bytes()
    ).hexdigest()
    assert manifest["pivot_assignments_sha256"] == sha256(
        (first_dir / "pivot_assignments.csv").read_bytes()
    ).hexdigest()
    assert manifest["population"] == {
        "episodes": 6,
        "scenes": 3,
        "schema_invalid": 0,
        "schema_valid": 6,
    }
    assert manifest["channel_folds"] == {
        "object": {"start_inclusive": 0, "stop_exclusive": 27},
        "region": {"start_inclusive": 27, "stop_exclusive": 37},
    }
    assert manifest["oracle_only_limitation"] == (
        "Angle selection uses ground-truth target raster cells and is not "
        "deployable performance."
    )
    assert set(manifest["artifact_sha256"]) == {
        "angle_scores.csv",
        "crossfit_results.csv",
        "pivot_assignments.csv",
        "summary.json",
        "bootstrap.json",
        "crossfit_control_intervals.png",
    }
    assert manifest["artifact_sha256"]["crossfit_control_intervals.png"] == sha256(
        (first_dir / "crossfit_control_intervals.png").read_bytes()
    ).hexdigest()

    summary = json.loads((first_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["population"] == manifest["population"]
    assert set(summary["pivots"]) == {mode.value for mode in PivotMode}
    assert set(summary["leave_one_scene_out_ranges"]) == {
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
    }
    assert set(summary["start_specific_contrasts"]) == {
        "true_start_minus_map_center",
        "true_start_minus_shuffled",
    }
    assert set(summary["gate"]["conditions"]) == {
        "true_start_symmetric_mean_at_least_0_01",
        "true_start_symmetric_ci_lower_above_zero",
        "both_true_start_directional_means_above_zero",
        "true_start_minus_map_center_ci_lower_above_zero",
        "true_start_minus_shuffled_ci_lower_above_zero",
    }
    assert summary["gate"]["decision"] == GateDecision.NO_GO.value

    bootstrap = json.loads(
        (first_dir / "bootstrap.json").read_text(encoding="utf-8")
    )
    assert bootstrap["contract"] == {
        "clusters": 3,
        "interval": "percentile",
        "pairing": "shared_scene_multiplicity_matrix",
        "percentiles": [2.5, 97.5],
        "repetitions": 10_000,
        "sampling_unit": "scene",
        "seed": 42,
        "weighting": "episode_macro",
    }
    assert set(bootstrap["endpoints"]) == set(
        summary["leave_one_scene_out_ranges"]
    )


def test_artifact_directory_validation_detects_changed_csv_byte(
    tmp_path: Path,
) -> None:
    """Breaks if post-write validation cannot detect a corrupted score table."""
    cases = artifact_cases()
    source_manifest = tmp_path / "prediction-manifest.json"
    source_manifest.write_text('{"source": "synthetic"}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    crossfit._run_cases(
        fixed_args(output_dir=output_dir),
        cases,
        expected_population=len(cases),
        expected_scenes=3,
        expected_valid=len(cases),
        expected_invalid=0,
        prediction_manifest_path=source_manifest,
    )

    crossfit.validate_artifact_directory(
        output_dir,
        expected_population=len(cases),
        expected_scenes=3,
        expected_valid=len(cases),
        expected_invalid=0,
    )
    angle_path = output_dir / "angle_scores.csv"
    angle_path.write_bytes(angle_path.read_bytes().replace(b"val_unseen", b"val_CHANGED", 1))
    with pytest.raises(ValueError, match="angle_scores.csv.*SHA-256"):
        crossfit.validate_artifact_directory(
            output_dir,
            expected_population=len(cases),
            expected_scenes=3,
            expected_valid=len(cases),
            expected_invalid=0,
        )


def test_artifact_directory_validation_detects_changed_png_tail(
    tmp_path: Path,
) -> None:
    output_dir, _, cases = write_artifact_fixture(tmp_path)
    plot_path = output_dir / "crossfit_control_intervals.png"
    plot_path.write_bytes(plot_path.read_bytes() + b"corrupt-tail")

    with pytest.raises(
        ValueError,
        match="crossfit_control_intervals.png.*SHA-256",
    ):
        crossfit.validate_artifact_directory(
            output_dir,
            expected_population=len(cases),
            expected_scenes=3,
            expected_valid=len(cases),
            expected_invalid=0,
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("duplicate_identity", "identities"),
        ("assignment_donor", "pivot assignments"),
        ("support_arithmetic", "support"),
        ("nonfinite_metric", "finite"),
        ("undeclared_angle", "angle"),
        ("paired_contrast", "summary.json"),
        ("bootstrap_interval", "bootstrap.json"),
        ("gate_input", "bootstrap.json"),
        ("gate_boolean", "summary.json"),
        ("gate_decision", "gate decision"),
        ("fixed_provenance", "provenance"),
        ("source_sha", "prediction source"),
        ("row_count", "row invariant"),
        ("header", "header invariant"),
    ),
)
def test_validator_rejects_semantic_corruption_after_hash_refresh(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    output_dir, _, cases = write_artifact_fixture(tmp_path)
    if corruption == "duplicate_identity":
        mutate_csv_field(
            output_dir,
            "pivot_assignments.csv",
            row_index=2,
            field="scene_id",
            value="scene-0",
        )
        mutate_csv_field(
            output_dir,
            "pivot_assignments.csv",
            row_index=2,
            field="example_id",
            value="episode-0-0",
        )
    elif corruption == "assignment_donor":
        mutate_csv_field(
            output_dir,
            "pivot_assignments.csv",
            row_index=0,
            field="donor_example_id",
            value="not-an-episode",
        )
    elif corruption == "support_arithmetic":
        mutate_csv_field(
            output_dir,
            "angle_scores.csv",
            row_index=0,
            field="object_input_support",
            value="2",
        )
    elif corruption == "nonfinite_metric":
        mutate_csv_field(
            output_dir,
            "crossfit_results.csv",
            row_index=0,
            field="delta_iou",
            value="nan",
        )
    elif corruption == "undeclared_angle":
        mutate_csv_field(
            output_dir,
            "angle_scores.csv",
            row_index=0,
            field="angle_degrees",
            value="45",
        )
    elif corruption in {"paired_contrast", "gate_boolean"}:
        summary_path = output_dir / "summary.json"
        summary = checked_json_object(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        if corruption == "paired_contrast":
            contrasts = checked_json_object(summary["start_specific_contrasts"])
            center = checked_json_object(contrasts["true_start_minus_map_center"])
            center["mean_delta_iou"] = 0.5
            contrasts["true_start_minus_map_center"] = center
            summary["start_specific_contrasts"] = contrasts
        else:
            gate = checked_json_object(summary["gate"])
            conditions = checked_json_object(gate["conditions"])
            conditions["true_start_symmetric_ci_lower_above_zero"] = True
            gate["conditions"] = conditions
            summary["gate"] = gate
        write_json_artifact(output_dir, "summary.json", summary)
    elif corruption in {"bootstrap_interval", "gate_input"}:
        bootstrap_path = output_dir / "bootstrap.json"
        bootstrap = checked_json_object(
            json.loads(bootstrap_path.read_text(encoding="utf-8"))
        )
        endpoints = checked_json_object(bootstrap["endpoints"])
        endpoint = checked_json_object(endpoints["true_start_symmetric"])
        if corruption == "bootstrap_interval":
            endpoint["ci_lower"] = -99.0
        else:
            endpoint["mean"] = 0.5
        endpoints["true_start_symmetric"] = endpoint
        bootstrap["endpoints"] = endpoints
        write_json_artifact(output_dir, "bootstrap.json", bootstrap)
    elif corruption in {
        "gate_decision",
        "fixed_provenance",
        "source_sha",
    }:
        manifest_path = output_dir / "manifest.json"
        manifest = checked_json_object(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if corruption == "gate_decision":
            manifest["gate_decision"] = GateDecision.GO.value
        elif corruption == "fixed_provenance":
            manifest["cache_model_key"] = "corrupted-model-key"
        else:
            manifest["prediction_manifest_sha256"] = "0" * 64
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif corruption == "row_count":
        path = output_dir / "crossfit_results.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        refresh_artifact_hash(output_dir, "crossfit_results.csv")
    elif corruption == "header":
        path = output_dir / "angle_scores.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("split", "bad_split", 1), encoding="utf-8")
        refresh_artifact_hash(output_dir, "angle_scores.csv")
    else:
        raise AssertionError(f"unhandled corruption fixture: {corruption}")

    with pytest.raises(ValueError, match=message):
        crossfit.validate_artifact_directory(
            output_dir,
            expected_population=len(cases),
            expected_scenes=3,
            expected_valid=len(cases),
            expected_invalid=0,
        )


def test_malformed_analysis_is_rejected_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = artifact_cases()
    source_manifest = tmp_path / "prediction-manifest.json"
    source_manifest.write_text('{"source": "synthetic"}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    original_evaluate = crossfit.evaluate_episode
    call_count = 0

    def inconsistent_evaluate(
        episode: EpisodeCase,
        assignment: PivotAssignment,
        angles: tuple[float, ...],
    ) -> tuple[EpisodePivotResult, ...]:
        nonlocal call_count
        call_count += 1
        results = original_evaluate(episode, assignment, angles)
        if call_count <= len(cases):
            return (
                replace(results[0], schema_valid=not results[0].schema_valid),
                *results[1:],
            )
        return results

    monkeypatch.setattr(crossfit, "evaluate_episode", inconsistent_evaluate)

    with pytest.raises(ValueError, match="evaluation from typed cases"):
        crossfit._run_cases(
            fixed_args(output_dir=output_dir),
            cases,
            expected_population=len(cases),
            expected_scenes=3,
            expected_valid=len(cases),
            expected_invalid=0,
            prediction_manifest_path=source_manifest,
        )
    assert not output_dir.exists()


def test_output_transaction_rejects_nonempty_directory_untouched(
    tmp_path: Path,
) -> None:
    """Breaks if a run can overwrite or mix with an existing artifact set."""
    cases = artifact_cases()
    source_manifest = tmp_path / "prediction-manifest.json"
    source_manifest.write_text('{"source": "synthetic"}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    marker = output_dir / "existing.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(ValueError, match="nonempty"):
        crossfit._run_cases(
            fixed_args(output_dir=output_dir),
            cases,
            expected_population=len(cases),
            expected_scenes=3,
            expected_valid=len(cases),
            expected_invalid=0,
            prediction_manifest_path=source_manifest,
        )

    assert tuple(output_dir.iterdir()) == (marker,)
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_analysis_validation_rejects_undeclared_angle() -> None:
    """Breaks if complete row counts can hide an altered angle membership."""
    cases = artifact_cases()
    assignments = build_pivot_assignments(cases)
    by_identity = {
        (assignment.scene_id, assignment.example_id): assignment
        for assignment in assignments
    }
    results = tuple(
        result
        for case in cases
        for result in evaluate_episode(
            case,
            by_identity[(case.scene_id, case.example_id)],
            (0.0, 90.0, 180.0, 270.0),
        )
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
    first = results[0]
    changed_scores = (
        replace(first.angle_scores[0], angle_degrees=45.0),
    ) + first.angle_scores[1:]
    malformed = (replace(first, angle_scores=changed_scores),) + results[1:]

    with pytest.raises(ValueError, match="angle membership"):
        crossfit._validate_analysis(
            cases=cases,
            assignments=assignments,
            results=malformed,
            summaries=summaries,
            ranges=ranges,
            estimates=estimates,
            gate_conditions=crossfit._gate_conditions(estimates),
            decision=decision,
            angle_rows=crossfit._angle_score_rows(malformed),
            crossfit_rows=crossfit._crossfit_result_rows(malformed),
            pivot_assignment_rows=crossfit._pivot_assignment_rows(assignments),
            summary_payload=crossfit._summary_payload(
                population={
                    "episodes": len(cases),
                    "scenes": 3,
                    "schema_valid": len(cases),
                    "schema_invalid": 0,
                },
                summaries=summaries,
                ranges=ranges,
                estimates=estimates,
                gate_conditions=crossfit._gate_conditions(estimates),
                decision=decision,
            ),
            bootstrap_payload=crossfit._bootstrap_payload(estimates, 3),
            expected_population=len(cases),
            expected_scenes=3,
            expected_valid=len(cases),
            expected_invalid=0,
            angles=(0.0, 90.0, 180.0, 270.0),
        )


def test_runner_uses_only_the_fixed_full_population_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Breaks if the public runner permits a partial or altered population."""
    args = fixed_args(output_dir=tmp_path / "output")
    manifest_path = tmp_path / "prediction-manifest.json"
    observed: Dict[str, object] = {}
    loaded_cases = artifact_cases()

    monkeypatch.setattr(
        crossfit,
        "_load_episode_cases",
        lambda received_args: (loaded_cases, manifest_path),
    )

    def fake_run_cases(
        received_args: CrossFitArgs,
        cases: tuple[EpisodeCase, ...],
        *,
        expected_population: int,
        expected_scenes: int,
        expected_valid: int,
        expected_invalid: int,
        prediction_manifest_path: Path,
    ) -> dict[str, object]:
        observed.update(
            {
                "args": received_args,
                "cases": cases,
                "contract": (
                    expected_population,
                    expected_scenes,
                    expected_valid,
                    expected_invalid,
                ),
                "manifest": prediction_manifest_path,
            }
        )
        return {"output_dir": received_args.output_dir}

    monkeypatch.setattr(crossfit, "_run_cases", fake_run_cases)

    assert crossfit.run_crossfit(args) == {"output_dir": args.output_dir}
    assert observed == {
        "args": args,
        "cases": loaded_cases,
        "contract": (1_839, 11, 1_830, 9),
        "manifest": manifest_path,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("cache_model_key", "other"),
        ("cognitive_map_namespace", "other.namespace"),
        ("angles", (0.0, 90.0)),
        ("shuffle_seed", 44),
        ("bootstrap_repetitions", 9999),
        ("bootstrap_seed", 41),
    ),
)
def test_args_reject_scientific_overrides(field: str, value: object) -> None:
    """Breaks if a caller can alter a fixed cross-fit scientific control."""
    with pytest.raises(ValueError):
        _validate_args(fixed_args(**{field: value}))


def test_args_cli_uses_dashed_names_and_has_no_partial_population_control() -> None:
    """Breaks if the public CLI reintroduces underlined or partial-run flags."""
    parsed = CrossFitArgs(underscores_to_dashes=True).parse_args(
        ["--cache-dir", "cache", "--output-dir", "output", "--quiet"]
    )

    assert parsed.cache_dir == Path("cache")
    assert parsed.output_dir == Path("output")
    assert parsed.quiet
    assert "limit" not in parsed.class_variables
    assert "partial_population" not in parsed.class_variables

    with pytest.raises(SystemExit):
        CrossFitArgs(underscores_to_dashes=True).parse_args(["--limit", "1"])


@pytest.mark.parametrize(
    ("cache_dir", "output_dir"),
    (
        (Path("same"), Path("same")),
        (Path("cache"), Path("cache/output")),
        (Path("output/cache"), Path("output")),
    ),
)
def test_args_reject_overlapping_cache_and_output_paths(
    cache_dir: Path, output_dir: Path
) -> None:
    """Breaks if reads and derived artifacts can share a directory tree."""
    with pytest.raises(ValueError, match="must not overlap"):
        _validate_args(fixed_args(cache_dir=cache_dir, output_dir=output_dir))


def test_prediction_manifest_requires_frozen_r2r_val_unseen_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Breaks if manifest validation stops pinning the cache provenance."""
    observed: dict[str, object] = {}
    split_dir = tmp_path / "split"

    def fake_split_dir(
        dataset: str, split: str, *, cache_dir: Path, model_key: str
    ) -> Path:
        observed["split"] = (dataset, split, cache_dir, model_key)
        return split_dir

    def fake_validate(path: Path, expected: Mapping[str, object]) -> None:
        observed["manifest"] = (path, expected)

    monkeypatch.setattr(crossfit, "llm_navigation_split_dir", fake_split_dir)
    monkeypatch.setattr(crossfit, "validate_llm_navigation_manifest", fake_validate)
    args = fixed_args(cache_dir=tmp_path / "cache", output_dir=tmp_path / "output")

    assert _prediction_manifest_path(args) == split_dir / "manifest.json"
    assert observed["split"] == (
        "R2R",
        "val_unseen",
        args.cache_dir,
        args.cache_model_key,
    )
    assert observed["manifest"] == (
        split_dir,
        {
            "dataset": "R2R",
            "split": "val_unseen",
            "generator": "llm-grid",
            "scale": 2,
            "cache_model_key": args.cache_model_key,
        },
    )


_VALID_PREDICTION_TEXT = json.dumps(
    {
        "predicted_regions": [],
        "predicted_objects": [],
        "regions": {},
        "objects": {},
        "direction_vectors": [[1.0, 0.0]] * 5,
    }
)
_EXPECTED_INVALID_IDS = frozenset(
    f"episode-{index:04d}" for index in range(1_830, 1_839)
)


@pytest.fixture(scope="module")
def full_episode_cases() -> tuple[EpisodeCase, ...]:
    """One complete immutable synthetic R2R val_unseen population."""
    empty_grid = np.zeros((37, 50, 50), dtype=np.bool_)
    return tuple(
        EpisodeCase(
            split="val_unseen",
            scene_id=f"scene-{index % 11}",
            example_id=f"episode-{index:04d}",
            schema_valid=index < 1_830,
            predicted_grid=empty_grid,
            target_grid=empty_grid,
            true_start_pivot=(float(index), float(index + 1)),
        )
        for index in range(1_839)
    )


@pytest.fixture(scope="module")
def full_loader_inputs() -> tuple[
    ExampleLoadResult[LLMGridExample], tuple[Dict[str, object], ...]
]:
    """Complete external-shaped examples and dataset items for loader tests."""
    empty_target = np.zeros((37, 50, 50), dtype=np.float32)
    empty_directions = np.zeros((5, 2), dtype=np.float32)
    examples = tuple(
        LLMGridExample(
            example_id=f"episode-{index:04d}",
            dataset="R2R",
            split="val_unseen",
            scene_id=f"scene-{index % 11}",
            episode_id=index,
            instruction="walk forward",
            raster_path=Path(f"/external/raster-{index:04d}.npz"),
        )
        for index in range(1_839)
    )
    items: list[Dict[str, object]] = []
    for example in examples:
        items.append(
            {
            "input_text": "input",
            "target_text": "target",
            "target_grid": empty_target,
            "target_direction_vectors": empty_directions,
            "example_id": example.example_id,
            "instruction": example.instruction,
            "start_position": (1.0, 2.0),
            "start_direction": (1.0, 0.0),
            "scene_id": example.scene_id,
            "dataset": "R2R",
            "split": "val_unseen",
            }
        )
    return (
        ExampleLoadResult(
            examples=examples,
            by_dataset={
                "R2R": SourceLoadStats(1_839, 1_839, ()),
                "RxR": SourceLoadStats(0, 0, ()),
            },
        ),
        tuple(items),
    )


def install_full_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loaded: ExampleLoadResult[LLMGridExample],
    items: tuple[Dict[str, object], ...],
    prediction_text: Callable[[Path], str],
) -> CrossFitArgs:
    """Keep manifest and population validation real while replacing external reads."""
    args = fixed_args(cache_dir=tmp_path / "cache", output_dir=tmp_path / "output")
    split_dir = crossfit.llm_navigation_split_dir(
        "R2R", "val_unseen", cache_dir=args.cache_dir, model_key=args.cache_model_key
    )
    split_dir.mkdir(parents=True)
    (split_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "R2R",
                "split": "val_unseen",
                "generator": "llm-grid",
                "scale": 2,
                "cache_model_key": args.cache_model_key,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(crossfit, "load_llm_grid_examples", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(crossfit, "LLMGridDataset", lambda examples, scale: items)
    original_read_text = Path.read_text

    def read_cache_text(
        path: Path, encoding: Optional[str] = None, errors: Optional[str] = None
    ) -> str:
        if path.suffix == ".txt":
            return prediction_text(path)
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_cache_text)
    return args


def test_loader_uses_real_manifest_and_population_validation_with_scaled_pivots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    full_loader_inputs: tuple[
        ExampleLoadResult[LLMGridExample], tuple[Dict[str, object], ...]
    ],
) -> None:
    """Breaks if loader bypasses the manifest/population path or skips scaling."""
    loaded, items = full_loader_inputs
    monkeypatch.setattr(crossfit, "CELL_SIZE", 0.25)
    args = install_full_loader(
        monkeypatch,
        tmp_path,
        loaded,
        items,
        lambda path: "{invalid JSON"
        if path.stem in _EXPECTED_INVALID_IDS
        else _VALID_PREDICTION_TEXT,
    )

    cases, manifest_path = _load_episode_cases(args)

    assert manifest_path.is_file()
    assert len(cases) == 1_839
    assert len({case.scene_id for case in cases}) == 11
    assert sum(case.schema_valid for case in cases) == 1_830
    assert sum(not case.schema_valid for case in cases) == 9
    assert cases[0].true_start_pivot == (2.0, 4.0)


@pytest.mark.parametrize("invalid_cache", ("missing", "invalid"))
def test_loader_retains_nine_expected_invalid_predictions_in_real_population(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    full_loader_inputs: tuple[
        ExampleLoadResult[LLMGridExample], tuple[Dict[str, object], ...]
    ],
    invalid_cache: str,
) -> None:
    """Breaks if expected cache failures leave the full evaluation denominator."""
    loaded, items = full_loader_inputs

    def prediction_text(path: Path) -> str:
        if path.stem in _EXPECTED_INVALID_IDS:
            if invalid_cache == "missing":
                raise FileNotFoundError(path)
            return "{invalid JSON"
        return _VALID_PREDICTION_TEXT

    cases, _ = _load_episode_cases(
        install_full_loader(monkeypatch, tmp_path, loaded, items, prediction_text)
    )

    assert sum(not case.schema_valid for case in cases) == 9
    assert all(not np.any(case.predicted_grid) for case in cases if not case.schema_valid)


@pytest.mark.parametrize("failure", (OSError("disk"), RuntimeError("parser")))
def test_loader_propagates_unexpected_cache_failures_from_complete_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    full_loader_inputs: tuple[
        ExampleLoadResult[LLMGridExample], tuple[Dict[str, object], ...]
    ],
    failure: Exception,
) -> None:
    """Breaks if unexpected cache faults are reclassified as invalid predictions."""
    loaded, items = full_loader_inputs

    def prediction_text(path: Path) -> str:
        raise failure

    with pytest.raises(type(failure), match=str(failure)):
        _load_episode_cases(
            install_full_loader(monkeypatch, tmp_path, loaded, items, prediction_text)
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("order", "dataset order"),
        ("target_shape", "target_grid must have shape"),
        ("prediction_shape", "predicted_grid must have shape"),
        ("nonfinite_start", "start_position coordinates must be finite"),
        ("duplicate_identity", "unique scene/example identities"),
    ),
)
def test_loader_reaches_each_complete_external_input_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    full_loader_inputs: tuple[
        ExampleLoadResult[LLMGridExample], tuple[Dict[str, object], ...]
    ],
    mutation: str,
    match: str,
) -> None:
    """Breaks if a malformed complete input reaches scoring or a wrong branch."""
    loaded, original_items = full_loader_inputs
    examples = loaded.examples
    items = original_items
    if mutation == "order":
        changed = original_items[0].copy()
        changed["example_id"] = "different"
        items = (changed,) + original_items[1:]
    elif mutation == "target_shape":
        changed = original_items[0].copy()
        changed["target_grid"] = np.zeros((37, 49, 50), dtype=np.float32)
        items = (changed,) + original_items[1:]
    elif mutation == "prediction_shape":
        monkeypatch.setattr(crossfit, "GRID_SHAPE", (37, 49, 50))
    elif mutation == "nonfinite_start":
        changed = original_items[0].copy()
        changed["start_position"] = (np.inf, 2.0)
        items = (changed,) + original_items[1:]
    else:
        examples = (
            examples[0],
            replace(examples[1], scene_id=examples[0].scene_id, example_id=examples[0].example_id),
        ) + examples[2:]
        changed = original_items[1].copy()
        changed["scene_id"] = examples[0].scene_id
        changed["example_id"] = examples[0].example_id
        items = (original_items[0], changed) + original_items[2:]
    changed_loaded = ExampleLoadResult(examples=examples, by_dataset=loaded.by_dataset)

    with pytest.raises(ValueError, match=match):
        _load_episode_cases(
            install_full_loader(
                monkeypatch,
                tmp_path,
                changed_loaded,
                items,
                lambda path: _VALID_PREDICTION_TEXT,
            )
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("scene_count", "exactly 11 scenes"),
        ("status_count", "exactly 1830 valid rows"),
        ("duplicate_identity", "unique scene/example identities"),
        ("nonempty_invalid", "invalid predictions must use an empty grid"),
    ),
)
def test_population_validator_rejects_one_mutated_full_invariant(
    full_episode_cases: tuple[EpisodeCase, ...], mutation: str, match: str
) -> None:
    """Breaks if fixed population constraints are hidden behind row-count failure."""
    cases = full_episode_cases
    if mutation == "scene_count":
        cases = tuple(
            replace(case, scene_id="scene-0") if case.scene_id == "scene-10" else case
            for case in cases
        )
    elif mutation == "status_count":
        cases = (replace(cases[0], schema_valid=False),) + cases[1:]
    elif mutation == "duplicate_identity":
        cases = (
            cases[0],
            replace(cases[1], scene_id=cases[0].scene_id, example_id=cases[0].example_id),
        ) + cases[2:]
    else:
        cases = cases[:-1] + (
            replace(
                cases[-1], predicted_grid=np.ones((37, 50, 50), dtype=np.bool_)
            ),
        )

    with pytest.raises(ValueError, match=match):
        crossfit._validate_episode_population(cases)
