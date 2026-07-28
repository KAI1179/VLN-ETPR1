from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional

import numpy as np
from numpy.typing import NDArray
import pytest

from prior.analyze.d2026_07_28 import llm_grid_transform_crossfit as crossfit
from prior.analyze.d2026_07_28.llm_grid_transform_crossfit import (
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
