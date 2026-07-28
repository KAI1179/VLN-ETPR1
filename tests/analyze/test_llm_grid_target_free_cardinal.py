from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from prior.analyze.d2026_07_28 import llm_grid_target_free_cardinal as target_free
from prior.analyze.llm_grid_registration import (
    WarpedGrid,
    warp_grid_about_pivot,
)
from prior.llm_grid_samples import serialize_grid_target


def test_quantize_heading_uses_clockwise_display_frame_and_declared_ties() -> None:
    """Breaks if heading uses the wrong display axis or tie ordering."""
    assert target_free.quantize_heading(np.asarray([0.0, 1.0], np.float32)) == 0.0
    assert target_free.quantize_heading(np.asarray([1.0, 0.0], np.float32)) == 90.0
    boundary = np.asarray([2**-0.5, 2**-0.5], np.float32)
    assert target_free.quantize_heading(boundary) == 0.0


@pytest.mark.parametrize("scene_id, example_id", (("", "example"), ("scene", "")))
def test_episode_key_rejects_empty_csv_identity_components(
    scene_id: str, example_id: str
) -> None:
    """Breaks if canonical assignment CSV rows can contain empty identity cells."""
    with pytest.raises(ValueError, match="must be non-empty"):
        target_free.EpisodeKey(scene_id, example_id)


@pytest.mark.parametrize(
    "bad",
    ([0.0, 0.0], [float("nan"), 1.0], [0.5, 0.0]),
)
def test_selector_input_rejects_invalid_start_direction(bad: list[float]) -> None:
    """Breaks if corrupt runtime heading metadata reaches the selector."""
    with pytest.raises(ValueError, match="start_direction"):
        target_free.SelectorInput(
            schema_valid=True,
            start_direction=np.asarray(bad, np.float32),
            predicted_grid=np.zeros((37, 50, 50), dtype=np.bool_),
            predicted_directions=np.zeros((5, 2), dtype=np.float32),
        )


def valid_selector_input(
    *,
    schema_valid: bool = True,
    start_direction: NDArray[np.float32] | None = None,
    predicted_directions: NDArray[np.float32] | None = None,
) -> target_free.SelectorInput:
    """Build a valid selector input with only the tested runtime fields changed."""
    return target_free.SelectorInput(
        schema_valid=schema_valid,
        start_direction=(
            np.asarray([0.0, 1.0], dtype=np.float32)
            if start_direction is None
            else start_direction
        ),
        predicted_grid=np.zeros((37, 50, 50), dtype=np.bool_),
        predicted_directions=(
            np.zeros((5, 2), dtype=np.float32)
            if predicted_directions is None
            else predicted_directions
        ),
    )


@pytest.mark.parametrize(
    ("mapping", "expected"),
    (
        (target_free.HeadingMapping(+1, 0.0), 90.0),
        (target_free.HeadingMapping(+1, 90.0), 180.0),
        (target_free.HeadingMapping(+1, 180.0), 270.0),
        (target_free.HeadingMapping(+1, 270.0), 0.0),
        (target_free.HeadingMapping(-1, 0.0), 270.0),
        (target_free.HeadingMapping(-1, 90.0), 0.0),
        (target_free.HeadingMapping(-1, 180.0), 90.0),
        (target_free.HeadingMapping(-1, 270.0), 180.0),
    ),
)
def test_heading_mapping_emits_each_declared_cardinal_correction(
    mapping: target_free.HeadingMapping, expected: float
) -> None:
    """Breaks if any frozen sign-and-offset mapping changes its physical angle."""
    assert mapping.angle_for(np.asarray([1.0, 0.0], dtype=np.float32)) == expected


@pytest.mark.parametrize(
    ("sign", "offset"),
    ((0, 0.0), (+1, 45.0), (True, 0.0)),
)
def test_heading_mapping_rejects_non_declared_identity_behavior(
    sign: int, offset: float
) -> None:
    """Breaks if invalid mappings silently become an identity correction."""
    with pytest.raises(ValueError):
        target_free.HeadingMapping(sign, offset)


def test_direct_selector_empty_prediction_is_identity_with_zero_margin() -> None:
    """Breaks if an empty prediction uses a directional tie as a correction."""
    selector_input = valid_selector_input(
        schema_valid=False,
        predicted_directions=np.zeros((5, 2), np.float32),
    )
    assert target_free.direct_heading_assignment(selector_input) == (0.0, 0.0)


def test_direct_selector_invalid_prediction_is_identity_with_zero_margin() -> None:
    """Breaks if schema-invalid predictions participate in direction selection."""
    selector_input = valid_selector_input(
        schema_valid=False,
        predicted_directions=np.asarray(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    assert target_free.direct_heading_assignment(selector_input) == (0.0, 0.0)


def test_direct_selector_uses_positive_angle_vector_sign_and_margin() -> None:
    """Breaks if physical positive angles rotate stored vectors in the wrong sign."""
    selector_input = valid_selector_input(
        start_direction=np.asarray([0.0, 1.0], dtype=np.float32),
        predicted_directions=np.asarray(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    angle, margin = target_free.direct_heading_assignment(selector_input)
    assert angle == 270.0
    assert margin == pytest.approx(1.0)


def test_direct_selector_retains_first_declared_angle_on_cosine_tie() -> None:
    """Breaks if equal cosine candidates replace the first physical angle."""
    diagonal = np.asarray([2**-0.5, 2**-0.5], dtype=np.float32)
    selector_input = valid_selector_input(
        start_direction=diagonal,
        predicted_directions=np.asarray(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    angle, margin = target_free.direct_heading_assignment(selector_input)
    assert angle == 0.0
    assert margin == pytest.approx(0.0, abs=1e-6)


def single_corner_cell_grid() -> NDArray[np.bool_]:
    grid = np.zeros((37, 50, 50), dtype=np.bool_)
    grid[0, 0, 0] = True
    return grid


def cardinal_warps(
    grid: NDArray[np.bool_], pivot: tuple[float, float]
) -> tuple[WarpedGrid, ...]:
    return tuple(warp_grid_about_pivot(grid, pivot, angle) for angle in target_free.ANGLE_ORDER)


def empty_target() -> NDArray[np.bool_]:
    return np.zeros((37, 50, 50), dtype=np.bool_)


def test_uniform_soft_score_keeps_out_of_frame_false_positive_mass() -> None:
    """Breaks if the common canvas clips padded cells before soft scoring."""
    hypotheses = cardinal_warps(single_corner_cell_grid(), pivot=(1.0, 1.0))
    score = target_free.uniform_soft_score(hypotheses, empty_target())
    assert any(
        hypothesis.bounds.row_min < 0 or hypothesis.bounds.col_min < 0
        for hypothesis in hypotheses
    )
    assert score.predicted_mass > 0.0
    assert score.intersection_mass == 0.0
    assert score.iou == 0.0


def test_uniform_soft_score_calculates_exact_soft_statistics() -> None:
    """Breaks if averaging, mass accounting, or soft precision metrics are wrong."""
    target = empty_target()
    target[0, 10, 10] = True
    hit = target.copy()
    miss = empty_target()
    miss[0, 10, 11] = True
    hypotheses = tuple(
        WarpedGrid(grid, warp_grid_about_pivot(grid, (25.0, 25.0), 0.0).bounds, 1)
        for grid in (hit, miss, empty_target(), empty_target())
    )

    score = target_free.uniform_soft_score(hypotheses, target)

    assert score.intersection_mass == pytest.approx(0.25)
    assert score.union_mass == pytest.approx(1.25)
    assert score.predicted_mass == pytest.approx(0.5)
    assert score.target_mass == pytest.approx(1.0)
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.25)
    assert score.f1 == pytest.approx(1.0 / 3.0)
    assert score.iou == pytest.approx(0.2)


def test_uniform_soft_score_returns_zero_metrics_for_empty_union() -> None:
    """Breaks if empty soft rasters produce undefined or nonzero metrics."""
    empty = empty_target()
    hypotheses = tuple(
        WarpedGrid(
            empty.copy(), warp_grid_about_pivot(empty, (25.0, 25.0), 0.0).bounds, 0
        )
        for _ in target_free.ANGLE_ORDER
    )

    score = target_free.uniform_soft_score(hypotheses, empty)

    assert score == target_free.SoftRasterScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class _Example:
    example_id: str
    dataset: str
    split: str
    scene_id: str
    raster_path: Path


@dataclass(frozen=True)
class _LoadedExamples:
    examples: tuple[_Example, ...]


def _selector_grid() -> NDArray[np.bool_]:
    grid = np.zeros((37, 50, 50), dtype=np.bool_)
    grid[0, 10, 12] = True
    grid[27, 11, 12] = True
    return grid


def _runtime(
    scene_id: str,
    example_id: str,
    heading: NDArray[np.float32],
    *,
    schema_valid: bool = True,
) -> target_free.RuntimeEpisode:
    return target_free.RuntimeEpisode(
        key=target_free.EpisodeKey(scene_id, example_id),
        split="val_seen",
        selector_input=target_free.SelectorInput(
            schema_valid=schema_valid,
            start_direction=heading,
            predicted_grid=_selector_grid() if schema_valid else empty_target(),
            predicted_directions=(
                np.asarray(
                    [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    dtype=np.float32,
                )
                if schema_valid
                else np.zeros((5, 2), dtype=np.float32)
            ),
        ),
        start_pivot=(25.0, 25.0),
    )


def synthetic_runtime_population() -> tuple[target_free.RuntimeEpisode, ...]:
    return (
        _runtime("scene-z", "example-z", np.asarray([0.0, 1.0], np.float32)),
        _runtime("scene-a", "example-a", np.asarray([1.0, 0.0], np.float32)),
        _runtime(
            "scene-b",
            "invalid",
            np.asarray([0.0, -1.0], np.float32),
            schema_valid=False,
        ),
    )


def frozen_lock() -> target_free.SelectorLock:
    mapping = target_free.HeadingMapping(+1, 0.0)
    scores = tuple(
        target_free.DevelopmentCandidateScore(
            "mapping", candidate.sign, int(candidate.offset_degrees), -1,
            1, 1, 1.0, 1.0, 1.0, candidate == mapping,
        )
        for candidate in (
            target_free.HeadingMapping(sign, offset)
            for sign in (+1, -1)
            for offset in target_free.ANGLE_ORDER
        )
    ) + tuple(
        target_free.DevelopmentCandidateScore(
            "global", -1, -1, int(angle), 1, 1, 1.0, 1.0, 1.0, angle == 0.0
        )
        for angle in target_free.ANGLE_ORDER
    )
    return target_free.SelectorLock(mapping, 0.0, scores, 8, 4)


def fail_if_called(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("target access during assignment")


def test_population_contract_and_runtime_loader_seal_manifest_and_sort_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Breaks if runtime loading uses an unsealed or target-bearing population."""
    root = tmp_path / "cache"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text('{"split":"val_seen"}', encoding="utf-8")
    first_raster = tmp_path / "first.npz"
    second_raster = tmp_path / "second.npz"
    for raster in (first_raster, second_raster):
        np.savez_compressed(
            raster,
            grid=np.ones((37, 100, 100), dtype=np.float32),
            direction_vectors=np.ones((5, 2), dtype=np.float32),
            start_position=np.asarray([10.0, 20.0], dtype=np.float32),
            start_direction_vector=np.asarray([0.0, 1.0], dtype=np.float32),
        )
    examples = _LoadedExamples(
        (
            _Example("late", "R2R", "val_seen", "scene-z", second_raster),
            _Example("early", "R2R", "val_seen", "scene-a", first_raster),
        )
    )
    valid_text = serialize_grid_target(
        np.zeros((37, 50, 50), dtype=np.float32),
        direction_vectors=np.zeros((5, 2), dtype=np.float32),
    )
    paths = {"late": tmp_path / "late.txt", "early": tmp_path / "early.txt"}
    paths["late"].write_text("malformed", encoding="utf-8")
    paths["early"].write_text(valid_text, encoding="utf-8")
    monkeypatch.setattr(target_free, "load_llm_grid_examples", lambda *_args, **_kwargs: examples)
    monkeypatch.setattr(target_free, "llm_navigation_split_dir", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        target_free,
        "llm_navigation_prediction_path",
        lambda _scene, example, *_args, **_kwargs: paths[example],
    )
    contract = target_free.PopulationContract(
        "val_seen", 2, 2, 1, 1, sha256(manifest.read_bytes()).hexdigest()
    )

    population = target_free.load_runtime_population(
        "val_seen", contract, tmp_path, "model", "namespace", quiet=True
    )

    assert [episode.key for episode in population] == [
        target_free.EpisodeKey("scene-a", "early"),
        target_free.EpisodeKey("scene-z", "late"),
    ]
    assert population[0].selector_input.schema_valid is True
    assert population[1].selector_input.schema_valid is False
    assert not population[1].selector_input.predicted_grid.any()
    assert population[0].start_pivot == pytest.approx((10.0, 20.0))


def test_runtime_loader_rejects_corrupt_start_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Breaks if corrupt runtime anchors are mistaken for bad predictions."""
    raster = tmp_path / "corrupt.npz"
    np.savez_compressed(
        raster,
        start_position=np.asarray([0.0, 0.0], dtype=np.float32),
        start_direction_vector=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    root = tmp_path / "cache"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        target_free,
        "load_llm_grid_examples",
        lambda *_args, **_kwargs: _LoadedExamples(
            (_Example("bad", "R2R", "val_seen", "scene-a", raster),)
        ),
    )
    monkeypatch.setattr(target_free, "llm_navigation_split_dir", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        target_free,
        "llm_navigation_prediction_path",
        lambda *_args, **_kwargs: tmp_path / "missing.txt",
    )
    contract = target_free.PopulationContract(
        "val_seen", 1, 1, 0, 1, sha256(manifest.read_bytes()).hexdigest()
    )

    with pytest.raises(ValueError, match="start_direction"):
        target_free.load_runtime_population(
            "val_seen", contract, tmp_path, "model", "namespace", quiet=True
        )


def _target_for(
    runtime: target_free.RuntimeEpisode, angle: float
) -> target_free.TargetEpisode:
    target = warp_grid_about_pivot(
        runtime.selector_input.predicted_grid, runtime.start_pivot, angle
    ).grid
    return target_free.TargetEpisode(
        runtime,
        target,
        np.zeros((5, 2), dtype=np.float32),
    )


def _mapping_targets(mapping: target_free.HeadingMapping) -> tuple[target_free.TargetEpisode, ...]:
    headings = (
        np.asarray([0.0, 1.0], np.float32),
        np.asarray([1.0, 0.0], np.float32),
        np.asarray([0.0, -1.0], np.float32),
        np.asarray([-1.0, 0.0], np.float32),
    )
    runtimes = tuple(
        _runtime("scene", f"episode-{index}", heading)
        for index, heading in enumerate(headings)
    )
    return tuple(
        _target_for(runtime, mapping.angle_for(runtime.selector_input.start_direction))
        for runtime in runtimes
    )


@pytest.mark.parametrize(
    "mapping",
    tuple(
        target_free.HeadingMapping(sign, offset)
        for sign in (+1, -1)
        for offset in target_free.ANGLE_ORDER
    ),
)
def test_develop_selector_can_uniquely_lock_each_heading_mapping(
    mapping: target_free.HeadingMapping,
) -> None:
    """Breaks if mapping calibration omits or aliases a declared candidate."""
    lock = target_free.develop_selector(_mapping_targets(mapping))

    assert lock.chosen_mapping == mapping
    assert lock.mapping_tied_candidate_count == 1
    assert len(lock.development_scores) == 12
    assert [score.candidate_kind for score in lock.development_scores] == [
        "mapping"
    ] * 8 + ["global"] * 4


@pytest.mark.parametrize("angle", target_free.ANGLE_ORDER)
def test_develop_selector_can_uniquely_lock_each_global_angle(angle: float) -> None:
    """Breaks if global calibration is not a complete declared-angle search."""
    targets = tuple(_target_for(runtime, angle) for runtime in synthetic_runtime_population()[:2])
    lock = target_free.develop_selector(targets)

    assert lock.chosen_global_angle == angle
    assert lock.global_tied_candidate_count == 1


def test_develop_selector_keeps_declared_order_on_aggregate_ties_and_scores_invalid_rows(
) -> None:
    """Breaks if ties or denominator accounting depend on valid-only filtering."""
    invalid = _runtime(
        "scene", "invalid", np.asarray([0.0, 1.0], np.float32), schema_valid=False
    )
    empty = target_free.TargetEpisode(
        invalid, empty_target(), np.zeros((5, 2), dtype=np.float32)
    )
    lock = target_free.develop_selector((empty,))

    assert lock.chosen_mapping == target_free.HeadingMapping(+1, 0.0)
    assert lock.chosen_global_angle == 0.0
    assert lock.mapping_tied_candidate_count == 8
    assert lock.global_tied_candidate_count == 4
    assert {score.episode_count for score in lock.development_scores} == {1}
    assert {score.valid_count for score in lock.development_scores} == {0}


def test_assignments_are_target_free_sorted_and_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if assignment sealing reads targets or emits unstable CSV bytes."""
    monkeypatch.setattr(target_free, "_load_target_arrays", fail_if_called)
    assignments = target_free.assign_population(synthetic_runtime_population(), frozen_lock())
    payload = target_free.assignment_csv_bytes(assignments)

    assert [assignment.key for assignment in assignments] == sorted(
        assignment.key for assignment in assignments
    )
    invalid = next(assignment for assignment in assignments if not assignment.schema_valid)
    assert (invalid.primary_angle_degrees, invalid.global_angle_degrees) == (0.0, 0.0)
    assert b"schema_valid" in payload
    assert b"false" in payload
    assert b"-0" not in payload
    assert all(cell for row in payload.decode("utf-8").splitlines() for cell in row.split(","))


def test_assignment_csv_uses_exact_float_boolean_and_target_independent_bytes() -> None:
    """Breaks if serialization changes canonical evidence after target mutation."""
    assignment = target_free.SelectorAssignment(
        target_free.EpisodeKey("scene-z", "example"),
        True,
        -0.0,
        0.0,
        0.0,
        270.0,
        1.2345678901234567,
    )
    baseline = target_free.assignment_csv_bytes((assignment,))
    target = _target_for(_runtime("scene", "target", np.asarray([0.0, 1.0], np.float32)), 0.0)
    mutated = target.target_grid.copy()
    mutated[0, 0, 0] = ~mutated[0, 0, 0]
    changed_target = target_free.TargetEpisode(target.runtime, mutated, target.target_directions)

    assert changed_target.target_grid[0, 0, 0] != target.target_grid[0, 0, 0]
    assert target_free.assignment_csv_bytes((assignment,)) == baseline
    assert b"1.2345678901234567" in baseline
    assert b"true" in baseline


def _two_family_target_episode(
    *, scene_id: str = "scene-a", example_id: str = "episode-a", schema_valid: bool = True
) -> tuple[target_free.TargetEpisode, target_free.SelectorAssignment]:
    """Build a literal two-family raster whose cardinal scores are hand-checkable."""
    prediction = np.zeros((37, 50, 50), dtype=np.bool_)
    prediction[0, 24, 25] = True
    prediction[27, 25, 26] = True
    runtime = target_free.RuntimeEpisode(
        key=target_free.EpisodeKey(scene_id, example_id),
        split="val_unseen",
        selector_input=target_free.SelectorInput(
            schema_valid=schema_valid,
            start_direction=np.asarray([0.0, 1.0], dtype=np.float32),
            predicted_grid=prediction if schema_valid else empty_target(),
            predicted_directions=(
                np.asarray(
                    [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    dtype=np.float32,
                )
                if schema_valid
                else np.zeros((5, 2), dtype=np.float32)
            ),
        ),
        start_pivot=(25.0, 25.0),
    )
    target_grid = empty_target()
    target_grid[0, 25, 24] = True
    target_grid[27, 25, 26] = True
    target = target_free.TargetEpisode(
        runtime,
        target_grid,
        np.asarray(
            [[-1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    assignment = target_free.SelectorAssignment(
        runtime.key,
        schema_valid,
        0.0,
        180.0 if schema_valid else 0.0,
        0.0,
        90.0 if schema_valid else 0.0,
        1.0 if schema_valid else 0.0,
    )
    return target, assignment


def test_score_population_emits_canonical_cardinal_and_episode_endpoints() -> None:
    """Breaks if a target score omits a frozen endpoint or reselects an angle."""
    target, assignment = _two_family_target_episode()

    angle_rows, episode_rows = target_free.score_population((target,), (assignment,))

    assert [row.angle_degrees for row in angle_rows] == list(target_free.ANGLE_ORDER)
    assert [row.key for row in angle_rows] == [target.runtime.key] * 4
    assert [(row.all_iou, row.object_iou, row.region_iou) for row in angle_rows] == [
        pytest.approx((1.0 / 3.0, 0.0, 1.0)),
        pytest.approx((0.0, 0.0, 0.0)),
        pytest.approx((1.0 / 3.0, 1.0, 0.0)),
        pytest.approx((0.0, 0.0, 0.0)),
    ]
    assert [(row.predicted_support, row.target_support, row.in_frame_support, row.out_of_frame_support, row.union) for row in angle_rows] == [
        (2, 2, 2, 0, 3),
        (2, 2, 2, 0, 4),
        (2, 2, 2, 0, 3),
        (2, 2, 2, 0, 4),
    ]
    assert [row.direction_cosine for row in angle_rows] == pytest.approx((-1.0, 0.0, 1.0, 0.0))

    assert len(episode_rows) == 1
    row = episode_rows[0]
    assert row.key == target.runtime.key
    assert (row.identity_iou, row.primary_iou, row.global_iou, row.direct_iou) == pytest.approx(
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0)
    )
    assert row.random_expected_iou == pytest.approx(1.0 / 6.0)
    assert (row.soft_identity_iou, row.soft_aggregate_iou, row.oracle_iou) == pytest.approx(
        (1.0 / 3.0, 1.0 / 7.0, 1.0 / 3.0)
    )
    assert (row.identity_object_iou, row.primary_object_iou) == pytest.approx((0.0, 1.0))
    assert (row.identity_region_iou, row.primary_region_iou) == pytest.approx((1.0, 0.0))
    assert (row.primary_predicted_support, row.primary_target_support) == (2, 2)
    assert (row.primary_in_frame_support, row.primary_out_of_frame_support, row.primary_union) == (2, 0, 3)
    assert (row.soft_prediction_mass, row.soft_target_mass, row.soft_intersection_mass, row.soft_union_mass) == pytest.approx(
        (2.0, 2.0, 0.5, 3.5)
    )
    assert (row.primary_direction_cosine, row.direct_direction_cosine) == pytest.approx((1.0, 0.0))


def test_score_population_keeps_invalid_empty_denominator_rows_and_rejects_key_mismatch() -> None:
    """Breaks if invalid episodes disappear or targets and sealed assignments drift."""
    invalid_target, invalid_assignment = _two_family_target_episode(schema_valid=False)

    angle_rows, episode_rows = target_free.score_population((invalid_target,), (invalid_assignment,))

    assert len(angle_rows) == 4
    assert len(episode_rows) == 1
    assert {row.all_iou for row in angle_rows} == {0.0}
    assert episode_rows[0].oracle_iou == 0.0
    wrong_assignment = target_free.SelectorAssignment(
        target_free.EpisodeKey("other-scene", "other-episode"), True, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    with pytest.raises(ValueError, match="exactly match"):
        target_free.score_population((invalid_target,), (wrong_assignment,))


def _episode_score_row(
    scene_id: str,
    example_id: str,
    *,
    identity_iou: float,
    primary_iou: float,
    global_iou: float,
    direct_iou: float | None = None,
    soft_aggregate_iou: float | None = None,
) -> target_free.EpisodeScoreRow:
    """Build an otherwise-neutral score row with explicit contrast endpoints."""
    direct = identity_iou if direct_iou is None else direct_iou
    soft = identity_iou if soft_aggregate_iou is None else soft_aggregate_iou
    return target_free.EpisodeScoreRow(
        target_free.EpisodeKey(scene_id, example_id), True, 0.0, 0.0, 0.0, 0.0,
        identity_iou, primary_iou, global_iou, direct, (identity_iou + primary_iou + global_iou + direct) / 4.0,
        identity_iou, soft, max(identity_iou, primary_iou, global_iou, direct),
        identity_iou, primary_iou, identity_iou, primary_iou,
        1, 1, 1, 0, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
    )


def test_paired_scene_bootstrap_uses_shared_duplicate_scene_resamples() -> None:
    """Breaks if scenes are independently resampled or unequal sizes become scene-macro."""
    rows = (
        _episode_score_row("scene-a", "one", identity_iou=0.2, primary_iou=0.2, global_iou=0.1),
        _episode_score_row("scene-a", "two", identity_iou=0.2, primary_iou=0.4, global_iou=0.2),
        _episode_score_row("scene-b", "one", identity_iou=0.2, primary_iou=1.0, global_iou=0.2),
    )

    intervals = target_free.paired_scene_bootstrap(rows, repetitions=10_000, seed=42)

    draws = np.random.default_rng(42).integers(0, 2, size=(10_000, 2))
    multiplicities = np.stack(((draws == 0).sum(axis=1), (draws == 1).sum(axis=1)), axis=1)
    expected = (multiplicities @ np.asarray((0.2, 0.8))) / (multiplicities @ np.asarray((2, 1)))
    assert intervals.primary_identity.mean == pytest.approx(1.0 / 3.0)
    assert (intervals.primary_identity.ci_lower, intervals.primary_identity.ci_upper) == pytest.approx(
        tuple(np.percentile(expected, (2.5, 97.5)))
    )
    expected_primary_global = (multiplicities @ np.asarray((0.3, 0.8))) / (
        multiplicities @ np.asarray((2, 1))
    )
    assert (intervals.primary_global.ci_lower, intervals.primary_global.ci_upper) == pytest.approx(
        tuple(np.percentile(expected_primary_global, (2.5, 97.5)))
    )
    assert intervals.global_identity.mean == pytest.approx(-1.0 / 30.0)


def test_leave_one_scene_out_uses_episode_macro_paired_difference() -> None:
    """Breaks if LOSO averages scene means or subtracts independent aggregates."""
    rows = (
        _episode_score_row("scene-a", "one", identity_iou=0.2, primary_iou=0.2, global_iou=0.1),
        _episode_score_row("scene-a", "two", identity_iou=0.2, primary_iou=0.4, global_iou=0.2),
        _episode_score_row("scene-b", "one", identity_iou=0.2, primary_iou=1.0, global_iou=0.2),
    )

    assert target_free.leave_one_scene_out(rows, "primary_identity") == pytest.approx((0.1, 0.8))
    with pytest.raises(ValueError, match="at least two"):
        target_free.leave_one_scene_out(rows[:2], "primary_identity")


def _contrast(
    *, mean: float = 0.02, ci_lower: float = 0.001, loso_min: float = 0.001
) -> target_free.ContrastInterval:
    return target_free.ContrastInterval(mean, ci_lower, 0.1, loso_min, 0.2)


def test_decision_gate_is_mutually_exclusive_with_frozen_precedence() -> None:
    """Breaks if any decision predicate is reordered, weakened, or uses inclusive zero."""
    selector = _contrast()
    global_interval = _contrast()
    assert target_free.classify_decision(selector, selector, global_interval, 0.01, 0.01, 0.01, 0.01, True).label is target_free.DecisionLabel.SELECTOR_GO
    assert target_free.classify_decision(selector, _contrast(ci_lower=0.0), global_interval, 0.01, 0.01, 0.01, 0.01, True).label is target_free.DecisionLabel.FIXED_CORRECTION_GO
    assert target_free.classify_decision(_contrast(loso_min=0.0), selector, global_interval, 0.01, 0.01, 0.0, 0.0, True).label is target_free.DecisionLabel.PARTIAL
    assert target_free.classify_decision(_contrast(mean=0.009, ci_lower=0.005), _contrast(mean=0.0, ci_lower=0.0), _contrast(mean=0.0, ci_lower=0.0), 0.0, 0.0, 0.0, 0.0, True).label is target_free.DecisionLabel.NO_GO
