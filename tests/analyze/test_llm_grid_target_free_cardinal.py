from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
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
    target_grid[27, 26, 24] = True
    target = target_free.TargetEpisode(
        runtime,
        target_grid,
        np.asarray(
            [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    direct_angle, direct_margin = target_free.direct_heading_assignment(runtime.selector_input)
    assignment = target_free.SelectorAssignment(
        runtime.key,
        schema_valid,
        0.0,
        180.0 if schema_valid else 0.0,
        0.0,
        direct_angle,
        direct_margin,
    )
    return target, assignment


def test_score_population_emits_canonical_cardinal_and_episode_endpoints() -> None:
    """Breaks if a target score omits a frozen endpoint or reselects an angle."""
    target, assignment = _two_family_target_episode()

    angle_rows, episode_rows = target_free.score_population((target,), (assignment,))

    assert assignment.direct_angle_degrees == 270.0
    assert [row.angle_degrees for row in angle_rows] == list(target_free.ANGLE_ORDER)
    assert [row.key for row in angle_rows] == [target.runtime.key] * 4
    assert [(row.all_iou, row.object_iou, row.region_iou) for row in angle_rows] == [
        pytest.approx((1.0 / 4.0, 0.0, 1.0 / 2.0)),
        pytest.approx((0.0, 0.0, 0.0)),
        pytest.approx((1.0 / 4.0, 1.0, 0.0)),
        pytest.approx((1.0 / 4.0, 0.0, 1.0 / 2.0)),
    ]
    assert [(row.predicted_support, row.target_support, row.in_frame_support, row.out_of_frame_support, row.union) for row in angle_rows] == [
        (2, 3, 2, 0, 4),
        (2, 3, 2, 0, 5),
        (2, 3, 2, 0, 4),
        (2, 3, 2, 0, 4),
    ]
    assert [row.direction_cosine for row in angle_rows] == pytest.approx((0.0, -1.0, 0.0, 1.0))

    assert len(episode_rows) == 1
    row = episode_rows[0]
    assert row.key == target.runtime.key
    assert (row.identity_iou, row.primary_iou, row.global_iou, row.direct_iou) == pytest.approx(
        (1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0)
    )
    assert row.random_expected_iou == pytest.approx(3.0 / 16.0)
    assert (row.soft_identity_iou, row.soft_aggregate_iou, row.oracle_iou) == pytest.approx(
        (1.0 / 4.0, 3.0 / 17.0, 1.0 / 4.0)
    )
    assert (
        row.soft_identity_object_iou,
        row.soft_aggregate_object_iou,
        row.soft_identity_region_iou,
        row.soft_aggregate_region_iou,
    ) == pytest.approx((0.0, 1.0 / 7.0, 1.0 / 2.0, 1.0 / 5.0))
    assert (row.identity_object_iou, row.primary_object_iou) == pytest.approx((0.0, 1.0))
    assert (row.identity_region_iou, row.primary_region_iou) == pytest.approx((1.0 / 2.0, 0.0))
    assert (row.primary_predicted_support, row.primary_target_support) == (2, 3)
    assert (row.primary_in_frame_support, row.primary_out_of_frame_support, row.primary_union) == (2, 0, 4)
    assert (row.soft_prediction_mass, row.soft_target_mass, row.soft_intersection_mass, row.soft_union_mass) == pytest.approx(
        (2.0, 3.0, 0.75, 4.25)
    )
    assert (row.primary_direction_cosine, row.direct_direction_cosine) == pytest.approx((0.0, 1.0))


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
        identity_iou, soft, identity_iou, soft,
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


def test_oracle_gain_summary_uses_aggregate_ratio_and_nonpositive_availability() -> None:
    """Breaks if oracle recovery averages unstable ratios or accepts zero gain."""
    positive = _episode_score_row(
        "scene-a", "positive", identity_iou=0.2, primary_iou=0.5, global_iou=0.3,
        direct_iou=0.1, soft_aggregate_iou=0.4,
    )
    nonpositive = _episode_score_row(
        "scene-b", "nonpositive", identity_iou=0.4, primary_iou=0.2, global_iou=0.3,
        direct_iou=0.1, soft_aggregate_iou=0.2,
    )

    positive_summary = target_free.oracle_gain_summary((positive,))
    nonpositive_summary = target_free.oracle_gain_summary((nonpositive,))

    assert positive_summary.available is True
    assert (
        positive_summary.primary_fraction,
        positive_summary.global_fraction,
        positive_summary.direct_fraction,
        positive_summary.soft_fraction,
    ) == pytest.approx((1.0, 1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0))
    assert nonpositive_summary.available is False
    assert (
        nonpositive_summary.primary_fraction,
        nonpositive_summary.global_fraction,
        nonpositive_summary.direct_fraction,
        nonpositive_summary.soft_fraction,
    ) == (0.0, 0.0, 0.0, 0.0)


def test_oracle_gain_summary_uses_ratio_of_unequal_row_sums() -> None:
    """Breaks if aggregate recovery is replaced by an average of row ratios."""
    small_oracle_gain = _episode_score_row(
        "scene-a", "small", identity_iou=0.2, primary_iou=0.5, global_iou=0.3,
        direct_iou=0.1, soft_aggregate_iou=0.4,
    )
    large_oracle_gain = _episode_score_row(
        "scene-b", "large", identity_iou=0.1, primary_iou=0.2, global_iou=0.1,
        direct_iou=0.9, soft_aggregate_iou=0.1,
    )

    summary = target_free.oracle_gain_summary((small_oracle_gain, large_oracle_gain))

    assert summary.available is True
    assert summary.primary_fraction == pytest.approx(4.0 / 11.0)


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


def test_decision_gate_requires_explicit_semantic_evidence() -> None:
    """Breaks if omitted family gains can fabricate a selector or fixed GO."""
    gate = getattr(target_free, "classify_decision")
    with pytest.raises(TypeError):
        gate(_contrast(), _contrast(), _contrast(), audit_passed=True)


def _artifact_fixture(
    tmp_path: Path,
) -> tuple[
    target_free.ArtifactBundle,
    target_free.ValidationInputs,
    tuple[target_free.TargetEpisode, ...],
    tuple[target_free.TargetEpisode, ...],
]:
    development_targets = _mapping_targets(target_free.HeadingMapping(+1, 0.0))
    selector_lock = target_free.develop_selector(development_targets)
    test_pairs = (
        _two_family_target_episode(scene_id="scene-a", example_id="episode-a"),
        _two_family_target_episode(scene_id="scene-b", example_id="episode-b"),
    )
    test_targets = tuple(pair[0] for pair in test_pairs)
    test_runtime = tuple(target.runtime for target in test_targets)
    assignments = target_free.assign_population(test_runtime, selector_lock)
    angle_rows, episode_rows = target_free.score_population(test_targets, assignments)
    development_population = target_free.PopulationSummary(4, 1, 4, 0)
    test_population = target_free.PopulationSummary(2, 2, 2, 0)
    population = target_free.PopulationArtifact(
        development_population, test_population
    )
    summary = target_free._build_summary_artifact(
        angle_rows, episode_rows, population
    )
    bootstrap = target_free._build_bootstrap_artifact(
        episode_rows, seed=42, repetitions=10_000
    )
    development_manifest = Path(
        "data/llm_navigation/"
        "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2/"
        "r2r/val_seen/manifest.json"
    )
    test_manifest = Path(
        "data/llm_navigation/"
        "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2/"
        "r2r/val_unseen/manifest.json"
    )
    sources = target_free.SourcesArtifact(
        target_free.SourceArtifact(
            str(development_manifest),
            sha256(development_manifest.read_bytes()).hexdigest(),
            "R2R",
            "val_seen",
        ),
        target_free.SourceArtifact(
            str(test_manifest),
            sha256(test_manifest.read_bytes()).hexdigest(),
            "R2R",
            "val_unseen",
        ),
    )
    protocol = target_free.ProtocolArtifact(
        "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2",
        "gt.legacy.r1p5.direction5.v1",
        2,
        (37, 50, 50),
        (0, 27),
        (27, 37),
        target_free.ANGLE_ORDER,
        tuple(
            (mapping.sign, int(mapping.offset_degrees))
            for mapping in (
                target_free.HeadingMapping(sign, angle)
                for sign in (+1, -1)
                for angle in target_free.ANGLE_ORDER
            )
        ),
        target_free.ANGLE_ORDER,
        0.0,
        42,
        10_000,
        0.01,
        "llm-grid-target-free-cardinal-v2",
        (
            "offline raster overlap is not navigation performance",
            "oracle and target-vector diagnostics are nondeployable",
        ),
    )
    schemas = target_free.SchemaArtifact(
        target_free.CsvHeadersArtifact(
            target_free.DEVELOPMENT_MAPPING_SCORES_HEADER,
            target_free.TEST_ASSIGNMENTS_HEADER,
            target_free.TEST_ANGLE_SCORES_HEADER,
            target_free.TEST_EPISODE_SCORES_HEADER,
        ),
        target_free.JsonTopLevelKeysArtifact(
            target_free.SELECTOR_LOCK_KEYS,
            target_free.SUMMARY_KEYS,
            target_free.BOOTSTRAP_KEYS,
            target_free.MANIFEST_KEYS,
        ),
        ".17g finite floats; negative zero normalized to 0",
        "UTF-8 lexicographic episode keys, then declared candidate/angle order",
        "UTF-8 sorted compact keys, no NaN, one trailing newline",
    )
    artifacts = target_free.build_artifacts(
        selector_lock,
        assignments,
        angle_rows,
        episode_rows,
        summary,
        bootstrap,
        target_free.ManifestInputs(
            sources,
            "a" * 40,
            protocol,
            population,
            schemas,
        ),
    )
    development_contract = target_free.PopulationContract(
        "val_seen",
        4,
        1,
        4,
        0,
        sources.development.sha256,
    )
    test_contract = target_free.PopulationContract(
        "val_unseen",
        2,
        2,
        2,
        0,
        sources.test.sha256,
    )
    validation_inputs = target_free.ValidationInputs(
        development_contract,
        test_contract,
        tmp_path / "cache",
        protocol.cache_model_key,
        protocol.cognitive_map_namespace,
        True,
    )
    return artifacts, validation_inputs, development_targets, test_targets


def _install_artifact_loaders(
    monkeypatch: pytest.MonkeyPatch,
    development_targets: tuple[target_free.TargetEpisode, ...],
    test_targets: tuple[target_free.TargetEpisode, ...],
) -> None:
    runtime_by_split = {
        "val_seen": tuple(target.runtime for target in development_targets),
        "val_unseen": tuple(target.runtime for target in test_targets),
    }
    targets_by_split = {
        "val_seen": development_targets,
        "val_unseen": test_targets,
    }
    monkeypatch.setattr(
        target_free,
        "load_runtime_population",
        lambda split, *_args, **_kwargs: runtime_by_split[split],
    )
    monkeypatch.setattr(
        target_free,
        "load_target_population",
        lambda runtimes, *_args, **_kwargs: targets_by_split[runtimes[0].split],
    )


def test_build_artifacts_emits_exact_canonical_transaction_and_decodable_plot(
    tmp_path: Path,
) -> None:
    """Breaks if any artifact filename, schema, hash, row count, or PNG drifts."""
    artifacts, _, _, _ = _artifact_fixture(tmp_path)
    files = artifacts.files()

    assert tuple(files) == (
        "manifest.json",
        "development_mapping_scores.csv",
        "selector_lock.json",
        "test_assignments.csv",
        "test_angle_scores.csv",
        "test_episode_scores.csv",
        "summary.json",
        "bootstrap.json",
        "target_free_control_intervals.png",
    )
    assert files["development_mapping_scores.csv"].splitlines()[0].decode() == (
        "candidate_kind,mapping_sign,heading_offset_degrees,global_angle_degrees,"
        "episode_count,valid_count,all_mean_iou,object_mean_iou,region_mean_iou,selected"
    )
    assert files["test_episode_scores.csv"].splitlines()[0].decode() == (
        "scene_id,example_id,schema_valid,heading_bin_degrees,primary_angle_degrees,"
        "global_angle_degrees,direct_angle_degrees,identity_iou,primary_iou,global_iou,"
        "direct_iou,random_expected_iou,soft_identity_iou,soft_aggregate_iou,oracle_iou,"
        "identity_object_iou,primary_object_iou,identity_region_iou,primary_region_iou,"
        "soft_identity_object_iou,soft_aggregate_object_iou,soft_identity_region_iou,"
        "soft_aggregate_region_iou,primary_predicted_support,primary_target_support,"
        "primary_in_frame_support,primary_out_of_frame_support,primary_union,"
        "soft_prediction_mass,soft_target_mass,soft_intersection_mass,soft_union_mass,"
        "primary_direction_cosine,direct_direction_cosine"
    )
    manifest = json.loads(files["manifest.json"])
    assert tuple(manifest) == tuple(sorted(target_free.MANIFEST_KEYS))
    assert manifest["schema_version"] == "llm-grid-target-free-cardinal-v2"
    entries = {entry["name"]: entry for entry in manifest["artifacts"]}
    assert set(entries) == set(files) - {"manifest.json"}
    for name, payload in files.items():
        assert payload
        if name != "manifest.json":
            assert entries[name]["sha256"] == sha256(payload).hexdigest()
            assert entries[name]["size_bytes"] == len(payload)
    image = plt.imread(BytesIO(files["target_free_control_intervals.png"]), format="png")
    assert image.ndim == 3
    assert np.isfinite(image).all()
    summary = json.loads(files["summary.json"])
    assert summary["means"]["soft_identity"] == pytest.approx(
        {"all_iou": 0.25, "object_iou": 0.0, "region_iou": 0.5}
    )
    assert summary["means"]["soft_aggregate"] == pytest.approx(
        {"all_iou": 3.0 / 17.0, "object_iou": 1.0 / 7.0, "region_iou": 0.2}
    )
    assert summary["decision"]["label"] in {
        "SELECTOR GO",
        "FIXED-CORRECTION GO",
        "PARTIAL",
        "NO GO",
    }


def test_publish_artifacts_validates_then_atomically_publishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Breaks if publication exposes a partial directory or skips semantic validation."""
    artifacts, validation_inputs, development_targets, test_targets = _artifact_fixture(
        tmp_path
    )
    _install_artifact_loaders(monkeypatch, development_targets, test_targets)
    output_dir = tmp_path / "published"

    target_free.publish_artifacts(output_dir, artifacts, validation_inputs)

    assert output_dir.is_dir()
    assert {path.name for path in output_dir.iterdir()} == set(artifacts.files())
    target_free.validate_artifact_directory(
        output_dir,
        expected_development_episodes=4,
        expected_development_scenes=1,
        expected_development_valid=4,
        expected_development_invalid=0,
        expected_test_episodes=2,
        expected_test_scenes=2,
        expected_test_valid=2,
        expected_test_invalid=0,
    )
    with pytest.raises(FileExistsError):
        target_free.publish_artifacts(output_dir, artifacts, validation_inputs)


@pytest.mark.parametrize(
    "name",
    (
        "manifest.json",
        "development_mapping_scores.csv",
        "selector_lock.json",
        "test_assignments.csv",
        "test_angle_scores.csv",
        "test_episode_scores.csv",
        "summary.json",
        "bootstrap.json",
        "target_free_control_intervals.png",
    ),
)
def test_validator_rejects_each_tampered_artifact(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Breaks if byte, hash, semantic, order, JSON-key, or PNG tampering survives."""
    artifacts, validation_inputs, development_targets, test_targets = _artifact_fixture(
        tmp_path
    )
    _install_artifact_loaders(monkeypatch, development_targets, test_targets)
    output_dir = tmp_path / "tampered"
    target_free.publish_artifacts(output_dir, artifacts, validation_inputs)
    path = output_dir / name
    if name.endswith(".json"):
        payload = json.loads(path.read_bytes())
        payload["unknown_key"] = True
        path.write_bytes(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
    elif name.endswith(".csv"):
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:1] + list(reversed(lines[1:]))))
    else:
        path.write_bytes(b"not a png")

    with pytest.raises(ValueError):
        target_free.validate_artifact_directory(
            output_dir,
            expected_development_episodes=4,
            expected_development_scenes=1,
            expected_development_valid=4,
            expected_development_invalid=0,
            expected_test_episodes=2,
            expected_test_scenes=2,
            expected_test_valid=2,
            expected_test_invalid=0,
        )


@pytest.mark.parametrize(
    ("name", "field"),
    (
        ("selector_lock.json", "chosen_global_angle"),
        ("summary.json", "decision"),
        ("bootstrap.json", "repetitions"),
    ),
)
def test_validator_rejects_rehashed_semantic_tampering(
    name: str,
    field: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Breaks if manifest rehashing can bless a wrong lock, bootstrap, or decision."""
    artifacts, validation_inputs, development_targets, test_targets = _artifact_fixture(
        tmp_path
    )
    _install_artifact_loaders(monkeypatch, development_targets, test_targets)
    output_dir = tmp_path / "semantic-tamper"
    target_free.publish_artifacts(output_dir, artifacts, validation_inputs)
    path = output_dir / name
    payload = json.loads(path.read_bytes())
    if field == "chosen_global_angle":
        payload[field]["angle_degrees"] = 90.0
    elif field == "decision":
        payload[field]["label"] = "SELECTOR GO"
    else:
        payload[field] = 999
    changed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(changed)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    entry = next(entry for entry in manifest["artifacts"] if entry["name"] == name)
    entry["sha256"] = sha256(changed).hexdigest()
    entry["size_bytes"] = len(changed)
    manifest_path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )

    with pytest.raises(ValueError):
        target_free.validate_artifact_directory(
            output_dir,
            expected_development_episodes=4,
            expected_development_scenes=1,
            expected_development_valid=4,
            expected_development_invalid=0,
            expected_test_episodes=2,
            expected_test_scenes=2,
            expected_test_valid=2,
            expected_test_invalid=0,
        )


def test_validator_rejects_relocated_source_with_identical_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Breaks if a manifest can redefine a frozen source path from its own bytes."""
    artifacts, _, development_targets, test_targets = _artifact_fixture(tmp_path)
    _install_artifact_loaders(monkeypatch, development_targets, test_targets)
    relocated = tmp_path / "relocated_manifest.json"
    frozen_source = Path(
        "data/llm_navigation/"
        "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2/"
        "r2r/val_seen/manifest.json"
    )
    relocated.write_bytes(frozen_source.read_bytes())
    manifest = json.loads(artifacts.manifest_json)
    manifest["sources"]["development"]["path"] = str(relocated)
    changed_manifest = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    output_dir = tmp_path / "relocated"
    output_dir.mkdir()
    for name, payload in artifacts.files().items():
        (output_dir / name).write_bytes(
            changed_manifest if name == "manifest.json" else payload
        )

    with pytest.raises(ValueError, match="source|frozen|path"):
        target_free.validate_artifact_directory(
            output_dir,
            expected_development_episodes=4,
            expected_development_scenes=1,
            expected_development_valid=4,
            expected_development_invalid=0,
            expected_test_episodes=2,
            expected_test_scenes=2,
            expected_test_valid=2,
            expected_test_invalid=0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("cache_model_key", "other-model"),
        ("cognitive_map_namespace", "other.namespace"),
        ("schema_version", "llm-grid-target-free-cardinal-v1"),
        ("angle_order", [90.0, 0.0, 180.0, 270.0]),
        ("mapping_order", [[-1, 0]]),
        ("bootstrap_seed", 7),
        ("bootstrap_repetitions", 999),
        ("gate_threshold", 0.02),
    ),
)
def test_validator_rejects_manifest_protocol_drift_from_frozen_constants(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Breaks if scientific expectations are derived from manifest protocol values."""
    artifacts, _, development_targets, test_targets = _artifact_fixture(tmp_path)
    _install_artifact_loaders(monkeypatch, development_targets, test_targets)
    manifest = json.loads(artifacts.manifest_json)
    manifest["protocol"][field] = value
    changed_manifest = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    output_dir = tmp_path / f"protocol-{field}"
    output_dir.mkdir()
    for name, payload in artifacts.files().items():
        (output_dir / name).write_bytes(
            changed_manifest if name == "manifest.json" else payload
        )

    with pytest.raises(ValueError):
        target_free.validate_artifact_directory(
            output_dir,
            expected_development_episodes=4,
            expected_development_scenes=1,
            expected_development_valid=4,
            expected_development_invalid=0,
            expected_test_episodes=2,
            expected_test_scenes=2,
            expected_test_valid=2,
            expected_test_invalid=0,
        )


def test_publish_failure_removes_only_exact_temporary_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Breaks if failed validation leaves partial output or removes sibling data."""
    artifacts, validation_inputs, _, _ = _artifact_fixture(tmp_path)
    output_dir = tmp_path / "official"
    preserved = tmp_path / "official.keep"
    preserved.mkdir()
    (preserved / "evidence").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        target_free,
        "_validate_artifact_directory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("injected")),
    )

    with pytest.raises(ValueError, match="injected"):
        target_free.publish_artifacts(output_dir, artifacts, validation_inputs)

    assert not output_dir.exists()
    assert (preserved / "evidence").read_text(encoding="utf-8") == "keep"
    assert not tuple(tmp_path.glob(".official.tmp-*"))


def test_fixed_tap_cli_rejects_protocol_and_surface_overrides(tmp_path: Path) -> None:
    """Breaks if the CLI permits limit/overwrite/resume or scientific drift."""
    args = target_free.TargetFreeArgs(
        underscores_to_dashes=True, explicit_bool=True
    ).parse_args(["--bootstrap-seed", "7"])
    with pytest.raises(ValueError, match="bootstrap_seed"):
        target_free._validate_args(args)
    with pytest.raises(SystemExit):
        target_free.TargetFreeArgs(underscores_to_dashes=True).parse_args(["--limit", "1"])
    with pytest.raises(SystemExit):
        target_free.TargetFreeArgs(underscores_to_dashes=True).parse_args(
            ["--overwrite"]
        )


def test_fixed_cli_refuses_dirty_worktree_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if uncommitted code can produce an ostensibly committed experiment."""
    def completed(
        command: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        stdout = " M prior/analyze/dirty.py\n" if command[1] == "status" else "b" * 40 + "\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(target_free.subprocess, "run", completed)
    monkeypatch.setattr(target_free, "load_runtime_population", fail_if_called)

    with pytest.raises(RuntimeError, match="clean"):
        target_free._run(target_free.TargetFreeArgs())


def test_clean_preflight_returns_exact_committed_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if manifest provenance is captured from a different HEAD."""
    calls: list[tuple[str, ...]] = []

    def completed(
        command: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        stdout = "" if command[1] == "status" else "c" * 40 + "\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(target_free.subprocess, "run", completed)

    assert target_free._preflight_git_commit() == "c" * 40
    assert calls == [
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        ("git", "rev-parse", "HEAD"),
    ]


def test_source_preflight_hashes_both_frozen_manifests_before_data_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Breaks if the evaluation manifest stays unchecked until after development."""
    corrupt_test_manifest = tmp_path / "manifest.json"
    corrupt_test_manifest.write_text('{"corrupt":true}\n', encoding="utf-8")
    monkeypatch.setattr(target_free, "_TEST_SOURCE_PATH", corrupt_test_manifest)
    monkeypatch.setattr(target_free, "load_runtime_population", fail_if_called)

    with pytest.raises(ValueError, match="SHA-256"):
        target_free._preflight_sources(target_free.TargetFreeArgs())
