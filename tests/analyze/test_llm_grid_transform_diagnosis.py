from __future__ import annotations

from dataclasses import replace
from typing import Tuple, cast

import numpy as np
import pytest

from prior.analyze.d2026_07_27.llm_grid_transform_diagnosis import (
    EpisodeTransformResult,
    FourWayDiagnosisArgs,
    _validate_args,
    _run_rows,
    evaluate_episode,
    bootstrap_scene_delta,
    start_pivot_for_scale,
)
from prior.analyze.llm_grid_registration import warp_grid_about_pivot
from prior.constants import OBJECT_CATEGORIES, REGION_CATEGORIES


def test_scale_two_start_pivot_uses_one_meter_cells() -> None:
    """Breaks if the scale-2 pivot converts metres with the wrong cell width."""
    assert start_pivot_for_scale((18.487621, 6.2339373), 2) == pytest.approx(
        (18.487621, 6.2339373)
    )


@pytest.mark.parametrize(
    "start_position_m",
    ((1.0,), (1.0, np.nan), (np.inf, 1.0)),
)
def test_start_pivot_rejects_nonfinite_or_noncoordinate_positions(
    start_position_m: tuple[float, ...],
) -> None:
    """Breaks if malformed start positions reach incidental geometry failures."""
    with pytest.raises(ValueError, match="start_position_m"):
        start_pivot_for_scale(cast(Tuple[float, float], start_position_m), 2)


def test_invalid_prediction_is_retained_as_zero_scored_identity_episode() -> None:
    """Breaks if invalid output is dropped or gains transform-specific credit."""
    target = np.zeros((OBJECT_CATEGORIES + REGION_CATEGORIES, 4, 4), dtype=np.bool_)
    target[0, 1, 2] = True

    result = evaluate_episode(
        split="val_unseen",
        scene_id="scene",
        example_id="episode",
        schema_valid=False,
        predicted_grid=np.zeros_like(target),
        target_grid=target,
        start_position_m=(1.0, 1.0),
        predicted_direction_vectors=np.zeros((5, 2), dtype=np.float32),
        target_direction_vectors=np.zeros((5, 2), dtype=np.float32),
        instruction="walk past the chair",
        angles=(0.0, 90.0, 180.0, 270.0),
    )

    assert result.schema_valid is False
    assert result.best_angle_degrees == 0.0
    assert result.identity_iou == 0.0
    assert result.best_iou == 0.0
    assert result.delta_iou == 0.0
    assert dict(result.angle_ious) == {
        0.0: 0.0,
        90.0: 0.0,
        180.0: 0.0,
        270.0: 0.0,
    }
    assert result.target_support == 1


def test_episode_evaluation_rejects_noncanonical_grid_and_direction_shapes() -> None:
    """Breaks if malformed public inputs bypass explicit evaluator validation."""
    target = np.zeros((OBJECT_CATEGORIES + REGION_CATEGORIES, 2, 2), dtype=np.bool_)
    with pytest.raises(ValueError, match="37"):
        evaluate_episode(
            split="val_unseen",
            scene_id="scene",
            example_id="episode",
            schema_valid=True,
            predicted_grid=np.zeros((36, 2, 2), dtype=np.bool_),
            target_grid=np.zeros((36, 2, 2), dtype=np.bool_),
            start_position_m=(1.0, 1.0),
            predicted_direction_vectors=np.zeros((5, 2), dtype=np.float32),
            target_direction_vectors=np.zeros((5, 2), dtype=np.float32),
            instruction="",
            angles=(0.0, 90.0, 180.0, 270.0),
        )
    with pytest.raises(ValueError, match="same shape"):
        evaluate_episode(
            split="val_unseen",
            scene_id="scene",
            example_id="episode",
            schema_valid=True,
            predicted_grid=target,
            target_grid=np.zeros((OBJECT_CATEGORIES + REGION_CATEGORIES, 3, 2), dtype=np.bool_),
            start_position_m=(1.0, 1.0),
            predicted_direction_vectors=np.zeros((5, 2), dtype=np.float32),
            target_direction_vectors=np.zeros((5, 2), dtype=np.float32),
            instruction="",
            angles=(0.0, 90.0, 180.0, 270.0),
        )
    with pytest.raises(ValueError, match="predicted_direction_vectors"):
        evaluate_episode(
            split="val_unseen",
            scene_id="scene",
            example_id="episode",
            schema_valid=True,
            predicted_grid=target,
            target_grid=target,
            start_position_m=(1.0, 1.0),
            predicted_direction_vectors=np.zeros((4, 2), dtype=np.float32),
            target_direction_vectors=np.zeros((5, 2), dtype=np.float32),
            instruction="",
            angles=(0.0, 90.0, 180.0, 270.0),
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("cache_model_key", "other-cache"),
        ("cognitive_map_namespace", "other.namespace"),
        ("bootstrap_repetitions", 99),
        ("bootstrap_seed", 7),
    ),
)
def test_args_reject_overrides_to_fixed_cache_contract(
    field: str, invalid_value: object
) -> None:
    """Breaks if a CLI override can change the selected cache experiment."""
    args = FourWayDiagnosisArgs()
    setattr(args, field, invalid_value)

    with pytest.raises(ValueError, match=field):
        _validate_args(args)


def test_sensitivity_metrics_reuse_the_all_channel_selected_angle() -> None:
    """Breaks if a sensitivity subset independently selects a better rotation."""
    predicted = np.zeros((OBJECT_CATEGORIES + REGION_CATEGORIES, 4, 4), dtype=np.bool_)
    predicted[0, 1, 2] = True
    predicted[OBJECT_CATEGORIES, 2, 1] = True
    target = np.zeros_like(predicted)
    target[0] = warp_grid_about_pivot(predicted[0:1], (2.0, 2.0), 90.0).grid[0]
    target[OBJECT_CATEGORIES] = warp_grid_about_pivot(
        predicted[OBJECT_CATEGORIES : OBJECT_CATEGORIES + 1], (2.0, 2.0), 270.0
    ).grid[0]

    result = evaluate_episode(
        split="val_unseen",
        scene_id="scene",
        example_id="episode",
        schema_valid=True,
        predicted_grid=predicted,
        target_grid=target,
        start_position_m=(2.0, 2.0),
        predicted_direction_vectors=np.zeros((5, 2), dtype=np.float32),
        target_direction_vectors=np.zeros((5, 2), dtype=np.float32),
        instruction="walk past the chair",
        angles=(0.0, 90.0, 180.0, 270.0),
    )

    assert result.best_angle_degrees == 90.0
    assert result.object_best_iou == pytest.approx(1.0)
    assert result.region_best_iou == 0.0


def test_mentioned_and_unmentioned_masks_partition_canonical_channels() -> None:
    """Breaks if region mention IDs omit the object-channel offset or masks overlap."""
    target = np.zeros((OBJECT_CATEGORIES + REGION_CATEGORIES, 4, 4), dtype=np.bool_)
    target[1, 1, 1] = True  # canonical object channel 1 is chair
    target[OBJECT_CATEGORIES, 2, 2] = True

    result = evaluate_episode(
        split="val_unseen",
        scene_id="scene",
        example_id="episode",
        schema_valid=True,
        predicted_grid=target.copy(),
        target_grid=target,
        start_position_m=(2.0, 2.0),
        predicted_direction_vectors=np.zeros((5, 2), dtype=np.float32),
        target_direction_vectors=np.zeros((5, 2), dtype=np.float32),
        instruction="walk past the chair",
        angles=(0.0, 90.0, 180.0, 270.0),
    )

    assert result.mentioned_target_support == 1
    assert result.unmentioned_target_support == 1
    assert result.mentioned_best_iou == 1.0
    assert result.unmentioned_best_iou == 1.0


def _synthetic_result(
    scene_id: str, example_id: str, delta_iou: float
) -> EpisodeTransformResult:
    empty = np.zeros((OBJECT_CATEGORIES + REGION_CATEGORIES, 2, 2), dtype=np.bool_)
    result = evaluate_episode(
        split="val_unseen",
        scene_id=scene_id,
        example_id=example_id,
        schema_valid=True,
        predicted_grid=empty,
        target_grid=empty,
        start_position_m=(1.0, 1.0),
        predicted_direction_vectors=np.zeros((5, 2), dtype=np.float32),
        target_direction_vectors=np.zeros((5, 2), dtype=np.float32),
        instruction="",
        angles=(0.0, 90.0, 180.0, 270.0),
    )
    return replace(result, best_iou=delta_iou, delta_iou=delta_iou)


def test_scene_bootstrap_is_deterministic_and_reports_interval() -> None:
    """Breaks if bootstrap samples episodes independently or uses an unseeded RNG."""
    rows = (
        _synthetic_result("scene-a", "a1", 0.0),
        _synthetic_result("scene-a", "a2", 0.2),
        _synthetic_result("scene-b", "b1", 0.4),
        _synthetic_result("scene-b", "b2", 0.6),
    )

    first = bootstrap_scene_delta(rows, repetitions=10_000, seed=42)
    second = bootstrap_scene_delta(rows, repetitions=10_000, seed=42)

    assert first == second
    assert first["mean_delta"] == pytest.approx(0.3)
    assert first["percentile_2_5"] == pytest.approx(0.1)
    assert first["percentile_97_5"] == pytest.approx(0.5)
    assert "percentile_2_5" in first
    assert "percentile_97_5" in first
    assert first["sampling_unit"] == "scene"


def test_synthetic_runner_writes_artifacts_and_rejects_bad_population(
    tmp_path,
) -> None:
    """Breaks if artifacts are incomplete or malformed populations are accepted."""
    rows = (
        _synthetic_result("scene-a", "a1", 0.0),
        _synthetic_result("scene-a", "a2", 0.2),
        _synthetic_result("scene-b", "b1", 0.4),
        _synthetic_result("scene-b", "b2", 0.6),
    )
    args = FourWayDiagnosisArgs()
    args.output_dir = tmp_path
    args.limit = 4
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("source", encoding="utf-8")

    _run_rows(
        args,
        rows,
        expected_population=4,
        prediction_manifest_path=source_manifest,
    )

    assert {
        "summary.json",
        "manifest.json",
        "bootstrap.json",
        "episodes.csv",
        "iou_delta_distribution.png",
        "angle_distribution.png",
    } <= {path.name for path in tmp_path.iterdir()}
    assert '"smoke": true' in (tmp_path / "manifest.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _run_rows(
            args,
            (rows[0], replace(rows[1], example_id="a1"), rows[2], rows[3]),
            expected_population=4,
            prediction_manifest_path=source_manifest,
        )
    with pytest.raises(ValueError, match="expected 5"):
        _run_rows(
            args,
            rows,
            expected_population=5,
            prediction_manifest_path=source_manifest,
        )


def test_full_runner_rejects_population_without_all_expected_scenes(tmp_path) -> None:
    """Breaks if a non-smoke population can bootstrap fewer than 11 scenes."""
    rows = tuple(
        _synthetic_result(f"scene-{index}", f"episode-{index}", 0.0)
        for index in range(4)
    )
    args = FourWayDiagnosisArgs()
    args.output_dir = tmp_path
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("source", encoding="utf-8")

    with pytest.raises(ValueError, match="11 unique scenes"):
        _run_rows(
            args,
            rows,
            expected_population=4,
            prediction_manifest_path=source_manifest,
        )


def test_runner_rejects_missing_prediction_manifest(tmp_path) -> None:
    """Breaks if provenance records a missing manifest with a null hash."""
    rows = (
        _synthetic_result("scene-a", "a1", 0.0),
        _synthetic_result("scene-a", "a2", 0.0),
        _synthetic_result("scene-b", "b1", 0.0),
        _synthetic_result("scene-b", "b2", 0.0),
    )
    args = FourWayDiagnosisArgs()
    args.output_dir = tmp_path
    args.limit = 4

    with pytest.raises(FileNotFoundError, match="prediction_manifest_path"):
        _run_rows(
            args,
            rows,
            expected_population=4,
            prediction_manifest_path=tmp_path / "missing-manifest.json",
        )
