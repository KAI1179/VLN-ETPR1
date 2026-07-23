from __future__ import annotations

import numpy as np
import pytest

from vlnce_baselines.models.etp_llm.llm_grid_input_dependence import (
    GridEpisode,
    GridSignature,
    compute_permutation_controls,
    summarize_paraphrase_consistency,
)


def test_grid_signature_scores_category_aware_cells_and_directions() -> None:
    prediction_grid = np.zeros((3, 2, 2), dtype=np.float32)
    prediction_grid[0, 0, 0] = 1.0
    prediction_grid[1, 1, 1] = 1.0
    target_grid = np.zeros_like(prediction_grid)
    target_grid[0, 0, 0] = 1.0
    target_grid[2, 1, 1] = 1.0
    prediction = GridSignature.from_arrays(
        prediction_grid,
        np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    target = GridSignature.from_arrays(
        target_grid,
        np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    )

    score = prediction.score_against(target)

    assert score.raster_iou == pytest.approx(1 / 3)
    assert score.cell_f1 == pytest.approx(0.5)
    assert score.category_f1 == pytest.approx(0.5)
    assert score.direction_cosine == pytest.approx(0.0)


def test_grid_signature_consistency_is_symmetric_and_tracks_padding() -> None:
    first_grid = np.zeros((2, 2, 2), dtype=np.float32)
    first_grid[0, 0, 0] = 1.0
    second_grid = np.zeros_like(first_grid)
    second_grid[0, 1, 1] = 1.0
    first = GridSignature.from_arrays(
        first_grid,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    second = GridSignature.from_arrays(
        second_grid,
        np.asarray([[-1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )

    consistency = first.consistency_with(second)

    assert consistency.raster_iou == 0.0
    assert consistency.category_jaccard == 1.0
    assert consistency.direction_cosine == pytest.approx(-0.5)
    assert consistency.direction_padding_accuracy == pytest.approx(0.5)


def test_paraphrase_summary_reports_invalid_pairs_without_silent_dropping() -> None:
    grid = np.zeros((1, 1, 1), dtype=np.float32)
    grid[0, 0, 0] = 1.0
    signature = GridSignature.from_arrays(
        grid,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    episodes = [
        GridEpisode(
            split="val_unseen",
            scene_id="scene",
            example_id=f"episode-{index}",
            trajectory_id=7,
            prediction=prediction,
            target=signature,
            prediction_status="valid" if prediction else "invalid",
        )
        for index, prediction in enumerate((signature, signature, None))
    ]

    summary, pairs = summarize_paraphrase_consistency(episodes)

    assert summary.episode_count == 3
    assert summary.trajectory_count == 1
    assert summary.pair_count == 3
    assert summary.valid_pair_count == 1
    assert summary.invalid_pair_count == 2
    assert summary.raster_iou == 1.0
    assert len(pairs) == 3


def test_paraphrase_target_contract_tolerates_direction_rounding() -> None:
    grid = np.ones((1, 1, 1), dtype=np.float32)
    first = GridSignature.from_arrays(
        grid,
        np.asarray([[0.6, 0.8]], dtype=np.float32),
    )
    second = GridSignature.from_arrays(
        grid,
        np.asarray([[0.6000001, 0.7999999]], dtype=np.float32),
    )
    episodes = [
        GridEpisode(
            split="val_unseen",
            scene_id="scene",
            example_id=f"episode-{index}",
            trajectory_id=7,
            prediction=target,
            target=target,
            prediction_status="valid",
        )
        for index, target in enumerate((first, second))
    ]

    summary, _ = summarize_paraphrase_consistency(episodes)

    assert summary.valid_pair_count == 1


def test_permutation_controls_remove_episode_correspondence() -> None:
    first_grid = np.zeros((2, 1, 2), dtype=np.float32)
    first_grid[0, 0, 0] = 1.0
    second_grid = np.zeros_like(first_grid)
    second_grid[1, 0, 1] = 1.0
    directions = np.asarray([[1.0, 0.0]], dtype=np.float32)
    first = GridSignature.from_arrays(first_grid, directions)
    second = GridSignature.from_arrays(second_grid, directions)
    episodes = [
        GridEpisode(
            split="val_unseen",
            scene_id="scene",
            example_id="first",
            trajectory_id=1,
            prediction=first,
            target=first,
            prediction_status="valid",
        ),
        GridEpisode(
            split="val_unseen",
            scene_id="scene",
            example_id="second",
            trajectory_id=2,
            prediction=second,
            target=second,
            prediction_status="valid",
        ),
    ]

    controls = compute_permutation_controls(episodes, seed=42)

    assert controls.matched.metrics.raster_iou == 1.0
    assert controls.global_permutation.metrics.raster_iou == 0.0
    assert controls.within_scene_permutation.metrics.raster_iou == 0.0
    assert controls.global_permutation.fixed_point_count == 0
    assert controls.within_scene_permutation.same_scene_rate == 1.0


def test_permutation_direction_excludes_invalid_predictions() -> None:
    grid = np.ones((1, 1, 1), dtype=np.float32)
    signature = GridSignature.from_arrays(
        grid,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    episodes = [
        GridEpisode(
            split="val_unseen",
            scene_id="scene",
            example_id=f"episode-{index}",
            trajectory_id=index,
            prediction=prediction,
            target=signature,
            prediction_status="valid" if prediction else "invalid",
        )
        for index, prediction in enumerate((signature, None))
    ]

    controls = compute_permutation_controls(episodes, seed=42)

    assert controls.matched.metrics.direction_cosine == 1.0
