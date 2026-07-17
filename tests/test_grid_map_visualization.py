from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prior.analyze.batch_vis import (
    BatchVisualizationArgs,
    _sample_prediction_paths,
    render_comparisons,
)
from prior.grid_map import BaseGridMap
from prior.grid_map._visualize import _MapOverlay, _grid_bounds


def test_batch_visualization_defaults_to_mixed_tag_free_grid_cache():
    assert (
        "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
        in str(BatchVisualizationArgs().prediction_root)
    )


def test_visualize_comparison_converts_context_to_grid_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted_map = BaseGridMap()
    ground_truth_map = BaseGridMap()
    captured: dict[str, object] = {}

    def fake_visualize_comparison(*args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "prior.grid_map._visualize.visualize_comparison",
        fake_visualize_comparison,
    )

    predicted_map.visualize_comparison(
        ground_truth_map,
        tmp_path / "comparison.png",
        instruction="Walk to the chair.",
        ground_truth_trajectory=[(1.0, 2.0), (2.0, 3.0)],
        trajectory_keypoints=[
            (0.0, 0.0),
            (1.0, 2.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        start_direction_vector=(1.0, 0.0),
    )

    assert captured["instruction"] == "Walk to the chair."
    assert captured["ground_truth_trajectory"] == [(2.0, 4.0), (4.0, 6.0)]
    assert captured["trajectory_keypoints"] == [(0.0, 0.0), (2.0, 4.0)]


def test_grid_bounds_cover_both_maps_and_episode_overlay() -> None:
    predicted_map = BaseGridMap()
    predicted_map.grid[1, 10, 20] = 1.0
    ground_truth_map = BaseGridMap()
    ground_truth_map.grid[2, 30, 40] = 1.0
    overlay = _MapOverlay(trajectory=((5.0, 50.0), (60.0, 2.0)))

    bounds = _grid_bounds(
        [predicted_map, ground_truth_map],
        overlay,
        auto_crop=True,
        crop_margin=2,
    )

    assert bounds == (3, 62, 0, 52)


def test_render_comparisons_loads_paired_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_root = tmp_path / "predictions"
    ground_truth_root = tmp_path / "ground-truth"
    output_root = tmp_path / "output"
    prediction_path = prediction_root / "scene" / "R2R_val_unseen_1.npz"
    ground_truth_path = ground_truth_root / "raster" / "scene" / "R2R_val_unseen_1.npz"
    boxes_path = ground_truth_root / "boxes" / "scene" / "R2R_val_unseen_1.npz"
    for path in (prediction_path, ground_truth_path, boxes_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    predicted_map = BaseGridMap()
    loaded_paths: list[Path] = []

    def fake_load(path: Path) -> BaseGridMap:
        loaded_paths.append(Path(path))
        return predicted_map if Path(path) == prediction_path else BaseGridMap()

    context = SimpleNamespace(
        instruction="Walk to the chair.",
        ground_truth_trajectory=[(1.0, 2.0), (2.0, 3.0)],
        trajectory_keypoints=[(1.0, 2.0)] * 5,
        start_direction_vector=(1.0, 0.0),
    )
    captured: dict[str, object] = {}

    def fake_comparison(
        self: BaseGridMap,
        ground_truth_map: BaseGridMap,
        save_path: Path,
        **kwargs: object,
    ) -> None:
        captured.update(kwargs)
        captured["save_path"] = save_path

    monkeypatch.setattr(BaseGridMap, "load", staticmethod(fake_load))
    monkeypatch.setattr(
        "prior.analyze.batch_vis.RelevantSemanticBoxes.load",
        lambda path: context,
    )
    monkeypatch.setattr(BaseGridMap, "visualize_comparison", fake_comparison)

    rendered = render_comparisons(
        prediction_root,
        ground_truth_root,
        output_root,
    )

    assert rendered == 1
    assert loaded_paths == [prediction_path, ground_truth_path]
    assert captured["instruction"] == context.instruction
    assert captured["ground_truth_trajectory"] == context.ground_truth_trajectory
    assert captured["save_path"] == output_root / "scene" / "R2R_val_unseen_1.png"


def test_render_comparisons_rejects_missing_boxes(tmp_path: Path) -> None:
    prediction_root = tmp_path / "predictions"
    ground_truth_root = tmp_path / "ground-truth"
    prediction_path = prediction_root / "scene" / "episode.npz"
    ground_truth_path = ground_truth_root / "raster" / "scene" / "episode.npz"
    prediction_path.parent.mkdir(parents=True)
    ground_truth_path.parent.mkdir(parents=True)
    (ground_truth_root / "boxes").mkdir()
    prediction_path.touch()
    ground_truth_path.touch()

    with pytest.raises(FileNotFoundError, match="Missing paired boxes"):
        render_comparisons(prediction_root, ground_truth_root, tmp_path / "output")


def test_sample_prediction_paths_is_deterministic(tmp_path: Path) -> None:
    prediction_root = tmp_path / "predictions"
    paths = [prediction_root / "scene" / f"episode_{index}.npz" for index in range(5)]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    first = _sample_prediction_paths(prediction_root, count=2, seed=7)
    second = _sample_prediction_paths(prediction_root, count=2, seed=7)

    assert first == second
    assert len(first) == 2


def test_sample_prediction_paths_rejects_excess_count(tmp_path: Path) -> None:
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()

    with pytest.raises(
        ValueError,
        match="count 1 exceeds available prediction count 0",
    ):
        _sample_prediction_paths(prediction_root, count=1, seed=0)
