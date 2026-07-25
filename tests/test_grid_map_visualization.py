from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prior.analyze.batch_vis import (
    BatchVisualizationArgs,
    _load_prediction_map,
    _sample_prediction_paths,
    render_comparisons,
    render_prediction_paths,
)
from prior.grid_map import BaseGridMap, CognitiveGridMap
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
        predicted_trajectory_keypoints=[
            (1.0, 1.0),
            (2.0, 2.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        ground_truth_trajectory=[(1.0, 2.0), (2.0, 3.0)],
        ground_truth_trajectory_keypoints=[
            (0.0, 0.0),
            (1.0, 2.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        start_direction_vector=(1.0, 0.0),
    )

    assert captured["instruction"] == "Walk to the chair."
    assert captured["predicted_trajectory_keypoints"] == [(2.0, 2.0), (4.0, 4.0)]
    assert captured["ground_truth_trajectory"] == [(2.0, 4.0), (4.0, 6.0)]
    assert captured["ground_truth_trajectory_keypoints"] == [
        (0.0, 0.0),
        (2.0, 4.0),
    ]


def test_visualize_comparison_does_not_fallback_to_ground_truth_waypoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "prior.grid_map._visualize.visualize_comparison",
        lambda *args, **kwargs: captured.update(kwargs),
    )

    BaseGridMap().visualize_comparison(
        BaseGridMap(),
        tmp_path / "comparison.png",
        instruction="Walk to the chair.",
        predicted_trajectory_keypoints=None,
        ground_truth_trajectory=[(1.0, 2.0), (2.0, 3.0)],
        ground_truth_trajectory_keypoints=[(1.0, 2.0)] * 5,
        start_direction_vector=(1.0, 0.0),
    )

    assert captured["predicted_trajectory_keypoints"] == []
    assert captured["ground_truth_trajectory_keypoints"] == [(2.0, 4.0)] * 5


def test_grid_bounds_cover_both_maps_and_episode_overlay() -> None:
    predicted_map = BaseGridMap()
    predicted_map.grid[1, 10, 20] = 1.0
    ground_truth_map = BaseGridMap()
    ground_truth_map.grid[2, 30, 40] = 1.0
    overlay = _MapOverlay(trajectory=((5.0, 50.0), (60.0, 2.0)))

    bounds = _grid_bounds(
        [predicted_map, ground_truth_map],
        [overlay],
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
        return BaseGridMap()

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

    def fake_load_prediction(path: Path):
        loaded_paths.append(Path(path))
        return predicted_map, [(4.0, 5.0)]

    monkeypatch.setattr(
        "prior.analyze.batch_vis._load_prediction_map",
        fake_load_prediction,
    )
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
    assert captured["predicted_trajectory_keypoints"] == [(4.0, 5.0)]
    assert captured["ground_truth_trajectory"] == context.ground_truth_trajectory
    assert (
        captured["ground_truth_trajectory_keypoints"]
        == context.trajectory_keypoints
    )
    assert captured["save_path"] == output_root / "scene" / "R2R_val_unseen_1.png"


def test_render_prediction_paths_preserves_requested_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_root = tmp_path / "predictions"
    ground_truth_root = tmp_path / "ground-truth"
    output_root = tmp_path / "output"
    prediction_paths = [
        prediction_root / "scene" / f"episode_{index}.npz" for index in range(3)
    ]
    for prediction_path in prediction_paths:
        ground_truth_path = (
            ground_truth_root / "raster" / "scene" / prediction_path.name
        )
        boxes_path = ground_truth_root / "boxes" / "scene" / prediction_path.name
        for path in (prediction_path, ground_truth_path, boxes_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    monkeypatch.setattr(
        "prior.analyze.batch_vis._load_prediction_map",
        lambda path: (BaseGridMap(), None),
    )
    monkeypatch.setattr(BaseGridMap, "load", staticmethod(lambda path: BaseGridMap()))
    monkeypatch.setattr(
        "prior.analyze.batch_vis.RelevantSemanticBoxes.load",
        lambda path: SimpleNamespace(
            instruction="Walk to the chair.",
            ground_truth_trajectory=[(1.0, 2.0)],
            trajectory_keypoints=[(1.0, 2.0)] * 5,
            start_direction_vector=(1.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        BaseGridMap,
        "visualize_comparison",
        lambda self, ground_truth_map, save_path, **kwargs: None,
    )

    rendered = render_prediction_paths(
        [prediction_paths[2], prediction_paths[0]],
        ground_truth_root,
        output_root,
    )

    assert rendered == (
        output_root / "scene" / "episode_2.png",
        output_root / "scene" / "episode_0.png",
    )


def test_render_prediction_paths_rejects_missing_requested_prediction(
    tmp_path: Path,
) -> None:
    missing_prediction_path = tmp_path / "predictions" / "scene" / "episode.npz"
    ground_truth_root = tmp_path / "ground-truth"
    (ground_truth_root / "raster").mkdir(parents=True)
    (ground_truth_root / "boxes").mkdir()

    with pytest.raises(FileNotFoundError, match="Missing prediction"):
        render_prediction_paths(
            [missing_prediction_path],
            ground_truth_root,
            tmp_path / "output",
        )


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


def test_load_prediction_map_detects_waypoint_capability(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.npz"
    boxes_path = tmp_path / "boxes.npz"
    BaseGridMap().save(grid_path)
    boxes_map = CognitiveGridMap()
    boxes_map.trajectory_keypoints = [
        (1.0, 2.0),
        (3.0, 4.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    boxes_map.start_direction_vector = (1.0, 0.0)
    boxes_map.save(boxes_path)

    _, grid_keypoints = _load_prediction_map(grid_path)
    loaded_boxes, boxes_keypoints = _load_prediction_map(boxes_path)

    assert grid_keypoints is None
    assert isinstance(loaded_boxes, CognitiveGridMap)
    assert boxes_keypoints == boxes_map.trajectory_keypoints


@pytest.mark.parametrize(
    ("trajectory_keypoints", "start_direction_vector", "message"),
    [
        (np.zeros((4, 2), dtype=np.float32), (1.0, 0.0), "must have shape"),
        (np.zeros((5, 2), dtype=np.bool_), (1.0, 0.0), "real numeric values"),
        (
            np.full((5, 2), np.nan, dtype=np.float32),
            (1.0, 0.0),
            "only finite values",
        ),
        (np.zeros((5, 2), dtype=np.float32), (1.0,), "must have shape"),
    ],
)
def test_load_prediction_map_rejects_malformed_waypoints(
    tmp_path: Path,
    trajectory_keypoints: np.ndarray,
    start_direction_vector: tuple[float, ...],
    message: str,
) -> None:
    path = tmp_path / "malformed.npz"
    np.savez_compressed(
        path,
        grid=BaseGridMap().grid,
        range_y=np.asarray([None, None], dtype=object),
        trajectory_keypoints=trajectory_keypoints,
        start_direction_vector=np.asarray(start_direction_vector, dtype=np.float32),
    )

    with pytest.raises(ValueError, match=message):
        _load_prediction_map(path)


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
