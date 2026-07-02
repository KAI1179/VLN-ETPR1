from pathlib import Path
import numpy as np
import pytest
import torch

from prior.grid_map import CognitiveGridMap, GroundTruthGridMap
from vlnce_baselines.models.etp_prior_gt import map_utils


def _map_with_value(value: float) -> GroundTruthGridMap:
    grid_map = GroundTruthGridMap()
    grid_map.grid[0, 0, 0] = value
    grid_map.range_y = [None, value]
    return grid_map


def test_ground_truth_grid_map_load_returns_ground_truth_grid_map(tmp_path: Path):
    grid_map = _map_with_value(1.0)
    path = tmp_path / "map.npz"
    grid_map.save(path)

    loaded = GroundTruthGridMap.load(path)

    assert isinstance(loaded, GroundTruthGridMap)
    np.testing.assert_array_equal(loaded.grid, grid_map.grid)
    assert loaded.range_y == grid_map.range_y


def test_cognitive_grid_map_load_matches_base_classmethod_contract(tmp_path: Path):
    grid_map = CognitiveGridMap()
    grid_map.grid[0, 0, 0] = 0.5
    grid_map.range_y = [None, 3.0]
    grid_map.trajectory_keypoints = [
        (1.0, 2.0),
        (2.0, 3.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    grid_map.start_direction_vector = (0.0, 1.0)
    path = tmp_path / "cognitive.npz"
    grid_map.save(path)

    loaded = CognitiveGridMap.load(path)

    assert isinstance(loaded, CognitiveGridMap)
    np.testing.assert_array_equal(loaded.grid, grid_map.grid)
    assert loaded.range_y == grid_map.range_y
    assert loaded.trajectory_keypoints == grid_map.trajectory_keypoints
    assert loaded.start_direction_vector == grid_map.start_direction_vector
    assert not hasattr(loaded, "positions")
    assert not hasattr(loaded, "direction_vectors")


def test_cognitive_grid_map_rejects_legacy_reference_path_cache(tmp_path: Path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        grid=CognitiveGridMap().grid,
        range_y=np.asarray([None, None], dtype=object),
        reference_path=np.asarray([(1.0, 2.0)], dtype=np.float32),
        start_direction_vector=np.asarray((0.0, 1.0), dtype=np.float32),
    )

    with pytest.raises(KeyError, match="trajectory_keypoints"):
        CognitiveGridMap.load(path)


def test_cognitive_grid_map_load_rejects_wrong_keypoint_shape(tmp_path: Path):
    path = tmp_path / "wrong-shape.npz"
    np.savez_compressed(
        path,
        grid=CognitiveGridMap().grid,
        range_y=np.asarray([None, None], dtype=object),
        trajectory_keypoints=np.asarray([(1.0, 2.0)], dtype=np.float32),
        start_direction_vector=np.asarray((0.0, 1.0), dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match=r"trajectory_keypoints must have shape \(5, 2\), got \(1, 2\)",
    ):
        CognitiveGridMap.load(path)


def test_cognitive_grid_map_save_rejects_wrong_keypoint_shape(tmp_path: Path):
    grid_map = CognitiveGridMap()
    grid_map.trajectory_keypoints = [(1.0, 2.0)]
    path = tmp_path / "wrong-shape.npz"

    with pytest.raises(
        ValueError,
        match=r"trajectory_keypoints must have shape \(5, 2\), got \(1, 2\)",
    ):
        grid_map.save(path)

    assert not path.exists()


def test_cognitive_grid_map_visualize_trims_zero_suffix_but_keeps_leading_origin(
    tmp_path: Path, monkeypatch
):
    grid_map = CognitiveGridMap()
    grid_map.trajectory_keypoints = [
        (0.0, 0.0),
        (1.0, 2.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    captured = {}

    def fake_visualize(*args, **kwargs):
        captured["positions"] = kwargs["positions"]

    monkeypatch.setattr("prior.grid_map._visualize.visualize", fake_visualize)

    grid_map.visualize(tmp_path / "map.png")

    assert captured["positions"] == [(0.0, 0.0), (2.0, 4.0)]


def test_cached_cognitive_map_to_tensors_loads_by_scene_and_cache_id(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cognitive_maps"
    scene_dir = cache_dir / "raster" / "scene"
    boxes_scene_dir = cache_dir / "boxes" / "scene"
    scene_dir.mkdir(parents=True)
    boxes_scene_dir.mkdir(parents=True)
    grid_map = CognitiveGridMap()
    grid_map.grid[0, 2, 3] = 1.0
    grid_map.trajectory_keypoints = [
        (1.0, 2.0),
        (2.0, 2.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    grid_map.start_direction_vector = (0.0, 1.0)
    grid_map.save(scene_dir / "R2R_train_1.npz")
    (boxes_scene_dir / "R2R_train_1.npz").touch()

    monkeypatch.setattr(map_utils, "VLNCE_COGNITIVE_MAP_DIR", cache_dir)

    tensors = map_utils.cached_cognitive_map_to_tensors(
        "scene.glb",
        "R2R_train_1",
    )

    assert tensors["grid"].shape == (
        map_utils.NUM_MAP_CATEGORIES,
        map_utils.SIZE,
        map_utils.SIZE,
    )
    assert tensors["grid"][0, 2, 3] == 1.0
    assert torch.equal(tensors["trajectory_keypoints"][0], torch.tensor([2.0, 4.0]))
    assert torch.equal(tensors["start_position"], torch.tensor([2.0, 4.0]))
    assert torch.equal(tensors["start_direction_vector"], torch.tensor([0.0, 1.0]))


def test_cached_cognitive_map_to_tensors_fails_for_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(map_utils, "VLNCE_COGNITIVE_MAP_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing cached cognitive map"):
        map_utils.cached_cognitive_map_to_tensors("scene.glb", "R2R_train_1")


def test_available_vlnce_cognitive_map_episode_ids_skips_missing(
    tmp_path, monkeypatch, capsys
):
    entries = [
        type(
            "Entry",
            (),
            {
                "scene_id": "scene",
                "episode_id": 1,
                "unique_id": "R2R_train_1",
            },
        )(),
        type(
            "Entry",
            (),
            {
                "scene_id": "scene",
                "episode_id": 2,
                "unique_id": "R2R_train_2",
            },
        )(),
    ]
    monkeypatch.setattr(
        map_utils.VLNCEEpisodeEntry,
        "iter_from",
        staticmethod(lambda dataset, splits: iter(entries)),
    )
    raster_scene_dir = tmp_path / "raster" / "scene"
    boxes_scene_dir = tmp_path / "boxes" / "scene"
    raster_scene_dir.mkdir(parents=True)
    boxes_scene_dir.mkdir(parents=True)
    (raster_scene_dir / "R2R_train_2.npz").touch()
    (boxes_scene_dir / "R2R_train_2.npz").touch()

    allowed = map_utils.available_vlnce_cognitive_map_episode_ids(
        "R2R",
        "train",
        tmp_path,
    )

    assert allowed == ["2"]
    assert (
        "finetuning_cognitive_maps: available=1 skipped_missing=1"
        in capsys.readouterr().out
    )


def test_tensor_rotation_preserves_zero_padded_trajectory_keypoints():
    tensors = {
        "grid": torch.zeros(1, 100, 100),
        "trajectory_keypoints": torch.tensor(
            [
                [2.0, 4.0],
                [6.0, 8.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        "start_position": torch.tensor([2.0, 4.0]),
        "start_direction_vector": torch.tensor([0.0, 1.0]),
    }

    rotated = map_utils.rotate_cognitive_map_tensors_by_right_angle(tensors, 1)

    assert torch.equal(
        rotated["trajectory_keypoints"],
        torch.tensor(
            [
                [95.0, 2.0],
                [91.0, 6.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(rotated["start_position"], torch.tensor([95.0, 2.0]))


def test_tensor_rotation_rotates_leading_origin_but_preserves_zero_suffix():
    tensors = {
        "grid": torch.zeros(1, 100, 100),
        "trajectory_keypoints": torch.tensor(
            [
                [0.0, 0.0],
                [6.0, 8.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        "start_position": torch.tensor([0.0, 0.0]),
        "start_direction_vector": torch.tensor([0.0, 1.0]),
    }

    rotated = map_utils.rotate_cognitive_map_tensors_by_right_angle(tensors, 1)

    assert torch.equal(
        rotated["trajectory_keypoints"],
        torch.tensor(
            [
                [99.0, 0.0],
                [91.0, 6.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(rotated["start_position"], torch.tensor([99.0, 0.0]))


def test_tensor_rotation_uses_max_valid_grid_indices():
    points = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 99.0],
            [99.0, 99.0],
            [99.0, 0.0],
        ]
    )

    rotated = map_utils._rotate_grid_points_by_right_angle(
        points, turns=1, rows=100, cols=100
    )

    assert torch.equal(
        rotated,
        torch.tensor(
            [
                [99.0, 0.0],
                [0.0, 0.0],
                [0.0, 99.0],
                [99.0, 99.0],
            ]
        ),
    )


def test_ground_truth_grid_map_has_no_scene_construction_api():
    assert not hasattr(GroundTruthGridMap, "from_scene")
    assert not hasattr(GroundTruthGridMap, "from_scene_id")
