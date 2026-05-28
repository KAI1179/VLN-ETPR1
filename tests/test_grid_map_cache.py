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
    grid_map.reference_path = [(1.0, 2.0), (2.0, 3.0)]
    grid_map.start_direction_vector = (0.0, 1.0)
    path = tmp_path / "cognitive.npz"
    grid_map.save(path)

    loaded = CognitiveGridMap.load(path)

    assert isinstance(loaded, CognitiveGridMap)
    np.testing.assert_array_equal(loaded.grid, grid_map.grid)
    assert loaded.range_y == grid_map.range_y
    assert loaded.reference_path == grid_map.reference_path
    assert loaded.start_direction_vector == grid_map.start_direction_vector
    assert not hasattr(loaded, "positions")
    assert not hasattr(loaded, "direction_vectors")


def test_cached_cognitive_map_to_tensors_loads_by_scene_and_cache_id(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cognitive_maps"
    scene_dir = cache_dir / "scene"
    scene_dir.mkdir(parents=True)
    grid_map = CognitiveGridMap()
    grid_map.grid[0, 2, 3] = 1.0
    grid_map.reference_path = [(1.0, 2.0)]
    grid_map.start_direction_vector = (0.0, 1.0)
    grid_map.save(scene_dir / "R2R_train_1.npz")

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
    assert torch.equal(tensors["reference_paths"][0], torch.tensor([2.0, 4.0]))
    assert torch.equal(tensors["start_direction_vector"], torch.tensor([0.0, 1.0]))


def test_cached_cognitive_map_to_tensors_fails_for_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(map_utils, "VLNCE_COGNITIVE_MAP_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing cached cognitive map"):
        map_utils.cached_cognitive_map_to_tensors("scene.glb", "R2R_train_1")


def test_ground_truth_grid_map_has_no_scene_construction_api():
    assert not hasattr(GroundTruthGridMap, "from_scene")
    assert not hasattr(GroundTruthGridMap, "from_scene_id")
