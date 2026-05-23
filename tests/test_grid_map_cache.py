from pathlib import Path
import numpy as np

from prior.grid_map import CognitiveGridMap, GroundTruthGridMap


def _map_with_value(value: float) -> GroundTruthGridMap:
    grid_map = GroundTruthGridMap()
    grid_map.grid[0, 0, 0] = value
    grid_map.offset_x = value + 1.0
    grid_map.offset_z = value + 2.0
    grid_map.range_y = [None, value]
    return grid_map


def test_ground_truth_grid_map_load_returns_ground_truth_grid_map(tmp_path: Path):
    grid_map = _map_with_value(1.0)
    path = tmp_path / "map.npz"
    grid_map.save(path)

    loaded = GroundTruthGridMap.load(path)

    assert isinstance(loaded, GroundTruthGridMap)
    np.testing.assert_array_equal(loaded.grid, grid_map.grid)
    assert loaded.offset_x == grid_map.offset_x
    assert loaded.offset_z == grid_map.offset_z
    assert loaded.range_y == grid_map.range_y


def test_cognitive_grid_map_load_matches_base_classmethod_contract(tmp_path: Path):
    grid_map = CognitiveGridMap()
    grid_map.grid[0, 0, 0] = 0.5
    grid_map.offset_x = 1.0
    grid_map.offset_z = 2.0
    grid_map.range_y = [None, 3.0]
    grid_map.positions = [(1.0, 2.0)]
    grid_map.direction_vectors = [(0.0, 1.0)]
    grid_map.start_direction_vector = (0.0, 1.0)
    path = tmp_path / "cognitive.npz"
    grid_map.save(path)

    loaded = CognitiveGridMap.load(path)

    assert isinstance(loaded, CognitiveGridMap)
    np.testing.assert_array_equal(loaded.grid, grid_map.grid)
    assert loaded.offset_x == grid_map.offset_x
    assert loaded.offset_z == grid_map.offset_z
    assert loaded.range_y == grid_map.range_y
    assert loaded.positions == grid_map.positions
    assert loaded.direction_vectors == grid_map.direction_vectors
    assert loaded.start_direction_vector == grid_map.start_direction_vector


def test_ground_truth_grid_map_has_no_scene_construction_api():
    assert not hasattr(GroundTruthGridMap, "from_scene")
    assert not hasattr(GroundTruthGridMap, "from_scene_id")
