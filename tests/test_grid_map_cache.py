from pathlib import Path
import numpy as np

from prior.grid_map import CognitiveGridMap, GroundTruthGridMap
from prior.grid_map import _construct


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


def test_construct_grid_maps_from_scene_id_uses_cached_levels_first(
    tmp_path: Path, monkeypatch
):
    scene_id = "scene"
    cache_dir = tmp_path / scene_id
    cache_dir.mkdir()
    _map_with_value(1.0).save(cache_dir / "0.npz")
    _map_with_value(0.0).save(cache_dir / "1.npz")

    def fail_load(*args, **kwargs):
        raise AssertionError("scene should not load when cache exists")

    _construct.construct_grid_maps_from_scene_id.cache_clear()
    monkeypatch.setattr(_construct, "SEMANTIC_MAP_DIR", tmp_path)
    monkeypatch.setattr(_construct.SemanticScene, "load_mp3d_house", fail_load)

    maps = _construct.construct_grid_maps_from_scene_id(scene_id)

    assert [float(m.grid[0, 0, 0]) for m in maps] == [1.0, 0.0]
    assert all(isinstance(m, GroundTruthGridMap) for m in maps)


def test_construct_grid_maps_from_scene_id_saves_constructed_levels(
    tmp_path: Path, monkeypatch
):
    scene_id = "scene"
    constructed_maps = [_map_with_value(1.0), _map_with_value(0.0)]

    _construct.construct_grid_maps_from_scene_id.cache_clear()
    monkeypatch.setattr(_construct, "SEMANTIC_MAP_DIR", tmp_path)
    monkeypatch.setattr(
        _construct.SemanticScene, "load_mp3d_house", lambda *a, **k: None
    )
    monkeypatch.setattr(
        _construct,
        "construct_grid_maps_from_scene",
        lambda semantic_scene: constructed_maps,
    )

    maps = _construct.construct_grid_maps_from_scene_id(scene_id)

    assert maps == constructed_maps
    assert (tmp_path / scene_id / "0.npz").exists()
    assert (tmp_path / scene_id / "1.npz").exists()
    cached_maps = [
        GroundTruthGridMap.load(tmp_path / scene_id / f"{level}.npz")
        for level in range(2)
    ]
    assert [float(m.grid[0, 0, 0]) for m in cached_maps] == [1.0, 0.0]
