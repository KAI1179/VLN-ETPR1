from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prior import blur_cognitive_maps


def test_blur_cognitive_map_grid_downsamples_then_upsamples() -> None:
    grid = np.zeros((2, 4, 4), dtype=np.float32)
    grid[0, 0, 1] = 0.25
    grid[0, 1, 0] = 1.0
    grid[1, 3, 3] = 0.6

    blurred = blur_cognitive_maps.blur_cognitive_map_grid(grid, scale=2)

    assert blurred.shape == grid.shape
    np.testing.assert_array_equal(
        blurred[0],
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert blurred[1, 2, 2] == pytest.approx(0.6)
    assert blurred[1, 3, 3] == pytest.approx(0.6)


def test_blur_cognitive_map_file_preserves_direction5_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    target = tmp_path / "target.npz"
    grid = np.zeros((37, 4, 4), dtype=np.float32)
    grid[3, 1, 1] = 1.0
    np.savez_compressed(
        source,
        grid=grid,
        range_y=np.asarray([None, 3.0], dtype=object),
        direction_vectors=np.asarray([[1.0, 0.0]] * 5, dtype=np.float32),
        start_direction_vector=np.asarray([0.0, 1.0], dtype=np.float32),
        start_position=np.asarray([4.0, 5.0], dtype=np.float32),
    )

    wrote = blur_cognitive_maps.blur_cognitive_map_file(source, target, scale=2)

    assert wrote is True
    data = np.load(target, allow_pickle=True)
    assert set(data.files) == {
        "grid",
        "range_y",
        "direction_vectors",
        "start_direction_vector",
        "start_position",
    }
    assert data["grid"].shape == (37, 4, 4)
    assert data["grid"][3, 0, 0] == pytest.approx(1.0)
    assert data["grid"][3, 1, 1] == pytest.approx(1.0)
    assert data["range_y"].tolist() == [None, 3.0]
    np.testing.assert_array_equal(
        data["direction_vectors"],
        np.asarray([[1.0, 0.0]] * 5, dtype=np.float32),
    )


def test_transform_namespace_copies_boxes_and_skips_existing(tmp_path: Path) -> None:
    source_root = tmp_path / "cognitive_maps" / "gt.legacy.r1p5.direction5.v1"
    raster_path = source_root / "raster" / "scene-a" / "item-1.npz"
    boxes_path = source_root / "boxes" / "scene-a" / "item-1.npz"
    raster_path.parent.mkdir(parents=True)
    boxes_path.parent.mkdir(parents=True)
    grid = np.zeros((37, 4, 4), dtype=np.float32)
    grid[1, 2, 2] = 1.0
    np.savez_compressed(raster_path, grid=grid)
    boxes_path.write_text(json.dumps({"box": 1}), encoding="utf-8")

    summary = blur_cognitive_maps.transform_cognitive_map_namespace(
        cache_root=tmp_path / "cognitive_maps",
        source_namespace="gt.legacy.r1p5.direction5.v1",
        target_namespace="gt.legacy.r1p5.direction5.blurred.v1",
        scale=2,
    )

    assert summary.raster_written == 1
    assert summary.boxes_written == 1
    target_root = tmp_path / "cognitive_maps" / "gt.legacy.r1p5.direction5.blurred.v1"
    assert (target_root / "raster" / "scene-a" / "item-1.npz").is_file()
    assert (target_root / "boxes" / "scene-a" / "item-1.npz").read_text(
        encoding="utf-8"
    ) == json.dumps({"box": 1})

    skipped = blur_cognitive_maps.transform_cognitive_map_namespace(
        cache_root=tmp_path / "cognitive_maps",
        source_namespace="gt.legacy.r1p5.direction5.v1",
        target_namespace="gt.legacy.r1p5.direction5.blurred.v1",
        scale=2,
    )

    assert skipped.raster_written == 0
    assert skipped.raster_skipped == 1
    assert skipped.boxes_written == 0
    assert skipped.boxes_skipped == 1
