from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prior.grid_map import CognitiveGridMap
from prior.llm_grid_samples import (
    SampleArgs,
    analyze_grid_sample,
    main,
    serialize_grid_target,
)


def _save_cognitive_map(path: Path, grid: np.ndarray) -> None:
    grid_map = CognitiveGridMap()
    grid_map.grid = grid.astype(np.float32)
    grid_map.trajectory_keypoints = [(0.0, 0.0)] * 5
    grid_map.start_direction_vector = (0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_map.save(path)


def test_serialize_grid_target_orders_nonzero_cells_and_keeps_soft_values() -> None:
    grid = np.zeros((3, 4, 4), dtype=np.float32)
    grid[2, 1, 0] = 0.25
    grid[0, 3, 2] = 1.0
    grid[0, 1, 2] = 0.5

    text = serialize_grid_target(grid)

    assert text == "g 0 1 2 0.5 ; g 0 3 2 ; g 2 1 0 0.25"


def test_serialize_grid_target_downsamples_by_max_pooling() -> None:
    grid = np.zeros((1, 4, 4), dtype=np.float32)
    grid[0, 1, 1] = 0.5
    grid[0, 2, 3] = 1.0

    text = serialize_grid_target(grid, scale=2)

    assert text == "g 0 0 0 0.5 ; g 0 1 1"


def test_analyze_grid_sample_reports_stats_with_whitespace_tokens(tmp_path: Path) -> None:
    grid = np.zeros((2, 4, 4), dtype=np.float32)
    grid[0, 0, 0] = 1.0
    grid[1, 3, 3] = 0.25
    path = tmp_path / "map.npz"
    _save_cognitive_map(path, grid)

    sample = analyze_grid_sample(path, scale=1)

    assert sample["target_text"] == "g 0 0 0 ; g 1 3 3 0.25"
    assert sample["stats"] == {
        "text_length": 22,
        "token_count": 10,
        "positive_cell_count": 2,
        "category_count": 2,
    }


def test_cli_samples_namespace_and_writes_manifest_samples_summary(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "cognitive_maps"
    grid = np.zeros((1, 4, 4), dtype=np.float32)
    grid[0, 0, 0] = 1.0
    _save_cognitive_map(
        root / "gt.bbox.r1p5.path5.v1" / "raster" / "scene" / "b.npz",
        grid,
    )
    _save_cognitive_map(
        root / "gt.bbox.r1p5.path5.v1" / "raster" / "scene" / "a.npz",
        grid,
    )
    monkeypatch.setattr(
        SampleArgs,
        "parse_args",
        lambda self, argv=None: self.from_dict(
            {
                "cache_root": root,
                "namespace": "gt.bbox.r1p5.path5.v1",
                "count": 1,
                "output_root": tmp_path / "runs",
                "scale": 1,
                "seed": 0,
                "tokenizer_path": None,
            }
        ),
    )

    main([])

    [output_dir] = (tmp_path / "runs").iterdir()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    samples = [
        json.loads(line)
        for line in (output_dir / "samples.jsonl").read_text().splitlines()
    ]
    summary = json.loads((output_dir / "summary.json").read_text())

    assert manifest["namespace"] == "gt.bbox.r1p5.path5.v1"
    assert manifest["scale"] == 1
    assert samples[0]["sample_id"] in {"a", "b"}
    assert samples[0]["target_text"] == "g 0 0 0"
    assert summary["sample_count"] == 1
    assert summary["stats"]["positive_cell_count"]["max"] == 1
