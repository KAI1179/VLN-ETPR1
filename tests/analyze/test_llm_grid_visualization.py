from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from prior.analyze.d2026_07_25.visualize_llm_grid import (
    VisualizationArgs,
    run_visualization,
    select_common_examples,
    write_comparison_sheet,
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)


def _write_raster(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_select_common_examples_keeps_sorted_historical_intersection(
    tmp_path: Path,
) -> None:
    historical_root = tmp_path / "historical"
    raster_roots = {epoch: tmp_path / f"raster-{epoch}" for epoch in (1, 2, 5, 10)}
    _write_png(historical_root / "scene-b" / "R2R_val_unseen_2.png")
    _write_png(historical_root / "scene-a" / "R2R_val_unseen_1.png")
    _write_raster(raster_roots[1] / "scene-a" / "R2R_val_unseen_1.npz")
    for raster_root in raster_roots.values():
        _write_raster(raster_root / "scene-b" / "R2R_val_unseen_2.npz")

    examples = select_common_examples(historical_root, raster_roots)

    assert [(example.scene_id, example.example_id) for example in examples] == [
        ("scene-b", "R2R_val_unseen_2"),
    ]
    assert examples[0].historical_path == (
        historical_root / "scene-b" / "R2R_val_unseen_2.png"
    )
    assert dict(examples[0].raster_paths) == {
        epoch: raster_roots[epoch] / "scene-b" / "R2R_val_unseen_2.npz"
        for epoch in (1, 2, 5, 10)
    }


def test_select_common_examples_rejects_duplicate_historical_identity(
    tmp_path: Path,
) -> None:
    historical_root = tmp_path / "historical"
    _write_png(historical_root / "scene" / "R2R_val_unseen_1.png")
    _write_png(historical_root / "archive" / "scene" / "R2R_val_unseen_1.png")

    with pytest.raises(ValueError, match="duplicate historical identity"):
        select_common_examples(historical_root, {})


def test_select_common_examples_rejects_unexpected_common_count(
    tmp_path: Path,
) -> None:
    historical_root = tmp_path / "historical"
    _write_png(historical_root / "scene" / "R2R_val_unseen_1.png")

    with pytest.raises(ValueError, match="expected 2 common examples, got 1"):
        select_common_examples(historical_root, {}, expected_count=2)


def test_write_comparison_sheet_arranges_five_panels_in_readable_grid(
    tmp_path: Path,
) -> None:
    columns = []
    for index in range(5):
        path = tmp_path / "columns" / f"column-{index}.png"
        _write_png(path)
        columns.append((f"Column {index}", path))
    output_path = tmp_path / "sheets" / "comparison.png"

    write_comparison_sheet(columns, output_path)

    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(output_path) as sheet:
        assert sheet.size == (16, 114)


@pytest.mark.parametrize(
    ("overlapping_root", "historical_relative", "output_relative"),
    (
        ("output_root", "historical", "historical"),
        ("output_root", "historical/nested", "historical"),
        ("output_root", "historical", "historical/nested"),
        ("docs_image_root", "historical", "historical"),
        ("docs_image_root", "historical/nested", "historical"),
        ("docs_image_root", "historical", "historical/nested"),
    ),
)
def test_run_visualization_rejects_historical_output_overlap_before_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overlapping_root: str,
    historical_relative: str,
    output_relative: str,
) -> None:
    args = VisualizationArgs()
    args.historical_root = tmp_path / historical_relative
    args.output_root = tmp_path / "separate-output"
    args.docs_image_root = tmp_path / "separate-docs"
    setattr(args, overlapping_root, tmp_path / output_relative)

    def fail_if_called(*args: object, **kwargs: object) -> tuple[Path, ...]:
        raise AssertionError("renderer called before overlap rejection")

    monkeypatch.setattr(
        "prior.analyze.d2026_07_25.visualize_llm_grid.render_prediction_paths",
        fail_if_called,
    )

    with pytest.raises(ValueError, match="historical root overlaps"):
        run_visualization(args)
