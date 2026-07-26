from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from prior.analyze.d2026_07_26.category_metrics import _CategoryCounts
from prior.analyze.d2026_07_26.presentation_visualizations import (
    EpisodeMetrics,
    _error_map,
    _observed_contour,
    _route_overlay,
    select_ordinary_examples,
)


def test_category_counts_pool_presence_and_category_cells() -> None:
    counts = _CategoryCounts(
        group="object",
        category_id=1,
        channel_id=1,
        category_name="chair",
        broad_environment_label=False,
    )
    counts.update(
        target=np.asarray([[True, True], [False, False]]),
        predicted=np.asarray([[True, False], [True, False]]),
        scene_id="scene-a",
        route_id=1,
        mentioned=True,
        schema_valid=True,
        missing=False,
    )
    counts.update(
        target=np.asarray([[False, False], [False, False]]),
        predicted=np.asarray([[True, False], [False, False]]),
        scene_id="scene-b",
        route_id=2,
        mentioned=False,
        schema_valid=False,
        missing=False,
    )

    row = counts.row(route_count=2, scene_count=2)

    assert row["presence_precision"] == 0.5
    assert row["presence_recall"] == 1.0
    assert row["presence_f1"] == 2 / 3
    assert row["spatial_precision"] == 1 / 3
    assert row["spatial_recall"] == 0.5
    assert row["spatial_iou"] == 0.25
    assert row["target_route_count"] == 1
    assert row["target_scene_count"] == 1
    assert row["schema_invalid_example_count"] == 1


def _metrics(
    example_id: str,
    *,
    iou: float,
    target_cells: int,
    schema_valid: float = 1.0,
) -> EpisodeMetrics:
    return EpisodeMetrics(
        scene_id=f"scene-{example_id}",
        example_id=example_id,
        instruction="instruction",
        raster_iou=iou,
        category_f1=0.5,
        target_cell_count=target_cells,
        raw={"schema_valid": str(schema_valid)},
    )


def test_ordinary_selection_uses_support_rich_zero_iou_failure(
    tmp_path: Path,
) -> None:
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps({
            "splits": {
                "val_unseen": {
                    "representatives": {
                        "median": {"example_id": "median"},
                        "largest_category_spatial_gap": {"example_id": "gap"},
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    rows = {
        "median": _metrics("median", iou=0.2, target_cells=10),
        "gap": _metrics("gap", iou=0.01, target_cells=20),
        "small-zero": _metrics("small-zero", iou=0.0, target_cells=30),
        "large-zero": _metrics("large-zero", iou=0.0, target_cells=100),
        "invalid": _metrics("invalid", iou=0.0, target_cells=200, schema_valid=0.0),
    }

    selected = select_ordinary_examples(diagnostics_path, rows)

    assert [(role, metrics.example_id) for role, metrics in selected] == [
        ("median", "median"),
        ("category_spatial_gap", "gap"),
        ("support_rich_failure", "large-zero"),
    ]


def test_unobserved_error_map_distinguishes_substitution() -> None:
    target = np.zeros((2, 2, 2), dtype=np.bool_)
    predicted = np.zeros_like(target)
    observed = np.zeros((2, 2), dtype=np.bool_)
    target[0, 0, 0] = True
    predicted[0, 0, 0] = True
    predicted[0, 0, 1] = True
    target[0, 1, 0] = True
    target[0, 1, 1] = True
    predicted[1, 1, 1] = True

    result = _error_map(predicted, target, observed)

    assert result.tolist() == [[1, 2], [3, 4]]


def test_observation_contour_uses_cell_centers_without_snapping_route() -> None:
    axis = Mock()
    observed = np.zeros((2, 3), dtype=np.bool_)

    _observed_contour(axis, observed)
    _route_overlay(
        axis,
        SimpleNamespace(ground_truth_trajectory=((0.1, 0.1), (1.1, 1.1))),
        scale=2,
    )

    contour_x, contour_y, _ = axis.contour.call_args.args
    assert contour_x.tolist() == [0.5, 1.5, 2.5]
    assert contour_y.tolist() == [0.5, 1.5]
    route_cols, route_rows = axis.plot.call_args.args
    assert route_cols == (0.1, 1.1)
    assert route_rows == (0.1, 1.1)
