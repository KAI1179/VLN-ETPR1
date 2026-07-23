from __future__ import annotations

import csv
from pathlib import Path

import pytest

from vlnce_baselines.models.etp_llm.llm_grid_paired_bootstrap import (
    compare_episode_csvs,
)


FIELDS = [
    "dataset",
    "split",
    "scene_id",
    "example_id",
    "category_aware_raster_iou",
    "cell_f1",
    "direction_vector_cosine",
    "direction_vector_cosine_support",
    "object_category_predicted_count",
    "object_category_target_count",
    "object_category_true_positive_count",
    "region_category_predicted_count",
    "region_category_target_count",
    "region_category_true_positive_count",
    "schema_valid",
]


def _write_episode_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(example_id: str, scene_id: str, value: float) -> dict[str, object]:
    return {
        "dataset": "R2R",
        "split": "val_unseen",
        "scene_id": scene_id,
        "example_id": example_id,
        "category_aware_raster_iou": value,
        "cell_f1": value,
        "direction_vector_cosine": value,
        "direction_vector_cosine_support": 1,
        "object_category_predicted_count": 1,
        "object_category_target_count": 1,
        "object_category_true_positive_count": value,
        "region_category_predicted_count": 1,
        "region_category_target_count": 1,
        "region_category_true_positive_count": value,
        "schema_valid": value,
    }


def test_paired_scene_bootstrap_reports_exact_equal_models(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    rows = [_row("a", "scene-a", 0.2), _row("b", "scene-b", 0.8)]
    _write_episode_csv(baseline, rows)
    _write_episode_csv(candidate, rows)

    result = compare_episode_csvs(baseline, candidate, repetitions=100, seed=42)

    metrics = result["splits"]["val_unseen"]["metrics"]
    assert metrics["category_aware_raster_iou"]["delta"] == 0.0
    assert metrics["category_aware_raster_iou"]["ci95"] == [0.0, 0.0]
    assert metrics["object_category_f1"]["delta"] == 0.0


def test_paired_scene_bootstrap_rejects_scene_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_episode_csv(baseline, [_row("a", "scene-a", 0.2)])
    _write_episode_csv(candidate, [_row("a", "scene-b", 0.3)])

    with pytest.raises(ValueError, match="scene mismatch"):
        compare_episode_csvs(baseline, candidate, repetitions=10, seed=42)
