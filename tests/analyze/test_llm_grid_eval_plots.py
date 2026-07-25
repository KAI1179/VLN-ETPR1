from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from prior.analyze.llm_grid_eval_records import EpisodeRecord
from prior.analyze.llm_grid_eval_plots import (
    plot_category_counts,
    plot_category_f1,
    plot_iou_histograms,
    plot_iou_survival,
)
from prior.analyze.d2026_07_25.plot_llm_grid_sanity import (
    SanityPlotArgs,
    _run_analysis,
    epoch_episode_paths,
)


def _record(*, epoch: int = 1, split: str = "val_seen", example_id: str = "a") -> EpisodeRecord:
    return EpisodeRecord(
        epoch=epoch,
        split=split,
        scene_id=f"scene-{split}",
        example_id=example_id,
        iou=0.5 if split == "val_seen" else 0.75,
        schema_valid=True,
        object_target=3,
        object_predicted=4,
        object_true_positive=2,
        region_target=2,
        region_predicted=3,
        region_true_positive=1,
        mentioned_object_predicted=2,
        unmentioned_object_predicted=2,
        mentioned_region_predicted=1,
        unmentioned_region_predicted=2,
    )


def _records_by_epoch() -> Mapping[int, tuple[EpisodeRecord, ...]]:
    records = (
        _record(split="val_seen", example_id="seen"),
        _record(split="val_unseen", example_id="unseen"),
    )
    return {
        epoch: tuple(replace(record, epoch=epoch) for record in records)
        for epoch in (1, 2, 5, 10)
    }


def test_plotters_write_png_images(tmp_path: Path) -> None:
    records_by_epoch = _records_by_epoch()
    output_paths = {
        "histograms": tmp_path / "plots" / "iou-histograms.png",
        "survival": tmp_path / "plots" / "iou-survival.png",
        "f1": tmp_path / "plots" / "category-f1.png",
        "object_counts": tmp_path / "plots" / "object-counts.png",
        "region_counts": tmp_path / "plots" / "region-counts.png",
    }

    plot_iou_histograms(records_by_epoch, output_paths["histograms"])
    plot_iou_survival(
        records_by_epoch,
        output_paths["survival"],
        thresholds=(0.0, 0.5, 1.0),
    )
    plot_category_f1(records_by_epoch, output_paths["f1"])
    plot_category_counts(records_by_epoch, output_paths["object_counts"], kind="object")
    plot_category_counts(records_by_epoch, output_paths["region_counts"], kind="region")

    for output_path in output_paths.values():
        assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plotters_reject_empty_record_mapping(tmp_path: Path) -> None:
    output_path = tmp_path / "plot.png"

    with pytest.raises(ValueError, match="records_by_epoch must not be empty"):
        plot_iou_histograms({}, output_path)
    with pytest.raises(ValueError, match="records_by_epoch must not be empty"):
        plot_iou_survival({}, output_path, thresholds=(0.0,))
    with pytest.raises(ValueError, match="records_by_epoch must not be empty"):
        plot_category_f1({}, output_path)
    with pytest.raises(ValueError, match="records_by_epoch must not be empty"):
        plot_category_counts({}, output_path, kind="object")


def test_plot_category_counts_rejects_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kind must be 'object' or 'region'"):
        plot_category_counts(_records_by_epoch(), tmp_path / "plot.png", kind="furniture")


def test_plotters_close_figures_when_an_evaluation_split_is_missing(
    tmp_path: Path,
) -> None:
    records_by_epoch = {
        epoch: (replace(_record(split="val_seen"), epoch=epoch),)
        for epoch in (1, 2, 5, 10)
    }
    figure_numbers = plt.get_fignums()

    with pytest.raises(ValueError, match="no records for split 'val_unseen'"):
        plot_iou_survival(
            records_by_epoch,
            tmp_path / "iou-survival.png",
            thresholds=(0.0,),
        )
    assert plt.get_fignums() == figure_numbers

    with pytest.raises(ValueError, match="no category F1 records for split 'val_unseen'"):
        plot_category_f1(records_by_epoch, tmp_path / "category-f1.png")
    assert plt.get_fignums() == figure_numbers


def test_sanity_analysis_writes_quantitative_artifacts_and_postpones_r2r_comparison(
    tmp_path: Path,
) -> None:
    sweep_root = tmp_path / "sweep"
    for epoch, path in epoch_episode_paths(sweep_root).items():
        path.parent.mkdir(parents=True)
        records = [
            replace(_record(split="val_seen", example_id="seen"), epoch=epoch),
            replace(_record(split="val_unseen", example_id="unseen"), epoch=epoch),
        ]
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    "split",
                    "scene_id",
                    "example_id",
                    "category_aware_raster_iou",
                    "schema_valid",
                    "object_category_target_count",
                    "object_category_predicted_count",
                    "object_category_true_positive_count",
                    "region_category_target_count",
                    "region_category_predicted_count",
                    "region_category_true_positive_count",
                    "mentioned_object_category_predicted_count",
                    "unmentioned_object_category_predicted_count",
                    "mentioned_region_category_predicted_count",
                    "unmentioned_region_category_predicted_count",
                ),
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "split": record.split,
                        "scene_id": record.scene_id,
                        "example_id": record.example_id,
                        "category_aware_raster_iou": record.iou,
                        "schema_valid": int(record.schema_valid),
                        "object_category_target_count": record.object_target,
                        "object_category_predicted_count": record.object_predicted,
                        "object_category_true_positive_count": record.object_true_positive,
                        "region_category_target_count": record.region_target,
                        "region_category_predicted_count": record.region_predicted,
                        "region_category_true_positive_count": record.region_true_positive,
                        "mentioned_object_category_predicted_count": record.mentioned_object_predicted,
                        "unmentioned_object_category_predicted_count": record.unmentioned_object_predicted,
                        "mentioned_region_category_predicted_count": record.mentioned_region_predicted,
                        "unmentioned_region_category_predicted_count": record.unmentioned_region_predicted,
                    }
                )

    args = SanityPlotArgs()
    args.sweep_root = sweep_root
    args.output_root = tmp_path / "analysis"
    args.docs_image_root = tmp_path / "docs-images"
    _run_analysis(args, expected_counts={"val_seen": 1, "val_unseen": 1})

    artifact_names = {
        "iou_histograms.csv",
        "iou_survival.csv",
        "category_f1.csv",
        "category_count_histograms.csv",
        "summary.json",
        "iou_histograms.png",
        "iou_survival.png",
        "category_f1.png",
        "object_category_counts.png",
        "region_category_counts.png",
    }
    assert {path.name for path in args.output_root.iterdir()} == artifact_names
    assert {path.name for path in args.docs_image_root.iterdir()} == {
        name for name in artifact_names if name.endswith(".png")
    }

    summary = json.loads((args.output_root / "summary.json").read_text())
    assert summary["population_counts"] == {
        "1": {"val_seen": 1, "val_unseen": 1},
        "2": {"val_seen": 1, "val_unseen": 1},
        "5": {"val_seen": 1, "val_unseen": 1},
        "10": {"val_seen": 1, "val_unseen": 1},
    }
    comparison = summary["r2r_only_comparison"]
    assert comparison["status"] == "postponed"
    assert comparison["missing_cache"] == (
        "data/llm_navigation/llm-grid-r2r-legacy-r1p5-direction5-scale2"
    )
    assert comparison["missing_evaluator_outputs"] == (
        "outputs/llm_grid_eval/r2r-only-checkpoint-sweep-r1p5/runs/*/episodes.csv"
    )
