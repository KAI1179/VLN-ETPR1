from __future__ import annotations

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
