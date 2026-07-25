"""Reusable quantitative figures for LLM-Grid evaluator episode records."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from prior.analyze.llm_grid_eval_records import (
    EpisodeRecord,
    category_count_histogram_rows,
    category_f1_rows,
    iou_histogram_rows,
    iou_survival_rows,
)


_EPOCHS = (1, 2, 5, 10)
_SPLITS = ("val_seen", "val_unseen")
_COUNT_PARTITIONS = ("total", "mentioned", "unmentioned")


def _require_records(records_by_epoch: Mapping[int, Sequence[EpisodeRecord]]) -> None:
    if not records_by_epoch:
        raise ValueError("records_by_epoch must not be empty")
    epochs = tuple(sorted(records_by_epoch))
    if epochs != _EPOCHS:
        raise ValueError(f"unsupported epochs: expected {list(_EPOCHS)}, got {list(epochs)}")


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_iou_histograms(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]],
    output_path: Path,
    bin_count: int = 20,
) -> None:
    """Plot fixed-range IoU histograms by checkpoint epoch and evaluation split."""
    _require_records(records_by_epoch)
    rows = iou_histogram_rows(records_by_epoch, bin_count=bin_count)
    figure, axes = plt.subplots(2, 4, figsize=(14, 6), squeeze=False)
    for split_index, split in enumerate(_SPLITS):
        for epoch_index, epoch in enumerate(_EPOCHS):
            axis = axes[split_index, epoch_index]
            histogram_rows = [
                row for row in rows if row["epoch"] == epoch and row["split"] == split
            ]
            starts = [float(row["bin_start"]) for row in histogram_rows]
            widths = [float(row["bin_end"]) - float(row["bin_start"]) for row in histogram_rows]
            counts = [int(row["count"]) for row in histogram_rows]
            axis.bar(starts, counts, width=widths, align="edge", color=f"C{epoch_index}")
            axis.set_xlim(0.0, 1.0)
            axis.set_xlabel("IoU")
            if epoch_index == 0:
                axis.set_ylabel("Episodes")
            if split_index == 0:
                axis.set_title(f"Epoch {epoch}")
            if epoch_index == 3:
                axis.annotate(
                    split,
                    xy=(1.04, 0.5),
                    xycoords="axes fraction",
                    rotation=-90,
                    ha="left",
                    va="center",
                )
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_iou_survival(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]],
    output_path: Path,
    thresholds: Sequence[float],
) -> None:
    """Plot IoU survival counts and population fractions by evaluation split."""
    _require_records(records_by_epoch)
    rows = iou_survival_rows(records_by_epoch, thresholds=thresholds)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    for split_index, split in enumerate(_SPLITS):
        axis = axes[0, split_index]
        population = sum(
            record.split == split for record in records_by_epoch[_EPOCHS[0]]
        )
        if population == 0:
            raise ValueError(f"no records for split {split!r}")
        for epoch_index, epoch in enumerate(_EPOCHS):
            survival_rows = sorted(
                (
                    row
                    for row in rows
                    if row["epoch"] == epoch and row["split"] == split
                ),
                key=lambda row: float(row["threshold"]),
            )
            axis.plot(
                [float(row["threshold"]) for row in survival_rows],
                [int(row["count"]) for row in survival_rows],
                label=f"Epoch {epoch}",
                color=f"C{epoch_index}",
            )
        axis.set_title(split)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0, population)
        axis.set_xlabel("IoU threshold")
        axis.set_ylabel("Episodes")
        fraction_axis = axis.secondary_yaxis(
            "right",
            functions=(
                lambda count, population=population: count / population,
                lambda fraction, population=population: fraction * population,
            ),
        )
        fraction_axis.set_ylabel("Population fraction")
        fraction_axis.set_ylim(0.0, 1.0)
        axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_category_f1(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]], output_path: Path
) -> None:
    """Plot episode-wise category F1 distributions and the predict-all baseline."""
    _require_records(records_by_epoch)
    rows = category_f1_rows(records_by_epoch)
    figure, axes = plt.subplots(3, 2, figsize=(10, 11), squeeze=False, sharey=True)
    for category_index, (category, field) in enumerate(
        (("Object", "object_f1"), ("Region", "region_f1"), ("Combined", "combined_f1"))
    ):
        for split_index, split in enumerate(_SPLITS):
            axis = axes[category_index, split_index]
            distributions = [
                [
                    float(row[field])
                    for row in rows
                    if row["source"] == "model"
                    and row["epoch"] == epoch
                    and row["split"] == split
                ]
                for epoch in _EPOCHS
            ]
            distributions.append(
                [
                    float(row[field])
                    for row in rows
                    if row["source"] == "predict_all" and row["split"] == split
                ]
            )
            if any(not distribution for distribution in distributions):
                raise ValueError(f"no category F1 records for split {split!r}")
            axis.boxplot(distributions, labels=["1", "2", "5", "10", "predict all"])
            axis.set_ylim(-0.05, 1.05)
            axis.set_ylabel("Episode-wise F1")
            axis.set_xlabel("Epoch")
            if category_index == 0:
                axis.set_title(split)
            if split_index == 0:
                axis.annotate(
                    category,
                    xy=(-0.28, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                )
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_category_counts(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]],
    output_path: Path,
    kind: str,
) -> None:
    """Plot predicted category-count distributions by partition and checkpoint."""
    if kind not in ("object", "region"):
        raise ValueError("kind must be 'object' or 'region'")
    _require_records(records_by_epoch)
    maximum_category_count = 27 if kind == "object" else 10
    edges = np.arange(-0.5, maximum_category_count + 1.5, 1.0)
    rows = category_count_histogram_rows(records_by_epoch)
    count_limit = max(
        int(row["count"])
        for row in rows
        if row["category"] == kind
    )
    figure, axes = plt.subplots(3, 4, figsize=(14, 9), squeeze=False, sharex=True)
    for partition_index, partition in enumerate(_COUNT_PARTITIONS):
        for epoch_index, epoch in enumerate(_EPOCHS):
            axis = axes[partition_index, epoch_index]
            for split_index, split in enumerate(_SPLITS):
                histogram_rows = [
                    row
                    for row in rows
                    if row["epoch"] == epoch
                    and row["split"] == split
                    and row["category"] == kind
                    and row["partition"] == partition
                ]
                values = [int(row["count"]) for row in histogram_rows]
                axis.stairs(values, edges, label=split, color=f"C{split_index}")
            axis.set_xlim(-0.5, maximum_category_count + 0.5)
            axis.set_ylim(0, max(1, count_limit))
            if epoch_index == 0:
                axis.set_ylabel("Episodes")
            if partition_index == 2:
                axis.set_xlabel("Predicted categories")
            if partition_index == 0:
                axis.set_title(f"Epoch {epoch}")
            if epoch_index == 3:
                axis.annotate(
                    partition,
                    xy=(1.04, 0.5),
                    xycoords="axes fraction",
                    rotation=-90,
                    ha="left",
                    va="center",
                )
    axes[0, 3].legend()
    figure.tight_layout()
    _save_figure(figure, output_path)
