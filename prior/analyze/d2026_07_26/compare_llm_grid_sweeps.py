"""Compare the R2R-only and R2R+RxR-EN LLM-Grid checkpoint sweeps."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from tap import Tap

from prior.analyze.llm_grid_eval_plots import (
    plot_category_counts,
    plot_category_f1,
    plot_iou_histograms,
    plot_iou_survival,
)
from prior.analyze.llm_grid_eval_records import (
    CSVValue,
    EpisodeRecord,
    category_count_histogram_rows,
    category_f1_rows,
    f1_summary,
    iou_histogram_rows,
    iou_survival_rows,
    load_episode_records,
    write_csv_rows,
)
from vlnce_baselines.models.etp_llm.llm_grid_eval_plot import EpochMetrics


_ALL_EPOCHS = tuple(range(1, 11))
_SELECTED_EPOCHS = (1, 2, 5, 10)
_SPLITS = ("val_seen", "val_unseen")
_EXPECTED_COUNTS = {"val_seen": 778, "val_unseen": 1839}
_IOU_THRESHOLDS = tuple(index / 20 for index in range(21))
_COMPARISON_METRICS = (
    ("Raster IoU", "category_aware_raster_iou"),
    ("Cell F1", "cell_f1"),
    ("Object category F1", "object_category_f1"),
    ("Region category F1", "region_category_f1"),
)
_SUMMARY_METRICS = tuple(metric for _, metric in _COMPARISON_METRICS) + (
    "schema_valid",
)


class SweepComparisonArgs(Tap):
    """Fixed inputs and destinations for the two-corpus comparison."""

    r2r_only_sweep_root: Path = Path(
        "outputs/llm_grid_eval/r2r-only-checkpoint-sweep-r1p5"
    )
    mixed_sweep_root: Path = Path("outputs/llm_grid_eval/checkpoint-sweep-r1p5")
    output_root: Path = Path("outputs/llm_grid_analysis/r2r_only_vs_r2r_rxr_sanity")
    docs_image_root: Path = Path("docs/images/llm_grid_r2r_only_vs_r2r_rxr_sanity")


@dataclass(frozen=True)
class LoadedSweep:
    """One complete ten-checkpoint evaluator sweep."""

    label: str
    root: Path
    records_by_epoch: Mapping[int, tuple[EpisodeRecord, ...]]
    metrics_by_epoch: Mapping[int, Mapping[str, float]]
    identities: frozenset[tuple[str, str, str]]

    @classmethod
    def load(
        cls,
        *,
        label: str,
        root: Path,
        expected_counts: Mapping[str, int],
    ) -> LoadedSweep:
        records_by_epoch = {
            epoch: load_episode_records(
                epoch, root / "runs" / f"{epoch - 1:03d}" / "episodes.csv"
            )
            for epoch in _ALL_EPOCHS
        }
        identities = _validate_populations(
            records_by_epoch,
            expected_counts=expected_counts,
            label=label,
        )
        aggregate = EpochMetrics.load(root / "metrics.json")
        epochs = tuple(result.epoch for result in aggregate)
        if epochs != _ALL_EPOCHS:
            raise ValueError(
                f"{label} aggregate epochs differ: "
                f"expected {list(_ALL_EPOCHS)}, got {list(epochs)}"
            )
        return cls(
            label=label,
            root=root,
            records_by_epoch=records_by_epoch,
            metrics_by_epoch={result.epoch: result.metrics for result in aggregate},
            identities=identities,
        )


def _record_identity(record: EpisodeRecord) -> tuple[str, str, str]:
    return record.split, record.scene_id, record.example_id


def _validate_populations(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]],
    *,
    expected_counts: Mapping[str, int],
    label: str,
) -> frozenset[tuple[str, str, str]]:
    epochs = tuple(sorted(records_by_epoch))
    if epochs != _ALL_EPOCHS:
        raise ValueError(
            f"{label} epochs differ: expected {list(_ALL_EPOCHS)}, got {list(epochs)}"
        )

    reference: Optional[frozenset[tuple[str, str, str]]] = None
    for epoch in _ALL_EPOCHS:
        records = records_by_epoch[epoch]
        counts = Counter(record.split for record in records)
        if dict(counts) != dict(expected_counts):
            raise ValueError(
                f"{label} epoch {epoch} split counts differ: "
                f"expected {dict(expected_counts)}, got {dict(counts)}"
            )
        identities = [_record_identity(record) for record in records]
        identity_set = frozenset(identities)
        if len(identity_set) != len(identities):
            raise ValueError(f"{label} epoch {epoch} has duplicate episode identities")
        if reference is None:
            reference = identity_set
        elif identity_set != reference:
            raise ValueError(f"{label} epoch {epoch} episode identities differ")

    if reference is None:
        raise ValueError(f"{label} has no episode identities")
    return reference


def _paths_overlap(first: Path, second: Path) -> bool:
    resolved_first = first.resolve()
    resolved_second = second.resolve()
    return (
        resolved_first == resolved_second
        or resolved_first in resolved_second.parents
        or resolved_second in resolved_first.parents
    )


def _refuse_input_overwrite(args: SweepComparisonArgs) -> None:
    for input_root in (args.r2r_only_sweep_root, args.mixed_sweep_root):
        for output_root in (args.output_root, args.docs_image_root):
            if _paths_overlap(input_root, output_root):
                raise ValueError(
                    f"evaluator input overlaps output: {input_root} and {output_root}"
                )


def _selected_records(
    sweep: LoadedSweep,
) -> dict[int, tuple[EpisodeRecord, ...]]:
    return {epoch: sweep.records_by_epoch[epoch] for epoch in _SELECTED_EPOCHS}


def _write_r2r_only_sanity(
    sweep: LoadedSweep,
    output_root: Path,
    docs_image_root: Path,
) -> dict[str, object]:
    records_by_epoch = _selected_records(sweep)
    output_root.mkdir(parents=True, exist_ok=True)
    category_rows = category_f1_rows(records_by_epoch)
    write_csv_rows(
        output_root / "iou_histograms.csv",
        iou_histogram_rows(records_by_epoch),
    )
    write_csv_rows(
        output_root / "iou_survival.csv",
        iou_survival_rows(records_by_epoch, thresholds=_IOU_THRESHOLDS),
    )
    write_csv_rows(output_root / "category_f1.csv", category_rows)
    write_csv_rows(
        output_root / "category_count_histograms.csv",
        category_count_histogram_rows(records_by_epoch),
    )

    figure_paths = {
        "iou_histograms": output_root / "iou_histograms.png",
        "iou_survival": output_root / "iou_survival.png",
        "category_f1": output_root / "category_f1.png",
        "object_category_counts": output_root / "object_category_counts.png",
        "region_category_counts": output_root / "region_category_counts.png",
    }
    plot_iou_histograms(records_by_epoch, figure_paths["iou_histograms"])
    plot_iou_survival(
        records_by_epoch,
        figure_paths["iou_survival"],
        thresholds=_IOU_THRESHOLDS,
    )
    plot_category_f1(records_by_epoch, figure_paths["category_f1"])
    plot_category_counts(
        records_by_epoch,
        figure_paths["object_category_counts"],
        kind="object",
    )
    plot_category_counts(
        records_by_epoch,
        figure_paths["region_category_counts"],
        kind="region",
    )

    docs_image_root.mkdir(parents=True, exist_ok=True)
    for path in figure_paths.values():
        shutil.copyfile(path, docs_image_root / path.name)

    return {
        "episode_macro_category_f1": f1_summary(category_rows),
        "figure_paths": {
            name: {
                "analysis": str(path),
                "docs": str(docs_image_root / path.name),
            }
            for name, path in sorted(figure_paths.items())
        },
        "invalid_counts": {
            str(epoch): {
                split: sum(
                    not record.schema_valid
                    for record in records_by_epoch[epoch]
                    if record.split == split
                )
                for split in _SPLITS
            }
            for epoch in _SELECTED_EPOCHS
        },
    }


def _metric_value(
    sweep: LoadedSweep,
    epoch: int,
    split: str,
    metric: str,
) -> float:
    key = f"{split}/{metric}"
    metrics = sweep.metrics_by_epoch[epoch]
    if key not in metrics:
        raise ValueError(f"{sweep.label} epoch {epoch} is missing metric {key}")
    return metrics[key]


def _comparison_rows(
    r2r_only: LoadedSweep,
    mixed: LoadedSweep,
) -> list[dict[str, CSVValue]]:
    rows: list[dict[str, CSVValue]] = []
    for epoch in _ALL_EPOCHS:
        for metric in _SUMMARY_METRICS:
            r2r_seen = _metric_value(r2r_only, epoch, "val_seen", metric)
            r2r_unseen = _metric_value(r2r_only, epoch, "val_unseen", metric)
            mixed_seen = _metric_value(mixed, epoch, "val_seen", metric)
            mixed_unseen = _metric_value(mixed, epoch, "val_unseen", metric)
            rows.append({
                "epoch": epoch,
                "metric": metric,
                "r2r_only_val_seen": r2r_seen,
                "r2r_only_val_unseen": r2r_unseen,
                "r2r_only_seen_unseen_gap": r2r_seen - r2r_unseen,
                "mixed_val_seen": mixed_seen,
                "mixed_val_unseen": mixed_unseen,
                "mixed_seen_unseen_gap": mixed_seen - mixed_unseen,
                "mixed_minus_r2r_only_val_seen": mixed_seen - r2r_seen,
                "mixed_minus_r2r_only_val_unseen": mixed_unseen - r2r_unseen,
                "mixed_minus_r2r_only_gap": (
                    (mixed_seen - mixed_unseen) - (r2r_seen - r2r_unseen)
                ),
            })
    return rows


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)


def _plot_corpus_comparison(
    r2r_only: LoadedSweep,
    mixed: LoadedSweep,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    corpora = (
        (r2r_only, "R2R only", "tab:blue"),
        (mixed, "R2R + RxR-EN", "tab:orange"),
    )
    for axis_index, (axis, (title, metric)) in enumerate(
        zip(axes.flat, _COMPARISON_METRICS)
    ):
        for sweep, corpus_label, color in corpora:
            for split, linestyle in (("val_seen", "-"), ("val_unseen", "--")):
                axis.plot(
                    _ALL_EPOCHS,
                    [
                        _metric_value(sweep, epoch, split, metric)
                        for epoch in _ALL_EPOCHS
                    ],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    label=f"{corpus_label} {split}",
                )
        axis.set_title(title)
        if axis_index >= 2:
            axis.set_xlabel("Epoch")
        axis.set_xticks(_ALL_EPOCHS)
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    figure.suptitle("LLM-Grid validation quality by training corpus", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_figure(figure, output_path)


def _plot_overfit_gaps(
    r2r_only: LoadedSweep,
    mixed: LoadedSweep,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    corpora = (
        (r2r_only, "R2R only", "tab:blue"),
        (mixed, "R2R + RxR-EN", "tab:orange"),
    )
    for axis_index, (axis, (title, metric)) in enumerate(
        zip(axes.flat, _COMPARISON_METRICS)
    ):
        for sweep, corpus_label, color in corpora:
            axis.plot(
                _ALL_EPOCHS,
                [
                    _metric_value(sweep, epoch, "val_seen", metric)
                    - _metric_value(sweep, epoch, "val_unseen", metric)
                    for epoch in _ALL_EPOCHS
                ],
                color=color,
                marker="o",
                label=corpus_label,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(title)
        if axis_index >= 2:
            axis.set_xlabel("Epoch")
        axis.set_xticks(_ALL_EPOCHS)
        axis.set_ylabel("val_seen - val_unseen")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("LLM-Grid seen–unseen gap by training corpus", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_figure(figure, output_path)


def _run_analysis(
    args: SweepComparisonArgs,
    *,
    expected_counts: Mapping[str, int],
) -> dict[str, object]:
    _refuse_input_overwrite(args)
    r2r_only = LoadedSweep.load(
        label="r2r_only",
        root=args.r2r_only_sweep_root,
        expected_counts=expected_counts,
    )
    mixed = LoadedSweep.load(
        label="r2r_rxr_en",
        root=args.mixed_sweep_root,
        expected_counts=expected_counts,
    )
    if r2r_only.identities != mixed.identities:
        raise ValueError("R2R-only and R2R+RxR-EN episode identities differ")

    r2r_only_root = args.output_root / "r2r_only_sanity"
    r2r_only_summary = _write_r2r_only_sanity(
        r2r_only,
        r2r_only_root,
        args.docs_image_root / "r2r_only_sanity",
    )
    comparison_rows = _comparison_rows(r2r_only, mixed)
    comparison_csv = args.output_root / "checkpoint_comparison.csv"
    write_csv_rows(comparison_csv, comparison_rows)

    comparison_figure = args.output_root / "checkpoint_comparison.png"
    gap_figure = args.output_root / "overfit_gaps.png"
    _plot_corpus_comparison(r2r_only, mixed, comparison_figure)
    _plot_overfit_gaps(r2r_only, mixed, gap_figure)
    args.docs_image_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(comparison_figure, args.docs_image_root / comparison_figure.name)
    shutil.copyfile(gap_figure, args.docs_image_root / gap_figure.name)

    summary = {
        "checkpoint_comparison": comparison_rows,
        "comparison_csv": str(comparison_csv),
        "expected_counts": dict(expected_counts),
        "figures": {
            "checkpoint_comparison": {
                "analysis": str(comparison_figure),
                "docs": str(args.docs_image_root / comparison_figure.name),
            },
            "overfit_gaps": {
                "analysis": str(gap_figure),
                "docs": str(args.docs_image_root / gap_figure.name),
            },
        },
        "inputs": {
            "r2r_only": str(args.r2r_only_sweep_root),
            "r2r_rxr_en": str(args.mixed_sweep_root),
        },
        "r2r_only_sanity": r2r_only_summary,
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"summary": summary, "summary_path": summary_path}


def run_analysis(args: SweepComparisonArgs) -> dict[str, object]:
    """Generate the complete fixed-population comparison."""
    return _run_analysis(args, expected_counts=_EXPECTED_COUNTS)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = SweepComparisonArgs(underscores_to_dashes=True).parse_args(argv)
    result = run_analysis(args)
    print(f"Wrote LLM-Grid corpus comparison to {result['summary_path']}")


if __name__ == "__main__":
    main()
