"""Plot an LLM-Grid checkpoint sweep from its aggregate metrics JSON."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from tap import Tap


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    metrics: dict[str, float]

    @classmethod
    def load(cls, path: Path) -> tuple[EpochMetrics, ...]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {"runs"}:
            raise ValueError("sweep metrics must contain exactly one 'runs' field")
        runs = raw["runs"]
        if not isinstance(runs, list) or not runs:
            raise ValueError("sweep metrics runs must be a non-empty list")

        results: list[EpochMetrics] = []
        for index, run in enumerate(runs):
            if not isinstance(run, dict):
                raise ValueError(f"sweep run {index} must be an object")
            fields: dict[str, object] = {}
            for key, value in run.items():
                if not isinstance(key, str):
                    raise ValueError(f"sweep run {index} field names must be strings")
                fields[key] = value
            label = fields.get("label")
            match = (
                re.fullmatch(r"epoch-(\d+)", label) if isinstance(label, str) else None
            )
            if match is None:
                raise ValueError(f"sweep run {index} label must be 'epoch-N'")
            metrics = fields.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"sweep run {index} metrics must be numeric")
            metric_values: dict[str, float] = {}
            for key, value in metrics.items():
                if not isinstance(key, str) or not isinstance(value, (int, float)):
                    raise ValueError(f"sweep run {index} metrics must be numeric")
                metric_values[key] = float(value)
            results.append(
                cls(
                    epoch=int(match.group(1)),
                    metrics=metric_values,
                )
            )
        epochs = [result.epoch for result in results]
        if len(set(epochs)) != len(epochs):
            raise ValueError("sweep metrics contain duplicate epochs")
        return tuple(sorted(results, key=lambda result: result.epoch))


class LLMGridEvalPlotArgs(Tap):
    input_path: Path = Path("outputs/llm_grid_eval/checkpoint-sweep-r1p5/metrics.json")
    output_path: Path = Path("outputs/llm_grid_eval/checkpoint-sweep-r1p5/metrics.png")
    mention_partition_output_path: Optional[Path] = None

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("underscores_to_dashes", True)
        super().__init__(*args, **kwargs)


PANELS = (
    (
        "Spatial quality",
        (
            ("Raster IoU", "category_aware_raster_iou"),
            ("Cell F1", "cell_f1"),
        ),
    ),
    (
        "Category presence",
        (
            ("Object F1", "object_category_f1"),
            ("Region F1", "region_category_f1"),
        ),
    ),
    (
        "Direction quality",
        (
            ("Direction cosine", "direction_vector_cosine"),
            ("Padding accuracy", "direction_vector_padding_accuracy"),
        ),
    ),
    ("Output reliability", (("Schema valid", "schema_valid"),)),
)

MENTION_PARTITION_REQUIRED_METRICS = frozenset(
    {
        f"{split}/{status}_{metric}"
        for split in ("val_seen", "val_unseen")
        for status in ("mentioned", "unmentioned")
        for metric in ("category_aware_raster_iou", "cell_f1")
    }
    | {
        f"{split}/{status}_{kind}_category_f1"
        for split in ("val_seen", "val_unseen")
        for status in ("mentioned", "unmentioned")
        for kind in ("object", "region")
    }
)


def plot_checkpoint_metrics(
    results: Sequence[EpochMetrics],
    output_path: Path,
) -> None:
    if not results:
        raise ValueError("cannot plot an empty checkpoint sweep")
    required = {
        f"{split}/{metric}"
        for split in ("val_seen", "val_unseen")
        for _title, panel_metrics in PANELS
        for _label, metric in panel_metrics
    }
    for result in results:
        missing = sorted(required - result.metrics.keys())
        if missing:
            raise ValueError(
                f"epoch-{result.epoch} is missing plot metrics: {', '.join(missing)}"
            )

    epochs = [result.epoch for result in results]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = ("tab:blue", "tab:orange")
    for axis, (title, panel_metrics) in zip(axes.flat, PANELS):
        for (metric_label, metric), color in zip(panel_metrics, colors):
            for split, linestyle in (("val_seen", "-"), ("val_unseen", "--")):
                axis.plot(
                    epochs,
                    [result.metrics[f"{split}/{metric}"] for result in results],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    label=f"{split} {metric_label}",
                )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    figure.suptitle("LLM-Grid R2R predictor quality by checkpoint")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_mention_partition_metrics(
    results: Sequence[EpochMetrics],
    output_path: Path,
) -> None:
    if not results:
        raise ValueError("cannot plot an empty checkpoint sweep")
    for result in results:
        missing = sorted(MENTION_PARTITION_REQUIRED_METRICS - result.metrics.keys())
        if missing:
            raise ValueError(
                f"epoch-{result.epoch} is missing mention-partition plot metrics: "
                f"{', '.join(missing)}"
            )

    epochs = [result.epoch for result in results]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    statuses = (
        ("Mentioned", "mentioned", "tab:blue"),
        ("Unmentioned", "unmentioned", "tab:orange"),
    )
    spatial_metrics = (
        ("Raster IoU", "category_aware_raster_iou", "-"),
        ("Cell F1", "cell_f1", "--"),
    )
    for axis, split in zip(axes[0], ("val_seen", "val_unseen")):
        for status_label, status, color in statuses:
            for metric_label, metric, linestyle in spatial_metrics:
                axis.plot(
                    epochs,
                    [
                        result.metrics[f"{split}/{status}_{metric}"]
                        for result in results
                    ],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    label=f"{status_label} {metric_label}",
                )
        axis.set_title(f"{split} spatial quality")

    for axis, (kind_label, kind) in zip(
        axes[1],
        (("Object", "object"), ("Region", "region")),
    ):
        for status_label, status, color in statuses:
            for split, linestyle in (("val_seen", "-"), ("val_unseen", "--")):
                axis.plot(
                    epochs,
                    [
                        result.metrics[
                            f"{split}/{status}_{kind}_category_f1"
                        ]
                        for result in results
                    ],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    label=f"{split} {status_label} F1",
                )
        axis.set_title(f"{kind_label} category presence")

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    figure.suptitle("LLM-Grid R2R quality by instruction mention partition")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = LLMGridEvalPlotArgs().parse_args(argv)
    results = EpochMetrics.load(args.input_path)
    plot_checkpoint_metrics(results, args.output_path)
    if args.mention_partition_output_path is not None:
        plot_mention_partition_metrics(results, args.mention_partition_output_path)


if __name__ == "__main__":
    main()
