"""Compute per-category LLM-Grid presence and spatial diagnostics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tap import Tap

from prior.constants import (
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    OBJECT_CATEGORIES,
)
from vlnce_baselines.models.etp_llm.llm_grid_eval import (
    _extract_instruction_mentions,
)
from vlnce_baselines.models.etp_llm.llm_grid_input_dependence import (
    _load_r2r_trajectory_ids,
)
from vlnce_baselines.models.etp_llm.llm_grid_train import (
    DEFAULT_GRID_NAMESPACE,
    GRID_SCALE,
    GRID_SHAPE,
    LLMGridDataset,
    LLMGridValidationError,
    load_llm_grid_examples,
    parse_grid_text,
)
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_prediction_path,
)


_BROAD_OBJECT_NAMES = frozenset(("void", "structure", "other", "free-space"))
_EXPECTED_EXAMPLES = 1839
_EXPECTED_SCENES = 11


class CategoryMetricArgs(Tap):
    """Inputs and destinations for the selected epoch-2 val_unseen analysis."""

    cache_dir: Path = Path("data/llm_navigation")
    cache_model_key: str = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2"
    cognitive_map_namespace: str = DEFAULT_GRID_NAMESPACE
    output_root: Path = Path(
        "docs/data/llm_grid_presentation_2026_07_26/category_metrics"
    )
    docs_image_path: Path = Path(
        "docs/images/llm_grid_presentation/category_presence_vs_spatial_iou.png"
    )
    min_target_scenes: int = 5


@dataclass
class _CategoryCounts:
    group: str
    category_id: int
    channel_id: int
    category_name: str
    broad_environment_label: bool
    example_count: int = 0
    schema_invalid_example_count: int = 0
    missing_prediction_count: int = 0
    target_examples: int = 0
    predicted_examples: int = 0
    presence_true_positives: int = 0
    target_cells: int = 0
    predicted_cells: int = 0
    spatial_true_positives: int = 0
    instruction_mentioned_examples: int = 0
    target_and_mentioned_examples: int = 0
    target_routes: set[tuple[str, int]] = field(default_factory=set)
    target_scenes: set[str] = field(default_factory=set)

    def update(
        self,
        *,
        target: NDArray[np.bool_],
        predicted: NDArray[np.bool_],
        scene_id: str,
        route_id: int,
        mentioned: bool,
        schema_valid: bool,
        missing: bool,
    ) -> None:
        target_present = bool(target.any())
        predicted_present = bool(predicted.any())
        self.example_count += 1
        self.schema_invalid_example_count += int(not schema_valid)
        self.missing_prediction_count += int(missing)
        self.target_examples += int(target_present)
        self.predicted_examples += int(predicted_present)
        self.presence_true_positives += int(target_present and predicted_present)
        self.target_cells += int(np.count_nonzero(target))
        self.predicted_cells += int(np.count_nonzero(predicted))
        self.spatial_true_positives += int(np.count_nonzero(target & predicted))
        self.instruction_mentioned_examples += int(mentioned)
        self.target_and_mentioned_examples += int(target_present and mentioned)
        if target_present:
            self.target_routes.add((scene_id, route_id))
            self.target_scenes.add(scene_id)

    def row(self, *, route_count: int, scene_count: int) -> dict[str, object]:
        spatial_union = (
            self.predicted_cells + self.target_cells - self.spatial_true_positives
        )
        return {
            "category_group": self.group,
            "category_id": self.category_id,
            "channel_id": self.channel_id,
            "category_name": self.category_name,
            "broad_environment_label": self.broad_environment_label,
            "example_count": self.example_count,
            "route_count": route_count,
            "scene_count": scene_count,
            "schema_invalid_example_count": self.schema_invalid_example_count,
            "missing_prediction_count": self.missing_prediction_count,
            "target_example_count": self.target_examples,
            "target_route_count": len(self.target_routes),
            "target_scene_count": len(self.target_scenes),
            "target_cell_count": self.target_cells,
            "predicted_example_count": self.predicted_examples,
            "predicted_cell_count": self.predicted_cells,
            "presence_true_positive_example_count": self.presence_true_positives,
            "presence_precision": _ratio(
                self.presence_true_positives, self.predicted_examples
            ),
            "presence_recall": _ratio(
                self.presence_true_positives, self.target_examples
            ),
            "presence_f1": _f1(
                self.presence_true_positives,
                self.predicted_examples,
                self.target_examples,
            ),
            "spatial_true_positive_cell_count": self.spatial_true_positives,
            "spatial_union_cell_count": spatial_union,
            "spatial_precision": _ratio(
                self.spatial_true_positives, self.predicted_cells
            ),
            "spatial_recall": _ratio(self.spatial_true_positives, self.target_cells),
            "spatial_f1": _f1(
                self.spatial_true_positives,
                self.predicted_cells,
                self.target_cells,
            ),
            "spatial_iou": _ratio(self.spatial_true_positives, spatial_union),
            "instruction_mentioned_example_count": (
                self.instruction_mentioned_examples
            ),
            "target_and_mentioned_example_count": (self.target_and_mentioned_examples),
            "target_mention_rate": _ratio(
                self.target_and_mentioned_examples, self.target_examples
            ),
        }


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _f1(true_positives: int, predicted: int, target: int) -> float:
    return _ratio(2 * true_positives, predicted + target)


def _accumulators() -> tuple[_CategoryCounts, ...]:
    objects = tuple(
        _CategoryCounts(
            group="object",
            category_id=category_id,
            channel_id=category_id,
            category_name=name,
            broad_environment_label=name in _BROAD_OBJECT_NAMES,
        )
        for category_id, name in enumerate(MAPPED_OBJECT_NAMES)
    )
    regions = tuple(
        _CategoryCounts(
            group="region",
            category_id=category_id,
            channel_id=OBJECT_CATEGORIES + category_id,
            category_name=name,
            broad_environment_label=False,
        )
        for category_id, name in enumerate(MAPPED_REGION_NAMES)
    )
    return objects + regions


def _mentioned_channels(
    object_categories: AbstractSet[int],
    region_categories: AbstractSet[int],
) -> frozenset[int]:
    return frozenset(object_categories) | frozenset(
        OBJECT_CATEGORIES + category for category in region_categories
    )


def collect_category_metrics(
    args: CategoryMetricArgs,
) -> tuple[dict[str, object], ...]:
    """Replay the selected cache and pool presence/cell counts by category."""
    loaded = load_llm_grid_examples(
        ("val_unseen",),
        quiet=True,
        cognitive_map_namespace=args.cognitive_map_namespace,
        datasets=("R2R",),
    )
    if len(loaded.examples) != _EXPECTED_EXAMPLES:
        raise ValueError(
            f"expected {_EXPECTED_EXAMPLES} val_unseen examples, "
            f"got {len(loaded.examples)}"
        )
    dataset = LLMGridDataset(loaded.examples, scale=GRID_SCALE)
    accumulators = _accumulators()
    trajectory_ids = _load_r2r_trajectory_ids()
    routes: set[tuple[str, int]] = set()
    scenes: set[str] = set()

    for example, item in zip(loaded.examples, dataset):
        if item["example_id"] != example.example_id:
            raise ValueError("LLM-Grid dataset order changed during category analysis")
        prediction_path = llm_navigation_prediction_path(
            example.scene_id,
            example.example_id,
            "R2R",
            "val_unseen",
            cache_dir=args.cache_dir,
            model_key=args.cache_model_key,
        )
        missing = not prediction_path.is_file()
        schema_valid = False
        predicted = np.zeros(GRID_SHAPE, dtype=np.bool_)
        if not missing:
            try:
                predicted = (
                    parse_grid_text(
                        prediction_path.read_text(encoding="utf-8"),
                        shape=GRID_SHAPE,
                    ).grid
                    > 0
                )
                schema_valid = True
            except LLMGridValidationError:
                pass
        target = item["target_grid"] > 0
        if target.shape != GRID_SHAPE:
            raise ValueError(
                f"unexpected target shape for {example.example_id}: {target.shape}"
            )
        mentioned = _mentioned_channels(
            *_extract_instruction_mentions(example.instruction)
        )
        try:
            route_id = trajectory_ids[("val_unseen", example.example_id)]
        except KeyError as error:
            raise KeyError(
                f"missing R2R trajectory ID for {example.example_id}"
            ) from error
        routes.add((example.scene_id, route_id))
        scenes.add(example.scene_id)
        for counts in accumulators:
            counts.update(
                target=target[counts.channel_id],
                predicted=predicted[counts.channel_id],
                scene_id=example.scene_id,
                route_id=route_id,
                mentioned=counts.channel_id in mentioned,
                schema_valid=schema_valid,
                missing=missing,
            )

    if len(scenes) != _EXPECTED_SCENES:
        raise ValueError(f"expected {_EXPECTED_SCENES} scenes, got {len(scenes)}")
    return tuple(
        counts.row(route_count=len(routes), scene_count=len(scenes))
        for counts in accumulators
    )


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("category metric rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _row_int(row: Mapping[str, object], field: str) -> int:
    value = row[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"category metric field {field!r} must be an integer")
    return value


def _row_float(row: Mapping[str, object], field: str) -> float:
    value = row[field]
    if not isinstance(value, float):
        raise TypeError(f"category metric field {field!r} must be a float")
    return value


def _important_rows(
    rows: Sequence[dict[str, object]],
    *,
    min_target_scenes: int,
) -> tuple[dict[str, object], ...]:
    return tuple(
        row
        for row in rows
        if not bool(row["broad_environment_label"])
        and _row_int(row, "target_scene_count") >= min_target_scenes
        and _row_int(row, "target_example_count") > 0
    )


def _labels_to_annotate(
    rows: Sequence[dict[str, object]],
) -> frozenset[str]:
    important = frozenset((
        "door",
        "cabinet",
        "table",
        "toilet",
        "circulation",
        "work/study",
        "other/miscellaneous",
    ))
    available = frozenset(str(row["category_name"]) for row in rows)
    return important & available


def plot_category_metrics(
    rows: Sequence[dict[str, object]],
    output_path: Path,
    *,
    min_target_scenes: int,
) -> None:
    """Plot only supported, non-broad categories while retaining raw full rows."""
    selected = _important_rows(rows, min_target_scenes=min_target_scenes)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    colors = {"object": "#4472C4", "region": "#ED7D31"}
    for axis, group, title in zip(
        axes,
        ("object", "region"),
        ("Specific objects", "Regions"),
    ):
        group_rows = tuple(row for row in selected if row["category_group"] == group)
        if not group_rows:
            raise ValueError(f"no supported {group} categories to plot")
        maximum_support = max(
            _row_int(row, "target_example_count") for row in group_rows
        )
        labels = _labels_to_annotate(group_rows)
        for row in group_rows:
            support = _row_int(row, "target_example_count")
            axis.scatter(
                _row_float(row, "presence_f1"),
                _row_float(row, "spatial_iou"),
                s=35 + 180 * np.sqrt(support / maximum_support),
                color=colors[group],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.8,
            )
            name = str(row["category_name"])
            if name in labels:
                axis.annotate(
                    name.replace("_", " "),
                    (
                        _row_float(row, "presence_f1"),
                        _row_float(row, "spatial_iou"),
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
        axis.axvline(0.5, color="#AAAAAA", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Category-presence F1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Pooled category-cell IoU")
    axes[0].text(
        0.52,
        0.11,
        "Category often present,\nbut poorly placed",
        fontsize=8,
        color="#666666",
    )
    for axis in axes:
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.01, 0.15)
    fig.suptitle(
        "LLM-Grid epoch 2, R2R val_unseen (1,839 episodes / 11 scenes)\n"
        f"Bubble area = target episodes; shown when target scenes ≥ "
        f"{min_target_scenes}; broad channels omitted",
        fontsize=12,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: CategoryMetricArgs) -> dict[str, object]:
    if args.min_target_scenes < 1:
        raise ValueError("min_target_scenes must be positive")
    rows = collect_category_metrics(args)
    csv_path = args.output_root / "category_metrics.csv"
    summary_path = args.output_root / "category_metrics.json"
    _write_csv(csv_path, rows)
    selected = _important_rows(rows, min_target_scenes=args.min_target_scenes)
    summary = {
        "cache_model_key": args.cache_model_key,
        "dataset": "R2R",
        "split": "val_unseen",
        "definitions": {
            "presence_f1": "pooled episode-level category presence F1",
            "spatial_iou": "pooled category-aware cell IoU across episodes",
        },
        "plot_filter": {
            "broad_channels_excluded": sorted(_BROAD_OBJECT_NAMES),
            "minimum_target_scenes": args.min_target_scenes,
        },
        "hardest_supported": [
            {
                "category_group": row["category_group"],
                "category_name": row["category_name"],
                "presence_f1": row["presence_f1"],
                "spatial_iou": row["spatial_iou"],
                "target_example_count": row["target_example_count"],
                "target_scene_count": row["target_scene_count"],
            }
            for row in sorted(
                selected,
                key=lambda row: (
                    _row_float(row, "spatial_iou"),
                    str(row["category_name"]),
                ),
            )[:8]
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_category_metrics(
        rows,
        args.docs_image_path,
        min_target_scenes=args.min_target_scenes,
    )
    return {
        "csv_path": csv_path,
        "summary_path": summary_path,
        "plot_path": args.docs_image_path,
        "row_count": len(rows),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = CategoryMetricArgs(underscores_to_dashes=True).parse_args(argv)
    result = run(args)
    print(
        f"Wrote {result['row_count']} category rows to {result['csv_path']} "
        f"and plot to {result['plot_path']}"
    )


if __name__ == "__main__":
    main()
