"""Render compact qualitative and observation-evidence presentation figures."""

from __future__ import annotations

import csv
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Mapping, Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from tap import Tap

from prior._coords import meters_to_grid
from prior.bbox import RelevantSemanticBoxes
from prior.constants import (
    MAPPED_OBJECT_COLORS,
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_COLORS,
    MAPPED_REGION_NAMES,
    OBJECT_CATEGORIES,
)
from prior.grid_map import BaseGridMap
from prior.grid_map._visualize import (
    _MapOverlay,
    _draw_grid_map,
    _draw_overlay,
    _grid_bounds,
)
from prior.llm_grid_samples import downsample_grid
from vlnce_baselines.models.etp_llm.llm_grid_evidence import GridEvidenceIndex
from vlnce_baselines.models.etp_llm.llm_grid_train import GRID_SHAPE, parse_grid_text
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_prediction_path,
)


_EPOCH_CACHE_KEYS = {
    2: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2",
    10: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree",
}
_BROAD_OBJECT_IDS = frozenset(
    MAPPED_OBJECT_NAMES.index(name)
    for name in ("void", "structure", "other", "free-space")
)
_ERROR_COLORS = ("#2CA02C", "#D62728", "#1F77B4", "#9467BD")
_ERROR_LABELS = ("Clean TP", "FP", "FN", "FP + FN / substitution")


class PresentationVisualizationArgs(Tap):
    """Fixed completed artifacts and presentation-facing destinations."""

    navigation_root: Path = Path("data/llm_navigation")
    ground_truth_root: Path = Path(
        "data/cognitive_maps/gt.legacy.r1p5.direction5.blurred.v1"
    )
    target_root: Path = Path("data/cognitive_maps/gt.legacy.r1p5.direction5.v1")
    checkpoint_sweep_root: Path = Path("outputs/llm_grid_eval/checkpoint-sweep-r1p5")
    oracle_eval_root: Path = Path(
        "outputs/llm_grid_eval/oracle-t0-v1-e2-matched-common"
    )
    oracle_cache_model_key: str = "llm-grid-oracle-t0-v1-e2-matched"
    evidence_root: Path = Path("data/llm_grid_oracle_evidence")
    evidence_key: str = "oracle-t0-v1"
    output_root: Path = Path(
        "docs/data/llm_grid_presentation_2026_07_26/visualizations"
    )
    docs_image_root: Path = Path("docs/images/llm_grid_presentation")


@dataclass(frozen=True)
class EpisodeMetrics:
    scene_id: str
    example_id: str
    instruction: str
    raster_iou: float
    category_f1: float
    target_cell_count: int
    raw: Mapping[str, str]


def _number(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric evaluator field {field!r}") from error
    if not np.isfinite(value):
        raise ValueError(f"non-finite evaluator field {field!r}")
    return value


def _count(row: Mapping[str, str], field: str) -> int:
    value = _number(row, field)
    if value != int(value) or value < 0:
        raise ValueError(f"evaluator field {field!r} must be a non-negative integer")
    return int(value)


def _combined_category_f1(row: Mapping[str, str]) -> float:
    true_positives = sum(
        _count(row, f"{group}_category_true_positive_count")
        for group in ("object", "region")
    )
    predicted = sum(
        _count(row, f"{group}_category_predicted_count")
        for group in ("object", "region")
    )
    target = sum(
        _count(row, f"{group}_category_target_count") for group in ("object", "region")
    )
    denominator = predicted + target
    return 0.0 if denominator == 0 else 2.0 * true_positives / denominator


def _load_episode_metrics(path: Path) -> dict[str, EpisodeMetrics]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        result: dict[str, EpisodeMetrics] = {}
        for row in reader:
            if row.get("split") != "val_unseen":
                continue
            example_id = row.get("example_id", "")
            scene_id = row.get("scene_id", "")
            instruction = row.get("instruction", "")
            if not example_id or not scene_id:
                raise ValueError(f"missing episode identity in {path}")
            if example_id in result:
                raise ValueError(f"duplicate evaluator episode {example_id} in {path}")
            result[example_id] = EpisodeMetrics(
                scene_id=scene_id,
                example_id=example_id,
                instruction=instruction,
                raster_iou=_number(row, "category_aware_raster_iou"),
                category_f1=_combined_category_f1(row),
                target_cell_count=_count(row, "target_cell_count"),
                raw=dict(row),
            )
    if len(result) != 1839:
        raise ValueError(f"expected 1839 val_unseen rows in {path}, got {len(result)}")
    return result


def select_ordinary_examples(
    diagnostics_path: Path,
    epoch_2_rows: Mapping[str, EpisodeMetrics],
) -> tuple[tuple[str, EpisodeMetrics], ...]:
    """Select median, category-placement gap, and support-rich zero-IoU failure."""
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    representatives = diagnostics["splits"]["val_unseen"]["representatives"]
    selected = [
        ("median", epoch_2_rows[representatives["median"]["example_id"]]),
        (
            "category_spatial_gap",
            epoch_2_rows[representatives["largest_category_spatial_gap"]["example_id"]],
        ),
    ]
    valid = tuple(
        metrics
        for metrics in epoch_2_rows.values()
        if _number(metrics.raw, "schema_valid") == 1.0
    )
    minimum_iou = min(metrics.raster_iou for metrics in valid)
    failure = min(
        (metrics for metrics in valid if metrics.raster_iou == minimum_iou),
        key=lambda metrics: (-metrics.target_cell_count, metrics.example_id),
    )
    selected.append(("support_rich_failure", failure))
    return tuple(selected)


def select_oracle_representative(
    rows: Mapping[str, EpisodeMetrics],
) -> EpisodeMetrics:
    """Select the episode nearest the joint overall/unobserved medians."""
    eligible = tuple(
        metrics
        for metrics in rows.values()
        if _number(metrics.raw, "schema_valid") == 1.0
        and _count(metrics.raw, "unobserved_target_cell_count") >= 20
    )
    if not eligible:
        raise ValueError("no schema-valid Oracle episodes with unobserved support")
    overall_median = median(metrics.raster_iou for metrics in eligible)
    unobserved_median = median(
        _number(metrics.raw, "unobserved_category_aware_raster_iou")
        for metrics in eligible
    )
    return min(
        eligible,
        key=lambda metrics: (
            abs(metrics.raster_iou - overall_median)
            + abs(
                _number(
                    metrics.raw,
                    "unobserved_category_aware_raster_iou",
                )
                - unobserved_median
            ),
            metrics.example_id,
        ),
    )


def _raster_path(
    navigation_root: Path,
    cache_key: str,
    metrics: EpisodeMetrics,
) -> Path:
    return (
        navigation_root
        / cache_key
        / "r2r"
        / "val_unseen"
        / "cognitive_maps"
        / "raster"
        / metrics.scene_id
        / f"{metrics.example_id}.npz"
    )


def _episode_overlay(boxes: RelevantSemanticBoxes) -> _MapOverlay:
    trajectory = tuple(
        meters_to_grid(float(position[0]), float(position[1]))
        for position in boxes.ground_truth_trajectory
    )
    keypoints = tuple(
        meters_to_grid(float(position[0]), float(position[1]))
        for position in boxes.trajectory_keypoints
        if tuple(position) != (0.0, 0.0)
    )
    return _MapOverlay(
        trajectory=trajectory,
        keypoints=keypoints,
        start_direction_vector=boxes.start_direction_vector,
    )


def _ordinary_figure(
    *,
    ground_truth: BaseGridMap,
    epoch_maps: Mapping[int, BaseGridMap],
    epoch_metrics: Mapping[int, EpisodeMetrics],
    boxes: RelevantSemanticBoxes,
    role: str,
    output_path: Path,
) -> None:
    overlay = _episode_overlay(boxes)
    maps = (ground_truth, epoch_maps[2], epoch_maps[10])
    bounds = _grid_bounds(
        maps, (overlay, overlay, overlay), auto_crop=True, crop_margin=5
    )
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    labels = (
        "Ground truth",
        f"Epoch 2\nIoU {epoch_metrics[2].raster_iou:.3f} · "
        f"category F1 {epoch_metrics[2].category_f1:.3f}",
        f"Epoch 10\nIoU {epoch_metrics[10].raster_iou:.3f} · "
        f"category F1 {epoch_metrics[10].category_f1:.3f}",
    )
    for column, (grid_map, label) in enumerate(zip(maps, labels)):
        _draw_grid_map(
            (axes[0, column], axes[1, column]),
            grid_map,
            bounds,
            overlay,
            label=label,
        )
    title = (
        f"{role.replace('_', ' ').title()}: {epoch_metrics[2].example_id}\n"
        + textwrap.fill(boxes.instruction, width=115)
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_ordinary_examples(
    args: PresentationVisualizationArgs,
) -> tuple[dict[str, object], ...]:
    epoch_rows = {
        epoch: _load_episode_metrics(
            args.checkpoint_sweep_root / "runs" / f"{epoch - 1:03d}" / "episodes.csv"
        )
        for epoch in (2, 10)
    }
    selected = select_ordinary_examples(
        args.checkpoint_sweep_root / "runs/001/diagnostics.json",
        epoch_rows[2],
    )
    records: list[dict[str, object]] = []
    for role, selected_metrics in selected:
        metrics_by_epoch = {
            epoch: epoch_rows[epoch][selected_metrics.example_id] for epoch in (2, 10)
        }
        if any(
            metrics.scene_id != selected_metrics.scene_id
            for metrics in metrics_by_epoch.values()
        ):
            raise ValueError(
                f"scene changed across epochs: {selected_metrics.example_id}"
            )
        gt_path = (
            args.ground_truth_root
            / "raster"
            / selected_metrics.scene_id
            / f"{selected_metrics.example_id}.npz"
        )
        boxes_path = (
            args.ground_truth_root
            / "boxes"
            / selected_metrics.scene_id
            / f"{selected_metrics.example_id}.npz"
        )
        raster_paths = {
            epoch: _raster_path(
                args.navigation_root,
                _EPOCH_CACHE_KEYS[epoch],
                selected_metrics,
            )
            for epoch in (2, 10)
        }
        for path in (gt_path, boxes_path, *raster_paths.values()):
            if not path.is_file():
                raise FileNotFoundError(path)
        output_path = args.docs_image_root / "qualitative" / f"{role}.png"
        _ordinary_figure(
            ground_truth=BaseGridMap.load(gt_path),
            epoch_maps={
                epoch: BaseGridMap.load(path) for epoch, path in raster_paths.items()
            },
            epoch_metrics=metrics_by_epoch,
            boxes=RelevantSemanticBoxes.load(boxes_path),
            role=role,
            output_path=output_path,
        )
        records.append({
            "role": role,
            "selection": (
                "epoch-2 evaluator representative"
                if role != "support_rich_failure"
                else "minimum epoch-2 IoU, then maximum target cells, then ID"
            ),
            "scene_id": selected_metrics.scene_id,
            "example_id": selected_metrics.example_id,
            "figure_path": str(output_path),
            "epochs": {
                str(epoch): {
                    "cache_model_key": _EPOCH_CACHE_KEYS[epoch],
                    "raster_iou": metrics_by_epoch[epoch].raster_iou,
                    "category_f1": metrics_by_epoch[epoch].category_f1,
                    "raster_path": str(raster_paths[epoch]),
                }
                for epoch in (2, 10)
            },
        })
    return tuple(records)


def _grid_axis(axis: Axes, title: str, bounds: tuple[int, int, int, int]) -> None:
    row_min, row_max, col_min, col_max = bounds
    axis.set_title(title, fontsize=10)
    axis.set_xlim(col_max + 1, col_min)
    axis.set_ylim(row_max + 1, row_min)
    axis.set_aspect("equal")
    axis.xaxis.set_ticks([])
    axis.yaxis.set_ticks([])


def _semantic_layers(
    axis: Axes,
    layers: NDArray[np.bool_],
    category_ids: Sequence[int],
    category_names: Sequence[str],
    colors: Sequence[str],
    bounds: tuple[int, int, int, int],
) -> None:
    row_min, row_max, col_min, col_max = bounds
    extent = [col_min, col_max + 1, row_max + 1, row_min]
    cropped = layers[:, row_min : row_max + 1, col_min : col_max + 1]
    active = sorted(
        (category for category in category_ids if cropped[category].any()),
        key=lambda category: int(np.count_nonzero(cropped[category])),
        reverse=True,
    )
    handles = []
    for category in active:
        rgba = np.zeros((*cropped[category].shape, 4), dtype=np.float32)
        rgba[..., :3] = mcolors.to_rgb(colors[category])
        rgba[..., 3] = cropped[category] * 0.78
        axis.imshow(rgba, interpolation="nearest", extent=extent)
        handles.append(
            mpatches.Patch(
                facecolor=colors[category],
                label=category_names[category].replace("_", " "),
            )
        )
    if handles:
        axis.legend(
            handles=handles,
            loc="upper left",
            fontsize=6,
            framealpha=0.88,
            ncols=1,
        )


def _observation_geometry(
    axis: Axes,
    observed: NDArray[np.bool_],
    free: NDArray[np.bool_],
    bounds: tuple[int, int, int, int],
) -> None:
    row_min, row_max, col_min, col_max = bounds
    extent = [col_min, col_max + 1, row_max + 1, row_min]
    unknown = ~observed[row_min : row_max + 1, col_min : col_max + 1]
    rgba = np.zeros((*unknown.shape, 4), dtype=np.float32)
    rgba[..., :3] = mcolors.to_rgb("#BFBFBF")
    rgba[..., 3] = unknown * 0.48
    axis.imshow(rgba, interpolation="nearest", extent=extent, zorder=-5)
    free_crop = free[row_min : row_max + 1, col_min : col_max + 1]
    free_rgba = np.zeros((*free_crop.shape, 4), dtype=np.float32)
    free_rgba[..., :3] = mcolors.to_rgb("#DDEBF7")
    free_rgba[..., 3] = free_crop * 0.85
    axis.imshow(free_rgba, interpolation="nearest", extent=extent, zorder=-4)
    _observed_contour(axis, observed)


def _observed_contour(
    axis: Axes,
    observed: NDArray[np.bool_],
) -> None:
    rows, cols = observed.shape
    axis.contour(
        np.arange(cols, dtype=np.float32) + 0.5,
        np.arange(rows, dtype=np.float32) + 0.5,
        observed.astype(np.float32),
        levels=(0.5,),
        colors=("#00A6D6",),
        linewidths=1.2,
        zorder=15,
    )


def _route_overlay(
    axis: Axes,
    boxes: RelevantSemanticBoxes,
    *,
    scale: float,
) -> None:
    trajectory = tuple(
        (row / scale, col / scale)
        for x, z in boxes.ground_truth_trajectory
        for row, col in (meters_to_grid(float(x), float(z)),)
    )
    _draw_overlay(axis, _MapOverlay(trajectory=trajectory))


def _oracle_bounds(
    target: NDArray[np.bool_],
    evidence: NDArray[np.bool_],
    predicted: NDArray[np.bool_],
    observed: NDArray[np.bool_],
    boxes: RelevantSemanticBoxes,
) -> tuple[int, int, int, int]:
    support = np.any(target | evidence | predicted, axis=0) | observed
    rows, cols = np.where(support)
    route_points = tuple(
        tuple(coordinate / 2 for coordinate in meters_to_grid(float(x), float(z)))
        for x, z in boxes.ground_truth_trajectory
    )
    rows = np.concatenate((
        rows,
        np.asarray([point[0] for point in route_points], dtype=np.float32),
    ))
    cols = np.concatenate((
        cols,
        np.asarray([point[1] for point in route_points], dtype=np.float32),
    ))
    return (
        max(0, int(np.floor(rows.min())) - 3),
        min(49, int(np.ceil(rows.max())) + 3),
        max(0, int(np.floor(cols.min())) - 3),
        min(49, int(np.ceil(cols.max())) + 3),
    )


def _error_map(
    predicted: NDArray[np.bool_],
    target: NDArray[np.bool_],
    observed: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    unobserved = ~observed
    true_positive = np.any(predicted & target, axis=0) & unobserved
    false_positive = np.any(predicted & ~target, axis=0) & unobserved
    false_negative = np.any(~predicted & target, axis=0) & unobserved
    result = np.zeros(observed.shape, dtype=np.uint8)
    result[true_positive & ~false_positive & ~false_negative] = 1
    result[false_positive & ~false_negative] = 2
    result[false_negative & ~false_positive] = 3
    result[false_positive & false_negative] = 4
    return result


def _binary_metrics(
    predicted: NDArray[np.bool_],
    target: NDArray[np.bool_],
) -> tuple[float, float, float]:
    true_positives = int(np.count_nonzero(predicted & target))
    predicted_count = int(np.count_nonzero(predicted))
    target_count = int(np.count_nonzero(target))
    union = predicted_count + target_count - true_positives
    iou = 0.0 if union == 0 else true_positives / union
    precision = 0.0 if predicted_count == 0 else true_positives / predicted_count
    recall = 0.0 if target_count == 0 else true_positives / target_count
    return iou, precision, recall


def _verify_metric(
    actual: float,
    row: Mapping[str, str],
    field: str,
) -> None:
    expected = _number(row, field)
    if not np.isclose(actual, expected, atol=1e-12):
        raise ValueError(
            f"oracle visualization metric mismatch for {field}: "
            f"computed {actual}, evaluator {expected}"
        )


def _draw_error(
    axis: Axes,
    error_map: NDArray[np.uint8],
    observed: NDArray[np.bool_],
    bounds: tuple[int, int, int, int],
) -> None:
    row_min, row_max, col_min, col_max = bounds
    cropped = error_map[row_min : row_max + 1, col_min : col_max + 1]
    masked = np.ma.masked_equal(cropped, 0)
    axis.imshow(
        masked,
        cmap=mcolors.ListedColormap(_ERROR_COLORS),
        norm=mcolors.BoundaryNorm((0.5, 1.5, 2.5, 3.5, 4.5), 4),
        interpolation="nearest",
        extent=[col_min, col_max + 1, row_max + 1, row_min],
    )
    _observed_contour(axis, observed)
    axis.legend(
        handles=[
            mpatches.Patch(facecolor=color, label=label)
            for color, label in zip(_ERROR_COLORS, _ERROR_LABELS)
        ],
        loc="upper left",
        fontsize=6,
        framealpha=0.88,
    )


def render_oracle_comparison(
    args: PresentationVisualizationArgs,
) -> dict[str, object]:
    rows = _load_episode_metrics(args.oracle_eval_root / "episodes.csv")
    metrics = select_oracle_representative(rows)
    evidence_index = GridEvidenceIndex.load(
        args.evidence_root,
        args.evidence_key,
        "R2R",
        "val_unseen",
    )
    episode = evidence_index.episode(metrics.example_id)
    if episode.scene_id != metrics.scene_id:
        raise ValueError(f"oracle evidence scene mismatch: {metrics.example_id}")
    evidence = evidence_index.load_evidence(metrics.example_id)
    prediction_path = llm_navigation_prediction_path(
        metrics.scene_id,
        metrics.example_id,
        "R2R",
        "val_unseen",
        cache_dir=args.navigation_root,
        model_key=args.oracle_cache_model_key,
    )
    predicted = (
        parse_grid_text(
            prediction_path.read_text(encoding="utf-8"),
            shape=GRID_SHAPE,
        ).grid
        > 0
    )
    target_path = (
        args.target_root / "raster" / metrics.scene_id / f"{metrics.example_id}.npz"
    )
    boxes_path = (
        args.target_root / "boxes" / metrics.scene_id / f"{metrics.example_id}.npz"
    )
    target = downsample_grid(BaseGridMap.load(target_path).grid, 2) > 0
    prompt_evidence = evidence.prompt_semantic_grid
    if target.shape != GRID_SHAPE or prompt_evidence.shape != GRID_SHAPE:
        raise ValueError("oracle comparison inputs are not aligned at 37x50x50")
    overall_iou, _, _ = _binary_metrics(predicted, target)
    unobserved = ~evidence.target_observed_mask
    unobserved_iou, unobserved_precision, unobserved_recall = _binary_metrics(
        predicted & unobserved[None, ...],
        target & unobserved[None, ...],
    )
    evidence_only_iou, _, _ = _binary_metrics(prompt_evidence, target)
    displayed_target = np.array(target, copy=True)
    displayed_evidence = np.array(prompt_evidence, copy=True)
    for category in _BROAD_OBJECT_IDS:
        displayed_target[category] = False
        displayed_evidence[category] = False
    displayed_evidence_iou, _, _ = _binary_metrics(
        displayed_evidence,
        displayed_target,
    )
    _verify_metric(overall_iou, metrics.raw, "category_aware_raster_iou")
    _verify_metric(
        evidence_only_iou,
        metrics.raw,
        "evidence_only_category_aware_raster_iou",
    )
    _verify_metric(
        unobserved_iou,
        metrics.raw,
        "unobserved_category_aware_raster_iou",
    )
    _verify_metric(
        unobserved_precision,
        metrics.raw,
        "unobserved_cell_precision",
    )
    _verify_metric(unobserved_recall, metrics.raw, "unobserved_cell_recall")
    boxes = RelevantSemanticBoxes.load(boxes_path)
    bounds = _oracle_bounds(
        target,
        prompt_evidence,
        predicted,
        evidence.target_observed_mask,
        boxes,
    )
    figure_path = args.docs_image_root / "oracle_t0_gt_evidence_prediction.png"
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), squeeze=False)
    column_titles = (
        "Full route-relevant GT",
        (
            "t=0 observation-bounded evidence\n"
            f"Displayed-category IoU {displayed_evidence_iou:.3f}"
        ),
        f"Raw LLM-Grid-Oracle prediction\nOverall IoU {metrics.raster_iou:.3f}",
        (
            "Unobserved category-aware errors\n"
            f"IoU {_number(metrics.raw, 'unobserved_category_aware_raster_iou'):.3f} · "
            f"P {_number(metrics.raw, 'unobserved_cell_precision'):.3f} · "
            f"R {_number(metrics.raw, 'unobserved_cell_recall'):.3f}"
        ),
    )
    group_specs = (
        (
            "Specific objects",
            tuple(
                category
                for category in range(OBJECT_CATEGORIES)
                if category not in _BROAD_OBJECT_IDS
            ),
            MAPPED_OBJECT_NAMES,
            MAPPED_OBJECT_COLORS,
            slice(0, OBJECT_CATEGORIES),
        ),
        (
            "Regions",
            tuple(range(len(MAPPED_REGION_NAMES))),
            MAPPED_REGION_NAMES,
            MAPPED_REGION_COLORS,
            slice(OBJECT_CATEGORIES, None),
        ),
    )
    for row_index, (group, category_ids, names, colors, channels) in enumerate(
        group_specs
    ):
        group_target = target[channels]
        group_evidence = prompt_evidence[channels]
        group_predicted = predicted[channels]
        for column, title in enumerate(column_titles):
            _grid_axis(
                axes[row_index, column],
                f"{title}\n{group}",
                bounds,
            )
        _semantic_layers(
            axes[row_index, 0],
            group_target,
            category_ids,
            names,
            colors,
            bounds,
        )
        _observation_geometry(
            axes[row_index, 1],
            evidence.target_observed_mask,
            evidence.target_free_mask,
            bounds,
        )
        _semantic_layers(
            axes[row_index, 1],
            group_evidence,
            category_ids,
            names,
            colors,
            bounds,
        )
        _semantic_layers(
            axes[row_index, 2],
            group_predicted,
            category_ids,
            names,
            colors,
            bounds,
        )
        _draw_error(
            axes[row_index, 3],
            _error_map(
                group_predicted,
                group_target,
                evidence.target_observed_mask,
            ),
            evidence.target_observed_mask,
            bounds,
        )
        _route_overlay(axes[row_index, 0], boxes, scale=2)
        _route_overlay(axes[row_index, 2], boxes, scale=2)
    fig.suptitle(
        f"Representative observation-evidence comparison: {metrics.example_id}\n"
        f"Instruction: {textwrap.fill(metrics.instruction.strip(), width=115)}",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Grey = unknown at t=0; pale blue = observed free; cyan = observed "
        "boundary; red line = continuous GT route.\n"
        "Displayed-category IoU excludes hidden broad channels "
        f"(void, structure, other, free-space); all-37-channel evidence-only "
        f"IoU = {evidence_only_iou:.3f}. The error panel scores only cells "
        "outside the observed mask.",
        ha="center",
        fontsize=9,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "role": "representative",
        "selection": (
            "nearest joint overall/unobserved median among schema-valid episodes "
            "with at least 20 unobserved target category-cells"
        ),
        "scene_id": metrics.scene_id,
        "example_id": metrics.example_id,
        "figure_path": str(figure_path),
        "prediction_path": str(prediction_path),
        "target_path": str(target_path),
        "evidence_manifest_sha256": evidence_index.manifest_sha256,
        "metrics": {
            "overall_iou": metrics.raster_iou,
            "evidence_only_iou": _number(
                metrics.raw, "evidence_only_category_aware_raster_iou"
            ),
            "displayed_category_evidence_iou": displayed_evidence_iou,
            "unobserved_iou": _number(
                metrics.raw, "unobserved_category_aware_raster_iou"
            ),
            "unobserved_precision": _number(metrics.raw, "unobserved_cell_precision"),
            "unobserved_recall": _number(metrics.raw, "unobserved_cell_recall"),
            "unobserved_target_cells": _count(
                metrics.raw, "unobserved_target_cell_count"
            ),
        },
    }


def run(args: PresentationVisualizationArgs) -> dict[str, object]:
    ordinary = render_ordinary_examples(args)
    oracle = render_oracle_comparison(args)
    manifest = {
        "ordinary_llm_grid": ordinary,
        "observation_evidence_comparison": oracle,
    }
    manifest_path = args.output_root / "qualitative_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"manifest_path": manifest_path, **manifest}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = PresentationVisualizationArgs(underscores_to_dashes=True).parse_args(argv)
    result = run(args)
    ordinary = result["ordinary_llm_grid"]
    if not isinstance(ordinary, tuple):
        raise TypeError("ordinary_llm_grid result must be a tuple")
    print(
        f"Wrote {len(ordinary)} qualitative figures and "
        f"the observation-evidence comparison; manifest: {result['manifest_path']}"
    )


if __name__ == "__main__":
    main()
