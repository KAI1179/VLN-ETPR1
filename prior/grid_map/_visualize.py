"""Visualization methods for grid maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..constants import (
    COLS,
    ENVIRONMENTAL_OBJECT_CATEGORIES,
    MAPPED_OBJECT_COLORS,
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_COLORS,
    MAPPED_REGION_NAMES,
    OBJECT_CATEGORIES,
    REGION_CATEGORIES,
    ROWS,
)

plt.switch_backend("Agg")

if TYPE_CHECKING:
    from . import BaseGridMap


GridPoint = Tuple[float, float]
GridBounds = Tuple[int, int, int, int]


@dataclass(frozen=True)
class _MapOverlay:
    trajectory: tuple[GridPoint, ...] = ()
    keypoints: tuple[GridPoint, ...] = ()
    start_direction_vector: Optional[Tuple[float, float]] = None


OBJECT_COLORMAP = mcolors.ListedColormap(MAPPED_OBJECT_COLORS)
REGION_COLORMAP = mcolors.ListedColormap(MAPPED_REGION_COLORS)


def _grid_bounds(
    grid_maps: Sequence[BaseGridMap],
    overlays: Sequence[_MapOverlay],
    *,
    auto_crop: bool,
    crop_margin: int,
) -> GridBounds:
    if not auto_crop:
        return 0, ROWS - 1, 0, COLS - 1

    rows: list[float] = []
    cols: list[float] = []
    for grid_map in grid_maps:
        data_mask = np.max(grid_map.grid, axis=0) > 0
        if data_mask.any():
            map_rows, map_cols = np.where(data_mask)
            rows.extend(map_rows.tolist())
            cols.extend(map_cols.tolist())

    for overlay in overlays:
        for row, col in (*overlay.trajectory, *overlay.keypoints):
            rows.append(row)
            cols.append(col)

    if not rows:
        return 0, ROWS - 1, 0, COLS - 1

    return (
        max(0, int(np.floor(min(rows))) - crop_margin),
        min(ROWS - 1, int(np.ceil(max(rows))) + crop_margin),
        max(0, int(np.floor(min(cols))) - crop_margin),
        min(COLS - 1, int(np.ceil(max(cols))) + crop_margin),
    )


def _setup_axis(ax: Axes, title: str, bounds: GridBounds) -> None:
    row_min, row_max, col_min, col_max = bounds
    ax.set_title(title)
    ax.set_xlabel("Column (grid coordinates)")
    ax.set_ylabel("Row (grid coordinates)")
    ax.set_xlim(col_max + 1, col_min)
    ax.set_ylim(row_max + 1, row_min)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


def _draw_overlay(ax: Axes, overlay: _MapOverlay) -> None:
    if overlay.trajectory:
        rows, cols = zip(*overlay.trajectory)
        ax.plot(cols, rows, color="red", linewidth=1.5, zorder=20)
        ax.scatter(cols[0], rows[0], color="limegreen", marker="o", zorder=21)
        ax.scatter(cols[-1], rows[-1], color="red", marker="*", zorder=21)

    for index, (row, col) in enumerate(overlay.keypoints, start=1):
        ax.scatter(col, row, color="white", edgecolor="black", zorder=22)
        ax.annotate(
            str(index),
            (col, row),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            zorder=23,
        )

    if (
        overlay.trajectory
        and overlay.start_direction_vector is not None
        and overlay.start_direction_vector != (0.0, 0.0)
    ):
        start_row, start_col = overlay.trajectory[0]
        sin_value, cos_value = overlay.start_direction_vector
        ax.annotate(
            "",
            xy=(start_col - sin_value * 1.5, start_row - cos_value * 1.5),
            xytext=(start_col, start_row),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
            zorder=24,
        )


def _draw_layers(
    ax: Axes,
    layers: np.ndarray,
    category_ids: Sequence[int],
    category_names: Sequence[str],
    colormap: mcolors.Colormap,
    bounds: GridBounds,
) -> None:
    row_min, row_max, col_min, col_max = bounds
    extent = [col_min, col_max + 1, row_max + 1, row_min]
    cropped = layers[:, row_min : row_max + 1, col_min : col_max + 1]
    active_categories = sorted(
        [category for category in category_ids if cropped[category].max() > 1e-6],
        key=lambda category: cropped[category].sum(),
        reverse=True,
    )
    legend_elements = []
    color_denominator = max(len(category_names) - 1, 1)
    for category in active_categories:
        confidence = cropped[category]
        color = np.asarray(colormap(category / color_denominator))
        rgba = np.zeros((*confidence.shape, 4), dtype=np.float32)
        rgba[..., :3] = color[:3]
        rgba[..., 3] = np.clip(confidence / confidence.max(), 0, 1) * 0.8
        ax.imshow(rgba, interpolation="nearest", extent=extent)
        legend_elements.append(
            mpatches.Patch(
                facecolor=color,
                label=f"{category}: {category_names[category]}",
            )
        )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="center right",
            bbox_to_anchor=(0, 0.5),
            fontsize=8,
            frameon=True,
        )


def _draw_grid_map(
    axes: tuple[Axes, Axes],
    grid_map: BaseGridMap,
    bounds: GridBounds,
    overlay: _MapOverlay,
    *,
    label: str = "",
) -> None:
    object_title = "Object Categories"
    if label:
        object_title = f"{label} — {object_title}"
    _setup_axis(
        axes[0],
        f"{object_title}\n"
        f"({OBJECT_CATEGORIES - len(ENVIRONMENTAL_OBJECT_CATEGORIES)} categories, "
        f"excluding {len(ENVIRONMENTAL_OBJECT_CATEGORIES)} environmental)",
        bounds,
    )
    object_grid = grid_map.grid[:OBJECT_CATEGORIES].copy()
    for category in ENVIRONMENTAL_OBJECT_CATEGORIES:
        object_grid[category] = 0
    _draw_layers(
        axes[0],
        object_grid,
        [
            category
            for category in range(OBJECT_CATEGORIES)
            if category not in ENVIRONMENTAL_OBJECT_CATEGORIES
        ],
        MAPPED_OBJECT_NAMES,
        OBJECT_COLORMAP,
        bounds,
    )
    _draw_overlay(axes[0], overlay)

    region_title = "Region Categories"
    if label:
        region_title = f"{label} — {region_title}"
    _setup_axis(axes[1], region_title, bounds)
    _draw_layers(
        axes[1],
        grid_map.grid[OBJECT_CATEGORIES:],
        range(REGION_CATEGORIES),
        MAPPED_REGION_NAMES,
        REGION_COLORMAP,
        bounds,
    )
    _draw_overlay(axes[1], overlay)


def visualize(
    grid_map: BaseGridMap,
    save_path: str | Path,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (16, 6),
    auto_crop: bool = True,
    crop_margin: int = 5,
    positions: Sequence[GridPoint] = (),
    start_direction_vector: Optional[Tuple[float, float]] = None,
) -> None:
    """Visualize one cognitive map as object and region panels."""
    overlay = _MapOverlay(
        trajectory=tuple(positions),
        keypoints=tuple(positions),
        start_direction_vector=start_direction_vector,
    )
    bounds = _grid_bounds(
        [grid_map], [overlay], auto_crop=auto_crop, crop_margin=crop_margin
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    _draw_grid_map((axes[0, 0], axes[0, 1]), grid_map, bounds, overlay)
    fig.suptitle(
        title or f"{grid_map.__class__.__name__} Visualization",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_comparison(
    predicted_map: BaseGridMap,
    ground_truth_map: BaseGridMap,
    save_path: str | Path,
    *,
    instruction: str,
    ground_truth_trajectory: Sequence[GridPoint],
    predicted_trajectory_keypoints: Sequence[GridPoint],
    ground_truth_trajectory_keypoints: Sequence[GridPoint],
    start_direction_vector: Tuple[float, float],
    figsize: tuple[int, int] = (24, 14),
    auto_crop: bool = True,
    crop_margin: int = 5,
) -> None:
    """Visualize predicted and ground-truth maps with shared episode context."""
    predicted_overlay = _MapOverlay(
        trajectory=tuple(predicted_trajectory_keypoints),
        keypoints=tuple(predicted_trajectory_keypoints),
        start_direction_vector=start_direction_vector,
    )
    ground_truth_overlay = _MapOverlay(
        trajectory=tuple(ground_truth_trajectory),
        keypoints=tuple(ground_truth_trajectory_keypoints),
        start_direction_vector=start_direction_vector,
    )
    bounds = _grid_bounds(
        [predicted_map, ground_truth_map],
        [predicted_overlay, ground_truth_overlay],
        auto_crop=auto_crop,
        crop_margin=crop_margin,
    )
    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)
    _draw_grid_map(
        (axes[0, 0], axes[1, 0]),
        predicted_map,
        bounds,
        predicted_overlay,
        label="Predicted",
    )
    _draw_grid_map(
        (axes[0, 1], axes[1, 1]),
        ground_truth_map,
        bounds,
        ground_truth_overlay,
        label="Ground truth",
    )
    fig.suptitle(instruction, fontsize=14, fontweight="bold", wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(grid_map: BaseGridMap) -> None:
    """Print a compact summary of one cognitive map."""
    print(f"\n{'=' * 70}")
    print(f"{grid_map.__class__.__name__} Summary")
    print(f"{'=' * 70}")
    print(f"Grid shape: {grid_map.grid.shape} (channels x rows x cols)")
    has_objects = np.any(grid_map.grid[:OBJECT_CATEGORIES] > 0)
    has_regions = np.any(grid_map.grid[OBJECT_CATEGORIES:] > 0)
    print("\nData present:")
    print(f"  Objects: {'Yes' if has_objects else 'No'}")
    print(f"  Regions: {'Yes' if has_regions else 'No'}")
    if not has_objects and not has_regions:
        print(f"{'=' * 70}\n")
        return

    any_presence = np.any(grid_map.grid > 0, axis=0)
    rows_with_data = np.any(any_presence, axis=1)
    cols_with_data = np.any(any_presence, axis=0)
    if not np.any(rows_with_data) or not np.any(cols_with_data):
        print(f"{'=' * 70}\n")
        return

    row_start, row_end = np.where(rows_with_data)[0][[0, -1]]
    col_start, col_end = np.where(cols_with_data)[0][[0, -1]]
    print(
        f"\nBounding box: rows [{row_start}:{row_end + 1}], "
        f"cols [{col_start}:{col_end + 1}]"
    )
    print(f"Size: {row_end - row_start + 1} x {col_end - col_start + 1}")
    print(f"{'=' * 70}\n")
