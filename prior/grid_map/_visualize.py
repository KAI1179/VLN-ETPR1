"""Visualization methods for grid maps. Should not be used directly; use provided class methods instead."""

from __future__ import annotations


from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

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

# Set matplotlib backend to Agg for non-interactive plotting
plt.switch_backend("Agg")

if TYPE_CHECKING:
    from . import BaseGridMap


OBJECT_COLORMAP = mcolors.ListedColormap(MAPPED_OBJECT_COLORS)
"""Colormap with 27 intuitive colors for object category visualization."""

REGION_COLORMAP = mcolors.ListedColormap(MAPPED_REGION_COLORS)
"""Colormap with 10 intuitive colors for region category visualization."""


def visualize(
    grid_map: BaseGridMap,
    save_path: str | Path,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (16, 6),
    auto_crop: bool = True,
    crop_margin: int = 5,
    positions: List[Tuple[float, float]] = [],
    start_direction_vector: Optional[Tuple[float, float]] = None,
) -> None:
    """Visualize the grid map using matplotlib.

    Creates separate visualizations for object and region categories, showing
    the dominant category at each grid cell.

    Args:
        grid_map: The grid map instance to visualize.
        save_path: Path to save the figure.
        title: Optional title for the figure. If None, uses class name.
        figsize: Figure size as (width, height) in inches.
        auto_crop: Whether to automatically crop to data bounding box.
        crop_margin: Number of cells to add as margin around data (only used if auto_crop=True).
        positions: Optional list of 2D positions (in continuous grid coordinates) to overlay on the visualizations.
        start_direction_vector: Optional start orientation as a (sin, cos) vector.
    """
    if title is None:
        title = f"{grid_map.__class__.__name__} Visualization"

    # Calculate data bounding box for auto-cropping
    row_min, row_max = 0, ROWS - 1
    col_min, col_max = 0, COLS - 1

    if auto_crop:
        # Find the bounding box of all non-zero data
        data_mask = np.max(grid_map.grid, axis=0) > 0
        if data_mask.any():
            rows_with_data = np.where(data_mask.any(axis=1))[0]
            cols_with_data = np.where(data_mask.any(axis=0))[0]
            row_min = max(0, rows_with_data.min() - crop_margin)
            row_max = min(ROWS - 1, rows_with_data.max() + crop_margin)
            col_min = max(0, cols_with_data.min() - crop_margin)
            col_max = min(COLS - 1, cols_with_data.max() + crop_margin)
    extent = [col_min, col_max + 1, row_max + 1, row_min]

    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    axes = axes[0]  # Get first row

    def setup_ax(ax, title_: str):
        ax.set_title(title_)
        ax.set_xlabel("Column (grid coordinates)")
        ax.set_ylabel("Row (grid coordinates)")
        ax.set_xlim(col_max + 1, col_min)  # x increases right -> left
        ax.set_ylim(row_max + 1, row_min)  # y increases top -> bottom
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    def draw_positions(ax):
        if positions:
            if start_direction_vector is not None and start_direction_vector != (
                0.0,
                0.0,
            ):
                start_row, start_col = positions[0]
                sin_value, cos_value = start_direction_vector
                ax.annotate(
                    "",
                    xy=(start_col - sin_value * 1.5, start_row - cos_value * 1.5),
                    xytext=(start_col, start_row),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
                )

            for i in range(len(positions) - 1):
                p_curr = positions[i]
                p_next = positions[i + 1]
                ax.annotate(
                    "",
                    xy=(p_next[1], p_next[0]),
                    xytext=(p_curr[1], p_curr[0]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1),
                )

    # == Show objects ==

    ax = axes[0]
    setup_ax(
        ax,
        f"Object Categories\n"
        f"({OBJECT_CATEGORIES - len(ENVIRONMENTAL_OBJECT_CATEGORIES)} categories, "
        f"excluding {len(ENVIRONMENTAL_OBJECT_CATEGORIES)} environmental)",
    )

    object_grid = grid_map.grid[:OBJECT_CATEGORIES, :, :]

    filtered_object_grid = object_grid.copy()
    for cat_id in ENVIRONMENTAL_OBJECT_CATEGORIES:
        filtered_object_grid[cat_id, :, :] = 0

    cropped_object_grid = filtered_object_grid[
        :, row_min : row_max + 1, col_min : col_max + 1
    ]

    eps = 1e-6

    object_cat_order = sorted(
        [
            cat_id
            for cat_id in range(OBJECT_CATEGORIES)
            if cat_id not in ENVIRONMENTAL_OBJECT_CATEGORIES
            and cropped_object_grid[cat_id].max() > eps
        ],
        key=lambda c: cropped_object_grid[c].sum(),
        reverse=True,  # large categories first, small categories last
    )

    object_legend_elements = []

    for cat_id in object_cat_order:
        conf = cropped_object_grid[cat_id]

        color = np.array(OBJECT_COLORMAP(cat_id / (OBJECT_CATEGORIES - 1)))

        rgba = np.zeros((*conf.shape, 4), dtype=np.float32)
        rgba[..., :3] = color[:3]

        # Per-category alpha normalization makes small blurred objects visible.
        alpha = conf / conf.max()
        rgba[..., 3] = np.clip(alpha, 0, 1) * 0.8

        ax.imshow(rgba, interpolation="nearest", extent=extent)

        object_legend_elements.append(
            mpatches.Patch(
                facecolor=color,
                label=f"{cat_id}: {MAPPED_OBJECT_NAMES[cat_id]}",
            )
        )

    if object_legend_elements:
        ax.legend(
            handles=object_legend_elements,
            loc="center right",
            bbox_to_anchor=(0, 0.5),
            fontsize=8,
            frameon=True,
        )

    draw_positions(ax)

    # == Show regions ==
    ax = axes[1]
    setup_ax(ax, "Region Categories")

    region_grid = grid_map.grid[OBJECT_CATEGORIES:, :, :]
    cropped_region_grid = region_grid[:, row_min : row_max + 1, col_min : col_max + 1]

    region_cat_order = sorted(
        [i for i in range(REGION_CATEGORIES) if cropped_region_grid[i].max() > eps],
        key=lambda c: cropped_region_grid[c].sum(),
        reverse=True,
    )

    region_legend_elements = []

    for region_id in region_cat_order:
        conf = cropped_region_grid[region_id]

        color = np.array(REGION_COLORMAP(region_id / (REGION_CATEGORIES - 1)))

        rgba = np.zeros((*conf.shape, 4), dtype=np.float32)
        rgba[..., :3] = color[:3]

        alpha = conf / conf.max()
        rgba[..., 3] = np.clip(alpha, 0, 1) * 0.8

        ax.imshow(rgba, interpolation="nearest", extent=extent)

        region_legend_elements.append(
            mpatches.Patch(
                facecolor=color,
                label=f"{region_id}: {MAPPED_REGION_NAMES[region_id]}",
            )
        )

    if region_legend_elements:
        ax.legend(
            handles=region_legend_elements,
            loc="center right",
            bbox_to_anchor=(0, 0.5),
            fontsize=8,
            frameon=True,
        )

    draw_positions(ax)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(grid_map: BaseGridMap) -> None:
    """Print a text summary of the grid map to terminal.

    Shows which grid cells contain objects or regions using ASCII characters.
    Useful for quick inspection without matplotlib.

    Args:
        grid_map: The grid map instance to summarize.
    """
    print(f"\n{'=' * 70}")
    print(f"{grid_map.__class__.__name__} Summary")
    print(f"{'=' * 70}")
    print(f"Grid shape: {grid_map.grid.shape} (channels x rows x cols)")

    # Check if there's any data
    has_objects = np.any(grid_map.grid[:OBJECT_CATEGORIES, :, :] > 0)
    has_regions = np.any(grid_map.grid[OBJECT_CATEGORIES:, :, :] > 0)

    print("\nData present:")
    print(f"  Objects: {'Yes' if has_objects else 'No'}")
    print(f"  Regions: {'Yes' if has_regions else 'No'}")

    if not has_objects and not has_regions:
        print(f"{'=' * 70}\n")
        return

    # Find the bounding box of non-zero values
    object_grid = grid_map.grid[:OBJECT_CATEGORIES, :, :]
    region_grid = grid_map.grid[OBJECT_CATEGORIES:, :, :]

    object_presence = np.any(object_grid > 0, axis=0)
    region_presence = np.any(region_grid > 0, axis=0)
    any_presence = object_presence | region_presence

    rows_with_data = np.any(any_presence, axis=1)
    cols_with_data = np.any(any_presence, axis=0)

    if not np.any(rows_with_data) or not np.any(cols_with_data):
        print(f"{'=' * 70}\n")
        return

    row_start, row_end = np.where(rows_with_data)[0][[0, -1]]
    col_start, col_end = np.where(cols_with_data)[0][[0, -1]]

    bbox_rows = row_end - row_start + 1
    bbox_cols = col_end - col_start + 1

    print(
        f"\nBounding box: rows [{row_start}:{row_end + 1}], cols [{col_start}:{col_end + 1}]"
    )
    print(f"Size: {bbox_rows} x {bbox_cols}")

    print(f"{'=' * 70}\n")
