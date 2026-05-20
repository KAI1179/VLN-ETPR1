"""Grid map module for spatial representations of object and region categories.

This module provides:
- BaseGridMap: Base class with core grid operations
- GroundTruthGridMap: Ground truth maps with binary (0/1) values
- CognitiveGridMap: Incomplete maps with only data near the path available, and the values are in range [0, 1], representing belief
- Construction: Functions to build maps from MP3D scenes (.from_scene, .from_scene_id, only for GroundTruthGridMap)
- Visualization: Methods for plotting and ASCII display (.visualize, .print_summary)
- Saving: `.save` saves the grid map to a `.npz` file.
"""

from __future__ import annotations
from pathlib import Path

from os import PathLike
from typing import List, Optional, Tuple, cast
from copy import deepcopy

import numpy as np
from habitat_sim.scene import SemanticScene
from numpy.typing import NDArray

from ..constants import (
    CELL_SIZE,
    COLS,
    OBJECT_CATEGORIES,
    REGION_CATEGORIES,
    ROWS,
)
from prior.directions import DirectionVector


class BaseGridMap:
    """
    Base class for grid-based spatial representations of object and region categories.

    This class provides the common data structure and interface for both ground truth
    and predicted grid maps. Both map types share the same grid dimensions and
    fundamental constraints but differ in their validation rules and typical operations.

    The grid represents spatial information in a discrete 2D grid where each cell
    contains probability distributions over object and region categories.
    """

    grid: NDArray[np.float32]
    """
    Inner grid data of dimension (OBJECT_CATEGORIES + REGION_CATEGORIES) x ROWS x COLS.

    Each element is in the range [0.0, 1.0], representing the confidence level of the
    presence of a specific category at a specific grid cell.
    """

    offset_x: float = 0.0
    """
    X offset to transform world coordinates to grid coordinates, in order to keep indexing positive.
    """

    offset_z: float = 0.0
    """
    Z offset to transform world coordinates to grid coordinates, in order to keep indexing positive.
    """

    range_y: List[Optional[float]]
    """
    Y range ``[min_y, max_y)`` of the level this map represents, in world coordinates.

    - ``None`` lower bound means the level is the lowest and accepts any Y below the upper bound.
    - ``None`` upper bound means the level is the highest and accepts any Y above the lower bound.
    - Both ``None`` means the map is single-level and accepts all waypoints regardless of Y.

    Derived from the minimum ``region.aabb.min.y`` across the level's regions, which
    corresponds to the floor surface Y after the MP3D coordinate rotation.  The upper
    bound is set to the next level's minimum region floor Y so that staircase waypoints
    are attributed to the correct level.
    """

    def __init__(self):
        """
        Initialize a grid map with zeros.
        """
        self.grid = np.zeros(
            (OBJECT_CATEGORIES + REGION_CATEGORIES, ROWS, COLS), dtype=np.float32
        )
        # Initialise as a mutable list so subclasses can update bounds without
        # sharing the same list across instances (class-level defaults would be
        # shared).
        self.range_y: List[Optional[float]] = [None, None]

    def __add__(self, other: BaseGridMap) -> BaseGridMap:
        """
        Add two grid maps element-wise, clamping values to [0.0, 1.0].

        Args:
            other: Another BaseGridMap instance to add.
        Returns:
            BaseGridMap: New grid map resulting from the addition.
        """
        result = BaseGridMap()
        result.grid = np.clip(self.grid + other.grid, 0.0, 1.0)
        return result

    def validate(self) -> bool:
        """
        Validate the grid map, ensuring all values are within [0.0, 1.0].
        Returns:
            bool: True if the grid map is valid, False otherwise.
        """
        if not np.all((0.0 <= self.grid) & (self.grid <= 1.0)):
            return False
        return True

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """
        Transform grid coordinates to world coordinates (cell center).

        Args:
            row: Row index of the grid cell.
            col: Column index of the grid cell.

        Returns:
            tuple[float, float]: World coordinates (x, z) of the cell center.
        """
        x = row * CELL_SIZE + CELL_SIZE / 2.0 + self.offset_x
        z = col * CELL_SIZE + CELL_SIZE / 2.0 + self.offset_z
        return (x, z)

    def world_to_grid(self, x: float, z: float) -> tuple[float, float]:
        """
        Transform world coordinates to continuous grid coordinates.

        Args:
            x: World x-coordinate.
            z: World z-coordinate.

        Returns:
            tuple[float, float]: Grid coordinates.
        """
        row = (x - self.offset_x) / CELL_SIZE
        col = (z - self.offset_z) / CELL_SIZE
        return (row, col)

    def get_object_cell(self, row: int, col: int, mapped_category: int) -> float:
        """
        Get the value of the object cell at (row, col) for the given mapped category.
        Args:
            row (int): Row index of the grid cell.
            col (int): Column index of the grid cell.
            mapped_category (int): Mapped object category index.
        Returns:
            float: Value of the cell if indices are valid, None otherwise.
        """
        if not (0 <= row < ROWS and 0 <= col < COLS):
            raise IndexError("Grid cell index out of bounds.")
        if not (0 <= mapped_category < OBJECT_CATEGORIES):
            raise IndexError("Mapped category index out of bounds.")
        return self.grid[mapped_category, row, col]

    def get_region_cell(self, row: int, col: int, mapped_category: int) -> float:
        """
        Get the value of the region cell at (row, col) for the given mapped category.
        Args:
            row (int): Row index of the grid cell.
            col (int): Column index of the grid cell.
            mapped_category (int): Mapped region category index.
        Returns:
            float: Value of the cell if indices are valid, None otherwise.
        """
        if not (0 <= row < ROWS and 0 <= col < COLS):
            raise IndexError("Grid cell index out of bounds.")
        if not (0 <= mapped_category < REGION_CATEGORIES):
            raise IndexError("Mapped category index out of bounds.")
        region_category_index = OBJECT_CATEGORIES + mapped_category
        return self.grid[region_category_index, row, col]

    def set_object_cell(
        self, row: int, col: int, mapped_category: int, value: int = 1
    ) -> bool:
        """
        Set the object cell at (row, col) to value for the given mapped category.
        Args:
            row (int): Row index of the grid cell.
            col (int): Column index of the grid cell.
            mapped_category (int): Mapped object category index.
            value (int): Value to set (default is 1).
        Returns:
            bool: True if the cell was set successfully, False otherwise.
        """
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return False
        self.grid[mapped_category, row, col] = value
        return True

    def set_region_cell(
        self, row: int, col: int, mapped_category: int, value: int = 1
    ) -> bool:
        """
        Set the region cell at (row, col) to value for the given mapped category.
        Args:
            row (int): Row index of the grid cell.
            col (int): Column index of the grid cell.
            mapped_category (int): Mapped region category index.
            value (int): Value to set (default is 1).
        Returns:
            bool: True if the cell was set successfully, False otherwise.
        """
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return False
        region_category_index = OBJECT_CATEGORIES + mapped_category
        self.grid[region_category_index, row, col] = value
        return True

    def visualize(
        self,
        save_path: str | Path,
        title: Optional[str] = None,
        figsize: tuple[int, int] = (16, 6),
        auto_crop: bool = True,
        crop_margin: int = 5,
    ) -> None:
        """Visualize the grid map using matplotlib.

        Creates separate visualizations for object and region categories, showing
        the dominant category at each grid cell.

        Args:
            save_path: Path to save the figure.
            title: Optional title for the figure. If None, uses class name.
            show_objects: Whether to show object categories visualization.
            show_regions: Whether to show region categories visualization.
            figsize: Figure size as (width, height) in inches.
            auto_crop: Whether to automatically crop to data bounding box.
            crop_margin: Number of cells to add as margin around data (only used if auto_crop=True).
        """
        from ._visualize import visualize

        visualize(
            self,
            save_path,
            title=title,
            figsize=figsize,
            auto_crop=auto_crop,
            crop_margin=crop_margin,
        )

    def print_summary(self) -> None:
        """Print a text summary of the grid map to terminal.

        Shows which grid cells contain objects or regions using ASCII characters.
        Useful for quick inspection without matplotlib.
        """
        from ._visualize import print_summary

        print_summary(self)

    def is_empty(self) -> bool:
        """Check if the grid map is empty (all zeros)."""
        return cast(bool, np.all(self.grid == 0.0))

    def save(self, save_path: str | PathLike[str] | np._SupportsWrite[bytes]):
        """Save the data with `np.savez_compressed`."""
        np.savez_compressed(
            save_path,
            grid=self.grid,
            offset_x=self.offset_x,
            offset_z=self.offset_z,
            range_y=np.asarray(self.range_y, dtype=object),
        )


class GroundTruthGridMap(BaseGridMap):
    """
    Grid map for ground truth data (semantic map).

    This class represents the actual ground truth spatial layout where each grid cell
    has a definitive object and region category assignment. Values are restricted to
    binary (0.0 or 1.0) to represent whether a category is present or absent in each cell.
    """

    def __init__(self):
        """
        Initialize a ground truth grid map with zeros.
        """
        super().__init__()

    def validate(self) -> bool:
        """
        Validate the ground truth grid map, ensuring all values are either 0.0 or 1.0.
        Returns:
            bool: True if the grid map is valid, False otherwise.
        """
        if not np.all(np.isin(self.grid, [0.0, 1.0])):
            return False
        return True

    def to_cognitive_map(
        self,
        instruction: str,
        positions: List[List[float]],
        start_direction_vector: DirectionVector,
    ) -> CognitiveGridMap:
        """
        Convert this ground truth grid map to a cognitive grid map, only
        keeping instruction-related categories along the path defined by positions.

        Args:
            instruction: Instruction text to determine relevant categories.
            positions: List of 3D positions (x, y, z) along the path.
            start_direction_vector: Start orientation as (sin, cos).

        Returns:
            CognitiveGridMap: New cognitive grid map with the same data.
        """
        from ._cognitive import build_cognitive_map

        return build_cognitive_map(
            self,
            instruction,
            positions,
            start_direction_vector=start_direction_vector,
        )

    @staticmethod
    def from_scene(semantic_scene: SemanticScene) -> List[GroundTruthGridMap]:
        """Constructs level-wise ground-truth grid maps from the given semantic scene. If you intend to work with MP3D scenes, consider using from_scene_id instead."""
        from ._construct import construct_grid_maps_from_scene

        return construct_grid_maps_from_scene(semantic_scene)

    @staticmethod
    def from_scene_id(scene_id: str) -> List[GroundTruthGridMap]:
        """Constructs level-wise ground-truth grid maps from the given MP3D scene ID. The return value is cached to improve performance, and copied to avoid mutation."""
        from ._construct import construct_grid_maps_from_scene_id

        return [deepcopy(m) for m in construct_grid_maps_from_scene_id(scene_id)]


class CognitiveGridMap(BaseGridMap):
    """
    Grid map for instruction-related data (cognitive map).

    This can be constructed either as predictions from instructions, or as GT
    from `GroundTruthGridMap`, by only keeping instruction-related categories
    along the path. Values are continuous in the range [0.0, 1.0], forming
    probability distributions over categories.
    """

    positions: List[Tuple[float, float]]
    """
    List of 2D positions (in continuous grid coordinates) along the path.
    """

    direction_vectors: List[Tuple[float, float]]
    """Direction vectors."""

    start_direction_vector: DirectionVector
    """Start orientation as a normalized (sin, cos) direction vector."""

    def __init__(self):
        """
        Initialize a cognitive grid map with zeros and empty positions.
        """
        super().__init__()
        self.positions = []
        self.direction_vectors = []

    def visualize(
        self,
        save_path: str | Path,
        title: str | None = None,
        figsize: tuple[int, int] = (16, 6),
        auto_crop: bool = True,
        crop_margin: int = 5,
    ) -> None:
        """Visualize the cognitive grid map using matplotlib.
        Creates separate visualizations for object and region categories, showing
        the dominant category at each grid cell, along with the path positions.

        Args:
            save_path: Path to save the figure.
            title: Optional title for the figure. If None, uses class name.
            show_objects: Whether to show object categories visualization.
            show_regions: Whether to show region categories visualization.
            figsize: Figure size as (width, height) in inches.
            auto_crop: Whether to automatically crop to data bounding box.
            crop_margin: Number of cells to add as margin around data (only used if auto_crop=True).
        """
        from ._visualize import visualize

        visualize(
            self,
            save_path,
            title=title,
            figsize=figsize,
            auto_crop=auto_crop,
            crop_margin=crop_margin,
            positions=self.positions,
            direction_vectors=self.direction_vectors,
            start_direction_vector=self.start_direction_vector,
        )

    def save(self, save_path: str | PathLike[str] | np._SupportsWrite[bytes]):
        """Save the cognitive map data with direction vectors."""
        np.savez_compressed(
            save_path,
            grid=self.grid,
            offset_x=self.offset_x,
            offset_z=self.offset_z,
            # range_y=np.asarray(self.range_y, dtype=object),
            direction_vectors=np.asarray(self.direction_vectors, dtype=np.float32),
            start_direction_vector=np.asarray(
                self.start_direction_vector,
                dtype=np.float32,
            ),
            start_position=np.asarray(self.positions[0], dtype=np.float32),
        )

    def validate(self) -> bool:
        """
        Validate the cognitive grid map, ensuring all values are within [0.0, 1.0].
        Returns:
            bool: True if the grid map is valid, False otherwise.
        """
        return super().validate()


def first_encountered_level(
    grid_maps: List[GroundTruthGridMap],
    positions: List[List[float]],
) -> Tuple[int, GroundTruthGridMap]:
    """Select the first map level encountered by a waypoint path."""
    if len(grid_maps) == 0:
        raise ValueError("GroundTruthGridMap.from_scene_id returned no levels")

    for position in positions:
        y = float(position[1])
        for level, grid_map in enumerate(grid_maps):
            lower, upper = grid_map.range_y
            if (lower is None or y >= lower) and (upper is None or y < upper):
                return level, grid_map

    return 0, grid_maps[0]


__all__ = [
    "BaseGridMap",
    "GroundTruthGridMap",
    "CognitiveGridMap",
    "first_encountered_level",
]
