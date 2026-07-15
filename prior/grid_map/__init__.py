"""Grid map module for spatial representations of object and region categories.

This module provides:
- BaseGridMap: Base class with core grid operations
- GroundTruthGridMap: Ground truth maps with binary (0/1) values
- CognitiveGridMap: Incomplete maps with only data near the path available, and the values are in range [0, 1], representing belief
- Visualization: Methods for plotting and ASCII display (.visualize, .print_summary)
- Saving: `.save` saves the grid map to a `.npz` file.
"""

from __future__ import annotations
from pathlib import Path

from os import PathLike
from typing import List, Optional, Sequence, Tuple, Type, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from .._coords import meters_to_grid
from ..constants import COLS, OBJECT_CATEGORIES, REGION_CATEGORIES, ROWS
from prior.directions import DirectionVector
from prior.trajectory import TRAJECTORY_KEYPOINT_COUNT


GridMapT = TypeVar("GridMapT", bound="BaseGridMap")
Point2D = Tuple[float, float]


def _trim_zero_padded_points(points: Sequence[Point2D]) -> Sequence[Point2D]:
    last_nonzero_index = next(
        (
            index
            for index in range(len(points) - 1, -1, -1)
            if points[index] != (0.0, 0.0)
        ),
        -1,
    )
    return points[: last_nonzero_index + 1]


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

    def visualize_comparison(
        self,
        ground_truth_map: BaseGridMap,
        save_path: str | Path,
        *,
        instruction: str,
        ground_truth_trajectory: Sequence[Point2D],
        trajectory_keypoints: Sequence[Point2D],
        start_direction_vector: DirectionVector,
        figsize: tuple[int, int] = (24, 14),
        auto_crop: bool = True,
        crop_margin: int = 5,
    ) -> None:
        """Compare this predicted map with ground truth in one shared frame."""
        from ._visualize import visualize_comparison

        visible_keypoints = _trim_zero_padded_points(trajectory_keypoints)
        visualize_comparison(
            self,
            ground_truth_map,
            save_path,
            instruction=instruction,
            ground_truth_trajectory=[
                meters_to_grid(float(position[0]), float(position[1]))
                for position in ground_truth_trajectory
            ],
            trajectory_keypoints=[
                meters_to_grid(float(position[0]), float(position[1]))
                for position in visible_keypoints
            ],
            start_direction_vector=start_direction_vector,
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
            range_y=np.asarray(self.range_y, dtype=object),
        )

    @classmethod
    def load(
        cls: Type[GridMapT],
        load_path: str | PathLike[str],
    ) -> GridMapT:
        """Load from given npz file."""
        data = np.load(load_path, allow_pickle=True)
        grid_map = cls()
        grid_map.grid = data["grid"]
        grid_map.range_y = list(data["range_y"].tolist())
        return grid_map


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


class CognitiveGridMap(BaseGridMap):
    """
    Grid map for instruction-related data (cognitive map).

    This can be constructed either as predictions from instructions, or as GT
    from `GroundTruthGridMap`, by only keeping instruction-related categories
    along the path. Values are continuous in the range [0.0, 1.0], forming
    probability distributions over categories.
    """

    trajectory_keypoints: List[Point2D]
    """
    Five level-local trajectory keypoints as ``[x, z]`` points.
    """

    start_direction_vector: DirectionVector
    """Start orientation as a normalized (sin, cos) direction vector."""

    def __init__(self):
        """
        Initialize a cognitive grid map with zeros and no trajectory keypoints.
        """
        super().__init__()
        self.trajectory_keypoints = []
        self.start_direction_vector = (0.0, 0.0)

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
        the dominant category at each grid cell, along with trajectory keypoints.

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

        visible_keypoints = _trim_zero_padded_points(self.trajectory_keypoints)
        visualize(
            self,
            save_path,
            title=title,
            figsize=figsize,
            auto_crop=auto_crop,
            crop_margin=crop_margin,
            positions=[
                meters_to_grid(float(position[0]), float(position[1]))
                for position in visible_keypoints
            ],
            start_direction_vector=self.start_direction_vector,
        )

    def save(self, save_path: str | PathLike[str] | np._SupportsWrite[bytes]):
        """Save cognitive map data with trajectory keypoints."""
        trajectory_keypoints = np.asarray(self.trajectory_keypoints, dtype=np.float32)
        expected_shape = (TRAJECTORY_KEYPOINT_COUNT, 2)
        if trajectory_keypoints.shape != expected_shape:
            raise ValueError(
                "trajectory_keypoints must have shape "
                f"{expected_shape}, got {trajectory_keypoints.shape}"
            )
        np.savez_compressed(
            save_path,
            grid=self.grid,
            range_y=np.asarray(self.range_y, dtype=object),
            trajectory_keypoints=trajectory_keypoints,
            start_direction_vector=np.asarray(
                self.start_direction_vector,
                dtype=np.float32,
            ),
        )

    @classmethod
    def load(
        cls: Type[CognitiveGridMap],
        load_path: str | PathLike[str],
    ) -> CognitiveGridMap:
        """Load from given npz file."""
        data = np.load(load_path, allow_pickle=True)
        grid_map = cls()
        grid_map.grid = data["grid"]
        grid_map.range_y = list(data["range_y"].tolist())
        trajectory_keypoints = data["trajectory_keypoints"]
        expected_shape = (TRAJECTORY_KEYPOINT_COUNT, 2)
        if trajectory_keypoints.shape != expected_shape:
            raise ValueError(
                "trajectory_keypoints must have shape "
                f"{expected_shape}, got {trajectory_keypoints.shape}"
            )
        grid_map.trajectory_keypoints = [
            (float(position[0]), float(position[1]))
            for position in trajectory_keypoints
        ]
        grid_map.start_direction_vector = tuple(data["start_direction_vector"])
        return grid_map

    def validate(self) -> bool:
        """
        Validate the cognitive grid map, ensuring all values are within [0.0, 1.0].
        Returns:
            bool: True if the grid map is valid, False otherwise.
        """
        return super().validate()


__all__ = [
    "BaseGridMap",
    "GroundTruthGridMap",
    "CognitiveGridMap",
]
