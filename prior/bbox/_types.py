"""Data types for semantic bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, field_validator

from prior.directions import DirectionVector
from prior.trajectory import (
    LevelTrajectory2D,
    Point2D,
    TRAJECTORY_KEYPOINT_COUNT,
    WorldTrajectory3D,
)
from ..constants import CELL_SIZE, COLS, MAX_DISTANCE_CELLS, ROWS

Axis2D = Tuple[float, float]
IRRELEVANT_MULTIPLIER = 0.6
"""Confidence multiplier for trajectory-near boxes not mentioned in the instruction."""

MAX_GRID_X = (ROWS - 1) * CELL_SIZE
MAX_GRID_Z = (COLS - 1) * CELL_SIZE


@dataclass(frozen=True)
class OBB2D:
    """2D object bounding box projected onto world X/Z axes."""

    center: Point2D
    half_extents: Point2D
    rotation: float = 0.0
    mentioned: bool = False


@dataclass(frozen=True)
class AABB2D:
    """2D axis-aligned bounding box projected onto world X/Z axes."""

    min: Point2D
    max: Point2D
    mentioned: bool = False


ObjectOBB2Ds = List[List[OBB2D]]
"""List indexed by mapped object category id, containing object OBBs of that type."""

RegionAABB2Ds = List[List[AABB2D]]
"""List indexed by mapped region category id, containing region AABBs of that type."""


class LevelSemanticBoxes(BaseModel):
    """2D semantic boxes for one MP3D semantic level."""

    objects: ObjectOBB2Ds
    regions: RegionAABB2Ds
    range_y: List[Optional[float]]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, payload=np.asarray(self.model_dump_json(indent=None)))

    @staticmethod
    def load(path: str | Path) -> "LevelSemanticBoxes":
        data = np.load(path)
        return LevelSemanticBoxes.model_validate_json(str(data["payload"]))


class RelevantSemanticBoxes(BaseModel):
    """Instruction/path-relevant semantic boxes for one selected scene level."""

    level_idx: int
    level: LevelSemanticBoxes
    instruction: str
    ground_truth_trajectory: LevelTrajectory2D
    trajectory_keypoints: List[Point2D]
    start_direction_vector: DirectionVector

    @field_validator("trajectory_keypoints")
    @classmethod
    def _validate_trajectory_keypoint_count(
        cls, points: List[Point2D]
    ) -> List[Point2D]:
        if len(points) != TRAJECTORY_KEYPOINT_COUNT:
            raise ValueError(
                "trajectory_keypoints must contain exactly "
                f"{TRAJECTORY_KEYPOINT_COUNT} points, got {len(points)}"
            )
        return points

    @staticmethod
    def _rotate_point_by_right_angle(point: Point2D, turns: int) -> Point2D:
        x, z = point
        turns %= 4
        if turns == 0:
            return (float(x), float(z))
        if turns == 1:
            return (float(MAX_GRID_Z - z), float(x))
        if turns == 2:
            return (float(MAX_GRID_X - x), float(MAX_GRID_Z - z))
        return (float(z), float(MAX_GRID_X - x))

    @staticmethod
    def _rotate_direction_by_right_angle(
        vector: DirectionVector, turns: int
    ) -> DirectionVector:
        sin_value, cos_value = vector
        turns %= 4
        if turns == 0:
            return (float(sin_value), float(cos_value))
        if turns == 1:
            return (float(cos_value), float(-sin_value))
        if turns == 2:
            return (float(-sin_value), float(-cos_value))
        return (float(-cos_value), float(sin_value))

    @staticmethod
    def _rotate_trajectory_keypoints_by_right_angle(
        points: List[Point2D], turns: int
    ) -> List[Point2D]:
        last_nonzero_idx = next(
            (
                index
                for index in range(len(points) - 1, -1, -1)
                if points[index] != (0.0, 0.0)
            ),
            -1,
        )
        return [
            (
                RelevantSemanticBoxes._rotate_point_by_right_angle(point, turns)
                if index <= last_nonzero_idx
                else (0.0, 0.0)
            )
            for index, point in enumerate(points)
        ]

    @staticmethod
    def _rotate_aabb_by_right_angle(box: AABB2D, turns: int) -> AABB2D:
        corners = [
            (box.min[0], box.min[1]),
            (box.min[0], box.max[1]),
            (box.max[0], box.min[1]),
            (box.max[0], box.max[1]),
        ]
        rotated_corners = [
            RelevantSemanticBoxes._rotate_point_by_right_angle(corner, turns)
            for corner in corners
        ]
        xs = [corner[0] for corner in rotated_corners]
        zs = [corner[1] for corner in rotated_corners]
        return AABB2D(
            min=(min(xs), min(zs)),
            max=(max(xs), max(zs)),
            mentioned=box.mentioned,
        )

    def rotate_by_right_angle(self, turns: int) -> "RelevantSemanticBoxes":
        turns %= 4
        rotated_objects = [
            [
                OBB2D(
                    center=self._rotate_point_by_right_angle(box.center, turns),
                    half_extents=box.half_extents,
                    rotation=box.rotation + turns * np.pi / 2.0,
                    mentioned=box.mentioned,
                )
                for box in boxes
            ]
            for boxes in self.level.objects
        ]
        rotated_regions = [
            [self._rotate_aabb_by_right_angle(box, turns) for box in boxes]
            for boxes in self.level.regions
        ]
        return RelevantSemanticBoxes(
            level_idx=self.level_idx,
            level=LevelSemanticBoxes(
                objects=rotated_objects,
                regions=rotated_regions,
                range_y=list(self.level.range_y),
            ),
            instruction=self.instruction,
            ground_truth_trajectory=[
                self._rotate_point_by_right_angle(point, turns)
                for point in self.ground_truth_trajectory
            ],
            trajectory_keypoints=self._rotate_trajectory_keypoints_by_right_angle(
                self.trajectory_keypoints, turns
            ),
            start_direction_vector=self._rotate_direction_by_right_angle(
                self.start_direction_vector, turns
            ),
        )

    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    @staticmethod
    def from_json(payload: str) -> "RelevantSemanticBoxes":
        return RelevantSemanticBoxes.model_validate_json(payload)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def load_json(path: str | Path) -> "RelevantSemanticBoxes":
        return RelevantSemanticBoxes.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix == ".json":
            self.save_json(path)
            return

        np.savez_compressed(path, payload=np.asarray(self.to_json(indent=None)))

    @staticmethod
    def load(path: str | Path) -> "RelevantSemanticBoxes":
        path = Path(path)
        if path.suffix == ".json":
            return RelevantSemanticBoxes.load_json(path)

        data = np.load(path)
        return RelevantSemanticBoxes.from_json(str(data["payload"]))

    def to_cognitive_map(self):
        """Convert relevant semantic boxes into one CognitiveGridMap."""
        from prior.grid_map import CognitiveGridMap

        from ._rasterize import _rasterize_level_semantic_boxes

        cognitive_map = CognitiveGridMap()
        cognitive_map.range_y = list(self.level.range_y)
        cognitive_map.start_direction_vector = self.start_direction_vector
        cognitive_map.trajectory_keypoints = [
            (float(x), float(z)) for x, z in self.trajectory_keypoints
        ]

        _rasterize_level_semantic_boxes(self.level, cognitive_map.grid)
        return cognitive_map


@dataclass
class SceneSemanticBoxes:
    """All level-wise semantic boxes for one MP3D scene."""

    levels: List[LevelSemanticBoxes]
    _level_origins: List[Point2D] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._level_origins:
            self._level_origins = [(0.0, 0.0) for _ in self.levels]
        if len(self._level_origins) != len(self.levels):
            raise ValueError("_level_origins length must match levels length")

    @staticmethod
    def from_scene_id(scene_id: str) -> "SceneSemanticBoxes":
        """Load scene semantic boxes from cache or build/cache them from MP3D. Cached."""
        from copy import deepcopy

        from ._construct import _scene_semantic_boxes_from_scene_id

        return deepcopy(_scene_semantic_boxes_from_scene_id(scene_id))

    @staticmethod
    def from_scene(semantic_scene: Any) -> "SceneSemanticBoxes":
        from ._construct import _construct_scene_semantic_boxes_from_scene

        return _construct_scene_semantic_boxes_from_scene(semantic_scene)

    def relevant_to(
        self,
        instruction: str,
        ground_truth_trajectory: WorldTrajectory3D,
        start_direction_vector: DirectionVector,
        max_distance: float = MAX_DISTANCE_CELLS * CELL_SIZE,
        category_extractor: Optional[Callable[[str], Tuple[Set[int], Set[int]]]] = None,
    ) -> RelevantSemanticBoxes:
        from ._relevance import _extract_relevant_semantic_boxes

        return _extract_relevant_semantic_boxes(
            self,
            instruction,
            ground_truth_trajectory,
            start_direction_vector=start_direction_vector,
            max_distance=max_distance,
            category_extractor=category_extractor,
        )
