"""Data types for semantic bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel

from prior.directions import DirectionVector
from ..constants import CELL_SIZE, MAX_DISTANCE_CELLS

Point2D = Tuple[float, float]
Axis2D = Tuple[float, float]
IRRELEVANT_MULTIPLIER = 0.6
"""Confidence multiplier for trajectory-near boxes not mentioned in the instruction."""


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
    offset_x: float = 0.0
    offset_z: float = 0.0

    def world_to_grid(self, x: float, z: float) -> tuple[float, float]:
        return ((x - self.offset_x) / CELL_SIZE, (z - self.offset_z) / CELL_SIZE)

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = row * CELL_SIZE + CELL_SIZE / 2.0 + self.offset_x
        z = col * CELL_SIZE + CELL_SIZE / 2.0 + self.offset_z
        return (x, z)

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
    reference_path: List[List[float]]
    start_direction_vector: DirectionVector

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

        from ._geometry import _is_position_in_level
        from ._rasterize import _rasterize_level_semantic_boxes

        cognitive_map = CognitiveGridMap()
        cognitive_map.offset_x = self.level.offset_x
        cognitive_map.offset_z = self.level.offset_z
        cognitive_map.range_y = list(self.level.range_y)
        cognitive_map.start_direction_vector = self.start_direction_vector
        cognitive_map.reference_path = [
            [float(x), float(y), float(z)]
            for x, y, z in self.reference_path
            if _is_position_in_level([x, y, z], self.level.range_y)
        ]

        _rasterize_level_semantic_boxes(self.level, cognitive_map.grid)
        return cognitive_map


@dataclass
class SceneSemanticBoxes:
    """All level-wise semantic boxes for one MP3D scene."""

    levels: List[LevelSemanticBoxes]

    @staticmethod
    def from_scene_id(scene_id: str) -> "SceneSemanticBoxes":
        """Load scene semantic boxes from cache or build/cache them from MP3D."""
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
        reference_path: List[List[float]],
        start_direction_vector: DirectionVector,
        max_distance: float = MAX_DISTANCE_CELLS * CELL_SIZE,
        category_extractor: Optional[Callable[[str], Tuple[Set[int], Set[int]]]] = None,
    ) -> RelevantSemanticBoxes:
        from ._relevance import _extract_relevant_semantic_boxes

        return _extract_relevant_semantic_boxes(
            self,
            instruction,
            reference_path,
            start_direction_vector=start_direction_vector,
            max_distance=max_distance,
            category_extractor=category_extractor,
        )
