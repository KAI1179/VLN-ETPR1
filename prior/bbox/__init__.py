"""2D semantic bounding boxes from MP3D scenes."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple, cast

import numpy as np
from habitat_sim.scene import (
    Mp3dObjectCategory,
    Mp3dRegionCategory,
    SemanticLevel,
    SemanticScene,
)
from magnum import Vector3
from pydantic import BaseModel

from prior import MP3D_DIR
from prior import DATA_DIR
from prior.directions import DirectionVector, world_delta_to_direction_vector
from ..constants import (
    CELL_SIZE,
    COLS,
    DIRECTION_VECTOR_CNT,
    DIRECTION_VECTOR_SIM,
    HABITAT_MP3D_ROTATION_VECTOR,
    MAX_DISTANCE_CELLS,
    OBJECT_CATEGORIES,
    OBJECT_MAPPING,
    REGION_CATEGORIES,
    REGION_MAPPING,
    ROWS,
)

Point2D = Tuple[float, float]
Axis2D = Tuple[float, float]
IRRELEVANT_MULTIPLIER = 0.6
"""Confidence multiplier for trajectory-near boxes not mentioned in the instruction."""

SEMANTIC_BOX_DIR = DATA_DIR / "semantic_boxes"
"""On-disk cache root for per-scene semantic boxes."""


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

    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)

    @staticmethod
    def from_json(payload: str) -> "LevelSemanticBoxes":
        return LevelSemanticBoxes.model_validate_json(payload)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def load_json(path: str | Path) -> "LevelSemanticBoxes":
        return LevelSemanticBoxes.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix == ".json":
            self.save_json(path)
            return

        np.savez_compressed(path, payload=np.asarray(self.to_json(indent=None)))

    @staticmethod
    def load(path: str | Path) -> "LevelSemanticBoxes":
        path = Path(path)
        if path.suffix == ".json":
            return LevelSemanticBoxes.load_json(path)

        data = np.load(path)
        return LevelSemanticBoxes.from_json(str(data["payload"]))


@dataclass
class RelevantSemanticBoxes:
    """Instruction/path-relevant semantic boxes for one selected scene level."""

    level_idx: int
    level: LevelSemanticBoxes
    instruction: str
    reference_path: List[List[float]]
    start_direction_vector: DirectionVector

    def to_cognitive_map(self):
        """Convert relevant semantic boxes into one CognitiveGridMap."""
        from prior.grid_map import CognitiveGridMap

        cognitive_map = CognitiveGridMap()
        cognitive_map.offset_x = self.level.offset_x
        cognitive_map.offset_z = self.level.offset_z
        cognitive_map.range_y = list(self.level.range_y)
        cognitive_map.direction_vectors = _extract_direction_vectors(
            self.reference_path
        )
        cognitive_map.start_direction_vector = self.start_direction_vector
        cognitive_map.positions = [
            self.level.world_to_grid(float(x), float(z))
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
        return deepcopy(_scene_semantic_boxes_from_scene_id(scene_id))

    @staticmethod
    def from_scene(semantic_scene: Any) -> "SceneSemanticBoxes":
        return _construct_scene_semantic_boxes_from_scene(semantic_scene)

    def relevant_to(
        self,
        instruction: str,
        reference_path: List[List[float]],
        start_direction_vector: DirectionVector,
        max_distance: float = MAX_DISTANCE_CELLS * CELL_SIZE,
        category_extractor: Optional[Callable[[str], Tuple[Set[int], Set[int]]]] = None,
    ) -> RelevantSemanticBoxes:
        return _extract_relevant_semantic_boxes(
            self,
            instruction,
            reference_path,
            start_direction_vector=start_direction_vector,
            max_distance=max_distance,
            category_extractor=category_extractor,
        )


def _aabb_min(aabb) -> Vector3:
    """Return AABB minimum corner."""
    return Vector3(aabb.center - (aabb.sizes / 2.0))


def _aabb_max(aabb) -> Vector3:
    """Return AABB maximum corner."""
    return Vector3(aabb.center + (aabb.sizes / 2.0))


def _xz(point) -> Point2D:
    """Project a 3D point/vector to 2D world X/Z coordinates."""
    return (float(point[0]), float(point[2]))


def _normalized_xz(vector) -> Axis2D:
    x, z = _xz(vector)
    norm = math.hypot(x, z)
    if norm == 0.0:
        return (1.0, 0.0)
    return (x / norm, z / norm)


def _obb_rotation_2d(obb) -> float:
    """Return projected local X-axis angle in world X/Z coordinates."""
    local_to_world = obb.local_to_world
    if hasattr(local_to_world, "transform_vector"):
        x_axis = _normalized_xz(local_to_world.transform_vector(Vector3(1.0, 0.0, 0.0)))
    else:
        # Habitat exposes OBB matrices as ndarray in some versions. Treat
        # column 0 as the local X basis vector in world coordinates.
        x_axis = _normalized_xz(
            (local_to_world[0][0], local_to_world[1][0], local_to_world[2][0])
        )
    return math.atan2(x_axis[1], x_axis[0])


def _empty_level_boxes() -> LevelSemanticBoxes:
    return LevelSemanticBoxes(
        objects=[[] for _ in range(OBJECT_CATEGORIES)],
        regions=[[] for _ in range(REGION_CATEGORIES)],
        range_y=[None, None],
    )


def _point_to_aabb_distance(point: Point2D, box: AABB2D) -> float:
    """Return shortest 2D distance from a point to an axis-aligned box."""
    px, pz = point
    dx = max(box.min[0] - px, 0.0, px - box.max[0])
    dz = max(box.min[1] - pz, 0.0, pz - box.max[1])
    return math.hypot(dx, dz)


def _point_to_obb_distance(point: Point2D, box: OBB2D) -> float:
    """Return shortest 2D distance from a point to a rotated object box."""
    dx = point[0] - box.center[0]
    dz = point[1] - box.center[1]
    cos_theta = math.cos(box.rotation)
    sin_theta = math.sin(box.rotation)

    local_x = dx * cos_theta + dz * sin_theta
    local_z = -dx * sin_theta + dz * cos_theta
    clamped_x = min(max(local_x, -box.half_extents[0]), box.half_extents[0])
    clamped_z = min(max(local_z, -box.half_extents[1]), box.half_extents[1])

    closest_x = box.center[0] + clamped_x * cos_theta - clamped_z * sin_theta
    closest_z = box.center[1] + clamped_x * sin_theta + clamped_z * cos_theta
    return math.hypot(point[0] - closest_x, point[1] - closest_z)


def _is_position_in_level(
    position: List[float], range_y: List[Optional[float]]
) -> bool:
    y = position[1]
    if range_y[0] is not None and y < range_y[0]:
        return False
    if range_y[1] is not None and y >= range_y[1]:
        return False
    return True


def _near_any_position(
    box,
    points: List[Point2D],
    max_distance: float,
) -> bool:
    if isinstance(box, OBB2D):
        distance_func = _point_to_obb_distance
    else:
        distance_func = _point_to_aabb_distance
    return any(distance_func(point, box) <= max_distance for point in points)


def _direction_vector_between(
    prev_position: List[float],
    next_position: List[float],
) -> DirectionVector:
    dx = next_position[0] - prev_position[0]
    dz = next_position[2] - prev_position[2]
    return world_delta_to_direction_vector(dx, dz)


def _extract_direction_vectors(
    positions: List[List[float]],
) -> List[DirectionVector]:
    directions: List[DirectionVector] = []

    for i in range(len(positions) - 1):
        direction = _direction_vector_between(positions[i], positions[i + 1])
        if directions:
            prev = directions[-1]
            similarity = direction[0] * prev[0] + direction[1] * prev[1]
            if similarity >= DIRECTION_VECTOR_SIM:
                continue

        directions.append(direction)
        if len(directions) >= DIRECTION_VECTOR_CNT:
            break

    while len(directions) < DIRECTION_VECTOR_CNT:
        directions.append((0.0, 0.0))

    return directions


def _first_encountered_level(
    scene: SceneSemanticBoxes,
    reference_path: List[List[float]],
) -> tuple[int, LevelSemanticBoxes]:
    if not scene.levels:
        raise ValueError("SceneSemanticBoxes contains no levels")
    if not reference_path:
        raise ValueError("reference_path is empty")

    for position in reference_path:
        for level_idx, level in enumerate(scene.levels):
            if _is_position_in_level(position, level.range_y):
                return level_idx, level

    ranges = [level.range_y for level in scene.levels]
    raise ValueError(
        f"reference_path does not intersect any semantic level range: {ranges}"
    )


def _extract_relevant_semantic_boxes(
    scene: SceneSemanticBoxes,
    instruction: str,
    reference_path: List[List[float]],
    start_direction_vector: DirectionVector,
    max_distance: float = MAX_DISTANCE_CELLS * CELL_SIZE,
    category_extractor: Optional[Callable[[str], Tuple[Set[int], Set[int]]]] = None,
) -> RelevantSemanticBoxes:
    """Return selected-level boxes near the path, marking categories in text."""
    if category_extractor is None:
        from prior.grid_map._cognitive import extract_categories

        category_extractor = extract_categories

    level_idx, level = _first_encountered_level(scene, reference_path)
    mentioned_objects, mentioned_regions = category_extractor(instruction)

    relevant_level = _empty_level_boxes()
    relevant_level.range_y = list(level.range_y)
    relevant_level.offset_x = level.offset_x
    relevant_level.offset_z = level.offset_z
    level_points = [
        (float(position[0]), float(position[2]))
        for position in reference_path
        if _is_position_in_level(position, level.range_y)
    ]

    for category_idx, boxes in enumerate(level.objects):
        mentioned = category_idx in mentioned_objects
        relevant_level.objects[category_idx] = [
            replace(box, mentioned=mentioned)
            for box in boxes
            if _near_any_position(box, level_points, max_distance)
        ]

    for category_idx, boxes in enumerate(level.regions):
        mentioned = category_idx in mentioned_regions
        relevant_level.regions[category_idx] = [
            replace(box, mentioned=mentioned)
            for box in boxes
            if _near_any_position(box, level_points, max_distance)
        ]

    return RelevantSemanticBoxes(
        level_idx=level_idx,
        level=relevant_level,
        instruction=instruction,
        reference_path=reference_path,
        start_direction_vector=start_direction_vector,
    )


def _grid_center_index_range(
    min_coord: float,
    max_coord: float,
    offset: float,
    limit: int,
) -> tuple[int, int] | None:
    start = math.ceil((min_coord - offset - CELL_SIZE / 2.0) / CELL_SIZE)
    end = math.floor((max_coord - offset - CELL_SIZE / 2.0) / CELL_SIZE)
    if end < 0 or start >= limit:
        return None
    return max(0, start), min(limit - 1, end)


def _rasterize_aabb(
    box: AABB2D,
    level: LevelSemanticBoxes,
    layer_idx: int,
    grid: np.ndarray,
    confidence: float,
) -> None:
    row_range = _grid_center_index_range(box.min[0], box.max[0], level.offset_x, ROWS)
    col_range = _grid_center_index_range(box.min[1], box.max[1], level.offset_z, COLS)
    if row_range is None or col_range is None:
        return
    row_start, row_end = row_range
    col_start, col_end = col_range
    grid[layer_idx, row_start : row_end + 1, col_start : col_end + 1] = np.maximum(
        grid[layer_idx, row_start : row_end + 1, col_start : col_end + 1],
        confidence,
    )


def _obb_bounds(box: OBB2D) -> tuple[float, float, float, float]:
    cos_theta = math.cos(box.rotation)
    sin_theta = math.sin(box.rotation)
    corners = []
    for sign_x in (-1.0, 1.0):
        for sign_z in (-1.0, 1.0):
            local_x = sign_x * box.half_extents[0]
            local_z = sign_z * box.half_extents[1]
            corners.append(
                (
                    box.center[0] + local_x * cos_theta - local_z * sin_theta,
                    box.center[1] + local_x * sin_theta + local_z * cos_theta,
                )
            )
    xs = [corner[0] for corner in corners]
    zs = [corner[1] for corner in corners]
    return (
        min(xs),
        max(xs),
        min(zs),
        max(zs),
    )


def _rasterize_obb(
    box: OBB2D,
    level: LevelSemanticBoxes,
    layer_idx: int,
    grid: np.ndarray,
    confidence: float,
) -> None:
    min_x, max_x, min_z, max_z = _obb_bounds(box)
    row_range = _grid_center_index_range(min_x, max_x, level.offset_x, ROWS)
    col_range = _grid_center_index_range(min_z, max_z, level.offset_z, COLS)
    if row_range is None or col_range is None:
        return

    row_start, row_end = row_range
    col_start, col_end = col_range
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            point = level.grid_to_world(row, col)
            if _point_to_obb_distance(point, box) <= 1e-6:
                grid[layer_idx, row, col] = max(grid[layer_idx, row, col], confidence)


def _rasterize_level_semantic_boxes(
    level: LevelSemanticBoxes,
    grid: np.ndarray,
) -> None:
    for category_idx, boxes in enumerate(level.objects):
        for box in boxes:
            confidence = 1.0 if box.mentioned else IRRELEVANT_MULTIPLIER
            _rasterize_obb(box, level, category_idx, grid, confidence)

    for category_idx, boxes in enumerate(level.regions):
        layer_idx = OBJECT_CATEGORIES + category_idx
        for box in boxes:
            confidence = 1.0 if box.mentioned else IRRELEVANT_MULTIPLIER
            _rasterize_aabb(box, level, layer_idx, grid, confidence)


def _load_scene_semantic_boxes_from_cache(scene_id: str) -> SceneSemanticBoxes | None:
    cache_dir = SEMANTIC_BOX_DIR / scene_id
    levels: List[LevelSemanticBoxes] = []
    level_idx = 0
    while (cache_dir / f"{level_idx}.npz").exists():
        levels.append(LevelSemanticBoxes.load(cache_dir / f"{level_idx}.npz"))
        level_idx += 1
    if not levels:
        return None
    return SceneSemanticBoxes(levels)


@lru_cache(maxsize=100)
def _scene_semantic_boxes_from_scene_id(scene_id: str) -> SceneSemanticBoxes:
    """Load cached scene semantic boxes or build/cache them from MP3D."""
    cached = _load_scene_semantic_boxes_from_cache(scene_id)
    if cached is not None:
        return cached

    scene_path = str(MP3D_DIR / scene_id / f"{scene_id}.house")
    semantic_scene = SemanticScene()
    SemanticScene.load_mp3d_house(
        scene_path, semantic_scene, cast(Any, HABITAT_MP3D_ROTATION_VECTOR)
    )
    scene_boxes = _construct_scene_semantic_boxes_from_scene(semantic_scene)

    cache_dir = SEMANTIC_BOX_DIR / scene_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    for level_idx, level in enumerate(scene_boxes.levels):
        level.save(cache_dir / f"{level_idx}.npz")

    return scene_boxes


def _construct_scene_semantic_boxes_from_scene(
    semantic_scene: SemanticScene,
) -> SceneSemanticBoxes:
    """Construct level-wise 2D semantic boxes from a loaded semantic scene."""
    if not semantic_scene.levels:
        return SceneSemanticBoxes([])

    def _level_floor_y(level) -> float:
        if level.regions:
            return min(_aabb_min(region.aabb).y for region in level.regions)
        return float(_aabb_min(level.aabb).y)

    level_pairs = sorted(
        ((float(_level_floor_y(lvl)), lvl) for lvl in semantic_scene.levels),
        key=lambda pair: pair[0],
    )

    floor_ys = [fy for fy, _ in level_pairs]
    level_boxes: List[LevelSemanticBoxes] = []

    for i, (_, semantic_level) in enumerate(level_pairs):
        boxes = _construct_level_semantic_boxes_from_level(semantic_level)
        boxes.range_y = [
            None if i == 0 else floor_ys[i],
            floor_ys[i + 1] if i + 1 < len(floor_ys) else None,
        ]
        level_boxes.append(boxes)

    return SceneSemanticBoxes(level_boxes)


def _construct_level_semantic_boxes_from_level(
    semantic_level: SemanticLevel,
) -> LevelSemanticBoxes:
    """Construct 2D semantic boxes from one semantic level."""
    boxes = _empty_level_boxes()
    level_aabb_min = _aabb_min(semantic_level.aabb)
    boxes.offset_x = float(level_aabb_min.x)
    boxes.offset_z = float(level_aabb_min.z)

    for region in semantic_level.regions:
        assert isinstance(region.category, Mp3dRegionCategory), (
            "Region category is not Mp3dRegionCategory"
        )
        mapped_region = REGION_MAPPING[region.category.index()]
        aabb_min = _aabb_min(region.aabb)
        aabb_max = _aabb_max(region.aabb)
        boxes.regions[mapped_region].append(
            AABB2D(min=_xz(aabb_min), max=_xz(aabb_max))
        )

        # semantic_level.objects is always empty for MP3D .house scenes; region
        # objects are populated and match the grid-map construction path.
        for obj in region.objects:
            assert isinstance(obj.category, Mp3dObjectCategory), (
                "Object category is not Mp3dObjectCategory"
            )
            mapped_object = OBJECT_MAPPING[obj.category.index()]
            obb = obj.obb
            boxes.objects[mapped_object].append(
                OBB2D(
                    center=_xz(obb.center),
                    half_extents=(
                        float(obb.half_extents[0]),
                        float(obb.half_extents[2]),
                    ),
                    rotation=_obb_rotation_2d(obb),
                )
            )

    return boxes


__all__ = [
    "AABB2D",
    "IRRELEVANT_MULTIPLIER",
    "LevelSemanticBoxes",
    "OBB2D",
    "RelevantSemanticBoxes",
    "SceneSemanticBoxes",
]
