"""Build and cache semantic boxes from Habitat semantic scenes."""

from __future__ import annotations

from functools import lru_cache
import json
from typing import Any, List, cast

from habitat_sim.scene import (
    Mp3dObjectCategory,
    Mp3dRegionCategory,
    SemanticLevel,
    SemanticScene,
)

from prior import DATA_DIR, MP3D_DIR

from ..constants import (
    HABITAT_MP3D_ROTATION_VECTOR,
    OBJECT_MAPPING,
    REGION_MAPPING,
)
from ._geometry import _aabb_max, _aabb_min, _empty_level_boxes, _obb_rotation_2d, _xz
from ._types import AABB2D, LevelSemanticBoxes, OBB2D, Point2D, SceneSemanticBoxes

SEMANTIC_BOX_DIR = DATA_DIR / "semantic_boxes"
"""On-disk cache root for per-scene semantic boxes."""


def _load_scene_semantic_boxes_from_cache(scene_id: str) -> SceneSemanticBoxes | None:
    cache_dir = SEMANTIC_BOX_DIR / scene_id
    origins_path = cache_dir / "origins.json"
    if not origins_path.exists():
        return None

    levels: List[LevelSemanticBoxes] = []
    level_idx = 0
    while (cache_dir / f"{level_idx}.npz").exists():
        levels.append(LevelSemanticBoxes.load(cache_dir / f"{level_idx}.npz"))
        level_idx += 1
    if not levels:
        return None

    level_origins = [
        (float(origin[0]), float(origin[1]))
        for origin in json.loads(origins_path.read_text(encoding="utf-8"))
    ]
    return SceneSemanticBoxes(levels, _level_origins=level_origins)


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
    (cache_dir / "origins.json").write_text(
        json.dumps(scene_boxes._level_origins),
        encoding="utf-8",
    )

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
    level_origins: List[Point2D] = []

    for i, (_, semantic_level) in enumerate(level_pairs):
        boxes, origin = _construct_level_semantic_boxes_from_level(semantic_level)
        boxes.range_y = [
            None if i == 0 else floor_ys[i],
            floor_ys[i + 1] if i + 1 < len(floor_ys) else None,
        ]
        level_boxes.append(boxes)
        level_origins.append(origin)

    return SceneSemanticBoxes(level_boxes, _level_origins=level_origins)


def _local_xz(point, origin: Point2D) -> Point2D:
    x, z = _xz(point)
    return (x - origin[0], z - origin[1])


def _construct_level_semantic_boxes_from_level(
    semantic_level: SemanticLevel,
) -> tuple[LevelSemanticBoxes, Point2D]:
    """Construct 2D semantic boxes from one semantic level."""
    boxes = _empty_level_boxes()
    level_aabb_min = _aabb_min(semantic_level.aabb)
    origin = _xz(level_aabb_min)

    for region in semantic_level.regions:
        assert isinstance(region.category, Mp3dRegionCategory), (
            "Region category is not Mp3dRegionCategory"
        )
        mapped_region = REGION_MAPPING[region.category.index()]
        aabb_min = _aabb_min(region.aabb)
        aabb_max = _aabb_max(region.aabb)
        boxes.regions[mapped_region].append(
            AABB2D(min=_local_xz(aabb_min, origin), max=_local_xz(aabb_max, origin))
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
                    center=_local_xz(obb.center, origin),
                    half_extents=(
                        float(obb.half_extents[0]),
                        float(obb.half_extents[2]),
                    ),
                    rotation=_obb_rotation_2d(obb),
                )
            )

    return boxes, origin
