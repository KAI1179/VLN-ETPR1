"""2D semantic bounding boxes from MP3D scenes."""

from __future__ import annotations

from ..constants import OBJECT_CATEGORIES, REGION_CATEGORIES
from ._construct import SEMANTIC_BOX_DIR, _scene_semantic_boxes_from_scene_id
from ._geometry import _point_to_obb_distance
from ._types import (
    AABB2D,
    Axis2D,
    IRRELEVANT_MULTIPLIER,
    LevelSemanticBoxes,
    OBB2D,
    ObjectOBB2Ds,
    Point2D,
    RegionAABB2Ds,
    RelevantSemanticBoxes,
    SceneSemanticBoxes,
)

__all__ = [
    "AABB2D",
    "Axis2D",
    "IRRELEVANT_MULTIPLIER",
    "LevelSemanticBoxes",
    "OBB2D",
    "OBJECT_CATEGORIES",
    "ObjectOBB2Ds",
    "Point2D",
    "REGION_CATEGORIES",
    "RegionAABB2Ds",
    "RelevantSemanticBoxes",
    "SEMANTIC_BOX_DIR",
    "SceneSemanticBoxes",
    "_point_to_obb_distance",
    "_scene_semantic_boxes_from_scene_id",
]
