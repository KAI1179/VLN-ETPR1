"""Structured T5-Boxes JSON schema and semantic-box conversion helpers."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple

import prior.bbox as bbox
from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES

Point2D = Tuple[float, float]
CategoryExtractor = Callable[[str], Tuple[Set[int], Set[int]]]

OBJECT_CATEGORY_TO_ID = {
    category: idx for idx, category in enumerate(MAPPED_OBJECT_NAMES)
}
REGION_CATEGORY_TO_ID = {
    category: idx for idx, category in enumerate(MAPPED_REGION_NAMES)
}


class T5BoxesValidationError(ValueError):
    """Raised when generated T5-Boxes JSON fails deterministic validation."""


@dataclass(frozen=True)
class ObjectBoxSpec:
    category: str
    center: Point2D
    half_extents: Point2D
    rotation: float = 0.0


@dataclass(frozen=True)
class RegionBoxSpec:
    category: str
    min: Point2D
    max: Point2D


@dataclass(frozen=True)
class T5BoxesSpec:
    objects: Tuple[ObjectBoxSpec, ...]
    regions: Tuple[RegionBoxSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "regions", tuple(self.regions))


def build_t5_boxes_input(
    dataset_tag: str,
    instruction: str,
    start_position: Sequence[float],
    start_direction: Sequence[float],
) -> str:
    """Build the scene-anonymous T5-Boxes prompt payload.

    ``start_position`` may be level-local ``(x, z)`` or scene-style
    ``(x, y, z)``; the prompt always stores level-local/projected ``[x, z]``.
    """
    payload = {
        "dataset": str(dataset_tag),
        "start_position": [
            _round_coord(value) for value in _xz_point(start_position, "start_position")
        ],
        "start_direction": [_round_rotation(value) for value in _point2(start_direction)],
        "instruction": str(instruction),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def relevant_semantic_boxes_to_spec(
    relevant: bbox.RelevantSemanticBoxes,
) -> T5BoxesSpec:
    """Convert indexed relevant semantic boxes to a category-name spec."""
    objects: List[ObjectBoxSpec] = []
    for category_id, boxes in enumerate(relevant.level.objects):
        category = MAPPED_OBJECT_NAMES[category_id]
        for box in boxes:
            objects.append(
                ObjectBoxSpec(
                    category=category,
                    center=tuple(_round_coord(value) for value in box.center),
                    half_extents=tuple(
                        _round_coord(value) for value in box.half_extents
                    ),
                    rotation=_round_rotation(box.rotation),
                )
            )

    regions: List[RegionBoxSpec] = []
    for category_id, boxes in enumerate(relevant.level.regions):
        category = MAPPED_REGION_NAMES[category_id]
        for box in boxes:
            regions.append(
                RegionBoxSpec(
                    category=category,
                    min=tuple(_round_coord(value) for value in box.min),
                    max=tuple(_round_coord(value) for value in box.max),
                )
            )

    return T5BoxesSpec(
        objects=sorted(objects, key=_object_sort_key),
        regions=sorted(regions, key=_region_sort_key),
    )


def spec_to_relevant_semantic_boxes(
    spec: T5BoxesSpec,
    instruction: str,
    level_idx: int,
    reference_path: Sequence[Sequence[float]],
    start_direction_vector: Sequence[float],
    range_y: Optional[List[Optional[float]]] = None,
    category_extractor: Optional[CategoryExtractor] = None,
) -> bbox.RelevantSemanticBoxes:
    """Convert a category-name spec back to indexed relevant semantic boxes."""
    if category_extractor is None:
        from prior.grid_map._cognitive import extract_categories

        category_extractor = extract_categories

    mentioned_objects, mentioned_regions = category_extractor(instruction)
    level = bbox.LevelSemanticBoxes(
        objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
        range_y=list(range_y) if range_y is not None else [None, None],
    )

    for raw_item in spec.objects:
        item = _normalize_object(raw_item)
        category_id = _object_category_id(item.category)
        level.objects[category_id].append(
            bbox.OBB2D(
                center=_point2(item.center),
                half_extents=_point2(item.half_extents),
                rotation=float(item.rotation),
                mentioned=category_id in mentioned_objects,
            )
        )

    for raw_item in spec.regions:
        item = _normalize_region(raw_item)
        category_id = _region_category_id(item.category)
        level.regions[category_id].append(
            bbox.AABB2D(
                min=_point2(item.min),
                max=_point2(item.max),
                mentioned=category_id in mentioned_regions,
            )
        )

    return bbox.RelevantSemanticBoxes(
        level_idx=int(level_idx),
        level=level,
        instruction=str(instruction),
        reference_path=[_path_point(point) for point in reference_path],
        start_direction_vector=_point2(start_direction_vector),
    )


def parse_t5_boxes_json(text: str) -> T5BoxesSpec:
    """Parse and validate a strict T5-Boxes JSON prediction."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise T5BoxesValidationError(f"malformed JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise T5BoxesValidationError("top-level JSON must be an object")
    if set(payload.keys()) != {"objects", "regions"}:
        raise T5BoxesValidationError(
            "top-level JSON must contain exactly objects and regions"
        )
    if not isinstance(payload["objects"], list):
        raise T5BoxesValidationError("objects must be an array")
    if not isinstance(payload["regions"], list):
        raise T5BoxesValidationError("regions must be an array")

    return T5BoxesSpec(
        objects=[_parse_object(item, idx) for idx, item in enumerate(payload["objects"])],
        regions=[_parse_region(item, idx) for idx, item in enumerate(payload["regions"])],
    )


def spec_to_json(spec: T5BoxesSpec) -> str:
    """Serialize a spec as normalized JSON with deterministic ordering."""
    normalized = T5BoxesSpec(
        objects=sorted(
            [_normalize_object(item) for item in spec.objects],
            key=_object_sort_key,
        ),
        regions=sorted(
            [_normalize_region(item) for item in spec.regions],
            key=_region_sort_key,
        ),
    )
    payload = {
        "objects": [
            {
                "category": item.category,
                "center": list(item.center),
                "half_extents": list(item.half_extents),
                "rotation": item.rotation,
            }
            for item in normalized.objects
        ],
        "regions": [
            {
                "category": item.category,
                "min": list(item.min),
                "max": list(item.max),
            }
            for item in normalized.regions
        ],
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def write_prediction_artifact(
    output_dir: str | Path,
    example_id: str,
    valid_spec: Optional[T5BoxesSpec] = None,
    invalid_text: Optional[str] = None,
    error: Optional[BaseException] = None,
) -> None:
    """Write either normalized valid JSON or invalid raw text plus error."""
    valid_mode = valid_spec is not None
    invalid_mode = invalid_text is not None or error is not None
    if valid_mode == invalid_mode:
        raise ValueError("provide exactly one of valid_spec or invalid_text/error")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / f"{_safe_artifact_stem(example_id)}.json"

    if valid_spec is not None:
        artifact_path.write_text(spec_to_json(valid_spec), encoding="utf-8")
        return

    payload = {
        "raw_text": "" if invalid_text is None else str(invalid_text),
        "error": "" if error is None else str(error),
    }
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _parse_object(item: Any, idx: int) -> ObjectBoxSpec:
    if not isinstance(item, dict):
        raise T5BoxesValidationError(f"objects[{idx}] must be an object")
    if set(item.keys()) != {"category", "center", "half_extents", "rotation"}:
        raise T5BoxesValidationError(
            f"objects[{idx}] must contain category, center, half_extents, rotation"
        )

    category = item["category"]
    _object_category_id(category)
    center = _finite_point(item["center"], f"objects[{idx}].center")
    half_extents = _finite_point(
        item["half_extents"], f"objects[{idx}].half_extents"
    )
    if half_extents[0] <= 0.0 or half_extents[1] <= 0.0:
        raise T5BoxesValidationError(
            f"objects[{idx}].half_extents must be positive"
        )
    rotation = _finite_number(item["rotation"], f"objects[{idx}].rotation")
    return ObjectBoxSpec(
        category=category,
        center=center,
        half_extents=half_extents,
        rotation=rotation,
    )


def _parse_region(item: Any, idx: int) -> RegionBoxSpec:
    if not isinstance(item, dict):
        raise T5BoxesValidationError(f"regions[{idx}] must be an object")
    if set(item.keys()) != {"category", "min", "max"}:
        raise T5BoxesValidationError(
            f"regions[{idx}] must contain category, min, max"
        )

    category = item["category"]
    _region_category_id(category)
    min_point = _finite_point(item["min"], f"regions[{idx}].min")
    max_point = _finite_point(item["max"], f"regions[{idx}].max")
    if max_point[0] <= min_point[0] or max_point[1] <= min_point[1]:
        raise T5BoxesValidationError(
            f"regions[{idx}].max must be greater than min on both axes"
        )
    return RegionBoxSpec(category=category, min=min_point, max=max_point)


def _normalize_object(item: ObjectBoxSpec) -> ObjectBoxSpec:
    _object_category_id(item.category)
    center = _finite_point(item.center, "object.center")
    half_extents = _finite_point(item.half_extents, "object.half_extents")
    if half_extents[0] <= 0.0 or half_extents[1] <= 0.0:
        raise T5BoxesValidationError("object.half_extents must be positive")
    return ObjectBoxSpec(
        category=item.category,
        center=tuple(_round_coord(value) for value in center),
        half_extents=tuple(_round_coord(value) for value in half_extents),
        rotation=_round_rotation(_finite_number(item.rotation, "object.rotation")),
    )


def _normalize_region(item: RegionBoxSpec) -> RegionBoxSpec:
    _region_category_id(item.category)
    min_point = _finite_point(item.min, "region.min")
    max_point = _finite_point(item.max, "region.max")
    if max_point[0] <= min_point[0] or max_point[1] <= min_point[1]:
        raise T5BoxesValidationError(
            "region.max must be greater than min on both axes"
        )
    return RegionBoxSpec(
        category=item.category,
        min=tuple(_round_coord(value) for value in min_point),
        max=tuple(_round_coord(value) for value in max_point),
    )


def _object_category_id(category: Any) -> int:
    if category not in OBJECT_CATEGORY_TO_ID:
        raise T5BoxesValidationError(f"unknown object category: {category!r}")
    return OBJECT_CATEGORY_TO_ID[category]


def _region_category_id(category: Any) -> int:
    if category not in REGION_CATEGORY_TO_ID:
        raise T5BoxesValidationError(f"unknown region category: {category!r}")
    return REGION_CATEGORY_TO_ID[category]


def _finite_point(value: Any, field_name: str) -> Point2D:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise T5BoxesValidationError(f"{field_name} must contain two numbers")
    return (
        _finite_number(value[0], f"{field_name}[0]"),
        _finite_number(value[1], f"{field_name}[1]"),
    )


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T5BoxesValidationError(f"{field_name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise T5BoxesValidationError(f"{field_name} must be a finite number")
    return value


def _point2(value: Sequence[float]) -> Point2D:
    return _finite_point(value, "point")


def _xz_point(value: Sequence[float], field_name: str) -> Point2D:
    if not isinstance(value, (list, tuple)):
        raise T5BoxesValidationError(f"{field_name} must contain two or three numbers")
    if len(value) == 2:
        return (
            _finite_number(value[0], f"{field_name}[0]"),
            _finite_number(value[1], f"{field_name}[1]"),
        )
    if len(value) == 3:
        return (
            _finite_number(value[0], f"{field_name}[0]"),
            _finite_number(value[2], f"{field_name}[2]"),
        )
    raise T5BoxesValidationError(f"{field_name} must contain two or three numbers")


def _path_point(value: Sequence[float]) -> Point2D:
    if not isinstance(value, (list, tuple)):
        raise T5BoxesValidationError(
            "reference_path points must contain two or three numbers"
        )
    if len(value) == 2:
        return _point2(value)
    if len(value) == 3:
        return (
            _finite_number(value[0], "path[0]"),
            _finite_number(value[2], "path[2]"),
        )
    raise T5BoxesValidationError(
        "reference_path points must contain two or three numbers"
    )


def _round_coord(value: float) -> float:
    return round(float(value), 1)


def _round_rotation(value: float) -> float:
    return round(float(value), 2)


def _object_sort_key(item: ObjectBoxSpec) -> Tuple[str, float, float, float]:
    return (
        item.category,
        item.center[0],
        item.center[1],
        -_object_area(item),
    )


def _region_sort_key(item: RegionBoxSpec) -> Tuple[str, float, float, float]:
    return (
        item.category,
        item.min[0],
        item.min[1],
        -_region_area(item),
    )


def _object_area(item: ObjectBoxSpec) -> float:
    return 4.0 * item.half_extents[0] * item.half_extents[1]


def _region_area(item: RegionBoxSpec) -> float:
    return (item.max[0] - item.min[0]) * (item.max[1] - item.min[1])


def _safe_artifact_stem(example_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(example_id))
    while ".." in stem:
        stem = stem.replace("..", "__")
    stem = stem.strip("._-")
    return stem or "prediction"
