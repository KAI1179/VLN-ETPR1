"""Structured LLM-Boxes schema and semantic-box conversion helpers."""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
TOP_LEVEL_KEYS = (
    "keypoints",
    "predicted_regions",
    "predicted_objects",
    "regions",
    "objects",
)


class LLMBoxesValidationError(ValueError):
    """Raised when generated LLM-Boxes text fails deterministic validation."""


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
class LLMBoxesSpec:
    objects: Sequence[ObjectBoxSpec]
    regions: Sequence[RegionBoxSpec]
    trajectory_keypoints: Sequence[Point2D] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "regions", tuple(self.regions))
        keypoints = tuple(
            _round_point(_point2(point)) for point in self.trajectory_keypoints
        )
        if keypoints and len(keypoints) != 5:
            raise LLMBoxesValidationError(
                "trajectory_keypoints must contain exactly five points"
            )
        object.__setattr__(
            self,
            "trajectory_keypoints",
            keypoints,
        )


@dataclass(frozen=True)
class LLMBoxesPartialParse:
    spec: LLMBoxesSpec
    dropped_text: str
    dropped_entity_count: int


def build_llm_boxes_input(
    dataset_tag: str,
    instruction: str,
    start_position: Sequence[float],
    start_direction: Sequence[float],
) -> str:
    """Build the scene-anonymous LLM-Boxes input text.

    ``start_position`` may be level-local ``(x, z)`` or scene-style
    ``(x, y, z)``; the text always stores level-local/projected ``(x, z)``.
    """
    start_x, start_z = _xz_point(start_position, "start_position")
    direction_x, direction_z = _point2(start_direction)
    return (
        f"dataset {dataset_tag} | "
        f"start x = {_round_coord(start_x)} | "
        f"start z = {_round_coord(start_z)} | "
        f"direction x = {_round_rotation(direction_x)} | "
        f"direction z = {_round_rotation(direction_z)} | "
        f"instruction {instruction}"
    )


def relevant_semantic_boxes_to_spec(
    relevant: bbox.RelevantSemanticBoxes,
) -> LLMBoxesSpec:
    """Convert indexed relevant semantic boxes to a category-name spec."""
    objects: List[ObjectBoxSpec] = []
    for category_id, boxes in enumerate(relevant.level.objects):
        category = MAPPED_OBJECT_NAMES[category_id]
        for box in boxes:
            objects.append(
                ObjectBoxSpec(
                    category=category,
                    center=_round_point(box.center),
                    half_extents=_round_positive_point(point=box.half_extents),
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
                    min=_round_point(box.min),
                    max=_round_point(box.max),
                )
            )

    return LLMBoxesSpec(
        objects=tuple(sorted(objects, key=_object_sort_key)),
        regions=tuple(sorted(regions, key=_region_sort_key)),
        trajectory_keypoints=tuple(
            _round_point(_trajectory_keypoint(point))
            for point in relevant.trajectory_keypoints
        ),
    )


def relevant_semantic_boxes_to_mentioned_spec(
    relevant: bbox.RelevantSemanticBoxes,
) -> LLMBoxesSpec:
    """Convert relevant semantic boxes to only instruction-mentioned entities."""
    objects: List[ObjectBoxSpec] = []
    for category_id, boxes in enumerate(relevant.level.objects):
        category = MAPPED_OBJECT_NAMES[category_id]
        for box in boxes:
            if not bool(getattr(box, "mentioned", False)):
                continue
            objects.append(
                ObjectBoxSpec(
                    category=category,
                    center=_round_point(box.center),
                    half_extents=_round_positive_point(box.half_extents),
                    rotation=_round_rotation(box.rotation),
                )
            )

    regions: List[RegionBoxSpec] = []
    for category_id, boxes in enumerate(relevant.level.regions):
        category = MAPPED_REGION_NAMES[category_id]
        for box in boxes:
            if not bool(getattr(box, "mentioned", False)):
                continue
            regions.append(
                RegionBoxSpec(
                    category=category,
                    min=_round_point(box.min),
                    max=_round_point(box.max),
                )
            )

    return LLMBoxesSpec(
        objects=tuple(sorted(objects, key=_object_sort_key)),
        regions=tuple(sorted(regions, key=_region_sort_key)),
        trajectory_keypoints=tuple(
            _round_point(_trajectory_keypoint(point))
            for point in relevant.trajectory_keypoints
        ),
    )


def spec_to_relevant_semantic_boxes(
    spec: LLMBoxesSpec,
    instruction: str,
    level_idx: int,
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
        ground_truth_trajectory=[],
        trajectory_keypoints=[
            _trajectory_keypoint(point) for point in spec.trajectory_keypoints
        ],
        start_direction_vector=_point2(start_direction_vector),
    )


def spec_to_llm_boxes_text(spec: LLMBoxesSpec) -> str:
    """Serialize a spec as compact JSON."""
    normalized = _normalize_spec(spec)

    regions: Dict[str, Dict[str, Any]] = {}
    for item in normalized.regions:
        regions.setdefault(item.category, {"boxes": []})["boxes"].append(
            {"min": list(item.min), "max": list(item.max)}
        )

    objects: Dict[str, Dict[str, Any]] = {}
    for item in normalized.objects:
        objects.setdefault(item.category, {"boxes": []})["boxes"].append(
            {
                "center": list(item.center),
                "half": list(item.half_extents),
                "rotation": item.rotation,
            }
        )

    raw = {
        "keypoints": [list(point) for point in normalized.trajectory_keypoints],
        "predicted_regions": list(regions.keys()),
        "predicted_objects": list(objects.keys()),
        "regions": regions,
        "objects": objects,
    }
    return json.dumps(raw, ensure_ascii=True, separators=(",", ":"))


def parse_llm_boxes_text(
    text: str, allow_trailing_incomplete: bool = False
) -> LLMBoxesSpec:
    """Parse and validate compact LLM-Boxes JSON."""
    result = parse_llm_boxes_text_partial(text)
    if result.dropped_text and not allow_trailing_incomplete:
        raise LLMBoxesValidationError("trailing text after JSON object")
    if not result.spec.trajectory_keypoints:
        raise LLMBoxesValidationError("trajectory keypoints are required")
    return result.spec


def parse_llm_boxes_text_partial(text: str) -> LLMBoxesPartialParse:
    """Parse the first complete JSON object and report trailing text."""
    stripped = text.strip()
    raw, end = _decode_first_json_value(stripped)
    dropped_text = stripped[end:].strip()
    return LLMBoxesPartialParse(
        spec=_parse_raw_llm_boxes_json(raw),
        dropped_text=dropped_text,
        dropped_entity_count=int(bool(dropped_text)),
    )


def _normalize_spec(spec: LLMBoxesSpec) -> LLMBoxesSpec:
    return LLMBoxesSpec(
        objects=tuple(
            sorted(
                [_normalize_object(item) for item in spec.objects],
                key=_object_sort_key,
            )
        ),
        regions=tuple(
            sorted(
                [_normalize_region(item) for item in spec.regions],
                key=_region_sort_key,
            )
        ),
        trajectory_keypoints=tuple(
            _round_point(_point2(point)) for point in spec.trajectory_keypoints
        ),
    )


def write_prediction_artifact(
    output_dir: str | Path,
    example_id: str,
    valid_spec: Optional[LLMBoxesSpec] = None,
    invalid_text: Optional[str] = None,
    error: Optional[BaseException] = None,
) -> None:
    """Write either normalized valid LLM text or invalid raw text plus error."""
    valid_mode = valid_spec is not None
    invalid_mode = invalid_text is not None or error is not None
    if valid_mode == invalid_mode:
        raise ValueError("provide exactly one of valid_spec or invalid_text/error")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / f"{_safe_artifact_stem(example_id)}.txt"

    if valid_spec is not None:
        artifact_path.write_text(
            f"{spec_to_llm_boxes_text(valid_spec)}\n",
            encoding="utf-8",
        )
        return

    raw_text = "" if invalid_text is None else str(invalid_text)
    error_text = "" if error is None else str(error)
    artifact_path.write_text(f"{raw_text}\n\n# error: {error_text}\n", encoding="utf-8")


def _decode_first_json_value(text: str) -> Tuple[Any, int]:
    if not text:
        raise LLMBoxesValidationError("LLM-Boxes output must be JSON")
    try:
        return json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as exc:
        raise LLMBoxesValidationError(f"invalid JSON: {exc.msg}") from exc


def _parse_raw_llm_boxes_json(raw: Any) -> LLMBoxesSpec:
    if not isinstance(raw, dict):
        raise LLMBoxesValidationError("LLM-Boxes output must be a JSON object")
    if tuple(raw.keys()) != TOP_LEVEL_KEYS:
        raise LLMBoxesValidationError(
            "LLM-Boxes JSON keys must be exactly "
            "keypoints, predicted_regions, predicted_objects, regions, objects"
        )

    trajectory_keypoints = _parse_keypoints(raw["keypoints"])
    predicted_regions = _parse_category_list(
        raw["predicted_regions"],
        "predicted_regions",
        REGION_CATEGORY_TO_ID,
    )
    predicted_objects = _parse_category_list(
        raw["predicted_objects"],
        "predicted_objects",
        OBJECT_CATEGORY_TO_ID,
    )
    regions = _parse_regions_by_category(raw["regions"], predicted_regions)
    objects = _parse_objects_by_category(raw["objects"], predicted_objects)
    return LLMBoxesSpec(
        objects=tuple(objects),
        regions=tuple(regions),
        trajectory_keypoints=tuple(trajectory_keypoints),
    )


def _parse_keypoints(raw_keypoints: Any) -> Tuple[Point2D, ...]:
    if not isinstance(raw_keypoints, list):
        raise LLMBoxesValidationError("keypoints must be a list")
    if len(raw_keypoints) != 5:
        raise LLMBoxesValidationError("trajectory keypoints are required")
    return tuple(
        _round_point(_finite_point(point, f"keypoints[{idx}]"))
        for idx, point in enumerate(raw_keypoints)
    )


def _parse_category_list(
    raw_categories: Any,
    name: str,
    category_to_id: Dict[str, int],
) -> List[str]:
    if not isinstance(raw_categories, list):
        raise LLMBoxesValidationError(f"{name} must be a list")
    categories: List[str] = []
    for idx, category in enumerate(raw_categories):
        if not isinstance(category, str):
            raise LLMBoxesValidationError(f"{name}[{idx}] must be a string")
        if category not in category_to_id:
            raise LLMBoxesValidationError(f"{name}[{idx}] is unknown: {category}")
        if category in categories:
            raise LLMBoxesValidationError(f"{name}[{idx}] is duplicated: {category}")
        categories.append(category)
    return categories


def _parse_regions_by_category(
    raw_regions: Any,
    predicted_regions: Sequence[str],
) -> Tuple[RegionBoxSpec, ...]:
    if not isinstance(raw_regions, dict):
        raise LLMBoxesValidationError("regions must be an object")
    if list(raw_regions.keys()) != list(predicted_regions):
        raise LLMBoxesValidationError("predicted_regions must match regions keys")

    regions: List[RegionBoxSpec] = []
    for category, raw_entity in raw_regions.items():
        _region_category_id(category)
        if not isinstance(raw_entity, dict):
            raise LLMBoxesValidationError(f"regions.{category} must be an object")
        if set(raw_entity) != {"boxes"}:
            raise LLMBoxesValidationError(f"regions.{category} must contain only boxes")
        raw_boxes = raw_entity["boxes"]
        if not isinstance(raw_boxes, list):
            raise LLMBoxesValidationError(f"regions.{category}.boxes must be a list")
        if not raw_boxes:
            raise LLMBoxesValidationError(f"regions.{category}.boxes must not be empty")
        for idx, raw_box in enumerate(raw_boxes):
            regions.append(_parse_region_box(category, raw_box, idx))
    return tuple(sorted(regions, key=_region_sort_key))


def _parse_objects_by_category(
    raw_objects: Any,
    predicted_objects: Sequence[str],
) -> Tuple[ObjectBoxSpec, ...]:
    if not isinstance(raw_objects, dict):
        raise LLMBoxesValidationError("objects must be an object")
    if list(raw_objects.keys()) != list(predicted_objects):
        raise LLMBoxesValidationError("predicted_objects must match objects keys")

    objects: List[ObjectBoxSpec] = []
    for category, raw_entity in raw_objects.items():
        _object_category_id(category)
        if not isinstance(raw_entity, dict):
            raise LLMBoxesValidationError(f"objects.{category} must be an object")
        if set(raw_entity) != {"boxes"}:
            raise LLMBoxesValidationError(f"objects.{category} must contain only boxes")
        raw_boxes = raw_entity["boxes"]
        if not isinstance(raw_boxes, list):
            raise LLMBoxesValidationError(f"objects.{category}.boxes must be a list")
        if not raw_boxes:
            raise LLMBoxesValidationError(f"objects.{category}.boxes must not be empty")
        for idx, raw_box in enumerate(raw_boxes):
            objects.append(_parse_object_box(category, raw_box, idx))
    return tuple(sorted(objects, key=_object_sort_key))


def _parse_region_box(category: str, raw_box: Any, idx: int) -> RegionBoxSpec:
    if not isinstance(raw_box, dict):
        raise LLMBoxesValidationError(
            f"regions.{category}.boxes[{idx}] must be an object"
        )
    if set(raw_box) != {"min", "max"}:
        raise LLMBoxesValidationError(
            f"regions.{category}.boxes[{idx}] must contain only min and max"
        )
    item = RegionBoxSpec(
        category=category,
        min=_finite_point(raw_box["min"], f"regions.{category}.boxes[{idx}].min"),
        max=_finite_point(raw_box["max"], f"regions.{category}.boxes[{idx}].max"),
    )
    return _normalize_region(item)


def _parse_object_box(category: str, raw_box: Any, idx: int) -> ObjectBoxSpec:
    if not isinstance(raw_box, dict):
        raise LLMBoxesValidationError(
            f"objects.{category}.boxes[{idx}] must be an object"
        )
    if set(raw_box) != {"center", "half", "rotation"}:
        raise LLMBoxesValidationError(
            f"objects.{category}.boxes[{idx}] must contain only center, half, rotation"
        )
    item = ObjectBoxSpec(
        category=category,
        center=_finite_point(
            raw_box["center"], f"objects.{category}.boxes[{idx}].center"
        ),
        half_extents=_finite_point(
            raw_box["half"],
            f"objects.{category}.boxes[{idx}].half",
        ),
        rotation=_finite_number(
            raw_box["rotation"],
            f"objects.{category}.boxes[{idx}].rotation",
        ),
    )
    return _normalize_object(item)


def _normalize_object(item: ObjectBoxSpec) -> ObjectBoxSpec:
    _object_category_id(item.category)
    center = _finite_point(item.center, "object.center")
    half_extents = _finite_point(item.half_extents, "object.half_extents")
    if half_extents[0] <= 0.0 or half_extents[1] <= 0.0:
        raise LLMBoxesValidationError("object.half_extents must be positive")
    return ObjectBoxSpec(
        category=item.category,
        center=_round_point(center),
        half_extents=_round_positive_point(half_extents),
        rotation=_round_rotation(_finite_number(item.rotation, "object.rotation")),
    )


def _normalize_region(item: RegionBoxSpec) -> RegionBoxSpec:
    _region_category_id(item.category)
    min_point = _finite_point(item.min, "region.min")
    max_point = _finite_point(item.max, "region.max")
    if max_point[0] <= min_point[0] or max_point[1] <= min_point[1]:
        raise LLMBoxesValidationError(
            "region.max must be greater than min on both axes"
        )
    return RegionBoxSpec(
        category=item.category,
        min=_round_point(min_point),
        max=_round_point(max_point),
    )


def _object_category_id(category: Any) -> int:
    if category not in OBJECT_CATEGORY_TO_ID:
        raise LLMBoxesValidationError(f"unknown object category: {category!r}")
    return OBJECT_CATEGORY_TO_ID[category]


def _region_category_id(category: Any) -> int:
    if category not in REGION_CATEGORY_TO_ID:
        raise LLMBoxesValidationError(f"unknown region category: {category!r}")
    return REGION_CATEGORY_TO_ID[category]


def _finite_point(value: Any, field_name: str) -> Point2D:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise LLMBoxesValidationError(f"{field_name} must contain two numbers")
    return (
        _finite_number(value[0], f"{field_name}[0]"),
        _finite_number(value[1], f"{field_name}[1]"),
    )


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMBoxesValidationError(f"{field_name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise LLMBoxesValidationError(f"{field_name} must be a finite number")
    return value


def _point2(value: Sequence[float]) -> Point2D:
    return _finite_point(value, "point")


def _xz_point(value: Sequence[float], field_name: str) -> Point2D:
    if not isinstance(value, (list, tuple)):
        raise LLMBoxesValidationError(f"{field_name} must contain two or three numbers")
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
    raise LLMBoxesValidationError(f"{field_name} must contain two or three numbers")


def _trajectory_keypoint(value: Sequence[float]) -> Point2D:
    if not isinstance(value, (list, tuple)):
        raise LLMBoxesValidationError(
            "trajectory_keypoints points must contain two or three numbers"
        )
    if len(value) == 2:
        return _point2(value)
    if len(value) == 3:
        return (
            _finite_number(value[0], "trajectory_keypoint[0]"),
            _finite_number(value[2], "trajectory_keypoint[2]"),
        )
    raise LLMBoxesValidationError(
        "trajectory_keypoints points must contain two or three numbers"
    )


def _round_coord(value: float) -> float:
    return round(float(value), 1)


def _round_point(point: Point2D) -> Point2D:
    return (_round_coord(point[0]), _round_coord(point[1]))


def _round_positive_coord(value: float) -> float:
    rounded = _round_coord(value)
    if rounded <= 0.0 and value > 0.0:
        return 0.1
    return rounded


def _round_positive_point(point: Point2D) -> Point2D:
    return (_round_positive_coord(point[0]), _round_positive_coord(point[1]))


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
