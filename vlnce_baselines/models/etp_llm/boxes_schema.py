"""Structured LLM-Boxes schema and semantic-box conversion helpers."""

from __future__ import annotations

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "regions", tuple(self.regions))


@dataclass(frozen=True)
class LLMBoxesPartialParse:
    spec: LLMBoxesSpec
    dropped_text: str
    dropped_entity_count: int


class _LLMBoxesIncompleteEntity(LLMBoxesValidationError):
    pass


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
                    half_extents=_round_positive_point(box.half_extents),
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
    )


def spec_to_relevant_semantic_boxes(
    spec: LLMBoxesSpec,
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


def spec_to_llm_boxes_text(spec: LLMBoxesSpec) -> str:
    """Serialize a spec in the compact LLM-friendly linear grammar."""
    normalized = _normalize_spec(spec)
    parts: List[str] = []
    for item in normalized.objects:
        parts.append(
            f"obj {item.category} {item.center[0]} {item.center[1]} "
            f"{item.half_extents[0]} {item.half_extents[1]} {item.rotation}"
        )
    for item in normalized.regions:
        parts.append(
            f"reg {item.category} {item.min[0]} {item.min[1]} "
            f"{item.max[0]} {item.max[1]}"
        )
    return " ; ".join(parts) if parts else "none"


def parse_llm_boxes_text(
    text: str, allow_trailing_incomplete: bool = False
) -> LLMBoxesSpec:
    """Parse and validate the compact LLM-friendly linear grammar."""
    result = parse_llm_boxes_text_partial(text)
    if result.dropped_text and not allow_trailing_incomplete:
        raise LLMBoxesValidationError("trailing incomplete entity")
    return result.spec


def parse_llm_boxes_text_partial(text: str) -> LLMBoxesPartialParse:
    """Parse complete entities and drop one trailing incomplete entity."""
    stripped = text.strip()
    if stripped == "none" or stripped == "":
        return LLMBoxesPartialParse(
            spec=LLMBoxesSpec(objects=(), regions=()),
            dropped_text="",
            dropped_entity_count=0,
        )

    objects: List[ObjectBoxSpec] = []
    regions: List[RegionBoxSpec] = []
    entities = [entity.strip() for entity in stripped.split(";")]
    for idx, entity in enumerate(entities):
        if not entity:
            continue
        try:
            _parse_llm_boxes_entity(entity, idx, objects, regions)
        except _LLMBoxesIncompleteEntity:
            if idx == len(entities) - 1:
                return LLMBoxesPartialParse(
                    spec=LLMBoxesSpec(objects=tuple(objects), regions=tuple(regions)),
                    dropped_text=entity,
                    dropped_entity_count=1,
                )
            raise LLMBoxesValidationError("incomplete entity before end of output")
    return LLMBoxesPartialParse(
        spec=LLMBoxesSpec(objects=tuple(objects), regions=tuple(regions)),
        dropped_text="",
        dropped_entity_count=0,
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


def _parse_object(item: Any, idx: int) -> ObjectBoxSpec:
    if not isinstance(item, dict):
        raise LLMBoxesValidationError(f"objects[{idx}] must be an object")
    if set(item.keys()) != {"category", "center", "half_extents", "rotation"}:
        raise LLMBoxesValidationError(
            f"objects[{idx}] must contain category, center, half_extents, rotation"
        )

    category = item["category"]
    _object_category_id(category)
    center = _finite_point(item["center"], f"objects[{idx}].center")
    half_extents = _finite_point(item["half_extents"], f"objects[{idx}].half_extents")
    if half_extents[0] <= 0.0 or half_extents[1] <= 0.0:
        raise LLMBoxesValidationError(f"objects[{idx}].half_extents must be positive")
    rotation = _finite_number(item["rotation"], f"objects[{idx}].rotation")
    return ObjectBoxSpec(
        category=category,
        center=center,
        half_extents=half_extents,
        rotation=rotation,
    )


def _parse_region(item: Any, idx: int) -> RegionBoxSpec:
    if not isinstance(item, dict):
        raise LLMBoxesValidationError(f"regions[{idx}] must be an object")
    if set(item.keys()) != {"category", "min", "max"}:
        raise LLMBoxesValidationError(f"regions[{idx}] must contain category, min, max")

    category = item["category"]
    _region_category_id(category)
    min_point = _finite_point(item["min"], f"regions[{idx}].min")
    max_point = _finite_point(item["max"], f"regions[{idx}].max")
    if max_point[0] <= min_point[0] or max_point[1] <= min_point[1]:
        raise LLMBoxesValidationError(
            f"regions[{idx}].max must be greater than min on both axes"
        )
    return RegionBoxSpec(category=category, min=min_point, max=max_point)


def _parse_llm_boxes_entity(
    raw_entity: str,
    idx: int,
    objects: List[ObjectBoxSpec],
    regions: List[RegionBoxSpec],
) -> None:
    parts = raw_entity.split()
    if not parts:
        raise LLMBoxesValidationError(f"entity[{idx}] must not be empty")

    entity_type = parts[0]
    if entity_type == "obj":
        if len(parts) < 7:
            raise _LLMBoxesIncompleteEntity(f"entity[{idx}] object is incomplete")
        category = " ".join(parts[1:-5])
        if not category:
            raise _LLMBoxesIncompleteEntity(f"entity[{idx}] object category is missing")
        center_x, center_z, half_x, half_z, rotation = _parse_trailing_numbers(
            parts[-5:], f"entity[{idx}]"
        )
        item = ObjectBoxSpec(
            category=category,
            center=(center_x, center_z),
            half_extents=(half_x, half_z),
            rotation=rotation,
        )
        objects.append(_normalize_object(item))
        return

    if entity_type == "reg":
        if len(parts) < 6:
            raise _LLMBoxesIncompleteEntity(f"entity[{idx}] region is incomplete")
        category = " ".join(parts[1:-4])
        if not category:
            raise _LLMBoxesIncompleteEntity(f"entity[{idx}] region category is missing")
        min_x, min_z, max_x, max_z = _parse_trailing_numbers(
            parts[-4:], f"entity[{idx}]"
        )
        item = RegionBoxSpec(
            category=category,
            min=(min_x, min_z),
            max=(max_x, max_z),
        )
        regions.append(_normalize_region(item))
        return

    raise LLMBoxesValidationError(f"entity[{idx}] must start with obj or reg")


def _parse_trailing_numbers(tokens: Sequence[str], field_name: str) -> Tuple[float, ...]:
    values: List[float] = []
    for idx, token in enumerate(tokens):
        try:
            value = float(token)
        except ValueError as exc:
            raise _LLMBoxesIncompleteEntity(
                f"{field_name}[{idx}] must be a finite number"
            ) from exc
        values.append(_finite_number(value, f"{field_name}[{idx}]"))
    return tuple(values)


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
        raise LLMBoxesValidationError("region.max must be greater than min on both axes")
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


def _path_point(value: Sequence[float]) -> Point2D:
    if not isinstance(value, (list, tuple)):
        raise LLMBoxesValidationError(
            "reference_path points must contain two or three numbers"
        )
    if len(value) == 2:
        return _point2(value)
    if len(value) == 3:
        return (
            _finite_number(value[0], "path[0]"),
            _finite_number(value[2], "path[2]"),
        )
    raise LLMBoxesValidationError(
        "reference_path points must contain two or three numbers"
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
