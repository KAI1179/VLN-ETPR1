"""T5-Boxes helpers for structured cognitive-map specifications."""

from .boxes_schema import (
    OBJECT_CATEGORY_TO_ID,
    REGION_CATEGORY_TO_ID,
    ObjectBoxSpec,
    RegionBoxSpec,
    T5BoxesSpec,
    T5BoxesValidationError,
    build_t5_boxes_input,
    parse_t5_boxes_text,
    parse_t5_boxes_text_partial,
    relevant_semantic_boxes_to_mentioned_spec,
    relevant_semantic_boxes_to_spec,
    spec_to_t5_boxes_text,
    spec_to_relevant_semantic_boxes,
    write_prediction_artifact,
)

__all__ = [
    "OBJECT_CATEGORY_TO_ID",
    "REGION_CATEGORY_TO_ID",
    "ObjectBoxSpec",
    "RegionBoxSpec",
    "T5BoxesSpec",
    "T5BoxesValidationError",
    "build_t5_boxes_input",
    "parse_t5_boxes_text",
    "parse_t5_boxes_text_partial",
    "relevant_semantic_boxes_to_mentioned_spec",
    "relevant_semantic_boxes_to_spec",
    "spec_to_t5_boxes_text",
    "spec_to_relevant_semantic_boxes",
    "write_prediction_artifact",
]
