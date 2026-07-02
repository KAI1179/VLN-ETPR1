"""Convert relevant semantic boxes into cognitive-map decoder targets."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from prior.bbox import OBB2D, RelevantSemanticBoxes
from prior.constants import DEPTH, WIDTH

from .map_utils import NUM_MAP_CATEGORIES


@dataclass(frozen=True)
class CognitiveMapBoxTarget:
    labels: torch.Tensor
    boxes: torch.Tensor

    def to(self, device: torch.device) -> "CognitiveMapBoxTarget":
        return CognitiveMapBoxTarget(
            labels=self.labels.to(device),
            boxes=self.boxes.to(device),
        )


def relevant_semantic_boxes_to_decoder_target(
    relevant: RelevantSemanticBoxes,
) -> CognitiveMapBoxTarget:
    labels: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []

    for category_id, category_boxes in enumerate(relevant.level.objects):
        for box in category_boxes:
            labels.append(category_id)
            boxes.append(_xyxy_to_normalized_cxczwh(_obb_envelope_xyxy(box)))

    object_category_count = len(relevant.level.objects)
    for category_id, category_boxes in enumerate(relevant.level.regions):
        for box in category_boxes:
            labels.append(object_category_count + category_id)
            boxes.append(
                _xyxy_to_normalized_cxczwh(
                    (box.min[0], box.min[1], box.max[0], box.max[1])
                )
            )

    if any(label < 0 or label >= NUM_MAP_CATEGORIES for label in labels):
        raise ValueError(f"decoder target labels must be in [0, {NUM_MAP_CATEGORIES})")

    return CognitiveMapBoxTarget(
        labels=torch.tensor(labels, dtype=torch.int64),
        boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
    )


def _obb_envelope_xyxy(box: OBB2D) -> tuple[float, float, float, float]:
    cos_value = math.cos(float(box.rotation))
    sin_value = math.sin(float(box.rotation))
    hx, hz = box.half_extents
    envelope_hx = abs(cos_value) * float(hx) + abs(sin_value) * float(hz)
    envelope_hz = abs(sin_value) * float(hx) + abs(cos_value) * float(hz)
    cx, cz = box.center
    return (
        float(cx) - envelope_hx,
        float(cz) - envelope_hz,
        float(cx) + envelope_hx,
        float(cz) + envelope_hz,
    )


def _xyxy_to_normalized_cxczwh(
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, z0, x1, z1 = box
    x0 = _clamp(float(x0), 0.0, WIDTH)
    z0 = _clamp(float(z0), 0.0, DEPTH)
    x1 = _clamp(float(x1), 0.0, WIDTH)
    z1 = _clamp(float(z1), 0.0, DEPTH)
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_z, max_z = min(z0, z1), max(z0, z1)
    width = max_x - min_x
    depth = max_z - min_z
    return (
        (min_x + width / 2.0) / WIDTH,
        (min_z + depth / 2.0) / DEPTH,
        width / WIDTH,
        depth / DEPTH,
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
