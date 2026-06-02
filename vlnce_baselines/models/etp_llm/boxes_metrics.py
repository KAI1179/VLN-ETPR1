"""Category-aware metrics for LLM-Boxes predictions."""

from __future__ import annotations

from collections import Counter
from typing import Counter as CounterType, Tuple, TypedDict

import numpy as np

import prior.bbox as bbox

from .boxes_schema import LLMBoxesSpec

EntityKey = Tuple[str, str]


class CategoryF1Metrics(TypedDict):
    category_precision: float
    category_recall: float
    category_f1: float


class CategoryAwareRasterMetrics(TypedDict):
    category_aware_raster_iou: float
    category_aware_raster_recall: float
    category_aware_raster_support: float


class LLMBoxesPredictionMetrics(CategoryF1Metrics, CategoryAwareRasterMetrics):
    pass


def category_f1(pred: LLMBoxesSpec, target: LLMBoxesSpec) -> CategoryF1Metrics:
    """Compute precision, recall, and F1 over semantic entity category counts."""
    pred_counts = _entity_counts(pred)
    target_counts = _entity_counts(target)
    true_positive = sum(
        min(pred_counts[key], target_counts[key])
        for key in pred_counts.keys() | target_counts.keys()
    )
    pred_total = sum(pred_counts.values())
    target_total = sum(target_counts.values())

    precision = _safe_divide(true_positive, pred_total)
    recall = _safe_divide(true_positive, target_total)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    return {
        "category_precision": precision,
        "category_recall": recall,
        "category_f1": f1,
    }


def category_aware_raster_metrics(
    pred: bbox.RelevantSemanticBoxes,
    target: bbox.RelevantSemanticBoxes,
    threshold: float = 0.0,
) -> CategoryAwareRasterMetrics:
    """Compute category-aware raster metrics over GT-present channels only.

    IoU and recall are macro-averaged over channels where the target has at
    least one positive cell. Predicted-only channels are ignored by these
    raster metrics and are handled by category F1. When the target has no
    GT-present channels, IoU and recall are defined as 0.0 with zero support.
    """
    pred_grid = pred.to_cognitive_map().grid > threshold
    target_grid = target.to_cognitive_map().grid > threshold
    target_channels = target_grid.reshape(target_grid.shape[0], -1)
    gt_present_channels = np.flatnonzero(target_channels.any(axis=1))

    if gt_present_channels.size == 0:
        return {
            "category_aware_raster_iou": 0.0,
            "category_aware_raster_recall": 0.0,
            "category_aware_raster_support": 0.0,
        }

    ious = []
    recalls = []
    for channel_idx in gt_present_channels:
        pred_channel = pred_grid[channel_idx]
        target_channel = target_grid[channel_idx]
        intersection = np.logical_and(pred_channel, target_channel).sum()
        union = np.logical_or(pred_channel, target_channel).sum()
        target_positive = target_channel.sum()

        ious.append(_safe_divide(float(intersection), float(union)))
        recalls.append(_safe_divide(float(intersection), float(target_positive)))

    return {
        "category_aware_raster_iou": float(np.mean(ious)),
        "category_aware_raster_recall": float(np.mean(recalls)),
        "category_aware_raster_support": float(gt_present_channels.size),
    }


def evaluate_llm_boxes_prediction(
    pred_spec: LLMBoxesSpec,
    target_spec: LLMBoxesSpec,
    pred_relevant: bbox.RelevantSemanticBoxes,
    target_relevant: bbox.RelevantSemanticBoxes,
) -> LLMBoxesPredictionMetrics:
    """Evaluate one LLM-Boxes prediction with category and raster metrics.

    Raster IoU and recall macro-average over GT-present channels only;
    predicted-only channels are ignored there and penalized by category F1.
    """
    return {
        **category_f1(pred_spec, target_spec),
        **category_aware_raster_metrics(pred_relevant, target_relevant),
    }


def _entity_counts(spec: LLMBoxesSpec) -> CounterType[EntityKey]:
    counts: CounterType[EntityKey] = Counter()
    counts.update(("object", item.category) for item in spec.objects)
    counts.update(("region", item.category) for item in spec.regions)
    return counts


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
