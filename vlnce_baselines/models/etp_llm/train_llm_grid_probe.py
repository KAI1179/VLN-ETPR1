"""Dataset, training, and evaluation CLI for the LLM-Grid-Probe milestone."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, cast

import numpy as np
from numpy.typing import NDArray

GRID_CHANNELS = 37
GRID_SCALE = 2
GRID_SHAPE = (GRID_CHANNELS, 50, 50)
DEFAULT_GRID_NAMESPACE = "gt.legacy.r1p5.direction5.v1"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "llm_grid_probe_system.md"


class LLMGridProbeValidationError(ValueError):
    """Raised when generated LLM-Grid-Probe JSON fails validation."""


class _LLMGridProbeJSONError(LLMGridProbeValidationError):
    """Raised when generated text is not valid JSON."""


@dataclass(frozen=True)
class ParsedGridProbe:
    grid: NDArray[np.float32]
    record_count: int
    duplicate_record_count: int


def parse_grid_probe_text(
    text: str,
    shape: Tuple[int, int, int] = GRID_SHAPE,
) -> ParsedGridProbe:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise _LLMGridProbeJSONError(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise LLMGridProbeValidationError("top-level JSON value must be an object")
    if set(payload) != {"grid"}:
        raise LLMGridProbeValidationError("top-level JSON object must contain only grid")
    records = payload["grid"]
    if not isinstance(records, list):
        raise LLMGridProbeValidationError("grid must be a list")

    channels, rows, cols = shape
    grid = np.zeros(shape, dtype=np.float32)
    seen: set[tuple[int, int, int]] = set()
    duplicates = 0
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, list) or len(raw_record) not in (3, 4):
            raise LLMGridProbeValidationError(
                f"grid[{index}] must be [category,row,col] or [category,row,col,value]"
            )
        category, row, col = raw_record[:3]
        if not all(type(item) is int for item in (category, row, col)):
            raise LLMGridProbeValidationError(f"grid[{index}] category,row,col must be ints")
        if not (0 <= category < channels and 0 <= row < rows and 0 <= col < cols):
            raise LLMGridProbeValidationError(f"grid[{index}] index out of bounds")
        value = 1.0
        if len(raw_record) == 4:
            raw_value = raw_record[3]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise LLMGridProbeValidationError(f"grid[{index}] value must be numeric")
            value = float(raw_value)
        if not (0.0 <= value <= 1.0):
            raise LLMGridProbeValidationError(f"grid[{index}] value out of range")
        key = (category, row, col)
        if key in seen:
            duplicates += 1
        seen.add(key)
        grid[category, row, col] = max(float(grid[category, row, col]), value)
    return ParsedGridProbe(
        grid=grid,
        record_count=len(records),
        duplicate_record_count=duplicates,
    )


def _binary_grid(grid: NDArray[np.float32]) -> NDArray[np.bool_]:
    return np.asarray(grid > 0, dtype=np.bool_)


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _grid_metrics(
    pred_grid: NDArray[np.float32],
    target_grid: NDArray[np.float32],
) -> Dict[str, float]:
    pred = _binary_grid(pred_grid)
    target = _binary_grid(target_grid)
    intersection = int(np.logical_and(pred, target).sum())
    pred_count = int(pred.sum())
    target_count = int(target.sum())
    union = int(np.logical_or(pred, target).sum())
    precision = _safe_div(intersection, pred_count)
    recall = _safe_div(intersection, target_count)
    f1 = _safe_div(2 * intersection, pred_count + target_count)
    return {
        "cell_precision": precision,
        "cell_recall": recall,
        "cell_f1": f1,
        "category_aware_raster_iou": _safe_div(intersection, union),
        "category_aware_raster_recall": recall,
        "category_aware_raster_support": float(target_count),
        "predicted_cell_count": float(pred_count),
        "target_cell_count": float(target_count),
    }


def evaluate_grid_probe_prediction(
    generated_text: str,
    target_grid: NDArray[np.float32],
) -> Dict[str, float]:
    try:
        shape = cast(Tuple[int, int, int], target_grid.shape)
        parsed = parse_grid_probe_text(generated_text, shape=shape)
    except LLMGridProbeValidationError as error:
        return {
            "json_valid": float(not isinstance(error, _LLMGridProbeJSONError)),
            "schema_valid": 0.0,
            "record_count": 0.0,
            "duplicate_record_count": 0.0,
            "duplicate_record_rate": 0.0,
            "cell_precision": 0.0,
            "cell_recall": 0.0,
            "cell_f1": 0.0,
            "category_aware_raster_iou": 0.0,
            "category_aware_raster_recall": 0.0,
            "category_aware_raster_support": float(
                np.count_nonzero(target_grid > 0)
            ),
            "predicted_cell_count": 0.0,
            "target_cell_count": float(np.count_nonzero(target_grid > 0)),
        }
    metrics = _grid_metrics(parsed.grid, target_grid)
    metrics.update(
        {
            "json_valid": 1.0,
            "schema_valid": 1.0,
            "record_count": float(parsed.record_count),
            "duplicate_record_count": float(parsed.duplicate_record_count),
            "duplicate_record_rate": _safe_div(
                parsed.duplicate_record_count,
                parsed.record_count,
            ),
        }
    )
    return metrics
