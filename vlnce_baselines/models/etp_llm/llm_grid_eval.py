"""Score cached R2R LLM-Grid predictions without running model inference."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import AbstractSet, cast, Dict, FrozenSet, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from tap import Tap

from prior.constants import OBJECT_CATEGORIES

from .llm_grid_train import (
    DEFAULT_GRID_NAMESPACE,
    GRID_SCALE,
    LLMGridDataset,
    LLMGridValidationError,
    _LLMGridJSONError,
    _progress,
    load_llm_grid_examples,
    parse_grid_text,
)
from .navigation import (
    DEFAULT_LLM_NAVIGATION_MODEL_KEY,
    llm_navigation_prediction_path,
    llm_navigation_split_dir,
    validate_llm_navigation_manifest,
)

EVAL_SPLITS = ("val_seen", "val_unseen")
EpisodeCSVValue = Union[str, int, float]


@dataclass(frozen=True)
class EpisodeEvaluation:
    cache_model_key: str
    dataset: str
    split: str
    scene_id: str
    example_id: str
    instruction: str
    instruction_word_count: int
    instruction_character_count: int
    prediction_character_count: int
    target_category_cell_density: float
    target_spatial_cell_count: int
    target_spatial_density: float
    target_direction_count: int
    metrics: Dict[str, float]

    def csv_row(self) -> Dict[str, EpisodeCSVValue]:
        return {
            "cache_model_key": self.cache_model_key,
            "dataset": self.dataset,
            "split": self.split,
            "scene_id": self.scene_id,
            "example_id": self.example_id,
            "instruction": self.instruction,
            "instruction_word_count": self.instruction_word_count,
            "instruction_character_count": self.instruction_character_count,
            "prediction_character_count": self.prediction_character_count,
            "target_category_cell_density": self.target_category_cell_density,
            "target_spatial_cell_count": self.target_spatial_cell_count,
            "target_spatial_density": self.target_spatial_density,
            "target_direction_count": self.target_direction_count,
            **self.metrics,
        }


def _binary_grid(grid: NDArray[np.float32]) -> NDArray[np.bool_]:
    return np.asarray(grid > 0, dtype=np.bool_)


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _category_presence_metrics(
    pred_grid: NDArray[np.float32],
    target_grid: NDArray[np.float32],
    mentioned_object_categories: AbstractSet[int],
    mentioned_region_categories: AbstractSet[int],
) -> Dict[str, float]:
    pred_present = _binary_grid(pred_grid).reshape(pred_grid.shape[0], -1).any(axis=1)
    target_present = (
        _binary_grid(target_grid).reshape(target_grid.shape[0], -1).any(axis=1)
    )
    metrics: Dict[str, float] = {}
    region_categories = pred_grid.shape[0] - OBJECT_CATEGORIES
    partitions = (
        (
            "object",
            slice(None, OBJECT_CATEGORIES),
            _category_mask(mentioned_object_categories, OBJECT_CATEGORIES, "object"),
        ),
        (
            "region",
            slice(OBJECT_CATEGORIES, None),
            _category_mask(mentioned_region_categories, region_categories, "region"),
        ),
    )
    for name, channels, mentioned_mask in partitions:
        category_pred = pred_present[channels]
        category_target = target_present[channels]
        for prefix, mask in (
            ("", np.ones_like(mentioned_mask)),
            ("mentioned_", mentioned_mask),
            ("unmentioned_", np.logical_not(mentioned_mask)),
        ):
            pred = category_pred[mask]
            target = category_target[mask]
            intersection = int(np.logical_and(pred, target).sum())
            pred_count = int(pred.sum())
            target_count = int(target.sum())
            stem = f"{prefix}{name}_category"
            metrics.update({
                f"{stem}_true_positive_count": float(intersection),
                f"{stem}_predicted_count": float(pred_count),
                f"{stem}_target_count": float(target_count),
                f"{stem}_precision": _safe_div(intersection, pred_count),
                f"{stem}_recall": _safe_div(intersection, target_count),
                f"{stem}_f1": _safe_div(
                    2 * intersection,
                    pred_count + target_count,
                ),
            })
    return metrics


def _category_mask(
    categories: AbstractSet[int],
    size: int,
    name: str,
) -> NDArray[np.bool_]:
    invalid = sorted(category for category in categories if not 0 <= category < size)
    if invalid:
        raise ValueError(f"invalid mentioned {name} category IDs: {invalid}")
    mask = np.zeros(size, dtype=np.bool_)
    mask[list(categories)] = True
    return mask


@lru_cache(maxsize=None)
def _extract_instruction_mentions(
    instruction: str,
) -> Tuple[FrozenSet[int], FrozenSet[int]]:
    from prior.grid_map._cognitive import extract_categories

    objects, regions = extract_categories(instruction)
    return frozenset(objects), frozenset(regions)


def _spatial_partition_metrics(
    pred_grid: NDArray[np.float32],
    target_grid: NDArray[np.float32],
    mentioned_object_categories: AbstractSet[int],
    mentioned_region_categories: AbstractSet[int],
) -> Dict[str, float]:
    region_categories = pred_grid.shape[0] - OBJECT_CATEGORIES
    mentioned_mask = np.concatenate((
        _category_mask(mentioned_object_categories, OBJECT_CATEGORIES, "object"),
        _category_mask(mentioned_region_categories, region_categories, "region"),
    ))
    pred = _binary_grid(pred_grid)
    target = _binary_grid(target_grid)
    metrics: Dict[str, float] = {}
    for name, mask in (
        ("mentioned", mentioned_mask),
        ("unmentioned", np.logical_not(mentioned_mask)),
    ):
        partition_pred = pred[mask]
        partition_target = target[mask]
        intersection = int(np.logical_and(partition_pred, partition_target).sum())
        predicted_count = int(partition_pred.sum())
        target_count = int(partition_target.sum())
        union = int(np.logical_or(partition_pred, partition_target).sum())
        metrics.update({
            f"{name}_cell_true_positive_count": float(intersection),
            f"{name}_predicted_cell_count": float(predicted_count),
            f"{name}_target_cell_count": float(target_count),
            f"{name}_category_aware_raster_union_count": float(union),
            f"{name}_spatial_target_episode_count": float(target_count > 0),
            f"{name}_cell_precision": _safe_div(intersection, predicted_count),
            f"{name}_cell_recall": _safe_div(intersection, target_count),
            f"{name}_cell_f1": _safe_div(
                2 * intersection,
                predicted_count + target_count,
            ),
            f"{name}_category_aware_raster_iou": _safe_div(intersection, union),
        })
    return metrics


def _grid_metrics(
    pred_grid: NDArray[np.float32],
    target_grid: NDArray[np.float32],
    mentioned_object_categories: AbstractSet[int],
    mentioned_region_categories: AbstractSet[int],
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
        **_spatial_partition_metrics(
            pred_grid,
            target_grid,
            mentioned_object_categories,
            mentioned_region_categories,
        ),
        **_category_presence_metrics(
            pred_grid,
            target_grid,
            mentioned_object_categories,
            mentioned_region_categories,
        ),
    }


def _zero_direction_vector_metrics() -> Dict[str, float]:
    return {
        "direction_vector_valid_rate": 0.0,
        "direction_vector_l2": 0.0,
        "direction_vector_l2_support": 0.0,
        "direction_vector_cosine": 0.0,
        "direction_vector_cosine_support": 0.0,
        "direction_vector_padding_accuracy": 0.0,
    }


def _direction_vector_metrics(
    pred_vectors: NDArray[np.float32],
    target_vectors: NDArray[np.float32],
) -> Dict[str, float]:
    pred_norms = np.linalg.norm(pred_vectors, axis=1)
    target_norms = np.linalg.norm(target_vectors, axis=1)
    pred_nonzero = pred_norms > 0.0
    target_nonzero = target_norms > 0.0
    non_padding_count = int(np.count_nonzero(target_nonzero))
    if non_padding_count == 0:
        cosine = 0.0
    else:
        dot_products = np.sum(
            pred_vectors[target_nonzero] * target_vectors[target_nonzero],
            axis=1,
        )
        denominators = pred_norms[target_nonzero] * target_norms[target_nonzero]
        row_cosines = np.divide(
            dot_products,
            denominators,
            out=np.zeros_like(dot_products, dtype=np.float32),
            where=denominators > 0.0,
        )
        cosine = float(np.mean(row_cosines))
    return {
        "direction_vector_valid_rate": 1.0,
        "direction_vector_l2": float(
            np.mean(np.linalg.norm(pred_vectors - target_vectors, axis=1))
        ),
        "direction_vector_l2_support": 1.0,
        "direction_vector_cosine": cosine,
        "direction_vector_cosine_support": float(non_padding_count > 0),
        "direction_vector_padding_accuracy": float(
            np.mean(pred_nonzero == target_nonzero)
        ),
    }


def evaluate_grid_prediction(
    generated_text: str,
    target_grid: NDArray[np.float32],
    target_direction_vectors: NDArray[np.float32],
    mentioned_object_categories: AbstractSet[int],
    mentioned_region_categories: AbstractSet[int],
) -> Dict[str, float]:
    try:
        shape = cast(Tuple[int, int, int], target_grid.shape)
        parsed = parse_grid_text(generated_text, shape=shape)
    except LLMGridValidationError as error:
        return {
            "json_valid": float(not isinstance(error, _LLMGridJSONError)),
            "schema_valid": 0.0,
            "record_count": 0.0,
            "duplicate_record_count": 0.0,
            "duplicate_record_rate": 0.0,
            "cell_precision": 0.0,
            "cell_recall": 0.0,
            "cell_f1": 0.0,
            "category_aware_raster_iou": 0.0,
            "category_aware_raster_recall": 0.0,
            "category_aware_raster_support": float(np.count_nonzero(target_grid > 0)),
            "predicted_cell_count": 0.0,
            "target_cell_count": float(np.count_nonzero(target_grid > 0)),
            **_spatial_partition_metrics(
                np.zeros_like(target_grid),
                target_grid,
                mentioned_object_categories,
                mentioned_region_categories,
            ),
            **_category_presence_metrics(
                np.zeros_like(target_grid),
                target_grid,
                mentioned_object_categories,
                mentioned_region_categories,
            ),
            **_zero_direction_vector_metrics(),
        }
    metrics = _grid_metrics(
        parsed.grid,
        target_grid,
        mentioned_object_categories,
        mentioned_region_categories,
    )
    metrics.update(
        _direction_vector_metrics(
            parsed.direction_vectors,
            target_direction_vectors,
        )
    )
    metrics.update({
        "json_valid": 1.0,
        "schema_valid": 1.0,
        "record_count": float(parsed.record_count),
        "duplicate_record_count": float(parsed.duplicate_record_count),
        "duplicate_record_rate": _safe_div(
            parsed.duplicate_record_count,
            parsed.record_count,
        ),
    })
    return metrics


class LLMGridEvalArgs(Tap):
    cache_dir: str = ""
    cache_model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY
    output_dir: Path = Path("outputs/llm_grid_eval")
    cognitive_map_namespace: str = DEFAULT_GRID_NAMESPACE
    limit: Optional[int] = None
    quiet: bool = False

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("underscores_to_dashes", True)
        super().__init__(*args, **kwargs)

    def process_args(self) -> None:
        if self.limit is not None and self.limit < 0:
            raise ValueError("--limit must be >= 0")


def _aggregate_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    weighted_keys = {
        "direction_vector_l2": "direction_vector_l2_support",
        "direction_vector_cosine": "direction_vector_cosine_support",
    }
    metrics: Dict[str, float] = {}
    for key in keys:
        support_key = weighted_keys.get(key)
        if support_key is None:
            metrics[key] = float(sum(row.get(key, 0.0) for row in rows) / len(rows))
            continue
        support = sum(row.get(support_key, 0.0) for row in rows)
        metrics[key] = (
            float(
                sum(row.get(key, 0.0) * row.get(support_key, 0.0) for row in rows)
                / support
            )
            if support
            else 0.0
        )
    return metrics


def _summarize_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    metrics = _aggregate_metrics(rows)
    metrics.pop("missing_prediction", None)
    for name in ("mentioned", "unmentioned"):
        true_positive_count = sum(
            row[f"{name}_cell_true_positive_count"] for row in rows
        )
        predicted_count = sum(row[f"{name}_predicted_cell_count"] for row in rows)
        target_count = sum(row[f"{name}_target_cell_count"] for row in rows)
        union_count = sum(
            row[f"{name}_category_aware_raster_union_count"] for row in rows
        )
        target_episode_count = sum(
            row[f"{name}_spatial_target_episode_count"] for row in rows
        )
        metrics.update({
            f"{name}_cell_true_positive_count": true_positive_count,
            f"{name}_predicted_cell_count": predicted_count,
            f"{name}_target_cell_count": target_count,
            f"{name}_category_aware_raster_union_count": union_count,
            f"{name}_spatial_target_episode_count": target_episode_count,
            f"{name}_cell_precision": _safe_div(
                int(true_positive_count),
                int(predicted_count),
            ),
            f"{name}_cell_recall": _safe_div(
                int(true_positive_count),
                int(target_count),
            ),
            f"{name}_cell_f1": _safe_div(
                int(2 * true_positive_count),
                int(predicted_count + target_count),
            ),
            f"{name}_category_aware_raster_iou": _safe_div(
                int(true_positive_count),
                int(union_count),
            ),
        })
    for name in (
        "object",
        "region",
        "mentioned_object",
        "unmentioned_object",
        "mentioned_region",
        "unmentioned_region",
    ):
        true_positive_count = sum(
            row[f"{name}_category_true_positive_count"] for row in rows
        )
        predicted_count = sum(row[f"{name}_category_predicted_count"] for row in rows)
        target_count = sum(row[f"{name}_category_target_count"] for row in rows)
        metrics.update({
            f"{name}_category_true_positive_count": true_positive_count,
            f"{name}_category_predicted_count": predicted_count,
            f"{name}_category_target_count": target_count,
            f"{name}_category_precision": _safe_div(
                int(true_positive_count),
                int(predicted_count),
            ),
            f"{name}_category_recall": _safe_div(
                int(true_positive_count),
                int(target_count),
            ),
            f"{name}_category_f1": _safe_div(
                int(2 * true_positive_count),
                int(predicted_count + target_count),
            ),
        })
    example_count = len(rows)
    missing_count = sum(row["missing_prediction"] for row in rows)
    metrics.update({
        "examples": float(example_count),
        "predictions": float(example_count - missing_count),
        "missing_prediction_count": float(missing_count),
        "missing_prediction_rate": (
            float(missing_count / example_count) if example_count else 0.0
        ),
    })
    return metrics


def evaluate_cache(args: LLMGridEvalArgs) -> Dict[str, float]:
    _validate_manifests(args)
    loaded = load_llm_grid_examples(
        EVAL_SPLITS,
        limit_per_dataset=args.limit,
        quiet=args.quiet,
        cognitive_map_namespace=args.cognitive_map_namespace,
        datasets=("R2R",),
    )
    dataset = LLMGridDataset(loaded.examples, scale=GRID_SCALE)
    rows: list[Dict[str, float]] = []
    rows_by_split: Dict[str, list[Dict[str, float]]] = {
        "val_seen": [],
        "val_unseen": [],
    }
    episode_evaluations: list[EpisodeEvaluation] = []
    for item in _progress(
        dataset,
        desc="eval cached LLM-Grid",
        quiet=args.quiet,
        total=len(dataset),
    ):
        prediction_path = llm_navigation_prediction_path(
            item["scene_id"],
            item["example_id"],
            "R2R",
            item["split"],
            cache_dir=args.cache_dir,
            model_key=args.cache_model_key,
        )
        missing = not prediction_path.is_file()
        generated_text = "" if missing else prediction_path.read_text(encoding="utf-8")
        mentioned_objects, mentioned_regions = _extract_instruction_mentions(
            item["instruction"]
        )
        row = evaluate_grid_prediction(
            generated_text,
            item["target_grid"],
            item["target_direction_vectors"],
            mentioned_objects,
            mentioned_regions,
        )
        row["missing_prediction"] = float(missing)
        rows.append(row)
        rows_by_split[item["split"]].append(row)
        target_grid = item["target_grid"]
        target_direction_vectors = item["target_direction_vectors"]
        target_spatial_cell_count = int(
            np.count_nonzero(np.any(target_grid > 0, axis=0))
        )
        episode_evaluations.append(
            EpisodeEvaluation(
                cache_model_key=args.cache_model_key,
                dataset="R2R",
                split=item["split"],
                scene_id=item["scene_id"],
                example_id=item["example_id"],
                instruction=item["instruction"],
                instruction_word_count=len(item["instruction"].split()),
                instruction_character_count=len(item["instruction"]),
                prediction_character_count=len(generated_text),
                target_category_cell_density=float(np.count_nonzero(target_grid > 0))
                / float(target_grid.size),
                target_spatial_cell_count=target_spatial_cell_count,
                target_spatial_density=float(target_spatial_cell_count)
                / float(target_grid.shape[1] * target_grid.shape[2]),
                target_direction_count=int(
                    np.count_nonzero(np.linalg.norm(target_direction_vectors, axis=1))
                ),
                metrics=row,
            )
        )

    metrics = _summarize_rows(rows)
    for name, split_rows in (("combined", rows), *rows_by_split.items()):
        split_metrics = _summarize_rows(split_rows)
        metrics.update({f"{name}/{key}": value for key, value in split_metrics.items()})
    _write_metrics(args.output_dir / "metrics.json", metrics)
    _write_episode_evaluations(
        args.output_dir / "episodes.csv",
        episode_evaluations,
    )
    _write_diagnostics(
        args.output_dir / "diagnostics.json",
        episode_evaluations,
    )
    return metrics


def _write_episode_evaluations(
    path: Path,
    evaluations: Sequence[EpisodeEvaluation],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = sorted({
        key for evaluation in evaluations for key in evaluation.metrics
    })
    fieldnames = [
        "cache_model_key",
        "dataset",
        "split",
        "scene_id",
        "example_id",
        "instruction",
        "instruction_word_count",
        "instruction_character_count",
        "prediction_character_count",
        "target_category_cell_density",
        "target_spatial_cell_count",
        "target_spatial_density",
        "target_direction_count",
        *metric_fields,
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation.csv_row() for evaluation in evaluations)


def _select_representative_episodes(
    evaluations: Sequence[EpisodeEvaluation],
) -> Dict[str, Dict[str, EpisodeCSVValue]]:
    valid = sorted(
        (
            evaluation
            for evaluation in evaluations
            if evaluation.metrics["schema_valid"] == 1.0
        ),
        key=lambda evaluation: (
            evaluation.metrics["category_aware_raster_iou"],
            evaluation.example_id,
        ),
    )
    if not valid:
        return {}
    selected = {
        "worst": valid[0],
        "median": valid[len(valid) // 2],
        "best": valid[-1],
        "largest_category_spatial_gap": max(
            valid,
            key=lambda evaluation: (
                _combined_category_f1(evaluation) - evaluation.metrics["cell_f1"],
                evaluation.example_id,
            ),
        ),
    }
    return {
        label: {
            "scene_id": evaluation.scene_id,
            "example_id": evaluation.example_id,
            "category_aware_raster_iou": evaluation.metrics[
                "category_aware_raster_iou"
            ],
            "cell_f1": evaluation.metrics["cell_f1"],
            "object_category_f1": evaluation.metrics["object_category_f1"],
            "region_category_f1": evaluation.metrics["region_category_f1"],
            "combined_category_f1": _combined_category_f1(evaluation),
        }
        for label, evaluation in selected.items()
    }


def _combined_category_f1(evaluation: EpisodeEvaluation) -> float:
    true_positive_count = sum(
        evaluation.metrics[f"{name}_category_true_positive_count"]
        for name in ("object", "region")
    )
    predicted_count = sum(
        evaluation.metrics[f"{name}_category_predicted_count"]
        for name in ("object", "region")
    )
    target_count = sum(
        evaluation.metrics[f"{name}_category_target_count"]
        for name in ("object", "region")
    )
    return _safe_div(
        int(2 * true_positive_count),
        int(predicted_count + target_count),
    )


def _diagnostic_values(evaluation: EpisodeEvaluation) -> Dict[str, float]:
    return {
        "instruction_word_count": float(evaluation.instruction_word_count),
        "prediction_character_count": float(evaluation.prediction_character_count),
        "target_category_cell_density": evaluation.target_category_cell_density,
        "target_spatial_density": evaluation.target_spatial_density,
        "target_direction_count": float(evaluation.target_direction_count),
        "object_category_target_count": evaluation.metrics[
            "object_category_target_count"
        ],
        "region_category_target_count": evaluation.metrics[
            "region_category_target_count"
        ],
        "category_aware_raster_iou": evaluation.metrics["category_aware_raster_iou"],
        "cell_f1": evaluation.metrics["cell_f1"],
        "object_category_f1": evaluation.metrics["object_category_f1"],
        "region_category_f1": evaluation.metrics["region_category_f1"],
        "direction_vector_cosine": evaluation.metrics["direction_vector_cosine"],
    }


def _distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0.0, "p10": None, "p50": None, "p90": None}
    return {
        "count": float(array.size),
        "p10": float(np.quantile(array, 0.1)),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _pearson(
    evaluations: Sequence[EpisodeEvaluation],
    x_field: str,
    y_field: str,
) -> Dict[str, Optional[float]]:
    pairs = [
        (
            _diagnostic_values(evaluation)[x_field],
            _diagnostic_values(evaluation)[y_field],
        )
        for evaluation in evaluations
        if evaluation.metrics["schema_valid"] == 1.0
    ]
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if len(pairs) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        correlation = None
    else:
        correlation = float(np.corrcoef(x, y)[0, 1])
    return {"pearson": correlation, "count": float(len(pairs))}


def _diagnostics_for_split(
    evaluations: Sequence[EpisodeEvaluation],
) -> Dict[str, object]:
    values = [_diagnostic_values(evaluation) for evaluation in evaluations]
    distribution_fields = (
        "category_aware_raster_iou",
        "cell_f1",
        "object_category_f1",
        "region_category_f1",
        "direction_vector_cosine",
        "instruction_word_count",
        "target_category_cell_density",
        "target_spatial_density",
    )
    correlation_inputs = (
        "instruction_word_count",
        "target_category_cell_density",
        "target_spatial_density",
        "object_category_target_count",
        "region_category_target_count",
        "object_category_f1",
        "region_category_f1",
    )
    correlation_outputs = ("category_aware_raster_iou", "cell_f1")
    distributions = {
        field: _distribution([
            row[field]
            for evaluation, row in zip(evaluations, values)
            if field != "direction_vector_cosine"
            or evaluation.metrics["direction_vector_cosine_support"] > 0.0
        ])
        for field in distribution_fields
    }
    return {
        "expected_count": len(evaluations),
        "missing_example_ids": [
            evaluation.example_id
            for evaluation in evaluations
            if evaluation.metrics["missing_prediction"] == 1.0
        ],
        "invalid_example_ids": [
            evaluation.example_id
            for evaluation in evaluations
            if evaluation.metrics["missing_prediction"] == 0.0
            and evaluation.metrics["schema_valid"] == 0.0
        ],
        "distributions": distributions,
        "correlations": {
            f"{x_field}__{y_field}": _pearson(evaluations, x_field, y_field)
            for x_field in correlation_inputs
            for y_field in correlation_outputs
        },
        "representatives": _select_representative_episodes(evaluations),
    }


def _write_diagnostics(
    path: Path,
    evaluations: Sequence[EpisodeEvaluation],
) -> None:
    payload = {
        "selection_metric": "category_aware_raster_iou",
        "distribution_population": "expected_episodes_except_metric_support_filters",
        "correlation_population": "schema_valid_predictions",
        "splits": {
            split: _diagnostics_for_split([
                evaluation for evaluation in evaluations if evaluation.split == split
            ])
            for split in EVAL_SPLITS
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_manifests(args: LLMGridEvalArgs) -> None:
    for split in EVAL_SPLITS:
        validate_llm_navigation_manifest(
            llm_navigation_split_dir(
                "R2R",
                split,
                cache_dir=args.cache_dir,
                model_key=args.cache_model_key,
            ),
            {
                "dataset": "R2R",
                "split": split,
                "generator": "llm-grid",
                "scale": GRID_SCALE,
                "cache_model_key": args.cache_model_key,
            },
        )


def _write_metrics(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    args = LLMGridEvalArgs().parse_args(argv)
    return evaluate_cache(args)


if __name__ == "__main__":
    main()
