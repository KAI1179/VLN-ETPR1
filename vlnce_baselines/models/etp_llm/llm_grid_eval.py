"""Score cached R2R LLM-Grid predictions without running model inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from tap import Tap

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
            **_zero_direction_vector_metrics(),
        }
    metrics = _grid_metrics(parsed.grid, target_grid)
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
        row = evaluate_grid_prediction(
            generated_text,
            item["target_grid"],
            item["target_direction_vectors"],
        )
        row["missing_prediction"] = float(missing)
        rows.append(row)
        rows_by_split[item["split"]].append(row)

    metrics = _summarize_rows(rows)
    for name, split_rows in (("combined", rows), *rows_by_split.items()):
        split_metrics = _summarize_rows(split_rows)
        metrics.update({f"{name}/{key}": value for key, value in split_metrics.items()})
    _write_metrics(args.output_dir / "metrics.json", metrics)
    return metrics


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
