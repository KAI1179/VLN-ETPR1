"""Score cached R2R LLM-Boxes predictions without running model inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from tap import Tap

from .boxes_metrics import evaluate_llm_boxes_prediction
from .boxes_schema import (
    parse_llm_boxes_text,
    spec_to_relevant_semantic_boxes,
)
from .llm_boxes_train import (
    DEFAULT_COGNITIVE_MAP_NAMESPACE,
    LLMBoxesDataset,
    _progress,
    load_llm_boxes_examples,
)
from .navigation import (
    DEFAULT_LLM_NAVIGATION_MODEL_KEY,
    llm_navigation_prediction_path,
    llm_navigation_split_dir,
    validate_llm_navigation_manifest,
)

EVAL_SPLITS = ("val_seen", "val_unseen")
AGGREGATE_METRIC_KEYS = {
    "category_precision": "category_precision",
    "category_recall": "category_recall",
    "category_f1": "category_f1",
    "category_aware_raster_iou": "category_aware_raster_iou",
    "category_aware_raster_recall": "category_aware_raster_recall",
    "category_aware_raster_support": "category_aware_raster_support_mean",
}


@dataclass
class _Accumulator:
    metric_sums: Dict[str, float] = field(
        default_factory=lambda: {key: 0.0 for key in AGGREGATE_METRIC_KEYS.values()}
    )
    example_count: int = 0
    prediction_count: int = 0
    schema_valid_count: int = 0
    partial_schema_valid_count: int = 0
    entity_valid_rate_sum: float = 0.0
    entity_valid_support_sum: float = 0.0

    def metrics(self) -> Dict[str, float]:
        metrics = {
            key: value / self.example_count if self.example_count else 0.0
            for key, value in self.metric_sums.items()
        }
        metrics.update({
            "examples": float(self.example_count),
            "predictions": float(self.prediction_count),
            "missing_prediction_count": float(
                self.example_count - self.prediction_count
            ),
            "missing_prediction_rate": _rate(
                self.example_count - self.prediction_count,
                self.example_count,
            ),
            "schema_valid_rate": _rate(
                self.schema_valid_count,
                self.example_count,
            ),
            "partial_schema_valid_rate": _rate(
                self.partial_schema_valid_count,
                self.example_count,
            ),
            "entity_valid_rate": (
                self.entity_valid_rate_sum / self.example_count
                if self.example_count
                else 0.0
            ),
            "entity_valid_support_mean": (
                self.entity_valid_support_sum / self.example_count
                if self.example_count
                else 0.0
            ),
        })
        metrics["format_parse_rate"] = metrics["schema_valid_rate"]
        return metrics


class LLMBoxesEvalArgs(Tap):
    cache_dir: str = ""
    cache_model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY
    output_dir: Path = Path("outputs/llm_boxes_eval")
    cognitive_map_namespace: str = DEFAULT_COGNITIVE_MAP_NAMESPACE
    limit: Optional[int] = None
    quiet: bool = False

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("underscores_to_dashes", True)
        super().__init__(*args, **kwargs)

    def process_args(self) -> None:
        if self.limit is not None and self.limit < 0:
            raise ValueError("--limit must be >= 0")


def evaluate_cache(args: LLMBoxesEvalArgs) -> Dict[str, float]:
    _validate_manifests(args)
    loaded = load_llm_boxes_examples(
        EVAL_SPLITS,
        limit_per_dataset=args.limit,
        quiet=args.quiet,
        cognitive_map_namespace=args.cognitive_map_namespace,
        datasets=("R2R",),
    )
    items = LLMBoxesDataset(loaded.examples)
    accumulators = {
        "combined": _Accumulator(),
        "val_seen": _Accumulator(),
        "val_unseen": _Accumulator(),
    }
    for item in _progress(
        items,
        desc="eval cached LLM-Boxes",
        quiet=args.quiet,
        total=len(items),
    ):
        selected = (accumulators["combined"], accumulators[item["split"]])
        for accumulator in selected:
            accumulator.example_count += 1

        prediction_path = llm_navigation_prediction_path(
            item["scene_id"],
            item["example_id"],
            "R2R",
            item["split"],
            cache_dir=args.cache_dir,
            model_key=args.cache_model_key,
        )
        if not prediction_path.is_file():
            continue
        generated_text = prediction_path.read_text(encoding="utf-8")
        entity_valid_rate, entity_valid_support = _entity_valid_stats(generated_text)
        for accumulator in selected:
            accumulator.prediction_count += 1
            accumulator.entity_valid_rate_sum += entity_valid_rate
            accumulator.entity_valid_support_sum += entity_valid_support

        try:
            pred_spec = parse_llm_boxes_text(
                generated_text,
                allow_trailing_incomplete=True,
            )
        except Exception:
            continue
        for accumulator in selected:
            accumulator.partial_schema_valid_count += 1
        try:
            parse_llm_boxes_text(generated_text)
        except Exception:
            pass
        else:
            for accumulator in selected:
                accumulator.schema_valid_count += 1

        pred_relevant = spec_to_relevant_semantic_boxes(
            pred_spec,
            instruction=item["instruction"],
            level_idx=item["level_idx"],
            start_direction_vector=item["start_direction"],
            range_y=item["target_relevant"].level.range_y,
        )
        row = evaluate_llm_boxes_prediction(
            pred_spec,
            item["target_spec"],
            pred_relevant,
            item["target_relevant"],
        )
        for source, metric_key in AGGREGATE_METRIC_KEYS.items():
            value = row.get(source)
            if isinstance(value, (int, float)):
                for accumulator in selected:
                    accumulator.metric_sums[metric_key] += float(value)

    metrics = accumulators["combined"].metrics()
    for name, accumulator in accumulators.items():
        metrics.update({
            f"{name}/{key}": value for key, value in accumulator.metrics().items()
        })
    _write_metrics(args.output_dir / "metrics.json", metrics)
    return metrics


def _validate_manifests(args: LLMBoxesEvalArgs) -> None:
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
                "cache_model_key": args.cache_model_key,
            },
        )


def _entity_valid_stats(text: str) -> tuple[float, int]:
    try:
        spec = parse_llm_boxes_text(text)
    except Exception:
        return 0.0, 0
    entity_count = len(spec.objects) + len(spec.regions)
    return (1.0 if entity_count else 0.0), entity_count


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _write_metrics(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    args = LLMBoxesEvalArgs().parse_args(argv)
    return evaluate_cache(args)


if __name__ == "__main__":
    main()
