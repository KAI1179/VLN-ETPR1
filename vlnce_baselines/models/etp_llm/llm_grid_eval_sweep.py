"""Compare cached LLM-Grid predictions from multiple ordered runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence

from tap import Tap

from .llm_grid_eval import LLMGridEvalArgs, evaluate_cache
from .llm_grid_train import DEFAULT_GRID_NAMESPACE


@dataclass(frozen=True)
class EvalRun:
    label: str
    cache_model_key: str

    @classmethod
    def load_manifest(cls, path: Path) -> tuple[EvalRun, ...]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("run manifest must be a non-empty JSON list")

        runs: list[EvalRun] = []
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"run manifest entry {index} must be an object")
            fields: dict[str, object] = {}
            for key, value in entry.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"run manifest entry {index} field names must be strings"
                    )
                fields[key] = value
            if set(fields) != {"label", "cache_model_key"}:
                raise ValueError(
                    f"run manifest entry {index} must contain exactly "
                    "'label' and 'cache_model_key'"
                )
            label = fields["label"]
            cache_model_key = fields["cache_model_key"]
            if not isinstance(label, str) or not label:
                raise ValueError(f"run manifest entry {index} has an invalid label")
            if not isinstance(cache_model_key, str) or not cache_model_key:
                raise ValueError(
                    f"run manifest entry {index} has an invalid cache_model_key"
                )
            runs.append(cls(label=label, cache_model_key=cache_model_key))

        cls._reject_duplicates([run.label for run in runs], "label")
        cls._reject_duplicates(
            [run.cache_model_key for run in runs],
            "cache_model_key",
        )
        return tuple(runs)

    @staticmethod
    def _reject_duplicates(values: Sequence[str], field: str) -> None:
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"duplicate {field}: {', '.join(duplicates)}")


@dataclass(frozen=True)
class EvalResult:
    label: str
    cache_model_key: str
    metrics: dict[str, float]


class LLMGridEvalSweepArgs(Tap):
    run_manifest: Path
    cache_dir: str = ""
    output_dir: Path = Path("outputs/llm_grid_eval_sweep")
    cognitive_map_namespace: str = DEFAULT_GRID_NAMESPACE
    limit: Optional[int] = None
    quiet: bool = False

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("underscores_to_dashes", True)
        super().__init__(*args, **kwargs)

    def process_args(self) -> None:
        if self.limit is not None and self.limit < 0:
            raise ValueError("--limit must be >= 0")


def evaluate_sweep(args: LLMGridEvalSweepArgs) -> tuple[EvalResult, ...]:
    runs = EvalRun.load_manifest(args.run_manifest)
    _invalidate_aggregate_outputs(args.output_dir)
    results: list[EvalResult] = []
    for index, run in enumerate(runs):
        eval_args = LLMGridEvalArgs()
        eval_args.cache_dir = args.cache_dir
        eval_args.cache_model_key = run.cache_model_key
        eval_args.output_dir = args.output_dir / "runs" / f"{index:03d}"
        eval_args.cognitive_map_namespace = args.cognitive_map_namespace
        eval_args.limit = args.limit
        eval_args.quiet = args.quiet
        results.append(
            EvalResult(
                label=run.label,
                cache_model_key=run.cache_model_key,
                metrics=evaluate_cache(eval_args),
            )
        )

    _validate_complete_comparable_results(results)
    _write_results(args.output_dir, results)
    return tuple(results)


def _invalidate_aggregate_outputs(output_dir: Path) -> None:
    for name in ("metrics.json", "metrics.csv"):
        (output_dir / name).unlink(missing_ok=True)


def _validate_complete_comparable_results(results: Sequence[EvalResult]) -> None:
    expected_examples: dict[str, float] = {}
    for result in results:
        for split in ("val_seen", "val_unseen"):
            missing = result.metrics[f"{split}/missing_prediction_count"]
            if missing:
                raise ValueError(
                    f"{result.label} has {int(missing)} missing {split} predictions"
                )
            examples = result.metrics[f"{split}/examples"]
            expected = expected_examples.setdefault(split, examples)
            if examples != expected:
                raise ValueError(
                    f"{result.label} has {int(examples)} {split} examples; "
                    f"expected {int(expected)}"
                )


def _write_results(output_dir: Path, results: Sequence[EvalResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "label": result.label,
            "cache_model_key": result.cache_model_key,
            "metrics": result.metrics,
        }
        for result in results
    ]
    json_path = output_dir / "metrics.json"
    json_temporary_path = json_path.with_suffix(".json.tmp")
    json_temporary_path.write_text(
        json.dumps({"runs": serializable}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metric_keys = sorted({key for result in results for key in result.metrics})
    csv_path = output_dir / "metrics.csv"
    csv_temporary_path = csv_path.with_suffix(".csv.tmp")
    with csv_temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["label", "cache_model_key", *metric_keys],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({
                "label": result.label,
                "cache_model_key": result.cache_model_key,
                **result.metrics,
            })
    json_temporary_path.replace(json_path)
    csv_temporary_path.replace(csv_path)


def main(argv: Optional[Sequence[str]] = None) -> tuple[EvalResult, ...]:
    args = LLMGridEvalSweepArgs().parse_args(argv)
    return evaluate_sweep(args)


if __name__ == "__main__":
    main()
