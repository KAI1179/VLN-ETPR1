"""Token distribution analysis for ETP LLM candidate datasets."""

from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from tap import Tap
from transformers import AutoTokenizer

from vlnce_baselines.models.etp_prior_gt.map_utils import (
    DEFAULT_COGNITIVE_MAP_NAMESPACE,
)

from .llm_boxes_train import (
    DEFAULT_MODEL_NAME_OR_PATH,
    LLMBoxesDataset,
    LLMBoxesItem,
    _compact_entity_count,
    _percentile,
    _render_chat_prompt,
    _target_text,
    _token_count,
    load_llm_boxes_examples,
    load_system_prompt,
)


@dataclass(frozen=True)
class TokenMeasurement:
    example_id: str
    split: str
    prompt_tokens: int
    input_tokens: int
    target_tokens: int
    target_entities: int


class TokenAnalysisArgs(Tap):
    target: Literal["llm-boxes"] = "llm-boxes"
    dataset: Literal["R2R", "RxR"] = "R2R"
    splits: str = "train,val_seen,val_unseen"
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH
    max_input_length: int = 1024
    max_new_tokens: int = 2048
    budgets: str = "1024,1152,1280,1408,1536,1664,1792,2048,2304,2560,3072"
    limit: Optional[int] = None
    cognitive_map_namespace: str = DEFAULT_COGNITIVE_MAP_NAMESPACE
    output_json: Optional[str] = None
    quiet: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("underscores_to_dashes", True)
        super().__init__(*args, **kwargs)


def analyze_llm_boxes_tokens(args: TokenAnalysisArgs) -> Dict[str, Any]:
    if args.target != "llm-boxes":
        raise ValueError(f"unsupported token analysis target: {args.target}")
    splits = _csv(args.splits)
    budgets = _int_csv(args.budgets)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    system_prompt = load_system_prompt()

    with contextlib.redirect_stdout(sys.stderr):
        examples = load_llm_boxes_examples(
            args.dataset,
            splits,
            limit=args.limit,
            quiet=args.quiet,
            skip_missing_cache=True,
            cognitive_map_namespace=args.cognitive_map_namespace,
        )

    measurements = _measure_llm_boxes_items(
        LLMBoxesDataset(examples),
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    return {
        "target": args.target,
        "dataset": args.dataset,
        "splits": splits,
        "example_count": len(measurements),
        "configured_budget": {
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "prompt_over_budget_count": sum(
                item.prompt_tokens > args.max_input_length for item in measurements
            ),
            "target_over_budget_count": sum(
                item.target_tokens > args.max_new_tokens for item in measurements
            ),
        },
        "prompt_tokens": _summarize(
            [item.prompt_tokens for item in measurements]
        ),
        "input_tokens": _summarize([item.input_tokens for item in measurements]),
        "target_tokens": _summarize([item.target_tokens for item in measurements]),
        "target_entities": _summarize(
            [item.target_entities for item in measurements]
        ),
        "over_budget": _over_budget(measurements, budgets),
        "by_split": _summarize_by_split(measurements, budgets),
        "largest_targets": [
            {
                "example_id": item.example_id,
                "split": item.split,
                "target_tokens": item.target_tokens,
                "target_entities": item.target_entities,
            }
            for item in sorted(
                measurements,
                key=lambda item: item.target_tokens,
                reverse=True,
            )[:10]
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = TokenAnalysisArgs().parse_args(argv)
    report = analyze_llm_boxes_tokens(args)
    text = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
    if args.output_json is not None:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return report


def _measure_llm_boxes_items(
    items: Iterable[LLMBoxesItem],
    tokenizer: Any,
    system_prompt: str,
) -> List[TokenMeasurement]:
    measurements: List[TokenMeasurement] = []
    for item in items:
        target_text = _target_text(item)
        prompt_text = _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        measurements.append(
            TokenMeasurement(
                example_id=item["example_id"],
                split=str(item.get("split", "")),
                prompt_tokens=_token_count(tokenizer, prompt_text),
                input_tokens=_token_count(tokenizer, item["input_text"]),
                target_tokens=_token_count(tokenizer, target_text),
                target_entities=_compact_entity_count(target_text),
            )
        )
    return measurements


def _summarize(values: Sequence[int]) -> Dict[str, float]:
    return {
        "n": float(len(values)),
        "mean": sum(values) / float(len(values)) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p97": _percentile(values, 0.97),
        "p98": _percentile(values, 0.98),
        "p99": _percentile(values, 0.99),
        "p99.5": _percentile(values, 0.995),
        "p99.9": _percentile(values, 0.999),
        "max": float(max(values)) if values else 0.0,
    }


def _over_budget(
    measurements: Sequence[TokenMeasurement],
    budgets: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    total = float(len(measurements))
    return {
        str(budget): {
            "count": count,
            "rate": count / total if total else 0.0,
        }
        for budget in budgets
        for count in [
            sum(item.target_tokens > budget for item in measurements),
        ]
    }


def _summarize_by_split(
    measurements: Sequence[TokenMeasurement],
    budgets: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    split_names = sorted({item.split for item in measurements})
    return {
        split: {
            "target_tokens": _summarize(
                [item.target_tokens for item in measurements if item.split == split]
            ),
            "over_budget": _over_budget(
                [item for item in measurements if item.split == split],
                budgets,
            ),
        }
        for split in split_names
    }


def _csv(text: str) -> List[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("CSV argument must contain at least one value")
    return values


def _int_csv(text: str) -> List[int]:
    return [int(value) for value in _csv(text)]


if __name__ == "__main__":
    main()
