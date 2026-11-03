"""Train graph text with the existing LLM-Grid SFT loop and manifest contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from prior.cogmap_v2.graph import output_path
from prior.cogmap_v2.llm_data import REPO_ROOT, SYSTEM_PROMPT_PATH
from .llm_grid_train import (
    LLMGridArgs,
    _load_grid_training_tokenizer,
    train_model,
)
from .llm_boxes_train import _render_chat_completion, _render_chat_prompt
from .sft import TrainingManifest, _write_training_manifest, rendered_token_counts


class LLMGraphArgs(LLMGridArgs):
    training_data: Path = Path("data_v2/llm_graph/train.jsonl")
    output_dir: str = "data_v2/llm_graph/runs/r2r"
    tokenizer_name_or_path: str = ""
    """Tokenizer-only override for CPU --dry-run; training uses the model tokenizer."""
    system_prompt_path: Path = SYSTEM_PROMPT_PATH
    max_dropped_fraction: float = 0.005
    max_new_tokens: int = 640
    max_sequence_length: int = 1792
    cuda_cache_clear_min_sequence_length: int = 1024
    dry_run: bool = False


def build_graph_training_manifest(
    args: LLMGraphArgs,
    tokenizer: Any,
    system_prompt: str,
) -> TrainingManifest:
    source = REPO_ROOT / args.training_data
    rows = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("Graph training data is empty")
    seen = set()
    items = []
    dropped = []
    for row in rows:
        if set(row) != {"example_id", "dataset", "input_text", "target_text"}:
            raise ValueError(
                "Graph training rows must match the four-field SFT contract"
            )
        if not all(isinstance(value, str) and value for value in row.values()):
            raise ValueError("Training row fields must be nonempty strings")
        if row["dataset"] != "R2R" or not row["example_id"].startswith("R2R_train_"):
            raise ValueError("The graph corpus currently contains R2R train only")
        if row["example_id"] in seen:
            raise ValueError(f"Duplicate example ID: {row['example_id']}")
        seen.add(row["example_id"])
        prompt = _render_chat_prompt(tokenizer, system_prompt, row["input_text"])
        completion = _render_chat_completion(
            tokenizer, system_prompt, row["input_text"], row["target_text"]
        )
        counts = rendered_token_counts(tokenizer, prompt, completion)
        if (
            counts.prompt_tokens > args.max_input_length
            or counts.completion_tokens > args.max_new_tokens
            or counts.sequence_tokens > args.max_sequence_length
        ):
            dropped.append(row["example_id"])
            continue
        items.append({
            **row,
            "prompt_tokens": counts.prompt_tokens,
            "completion_tokens": counts.completion_tokens,
            "sequence_tokens": counts.sequence_tokens,
        })
    fraction = len(dropped) / len(rows)
    if fraction > args.max_dropped_fraction:
        raise ValueError(
            f"Graph token budgets drop {len(dropped)}/{len(rows)} "
            f"({fraction:.2%}) > {args.max_dropped_fraction:.2%}"
        )
    if not items:
        raise ValueError("Token filtering retained no graph training examples")
    return TrainingManifest(
        metadata={
            "candidate": "llm_graph",
            "datasets": ["R2R"],
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "skipped_over_budget_count": len(dropped),
            "skipped_over_budget_fraction": fraction,
            "dropped_example_ids": dropped,
            "token_budgets": {
                "max_input_length": args.max_input_length,
                "max_new_tokens": args.max_new_tokens,
                "max_sequence_length": args.max_sequence_length,
            },
            "fixed_corpus_metrics": {
                "r2r_discovered": len(rows),
                "r2r_retained": len(items),
                "rxr_discovered": 0,
            },
        },
        items=tuple(items),
    )


def main(argv: Optional[List[str]] = None) -> Dict[str, float]:
    args = LLMGraphArgs().parse_args(argv)
    if args.tokenizer_name_or_path and not args.dry_run:
        raise ValueError("--tokenizer-name-or-path is a CPU --dry-run override only")
    args.output_dir = str(output_path(Path(args.output_dir)))
    system_prompt = (REPO_ROOT / args.system_prompt_path).read_text().strip()
    if args.dry_run:
        tokenizer_source = args.tokenizer_name_or_path or args.model_name_or_path
        tokenizer = _load_grid_training_tokenizer(tokenizer_source)
        manifest = build_graph_training_manifest(args, tokenizer, system_prompt)
        destination = output_path(
            Path(args.output_dir) / "artifacts/training_manifest.jsonl"
        )
        _write_training_manifest(destination, manifest)
        metrics = {
            "examples": float(len(manifest.items)),
            "skipped_over_budget_count": float(
                manifest.metadata["skipped_over_budget_count"]
            ),
        }
        output_path(Path(args.output_dir) / "dry_run.json").write_text(
            json.dumps(
                {
                    "gpu_training_started": False,
                    "manifest": str(destination),
                    **metrics,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(metrics, indent=2))
        return metrics
    return train_model(
        args,
        system_prompt_override=system_prompt,
        manifest_builder=lambda tokenizer, prompt: build_graph_training_manifest(
            args, tokenizer, prompt
        ),
    )


if __name__ == "__main__":
    main()
