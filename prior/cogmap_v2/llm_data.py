"""Export semantic graph SFT rows and measure Llama completion budgets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from tap import Tap

from model_paths import LLAMA_3_1_8B_INSTRUCT_MODEL
from .graph import CognitiveGraph, output_path
from .serialize import graph_to_text

REPO_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT_PATH = (
    REPO_ROOT / "vlnce_baselines/models/etp_llm/prompts/llm_graph_system.md"
)


def graph_prompt(graph: CognitiveGraph) -> str:
    """Known instruction and start pose only; no identity or route supervision."""
    position, direction = graph.meta.require_start_pose()
    return (
        f"Instruction: {graph.meta.instruction}\n"
        "Start pose (level-local meters): "
        f"position=({position[0]:.1f}, {position[1]:.1f}); "
        f"direction=({direction[0]:.4f}, {direction[1]:.4f})"
    )


def token_summary(counts: Sequence[int]) -> dict:
    values = np.asarray(counts, dtype=np.int64)
    if not len(values):
        raise ValueError("Cannot summarize empty token counts")
    return {
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": int(values.max()),
    }


def export_data(
    graph_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    *,
    tokenizer_source: str,
    max_new_tokens: Optional[int] = None,
) -> dict:
    """Write the same input_text/target_text contract as LLM-Grid training."""
    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()
    output_dir = output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "datasets": ["R2R"],
        "rxr_reason": "Graph CLI currently supports R2R only",
        "tokenizer_source": tokenizer_source,
        "tokenizer_vocabulary_size": len(tokenizer),
        "splits": {},
    }
    train_raw_counts = []
    train_completion_counts = []
    train_prompt_counts = []
    train_sequence_counts = []
    for split in ("train", "val_seen", "val_unseen"):
        paths = sorted((graph_dir / split).glob("*/*.json"))
        if not paths:
            raise FileNotFoundError(f"No graphs under {graph_dir / split}")
        destination = output_path(output_dir / f"{split}.jsonl")
        with destination.open("w", encoding="utf-8") as stream:
            for path in paths:
                graph = CognitiveGraph.load(path)
                if graph.meta.episode_id is None:
                    raise ValueError(f"Missing episode identity: {path}")
                target = graph_to_text(graph)
                prompt = graph_prompt(graph)
                row = {
                    "example_id": f"R2R_{split}_{graph.meta.episode_id}",
                    "dataset": "R2R",
                    "input_text": prompt,
                    "target_text": target,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if split == "train":
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                    prefix = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    sequence = tokenizer.apply_chat_template(
                        messages + [{"role": "assistant", "content": target}],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    if not sequence.startswith(prefix):
                        raise ValueError(
                            "Tokenizer chat template is not completion-prefix compatible"
                        )

                    def count(text: str) -> int:
                        return len(tokenizer.encode(text, add_special_tokens=False))

                    prompt_count, sequence_count = count(prefix), count(sequence)
                    train_raw_counts.append(count(target))
                    train_completion_counts.append(sequence_count - prompt_count)
                    train_prompt_counts.append(prompt_count)
                    train_sequence_counts.append(sequence_count)
        report["splits"][split] = {
            "episodes": len(paths),
            "file": str(destination.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    ordered = sorted(train_completion_counts)
    # Choose the smallest 128-token bucket retaining at least 99.5% of train.
    minimum = ordered[math.ceil(0.995 * len(ordered)) - 1]
    budget = (
        max_new_tokens if max_new_tokens is not None else math.ceil(minimum / 128) * 128
    )
    if budget <= 0:
        raise ValueError("max_new_tokens must be positive")
    dropped = sum(count > budget for count in train_completion_counts)
    if dropped / len(ordered) > 0.005:
        raise ValueError(
            f"Completion budget drops {dropped}/{len(ordered)}, exceeding 0.5%"
        )
    report.update({
        "raw_target_tokens": token_summary(train_raw_counts),
        "rendered_completion_tokens": token_summary(train_completion_counts),
        "rendered_prompt_tokens": token_summary(train_prompt_counts),
        "rendered_sequence_tokens": token_summary(train_sequence_counts),
        "max_new_tokens": budget,
        "dropped_completions": dropped,
        "dropped_fraction": dropped / len(ordered),
        "max_input_length": 1152,
        "max_sequence_length": 1152 + budget,
    })
    output_path(output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


class DataArgs(Tap):
    graph_dir: Path = Path("data_v2/cogmap_v2/graphs")
    output_dir: Path = Path("data_v2/llm_graph")
    tokenizer: str = LLAMA_3_1_8B_INSTRUCT_MODEL
    max_new_tokens: Optional[int] = None


def main() -> None:
    from transformers import AutoTokenizer

    args = DataArgs(underscores_to_dashes=True).parse_args()
    source = Path(args.tokenizer)
    if not source.is_absolute():
        source = REPO_ROOT / source
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    export_data(
        REPO_ROOT / args.graph_dir,
        args.output_dir,
        tokenizer,
        tokenizer_source=str(source),
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
