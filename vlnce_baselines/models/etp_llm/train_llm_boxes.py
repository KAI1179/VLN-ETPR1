"""Dataset, training, and evaluation CLI for the LLM-Boxes milestone."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sized
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import prior.bbox as bbox
from prior.bbox import SceneSemanticBoxes
from prior.vlnce import VLNCEEpisodeEntry
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .boxes_metrics import evaluate_llm_boxes_prediction
from .boxes_schema import (
    LLMBoxesSpec,
    build_llm_boxes_input,
    parse_llm_boxes_text,
    parse_llm_boxes_text_partial,
    relevant_semantic_boxes_to_mentioned_spec,
    spec_to_llm_boxes_text,
    spec_to_relevant_semantic_boxes,
    write_prediction_artifact,
)

DEFAULT_MODEL_NAME_OR_PATH = "data/models/Llama-3.1-8B-Instruct"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "llm_boxes_system.md"
TRAIN_SPLITS = ("train",)
EVAL_SPLITS = ("val_seen", "val_unseen")
AGGREGATE_METRIC_KEYS: Dict[str, str] = {
    "category_precision": "category_precision",
    "category_recall": "category_recall",
    "category_f1": "category_f1",
    "category_aware_raster_iou": "category_aware_raster_iou",
    "category_aware_raster_recall": "category_aware_raster_recall",
    "category_aware_raster_support": "category_aware_raster_support_mean",
}


class LLMBoxesItem(TypedDict, total=False):
    input_text: str
    target_text: str
    example_id: str
    target_spec: LLMBoxesSpec
    target_relevant: bbox.RelevantSemanticBoxes
    instruction: str
    level_idx: int
    reference_path: Sequence[Sequence[float]]
    start_direction: Sequence[float]


@dataclass
class LLMBoxesExample:
    example_id: str
    dataset_tag: str
    split: str
    episode_id: int
    instruction: str
    start_position: Sequence[float]
    start_direction: Sequence[float]
    reference_path: Sequence[Sequence[float]]
    target_relevant: bbox.RelevantSemanticBoxes
    target_spec: LLMBoxesSpec = field(init=False)

    def __post_init__(self) -> None:
        self.target_spec = relevant_semantic_boxes_to_mentioned_spec(
            self.target_relevant
        )


def load_llm_boxes_examples(
    dataset: Literal["R2R", "RxR"],
    splits: Iterable[str],
    limit: Optional[int] = None,
    quiet: bool = False,
) -> List[LLMBoxesExample]:
    """Load VLN-CE episodes and attach target relevant semantic boxes."""
    if limit == 0:
        return []

    examples: List[LLMBoxesExample] = []
    scene_cache: Dict[str, SceneSemanticBoxes] = {}
    episodes = _progress(
        VLNCEEpisodeEntry.iter_from(dataset, splits=splits),
        desc="load LLM-Boxes examples",
        quiet=quiet,
        total=limit,
    )
    for episode in episodes:
        scene_boxes = scene_cache.get(episode.scene_id)
        if scene_boxes is None:
            scene_boxes = SceneSemanticBoxes.from_scene_id(episode.scene_id)
            scene_cache[episode.scene_id] = scene_boxes

        target_relevant = scene_boxes.relevant_to(
            episode.instruction,
            episode.reference_path,
            episode.start_direction_vector,
        )
        examples.append(
            LLMBoxesExample(
                example_id=episode.unique_id,
                dataset_tag=episode.dataset,
                split=episode.split,
                episode_id=episode.episode_id,
                instruction=episode.instruction,
                start_position=episode.start_position,
                start_direction=episode.start_direction_vector,
                reference_path=episode.reference_path,
                target_relevant=target_relevant,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


class LLMBoxesDataset(Dataset):
    def __init__(self, examples: Sequence[LLMBoxesExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LLMBoxesItem:
        example = self.examples[index]
        return {
            "input_text": build_llm_boxes_input(
                example.dataset_tag,
                example.instruction,
                _level_local_start_position(example),
                example.start_direction,
            ),
            "target_text": spec_to_llm_boxes_text(example.target_spec),
            "example_id": example.example_id,
            "target_spec": example.target_spec,
            "target_relevant": example.target_relevant,
            "instruction": example.instruction,
            "level_idx": example.target_relevant.level_idx,
            "reference_path": example.target_relevant.reference_path,
            "start_direction": example.start_direction,
        }

    def __iter__(self) -> Iterator[LLMBoxesItem]:
        for index in range(len(self)):
            yield self[index]


def collate_llm_boxes_batch(
    batch: Sequence[LLMBoxesItem],
    tokenizer: Any,
    system_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    target_texts = [
        truncate_llm_boxes_text_at_entity_boundary(
            _target_text(item), tokenizer, max_new_tokens
        )
        for item in batch
    ]
    prompt_texts = [
        _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        for item in batch
    ]
    full_texts = [
        _render_chat_completion(
            tokenizer,
            system_prompt,
            item["input_text"],
            target_text,
        )
        for item, target_text in zip(batch, target_texts)
    ]
    prompt_lengths = [_token_count(tokenizer, text) for text in prompt_texts]
    max_length = max_input_length + max_new_tokens
    encoded = tokenizer(
        full_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded["labels"] = _causal_lm_labels(
        encoded["input_ids"],
        prompt_lengths,
        tokenizer.pad_token_id,
    )
    encoded["prompt_lengths"] = prompt_lengths
    encoded["example_ids"] = [item["example_id"] for item in batch]
    encoded["items"] = list(batch)
    return encoded


def collate_llm_boxes_prompt_batch(
    batch: Sequence[LLMBoxesItem],
    tokenizer: Any,
    system_prompt: str,
    max_input_length: int,
) -> Dict[str, Any]:
    prompt_texts = [
        _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        for item in batch
    ]
    encoded = tokenizer(
        prompt_texts,
        max_length=max_input_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    prompt_width = _encoded_width(encoded["input_ids"])
    encoded["prompt_lengths"] = [prompt_width] * len(batch)
    encoded["example_ids"] = [item["example_id"] for item in batch]
    encoded["items"] = list(batch)
    return encoded


def train_model(args: argparse.Namespace) -> Dict[str, float]:
    """Fine-tune a causal language model on LLM-Boxes examples."""
    if args.finetune_method == "full":
        raise NotImplementedError("full fine-tuning is not implemented for LLM-Boxes")

    quiet = bool(getattr(args, "quiet", False))
    examples = load_llm_boxes_examples(
        args.dataset,
        TRAIN_SPLITS,
        limit=args.limit,
        quiet=quiet,
    )
    if not examples:
        raise ValueError("No LLM-Boxes training examples were loaded")

    import torch
    from torch.utils.data import DataLoader

    system_prompt = load_system_prompt()
    _write_run_system_prompt(args.output_dir, system_prompt)
    model, tokenizer = _load_causal_lm_model_and_tokenizer(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
    )
    model = _apply_lora(model, args)
    device = torch.device(args.device)
    model.to(device)

    dataset = LLMBoxesDataset(examples)
    text_stats = compute_llm_text_stats(
        dataset,
        tokenizer,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_llm_boxes_batch(
            batch,
            tokenizer,
            system_prompt,
            args.max_input_length,
            args.max_new_tokens,
        ),
    )
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
    )

    model.train()
    total_loss = 0.0
    steps = 0
    for epoch in range(args.epochs):
        progress_loader = _progress(
            loader,
            desc=f"train epoch {epoch + 1}/{args.epochs}",
            quiet=quiet,
            total=len(loader),
        )
        for batch in progress_loader:
            model_inputs = _model_batch(batch, device)
            outputs = model(**model_inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach().cpu())
            steps += 1
            set_postfix = getattr(progress_loader, "set_postfix", None)
            if callable(set_postfix):
                set_postfix(loss=float(loss.detach().cpu()))
        save_llm_boxes_checkpoint(
            model,
            tokenizer,
            _checkpoint_dir(args.output_dir, f"epoch-{epoch + 1}"),
        )

    save_llm_boxes_checkpoint(
        model, tokenizer, _checkpoint_dir(args.output_dir, "final")
    )
    return {
        "train_loss": total_loss / steps if steps else 0.0,
        "steps": float(steps),
        **text_stats,
    }


def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: Iterable[LLMBoxesItem],
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Generate, validate, artifact, and score LLM-Boxes predictions."""
    import torch

    if hasattr(model, "to"):
        model.to(args.device)
    if hasattr(model, "eval"):
        model.eval()

    system_prompt = getattr(args, "system_prompt", None) or load_system_prompt()
    _write_run_system_prompt(args.output_dir, system_prompt)
    loader = _iter_collated_batches(
        dataset,
        args.batch_size,
        lambda batch: collate_llm_boxes_prompt_batch(
            batch,
            tokenizer,
            system_prompt,
            args.max_input_length,
        ),
    )
    text_stats = compute_llm_text_stats(
        dataset,
        tokenizer,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )
    progress_loader = _progress(
        loader,
        desc="eval LLM-Boxes",
        quiet=bool(getattr(args, "quiet", False)),
        total=_batch_count(dataset, args.batch_size),
    )

    metric_sums: Dict[str, float] = {key: 0.0 for key in AGGREGATE_METRIC_KEYS.values()}
    example_count = 0
    schema_valid_count = 0
    partial_schema_valid_count = 0
    entity_valid_rate_sum = 0.0
    entity_valid_support_sum = 0.0

    with torch.no_grad():
        for batch in progress_loader:
            model_inputs = _model_batch(batch, args.device, include_labels=False)
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            decoded = [
                decode_generated_completion(tokenizer, sequence, prompt_length)
                for sequence, prompt_length in zip(generated, batch["prompt_lengths"])
            ]

            for item, generated_text in zip(batch["items"], decoded):
                example_count += 1
                entity_valid_rate, entity_valid_support = _entity_valid_stats(
                    generated_text
                )
                entity_valid_rate_sum += entity_valid_rate
                entity_valid_support_sum += entity_valid_support

                try:
                    pred_spec = parse_llm_boxes_text(
                        generated_text,
                        allow_trailing_incomplete=True,
                    )
                except Exception as exc:
                    write_prediction_artifact(
                        _artifact_dir(args.output_dir),
                        item["example_id"],
                        invalid_text=generated_text,
                        error=exc,
                    )
                    continue

                partial_schema_valid_count += 1
                try:
                    parse_llm_boxes_text(generated_text)
                except Exception:
                    pass
                else:
                    schema_valid_count += 1

                write_prediction_artifact(
                    _artifact_dir(args.output_dir),
                    item["example_id"],
                    valid_spec=pred_spec,
                )
                pred_relevant = spec_to_relevant_semantic_boxes(
                    pred_spec,
                    instruction=item["instruction"],
                    level_idx=item["level_idx"],
                    reference_path=item["reference_path"],
                    start_direction_vector=item["start_direction"],
                    range_y=item["target_relevant"].level.range_y,
                )
                metrics = evaluate_llm_boxes_prediction(
                    pred_spec,
                    item["target_spec"],
                    pred_relevant,
                    item["target_relevant"],
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metric_key = AGGREGATE_METRIC_KEYS.get(key)
                        if metric_key is not None:
                            metric_sums[metric_key] += float(value)

    metrics = (
        {key: value / example_count for key, value in metric_sums.items()}
        if example_count
        else {}
    )
    metrics["examples"] = float(example_count)
    metrics["schema_valid_rate"] = (
        float(schema_valid_count) / float(example_count) if example_count else 0.0
    )
    metrics["format_parse_rate"] = metrics["schema_valid_rate"]
    metrics["partial_schema_valid_rate"] = (
        float(partial_schema_valid_count) / float(example_count)
        if example_count
        else 0.0
    )
    metrics["entity_valid_rate"] = (
        entity_valid_rate_sum / float(example_count) if example_count else 0.0
    )
    metrics["entity_valid_support_mean"] = (
        entity_valid_support_sum / float(example_count) if example_count else 0.0
    )
    metrics.update(text_stats)
    return metrics


def save_llm_boxes_checkpoint(
    model: Any, tokenizer: Any, output_dir: str | Path
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


def _load_causal_lm_model_and_tokenizer(
    model_name_or_path: str,
    torch_dtype: str,
) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=_resolve_torch_dtype(torch_dtype),
    )
    return model, tokenizer


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _add_common_args(subparsers.add_parser("train", help="Fine-tune LLM-Boxes"))
    _add_common_args(subparsers.add_parser("eval", help="Evaluate LLM-Boxes"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    args = parse_args(argv)
    if args.mode == "train":
        return train_model(args)

    model, tokenizer = _load_causal_lm_model_and_tokenizer(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
    )
    examples = load_llm_boxes_examples(
        args.dataset,
        EVAL_SPLITS,
        limit=args.limit,
        quiet=args.quiet,
    )
    metrics = evaluate_model(model, tokenizer, LLMBoxesDataset(examples), args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name-or-path",
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="Pretrained or checkpoint path for the causal language model.",
    )
    parser.add_argument("--output-dir", default=Path("./data/logs/llm/"))
    parser.add_argument("--dataset", default="R2R", choices=["R2R", "RxR"])
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--finetune-method",
        default="lora",
        choices=["lora", "full"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model loading dtype. auto uses the dtype declared by the checkpoint.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")


def load_system_prompt() -> str:
    prompt = DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"System prompt is empty: {DEFAULT_SYSTEM_PROMPT_PATH}")
    return prompt


def _write_run_system_prompt(output_dir: str | Path, system_prompt: str) -> None:
    artifact_dir = _artifact_dir(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "system_prompt.md").write_text(
        f"{system_prompt}\n",
        encoding="utf-8",
    )


def _render_chat_prompt(tokenizer: Any, system_prompt: str, user_text: str) -> str:
    return str(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def _render_chat_completion(
    tokenizer: Any,
    system_prompt: str,
    user_text: str,
    assistant_text: str,
) -> str:
    return str(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    )


def _causal_lm_labels(
    input_ids: Any,
    prompt_lengths: Sequence[int],
    pad_token_id: Optional[int],
) -> Any:
    if hasattr(input_ids, "clone"):
        labels = input_ids.clone()
        for row_idx, prompt_length in enumerate(prompt_lengths):
            labels[row_idx, :prompt_length] = -100
        if pad_token_id is not None:
            labels = labels.masked_fill(input_ids == pad_token_id, -100)
        return labels

    labels = []
    for row, prompt_length in zip(input_ids, prompt_lengths):
        labels.append(
            [
                -100 if idx < prompt_length or token == pad_token_id else token
                for idx, token in enumerate(row)
            ]
        )
    return labels


def decode_generated_completion(
    tokenizer: Any,
    generated_ids: Sequence[int],
    prompt_length: int,
) -> str:
    completion_ids = list(generated_ids)[prompt_length:]
    return str(
        tokenizer.batch_decode([completion_ids], skip_special_tokens=True)[0]
    ).strip()


def _apply_lora(model: Any, args: argparse.Namespace) -> Any:
    if args.finetune_method != "lora":
        raise ValueError(f"Unsupported finetune method: {args.finetune_method}")
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return get_peft_model(model, config)


def _resolve_torch_dtype(torch_dtype: str) -> Any:
    if torch_dtype == "auto":
        return "auto"

    import torch

    if torch_dtype == "float32":
        return torch.float32
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def _level_local_start_position(example: LLMBoxesExample) -> Sequence[float]:
    if example.target_relevant.reference_path:
        return example.target_relevant.reference_path[0]
    return example.start_position


def _artifact_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "artifacts"


def _checkpoint_dir(output_dir: str | Path, name: str) -> Path:
    return Path(output_dir) / "checkpoints" / name


def _model_batch(
    batch: Dict[str, Any],
    device: Any,
    include_labels: bool = True,
) -> Dict[str, Any]:
    keys = ["input_ids", "attention_mask"]
    if include_labels:
        keys.append("labels")

    model_inputs = {key: batch[key] for key in keys if key in batch}
    for key, value in list(model_inputs.items()):
        if hasattr(value, "to"):
            model_inputs[key] = value.to(device)
    return model_inputs


def _encoded_width(input_ids: Any) -> int:
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    return len(input_ids[0]) if input_ids else 0


def _target_text(item: LLMBoxesItem) -> str:
    if "target_text" in item:
        return item["target_text"]
    return spec_to_llm_boxes_text(item["target_spec"])


def truncate_llm_boxes_text_at_entity_boundary(
    text: str,
    tokenizer: Any,
    max_tokens: int,
) -> str:
    """Truncate compact LLM-Boxes text without keeping a partial entity."""
    if _token_count(tokenizer, text) <= max_tokens:
        return text
    if text.strip() == "none":
        return text

    kept_entities: List[str] = []
    for entity in [item.strip() for item in text.split(";") if item.strip()]:
        candidate = " ; ".join([*kept_entities, entity])
        if _token_count(tokenizer, candidate) > max_tokens:
            break
        kept_entities.append(entity)
    return " ; ".join(kept_entities) if kept_entities else "none"


def compute_llm_text_stats(
    dataset: Iterable[LLMBoxesItem],
    tokenizer: Any,
    max_input_length: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    input_lengths: List[int] = []
    target_lengths: List[int] = []
    target_entity_counts: List[int] = []
    input_truncated = 0
    target_truncated = 0
    examples = 0

    for item in dataset:
        examples += 1
        input_text = item["input_text"]
        target_text = _target_text(item)
        input_length = _token_count(tokenizer, input_text)
        target_length = _token_count(tokenizer, target_text)
        input_lengths.append(input_length)
        target_lengths.append(target_length)
        target_entity_counts.append(_compact_entity_count(target_text))
        if input_length > max_input_length:
            input_truncated += 1
        if target_length > max_new_tokens:
            target_truncated += 1

    return {
        "input_token_p50": _percentile(input_lengths, 0.50),
        "input_token_p90": _percentile(input_lengths, 0.90),
        "input_token_p95": _percentile(input_lengths, 0.95),
        "input_token_max": float(max(input_lengths)) if input_lengths else 0.0,
        "target_token_p50": _percentile(target_lengths, 0.50),
        "target_token_p90": _percentile(target_lengths, 0.90),
        "target_token_p95": _percentile(target_lengths, 0.95),
        "target_token_max": float(max(target_lengths)) if target_lengths else 0.0,
        "target_entity_p50": _percentile(target_entity_counts, 0.50),
        "target_entity_p90": _percentile(target_entity_counts, 0.90),
        "target_entity_p95": _percentile(target_entity_counts, 0.95),
        "target_entity_max": (
            float(max(target_entity_counts)) if target_entity_counts else 0.0
        ),
        "input_truncation_rate": (
            float(input_truncated) / float(examples) if examples else 0.0
        ),
        "target_truncation_rate": (
            float(target_truncated) / float(examples) if examples else 0.0
        ),
    }


def _iter_collated_batches(
    dataset: Iterable[LLMBoxesItem],
    batch_size: int,
    collate_fn,
) -> Iterator[Dict[str, Any]]:
    batch: List[LLMBoxesItem] = []
    for item in dataset:
        batch.append(item)
        if len(batch) >= batch_size:
            yield collate_fn(batch)
            batch = []
    if batch:
        yield collate_fn(batch)


def _progress(
    iterable: Iterable[Any],
    desc: str,
    quiet: bool,
    total: Optional[int] = None,
) -> Iterable[Any]:
    return tqdm(
        iterable,
        desc=desc,
        disable=quiet,
        dynamic_ncols=True,
        total=total,
    )


def _batch_count(dataset: Iterable[Any], batch_size: int) -> Optional[int]:
    if not isinstance(dataset, Sized):
        return None
    item_count = len(dataset)
    return (item_count + batch_size - 1) // batch_size


def _entity_valid_stats(text: str) -> Tuple[float, int]:
    stripped = text.strip()
    if stripped == "none" or stripped == "":
        try:
            parse_llm_boxes_text(stripped)
        except Exception:
            return 0.0, 0
        return 1.0, 0

    try:
        result = parse_llm_boxes_text_partial(stripped)
    except Exception:
        entities = [entity.strip() for entity in stripped.split(";") if entity.strip()]
        entities = [
            entity
            for entity in entities
            if entity.split() and entity.split()[0] in {"obj", "reg"}
        ]
        if not entities:
            return 0.0, 0
        valid_entities = sum(1 for entity in entities if _entity_is_valid(entity))
        return float(valid_entities) / float(len(entities)), len(entities)

    total_entities = (
        len(result.spec.objects)
        + len(result.spec.regions)
        + result.dropped_entity_count
    )
    if total_entities == 0:
        return 0.0, 0
    valid_entities = len(result.spec.objects) + len(result.spec.regions)
    return float(valid_entities) / float(total_entities), total_entities


def _entity_is_valid(text: str) -> bool:
    try:
        parse_llm_boxes_text(text)
    except Exception:
        return False
    return True


def _token_count(tokenizer: Any, text: str) -> int:
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())


def _compact_entity_count(text: str) -> int:
    if text.strip() == "none" or text.strip() == "":
        return 0
    return len([entity for entity in text.split(";") if entity.strip()])


def _percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    if quantile == 0.50:
        middle = len(sorted_values) // 2
        if len(sorted_values) % 2:
            return float(sorted_values[middle])
        return float((sorted_values[middle - 1] + sorted_values[middle]) / 2.0)
    index = math.ceil(quantile * len(sorted_values)) - 1
    return float(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def _default_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
