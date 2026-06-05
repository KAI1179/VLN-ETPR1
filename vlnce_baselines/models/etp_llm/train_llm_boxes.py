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
    start_position: Sequence[float]
    scene_id: str


@dataclass
class LLMBoxesExample:
    example_id: str
    dataset_tag: str
    split: str
    scene_id: str
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
    episodes = _progress(
        VLNCEEpisodeEntry.iter_from(dataset, splits=splits),
        desc="load LLM-Boxes examples",
        quiet=quiet,
        total=limit,
    )
    for episode in episodes:
        scene_boxes = SceneSemanticBoxes.from_scene_id(episode.scene_id)

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
                scene_id=episode.scene_id,
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
            "start_position": _level_local_start_position(example),
            "scene_id": example.scene_id,
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
    _validate_supervised_labels(
        encoded["labels"],
        [item["example_id"] for item in batch],
        prompt_lengths,
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
        device_map=_normalize_device_map(args.device_map),
    )
    model = _apply_lora(model, args)
    _cast_trainable_parameters_to_float32(model)
    device = torch.device(args.device)
    if not _model_uses_device_map(model):
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
    trainable_parameters = [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)

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
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(
                    _non_finite_step_message("loss", epoch + 1, steps + 1, batch)
                )
            loss.backward()
            max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))
            grad_norm = None
            if max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_grad_norm,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(grad_norm.detach()):
                    raise FloatingPointError(
                        _non_finite_step_message(
                            "gradient norm",
                            epoch + 1,
                            steps + 1,
                            batch,
                            value=float(grad_norm.detach().cpu()),
                        )
                    )
            optimizer.step()
            optimizer.zero_grad()
            _validate_trainable_parameters_finite(
                model,
                context=_non_finite_step_message(
                    "trainable parameter",
                    epoch + 1,
                    steps + 1,
                    batch,
                    value=(
                        float(grad_norm.detach().cpu())
                        if grad_norm is not None
                        else None
                    ),
                ),
            )
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

    if hasattr(model, "to") and not _model_uses_device_map(model):
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
    device_map: Optional[str] = None,
) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": "auto",
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
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
        device_map=_normalize_device_map(args.device_map),
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
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--finetune-method",
        default="lora",
        choices=["lora", "full"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Clip trainable parameter gradients to this norm; use 0 to disable.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=["auto", "balanced", "balanced_low_0", "sequential", "none"],
        help=(
            "Optional Transformers/Accelerate model-parallel device map. "
            "Use none for ordinary single-device loading."
        ),
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


def _validate_supervised_labels(
    labels: Any,
    example_ids: Sequence[str],
    prompt_lengths: Sequence[int],
) -> None:
    bad_examples: List[str] = []
    for row_idx, example_id in enumerate(example_ids):
        row = labels[row_idx]
        if hasattr(row, "ne"):
            supervised_tokens = int(row.ne(-100).sum().item())
            sequence_tokens = int(row.shape[-1])
        else:
            supervised_tokens = sum(1 for token in row if token != -100)
            sequence_tokens = len(row)
        if supervised_tokens == 0:
            bad_examples.append(
                (
                    f"{example_id} "
                    f"(prompt_tokens={prompt_lengths[row_idx]}, "
                    f"sequence_tokens={sequence_tokens})"
                )
            )

    if bad_examples:
        joined_examples = ", ".join(bad_examples)
        raise ValueError(
            "No supervised target tokens remain after tokenization/truncation "
            f"for: {joined_examples}. Increase --max-input-length or "
            "--max-new-tokens, or shorten the system prompt."
        )


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
        r=32,
        lora_alpha=64,
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


def _cast_trainable_parameters_to_float32(model: Any) -> None:
    for param in model.parameters():
        if getattr(param, "requires_grad", False) and hasattr(param, "data"):
            param.data = param.data.float()


def _validate_trainable_parameters_finite(model: Any, context: str) -> None:
    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        data = getattr(param, "data", None)
        if data is None or not hasattr(data, "isfinite"):
            continue
        if not data.isfinite().all().item():
            raise FloatingPointError(f"{context}; non-finite parameter={name}")


def _non_finite_step_message(
    kind: str,
    epoch: int,
    step: int,
    batch: Dict[str, Any],
    value: Optional[float] = None,
) -> str:
    parts = [f"Non-finite training {kind}", f"epoch={epoch}", f"step={step}"]
    if value is not None:
        parts.append(f"value={value}")
    example_ids = batch.get("example_ids")
    if example_ids:
        parts.append(f"examples={list(example_ids)}")
    labels = batch.get("labels")
    if labels is not None:
        parts.append(f"supervised_tokens={_supervised_token_counts(labels)}")
    return "; ".join(parts)


def _supervised_token_counts(labels: Any) -> List[int]:
    if hasattr(labels, "ne"):
        return [int(row.ne(-100).sum().item()) for row in labels]
    return [sum(1 for token in row if token != -100) for row in labels]


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


def _model_uses_device_map(model: Any) -> bool:
    return bool(getattr(model, "hf_device_map", None))


def _normalize_device_map(device_map: Optional[str]) -> Optional[str]:
    if device_map == "none":
        return None
    return device_map


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
