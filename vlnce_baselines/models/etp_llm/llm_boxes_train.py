"""Dataset, training, and evaluation CLI for the LLM-Boxes milestone."""

from __future__ import annotations

import json
import math
import warnings
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
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import torch
from tap import Tap
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import prior.bbox as bbox
from model_paths import LLAMA_3_1_8B_INSTRUCT_MODEL
from prior.bbox import RelevantSemanticBoxes
from prior.trajectory import WorldTrajectory3D
from prior.vlnce import VLNCEEpisodeEntry
from vlnce_baselines.models.etp_prior_gt.map_utils import (
    DEFAULT_COGNITIVE_MAP_NAMESPACE,
    cognitive_map_boxes_cache_path,
)

from .boxes_metrics import evaluate_llm_boxes_prediction
from .boxes_schema import (
    LLMBoxesSpec,
    build_llm_map_input,
    parse_llm_boxes_text,
    parse_llm_boxes_text_partial,
    relevant_semantic_boxes_to_mentioned_spec,
    spec_to_llm_boxes_text,
    spec_to_relevant_semantic_boxes,
    write_prediction_artifact,
)
from .sft import (
    ExampleLoadResult,
    LengthFilterResult,
    LengthGroupedBatchSampler,
    SourceLoadStats,
    TrainingIndex,
    TrainingManifest,
    configure_peft_fsdp,
    distributed_batch_metrics,
    enable_gradient_checkpointing as _enable_gradient_checkpointing,
    fixed_corpus_metrics,
    make_sft_accelerator,
    load_or_create_training_manifest,
    reduce_training_totals,
    rendered_token_counts,
    run_backward_preflight,
    save_peft_checkpoint,
    scale_training_loss,
    validate_fixed_corpus,
    validate_distributed_device_map,
)

DEFAULT_MODEL_NAME_OR_PATH = LLAMA_3_1_8B_INSTRUCT_MODEL
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "llm_boxes_system.md"
TRAIN_SPLITS = ("train",)
EVAL_SPLITS = ("val_seen", "val_unseen")
TRAINING_MANIFEST_NAME = "training_manifest.jsonl"
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
    split: str
    level_idx: int
    trajectory_keypoints: Sequence[Sequence[float]]
    start_direction: Sequence[float]
    start_position: Sequence[float]
    scene_id: str
    dataset: Literal["R2R", "RxR"]
    training_weight: float
    is_padding: bool


@dataclass
class LLMBoxesExample:
    example_id: str
    dataset: Literal["R2R", "RxR"]
    split: str
    scene_id: str
    episode_id: int
    instruction: str
    start_position: Sequence[float]
    start_direction: Sequence[float]
    ground_truth_trajectory: WorldTrajectory3D
    target_relevant: bbox.RelevantSemanticBoxes
    target_spec: LLMBoxesSpec = field(init=False)

    def __post_init__(self) -> None:
        self.target_spec = relevant_semantic_boxes_to_mentioned_spec(
            self.target_relevant
        )


@dataclass(frozen=True)
class _PreparedBoxesTrainingCorpus:
    load_stats: Mapping[str, SourceLoadStats]
    filtered: LengthFilterResult[LLMBoxesItem]
    sequence_lengths: Tuple[int, ...]
    text_stats: Mapping[str, float]
    corpus_metrics: Mapping[str, float]


@dataclass
class _BoxesEvaluationAccumulator:
    metric_sums: Dict[str, float] = field(
        default_factory=lambda: {key: 0.0 for key in AGGREGATE_METRIC_KEYS.values()}
    )
    example_count: int = 0
    schema_valid_count: int = 0
    partial_schema_valid_count: int = 0
    entity_valid_rate_sum: float = 0.0
    entity_valid_support_sum: float = 0.0

    def metrics(self) -> Dict[str, float]:
        metrics = (
            {key: value / self.example_count for key, value in self.metric_sums.items()}
            if self.example_count
            else {key: 0.0 for key in self.metric_sums}
        )
        metrics["examples"] = float(self.example_count)
        metrics["schema_valid_rate"] = _safe_rate(
            self.schema_valid_count, self.example_count
        )
        metrics["format_parse_rate"] = metrics["schema_valid_rate"]
        metrics["partial_schema_valid_rate"] = _safe_rate(
            self.partial_schema_valid_count, self.example_count
        )
        metrics["entity_valid_rate"] = (
            self.entity_valid_rate_sum / self.example_count
            if self.example_count
            else 0.0
        )
        metrics["entity_valid_support_mean"] = (
            self.entity_valid_support_sum / self.example_count
            if self.example_count
            else 0.0
        )
        return metrics


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def load_llm_boxes_examples(
    splits: Iterable[str],
    limit_per_dataset: Optional[int] = None,
    quiet: bool = False,
    skip_missing_cache: bool = False,
    cognitive_map_namespace: str = DEFAULT_COGNITIVE_MAP_NAMESPACE,
) -> ExampleLoadResult[LLMBoxesExample]:
    """Load VLN-CE episodes and attach cached target relevant semantic boxes."""
    examples: List[LLMBoxesExample] = []
    skipped_missing_cache: List[Tuple[str, str]] = []
    discovered = {"R2R": 0, "RxR": 0}
    loaded = {"R2R": 0, "RxR": 0}
    missing = {"R2R": [], "RxR": []}
    episodes = _progress(
        VLNCEEpisodeEntry.iter_r2r_rxr(
            splits=splits,
            limit_per_dataset=limit_per_dataset,
        ),
        desc="load LLM-Boxes examples",
        quiet=quiet,
        total=(limit_per_dataset * 2 if limit_per_dataset is not None else None),
    )
    for episode in episodes:
        dataset = _episode_dataset(episode.dataset, "LLM-Boxes")
        discovered[dataset] += 1
        try:
            target_relevant = RelevantSemanticBoxes.load(
                cognitive_map_boxes_cache_path(
                    episode.scene_id,
                    episode.unique_id,
                    namespace=cognitive_map_namespace,
                )
            )
        except FileNotFoundError as error:
            if not skip_missing_cache:
                raise
            warnings.warn(
                f"skipping {episode.unique_id}: missing cached boxes: {error.filename}",
                RuntimeWarning,
                stacklevel=2,
            )
            skipped_missing_cache.append((episode.unique_id, str(error.filename)))
            missing[dataset].append(episode.unique_id)
            continue
        examples.append(
            LLMBoxesExample(
                example_id=episode.unique_id,
                dataset=dataset,
                split=episode.split,
                scene_id=episode.scene_id,
                episode_id=episode.episode_id,
                instruction=episode.instruction,
                start_position=episode.start_position,
                start_direction=episode.start_direction_vector,
                ground_truth_trajectory=episode.ground_truth_trajectory,
                target_relevant=target_relevant,
            )
        )
        loaded[dataset] += 1
    if skipped_missing_cache and not quiet:
        print(f"skipped_missing_cache={len(skipped_missing_cache)}")
        for example_id, path in skipped_missing_cache:
            print(f"  {example_id}: {path}")
    return ExampleLoadResult(
        examples=tuple(examples),
        by_dataset={
            dataset: SourceLoadStats(
                discovered=discovered[dataset],
                loaded=loaded[dataset],
                missing_cache_example_ids=tuple(missing[dataset]),
            )
            for dataset in ("R2R", "RxR")
        },
    )


def _episode_dataset(
    dataset: str,
    target: str,
) -> Literal["R2R", "RxR"]:
    if dataset == "R2R":
        return "R2R"
    if dataset == "RxR":
        return "RxR"
    raise ValueError(f"unsupported {target} dataset: {dataset}")


class LLMBoxesDataset(Dataset):
    def __init__(self, examples: Sequence[LLMBoxesExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LLMBoxesItem:
        example = self.examples[index]
        return {
            "input_text": build_llm_map_input(
                example.instruction,
                _level_local_start_position(example),
                example.start_direction,
            ),
            "target_text": spec_to_llm_boxes_text(example.target_spec),
            "example_id": example.example_id,
            "target_spec": example.target_spec,
            "target_relevant": example.target_relevant,
            "instruction": example.instruction,
            "split": example.split,
            "level_idx": example.target_relevant.level_idx,
            "trajectory_keypoints": example.target_relevant.trajectory_keypoints,
            "start_direction": example.start_direction,
            "start_position": _level_local_start_position(example),
            "scene_id": example.scene_id,
            "dataset": example.dataset,
        }

    def __iter__(self) -> Iterator[LLMBoxesItem]:
        for index in range(len(self)):
            yield self[index]


class LLMBoxesItemDataset(Dataset):
    def __init__(self, items: Sequence[LLMBoxesItem]) -> None:
        self.items = tuple(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: Union[int, TrainingIndex]) -> LLMBoxesItem:
        if isinstance(index, TrainingIndex):
            item = self.items[index.index].copy()
            item["training_weight"] = index.loss_scale
            item["is_padding"] = index.is_padding
            return item
        return self.items[index]

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
    training_weights = torch.tensor(
        [item["training_weight"] for item in batch],
        dtype=torch.float32,
    )
    is_padding = torch.tensor(
        [item["is_padding"] for item in batch],
        dtype=torch.bool,
    )
    # Tokenizer output tensors use shape (B, T), where T is padded to the
    # longest prompt+completion sequence in this batch, capped by max_length.
    target_texts = [_target_text(item) for item in batch]
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
        add_special_tokens=False,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    # labels: (B, T). Prompt and padding positions are masked with -100 so the
    # causal LM loss only trains on compact LLM-Boxes completion tokens.
    encoded["labels"] = _causal_lm_labels(
        encoded["input_ids"],
        encoded["attention_mask"],
        prompt_lengths,
    )
    _validate_supervised_labels(
        encoded["labels"],
        [item["example_id"] for item in batch],
        prompt_lengths,
    )
    encoded["prompt_lengths"] = prompt_lengths
    encoded["example_ids"] = [item["example_id"] for item in batch]
    encoded["items"] = list(batch)
    encoded["training_weights"] = training_weights
    encoded["is_padding"] = is_padding
    return encoded


def collate_llm_boxes_prompt_batch(
    batch: Sequence[LLMBoxesItem],
    tokenizer: Any,
    system_prompt: str,
    max_input_length: int,
) -> Dict[str, Any]:
    # Prompt-only generation input: input_ids/attention_mask have shape (B, T).
    prompt_texts = [
        _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        for item in batch
    ]
    encoded = tokenizer(
        prompt_texts,
        add_special_tokens=False,
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


def _build_boxes_training_manifest(
    args: LLMBoxesArgs,
    tokenizer: Any,
    system_prompt: str,
    quiet: bool,
) -> TrainingManifest:
    load_result = load_llm_boxes_examples(
        TRAIN_SPLITS,
        limit_per_dataset=args.limit_per_dataset,
        quiet=quiet,
        skip_missing_cache=True,
        cognitive_map_namespace=args.cognitive_map_namespace,
    )
    validate_fixed_corpus(load_result.by_dataset)
    all_items = tuple(LLMBoxesDataset(load_result.examples))
    filtered = filter_llm_boxes_items_for_length(
        all_items,
        tokenizer,
        system_prompt,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )
    validate_fixed_corpus(load_result.by_dataset, retained_items=filtered.kept)
    rows = []
    for item in all_items:
        prompt_text = _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        completion_text = _render_chat_completion(
            tokenizer,
            system_prompt,
            item["input_text"],
            _target_text(item),
        )
        counts = rendered_token_counts(tokenizer, prompt_text, completion_text)
        rows.append(
            {
                "input_text": item["input_text"],
                "target_text": _target_text(item),
                "example_id": item["example_id"],
                "dataset": item["dataset"],
                "prompt_tokens": counts.prompt_tokens,
                "completion_tokens": counts.completion_tokens,
                "sequence_tokens": counts.sequence_tokens,
            }
        )
    text_stats = compute_llm_text_stats(
        filtered.kept,
        tokenizer,
        system_prompt,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )
    corpus_metrics = fixed_corpus_metrics(load_result.by_dataset, all_items, filtered)
    return TrainingManifest(
        metadata={
            "candidate": "llm_boxes",
            "source_stats": {
                dataset: {
                    "discovered": stats.discovered,
                    "loaded": stats.loaded,
                    "missing_cache_example_ids": list(stats.missing_cache_example_ids),
                }
                for dataset, stats in load_result.by_dataset.items()
            },
            "dropped_prompt_example_ids": list(filtered.dropped_prompt_example_ids),
            "dropped_completion_example_ids": list(
                filtered.dropped_completion_example_ids
            ),
            "text_stats": text_stats,
            "corpus_metrics": corpus_metrics,
        },
        items=tuple(rows),
    )


def _boxes_training_corpus_from_manifest(
    manifest: TrainingManifest,
) -> _PreparedBoxesTrainingCorpus:
    metadata = manifest.metadata
    if metadata.get("candidate") != "llm_boxes":
        raise ValueError("training manifest candidate must be llm_boxes")
    load_stats = _source_stats_from_manifest(metadata.get("source_stats"))
    kept_items: List[LLMBoxesItem] = []
    sequence_lengths: List[int] = []
    dropped_ids = set(
        _manifest_string_tuple(metadata, "dropped_prompt_example_ids")
    ) | set(_manifest_string_tuple(metadata, "dropped_completion_example_ids"))
    for row in manifest.items:
        dataset = row.get("dataset")
        if dataset not in ("R2R", "RxR"):
            raise ValueError("training manifest item dataset must be R2R or RxR")
        item: LLMBoxesItem = {
            "input_text": _manifest_string(row, "input_text"),
            "target_text": _manifest_string(row, "target_text"),
            "example_id": _manifest_string(row, "example_id"),
            "dataset": dataset,
        }
        if item["example_id"] not in dropped_ids:
            sequence_tokens = row.get("sequence_tokens")
            if not isinstance(sequence_tokens, int) or sequence_tokens < 1:
                raise ValueError(
                    "training manifest item sequence_tokens must be a positive integer"
                )
            kept_items.append(item)
            sequence_lengths.append(sequence_tokens)
    filtered = LengthFilterResult(
        kept=tuple(kept_items),
        dropped_prompt_example_ids=_manifest_string_tuple(
            metadata, "dropped_prompt_example_ids"
        ),
        dropped_completion_example_ids=_manifest_string_tuple(
            metadata, "dropped_completion_example_ids"
        ),
    )
    return _PreparedBoxesTrainingCorpus(
        load_stats=load_stats,
        filtered=filtered,
        sequence_lengths=tuple(sequence_lengths),
        text_stats=_manifest_float_mapping(metadata, "text_stats"),
        corpus_metrics=_manifest_float_mapping(metadata, "corpus_metrics"),
    )


def _source_stats_from_manifest(value: Any) -> Mapping[str, SourceLoadStats]:
    if not isinstance(value, dict):
        raise ValueError("training manifest source_stats must be an object")
    result: Dict[str, SourceLoadStats] = {}
    for dataset in ("R2R", "RxR"):
        raw = value.get(dataset)
        if not isinstance(raw, dict):
            raise ValueError(f"training manifest source_stats.{dataset} is missing")
        discovered = raw.get("discovered")
        loaded = raw.get("loaded")
        if not isinstance(discovered, int) or not isinstance(loaded, int):
            raise ValueError(
                f"training manifest source_stats.{dataset} counts must be integers"
            )
        result[dataset] = SourceLoadStats(
            discovered=discovered,
            loaded=loaded,
            missing_cache_example_ids=_manifest_string_tuple(
                raw, "missing_cache_example_ids"
            ),
        )
    return result


def _manifest_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"training manifest {key} must be a string")
    return value


def _manifest_string_tuple(row: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"training manifest {key} must be a list of strings")
    return tuple(value)


def _manifest_float_mapping(
    metadata: Mapping[str, Any], key: str
) -> Mapping[str, float]:
    value = metadata.get(key)
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(number, (int, float))
        for name, number in value.items()
    ):
        raise ValueError(f"training manifest {key} must contain numeric values")
    return {name: float(number) for name, number in value.items()}


def train_model(args: LLMBoxesArgs) -> Dict[str, float]:
    """Fine-tune a causal language model on LLM-Boxes examples."""
    _validate_llm_boxes_training_args(args)
    if args.finetune_method == "full":
        raise NotImplementedError("full fine-tuning is not implemented for LLM-Boxes")

    accelerator = make_sft_accelerator(args.gradient_accumulation_steps)
    validate_distributed_device_map(accelerator, args.device_map)
    batch_metrics = distributed_batch_metrics(
        accelerator,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
    )
    quiet = bool(getattr(args, "quiet", False))
    system_prompt = load_system_prompt()
    if accelerator.is_main_process:
        _write_run_system_prompt(args.output_dir, system_prompt)
    tokenizer = _load_causal_lm_tokenizer(args.model_name_or_path)
    manifest = load_or_create_training_manifest(
        accelerator,
        _artifact_dir(args.output_dir) / TRAINING_MANIFEST_NAME,
        lambda: _build_boxes_training_manifest(args, tokenizer, system_prompt, quiet),
    )
    corpus = _boxes_training_corpus_from_manifest(manifest)
    validate_fixed_corpus(
        corpus.load_stats,
        retained_items=corpus.filtered.kept,
    )

    model = _load_causal_lm_model(
        args.model_name_or_path,
        device_map=(
            None
            if accelerator.num_processes > 1
            else _normalize_device_map(args.device_map)
        ),
    )
    model = _apply_lora(model, args)
    _cast_trainable_parameters_to_float32(model)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    configure_peft_fsdp(accelerator, model)

    filtered_dataset = LLMBoxesItemDataset(corpus.filtered.kept)
    sequence_lengths = corpus.sequence_lengths
    batch_sampler = LengthGroupedBatchSampler(
        sequence_lengths,
        batch_size=args.per_device_batch_size,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=args.seed,
    )
    loader = DataLoader(
        filtered_dataset,
        batch_sampler=batch_sampler,
        collate_fn=lambda batch: collate_llm_boxes_batch(
            batch,
            tokenizer,
            system_prompt,
            args.max_input_length,
            args.max_new_tokens,
        ),
    )
    model = accelerator.prepare(model)
    trainable_parameters = [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    optimizer = accelerator.prepare(optimizer)

    model.train()
    longest_index = max(
        range(len(sequence_lengths)),
        key=lambda index: sequence_lengths[index],
    )
    longest_item = filtered_dataset[TrainingIndex(longest_index)]
    if accelerator.is_main_process and not quiet:
        print(
            "preflight_longest_sequence="
            f"{longest_item['example_id']}:{sequence_lengths[longest_index]}"
        )
    preflight_batch = collate_llm_boxes_batch(
        [longest_item],
        tokenizer,
        system_prompt,
        args.max_input_length,
        args.max_new_tokens,
    )
    run_backward_preflight(
        accelerator,
        model,
        optimizer,
        _model_batch(preflight_batch, accelerator.device),
        example_id=longest_item["example_id"],
        sequence_tokens=sequence_lengths[longest_index],
    )
    local_loss_sum = 0.0
    local_example_count = 0
    local_batch_count = 0
    optimizer_steps = 0
    for epoch in range(args.epochs):
        batch_sampler.set_epoch(epoch)
        progress_loader = _progress(
            loader,
            desc=f"train epoch {epoch + 1}/{args.epochs}",
            quiet=quiet,
            total=len(loader),
        )
        for batch in progress_loader:
            with accelerator.accumulate(model):
                outputs = model(**_model_batch(batch, accelerator.device))
                loss = outputs.loss
                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError(
                        _non_finite_step_message(
                            "loss",
                            epoch + 1,
                            local_batch_count + 1,
                            batch,
                        )
                    )
                weighted_loss = scale_training_loss(
                    loss,
                    batch["training_weights"],
                    batch["is_padding"],
                )
                accelerator.backward(weighted_loss)
                grad_norm = None
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )
                    if not torch.isfinite(grad_norm.detach()):
                        raise FloatingPointError(
                            _non_finite_step_message(
                                "gradient norm",
                                epoch + 1,
                                local_batch_count + 1,
                                batch,
                                value=float(grad_norm.detach().cpu()),
                            )
                        )
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    optimizer_steps += 1
                    _validate_trainable_parameters_finite(
                        model,
                        context=_non_finite_step_message(
                            "trainable parameter",
                            epoch + 1,
                            local_batch_count + 1,
                            batch,
                            value=(
                                float(grad_norm.detach().cpu())
                                if grad_norm is not None
                                else None
                            ),
                        ),
                    )
            real_example_count = int((~batch["is_padding"].bool()).sum().item())
            local_loss_sum += float(loss.detach().cpu()) * real_example_count
            local_example_count += real_example_count
            local_batch_count += 1
            set_postfix = getattr(progress_loader, "set_postfix", None)
            if callable(set_postfix):
                set_postfix(loss=float(loss.detach().cpu()))
        save_peft_checkpoint(
            accelerator,
            model,
            tokenizer,
            _checkpoint_dir(args.output_dir, f"epoch-{epoch + 1}"),
        )

    save_peft_checkpoint(
        accelerator,
        model,
        tokenizer,
        _checkpoint_dir(args.output_dir, "final"),
    )
    training_totals = reduce_training_totals(
        accelerator,
        local_loss_sum,
        local_example_count,
        local_batch_count,
    )
    metrics = {
        "train_loss": training_totals["loss"],
        "examples": training_totals["example_count"],
        "steps": training_totals["batch_count"],
        "optimizer_steps": float(optimizer_steps),
        "batches_per_epoch": training_totals["batch_count"] / args.epochs,
        "optimizer_steps_per_epoch": float(optimizer_steps) / args.epochs,
        "world_size": float(batch_metrics.world_size),
        "per_device_batch_size": float(batch_metrics.per_device_batch_size),
        "gradient_accumulation_steps": float(batch_metrics.gradient_accumulation_steps),
        "global_batch_size": float(batch_metrics.global_batch_size),
        "dropped_prompt_examples": float(
            len(corpus.filtered.dropped_prompt_example_ids)
        ),
        "dropped_completion_examples": float(
            len(corpus.filtered.dropped_completion_example_ids)
        ),
        **corpus.corpus_metrics,
        **corpus.text_stats,
    }
    if accelerator.is_main_process:
        _write_json(Path(args.output_dir) / "metrics.json", metrics)
    return metrics


def _evaluate_loaded_model(
    model: Any,
    tokenizer: Any,
    dataset: Iterable[LLMBoxesItem],
    args: Any,
    device: Any = None,
) -> Dict[str, float]:
    """Generate, validate, artifact, and score LLM-Boxes predictions."""
    evaluation_device = args.device if device is None else device
    if hasattr(model, "to") and not _model_uses_device_map(model):
        model.to(evaluation_device)
    if hasattr(model, "eval"):
        model.eval()
    tokenizer.padding_side = "left"

    system_prompt = getattr(args, "system_prompt", None) or load_system_prompt()
    _write_run_system_prompt(args.output_dir, system_prompt)
    items = tuple(dataset)
    loader = _iter_collated_batches(
        items,
        args.per_device_batch_size,
        lambda batch: collate_llm_boxes_prompt_batch(
            batch,
            tokenizer,
            system_prompt,
            args.max_input_length,
        ),
    )
    progress_loader = _progress(
        loader,
        desc="eval LLM-Boxes",
        quiet=bool(getattr(args, "quiet", False)),
        total=_batch_count(items, args.per_device_batch_size),
    )

    accumulators = {
        "combined": _BoxesEvaluationAccumulator(),
        "r2r": _BoxesEvaluationAccumulator(),
        "rxr": _BoxesEvaluationAccumulator(),
    }

    with torch.no_grad():
        for batch in progress_loader:
            model_inputs = _model_batch(
                batch,
                evaluation_device,
                include_labels=False,
            )
            generated = model.generate(
                **model_inputs,
                **_generation_kwargs(tokenizer, args.max_new_tokens),
            )
            decoded = [
                decode_generated_completion(tokenizer, sequence, prompt_length)
                for sequence, prompt_length in zip(generated, batch["prompt_lengths"])
            ]

            for item, generated_text in zip(batch["items"], decoded):
                item_accumulators = (
                    accumulators["combined"],
                    accumulators[item["dataset"].lower()],
                )
                entity_valid_rate, entity_valid_support = _entity_valid_stats(
                    generated_text
                )
                for accumulator in item_accumulators:
                    accumulator.example_count += 1
                    accumulator.entity_valid_rate_sum += entity_valid_rate
                    accumulator.entity_valid_support_sum += entity_valid_support

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

                if not pred_spec.trajectory_keypoints:
                    write_prediction_artifact(
                        _artifact_dir(args.output_dir),
                        item["example_id"],
                        invalid_text=generated_text,
                        error=ValueError("trajectory keypoints are required"),
                    )
                    continue

                for accumulator in item_accumulators:
                    accumulator.partial_schema_valid_count += 1
                try:
                    parse_llm_boxes_text(generated_text)
                except Exception:
                    pass
                else:
                    for accumulator in item_accumulators:
                        accumulator.schema_valid_count += 1

                pred_relevant = spec_to_relevant_semantic_boxes(
                    pred_spec,
                    instruction=item["instruction"],
                    level_idx=item["level_idx"],
                    start_direction_vector=item["start_direction"],
                    range_y=item["target_relevant"].level.range_y,
                )
                write_prediction_artifact(
                    _artifact_dir(args.output_dir),
                    item["example_id"],
                    valid_spec=pred_spec,
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
                            for accumulator in item_accumulators:
                                accumulator.metric_sums[metric_key] += float(value)

    metrics = accumulators["combined"].metrics()
    for prefix, dataset_items in (
        ("combined", items),
        ("r2r", tuple(item for item in items if item["dataset"] == "R2R")),
        ("rxr", tuple(item for item in items if item["dataset"] == "RxR")),
    ):
        group_metrics = accumulators[prefix].metrics()
        group_metrics.update(
            compute_llm_text_stats(
                dataset_items,
                tokenizer,
                system_prompt,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
            )
        )
        metrics.update(
            {f"{prefix}/{name}": value for name, value in group_metrics.items()}
        )
        if prefix == "combined":
            metrics.update(group_metrics)
    return metrics


def evaluate_model(args: LLMBoxesArgs) -> Dict[str, float]:
    """Load eval data/model, generate predictions, and write eval metrics."""
    accelerator = make_sft_accelerator(1)
    validate_distributed_device_map(accelerator, args.device_map)
    if not accelerator.is_main_process:
        return {}

    load_result = load_llm_boxes_examples(
        EVAL_SPLITS,
        limit_per_dataset=args.limit_per_dataset,
        quiet=args.quiet,
        skip_missing_cache=True,
        cognitive_map_namespace=args.cognitive_map_namespace,
    )
    validate_fixed_corpus(load_result.by_dataset)

    model_path = args.checkpoint_path or args.model_name_or_path
    model, tokenizer = _load_causal_lm_model_and_tokenizer(
        model_path,
        device_map=(
            None
            if accelerator.num_processes > 1
            else _normalize_device_map(args.device_map)
        ),
    )
    eval_items = tuple(LLMBoxesDataset(load_result.examples))
    validate_fixed_corpus(
        load_result.by_dataset,
        retained_items=eval_items,
    )
    metrics = _evaluate_loaded_model(
        model,
        tokenizer,
        eval_items,
        args,
        device=accelerator.device,
    )
    metrics.update(
        fixed_corpus_metrics(
            load_result.by_dataset,
            eval_items,
            LengthFilterResult(
                kept=eval_items,
                dropped_prompt_example_ids=(),
                dropped_completion_example_ids=(),
            ),
        )
    )
    _write_json(Path(args.output_dir) / "metrics.json", metrics)
    return metrics


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_causal_lm_model_and_tokenizer(
    model_name_or_path: str,
    device_map: Optional[str] = None,
) -> Tuple[Any, Any]:
    return (
        _load_causal_lm_model(model_name_or_path, device_map=device_map),
        _load_causal_lm_tokenizer(model_name_or_path),
    )


def _load_causal_lm_tokenizer(model_name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_causal_lm_model(
    model_name_or_path: str,
    device_map: Optional[str] = None,
) -> Any:
    from transformers import AutoModelForCausalLM

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": "auto",
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)


class LLMBoxesArgs(Tap):
    mode: Literal["train", "eval"]
    """Run mode."""
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH
    """Pretrained or checkpoint path for the causal language model."""
    checkpoint_path: Optional[str] = None
    output_dir: str = "outputs/llm_boxes"
    max_input_length: int = 1152
    max_new_tokens: int = 4096
    finetune_method: Literal["lora", "full"] = "lora"
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    epochs: int = 10
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    """Clip trainable parameter gradients to this norm; use 0 to disable."""
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    limit_per_dataset: Optional[int] = None
    seed: int = 42
    device: str = ""
    device_map: Literal["auto", "balanced", "balanced_low_0", "sequential", "none"] = (
        "auto"
    )
    """Optional Transformers/Accelerate model-parallel device map."""
    cognitive_map_namespace: str = DEFAULT_COGNITIVE_MAP_NAMESPACE
    """Cached boxes namespace under the cognitive-map cache root."""
    quiet: bool = False
    """Disable progress bars."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("underscores_to_dashes", True)
        super().__init__(*args, **kwargs)

    def configure(self) -> None:
        self.add_argument("mode")

    def process_args(self) -> None:
        if not self.device:
            self.device = _default_device()
        if self.mode == "train":
            _validate_llm_boxes_training_args(self)


def _validate_llm_boxes_training_args(args: LLMBoxesArgs) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be >= 1")
    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            "LLM-Boxes training requires "
            "--gradient-accumulation-steps 1 because its pre-partitioned "
            "loader cannot flush partial accumulation windows"
        )
    if not args.gradient_checkpointing:
        raise ValueError("--gradient-checkpointing is required for LLM-Boxes training")


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(argv: Optional[Sequence[str]] = None) -> LLMBoxesArgs:
    return LLMBoxesArgs().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    args = parse_args(argv)
    if args.mode == "train":
        return train_model(args)
    return evaluate_model(args)


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
    attention_mask: Any,
    prompt_lengths: Sequence[int],
) -> Any:
    if hasattr(input_ids, "clone"):
        labels = input_ids.clone()
        for row_idx, prompt_length in enumerate(prompt_lengths):
            labels[row_idx, :prompt_length] = -100
        labels = labels.masked_fill(attention_mask == 0, -100)
        return labels

    labels = []
    for row, mask, prompt_length in zip(input_ids, attention_mask, prompt_lengths):
        labels.append(
            [
                -100 if idx < prompt_length or mask[idx] == 0 else token
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


def _generation_kwargs(tokenizer: Any, max_new_tokens: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def _apply_lora(model: Any, args: LLMBoxesArgs) -> Any:
    if args.finetune_method != "lora":
        raise ValueError(f"Unsupported finetune method: {args.finetune_method}")
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(args.lora_target_modules),
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
    if example.target_relevant.trajectory_keypoints:
        return example.target_relevant.trajectory_keypoints[0]
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


def filter_llm_boxes_items_for_length(
    dataset: Iterable[LLMBoxesItem],
    tokenizer: Any,
    system_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
) -> LengthFilterResult[LLMBoxesItem]:
    """Drop items whose rendered prompt or target would be truncated."""
    kept: List[LLMBoxesItem] = []
    dropped_prompt: List[str] = []
    dropped_completion: List[str] = []
    for item in dataset:
        prompt_text = _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        completion_text = _render_chat_completion(
            tokenizer,
            system_prompt,
            item["input_text"],
            _target_text(item),
        )
        counts = rendered_token_counts(tokenizer, prompt_text, completion_text)
        prompt_over_budget = counts.prompt_tokens > max_input_length
        completion_over_budget = counts.completion_tokens > max_new_tokens
        if prompt_over_budget:
            dropped_prompt.append(item["example_id"])
        if completion_over_budget:
            dropped_completion.append(item["example_id"])
        if not prompt_over_budget and not completion_over_budget:
            kept.append(item)
    return LengthFilterResult(
        kept=tuple(kept),
        dropped_prompt_example_ids=tuple(dropped_prompt),
        dropped_completion_example_ids=tuple(dropped_completion),
    )


def compute_llm_text_stats(
    dataset: Iterable[LLMBoxesItem],
    tokenizer: Any,
    system_prompt: str,
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
        target_text = _target_text(item)
        prompt_text = _render_chat_prompt(
            tokenizer,
            system_prompt,
            item["input_text"],
        )
        completion_text = _render_chat_completion(
            tokenizer,
            system_prompt,
            item["input_text"],
            target_text,
        )
        counts = rendered_token_counts(tokenizer, prompt_text, completion_text)
        input_length = counts.prompt_tokens
        target_length = counts.completion_tokens
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
    try:
        spec = parse_llm_boxes_text(text)
    except Exception:
        return 0.0, 0
    entity_count = len(spec.objects) + len(spec.regions)
    return (1.0 if entity_count else 0.0), entity_count


def _token_count(tokenizer: Any, text: str) -> int:
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())


def _compact_entity_count(text: str) -> int:
    try:
        result = parse_llm_boxes_text_partial(text)
    except Exception:
        return 0
    return len(result.spec.objects) + len(result.spec.regions)


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


if __name__ == "__main__":
    main()
