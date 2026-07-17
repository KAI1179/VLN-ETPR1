"""Shared supervised fine-tuning helpers for ETP LLM candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType, broadcast_object_list
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.utils.data import Sampler

ItemT = TypeVar("ItemT")
MetricItemT = TypeVar("MetricItemT", bound=Mapping[str, Any])


@dataclass(frozen=True)
class RenderedTokenCounts:
    prompt_tokens: int
    completion_tokens: int
    sequence_tokens: int


@dataclass(frozen=True)
class SourceLoadStats:
    discovered: int
    loaded: int
    missing_cache_example_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ExampleLoadResult(Generic[ItemT]):
    examples: Tuple[ItemT, ...]
    by_dataset: Mapping[str, SourceLoadStats]


@dataclass(frozen=True)
class LengthFilterResult(Generic[ItemT]):
    kept: Tuple[ItemT, ...]
    dropped_prompt_example_ids: Tuple[str, ...]
    dropped_completion_example_ids: Tuple[str, ...]
    dropped_sequence_example_ids: Tuple[str, ...]


@dataclass(frozen=True)
class TrainingIndex:
    index: int
    loss_scale: float = 1.0
    is_padding: bool = False


@dataclass(frozen=True)
class SFTBatchMetrics:
    world_size: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int


@dataclass(frozen=True)
class TrainingManifest:
    metadata: Mapping[str, Any]
    items: Tuple[Mapping[str, Any], ...]


TRAINING_MANIFEST_SCHEMA_VERSION = 1
TRAINING_MANIFEST_ITEM_KEYS = {
    "example_id",
    "dataset",
    "input_text",
    "target_text",
    "prompt_tokens",
    "completion_tokens",
    "sequence_tokens",
}


def load_or_create_training_manifest(
    accelerator: Accelerator,
    path: str | Path,
    build: Callable[[], TrainingManifest],
) -> TrainingManifest:
    manifest_path = Path(path)
    build_error: Exception | None = None
    outcome: List[Dict[str, str] | None] = [None]
    if accelerator.is_main_process:
        try:
            _write_training_manifest(manifest_path, build())
        except Exception as error:
            build_error = error
            outcome[0] = {
                "error_type": type(error).__name__,
                "message": str(error),
            }
    broadcast_object_list(outcome)
    if outcome[0] is not None:
        if build_error is not None:
            raise build_error
        raise RuntimeError(
            "training manifest startup failed on rank 0: "
            f"{outcome[0]['error_type']}: {outcome[0]['message']}"
        )
    accelerator.wait_for_everyone()
    return _read_training_manifest(manifest_path)


def _write_training_manifest(path: Path, manifest: TrainingManifest) -> None:
    _validate_training_manifest(manifest, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as manifest_file:
            header = {
                "record_type": "metadata",
                "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
                "metadata": dict(manifest.metadata),
            }
            manifest_file.write(json.dumps(header, sort_keys=True) + "\n")
            for item in manifest.items:
                record = {"record_type": "item", "item": dict(item)}
                manifest_file.write(json.dumps(record, sort_keys=True) + "\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_training_manifest(path: Path) -> TrainingManifest:
    with path.open(encoding="utf-8") as manifest_file:
        header_line = manifest_file.readline()
        if not header_line:
            raise ValueError(f"training manifest is empty: {path}")
        header = json.loads(header_line)
        if (
            not isinstance(header, dict)
            or set(header) != {"record_type", "schema_version", "metadata"}
            or header.get("record_type") != "metadata"
        ):
            raise ValueError(f"training manifest metadata header is invalid: {path}")
        if header.get("schema_version") != TRAINING_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "training manifest schema_version must be "
                f"{TRAINING_MANIFEST_SCHEMA_VERSION}: {path}"
            )
        metadata = header.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"training manifest metadata must be an object: {path}")
        items: List[Mapping[str, Any]] = []
        for line_number, line in enumerate(manifest_file, start=2):
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or set(record) != {"record_type", "item"}
                or record.get("record_type") != "item"
            ):
                raise ValueError(
                    f"training manifest item record is invalid at line {line_number}: "
                    f"{path}"
                )
            item = record.get("item")
            if not isinstance(item, dict):
                raise ValueError(
                    f"training manifest item must be an object at line {line_number}: "
                    f"{path}"
                )
            items.append(item)
    manifest = TrainingManifest(metadata=metadata, items=tuple(items))
    _validate_training_manifest(manifest, path)
    return manifest


def _validate_training_manifest(manifest: TrainingManifest, path: Path) -> None:
    if not isinstance(manifest.metadata, Mapping):
        raise ValueError(f"training manifest metadata must be a mapping: {path}")
    if not manifest.items:
        raise ValueError(f"training manifest has no items: {path}")
    seen_example_ids = set()
    for index, item in enumerate(manifest.items):
        if set(item) != TRAINING_MANIFEST_ITEM_KEYS:
            raise ValueError(
                f"training manifest item {index} has invalid fields: {path}"
            )
        example_id = item["example_id"]
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(
                f"training manifest item {index} has invalid example_id: {path}"
            )
        if example_id in seen_example_ids:
            raise ValueError(
                f"training manifest has duplicate example_id {example_id}: {path}"
            )
        seen_example_ids.add(example_id)
        if item["dataset"] not in ("R2R", "RxR"):
            raise ValueError(
                f"training manifest item {index} has invalid dataset: {path}"
            )
        for field_name in ("input_text", "target_text"):
            if not isinstance(item[field_name], str):
                raise ValueError(
                    f"training manifest item {index} has invalid {field_name}: {path}"
                )
        counts = []
        for field_name in (
            "prompt_tokens",
            "completion_tokens",
            "sequence_tokens",
        ):
            value = item[field_name]
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"training manifest item {index} has invalid {field_name}: {path}"
                )
            counts.append(value)
        prompt_tokens, completion_tokens, sequence_tokens = counts
        if sequence_tokens <= 0 or sequence_tokens != (
            prompt_tokens + completion_tokens
        ):
            raise ValueError(
                f"training manifest item {index} has inconsistent token counts: {path}"
            )


def rendered_token_counts(
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
) -> RenderedTokenCounts:
    prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    sequence_tokens = len(tokenizer.encode(completion_text, add_special_tokens=False))
    if sequence_tokens < prompt_tokens:
        raise ValueError("completion rendering is shorter than prompt rendering")
    return RenderedTokenCounts(
        prompt_tokens=prompt_tokens,
        completion_tokens=sequence_tokens - prompt_tokens,
        sequence_tokens=sequence_tokens,
    )


def fixed_corpus_metrics(
    load_stats: Mapping[str, SourceLoadStats],
    items: Sequence[MetricItemT],
    filtered: LengthFilterResult[MetricItemT],
) -> Dict[str, float]:
    dataset_by_id = {str(item["example_id"]): str(item["dataset"]) for item in items}
    kept_ids = {str(item["example_id"]) for item in filtered.kept}
    prompt_dropped_ids = set(filtered.dropped_prompt_example_ids)
    completion_dropped_ids = set(filtered.dropped_completion_example_ids)
    sequence_dropped_ids = set(filtered.dropped_sequence_example_ids)
    metrics: Dict[str, float] = {}
    for dataset, prefix in (("R2R", "r2r"), ("RxR", "rxr")):
        stats = load_stats[dataset]
        metrics.update(
            {
                f"{prefix}_discovered": float(stats.discovered),
                f"{prefix}_loaded": float(stats.loaded),
                f"{prefix}_missing_cache": float(len(stats.missing_cache_example_ids)),
                f"{prefix}_prompt_dropped": float(
                    sum(
                        dataset_by_id[item_id] == dataset
                        for item_id in prompt_dropped_ids
                    )
                ),
                f"{prefix}_completion_dropped": float(
                    sum(
                        dataset_by_id[item_id] == dataset
                        for item_id in completion_dropped_ids
                    )
                ),
                f"{prefix}_sequence_dropped": float(
                    sum(
                        dataset_by_id[item_id] == dataset
                        for item_id in sequence_dropped_ids
                    )
                ),
                f"{prefix}_retained": float(
                    sum(dataset_by_id[item_id] == dataset for item_id in kept_ids)
                ),
            }
        )
    for name in (
        "discovered",
        "loaded",
        "missing_cache",
        "prompt_dropped",
        "completion_dropped",
        "sequence_dropped",
        "retained",
    ):
        metrics[f"combined_{name}"] = metrics[f"r2r_{name}"] + metrics[f"rxr_{name}"]
    return metrics


def validate_fixed_corpus(
    load_stats: Mapping[str, SourceLoadStats],
    *,
    retained_items: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Require both fixed corpus sources at every completed data stage."""
    for dataset in ("R2R", "RxR"):
        stats = load_stats.get(dataset)
        if stats is None or stats.discovered == 0:
            raise ValueError(f"fixed corpus source {dataset} discovered zero examples")
        if stats.loaded == 0:
            raise ValueError(f"fixed corpus source {dataset} loaded zero examples")
    if retained_items is None:
        return
    retained_datasets = {str(item["dataset"]) for item in retained_items}
    for dataset in ("R2R", "RxR"):
        if dataset not in retained_datasets:
            raise ValueError(f"fixed corpus source {dataset} retained zero examples")


def make_sft_accelerator(gradient_accumulation_steps: int) -> Accelerator:
    accelerator_kwargs: Dict[str, Any] = {}
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            limit_all_gathers=True,
            use_orig_params=False,
            sync_module_states=True,
            activation_checkpointing=False,
        )
        expected = {
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "limit_all_gathers": True,
            "use_orig_params": False,
            "sync_module_states": True,
            "activation_checkpointing": False,
        }
        conflicts = [
            name
            for name, value in expected.items()
            if getattr(fsdp_plugin, name) != value
        ]
        if conflicts:
            raise ValueError(
                "conflicting FSDP environment configuration: " + ", ".join(conflicts)
            )
        accelerator_kwargs["fsdp_plugin"] = fsdp_plugin
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        **accelerator_kwargs,
    )


def configure_peft_fsdp(accelerator: Accelerator, model: Any) -> None:
    if accelerator.distributed_type != DistributedType.FSDP:
        return
    from peft.utils.other import fsdp_auto_wrap_policy

    plugin = cast(FullyShardedDataParallelPlugin, accelerator.state.fsdp_plugin)
    plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)


def save_peft_checkpoint(
    accelerator: Accelerator,
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        output_path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            output_path,
            state_dict=state_dict,
            is_main_process=True,
        )
        tokenizer.save_pretrained(output_path)
    accelerator.wait_for_everyone()


def run_backward_preflight(
    accelerator: Accelerator,
    model: Any,
    optimizer: Any,
    model_inputs: Mapping[str, Any],
    *,
    example_id: str,
    sequence_tokens: int,
) -> None:
    log_fsdp_layout(accelerator, model)
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()
        log_cuda_memory(accelerator, stage="after_empty_cache")
    try:
        loss = model(**model_inputs).loss
        log_cuda_memory(accelerator, stage="after_forward")
        accelerator.backward(loss)
    except torch.cuda.OutOfMemoryError as error:
        raise torch.cuda.OutOfMemoryError(
            "longest-sequence preflight failed for "
            f"{example_id} ({sequence_tokens} tokens): {error}"
        ) from error
    optimizer.zero_grad(set_to_none=True)
    del loss
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()
        log_cuda_memory(accelerator, stage="after_preflight_cleanup")


def clear_cuda_cache_for_long_sequences(
    accelerator: Accelerator,
    *,
    local_sequence_tokens: int,
    minimum_sequence_tokens: int,
) -> bool:
    long_sequence = torch.tensor(
        [int(local_sequence_tokens >= minimum_sequence_tokens)],
        dtype=torch.int64,
        device=accelerator.device,
    )
    any_long_sequence = bool(
        accelerator.reduce(long_sequence, reduction="sum").item()
    )
    if any_long_sequence and accelerator.device.type == "cuda":
        torch.cuda.empty_cache()
    return any_long_sequence


def log_cuda_memory(accelerator: Accelerator, *, stage: str) -> None:
    if accelerator.device.type != "cuda":
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info(accelerator.device)
    memory_stats = torch.cuda.memory_stats(accelerator.device)
    gib = 1024**3
    allocated_bytes = torch.cuda.memory_allocated(accelerator.device)
    reserved_bytes = torch.cuda.memory_reserved(accelerator.device)
    inactive_split_bytes = memory_stats["inactive_split_bytes.all.current"]
    print(
        f"preflight_cuda_memory stage={stage} rank={accelerator.process_index} "
        f"free_gib={free_bytes / gib:.2f} total_gib={total_bytes / gib:.2f} "
        f"allocated_gib={allocated_bytes / gib:.2f} "
        f"reserved_gib={reserved_bytes / gib:.2f} "
        f"reserved_unused_gib={(reserved_bytes - allocated_bytes) / gib:.2f} "
        f"inactive_split_gib={inactive_split_bytes / gib:.2f}",
        flush=True,
    )


def log_fsdp_layout(accelerator: Accelerator, model: Any) -> None:
    if accelerator.distributed_type != DistributedType.FSDP:
        return
    wrappers = [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if isinstance(module, FullyShardedDataParallel)
    ]
    if not wrappers:
        raise RuntimeError("FSDP training model contains no FSDP wrappers")
    world_size = int(accelerator.num_processes)
    largest_name, largest_bytes = max(
        (
            name,
            sum(
                parameter.numel() * parameter.element_size() * world_size
                for parameter in module.parameters(recurse=False)
            ),
        )
        for name, module in wrappers
    )
    if accelerator.is_main_process:
        print(
            f"fsdp_layout wrappers={len(wrappers)} "
            f"largest_wrapper={largest_name} "
            "largest_estimated_unsharded_flat_parameter_gib="
            f"{largest_bytes / 1024**3:.2f}",
            flush=True,
        )


def validate_distributed_device_map(
    accelerator: Accelerator,
    device_map: str,
) -> None:
    if accelerator.num_processes > 1 and device_map != "none":
        raise ValueError("--device-map must be none when WORLD_SIZE > 1")


def distributed_batch_metrics(
    accelerator: Accelerator,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> SFTBatchMetrics:
    world_size = int(accelerator.num_processes)
    if world_size > 1 and per_device_batch_size != 1:
        raise ValueError(
            "distributed LLM finetuning currently requires --per-device-batch-size 1"
        )
    return SFTBatchMetrics(
        world_size=world_size,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_batch_size=(
            world_size * per_device_batch_size * gradient_accumulation_steps
        ),
    )


def reduce_training_totals(
    accelerator: Accelerator,
    loss_sum: float,
    example_count: int,
    batch_count: int,
) -> Dict[str, float]:
    """Reduce unscaled real-example loss/support and synchronized batch count.

    ``loss_sum`` must exclude synthetic padding and must not include
    ``TrainingIndex.loss_scale``. ``example_count`` must count only real
    examples. ``batch_count`` is the local device-batch count and must match on
    every rank.
    """
    local_totals = torch.tensor(
        [loss_sum, example_count, batch_count, batch_count**2],
        dtype=torch.float64,
        device=accelerator.device,
    )
    global_totals = accelerator.reduce(local_totals, reduction="sum")
    (
        global_loss_sum,
        global_example_count,
        global_device_batch_count,
        global_device_batch_count_squared,
    ) = (float(value.item()) for value in global_totals)
    world_size = int(accelerator.num_processes)
    if global_device_batch_count_squared * world_size != global_device_batch_count**2:
        raise ValueError("all ranks must report equal local batch counts")
    synchronized_batch_count = global_device_batch_count / world_size
    return {
        "loss": (
            global_loss_sum / global_example_count if global_example_count else 0.0
        ),
        "loss_sum": global_loss_sum,
        "example_count": global_example_count,
        "batch_count": synchronized_batch_count,
    }


def scale_training_loss(
    loss: torch.Tensor,
    training_weights: torch.Tensor,
    is_padding: torch.Tensor,
) -> torch.Tensor:
    weights = training_weights.reshape(-1)
    padding = is_padding.reshape(-1).bool()
    if weights.numel() == 0 or weights.shape != padding.shape:
        raise ValueError("training weights and padding flags must be non-empty peers")
    if not torch.equal(weights == 0, padding):
        raise ValueError("only synthetic padding may have zero training weight")
    if not torch.all(weights == weights[0]):
        raise ValueError("all examples in a training batch must use one loss scale")
    return loss * weights[0].to(device=loss.device, dtype=loss.dtype)


class LengthGroupedBatchSampler(Sampler[List[TrainingIndex]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if world_size < 1:
            raise ValueError("world_size must be at least 1")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        if world_size > 1 and batch_size != 1:
            raise ValueError(
                "distributed LLM finetuning currently requires "
                "--per-device-batch-size 1"
            )
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.sorted_indices = sorted(
            range(len(lengths)), key=lambda index: lengths[index]
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[TrainingIndex]]:
        global_batch_size = self.batch_size * self.world_size
        groups = [
            self.sorted_indices[start : start + global_batch_size]
            for start in range(0, len(self.sorted_indices), global_batch_size)
        ]
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(groups), generator=generator).tolist()
        for group_index in order:
            group = groups[group_index]
            rank_batches = [
                group[start : start + self.batch_size]
                for start in range(0, len(group), self.batch_size)
            ]
            real_rank_count = len(rank_batches)
            if self.rank < real_rank_count:
                loss_scale = self.world_size / real_rank_count
                yield [
                    TrainingIndex(index=index, loss_scale=loss_scale)
                    for index in rank_batches[self.rank]
                ]
            else:
                yield [
                    TrainingIndex(
                        index=group[0],
                        loss_scale=0.0,
                        is_padding=True,
                    )
                ]

    def __len__(self) -> int:
        batch_count = (
            len(self.sorted_indices) + self.batch_size - 1
        ) // self.batch_size
        return (batch_count + self.world_size - 1) // self.world_size


def enable_gradient_checkpointing(model: Any) -> None:
    gradient_checkpointing_enable = getattr(
        model,
        "gradient_checkpointing_enable",
        None,
    )
    if gradient_checkpointing_enable is None:
        raise AttributeError(
            "Model does not support gradient checkpointing; rerun without "
            "--gradient-checkpointing"
        )
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    enable_input_require_grads = getattr(
        model,
        "enable_input_require_grads",
        None,
    )
    if enable_input_require_grads is not None:
        enable_input_require_grads()
    else:
        get_input_embeddings = getattr(model, "get_input_embeddings", None)
        if get_input_embeddings is None:
            raise AttributeError(
                "Model does not expose enable_input_require_grads or "
                "get_input_embeddings; rerun without --gradient-checkpointing"
            )
        input_embeddings = get_input_embeddings()
        if input_embeddings is None:
            raise AttributeError(
                "Model returned no input embeddings; rerun without "
                "--gradient-checkpointing"
            )

        def make_inputs_require_grad(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            output.requires_grad_(True)

        input_embeddings.register_forward_hook(make_inputs_require_grad)
    gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
