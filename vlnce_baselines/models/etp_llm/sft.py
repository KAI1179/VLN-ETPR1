"""Shared supervised fine-tuning helpers for ETP LLM candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
)

import torch
from accelerate import Accelerator
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


def rendered_token_counts(
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
) -> RenderedTokenCounts:
    prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    sequence_tokens = len(
        tokenizer.encode(completion_text, add_special_tokens=False)
    )
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
    dataset_by_id = {
        str(item["example_id"]): str(item["dataset"]) for item in items
    }
    kept_ids = {str(item["example_id"]) for item in filtered.kept}
    prompt_dropped_ids = set(filtered.dropped_prompt_example_ids)
    completion_dropped_ids = set(filtered.dropped_completion_example_ids)
    metrics: Dict[str, float] = {}
    for dataset, prefix in (("R2R", "r2r"), ("RxR", "rxr")):
        stats = load_stats[dataset]
        metrics.update(
            {
                f"{prefix}_discovered": float(stats.discovered),
                f"{prefix}_loaded": float(stats.loaded),
                f"{prefix}_missing_cache": float(
                    len(stats.missing_cache_example_ids)
                ),
                f"{prefix}_prompt_dropped": float(
                    sum(dataset_by_id[item_id] == dataset for item_id in prompt_dropped_ids)
                ),
                f"{prefix}_completion_dropped": float(
                    sum(
                        dataset_by_id[item_id] == dataset
                        for item_id in completion_dropped_ids
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
        "retained",
    ):
        metrics[f"combined_{name}"] = metrics[f"r2r_{name}"] + metrics[
            f"rxr_{name}"
        ]
    return metrics


def make_sft_accelerator(gradient_accumulation_steps: int) -> Accelerator:
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
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
            "distributed LLM finetuning currently requires "
            "--per-device-batch-size 1"
        )
    return SFTBatchMetrics(
        world_size=world_size,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_batch_size=(
            world_size
            * per_device_batch_size
            * gradient_accumulation_steps
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
    if (
        global_device_batch_count_squared * world_size
        != global_device_batch_count**2
    ):
        raise ValueError("all ranks must report equal local batch counts")
    synchronized_batch_count = global_device_batch_count / world_size
    return {
        "loss": (
            global_loss_sum / global_example_count
            if global_example_count
            else 0.0
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
        batches = [
            self.sorted_indices[start : start + self.batch_size]
            for start in range(0, len(self.sorted_indices), self.batch_size)
        ]
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(batches), generator=generator).tolist()
        shuffled_batches = [batches[index] for index in order]
        for start in range(0, len(shuffled_batches), self.world_size):
            rank_group = shuffled_batches[start : start + self.world_size]
            real_rank_count = len(rank_group)
            if self.rank < real_rank_count:
                loss_scale = self.world_size / real_rank_count
                yield [
                    TrainingIndex(index=index, loss_scale=loss_scale)
                    for index in rank_group[self.rank]
                ]
            else:
                yield [
                    TrainingIndex(
                        index=self.sorted_indices[0],
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
    gradient_checkpointing_enable()
