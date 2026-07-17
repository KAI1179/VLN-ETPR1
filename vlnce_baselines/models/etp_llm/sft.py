"""Shared supervised fine-tuning helpers for ETP LLM candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import torch
from torch.utils.data import BatchSampler

ItemT = TypeVar("ItemT")


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


class LengthGroupedBatchSampler(BatchSampler):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(range(len(lengths)), batch_size=batch_size, drop_last=False)
        self.generator = generator
        self.sorted_indices = sorted(
            range(len(lengths)), key=lambda index: lengths[index]
        )

    def __iter__(self) -> Iterator[List[int]]:
        batches = [
            self.sorted_indices[start : start + self.batch_size]
            for start in range(0, len(self.sorted_indices), self.batch_size)
        ]
        full_batches = [batch for batch in batches if len(batch) == self.batch_size]
        tail_batches = [batch for batch in batches if len(batch) != self.batch_size]
        order = torch.randperm(
            len(full_batches),
            generator=self.generator,
        ).tolist()
        for batch_index in order:
            yield full_batches[batch_index]
        for batch in tail_batches:
            yield batch

    def __len__(self) -> int:
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


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
