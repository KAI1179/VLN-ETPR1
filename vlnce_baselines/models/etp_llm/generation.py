"""Shared causal-LM generation helpers."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple


def generate_with_oom_splitting(
    model: Any,
    model_inputs: Dict[str, Any],
    generation_kwargs: Dict[str, Any],
) -> Tuple[List[Any], int]:
    """Generate a batch, recursively splitting CUDA OOM batches in order."""
    import torch

    batch_size = len(model_inputs["input_ids"])
    try:
        return list(model.generate(**model_inputs, **generation_kwargs)), 0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if batch_size <= 1:
            raise
        midpoint = batch_size // 2
        warnings.warn(
            f"CUDA OOM during generation; splitting batch of {batch_size} into "
            f"{midpoint} and {batch_size - midpoint}",
            RuntimeWarning,
            stacklevel=2,
        )
        left, left_retries = generate_with_oom_splitting(
            model,
            _slice_model_inputs(model_inputs, 0, midpoint),
            generation_kwargs,
        )
        right, right_retries = generate_with_oom_splitting(
            model,
            _slice_model_inputs(model_inputs, midpoint, batch_size),
            generation_kwargs,
        )
        return left + right, 1 + left_retries + right_retries


def _slice_model_inputs(
    model_inputs: Dict[str, Any],
    start: int,
    end: int,
) -> Dict[str, Any]:
    return {key: value[start:end] for key, value in model_inputs.items()}
