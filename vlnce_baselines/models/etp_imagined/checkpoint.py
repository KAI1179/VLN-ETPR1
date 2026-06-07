"""Strict checkpoint loading for cognitive-map modules."""

from pathlib import Path
from typing import Mapping

import torch


def load_complete_state_dict(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    checkpoint_path: Path,
    module_name: str,
) -> None:
    expected_keys = set(module.state_dict())
    actual_keys = set(state_dict)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            f"Incompatible checkpoint {checkpoint_path} for {module_name}; "
            f"missing keys: {missing_keys}; unexpected keys: {unexpected_keys}"
        )
    try:
        module.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Incompatible checkpoint {checkpoint_path} for {module_name}: {exc}"
        ) from exc
