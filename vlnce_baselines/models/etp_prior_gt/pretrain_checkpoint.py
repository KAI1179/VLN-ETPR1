"""Explicit mapping from pretraining checkpoints into navigation modules."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional

import torch

from vlnce_baselines.models.etp_imagined.checkpoint import (
    load_complete_state_dict,
)


FUSION_SOURCE_PREFIX = "bert.global_encoder.graph_map_attention."
MAP_ENCODER_SOURCE_PREFIX = "map_encoder."
_PRETRAIN_HEAD_PREFIXES = (
    "graph_query_text.",
    "graph_attentioned_txt_embeds_transform.",
    "global_sap_head.",
)


def load_pretraining_checkpoint(path: str) -> Mapping[str, torch.Tensor]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping: {checkpoint_path}")
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise TypeError(
            f"Checkpoint state_dict must be a mapping: {checkpoint_path}"
        )

    normalized: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for raw_key, value in raw_state.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Checkpoint state_dict must map strings to tensors: {checkpoint_path}"
            )
        key = raw_key[len("module.") :] if raw_key.startswith("module.") else raw_key
        if key in normalized:
            raise ValueError(
                f"Checkpoint key normalization collision for {key}: {checkpoint_path}"
            )
        normalized[key] = value
    return normalized


def map_pretraining_state_to_vlnbert(
    state_dict: Mapping[str, torch.Tensor],
) -> "OrderedDict[str, torch.Tensor]":
    """Map pretraining names to the runtime VLN-BERT names explicitly."""

    mapped: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in state_dict.items():
        target: Optional[str]
        if key.startswith(FUSION_SOURCE_PREFIX):
            suffix = key[len(FUSION_SOURCE_PREFIX) :]
            target = f"bert.graph_map_attention.{suffix}"
        elif key.startswith("bert."):
            target = key
        elif key.startswith(_PRETRAIN_HEAD_PREFIXES):
            target = f"bert.{key}"
        else:
            target = None
        if target is None:
            continue
        if target in mapped:
            raise ValueError(f"Pretraining checkpoint mapping collision for {target}")
        mapped[target] = value
    return mapped


def load_checkpoint_submodule(
    module: torch.nn.Module,
    checkpoint_state: Optional[Mapping[str, torch.Tensor]],
    *,
    checkpoint_path: Path,
    source_prefix: str,
    module_name: str,
    required: bool,
) -> bool:
    """Load one complete submodule or fail on absent/partial/incompatible state."""

    state = OrderedDict()
    if checkpoint_state is not None:
        for key, value in checkpoint_state.items():
            if key.startswith(source_prefix):
                state[key[len(source_prefix) :]] = value

    if not state:
        if required:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no complete {module_name} state "
                f"under {source_prefix}"
            )
        return False

    load_complete_state_dict(
        module,
        state,
        checkpoint_path,
        module_name,
    )
    return True


def validate_pretraining_initialization(
    model: torch.nn.Module,
    checkpoint_state: Mapping[str, torch.Tensor],
    checkpoint_path: Path,
    *,
    allowed_missing_prefixes=(),
) -> None:
    """Fail closed except for explicitly new initialization-only subtrees."""

    expected = model.state_dict()
    expected_keys = set(expected)
    actual_keys = set(checkpoint_state)
    missing = sorted(
        key
        for key in expected_keys - actual_keys
        if not key.startswith(tuple(allowed_missing_prefixes))
    )
    unexpected = sorted(actual_keys - expected_keys)
    shape_mismatches = sorted(
        key
        for key in expected_keys & actual_keys
        if expected[key].shape != checkpoint_state[key].shape
    )
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            f"Incompatible pretraining initialization checkpoint {checkpoint_path}; "
            f"missing keys: {missing}; unexpected keys: {unexpected}; "
            f"shape mismatches: {shape_mismatches}"
        )
