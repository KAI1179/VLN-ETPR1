"""Explicit trainable-module and learning-rate contracts for navigation stages."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch


FULL_OPTIMIZER_PROFILE = "full"
ONLINE_FUSION_OPTIMIZER_PROFILE = "online_fusion"
OPTIMIZER_PROFILES = (
    FULL_OPTIMIZER_PROFILE,
    ONLINE_FUSION_OPTIMIZER_PROFILE,
)

_PRETRAIN_ONE_X_PREFIXES = (
    "map_encoder.",
    "bert.global_encoder.graph_map_attention.",
)
_PRETRAIN_POINT_ONE_X_PREFIXES = (
    "bert.global_encoder.",
    "graph_query_text.",
    "graph_attentioned_txt_embeds_transform.",
    "global_sap_head.",
    "mlm_head.",
)
_DAGGER_ONE_X_PARTS = (
    ".vln_bert.graph_map_attention.",
    ".online_fusion.",
)
_DAGGER_POINT_ONE_X_PARTS = (
    ".map_encoder.",
    ".vln_bert.global_encoder.",
    ".vln_bert.graph_query_text.",
    ".vln_bert.graph_attentioned_txt_embeds_transform.",
    ".vln_bert.global_sap_head.",
)
_NO_DECAY_PARTS = ("bias", "LayerNorm.bias", "LayerNorm.weight")


def configure_pretrain_optimizer_profile(
    model: torch.nn.Module,
    profile: str,
) -> Dict[str, float]:
    """Set pretraining ``requires_grad`` and return the LR scale per tensor."""

    return _configure_profile(
        model,
        profile,
        one_x_match=lambda name: _without_distributed_prefix(name).startswith(
            _PRETRAIN_ONE_X_PREFIXES
        ),
        point_one_x_match=lambda name: _without_distributed_prefix(name).startswith(
            _PRETRAIN_POINT_ONE_X_PREFIXES
        ),
    )


def configure_dagger_optimizer_profile(
    policy: torch.nn.Module,
    profile: str,
) -> Dict[str, float]:
    """Set DAgger ``requires_grad`` and return the LR scale per tensor."""

    return _configure_profile(
        policy,
        profile,
        one_x_match=lambda name: any(part in f".{name}" for part in _DAGGER_ONE_X_PARTS),
        point_one_x_match=lambda name: any(
            part in f".{name}" for part in _DAGGER_POINT_ONE_X_PARTS
        ),
    )


def _configure_profile(model, profile, one_x_match, point_one_x_match):
    if profile not in OPTIMIZER_PROFILES:
        raise ValueError(
            f"Unknown optimizer profile {profile!r}; expected one of "
            f"{OPTIMIZER_PROFILES}"
        )

    named_parameters = list(model.named_parameters())
    if profile == FULL_OPTIMIZER_PROFILE:
        scales = {
            name: 1.0 for name, param in named_parameters if param.requires_grad
        }
    else:
        scales = {}
        for name, param in named_parameters:
            if one_x_match(name):
                scale = 1.0
            elif point_one_x_match(name):
                scale = 0.1
            else:
                scale = None
            param.requires_grad_(scale is not None)
            if scale is not None:
                scales[name] = scale

    if not scales:
        raise RuntimeError(f"Optimizer profile {profile!r} selected no parameters")
    return scales


def _without_distributed_prefix(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def build_lr_parameter_groups(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    lr_scales: Dict[str, float],
    *,
    learning_rate: float,
    weight_decay: float,
) -> List[dict]:
    """Build decay/non-decay groups without losing the profile's LR tiers."""

    parameters = list(named_parameters)
    trainable_names = {name for name, param in parameters if param.requires_grad}
    if trainable_names != set(lr_scales):
        missing = sorted(trainable_names - set(lr_scales))
        extra = sorted(set(lr_scales) - trainable_names)
        raise RuntimeError(
            "Optimizer LR profile does not match requires_grad parameters; "
            f"missing_scales={missing}, extra_scales={extra}"
        )

    grouped = defaultdict(list)
    for name, param in parameters:
        if not param.requires_grad:
            continue
        scale = lr_scales[name]
        decay = not any(part in name for part in _NO_DECAY_PARTS)
        grouped[(scale, decay)].append((name, param))

    result = []
    seen = set()
    for (scale, decay), entries in sorted(grouped.items(), reverse=True):
        ids = [id(param) for _, param in entries]
        if seen.intersection(ids):
            raise RuntimeError("Optimizer profile selected duplicate parameters")
        seen.update(ids)
        result.append({
            "params": [param for _, param in entries],
            "lr": learning_rate * scale,
            "lr_scale": scale,
            "weight_decay": weight_decay if decay else 0.0,
            "profile_names": tuple(name for name, _ in entries),
        })

    expected = {id(param) for _, param in parameters if param.requires_grad}
    if seen != expected:
        raise RuntimeError("Optimizer groups do not cover every trainable parameter")
    return result


def optimizer_profile_summary(
    lr_scales: Dict[str, float],
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
) -> str:
    parameter_counts = {
        name: parameter.numel() for name, parameter in named_parameters
    }
    missing = sorted(set(lr_scales) - set(parameter_counts))
    if missing:
        raise RuntimeError(
            "Optimizer profile summary cannot find parameters: " f"{missing}"
        )
    grouped = defaultdict(list)
    for name, scale in lr_scales.items():
        grouped[scale].append(name)
    sections = []
    for scale in sorted(grouped, reverse=True):
        names = sorted(grouped[scale])
        total = sum(parameter_counts[name] for name in names)
        sections.append(
            f"lr_scale={scale:g}; tensors={len(names)}; parameters={total}:\n"
            + "\n".join(
                f"{name} ({parameter_counts[name]})" for name in names
            )
        )
    return "\n".join(sections)
