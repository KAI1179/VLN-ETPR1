"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""

from torch.optim import Adam, Adamax

from .adamw import AdamW
from .rangerlars import RangerLars


def _canonical_parameter_name(name):
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def build_optimizer(model, opts):
    param_optimizer = [
        (_canonical_parameter_name(name), parameter)
        for name, parameter in model.named_parameters()
    ]
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    trainable = [(n, p) for n, p in param_optimizer if p.requires_grad]
    if getattr(opts, "pose_gated_map", False):
        new_module = "route_map_updater"
        new_prefix = f"{new_module}."
        new_trainable = [
            (n, p)
            for n, p in trainable
            if n == new_module or n.startswith(new_prefix)
        ]
        existing_trainable = [
            (n, p)
            for n, p in trainable
            if n != new_module and not n.startswith(new_prefix)
        ]
        if not new_trainable:
            raise RuntimeError(
                "pose-gated optimizer found no trainable route_map_updater parameters"
            )
        if not existing_trainable:
            raise RuntimeError(
                "pose-gated optimizer found no trainable existing-model parameters"
            )
        missing_parts = [
            part
            for part in ("visual_head", "route_gate", "refiner")
            if not any(n.startswith(f"{new_prefix}{part}.") for n, _ in new_trainable)
        ]
        if missing_parts:
            raise RuntimeError(
                "pose-gated optimizer is missing trainable updater parts: "
                + ", ".join(missing_parts)
            )
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in new_trainable
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": opts.weight_decay,
                "lr": opts.learning_rate,
            },
            {
                "params": [
                    p for n, p in new_trainable
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": opts.learning_rate,
            },
            {
                "params": [
                    p for n, p in existing_trainable
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": opts.weight_decay,
                "lr": opts.learning_rate * 0.1,
            },
            {
                "params": [
                    p for n, p in existing_trainable
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": opts.learning_rate * 0.1,
            },
        ]
        grouped_ids = [
            id(parameter)
            for group in optimizer_grouped_parameters
            for parameter in group["params"]
        ]
        trainable_ids = {id(parameter) for _, parameter in trainable}
        if (
            len(grouped_ids) != len(set(grouped_ids))
            or set(grouped_ids) != trainable_ids
        ):
            raise RuntimeError(
                "pose-gated optimizer parameter groups must cover each trainable "
                "parameter exactly once"
            )
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in trainable
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": opts.weight_decay,
            },
            {
                "params": [
                    p for n, p in trainable if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    # currently Adam only
    if opts.optim == "adam":
        OptimCls = Adam
    elif opts.optim == "adamax":
        OptimCls = Adamax
    elif opts.optim == "adamw":
        OptimCls = AdamW
    elif opts.optim == "rangerlars":
        OptimCls = RangerLars
    else:
        raise ValueError("invalid optimizer")
    optimizer = OptimCls(
        optimizer_grouped_parameters, lr=opts.learning_rate, betas=opts.betas
    )
    for group in optimizer.param_groups:
        group["lr_scale"] = group["lr"] / opts.learning_rate
    if getattr(opts, "pose_gated_map", False):
        new_parameter_ids = {id(parameter) for _, parameter in new_trainable}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                expected_scale = 1.0 if id(parameter) in new_parameter_ids else 0.1
                if abs(group["lr_scale"] - expected_scale) > 1e-12:
                    raise RuntimeError(
                        "pose-gated optimizer assigned an incorrect learning-rate scale"
                    )
    return optimizer
