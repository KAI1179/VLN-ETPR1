"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""

from torch.optim import Adam, Adamax

from .adamw import AdamW
from .rangerlars import RangerLars


def build_optimizer(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    trainable = [(n, p) for n, p in param_optimizer if p.requires_grad]
    if getattr(opts, "pose_gated_map", False):
        new_module = "route_map_updater"
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in trainable
                    if n.startswith(new_module) and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": opts.weight_decay,
                "lr": opts.learning_rate,
            },
            {
                "params": [
                    p for n, p in trainable
                    if n.startswith(new_module) and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": opts.learning_rate,
            },
            {
                "params": [
                    p for n, p in trainable
                    if not n.startswith(new_module) and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": opts.weight_decay,
                "lr": opts.learning_rate * 0.1,
            },
            {
                "params": [
                    p for n, p in trainable
                    if not n.startswith(new_module) and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": opts.learning_rate * 0.1,
            },
        ]
    else:
        optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in trainable if not any(nd in n for nd in no_decay)
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
    return optimizer
