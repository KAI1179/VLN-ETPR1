"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""

from torch.optim import Adam, Adamax

from .adamw import AdamW
from .rangerlars import RangerLars
from vlnce_baselines.models.optimizer_profiles import build_lr_parameter_groups


def build_optimizer(model, opts, lr_scales=None):
    param_optimizer = list(model.named_parameters())
    if lr_scales is None:
        lr_scales = {
            name: 1.0 for name, param in param_optimizer if param.requires_grad
        }
    optimizer_grouped_parameters = build_lr_parameter_groups(
        param_optimizer,
        lr_scales,
        learning_rate=opts.learning_rate,
        weight_decay=opts.weight_decay,
    )

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
    return optimizer
