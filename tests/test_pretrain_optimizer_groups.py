from types import SimpleNamespace

import pytest
from torch import nn

from pretrain_src.pretrain_src.optim.misc import build_optimizer


class _RouteMapUpdater(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_head = nn.Linear(2, 2)
        self.route_gate = nn.Linear(2, 1)
        self.refiner = nn.Linear(2, 2)


class _TinyModel(nn.Module):
    def __init__(self, with_updater=True):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        if with_updater:
            self.route_map_updater = _RouteMapUpdater()


class _DistributedNameWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


def _opts():
    return SimpleNamespace(
        pose_gated_map=True,
        weight_decay=0.01,
        learning_rate=5e-5,
        optim="adamw",
        betas=[0.9, 0.98],
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_pose_gated_optimizer_assigns_lr_tiers_with_distributed_prefix(wrapped):
    model = _TinyModel()
    if wrapped:
        model = _DistributedNameWrapper(model)
    optimizer = build_optimizer(model, _opts())

    parameter_scale = {
        id(parameter): group["lr_scale"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for name, parameter in model.named_parameters():
        canonical_name = name[len("module.") :] if name.startswith("module.") else name
        expected = 1.0 if canonical_name.startswith("route_map_updater.") else 0.1
        assert parameter_scale[id(parameter)] == expected


def test_pose_gated_optimizer_rejects_missing_updater():
    with pytest.raises(RuntimeError, match="no trainable route_map_updater"):
        build_optimizer(_DistributedNameWrapper(_TinyModel(False)), _opts())
