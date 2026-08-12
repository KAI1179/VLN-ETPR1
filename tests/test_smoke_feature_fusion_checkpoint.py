from collections import OrderedDict

import pytest
import torch

from scripts.smoke_feature_fusion_checkpoint import (
    Arguments,
    _normalize_checkpoint_state,
)


def test_normalizes_only_known_ddp_prefix():
    tensor = torch.zeros(1)
    state = OrderedDict([
        ("net.module.map_encoder.weight", tensor),
        ("module.unrelated", tensor),
    ])

    normalized = _normalize_checkpoint_state(state)

    assert list(normalized) == ["net.map_encoder.weight", "module.unrelated"]


def test_rejects_normalization_collision():
    tensor = torch.zeros(1)
    state = OrderedDict([
        ("net.module.map_encoder.weight", tensor),
        ("net.map_encoder.weight", tensor),
    ])

    with pytest.raises(ValueError, match="normalization collision"):
        _normalize_checkpoint_state(state)


def test_cli_uses_dashed_options():
    args = Arguments(underscores_to_dashes=True).parse_args([
        "--checkpoint",
        "model.pth",
        "--sha256",
        "0" * 64,
        "--size",
        "1",
        "--iteration",
        "2",
        "--policy",
        "PriorGTTry5Policy",
        "--source",
        "prior_gt",
        "--scene",
        "scene",
        "--episode-id",
        "3",
    ])

    assert args.episode_id == "3"
