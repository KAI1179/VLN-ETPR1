from collections import OrderedDict

import pytest
import torch

from scripts.smoke_feature_fusion_checkpoint import _normalize_checkpoint_state


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
