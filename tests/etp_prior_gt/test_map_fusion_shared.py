import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_shared_map_fusion():
    spec = importlib.util.spec_from_file_location(
        "shared_map_fusion_under_test",
        ROOT / "vlnce_baselines/models/etp_prior_gt/map_fusion.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_bidirectional_map_token_fusion_masks_stop_token():
    map_fusion = _load_shared_map_fusion()
    fusion = map_fusion.BidirectionalMapTokenFusion(
        hidden_size=16, num_heads=4, dropout=0.0
    )
    gmap_embeds = torch.randn(1, 4, 16)
    map_tokens = torch.randn(1, 3, 16)
    map_token_masks = torch.ones(1, 3, dtype=torch.bool)
    gmap_masks = torch.tensor([[True, True, True, False]])

    graph_key_padding_mask = fusion._graph_key_padding_mask(gmap_embeds, gmap_masks)

    assert torch.equal(
        graph_key_padding_mask,
        torch.tensor([[True, False, False, True]]),
    )
