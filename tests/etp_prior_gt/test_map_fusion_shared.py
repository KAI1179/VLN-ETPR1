import torch

from vlnce_baselines.models.etp_prior_gt.map_fusion import (
    BidirectionalMapTokenFusion,
    GraphMapCrossAttention,
    build_map_token_fusion,
)


def test_shared_bidirectional_map_token_fusion_masks_stop_token():
    fusion = BidirectionalMapTokenFusion(hidden_size=16, num_heads=4, dropout=0.0)
    gmap_embeds = torch.randn(1, 4, 16)
    gmap_masks = torch.tensor([[True, True, True, False]])

    graph_key_padding_mask = fusion._graph_key_padding_mask(gmap_embeds, gmap_masks)

    assert torch.equal(
        graph_key_padding_mask,
        torch.tensor([[True, False, False, True]]),
    )


def test_map_token_fusion_factory_selects_current_default():
    fusion = build_map_token_fusion(
        "current",
        hidden_size=16,
        num_heads=4,
        dropout=0.0,
    )

    assert isinstance(fusion, BidirectionalMapTokenFusion)


def test_map_token_fusion_factory_selects_try5_one_way_module():
    fusion = build_map_token_fusion(
        "try5",
        hidden_size=16,
        num_heads=4,
        dropout=0.0,
    )

    assert isinstance(fusion, GraphMapCrossAttention)
    assert hasattr(fusion, "attention")
    assert hasattr(fusion, "residual_projection")


def test_try5_fusion_does_not_update_map_tokens():
    fusion = build_map_token_fusion(
        "try5",
        hidden_size=16,
        num_heads=4,
        dropout=0.0,
    )

    _, updated_map_tokens = fusion(
        torch.randn(1, 4, 16),
        torch.ones(1, 4, dtype=torch.bool),
        torch.randn(1, 8, 16),
        torch.ones(1, 8, dtype=torch.bool),
    )

    assert updated_map_tokens is None
