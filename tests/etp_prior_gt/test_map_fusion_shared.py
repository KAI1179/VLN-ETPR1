import torch

from vlnce_baselines.models.etp_prior_gt.map_fusion import BidirectionalMapTokenFusion


def test_shared_bidirectional_map_token_fusion_masks_stop_token():
    fusion = BidirectionalMapTokenFusion(hidden_size=16, num_heads=4, dropout=0.0)
    gmap_embeds = torch.randn(1, 4, 16)
    gmap_masks = torch.tensor([[True, True, True, False]])

    graph_key_padding_mask = fusion._graph_key_padding_mask(gmap_embeds, gmap_masks)

    assert torch.equal(
        graph_key_padding_mask,
        torch.tensor([[True, False, False, True]]),
    )
