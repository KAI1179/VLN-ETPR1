"""Shared cognitive-map and graph-token fusion modules."""

import torch
import torch.nn as nn


class GraphMapCrossAttention(nn.Module):
    """Try5 one-way map-token fusion: graph nodes attend to fixed map tokens."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.residual_projection = nn.Linear(hidden_size, hidden_size)
        self._zero_residual_projection()

    def _zero_residual_projection(self):
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def _map_key_padding_mask(self, map_token_masks):
        if map_token_masks is None:
            return None
        if map_token_masks.any(dim=1).logical_not().any():
            map_token_masks = map_token_masks.clone()
            map_token_masks[map_token_masks.any(dim=1).logical_not(), 0] = True
        return map_token_masks.logical_not()

    def forward(self, gmap_embeds, gmap_masks, map_tokens, map_token_masks):
        if map_tokens is None:
            return gmap_embeds, None

        map_context, _ = self.attention(
            gmap_embeds,
            map_tokens,
            map_tokens,
            key_padding_mask=self._map_key_padding_mask(map_token_masks),
            need_weights=False,
        )
        return gmap_embeds + self.residual_projection(map_context), None


class BidirectionalMapTokenFusion(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.map_from_graph_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.graph_from_map_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.map_residual_projection = nn.Linear(hidden_size, hidden_size)
        self.graph_residual_projection = nn.Linear(hidden_size, hidden_size)
        self._zero_residual_projection()

    def _zero_residual_projection(self):
        nn.init.zeros_(self.map_residual_projection.weight)
        nn.init.zeros_(self.map_residual_projection.bias)
        nn.init.zeros_(self.graph_residual_projection.weight)
        nn.init.zeros_(self.graph_residual_projection.bias)

    def _map_key_padding_mask(self, map_token_masks):
        if map_token_masks is None:
            return None
        if map_token_masks.any(dim=1).logical_not().any():
            # MultiheadAttention returns NaNs for rows with all keys masked.
            map_token_masks = map_token_masks.clone()
            map_token_masks[map_token_masks.any(dim=1).logical_not(), 0] = True
        return map_token_masks.logical_not()

    def _graph_key_padding_mask(self, gmap_embeds, gmap_masks):
        if gmap_masks is None:
            graph_key_padding_mask = torch.zeros(
                gmap_embeds.shape[:2],
                dtype=torch.bool,
                device=gmap_embeds.device,
            )
        else:
            graph_key_padding_mask = gmap_masks.logical_not().clone()
        if graph_key_padding_mask.size(1) > 0:
            graph_key_padding_mask[:, 0] = True
        fully_masked = graph_key_padding_mask.all(dim=1)
        if fully_masked.any():
            # Numeric fallback for degenerate rows; normal rollouts have a real graph node.
            graph_key_padding_mask[fully_masked, 0] = False
        return graph_key_padding_mask

    def forward(self, gmap_embeds, gmap_masks, map_tokens, map_token_masks):
        if map_tokens is None:
            return gmap_embeds, None

        graph_key_padding_mask = self._graph_key_padding_mask(gmap_embeds, gmap_masks)
        map_context, _ = self.map_from_graph_attention(
            map_tokens,
            gmap_embeds,
            gmap_embeds,
            key_padding_mask=graph_key_padding_mask,
            need_weights=False,
        )
        updated_map_tokens = map_tokens + self.map_residual_projection(map_context)

        map_key_padding_mask = self._map_key_padding_mask(map_token_masks)
        graph_context, _ = self.graph_from_map_attention(
            gmap_embeds,
            updated_map_tokens,
            updated_map_tokens,
            key_padding_mask=map_key_padding_mask,
            need_weights=False,
        )
        updated_gmap_embeds = gmap_embeds + self.graph_residual_projection(
            graph_context
        )
        return updated_gmap_embeds, updated_map_tokens


def build_map_token_fusion(
    architecture: str,
    hidden_size: int,
    num_heads: int,
    dropout: float = 0.1,
) -> nn.Module:
    if architecture == "current":
        return BidirectionalMapTokenFusion(hidden_size, num_heads, dropout)
    if architecture == "try5":
        return GraphMapCrossAttention(hidden_size, num_heads, dropout)
    raise ValueError(f"Unknown navigation architecture: {architecture}")
