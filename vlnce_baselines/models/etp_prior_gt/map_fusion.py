"""Shared cognitive-map and graph-token fusion modules."""

import torch
import torch.nn as nn

# Geometry of the map encoder's spatial tokens (EmbeddingGridMapEncoder): a
# 100x100 raster of 0.5 m cells cut by a stride-10 convolution into 10x10
# tokens, flattened row-major, followed by one metadata token.
MAP_TOKEN_GRID = 10
MAP_CELLS_PER_TOKEN = 10
MAP_SPATIAL_TOKENS = MAP_TOKEN_GRID * MAP_TOKEN_GRID
# Per-head locality scales (in tokens, 1 token = 5 m) for coordinate fusion.
# A third of the heads stay purely content-based; the rest are pulled towards
# the map tokens around each graph node at 2.5 m to 20 m scales.
LOCALITY_SIGMAS_TOKENS = (0.5, 1.0, 2.0, 4.0)


def map_token_centers() -> torch.Tensor:
    """(100, 2) centres of the spatial tokens in raster-cell coordinates."""
    index = torch.arange(MAP_TOKEN_GRID, dtype=torch.float32)
    centers = index * MAP_CELLS_PER_TOKEN + MAP_CELLS_PER_TOKEN / 2.0
    rows, cols = torch.meshgrid(centers, centers, indexing="ij")
    return torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=-1)


def locality_coefficients(num_heads: int) -> torch.Tensor:
    """(num_heads,) coefficients c_h of the attention bias -c_h * d^2."""
    content_heads = num_heads // 3
    coefficients = [0.0] * content_heads
    for index in range(num_heads - content_heads):
        sigma = LOCALITY_SIGMAS_TOKENS[index % len(LOCALITY_SIGMAS_TOKENS)]
        coefficients.append(1.0 / (2.0 * sigma * sigma))
    return torch.tensor(sorted(coefficients), dtype=torch.float32)


def _fp32(tensor: torch.Tensor):
    """Disable autocast on the tensor's device.

    Pretraining ran the map fusion in fp32; DAgger rollouts run under fp16
    autocast.  Keeping the trained fusion in fp32 stops its activations from
    overflowing the fp16 range (the first symptom is NaN action probabilities).
    """
    return torch.autocast(device_type=tensor.device.type, enabled=False)


class GraphMapCrossAttention(nn.Module):
    """Try5 one-way map-token fusion: graph nodes attend to fixed map tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        coordinate_fusion: bool = False,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.residual_projection = nn.Linear(hidden_size, hidden_size)
        self._zero_residual_projection()
        # Coordinate fusion adds a fixed distance bias between each graph node's
        # map-frame position and each spatial token's centre to the attention
        # logits.  Non-persistent buffers: the state_dict (and so every
        # checkpoint) is identical with the flag on or off.
        self.num_heads = num_heads
        self.coordinate_fusion = coordinate_fusion
        if coordinate_fusion:
            self.register_buffer("token_centers", map_token_centers(), persistent=False)
            self.register_buffer(
                "locality_coef", locality_coefficients(num_heads), persistent=False
            )

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

    def coordinate_bias(self, node_coords, num_map_tokens):
        """Additive attention bias (B * num_heads, G, num_map_tokens).

        node_coords: (B, G, 2) map-frame (row, col) in raster cells; rows that
        are not finite (STOP slot, padding) get no bias, and neither does the
        metadata token.
        """
        batch_size, num_nodes, _ = node_coords.shape
        valid = torch.isfinite(node_coords).all(dim=-1)
        coords = torch.where(
            valid.unsqueeze(-1), node_coords, torch.zeros_like(node_coords)
        )
        delta = (coords.unsqueeze(2) - self.token_centers) / MAP_CELLS_PER_TOKEN
        squared_distance = delta.pow(2).sum(dim=-1)  # (B, G, 100) in tokens^2
        bias = -self.locality_coef.view(1, -1, 1, 1) * squared_distance.unsqueeze(1)
        bias = bias * valid.view(batch_size, 1, num_nodes, 1)
        extra = bias.new_zeros(
            batch_size, self.num_heads, num_nodes, num_map_tokens - MAP_SPATIAL_TOKENS
        )
        bias = torch.cat([bias, extra], dim=-1)
        return bias.reshape(batch_size * self.num_heads, num_nodes, num_map_tokens)

    def forward(
        self, gmap_embeds, gmap_masks, map_tokens, map_token_masks, node_coords=None
    ):
        if map_tokens is None:
            return gmap_embeds, None
        if self.coordinate_fusion and node_coords is None:
            raise ValueError(
                "MAP_ENCODER.coordinate_fusion is on but no gmap_map_coords were "
                "passed; this trainer does not provide node map coordinates"
            )

        with _fp32(gmap_embeds):
            gmap_embeds = gmap_embeds.float()
            map_tokens = map_tokens.float()
            attn_mask = None
            if self.coordinate_fusion:
                attn_mask = self.coordinate_bias(
                    node_coords.float(), map_tokens.size(1)
                )
            map_context, _ = self.attention(
                gmap_embeds,
                map_tokens,
                map_tokens,
                key_padding_mask=self._map_key_padding_mask(map_token_masks),
                attn_mask=attn_mask,
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

        with _fp32(gmap_embeds):
            gmap_embeds = gmap_embeds.float()
            map_tokens = map_tokens.float()
            graph_key_padding_mask = self._graph_key_padding_mask(
                gmap_embeds, gmap_masks
            )
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
    coordinate_fusion: bool = False,
) -> nn.Module:
    if architecture == "current":
        if coordinate_fusion:
            raise ValueError("coordinate_fusion is only implemented for try5")
        return BidirectionalMapTokenFusion(hidden_size, num_heads, dropout)
    if architecture == "try5":
        return GraphMapCrossAttention(
            hidden_size, num_heads, dropout, coordinate_fusion=coordinate_fusion
        )
    raise ValueError(f"Unknown navigation architecture: {architecture}")
