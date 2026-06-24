"""Decode updated map tokens into dense cognitive-map logits."""

from __future__ import annotations

import torch
import torch.nn as nn

from .map_utils import (
    MAP_SPATIAL_TOKEN_COUNT,
    MAP_TOKEN_COUNT,
    MAP_TOKEN_GRID_SIZE,
    NUM_MAP_CATEGORIES,
    SIZE,
)


class CognitiveMapDecoder(nn.Module):
    """Decode the spatial updated map tokens into an updated cognitive map."""

    def __init__(self, hidden_size: int):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_size = hidden_size
        decoder_hidden_size = max(hidden_size // 2, NUM_MAP_CATEGORIES)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, decoder_hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(SIZE, SIZE), mode="bilinear", align_corners=False),
            nn.Conv2d(
                decoder_hidden_size,
                decoder_hidden_size,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(decoder_hidden_size, NUM_MAP_CATEGORIES, kernel_size=1),
        )

    def _validate_updated_map_tokens(self, updated_map_tokens: torch.Tensor) -> None:
        if updated_map_tokens.dim() != 3:
            raise ValueError(
                "updated_map_tokens must have shape "
                f"(B, {MAP_TOKEN_COUNT}, {self.hidden_size}), "
                f"got {tuple(updated_map_tokens.shape)}"
            )
        if updated_map_tokens.shape[1:] != (MAP_TOKEN_COUNT, self.hidden_size):
            raise ValueError(
                "updated_map_tokens must have shape "
                f"(B, {MAP_TOKEN_COUNT}, {self.hidden_size}), "
                f"got {tuple(updated_map_tokens.shape)}"
            )

    def forward(self, updated_map_tokens: torch.Tensor) -> torch.Tensor:
        """Return dense updated cognitive-map logits with shape (B, 37, 100, 100)."""
        self._validate_updated_map_tokens(updated_map_tokens)
        batch_size = updated_map_tokens.shape[0]
        spatial_tokens = updated_map_tokens[:, :MAP_SPATIAL_TOKEN_COUNT]
        feature_map = spatial_tokens.transpose(1, 2).reshape(
            batch_size,
            self.hidden_size,
            MAP_TOKEN_GRID_SIZE,
            MAP_TOKEN_GRID_SIZE,
        )
        return self.decoder(feature_map)
