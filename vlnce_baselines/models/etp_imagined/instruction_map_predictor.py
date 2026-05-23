import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from vlnce_baselines.models.etp_prior_gt.map_utils import (
    REFERENCE_PATH_LENGTH,
    NUM_MAP_CATEGORIES,
    SIZE,
)


MAP_QUERY_GRID_SIZE = 10
MAP_QUERY_COUNT = MAP_QUERY_GRID_SIZE * MAP_QUERY_GRID_SIZE
START_METADATA_DIM = 4


class InstructionLatentMapBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        map_queries: torch.Tensor,
        txt_embeds: torch.Tensor,
        txt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        txt_out, _ = self.cross_attn(
            map_queries,
            txt_embeds,
            txt_embeds,
            key_padding_mask=txt_key_padding_mask,
            need_weights=False,
        )
        map_queries = self.norm1(map_queries + self.dropout(txt_out))
        attn_out, _ = self.self_attn(
            map_queries,
            map_queries,
            map_queries,
            need_weights=False,
        )
        map_queries = self.norm2(map_queries + self.dropout(attn_out))
        return self.norm3(map_queries + self.dropout(self.ffn(map_queries)))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        norm_groups = min(8, out_channels)
        while out_channels % norm_groups != 0:
            norm_groups -= 1
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InstructionCognitiveMapPredictor(nn.Module):
    """OccWorld-style latent map prior from instruction text embeddings."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        latent_grid_size: int = MAP_QUERY_GRID_SIZE,
        decoder_channels: int = 256,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, got {hidden_size}, {num_heads}"
            )
        if latent_grid_size <= 0:
            raise ValueError(
                f"latent_grid_size must be positive, got {latent_grid_size}"
            )
        if decoder_channels % 8 != 0:
            raise ValueError(
                f"decoder_channels must be divisible by 8, got {decoder_channels}"
            )
        self.hidden_size = hidden_size
        self.latent_grid_size = latent_grid_size
        self.latent_count = latent_grid_size * latent_grid_size
        self.map_queries = nn.Parameter(torch.zeros(1, self.latent_count, hidden_size))
        self.map_pos = nn.Parameter(torch.zeros(1, self.latent_count, hidden_size))
        self.start_metadata_projection = nn.Sequential(
            nn.Linear(START_METADATA_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layers = nn.ModuleList(
            [
                InstructionLatentMapBlock(hidden_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_size, decoder_channels),
            nn.GELU(),
            nn.LayerNorm(decoder_channels),
        )
        upsample_blocks = []
        channels = decoder_channels
        current_size = latent_grid_size
        while current_size < SIZE:
            next_channels = max(NUM_MAP_CATEGORIES * 2, channels // 2)
            upsample_blocks.append(UpsampleBlock(channels, next_channels, dropout))
            channels = next_channels
            current_size *= 2
        self.decoder = nn.Sequential(*upsample_blocks)
        self.output_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, NUM_MAP_CATEGORIES, kernel_size=1),
        )
        self.reference_path_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, REFERENCE_PATH_LENGTH * 2),
        )
        nn.init.normal_(self.map_queries, std=0.02)
        nn.init.normal_(self.map_pos, std=0.02)

    def _validate_inputs(
        self,
        txt_embeds: torch.Tensor,
        txt_masks: torch.Tensor,
        start_direction_vectors: torch.Tensor,
        start_positions: torch.Tensor,
    ) -> None:
        if txt_embeds.dim() != 3:
            raise ValueError(
                f"txt_embeds must have shape (B, L, H), got {tuple(txt_embeds.shape)}"
            )
        if txt_masks.dim() != 2 or txt_masks.shape != txt_embeds.shape[:2]:
            raise ValueError(
                f"txt_masks must have shape {tuple(txt_embeds.shape[:2])}, got {tuple(txt_masks.shape)}"
            )
        if txt_embeds.shape[-1] != self.hidden_size:
            raise ValueError(
                f"txt_embeds hidden size must be {self.hidden_size}, got {txt_embeds.shape[-1]}"
            )
        if start_direction_vectors.dim() != 2 or start_direction_vectors.shape != (
            txt_embeds.shape[0],
            2,
        ):
            raise ValueError(
                "start_direction_vectors must have shape "
                f"({txt_embeds.shape[0]}, 2), got {tuple(start_direction_vectors.shape)}"
            )
        if start_positions.dim() != 2 or start_positions.shape != (
            txt_embeds.shape[0],
            2,
        ):
            raise ValueError(
                f"start_positions must have shape ({txt_embeds.shape[0]}, 2), got {tuple(start_positions.shape)}"
            )

    def forward(
        self,
        txt_embeds: torch.Tensor,
        txt_masks: torch.Tensor,
        start_direction_vectors: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = txt_embeds.shape[0]
        if start_direction_vectors is None:
            start_direction_vectors = txt_embeds.new_zeros(batch_size, 2)
        if start_positions is None:
            start_positions = txt_embeds.new_zeros(batch_size, 2)
        self._validate_inputs(
            txt_embeds, txt_masks, start_direction_vectors, start_positions
        )
        batch_size = txt_embeds.shape[0]
        queries = self.map_queries + self.map_pos
        queries = queries.expand(batch_size, -1, -1)
        start_metadata = torch.cat([start_direction_vectors, start_positions], dim=1)
        queries = queries + self.start_metadata_projection(start_metadata).unsqueeze(1)
        txt_key_padding_mask = txt_masks.logical_not()
        for layer in self.layers:
            queries = layer(queries, txt_embeds, txt_key_padding_mask)
        queries = self.output_norm(queries)
        reference_paths = self.reference_path_head(queries.mean(dim=1)).view(
            batch_size,
            REFERENCE_PATH_LENGTH,
            2,
        )
        latent = self.latent_projection(queries)
        latent = latent.transpose(1, 2).reshape(
            batch_size,
            -1,
            self.latent_grid_size,
            self.latent_grid_size,
        )
        logits = self.output_head(self.decoder(latent))
        if logits.shape[-2:] != (SIZE, SIZE):
            logits = F.interpolate(
                logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False
            )
        return logits, reference_paths
