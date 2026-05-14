import torch
import torch.nn as nn
import torch.nn.functional as F

from vlnce_baselines.models.etp_prior_gt.map_utils import (
    NUM_MAP_CATEGORIES,
    SIZE,
)


MAP_QUERY_GRID_SIZE = 10
MAP_QUERY_COUNT = MAP_QUERY_GRID_SIZE * MAP_QUERY_GRID_SIZE


class InstructionMapDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
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
        attn_out, _ = self.self_attn(
            map_queries,
            map_queries,
            map_queries,
            need_weights=False,
        )
        map_queries = self.norm1(map_queries + self.dropout(attn_out))
        txt_out, _ = self.cross_attn(
            map_queries,
            txt_embeds,
            txt_embeds,
            key_padding_mask=txt_key_padding_mask,
            need_weights=False,
        )
        map_queries = self.norm2(map_queries + self.dropout(txt_out))
        return self.norm3(map_queries + self.dropout(self.ffn(map_queries)))


class InstructionCognitiveMapPredictor(nn.Module):
    """Predict a dense cognitive-map prior from instruction text embeddings."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size must be divisible by num_heads, got {hidden_size}, {num_heads}")
        self.hidden_size = hidden_size
        self.map_queries = nn.Parameter(torch.zeros(1, MAP_QUERY_COUNT, hidden_size))
        self.map_pos = nn.Parameter(torch.zeros(1, MAP_QUERY_COUNT, hidden_size))
        self.layers = nn.ModuleList(
            [
                InstructionMapDecoderLayer(hidden_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.coarse_head = nn.Linear(hidden_size, NUM_MAP_CATEGORIES)
        self.refine_head = nn.Sequential(
            nn.Conv2d(NUM_MAP_CATEGORIES, NUM_MAP_CATEGORIES, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(NUM_MAP_CATEGORIES, NUM_MAP_CATEGORIES, kernel_size=1),
        )
        nn.init.normal_(self.map_queries, std=0.02)
        nn.init.normal_(self.map_pos, std=0.02)

    def _validate_inputs(self, txt_embeds: torch.Tensor, txt_masks: torch.Tensor) -> None:
        if txt_embeds.dim() != 3:
            raise ValueError(f"txt_embeds must have shape (B, L, H), got {tuple(txt_embeds.shape)}")
        if txt_masks.dim() != 2 or txt_masks.shape != txt_embeds.shape[:2]:
            raise ValueError(
                f"txt_masks must have shape {tuple(txt_embeds.shape[:2])}, got {tuple(txt_masks.shape)}"
            )
        if txt_embeds.shape[-1] != self.hidden_size:
            raise ValueError(
                f"txt_embeds hidden size must be {self.hidden_size}, got {txt_embeds.shape[-1]}"
            )

    def forward(self, txt_embeds: torch.Tensor, txt_masks: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(txt_embeds, txt_masks)
        batch_size = txt_embeds.shape[0]
        queries = self.map_queries + self.map_pos
        queries = queries.expand(batch_size, -1, -1)
        txt_key_padding_mask = txt_masks.logical_not()
        for layer in self.layers:
            queries = layer(queries, txt_embeds, txt_key_padding_mask)
        queries = self.output_norm(queries)
        coarse_logits = self.coarse_head(queries)
        coarse_logits = coarse_logits.transpose(1, 2).reshape(
            batch_size,
            NUM_MAP_CATEGORIES,
            MAP_QUERY_GRID_SIZE,
            MAP_QUERY_GRID_SIZE,
        )
        logits = F.interpolate(
            coarse_logits,
            size=(SIZE, SIZE),
            mode="bilinear",
            align_corners=False,
        )
        return self.refine_head(logits)
