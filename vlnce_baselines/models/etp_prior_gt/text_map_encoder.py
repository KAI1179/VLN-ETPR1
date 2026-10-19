"""Text-form cognitive map input: the ablation of the raster map encoder.

``MODEL.MAP_ENCODER.input_form text`` replaces ``EmbeddingGridMapEncoder``
with this module. The cognitive map is serialized into one short sentence
per entity (every nonzero raster channel, plus one heading sentence for the
direction vectors); each sentence is encoded by a frozen copy of the
policy's own instruction encoder, the first-token vector is projected by a
trainable linear layer, and the resulting entity tokens (variable count,
masked) plus the usual metadata token go to the unchanged graph-map fusion.

Nothing of the raster encoder exists in this mode: no category projection,
no spatial tokenizer, no map token transformer.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike, NDArray

from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES, OBJECT_CATEGORIES
from vlnce_baselines.models.etp_prior_gt.map_encoder import MAP_METADATA_DIM

MAX_ENTITY_TOKENS = 64
HEADING_DECIMALS = 2


def _downsample_max(grid: NDArray[np.float32], scale: int) -> NDArray[np.float32]:
    if scale == 1:
        return grid
    channels, rows, cols = grid.shape
    if rows % scale or cols % scale:
        raise ValueError(f"grid {grid.shape} not divisible by scale {scale}")
    return grid.reshape(channels, rows // scale, scale, cols // scale, scale).max(axis=(2, 4))


def map_text_entities(
    grid: ArrayLike,
    direction_vectors: ArrayLike,
    scale: int = 2,
) -> List[str]:
    """One sentence per nonzero category plus one heading sentence.

    Coordinates are on the downsampled grid (scale 2 -> 50x50 cells, the LLM
    output resolution). Example:
    ``region hallway: 23 cells, center (14,27), extent 6x11``.
    """
    sampled = _downsample_max(np.asarray(grid, dtype=np.float32), scale)
    directions = np.asarray(direction_vectors, dtype=np.float32)
    if directions.shape != (5, 2):
        raise ValueError(f"direction_vectors must be (5, 2), got {directions.shape}")
    sentences: List[str] = []
    for category in range(sampled.shape[0]):
        cells = np.argwhere(sampled[category] > 0)
        if len(cells) == 0:
            continue
        if category < OBJECT_CATEGORIES:
            kind, name = "object", MAPPED_OBJECT_NAMES[category]
        else:
            kind, name = "region", MAPPED_REGION_NAMES[category - OBJECT_CATEGORIES]
        center = cells.mean(axis=0)
        extent = cells.max(axis=0) - cells.min(axis=0) + 1
        sentences.append(
            f"{kind} {name.replace('_', ' ')}: {len(cells)} cells, "
            f"center ({int(round(center[0]))},{int(round(center[1]))}), "
            f"extent {int(extent[0])}x{int(extent[1])}"
        )
    heading = " ".join(
        f"({v[0]:.{HEADING_DECIMALS}f},{v[1]:.{HEADING_DECIMALS}f})" for v in directions
    )
    sentences.append(f"heading: {heading}")
    return sentences


class MapTextTokenizer:
    """BERT word-piece ids for entity sentences, [CLS] ... [SEP], padded."""

    def __init__(self, vocab_dir: str, pad_id: int = 1, max_length: int = MAX_ENTITY_TOKENS):
        from transformers import BertTokenizer

        self.tokenizer = BertTokenizer.from_pretrained(vocab_dir)
        self.pad_id = pad_id
        self.max_length = max_length
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id

    def __call__(self, sentences: Sequence[str]) -> torch.Tensor:
        """(N, max_length) long tensor, one row per sentence."""
        rows = []
        for sentence in sentences:
            ids = self.tokenizer.encode(sentence, add_special_tokens=False)
            ids = [self.cls_id] + ids[: self.max_length - 2] + [self.sep_id]
            ids += [self.pad_id] * (self.max_length - len(ids))
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long)


class TextMapEncoder(nn.Module):
    """Entity sentences -> map tokens (B, N+1, hidden) with masks (B, N+1)."""

    def __init__(self, vln_bert: nn.Module, hidden_size: int, task_type: int = 1):
        super().__init__()
        self.task_type = task_type
        # Frozen copy of the instruction encoder (pretrained weights are
        # already loaded into vln_bert when the policy builds this module).
        self.embeddings = deepcopy(vln_bert.embeddings)
        self.lang_encoder = deepcopy(vln_bert.lang_encoder)
        for module in (self.embeddings, self.lang_encoder):
            for param in module.parameters():
                param.requires_grad = False
        self.entity_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        # Same structure as EmbeddingGridMapEncoder.metadata_encoder; the
        # direction-vector slots of the 14-d input are always zero here
        # (the heading sentence carries them).
        self.metadata_encoder = nn.Sequential(
            nn.Linear(MAP_METADATA_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def train(self, mode: bool = True):
        super().train(mode)
        self.embeddings.eval()
        self.lang_encoder.eval()
        return self

    def forward(
        self,
        map_text_ids: torch.Tensor,
        map_text_masks: torch.Tensor,
        start_direction_vectors: torch.Tensor,
        start_positions: torch.Tensor,
        pad_id: int = 1,
    ):
        """map_text_ids: (B, N, L) long; map_text_masks: (B, N) bool (entity present)."""
        batch_size, entity_count, _ = map_text_ids.shape
        flat_ids = map_text_ids[map_text_masks]  # (M, L)
        with torch.no_grad():
            txt_masks = flat_ids != pad_id
            task_encoding = torch.where(
                txt_masks, torch.full_like(flat_ids, self.task_type), torch.zeros_like(flat_ids)
            )
            embeds = self.embeddings(flat_ids, task_encoding, token_type_ids=torch.zeros_like(flat_ids))
            embeds = self.lang_encoder(embeds, txt_masks)  # (M, L, H)
        entity_vectors = self.entity_projection(embeds[:, 0].float())  # (M, H)
        hidden = entity_vectors.shape[-1]
        entity_tokens = entity_vectors.new_zeros(batch_size, entity_count, hidden)
        entity_tokens[map_text_masks] = entity_vectors

        metadata = torch.cat(
            [
                start_positions.new_zeros(batch_size, MAP_METADATA_DIM - 4),
                start_direction_vectors.float(),
                start_positions.float(),
            ],
            dim=1,
        )
        metadata_token = self.metadata_encoder(metadata).unsqueeze(1)
        map_tokens = self.output_norm(torch.cat([entity_tokens, metadata_token], dim=1))
        map_token_masks = torch.cat(
            [map_text_masks, map_text_masks.new_ones(batch_size, 1)], dim=1
        )
        return map_tokens, map_token_masks
