import logging
from typing import List, Optional, Tuple
from .map_utils import (
    MAPPED_OBJECT_NAMES,
    MAPPED_REGION_NAMES,
    NUM_MAP_CATEGORIES,
    SIZE,
    TRAJECTORY_KEYPOINT_COUNT,
)

import clip
import torch
import torch.nn as nn

from model_paths import CLIP_VIT_B32_MODEL

logger = logging.getLogger(__name__)
CLIP_MODEL_NAME = CLIP_VIT_B32_MODEL
CLIP_EMBEDDING_DIM = 512
MAP_METADATA_DIM = TRAJECTORY_KEYPOINT_COUNT * 2 + 2 + 2
MAP_TOKEN_GRID_SIZE = 10
MAP_SPATIAL_TOKEN_COUNT = MAP_TOKEN_GRID_SIZE * MAP_TOKEN_GRID_SIZE
MAP_TOKEN_COUNT = MAP_SPATIAL_TOKEN_COUNT + 1
MAP_TRANSFORMER_LAYERS = 2
MAP_TRANSFORMER_HEADS = 8

DEFAULT_CATEGORY_NAMES = MAPPED_OBJECT_NAMES + MAPPED_REGION_NAMES

CLIP_DEVICE = "cpu"
CLIP_MODEL, _ = clip.load(CLIP_MODEL_NAME, device=CLIP_DEVICE)
CLIP_MODEL.eval()
for param in CLIP_MODEL.parameters():
    param.requires_grad_(False)


def _normalize_category_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").replace("/", " or ").strip()


def _default_category_prompts(
    category_names: List[str], num_object_categories: int
) -> List[str]:
    prompts = []
    for idx, category_name in enumerate(category_names):
        readable_name = _normalize_category_name(category_name)
        if idx < num_object_categories:
            prompts.append(f"an indoor navigation map cell containing {readable_name}")
        else:
            prompts.append(f"an indoor navigation map region for {readable_name}")
    return prompts


def _load_clip_text_embeddings(category_names: List[str]) -> torch.Tensor:
    prompts = _default_category_prompts(category_names, len(MAPPED_OBJECT_NAMES))

    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(CLIP_DEVICE)
        text_features = CLIP_MODEL.encode_text(tokens).float()
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

    if text_features.shape != (len(category_names), CLIP_EMBEDDING_DIM):
        raise ValueError(
            f"Expected CLIP text features with shape ({len(category_names)}, {CLIP_EMBEDDING_DIM}), "
            f"got {tuple(text_features.shape)}"
        )
    return text_features


def _build_category_projection_weights() -> torch.Tensor:
    try:
        return _load_clip_text_embeddings(DEFAULT_CATEGORY_NAMES)
    except Exception as exc:
        logger.exception("Failed to initialize map category embeddings from CLIP text")
        raise RuntimeError("CLIP-based map category initialization failed") from exc


class NormFirstTransformerEncoderLayer(nn.Module):
    """PyTorch 1.9-compatible pre-norm transformer encoder layer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            MAP_TRANSFORMER_HEADS,
            dropout=0.0,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.0)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if is_causal:
            raise ValueError("Causal map-token attention is unsupported")
        attn_input = self.norm1(src)
        attn_output, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = src + self.dropout(attn_output)
        ff_input = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(ff_input))))
        return src + self.dropout(ff_output)


class EmbeddingGridMapEncoder(nn.Module):
    """Encode a dense cognitive map into map tokens for VLN fusion.

    The category channels are first projected with a 1x1 convolution whose
    weights are initialized from fixed CLIP category text embeddings. A spatial
    tokenizer emits a 10x10 token grid, and a metadata token is appended before
    transformer encoding.
    """

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        if hidden_size % MAP_TRANSFORMER_HEADS != 0:
            raise ValueError(
                f"hidden_size must be divisible by {MAP_TRANSFORMER_HEADS}, got {hidden_size}"
            )
        self.hidden_size = hidden_size

        self.category_projection = nn.Conv2d(
            NUM_MAP_CATEGORIES, CLIP_EMBEDDING_DIM, kernel_size=1, bias=False
        )
        init_embeds = _build_category_projection_weights()
        with torch.no_grad():
            self.category_projection.weight.copy_(
                init_embeds.t().unsqueeze(-1).unsqueeze(-1)
            )

        self.spatial_tokenizer = nn.Conv2d(
            CLIP_EMBEDDING_DIM,
            hidden_size,
            kernel_size=10,
            stride=10,
        )
        self.spatial_token_norm = nn.LayerNorm(hidden_size)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(MAP_METADATA_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        encoder_layer = NormFirstTransformerEncoderLayer(hidden_size)
        self.token_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=MAP_TRANSFORMER_LAYERS,
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def _validate_inputs(
        self,
        cognitive_crop: torch.Tensor,
        trajectory_keypoints: torch.Tensor,
        start_direction_vectors: torch.Tensor,
        start_positions: torch.Tensor,
    ) -> int:
        if cognitive_crop.dim() != 4 or cognitive_crop.shape[1:] != (
            NUM_MAP_CATEGORIES,
            SIZE,
            SIZE,
        ):
            raise ValueError(
                "cognitive_crop must have shape "
                f"(B, {NUM_MAP_CATEGORIES}, {SIZE}, {SIZE}), got {tuple(cognitive_crop.shape)}"
            )
        batch_size = cognitive_crop.shape[0]
        if trajectory_keypoints.dim() != 3 or trajectory_keypoints.shape[1:] != (
            TRAJECTORY_KEYPOINT_COUNT,
            2,
        ):
            raise ValueError(
                "trajectory_keypoints must have shape "
                f"(B, {TRAJECTORY_KEYPOINT_COUNT}, 2), got {tuple(trajectory_keypoints.shape)}"
            )
        if start_direction_vectors.dim() != 2 or start_direction_vectors.shape[1:] != (
            2,
        ):
            raise ValueError(
                "start_direction_vectors must have shape "
                f"(B, 2), got {tuple(start_direction_vectors.shape)}"
            )
        if start_positions.dim() != 2 or start_positions.shape[1:] != (2,):
            raise ValueError(
                f"start_positions must have shape (B, 2), got {tuple(start_positions.shape)}"
            )
        if (
            cognitive_crop.shape[0] != trajectory_keypoints.shape[0]
            or cognitive_crop.shape[0] != start_direction_vectors.shape[0]
            or cognitive_crop.shape[0] != start_positions.shape[0]
        ):
            raise ValueError(
                "batch sizes must match for cognitive_crop, trajectory_keypoints, "
                "start_direction_vectors, and start_positions"
            )
        return batch_size

    def forward(
        self,
        cognitive_crop: torch.Tensor,
        trajectory_keypoints: torch.Tensor,
        start_direction_vectors: torch.Tensor,
        start_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
        cognitive_crop: (B, CATEGORIES, H, W)
        trajectory_keypoints: (B, TRAJECTORY_KEYPOINT_COUNT, 2)
        start_direction_vectors: (B, 2)
        start_positions: (B, 2)

        Returns:
        map_tokens: (B, MAP_TOKEN_COUNT, hidden_size)
        map_token_masks: (B, MAP_TOKEN_COUNT)
        """
        batch_size = self._validate_inputs(
            cognitive_crop,
            trajectory_keypoints,
            start_direction_vectors,
            start_positions,
        )

        # cognitive_crop: (B, 37, 100, 100) -> embedding_map: (B, 512, 100, 100).
        embedding_map = self.category_projection(cognitive_crop)
        spatial_tokens = self.spatial_tokenizer(
            embedding_map
        )  # (B, hidden_size, 10, 10), kernel_size=10, stride=10.
        spatial_tokens = spatial_tokens.flatten(start_dim=2).transpose(
            1, 2
        )  # (B, 100, hidden_size)
        spatial_tokens = self.spatial_token_norm(spatial_tokens)

        # metadata: (B, 14) = flattened keypoints (5*2), direction (2), start (2).
        metadata = torch.cat(
            [
                trajectory_keypoints.flatten(start_dim=1),
                start_direction_vectors,
                start_positions,
            ],
            dim=1,
        )
        # metadata_token: (B, 1, hidden_size), appended after the 10x10 map tokens.
        metadata_token = self.metadata_encoder(metadata).unsqueeze(1)

        # map_tokens: (B, 101, hidden_size). The mask is all true because the
        # dense raster always emits all 100 spatial tokens plus one metadata token.
        map_tokens = torch.cat([spatial_tokens, metadata_token], dim=1)
        map_tokens = self.token_transformer(map_tokens)
        map_tokens = self.output_norm(map_tokens)
        map_token_masks = torch.ones(
            batch_size,
            MAP_TOKEN_COUNT,
            dtype=torch.bool,
            device=cognitive_crop.device,
        )
        return map_tokens, map_token_masks
