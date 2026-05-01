import logging
from typing import List

import clip
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_EMBEDDING_DIM = 512

MAPPED_OBJECT_NAMES = [
    "void",
    "chair",
    "door",
    "table",
    "cushion",
    "sofa",
    "bed",
    "plant",
    "sink",
    "toilet",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "appliances",
    "structure",
    "other",
    "free-space",
    "picture",
    "cabinet",
    "chest_of_drawers",
    "stool",
    "towel",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes",
]

MAPPED_REGION_NAMES = [
    "outdoor/semi-outdoor",
    "living/social space",
    "recreation/fitness",
    "utility/service",
    "work/study",
    "circulation",
    "private room",
    "bathroom/sanitary",
    "dining/food",
    "other/miscellaneous",
]

DEFAULT_CATEGORY_NAMES = MAPPED_OBJECT_NAMES + MAPPED_REGION_NAMES
NUM_MAP_CATEGORIES = len(DEFAULT_CATEGORY_NAMES)


def _normalize_category_name(name: str) -> str:
    return (
        name.replace("_", " ")
        .replace("-", " ")
        .replace("/", " or ")
        .strip()
    )


def _default_category_prompts(category_names: List[str], num_object_categories: int) -> List[str]:
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
    device = "cpu"
    model, _ = clip.load(CLIP_MODEL_NAME, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        tokens = clip.tokenize(prompts)
        text_features = model.encode_text(tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    del model

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


def _group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = _group_norm(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = _group_norm(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.act(x)


class EmbeddingGridMapEncoder(nn.Module):
    """Encode a dense cognitive map into the VLN hidden space.

    The category channels are first projected with a 1x1 convolution whose
    weights are initialized from semantic category embeddings. A residual CNN
    then aggregates spatial structure before global pooling and projection into
    the 768-d navigation hidden space.
    """

    def __init__(
        self,
        output_size: int = 768,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.category_projection = nn.Conv2d(
            NUM_MAP_CATEGORIES, CLIP_EMBEDDING_DIM, kernel_size=1, bias=False
        )
        init_embeds = _build_category_projection_weights()
        with torch.no_grad():
            self.category_projection.weight.copy_(init_embeds.t().unsqueeze(-1).unsqueeze(-1))

        reduced_size = max(hidden_size, output_size // 6)
        self.encoder = nn.Sequential(
            nn.Conv2d(CLIP_EMBEDDING_DIM, reduced_size, kernel_size=1, bias=False),
            _group_norm(reduced_size),
            nn.GELU(),
            ResidualConvBlock(reduced_size, reduced_size, stride=1),
            ResidualConvBlock(reduced_size, reduced_size * 2, stride=2),
            ResidualConvBlock(reduced_size * 2, reduced_size * 2, stride=1),
            ResidualConvBlock(reduced_size * 2, reduced_size * 4, stride=2),
            ResidualConvBlock(reduced_size * 4, reduced_size * 4, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(reduced_size * 4, output_size),
            nn.LayerNorm(output_size),
        )

        # Keep navigation behavior identical to the R1 baseline at step 0.
        nn.init.zeros_(self.encoder[-2].weight)
        nn.init.zeros_(self.encoder[-2].bias)

    def forward(self, cognitive_crop: torch.Tensor) -> torch.Tensor:
        """Args:
        cognitive_crop: (B, CATEGORIES, H, W)

        Returns:
        (B, output_size)
        """
        embedding_map = self.category_projection(cognitive_crop)
        return self.encoder(embedding_map)
