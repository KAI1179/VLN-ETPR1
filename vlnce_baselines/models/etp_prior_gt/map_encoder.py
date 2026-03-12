import numpy as np
import torch
import torch.nn as nn


class EmbeddingGridMapEncoder(nn.Module):
    """Convert a cognitive map crop to embeddings, then encode with a CNN.

    Two stages:
      1. Weighted-average embedding: (B, CATEGORIES, H, W) -> (B, EMBEDDING_DIM, H, W)
         via matmul with a (CATEGORIES, EMBEDDING_DIM) category embedding matrix.
      2. CNN encoder: (B, EMBEDDING_DIM, H, W) -> (B, output_size)
    """

    def __init__(
        self,
        num_categories: int = 37,
        embedding_dim: int = 512,
        output_size: int = 768,
        category_embeds_path: str = "",
    ):
        super().__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim

        if category_embeds_path:
            init_embeds = torch.from_numpy(np.load(category_embeds_path)).float()
            assert init_embeds.shape == (num_categories, embedding_dim), (
                f"Expected category_embeds shape ({num_categories}, {embedding_dim}), "
                f"got {init_embeds.shape}"
            )
            self.category_embeds = nn.Parameter(init_embeds)
        else:
            self.category_embeds = nn.Parameter(
                torch.randn(num_categories, embedding_dim) * 0.02
            )

        self.encoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_size),
            nn.LayerNorm(output_size),
        )
        # Zero-init the output projection so map_embeds == 0 at the start of
        # training, keeping navigation identical to the R1 baseline until the
        # map encoder learns a useful signal.
        nn.init.zeros_(self.encoder[-2].weight)
        nn.init.zeros_(self.encoder[-2].bias)

    def forward(self, cognitive_crop: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            cognitive_crop: (B, CATEGORIES, H, W) — cropped cognitive map
        Returns:
            (B, output_size) — map feature vector
        """
        B, C, H, W = cognitive_crop.shape
        # Weighted-average embedding: (B, H, W, C) @ (C, D) -> (B, H, W, D)
        crop_flat = cognitive_crop.permute(0, 2, 3, 1)          # (B, H, W, C)
        embedding_map = crop_flat @ self.category_embeds         # (B, H, W, D)
        embedding_map = embedding_map.permute(0, 3, 1, 2)       # (B, D, H, W)
        return self.encoder(embedding_map)
