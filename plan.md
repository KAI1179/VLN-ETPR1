# Integration Plan: Embedding Grid Map into ETP-R1 Navigation Pipeline

## Goal

Add an embedding grid map as an additional spatial representation to the ETP-R1 navigation pipeline. The embedding grid map (ROWS × COLS × EMBEDDING_DIM) is derived from a cognitive grid map (CATEGORIES × ROWS × COLS) by taking weighted sums of category embeddings. This supplements the existing graph-based spatial representation (`GraphMap`) with dense, instruction-aware semantic context.

**Key constraint**: The **GT cognitive maps** (CATEGORIES × ROWS × COLS) are **precomputed offline** and saved as `.npy` files (~35 MB/scene). The weighted-average embedding step is performed **at runtime** on GPU, keeping storage minimal. No runtime dependency on `data.prior` or spaCy.

**Key constraint**: The author's ETP-R1 code is **not modified**. All new model code lives in a new folder `vlnce_baselines/models/etp_prior_gt/`.

---

## Architecture Overview

### Current Pipeline (ETP-R1)

```
Instruction ──→ XLM-RoBERTa ──→ txt_embeds
                                    │
RGB/Depth ──→ CLIP + ResNet ──→ Waypoints ──→ GraphMap ──→ gmap_*_fts
                                                              │
                                           ┌──────────────────┘
                                           ▼
                              GlocalTextPathNavCMT.forward_navigation()
                                           │
                                           ▼
                                     action logits
```

### Proposed Pipeline (with Embedding Grid Map)

```
Instruction ──→ XLM-RoBERTa ──→ txt_embeds
                                    │
RGB/Depth ──→ CLIP + ResNet ──→ Waypoints ──→ GraphMap ──→ gmap_*_fts
                                                              │
Precomputed GT cog. map (.npy)                                │
  (CATEGORIES × ROWS × COLS)                                  │
         │                                                    │
    [crop + load]                                             │
         │                                                    │
  (CATEGORIES × crop_H × crop_W)                             │
         │                                                    │
  category_embeds (CATEGORIES × EMBEDDING_DIM)                │
         │       ← weighted avg (matmul on GPU)               │
         ▼                                                    │
  (crop_H × crop_W × EMBEDDING_DIM)                          │
         │                                                    │
    MapEncoder (CNN)                                          │
         │                                                    │
    map_embeds (batch × hidden_size)                          │
         │                                                    │
         └────────────────────────────────────┐               │
                                              ▼               │
                        GlocalTextPathNavCMT_PriorGT.forward_navigation()
                                           │
                                           ▼
                                     action logits
```

---

## Precomputation Pipeline (Offline, uses `data/prior/`)

**GT cognitive maps** (not embedding maps) are built offline and stored on disk. The embedding step happens at runtime, keeping storage at ~35 MB per scene level instead of 286 MB–1.4 GB.

### Storage Comparison

| What is stored | Shape | Size/scene level | 61 scenes |
|----------------|-------|------------------|-----------|
| GT cognitive map (new) | (37, 500, 500) float32 | **35 MB** | **~2.1 GB** |
| Embedding map, spaCy | (500, 500, 300) float32 | 286 MB | ~17 GB |
| Embedding map, CLIP | (500, 500, 512) float32 | 488 MB | ~29 GB |
| Embedding map, BERT/XLM-R | (500, 500, 768) float32 | 1.4 GB | ~85 GB |

The GT cognitive map is **8–40× smaller** than pre-embedded maps, with the embedding method choice deferred to runtime.

### Precomputation Script: `data/prior/__main__.py`

**Output for each episode**:
- `data/cognitive_maps/{scene_id}/episode_{episode_id}.npy` — shape `(37, 500, 500)` float32
- `data/cognitive_maps/{scene_id}/episode_{episode_id}_meta.npz` — contains metadata

See `data/prior/README.md` for detailed information.

### Runtime Embedding Step (on GPU, in MapEncoder)

The category embeddings are a small fixed matrix of shape `(CATEGORIES, EMBEDDING_DIM)`. At runtime:

```python
# cognitive_crop: (batch, CATEGORIES, crop_H, crop_W)  — loaded from .npy, cropped
# category_embeds: (CATEGORIES, EMBEDDING_DIM)          — stored as nn.Embedding or buffer

# Reshape for matmul:
B, C, H, W = cognitive_crop.shape
crop_flat = cognitive_crop.permute(0, 2, 3, 1)          # (B, H, W, C)
embedding_map = crop_flat @ category_embeds              # (B, H, W, EMBEDDING_DIM)
embedding_map = embedding_map.permute(0, 3, 1, 2)       # (B, EMBEDDING_DIM, H, W)
# → feed into CNN encoder
```

This matmul is fast on GPU (101×101×37 × 37×EMBEDDING_DIM ≈ 113K × EMBEDDING_DIM multiply-adds per sample) and avoids storing the large embedding maps on disk.

---

## Embedding Method Comparison

The `data/prior/` code currently uses spaCy `en_core_web_lg` (300-dim GloVe vectors) to embed category names, although the code is not used when generating GT cognitive maps. Here is a comparison of alternatives:

### Option 1: spaCy GloVe (current, `en_core_web_lg`)

| Aspect | Details |
|--------|---------|
| **Dimension** | 300 |
| **Pros** | Simple and fast; well-understood word-level embeddings; good coverage of common nouns (furniture, rooms); lightweight inference (no GPU needed for embedding lookup); already implemented in `data/prior/` |
| **Cons** | Static word embeddings — no context sensitivity ("table" as furniture vs. "table" in "multiplication table" are identical); poor multi-word handling (spaCy averages token vectors for multi-word phrases like "chest of drawers"); no alignment with visual features (CLIP uses a separate embedding space); large model download (~750 MB) just for embeddings; 300-dim is smaller than the model's hidden_size (768), requiring a learned projection |
| **Best for** | Quick prototyping; baseline comparison |

### Option 2: CLIP Text Encoder (`ViT-B/32`)

| Aspect | Details |
|--------|---------|
| **Dimension** | 512 |
| **Pros** | **Vision-language aligned** — category embeddings live in the same space as the RGB features already used by ETP's `CLIPEncoder`, so the map and visual observations speak the same "language"; handles multi-word categories naturally (full sentence encoding); 512-dim is closer to hidden_size (768 for R2R, 512 for RxR — exact match for RxR); CLIP is already loaded in the pipeline (`CLIPEncoder` in `resnet_encoders.py`), so text encoder can be reused at precompute time with zero extra runtime cost |
| **Cons** | CLIP text embeddings are optimized for image-text matching, not fine-grained spatial semantics — "chair" and "sofa" may be closer than desired since both are "sit-able"; requires GPU at precompute time (minor, since precomputation is offline); slightly larger map files (512 vs. 300 per cell) |
| **Best for** | Leveraging existing CLIP alignment; especially strong for R2R where visual grounding matters |

### Option 3: BERT / XLM-RoBERTa Embeddings

| Aspect | Details |
|--------|---------|
| **Dimension** | 768 (BERT-base) or 768 (XLM-RoBERTa-base, already used by ETP for instructions) |
| **Pros** | Contextual embeddings — can encode category names with richer semantics; 768-dim matches ETP's hidden_size exactly (no projection needed for R2R); XLM-RoBERTa is already loaded by the model, so category embeddings could be extracted from the same language encoder used for instructions, ensuring same embedding space as `txt_embeds`; multilingual support (XLM-R) benefits RxR (Hindi, Telugu) |
| **Cons** | Heavier precomputation (transformer forward pass per category, though only ~37 categories so negligible); no vision alignment — category embeddings and RGB features are in different spaces; contextual BERT embeddings are sentence-dependent, so embedding isolated words like "chair" gives less benefit than full sentences; 768-dim maps are 2.5× larger than spaCy (500×500×768 ≈ 1.4 GB per map) |
| **Best for** | Matching the instruction encoder's representation; multilingual scenarios (RxR) |

### Summary Table

| Method | Dim | Vision-aligned | Multilingual | Runtime dep. | Matches hidden_size |
|--------|-----|---------------|-------------|-------------|-------------------|
| spaCy GloVe | 300 | No | No | None (embeddings stored as buffer) | No (needs projection) |
| CLIP ViT-B/32 | 512 | **Yes** | No | None (embeddings stored as buffer) | RxR only |
| BERT-base | 768 | No | No | None (embeddings stored as buffer) | R2R |
| XLM-RoBERTa | 768 | No | **Yes** | None (embeddings stored as buffer) | R2R |

Since the GT cognitive map (37 × 500 × 500, ~35 MB) is stored instead of the full embedding map, **map storage is identical regardless of embedding method** (~2.1 GB total for 61 scenes). The category embedding matrix (37 × EMBEDDING_DIM, <120 KB) is stored as an `nn.Parameter` or buffer in the model, making the embedding method a runtime config choice with zero impact on disk usage.

**Recommendation**: Start with **CLIP** for R2R (vision-language alignment, already in pipeline) and **XLM-RoBERTa** for RxR (multilingual, same space as instruction encoder). Category embeddings can be precomputed once from each method and saved as a small `.npy` file (~120 KB), or computed at model init time.

---

## Existing Code Inventory

### Reference Code (`data/prior/`) — used only for precomputation

| File | Key Exports | Purpose |
|------|-------------|---------|
| `constants.py` | `ROWS=500`, `COLS=500`, `CELL_SIZE=0.1`, `OBJECT_CATEGORIES=27`, `REGION_CATEGORIES=10` | Grid and category constants |
| `grid_map/__init__.py` | `BaseGridMap`, `GroundTruthGridMap`, `CognitiveGridMap`, `EmbeddingGridMap` | Grid map class hierarchy |
| `grid_map/_cognitive.py` | `build_cognitive_map()`, `build_embedding_map()`, `extract_categories()` | Map construction (offline only) |
| `grid_map/_construct.py` | `from_scene()`, `from_scene_id()` | GT map construction from MP3D scenes |

### Author's Code (ETP-R1) — **not modified**

| File | Key Classes/Functions | Role in Pipeline |
|------|----------------------|------------------|
| `vlnce_baselines/models/R1Policy.py` | `R1Policy`, `ETP(Net)`, `Critic` | Main policy; `ETP.forward()` dispatches via `mode=` |
| `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py` | `GlocalTextPathNavCMT` | Core VLN-BERT with `forward_txt()`, `forward_panorama()`, `forward_navigation()` |
| `vlnce_baselines/models/etp/ETP_R1_vlnbert_init.py` | `get_vlnbert_models()` | Model config + init |
| `vlnce_baselines/models/graph_utils.py` | `GraphMap`, `FloydGraph`, `estimate_cand_pos()` | Graph-based spatial representation |
| `vlnce_baselines/ss_trainer_ETP_R1.py` | `RLTrainer` (SS) | Training loop |
| `vlnce_baselines/GRPO_trainer_ETP_R1.py` | `RLTrainer` (GRPO) | RL fine-tuning loop |
| `vlnce_baselines/config/default.py` | `_C.MODEL.*` | Configuration tree |

---

## Placement Decision

All new model code lives in `vlnce_baselines/models/etp_prior_gt/`, a self-contained variant of the ETP model augmented with the embedding grid map. The author's original `etp/` and `R1Policy.py` are untouched.

| New Code | Location | Why |
|----------|----------|-----|
| Policy + ETP variant | `vlnce_baselines/models/etp_prior_gt/policy.py` | New `PriorGTPolicy` and `ETP_PriorGT(Net)` extending the original with map support |
| VLN-BERT variant | `vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py` | Copy of `GlocalTextPathNavCMT` with `map_embeds` fusion in `forward_navigation()` |
| VLN-BERT init | `vlnce_baselines/models/etp_prior_gt/vlnbert_init.py` | Copy of init, pointing to new vilmodel |
| Map encoder | `vlnce_baselines/models/etp_prior_gt/map_encoder.py` | `EmbeddingGridMapEncoder` CNN module |
| Map loading utils | `vlnce_baselines/models/etp_prior_gt/map_utils.py` | Load precomputed GT cognitive `.npy`, crop around agent |
| Package init | `vlnce_baselines/models/etp_prior_gt/__init__.py` | Package marker |
| Config additions | `vlnce_baselines/config/default.py` | Add `_C.MODEL.MAP_ENCODER.*` section (non-breaking) |
| Trainer (SS) | `vlnce_baselines/ss_trainer_ETP_PriorGT.py` | New trainer extending `ss_trainer_ETP_R1` with map logic |
| Trainer (GRPO) | `vlnce_baselines/GRPO_trainer_ETP_PriorGT.py` | New trainer extending `GRPO_trainer_ETP_R1` with map logic |
| Precompute script | `precompute_cognitive_maps.py` (project root) | Offline script using `data/prior/` |

---

## Step-by-Step Implementation Plan

### Step 0: Precompute GT Cognitive Maps (Offline)

Create `precompute_cognitive_maps.py` at project root:

```python
"""Precompute GT cognitive grid maps for all MP3D scenes.

Stores the raw category-probability grid (37 × 500 × 500, ~35 MB each),
not the full embedding map. The weighted-average embedding is done at runtime.

Usage:
    python precompute_cognitive_maps.py --output_dir data/cognitive_maps
"""
import argparse, os
import numpy as np
from data.prior.grid_map import GroundTruthGridMap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/cognitive_maps")
    parser.add_argument("--scenes", nargs="*", help="Scene IDs to process (default: all)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    scene_ids = args.scenes or get_all_scene_ids()
    for scene_id in scene_ids:
        gt_maps = GroundTruthGridMap.from_scene_id(scene_id)
        for level_idx, gt_map in enumerate(gt_maps):
            # gt_map.grid shape: (CATEGORIES, ROWS, COLS) = (37, 500, 500)
            np.save(
                os.path.join(args.output_dir, f"{scene_id}_level{level_idx}.npy"),
                gt_map.grid,
            )
            np.savez(
                os.path.join(args.output_dir, f"{scene_id}_level{level_idx}_meta.npz"),
                offset_x=gt_map.offset_x,
                offset_z=gt_map.offset_z,
                range_y=np.array(gt_map.range_y, dtype=object),
            )
        print(f"[{scene_id}] saved {len(gt_maps)} level(s)")

if __name__ == "__main__":
    main()
```

### Step 1: Add Configuration (`vlnce_baselines/config/default.py`)

Add a `MAP_ENCODER` section under `_C.MODEL` (non-breaking — existing code ignores it):

```python
_C.MODEL.MAP_ENCODER = CN()
_C.MODEL.MAP_ENCODER.enabled = False                              # Toggle feature on/off
_C.MODEL.MAP_ENCODER.num_categories = 37                          # OBJECT_CATEGORIES + REGION_CATEGORIES
_C.MODEL.MAP_ENCODER.embedding_dim = 512                          # Category embedding dim (CLIP=512, spaCy=300, BERT=768)
_C.MODEL.MAP_ENCODER.output_size = 768                            # Must match hidden_size for fusion
_C.MODEL.MAP_ENCODER.crop_radius = 50                             # Cells around agent (±5m at 0.1m/cell)
_C.MODEL.MAP_ENCODER.precomputed_dir = "data/cognitive_maps"      # Path to precomputed GT cognitive maps
_C.MODEL.MAP_ENCODER.category_embeds_path = ""                    # Path to (37, EMBEDDING_DIM) .npy; empty = learn from scratch
```

### Step 2: Create `vlnce_baselines/models/etp_prior_gt/` Package

#### `__init__.py`

Empty package marker.

#### `map_encoder.py` — Category Embedding + CNN Encoder

```python
import torch
import torch.nn as nn
import numpy as np

class EmbeddingGridMapEncoder(nn.Module):
    """Convert a GT cognitive map crop to embeddings, then encode with a CNN.

    Two stages:
      1. Weighted-average embedding: (B, CATEGORIES, H, W) → (B, EMBEDDING_DIM, H, W)
         via matmul with a (CATEGORIES, EMBEDDING_DIM) category embedding matrix.
      2. CNN encoder: (B, EMBEDDING_DIM, H, W) → (B, output_size)
    """

    def __init__(self, num_categories=37, embedding_dim=512, output_size=768,
                 category_embeds_path=""):
        super().__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim

        # Category embedding matrix — can be initialized from precomputed vectors
        if category_embeds_path:
            init_embeds = torch.from_numpy(np.load(category_embeds_path)).float()
            assert init_embeds.shape == (num_categories, embedding_dim)
            self.category_embeds = nn.Parameter(init_embeds)
        else:
            self.category_embeds = nn.Parameter(
                torch.randn(num_categories, embedding_dim) * 0.02
            )

        # CNN encoder
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

    def forward(self, cognitive_crop):
        """Forward pass.

        Args:
            cognitive_crop: (B, CATEGORIES, H, W) — cropped GT cognitive map
        Returns:
            (B, output_size) — map feature vector
        """
        B, C, H, W = cognitive_crop.shape
        # Weighted-average embedding: (B, H, W, C) @ (C, D) → (B, H, W, D)
        crop_flat = cognitive_crop.permute(0, 2, 3, 1)            # (B, H, W, C)
        embedding_map = crop_flat @ self.category_embeds           # (B, H, W, D)
        embedding_map = embedding_map.permute(0, 3, 1, 2)         # (B, D, H, W)
        return self.encoder(embedding_map)
```

**Key design**: The category embedding matrix is an `nn.Parameter`, so it can be:
- Initialized from precomputed CLIP/spaCy/XLM-R vectors (via `category_embeds_path`)
- Fine-tuned end-to-end during training
- Or learned from scratch (random init)

#### `map_utils.py` — Load Precomputed GT Cognitive Maps & Crop

```python
import os
import numpy as np
import torch

class PrecomputedCognitiveMap:
    """Lightweight wrapper for a precomputed GT cognitive grid map (.npy)."""

    def __init__(self, grid, offset_x, offset_z, cell_size=0.1):
        self.grid = grid          # (CATEGORIES, ROWS, COLS)
        self.offset_x = offset_x
        self.offset_z = offset_z
        self.cell_size = cell_size

    def world_to_grid(self, x, z):
        row = int((x - self.offset_x) // self.cell_size)
        col = int((z - self.offset_z) // self.cell_size)
        return row, col

def load_cognitive_map(precomputed_dir, scene_id, level_idx=0):
    """Load a precomputed GT cognitive map from disk."""
    grid = np.load(os.path.join(precomputed_dir, f"{scene_id}_level{level_idx}.npy"))
    meta = np.load(os.path.join(precomputed_dir, f"{scene_id}_level{level_idx}_meta.npz"),
                   allow_pickle=True)
    return PrecomputedCognitiveMap(
        grid=grid,
        offset_x=float(meta["offset_x"]),
        offset_z=float(meta["offset_z"]),
    )

def crop_cognitive_map(cog_map, agent_x, agent_z, crop_radius=50):
    """Extract a local crop of the cognitive map centered on the agent.

    Returns: torch tensor of shape (CATEGORIES, crop_size, crop_size)
    """
    row, col = cog_map.world_to_grid(agent_x, agent_z)
    grid = cog_map.grid  # (CATEGORIES, ROWS, COLS)
    # Pad spatial dims only (not category dim)
    padded = np.pad(grid,
                    ((0, 0), (crop_radius, crop_radius), (crop_radius, crop_radius)),
                    mode='constant')
    pr, pc = row + crop_radius, col + crop_radius
    crop = padded[:, pr - crop_radius : pr + crop_radius + 1,
                     pc - crop_radius : pc + crop_radius + 1]
    return torch.from_numpy(np.ascontiguousarray(crop)).float()
```

#### `vilmodel_cmt.py` — Copy of `ETP_R1_vilmodel_cmt.py` with Map Fusion

A copy of the author's `GlocalTextPathNavCMT` with one change: `forward_navigation()` accepts an optional `map_embeds` parameter and adds it to `gmap_embeds` before the cross-modal encoder.

Changed method signature and fusion point:

```python
def forward_navigation(
    self, txt_embeds, txt_masks,
    gmap_vpids, gmap_step_ids,
    gmap_img_fts, gmap_pos_fts,
    gmap_masks, gmap_visited_masks, gmap_pair_dists, gmap_task_embeddings,
    map_embeds=None,  # NEW
):
    ...
    gmap_embeds = gmap_img_fts + \
                  self.global_encoder.gmap_step_embeddings(gmap_step_ids) + \
                  task_type_encoding + \
                  self.global_encoder.gmap_pos_embeddings(gmap_pos_fts)

    # NEW: fuse embedding grid map context
    if map_embeds is not None:
        gmap_embeds = gmap_embeds + map_embeds.unsqueeze(1)

    ...  # rest unchanged
```

#### `vlnbert_init.py` — Copy of `ETP_R1_vlnbert_init.py`

Points to `etp_prior_gt.vilmodel_cmt.GlocalTextPathNavCMT` instead of `etp.ETP_R1_vilmodel_cmt.GlocalTextPathNavCMT`.

#### `policy.py` — New Policy & ETP Variant

```python
from vlnce_baselines.models.etp_prior_gt.vlnbert_init import get_vlnbert_models
from vlnce_baselines.models.etp_prior_gt.map_encoder import EmbeddingGridMapEncoder
# Reuse shared components from the original:
from vlnce_baselines.models.encoders.resnet_encoders import VlnResnetDepthEncoder, CLIPEncoder
from vlnce_baselines.models.policy import ILPolicy

@baseline_registry.register_policy
class PriorGTPolicy(ILPolicy):
    ...

class ETP_PriorGT(Net):
    def __init__(self, observation_space, model_config, num_actions, dropout_rate):
        # Same init as original ETP, plus:
        if model_config.MAP_ENCODER.enabled:
            self.map_encoder = EmbeddingGridMapEncoder(
                num_categories=model_config.MAP_ENCODER.num_categories,
                embedding_dim=model_config.MAP_ENCODER.embedding_dim,
                output_size=model_config.MAP_ENCODER.output_size,
                category_embeds_path=model_config.MAP_ENCODER.category_embeds_path,
            )

    def forward(self, mode=None, ..., cognitive_crops=None, map_embeds=None, ...):
        if mode == 'language':
            ...  # same as original
        elif mode == 'waypoint':
            ...  # same as original
        elif mode == 'panorama':
            ...  # same as original
        elif mode == 'map_encoding':
            # cognitive_crops: (batch, CATEGORIES, crop_H, crop_W)
            return self.map_encoder(cognitive_crops)  # → (batch, output_size)
        elif mode == 'navigation':
            outs = self.vln_bert.forward_navigation(
                ..., map_embeds=map_embeds,  # pass through
            )
            return outs
```

### Step 3: Create the Trainers

#### `vlnce_baselines/ss_trainer_ETP_PriorGT.py`

Copy of `ss_trainer_ETP_R1.py` with these additions:

1. At episode start: load precomputed GT cognitive maps from `data/cognitive_maps/`
2. At each step: crop cognitive map around agent position, encode via `mode='map_encoding'` (matmul + CNN)
3. Pass `map_embeds` into `mode='navigation'`

```python
from vlnce_baselines.models.etp_prior_gt.map_utils import load_cognitive_map, crop_cognitive_map

# In episode setup:
cognitive_maps = [
    load_cognitive_map(self.config.MODEL.MAP_ENCODER.precomputed_dir, scene_id)
    for scene_id in current_scene_ids
]

# In step loop, after panorama encoding:
cognitive_crops = torch.stack([
    crop_cognitive_map(cognitive_maps[i], cur_pos[i][0], cur_pos[i][2],
                       self.config.MODEL.MAP_ENCODER.crop_radius)
    for i in range(self.envs.num_envs)
]).to(self.device)
# Embedding + CNN encoding happens inside the model:
map_embeds = self.policy.net(mode='map_encoding', cognitive_crops=cognitive_crops)

nav_inputs['map_embeds'] = map_embeds
```

#### `vlnce_baselines/GRPO_trainer_ETP_PriorGT.py`

Same pattern applied to GRPO trainer.

---

## Data Flow Summary

```
OFFLINE (precompute_cognitive_maps.py):
  MP3D scene ──→ data/prior ──→ GroundTruthGridMap
                                      │
                                 gt_map.grid
                                      │
                              data/cognitive_maps/{scene}_level{N}.npy
                              (37 × 500 × 500)  ← ~35 MB each

RUNTIME (no data.prior dependency):
  Episode Start:
    scene_id ──→ np.load("data/cognitive_maps/{scene}_level0.npy")
                       │
                 PrecomputedCognitiveMap (lightweight wrapper)

  Each Step:
    agent position ──→ crop_cognitive_map()
                            │
                      (37 × 101 × 101) tensor   ← ~1.5 MB, category probs
                            │
                      ┌─ Weighted-avg embedding (matmul on GPU) ─┐
                      │  crop (B,H,W,37) @ embeds (37,D) → (B,H,W,D)  │
                      └──────────────────────────────────────────┘
                            │
                      (EMBEDDING_DIM × 101 × 101)
                            │
                      CNN encoder
                            │
                      map_embeds (batch × 768)
                            │
                ┌───────────┘
                ▼
    gmap_embeds += map_embeds.unsqueeze(1)   ← additive fusion
                │
                ▼
    forward_navigation() → action logits
```

---

## File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `precompute_cognitive_maps.py` | **New file** | Offline script using `data/prior/` to save GT cognitive maps as `.npy` |
| `vlnce_baselines/models/etp_prior_gt/__init__.py` | **New file** | Package marker |
| `vlnce_baselines/models/etp_prior_gt/policy.py` | **New file** | `PriorGTPolicy`, `ETP_PriorGT(Net)` with map encoder |
| `vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py` | **New file** | `GlocalTextPathNavCMT` variant with `map_embeds` fusion |
| `vlnce_baselines/models/etp_prior_gt/vlnbert_init.py` | **New file** | Init pointing to new vilmodel |
| `vlnce_baselines/models/etp_prior_gt/map_encoder.py` | **New file** | `EmbeddingGridMapEncoder`: category embedding matmul + CNN |
| `vlnce_baselines/models/etp_prior_gt/map_utils.py` | **New file** | `load_cognitive_map()`, `crop_cognitive_map()` |
| `vlnce_baselines/config/default.py` | **Edit** | Add `_C.MODEL.MAP_ENCODER` section (non-breaking) |
| `vlnce_baselines/ss_trainer_ETP_PriorGT.py` | **New file** | SS trainer with map integration |
| `vlnce_baselines/GRPO_trainer_ETP_PriorGT.py` | **New file** | GRPO trainer with map integration |

**No changes** to any author files: `R1Policy.py`, `etp/`, `ss_trainer_ETP_R1.py`, `GRPO_trainer_ETP_R1.py`.

**No runtime dependency** on `data/prior/` or spaCy.

---

## Design Considerations

### Performance: Cropping vs. Full Map

The full 37×500×500 cognitive map is ~35 MB per sample. Cropping to a 37×101×101 window (±50 cells = ±5m) reduces this to ~1.5 MB per sample, and the subsequent matmul produces a EMBEDDING_DIM×101×101 tensor on GPU. This is fast and memory-efficient.

**Alternative**: Downsample the full map (e.g., 4×→125×125) to preserve global context at lower resolution.

### Fusion Strategy

Three options for fusing `map_embeds` with the existing `gmap_embeds`:

1. **Additive** (recommended for initial experiment): `gmap_embeds += map_embeds.unsqueeze(1)` — simple, no extra params, preserves existing architecture.
2. **Concatenation + projection**: `gmap_embeds = proj(cat(gmap_embeds, map_embeds))` — more expressive but adds parameters and changes the embedding dimension.
3. **Cross-attention**: Add map embeddings as a separate sequence in the transformer — most expressive but most complex to integrate.

### Instruction-Aware vs. Full GT Maps

Two strategies for the precomputed maps:

1. **Full GT cognitive maps** (recommended): Precompute once per scene. Simpler, no instruction dependency, works identically at train and test time. The model learns to focus on relevant regions via cross-attention with `txt_embeds`. The category embedding matrix can also be fine-tuned to capture instruction-relevant semantics.
2. **Instruction-filtered cognitive maps**: Precompute per (scene, instruction, path) triple. Richer signal but requires `data.prior` at runtime or massive precomputation across all instruction variants. Also requires reference paths at inference time.

### Loading Strategy & Caching

Precomputed GT cognitive maps are ~35 MB each (~2.1 GB for 61 scenes). Options:
- **Pre-load all** (recommended): 61 scenes × 35 MB ≈ 2.1 GB fits comfortably in RAM. Load all maps at trainer startup for fastest access.
- **Lazy load + LRU cache**: Load maps on first access per scene, keep the N most recent in memory. Useful if training on a subset of scenes.
- **Memory-mapped files**: Use `np.load(..., mmap_mode='r')` to avoid loading the full array into RAM — the OS pages in only the cropped region.

---

## Testing Strategy

1. **Unit test**: Verify `EmbeddingGridMapEncoder` dimensions — input `(1, 37, 101, 101)` → output `(1, 768)`.
2. **Precompute test**: Run `precompute_cognitive_maps.py` on one scene, verify `.npy` shape `(37, 500, 500)` and coordinate transforms.
3. **Integration test**: Run one training step with `PriorGTPolicy` and `MAP_ENCODER.enabled = True`, verify loss computes without errors.
4. **Ablation**: Compare metrics (SR, SPL, NDTW) on `val_seen` split: ETP-R1 baseline vs. ETP-PriorGT.
