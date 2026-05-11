# PriorGT Map Token Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PriorGT's pooled `map_embeds` fusion with spatial map tokens that global navigation graph nodes can attend to.

**Architecture:** `EmbeddingGridMapEncoder` will emit `(map_tokens, map_token_masks)` with 100 spatial tokens plus one metadata token. `GlocalTextPathNavCMT.forward_navigation` will cross-attend graph node embeddings to those map tokens through a small zero-initialized fusion module before the existing global encoder and action head.

**Tech Stack:** PyTorch, existing PriorGT policy/model files, lightweight pytest smoke tests, CLIP text initialization already present in `map_encoder.py`.

---

## File Structure

- Modify: `vlnce_baselines/models/etp_prior_gt/map_encoder.py`
  - Owns CLIP category projection, 10x10 map tokenization, metadata token creation, map-token transformer, and token mask generation.
- Modify: `vlnce_baselines/models/etp_prior_gt/policy.py`
  - Changes `mode="map_encoding"` to return map tokens and passes map-token tensors into navigation.
- Modify: `vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py`
  - Adds graph-to-map cross-attention fusion and replaces old broadcast `map_embeds` fusion.
- Modify: `vlnce_baselines/ss_trainer_ETP_PriorGT.py`
  - Stores `map_tokens` and `map_token_masks` in navigation inputs.
- Modify: `vlnce_baselines/GRPO_trainer_ETP_PriorGT.py`
  - Stores `map_tokens` and `map_token_masks` in navigation inputs and rollout step data.
- Create: `tests/etp_prior_gt/test_map_encoder_tokens.py`
  - Lightweight smoke tests for map encoder output shape and zero-token missing-map behavior.
- Create: `tests/etp_prior_gt/test_graph_map_cross_attention.py`
  - Lightweight smoke test for graph-to-map fusion shape and zero-initialized behavior.
- Modify: `vlnce_baselines/models/etp_prior_gt/README.md`
  - Update documentation after the code lands.

## Task 1: Add Map Encoder Token Smoke Tests

**Files:**
- Create: `tests/etp_prior_gt/test_map_encoder_tokens.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/etp_prior_gt/test_map_encoder_tokens.py` with this content:

```python
import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _install_fake_clip():
    fake_clip = types.ModuleType("clip")

    class FakeClipModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def eval(self):
            return self

        def encode_text(self, tokens):
            num_prompts = tokens.size(0)
            features = torch.arange(num_prompts * 512, dtype=torch.float32).view(num_prompts, 512)
            return features + 1.0

    def load(name, device="cpu"):
        return FakeClipModel(), None

    def tokenize(prompts):
        return torch.zeros(len(prompts), 77, dtype=torch.long)

    fake_clip.load = load
    fake_clip.tokenize = tokenize
    sys.modules["clip"] = fake_clip


def _load_priorgt_modules():
    _install_fake_clip()
    for name in [
        "vlnce_baselines",
        "vlnce_baselines.models",
        "vlnce_baselines.models.etp_prior_gt",
    ]:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    for mod_name, rel_path in [
        (
            "vlnce_baselines.models.etp_prior_gt.map_utils",
            "vlnce_baselines/models/etp_prior_gt/map_utils.py",
        ),
        (
            "vlnce_baselines.models.etp_prior_gt.map_encoder",
            "vlnce_baselines/models/etp_prior_gt/map_encoder.py",
        ),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

    return (
        sys.modules["vlnce_baselines.models.etp_prior_gt.map_utils"],
        sys.modules["vlnce_baselines.models.etp_prior_gt.map_encoder"],
    )


def test_map_encoder_returns_101_tokens_and_mask():
    map_utils, map_encoder = _load_priorgt_modules()
    encoder = map_encoder.EmbeddingGridMapEncoder()
    batch_size = 2
    grid = torch.randn(batch_size, map_utils.NUM_MAP_CATEGORIES, map_utils.SIZE, map_utils.SIZE)
    directions = torch.randn(batch_size, map_utils.DIRECTION_VECTOR_CNT, 2)
    starts = torch.randn(batch_size, 2)

    map_tokens, map_token_masks = encoder(grid, directions, starts)

    assert map_tokens.shape == (batch_size, 101, 768)
    assert map_token_masks.shape == (batch_size, 101)
    assert map_token_masks.dtype == torch.bool
    assert map_token_masks.all()


def test_map_encoder_rejects_bad_grid_shape():
    map_utils, map_encoder = _load_priorgt_modules()
    encoder = map_encoder.EmbeddingGridMapEncoder()
    bad_grid = torch.randn(2, map_utils.NUM_MAP_CATEGORIES - 1, map_utils.SIZE, map_utils.SIZE)
    directions = torch.randn(2, map_utils.DIRECTION_VECTOR_CNT, 2)
    starts = torch.randn(2, 2)

    try:
        encoder(bad_grid, directions, starts)
    except ValueError as exc:
        assert "cognitive_crop" in str(exc)
    else:
        raise AssertionError("Expected ValueError for bad cognitive_crop shape")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/etp_prior_gt/test_map_encoder_tokens.py -q`

Expected: FAIL because `EmbeddingGridMapEncoder.forward` still returns one tensor, not `(map_tokens, map_token_masks)`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/etp_prior_gt/test_map_encoder_tokens.py
git commit -m "test: add PriorGT map token encoder smoke tests"
```

## Task 2: Implement Map Token Encoder Output

**Files:**
- Modify: `vlnce_baselines/models/etp_prior_gt/map_encoder.py`

- [ ] **Step 1: Replace pooled map encoder internals with token output**

In `vlnce_baselines/models/etp_prior_gt/map_encoder.py`, update the constants near the top:

```python
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_EMBEDDING_DIM = 512
MAP_METADATA_DIM = DIRECTION_VECTOR_CNT * 2 + 2
MAP_TOKEN_GRID_SIZE = 10
MAP_SPATIAL_TOKEN_COUNT = MAP_TOKEN_GRID_SIZE * MAP_TOKEN_GRID_SIZE
MAP_TOKEN_COUNT = MAP_SPATIAL_TOKEN_COUNT + 1
MAP_TRANSFORMER_LAYERS = 2
MAP_TRANSFORMER_HEADS = 8
```

Replace the `EmbeddingGridMapEncoder` class body with this implementation:

```python
class EmbeddingGridMapEncoder(nn.Module):
    """Encode a dense cognitive map into graph-attendable map tokens."""

    def __init__(
        self,
        output_size: int = 768,
        hidden_size: int = 256,
        metadata_hidden_size: int = 256,
    ):
        super().__init__()
        self.output_size = output_size

        self.category_projection = nn.Conv2d(
            NUM_MAP_CATEGORIES, CLIP_EMBEDDING_DIM, kernel_size=1, bias=False
        )
        init_embeds = _build_category_projection_weights()
        with torch.no_grad():
            self.category_projection.weight.copy_(init_embeds.t().unsqueeze(-1).unsqueeze(-1))

        self.patch_projection = nn.Conv2d(
            CLIP_EMBEDDING_DIM,
            hidden_size,
            kernel_size=10,
            stride=10,
            bias=False,
        )
        self.patch_norm = nn.LayerNorm(hidden_size)
        self.token_projection = nn.Linear(hidden_size, output_size)

        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, MAP_SPATIAL_TOKEN_COUNT, output_size)
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(MAP_METADATA_DIM, metadata_hidden_size),
            nn.LayerNorm(metadata_hidden_size),
            nn.GELU(),
            nn.Linear(metadata_hidden_size, output_size),
            nn.LayerNorm(output_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_size,
            nhead=MAP_TRANSFORMER_HEADS,
            dim_feedforward=output_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=MAP_TRANSFORMER_LAYERS,
        )
        self.output_norm = nn.LayerNorm(output_size)

        nn.init.normal_(self.spatial_pos_embed, std=0.02)

    def _validate_inputs(
        self,
        cognitive_crop: torch.Tensor,
        direction_vectors: torch.Tensor,
        start_position: torch.Tensor,
    ) -> None:
        if cognitive_crop.dim() != 4 or cognitive_crop.shape[1:] != (
            NUM_MAP_CATEGORIES,
            100,
            100,
        ):
            raise ValueError(
                "cognitive_crop must have shape (B, 37, 100, 100); "
                f"got {tuple(cognitive_crop.shape)}"
            )
        batch_size = cognitive_crop.size(0)
        if direction_vectors.shape != (batch_size, DIRECTION_VECTOR_CNT, 2):
            raise ValueError(
                "direction_vectors must have shape (B, 5, 2); "
                f"got {tuple(direction_vectors.shape)}"
            )
        if start_position.shape != (batch_size, 2):
            raise ValueError(
                "start_position must have shape (B, 2); "
                f"got {tuple(start_position.shape)}"
            )

    def forward(
        self,
        cognitive_crop: torch.Tensor,
        direction_vectors: torch.Tensor,
        start_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return map tokens and token masks.

        Args:
            cognitive_crop: (B, 37, 100, 100)
            direction_vectors: (B, 5, 2)
            start_position: (B, 2)

        Returns:
            map_tokens: (B, 101, output_size)
            map_token_masks: (B, 101), true for valid tokens
        """
        self._validate_inputs(cognitive_crop, direction_vectors, start_position)
        embedding_map = self.category_projection(cognitive_crop)
        spatial_tokens = self.patch_projection(embedding_map).flatten(2).transpose(1, 2)
        spatial_tokens = self.patch_norm(spatial_tokens)
        spatial_tokens = self.token_projection(spatial_tokens)
        spatial_tokens = spatial_tokens + self.spatial_pos_embed

        metadata = torch.cat(
            [direction_vectors.flatten(start_dim=1), start_position],
            dim=1,
        )
        metadata_token = self.metadata_encoder(metadata).unsqueeze(1)
        map_tokens = torch.cat([spatial_tokens, metadata_token], dim=1)
        map_tokens = self.output_norm(self.token_encoder(map_tokens))
        map_token_masks = torch.ones(
            map_tokens.shape[:2],
            dtype=torch.bool,
            device=map_tokens.device,
        )
        return map_tokens, map_token_masks
```

Also update the typing import:

```python
from typing import List, Tuple
```

- [ ] **Step 2: Run the map encoder tests**

Run: `pytest tests/etp_prior_gt/test_map_encoder_tokens.py -q`

Expected: PASS.

- [ ] **Step 3: Run compile check**

Run: `python -m py_compile vlnce_baselines/models/etp_prior_gt/map_encoder.py`

Expected: no output and exit code 0.

- [ ] **Step 4: Commit**

```bash
git add vlnce_baselines/models/etp_prior_gt/map_encoder.py
git commit -m "feat: emit PriorGT map tokens"
```

## Task 3: Add Graph-To-Map Cross-Attention Tests

**Files:**
- Create: `tests/etp_prior_gt/test_graph_map_cross_attention.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/etp_prior_gt/test_graph_map_cross_attention.py` with this content:

```python
import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_vilmodel_cmt():
    transformers = types.ModuleType("transformers")

    class BertPreTrainedModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    transformers.BertPreTrainedModel = BertPreTrainedModel
    sys.modules.setdefault("transformers", transformers)

    for name in [
        "vlnce_baselines",
        "vlnce_baselines.common",
        "vlnce_baselines.models",
        "vlnce_baselines.models.etp_prior_gt",
    ]:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    ops = types.ModuleType("vlnce_baselines.common.ops")
    ops.create_transformer_encoder = lambda *args, **kwargs: None
    ops.extend_neg_masks = lambda masks: masks
    ops.gen_seq_masks = lambda lens: torch.ones(lens.size(0), int(lens.max()), dtype=torch.bool)
    ops.pad_tensors_wgrad = lambda tensors: torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    sys.modules["vlnce_baselines.common.ops"] = ops

    mod_name = "vlnce_baselines.models.etp_prior_gt.vilmodel_cmt"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        ROOT / "vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_graph_map_cross_attention_is_zero_initialized():
    module = _load_vilmodel_cmt()
    fusion = module.GraphMapCrossAttention(hidden_size=768, num_heads=12)
    graph_tokens = torch.randn(2, 4, 768)
    map_tokens = torch.randn(2, 101, 768)
    map_masks = torch.ones(2, 101, dtype=torch.bool)

    fused = fusion(graph_tokens, map_tokens, map_masks)

    assert fused.shape == graph_tokens.shape
    assert torch.allclose(fused, torch.zeros_like(fused))


def test_graph_map_cross_attention_handles_missing_tokens():
    module = _load_vilmodel_cmt()
    fusion = module.GraphMapCrossAttention(hidden_size=768, num_heads=12)
    graph_tokens = torch.randn(2, 4, 768)

    fused = fusion(graph_tokens, None, None)

    assert fused.shape == graph_tokens.shape
    assert torch.allclose(fused, torch.zeros_like(fused))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/etp_prior_gt/test_graph_map_cross_attention.py -q`

Expected: FAIL because `GraphMapCrossAttention` does not exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/etp_prior_gt/test_graph_map_cross_attention.py
git commit -m "test: add graph map cross attention smoke tests"
```

## Task 4: Implement Graph-To-Map Cross-Attention Fusion

**Files:**
- Modify: `vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py`

- [ ] **Step 1: Add the fusion module**

In `vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py`, add this class near the other model helper modules before `GlocalTextPathNavCMT`:

```python
class GraphMapCrossAttention(nn.Module):
    """Let global graph nodes attend to spatial map tokens."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm_graph = BertLayerNorm(hidden_size, eps=1e-12)
        self.norm_map = BertLayerNorm(hidden_size, eps=1e-12)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, graph_tokens, map_tokens=None, map_token_masks=None):
        if map_tokens is None:
            return torch.zeros_like(graph_tokens)
        graph_query = self.norm_graph(graph_tokens)
        map_context = self.norm_map(map_tokens)
        key_padding_mask = None
        if map_token_masks is not None:
            key_padding_mask = map_token_masks.logical_not()
        attended, _ = self.attn(
            graph_query,
            map_context,
            map_context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.output(attended)
```

- [ ] **Step 2: Instantiate the fusion module**

In `GlocalTextPathNavCMT.__init__`, after `self.global_encoder = GlobalMapEncoder(config)`, add:

```python
self.graph_map_cross_attention = GraphMapCrossAttention(
    hidden_size=config.hidden_size,
    num_heads=config.num_attention_heads,
)
```

- [ ] **Step 3: Update `forward_navigation` signature and fusion**

Change the signature from:

```python
map_embeds=None,
```

to:

```python
map_tokens=None,
map_token_masks=None,
```

Replace the old map fusion block:

```python
# Fuse embedding grid map context (broadcast over all graph nodes)
if map_embeds is not None:
    gmap_embeds = gmap_embeds + map_embeds.unsqueeze(1)
```

with:

```python
if map_tokens is not None:
    gmap_embeds = gmap_embeds + self.graph_map_cross_attention(
        gmap_embeds,
        map_tokens,
        map_token_masks,
    )
```

- [ ] **Step 4: Run graph map attention tests**

Run: `pytest tests/etp_prior_gt/test_graph_map_cross_attention.py -q`

Expected: PASS.

- [ ] **Step 5: Run compile check**

Run: `python -m py_compile vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py`

Expected: no output and exit code 0.

- [ ] **Step 6: Commit**

```bash
git add vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py
git commit -m "feat: fuse map tokens into global navigation"
```

## Task 5: Wire Map Tokens Through Policy and Trainers

**Files:**
- Modify: `vlnce_baselines/models/etp_prior_gt/policy.py`
- Modify: `vlnce_baselines/ss_trainer_ETP_PriorGT.py`
- Modify: `vlnce_baselines/GRPO_trainer_ETP_PriorGT.py`

- [ ] **Step 1: Update policy forward arguments**

In `vlnce_baselines/models/etp_prior_gt/policy.py`, change the forward argument tail from:

```python
cognitive_crops=None, direction_vectors=None, start_positions=None, map_embeds=None):
```

to:

```python
cognitive_crops=None, direction_vectors=None, start_positions=None,
map_tokens=None, map_token_masks=None):
```

Change the `map_encoding` comment and return path to:

```python
elif mode == 'map_encoding':
    # cognitive_crops: (B, 37, 100, 100) -> map tokens and masks
    assert self.map_encoder_enabled, "map_encoding mode requires MAP_ENCODER.enabled=True"
    return self.map_encoder(cognitive_crops, direction_vectors, start_positions)
```

Change the navigation call from:

```python
map_embeds=map_embeds,
```

to:

```python
map_tokens=map_tokens,
map_token_masks=map_token_masks,
```

- [ ] **Step 2: Update SS trainer map encoding block**

In `vlnce_baselines/ss_trainer_ETP_PriorGT.py`, replace:

```python
nav_inputs['map_embeds'] = self.policy.net(
    mode='map_encoding',
    cognitive_crops=cognitive_crops,
    direction_vectors=direction_vectors,
    start_positions=start_positions,
)
```

with:

```python
map_tokens, map_token_masks = self.policy.net(
    mode='map_encoding',
    cognitive_crops=cognitive_crops,
    direction_vectors=direction_vectors,
    start_positions=start_positions,
)
nav_inputs['map_tokens'] = map_tokens
nav_inputs['map_token_masks'] = map_token_masks
```

- [ ] **Step 3: Update GRPO trainer map encoding block**

In `vlnce_baselines/GRPO_trainer_ETP_PriorGT.py`, replace:

```python
current_map_embeds = self.policy.net(
    mode='map_encoding',
    cognitive_crops=cognitive_crops,
    direction_vectors=direction_vectors,
    start_positions=start_positions,
)
nav_inputs_for_gpu['map_embeds'] = current_map_embeds
```

with:

```python
current_map_tokens, current_map_token_masks = self.policy.net(
    mode='map_encoding',
    cognitive_crops=cognitive_crops,
    direction_vectors=direction_vectors,
    start_positions=start_positions,
)
nav_inputs_for_gpu['map_tokens'] = current_map_tokens
nav_inputs_for_gpu['map_token_masks'] = current_map_token_masks
```

- [ ] **Step 4: Update GRPO rollout storage names**

In `vlnce_baselines/GRPO_trainer_ETP_PriorGT.py`, replace the replay restore block currently around lines 751-760:

```python
if "map_embeds" in step_data:
    step_map_embeds = step_data["map_embeds"]
    # map_embeds saved during rollout are typically in the
    # current active-env order already. Re-index only when
    # the stored tensor is in original-batch layout.
    if step_map_embeds.size(0) == len(active_indices_in_original_batch):
        map_embeds_for_step = step_map_embeds
    else:
        map_embeds_for_step = step_map_embeds[active_indices_in_original_batch]
    nav_inputs_cuda['map_embeds'] = map_embeds_for_step.to(self.device, non_blocking=True)
```

with:

```python
if "map_tokens" in step_data and "map_token_masks" in step_data:
    step_map_tokens = step_data["map_tokens"]
    step_map_token_masks = step_data["map_token_masks"]
    if step_map_tokens.size(0) == len(active_indices_in_original_batch):
        map_tokens_for_step = step_map_tokens
        map_token_masks_for_step = step_map_token_masks
    else:
        map_tokens_for_step = step_map_tokens[active_indices_in_original_batch]
        map_token_masks_for_step = step_map_token_masks[active_indices_in_original_batch]
    nav_inputs_cuda["map_tokens"] = map_tokens_for_step.to(self.device, non_blocking=True)
    nav_inputs_cuda["map_token_masks"] = map_token_masks_for_step.to(self.device, non_blocking=True)
```

At the start of the rollout map encoding block currently around line 1036, replace:

```python
current_map_embeds = None
```

with:

```python
current_map_tokens = None
current_map_token_masks = None
```

Replace the step-data save block currently around lines 1076-1077:

```python
if current_map_embeds is not None:
    data_this_stepk["map_embeds"] = current_map_embeds.detach().cpu()
```

with:

```python
if current_map_tokens is not None:
    data_this_stepk["map_tokens"] = current_map_tokens.detach().cpu()
    data_this_stepk["map_token_masks"] = current_map_token_masks.detach().cpu()
```

- [ ] **Step 5: Verify no old map embeds remain in PriorGT path**

Run: `rg -n "map_embeds" vlnce_baselines/models/etp_prior_gt vlnce_baselines/ss_trainer_ETP_PriorGT.py vlnce_baselines/GRPO_trainer_ETP_PriorGT.py`

Expected: no output.

- [ ] **Step 6: Run compile checks**

Run:

```bash
python -m py_compile \
  vlnce_baselines/models/etp_prior_gt/policy.py \
  vlnce_baselines/ss_trainer_ETP_PriorGT.py \
  vlnce_baselines/GRPO_trainer_ETP_PriorGT.py
```

Expected: no output and exit code 0.

- [ ] **Step 7: Commit**

```bash
git add \
  vlnce_baselines/models/etp_prior_gt/policy.py \
  vlnce_baselines/ss_trainer_ETP_PriorGT.py \
  vlnce_baselines/GRPO_trainer_ETP_PriorGT.py
git commit -m "feat: pass map tokens through PriorGT policy"
```

## Task 6: Add End-To-End Static Verification

**Files:**
- Modify: `tests/etp_prior_gt/test_map_encoder_tokens.py`
- Modify: `tests/etp_prior_gt/test_graph_map_cross_attention.py`

- [ ] **Step 1: Add an initialization behavior assertion to map encoder test**

Append this test to `tests/etp_prior_gt/test_map_encoder_tokens.py`:

```python
def test_map_encoder_outputs_finite_tokens_for_zero_map():
    map_utils, map_encoder = _load_priorgt_modules()
    encoder = map_encoder.EmbeddingGridMapEncoder()
    batch_size = 2
    grid = torch.zeros(batch_size, map_utils.NUM_MAP_CATEGORIES, map_utils.SIZE, map_utils.SIZE)
    directions = torch.zeros(batch_size, map_utils.DIRECTION_VECTOR_CNT, 2)
    starts = torch.zeros(batch_size, 2)

    map_tokens, map_token_masks = encoder(grid, directions, starts)

    assert torch.isfinite(map_tokens).all()
    assert map_token_masks.all()
```

- [ ] **Step 2: Add a masked-token assertion to cross-attention test**

Append this test to `tests/etp_prior_gt/test_graph_map_cross_attention.py`:

```python
def test_graph_map_cross_attention_accepts_partial_masks():
    module = _load_vilmodel_cmt()
    fusion = module.GraphMapCrossAttention(hidden_size=768, num_heads=12)
    graph_tokens = torch.randn(2, 4, 768)
    map_tokens = torch.randn(2, 101, 768)
    map_masks = torch.ones(2, 101, dtype=torch.bool)
    map_masks[:, 50:] = False

    fused = fusion(graph_tokens, map_tokens, map_masks)

    assert fused.shape == graph_tokens.shape
    assert torch.isfinite(fused).all()
```

- [ ] **Step 3: Run all PriorGT map tests**

Run: `pytest tests/etp_prior_gt -q`

Expected: PASS.

- [ ] **Step 4: Run combined compile check**

Run:

```bash
python -m py_compile \
  vlnce_baselines/models/etp_prior_gt/map_encoder.py \
  vlnce_baselines/models/etp_prior_gt/policy.py \
  vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py \
  vlnce_baselines/ss_trainer_ETP_PriorGT.py \
  vlnce_baselines/GRPO_trainer_ETP_PriorGT.py
```

Expected: no output and exit code 0.

- [ ] **Step 5: Commit**

```bash
git add tests/etp_prior_gt
git commit -m "test: cover PriorGT map token fusion"
```

## Task 7: Update Documentation

**Files:**
- Modify: `vlnce_baselines/models/etp_prior_gt/README.md`

- [ ] **Step 1: Update architecture description**

In `vlnce_baselines/models/etp_prior_gt/README.md`, replace references to a single `(B, 768)` map vector with:

```markdown
The current PriorGT map encoder outputs `(B, 101, 768)` map tokens and a `(B, 101)` token mask.
The first 100 tokens correspond to a `10 x 10` spatial token grid over the cognitive map.
The final token encodes `direction_vectors` and `start_position` metadata.
Navigation graph nodes cross-attend to these map tokens before the existing global navigation encoder.
```

- [ ] **Step 2: Update fusion description**

Replace wording that says the map vector is broadcast-added to all graph nodes with:

```markdown
Map fusion happens through graph-to-map cross-attention in the global navigation branch.
The fusion projection is zero-initialized so enabling the map path starts close to the R1 baseline.
```

- [ ] **Step 3: Run a docs grep for stale old interface**

Run: `rg -n "map_embeds|single .*768|broadcast|pooled" vlnce_baselines/models/etp_prior_gt/README.md docs/superpowers/specs/2026-05-11-priorgt-map-token-fusion-design.md`

Expected: no stale references to the old runtime interface. References in the spec context section are acceptable only when explicitly describing the old behavior.

- [ ] **Step 4: Commit**

```bash
git add vlnce_baselines/models/etp_prior_gt/README.md
git commit -m "docs: describe PriorGT map token fusion"
```

## Task 8: Final Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run full targeted test suite**

Run: `pytest tests/etp_prior_gt -q`

Expected: PASS.

- [ ] **Step 2: Run targeted compile checks**

Run:

```bash
python -m py_compile \
  vlnce_baselines/models/etp_prior_gt/map_utils.py \
  vlnce_baselines/models/etp_prior_gt/map_encoder.py \
  vlnce_baselines/models/etp_prior_gt/policy.py \
  vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py \
  vlnce_baselines/ss_trainer_ETP_PriorGT.py \
  vlnce_baselines/GRPO_trainer_ETP_PriorGT.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Verify old interface is gone**

Run: `rg -n "map_embeds" vlnce_baselines/models/etp_prior_gt vlnce_baselines/ss_trainer_ETP_PriorGT.py vlnce_baselines/GRPO_trainer_ETP_PriorGT.py`

Expected: no output.

- [ ] **Step 4: Inspect git history**

Run: `git log --oneline -8`

Expected: recent commits show tests, encoder tokens, graph fusion, policy wiring, docs, and final verification work.
