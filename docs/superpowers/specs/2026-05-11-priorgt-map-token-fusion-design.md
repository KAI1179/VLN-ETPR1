# PriorGT Map Token Fusion Design

## Context

PriorGT currently encodes each precomputed cognitive map into one `(B, 768)` vector and
broadcast-adds that vector to every global navigation graph node. This keeps the integration
small, but it discards spatial structure: every graph node receives the same map summary.

The SGM reference code in `data/SGM-main` uses a more token-oriented pattern. It patchifies
semantic maps with a ViT-style `PatchEmbed`, runs transformer blocks over map patch tokens,
and injects category language features through cross-attention. Its RoBERTa features are
precomputed category tensors, for example `chair.pt` has shape `(1, 25, 768)`.

For PriorGT, the useful idea to borrow is not the MAE reconstruction decoder. The useful idea
is token-level semantic map reasoning: preserve spatial map tokens and let navigation graph
nodes attend to them.

## Goals

- Replace global pooled `map_embeds` fusion with token-level map fusion.
- Keep map information focused on the global navigation graph branch.
- Preserve the existing waypoint and panorama candidate generation paths.
- Keep initialization close to baseline behavior with zero-initialized or gated map fusion.
- Continue using tensor inputs to the model, not `PrecomputedCognitiveMap` objects.

## Non-Goals

- Do not add map conditioning to waypoint prediction in this phase.
- Do not port SGM's MAE decoder or map reconstruction loss.
- Do not switch the current CLIP category initialization to RoBERTa in this phase.
- Do not make the model consume data container objects directly.

## Reference Pattern From SGM

SGM's cross variant follows this sequence:

1. Load precomputed RoBERTa category features from `.pt` files.
2. Inspect each semantic map to identify which category channels are present.
3. Concatenate present category token sequences into a fixed context tensor.
4. Patchify the semantic map with a ViT patch embedding.
5. Add positional embeddings and run transformer blocks over map patch tokens.
6. Cross-attend map tokens to language/category context tokens.
7. Decode masked map patches for reconstruction.

PriorGT should borrow steps 4-6 conceptually, but adapt them to navigation. The output should
be map tokens for graph-node attention, not reconstructed map patches.

## Proposed Architecture

### Map Encoder Output

The map encoder should output spatial tokens and masks:

```python
map_tokens: torch.Tensor       # (B, T, 768)
map_token_masks: torch.Tensor  # (B, T), true for valid tokens
```

`T` should be `101`: 100 spatial tokens from a `10 x 10` map token grid plus one metadata
token. For the current `100 x 100` maps, use `10 x 10` patches or an equivalent stride-10
projection. The output hidden size must remain `768`.

### Map Encoder Internals

The encoder should keep the current category and metadata inputs:

```python
cognitive_crops: torch.Tensor   # (B, 37, 100, 100)
direction_vectors: torch.Tensor # (B, 5, 2)
start_positions: torch.Tensor   # (B, 2)
```

The visual branch should start with the current fixed-vocabulary category projection:

- `Conv2d(37, 512, kernel_size=1, bias=False)`
- weights initialized from CLIP text embeddings of the 37 object + region labels

Then it should produce map tokens instead of immediately global-pooling:

- patch/downsample the projected map into a `10 x 10` spatial token grid
- add 2-D positional embeddings
- run a small transformer stack over map tokens
- encode direction vectors and start position into one metadata token
- append the metadata token to the spatial token sequence
- project tokens to `768`

### Navigation Fusion

The navigation branch should accept:

```python
map_tokens: Optional[torch.Tensor]
map_token_masks: Optional[torch.Tensor]
```

Inside `forward_navigation`, after constructing initial `gmap_embeds`, graph nodes should
cross-attend to map tokens:

```python
gmap_embeds = image_embeds + step_embeds + task_embeds + pos_embeds
map_context = graph_to_map_cross_attention(gmap_embeds, map_tokens, map_token_masks)
gmap_embeds = gmap_embeds + gated_map_context
```

The existing global encoder, graph-query-text fusion, and action head should remain after this
fusion step.

The map-fusion branch should be zero-initialized or controlled by a gate initialized to zero.
At step 0, enabling the map path should not materially change navigation logits.

## Data Flow

`PrecomputedCognitiveMap` remains a loader/container type. Trainers should batch plain tensors:

```python
cognitive_crops = torch.stack([...])      # (B, 37, 100, 100)
direction_vectors = torch.stack([...])    # (B, 5, 2)
start_positions = torch.stack([...])      # (B, 2)
```

The policy flow becomes:

1. Trainers call `policy.net(mode="map_encoding", ...)`.
2. The map encoder returns `map_tokens` and `map_token_masks`.
3. Trainers pass those into `policy.net(mode="navigation", ...)`.
4. `forward_navigation` cross-attends graph nodes to map tokens.

The old `map_embeds` argument should be removed or retained only behind a short-lived
compatibility path during migration. The preferred final interface is token-based.

## Missing Map Handling

When a map is missing, the fallback tensors should be:

- zero grid
- zero direction vectors
- zero start position

The map encoder may still emit fixed-shape zero-derived tokens with valid masks. This avoids
all-masked attention edge cases. The zero-initialized fusion gate/output keeps behavior stable
for missing maps and at initialization.

## Shape and Error Checks

Implementation should assert or validate:

- `cognitive_crops` is `(B, 37, 100, 100)`
- `direction_vectors` is `(B, 5, 2)`
- `start_positions` is `(B, 2)`
- `map_tokens` is `(B, T, 768)`
- `map_token_masks` is `(B, T)`
- if `MODEL.MAP_ENCODER.enabled` is true, navigation receives map tokens

Failures should be explicit shape errors rather than silent fallback behavior.

## Testing Plan

Minimum validation:

- Map encoder smoke test: dummy grid + metadata produces `(B, T, 768)` tokens and `(B, T)` masks.
- Navigation smoke test: `forward_navigation(..., map_tokens=..., map_token_masks=...)` returns finite logits with expected shape.
- Missing-map test: zero grid and zero metadata do not crash and produce finite logits.
- Initialization test: with the zero-initialized fusion gate/output, map fusion has zero or near-zero effect at initialization.
- Rollout path test: `SS-ETP-PriorGT` and `GRPO-ETP-PriorGT` pass map tokens into navigation when map encoder is enabled.

## Implementation Decisions

Use these defaults for implementation:

- Map token resolution: `10 x 10` spatial tokens from `100 x 100` maps.
- Metadata conditioning: one appended metadata token produced from `(5 * 2 + 2)` normalized metadata values.
- Navigation fusion: a dedicated graph-to-map cross-attention module in `vilmodel_cmt.py`.
- Stability: zero-initialize the fusion output projection or scalar gate so the new branch is inactive at step 0.
