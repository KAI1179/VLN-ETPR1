# Cognitive Map Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decode updated map tokens into an updated cognitive map and train it with dense supervision before changing cognitive-map storage for MaskFormer-style per-entity masks.

**Architecture:** Add a compact decoder that consumes the 100 spatial updated map tokens, reshapes them to a 10x10 feature map, and predicts `(B, 37, 100, 100)` logits. Make updated map tokens a first-class navigation result in pretraining instead of discarding them inside `GlocalTextPathCMT`.

**Tech Stack:** PyTorch, pytest, ruff, ty.

---

### Task 1: Dense Decoder Module

**Files:**
- Create: `vlnce_baselines/models/etp_prior_gt/map_decoder.py`
- Test: `tests/etp_prior_gt/test_map_decoder.py`

- [x] **Step 1: Write the failing decoder test**

```python
def test_cognitive_map_decoder_predicts_dense_logits_from_spatial_tokens():
    decoder = CognitiveMapDecoder(hidden_size=32)
    updated_map_tokens = torch.randn(2, 101, 32)

    logits = decoder(updated_map_tokens)

    assert logits.shape == (2, 37, 100, 100)
    assert torch.isfinite(logits).all()
```

- [x] **Step 2: Implement the decoder**

```python
class CognitiveMapDecoder(nn.Module):
    def forward(self, updated_map_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = updated_map_tokens.shape[0]
        spatial_tokens = updated_map_tokens[:, :100]
        feature_map = spatial_tokens.transpose(1, 2).reshape(
            batch_size,
            self.hidden_size,
            10,
            10,
        )
        return self.decoder(feature_map)
```

- [x] **Step 3: Add explicit shape failures**

```python
with pytest.raises(ValueError, match="updated_map_tokens"):
    decoder(torch.randn(2, 100, 32))
```

### Task 2: Updated Map Tokens as Public Navigation Result

**Files:**
- Modify: `pretrain_src/pretrain_src/model/vilmodel.py`
- Modify: `pretrain_src/pretrain_src/model/pretrain_cmt.py`

- [x] **Step 1: Replace tuple return with a named navigation output**

```python
@dataclass
class NavigationModelOutput:
    txt_embeds: torch.Tensor
    gmap_embeds: torch.Tensor
    updated_map_tokens: Optional[torch.Tensor]
```

- [x] **Step 2: Update MLM/SAP callers**

```python
navigation_output = self.bert(...)
txt_embeds = navigation_output.txt_embeds
updated_map_tokens = navigation_output.updated_map_tokens
```

### Task 3: Dense Updated Cognitive Map Loss

**Files:**
- Modify: `pretrain_src/pretrain_src/model/pretrain_cmt.py`
- Test: `tests/etp_prior_gt/test_map_decoder.py`

- [x] **Step 1: Instantiate the decoder when map encoder is enabled**

```python
self.map_decoder = CognitiveMapDecoder(hidden_size=self.config.hidden_size)
```

- [x] **Step 2: Decode updated map tokens in pretraining**

```python
updated_map_logits = self.map_decoder(updated_map_tokens)
decoder_loss = F.binary_cross_entropy_with_logits(updated_map_logits, batch["cognitive_maps"])
```

- [x] **Step 3: Add decoder loss to the existing map loss path**

```python
return losses + self.map_loss_weight * decoder_loss
```

### Task 4: Documentation and Verification

**Files:**
- Modify: `docs/NOTE.md`
- Modify: `CONTEXT.md`

- [x] **Step 1: Document deferred MaskFormer storage change**
- [x] **Step 2: Document the cognitive map decoder term**
- [x] **Step 3: Run targeted tests**

```bash
pytest tests/etp_prior_gt/test_map_decoder.py tests/etp_prior_gt/test_graph_map_cross_attention.py tests/etp_prior_gt/test_map_fusion_shared.py -q
ruff check vlnce_baselines/models/etp_prior_gt/map_decoder.py pretrain_src/pretrain_src/model/vilmodel.py pretrain_src/pretrain_src/model/pretrain_cmt.py tests/etp_prior_gt/test_map_decoder.py
ty check vlnce_baselines/models/etp_prior_gt/map_decoder.py pretrain_src/pretrain_src/model/vilmodel.py pretrain_src/pretrain_src/model/pretrain_cmt.py tests/etp_prior_gt/test_map_decoder.py
```
