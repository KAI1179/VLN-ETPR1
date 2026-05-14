# ETP Imagined

`etp_imagined` is the instruction-only cognitive-map variant of PriorGT.

PriorGT consumes precomputed ground-truth cognitive maps. ETP Imagined predicts a
soft cognitive map from the instruction text embeddings, then feeds that predicted
map through the existing PriorGT map-token encoder and graph-map cross-attention.

## Architecture

```text
instruction -> VLN text embeddings
text embeddings -> InstructionCognitiveMapPredictor
predicted grid logits -> sigmoid -> soft cognitive grid (B,37,100,100)
soft grid -> PriorGT map_encoding / EmbeddingGridMapEncoder
map tokens -> GraphMapCrossAttention -> navigation logits
```

`InstructionCognitiveMapPredictor` uses 100 learned `10x10` map queries. The
queries self-attend, cross-attend to instruction tokens, then produce 37-channel
coarse map logits that are upsampled to `100x100`.

## Training

Use:

- `TRAINER_NAME SS-ETP-Imagined`
- `MODEL.policy_name ImaginedPolicy`

During SS training, ground-truth cognitive maps are loaded only as supervision
targets for `BCEWithLogitsLoss`. They are not passed to navigation. During eval
or inference, no cognitive-map file is required.

The predictor is exposed separately through `mode="predict_cognitive_map"`.
`SS-ETP-Imagined` composes it with the existing PriorGT `mode="map_encoding"`
path, so the imagined model is conceptually the GT model with the GT map source
replaced by predicted maps.

The auxiliary map loss weight is:

```text
MODEL.MAP_ENCODER.map_loss_weight
```

Default: `0.1`.

## Scope

Current implementation covers SS/DAgger. GRPO and pretraining support are not
yet migrated to the imagined-map path.
