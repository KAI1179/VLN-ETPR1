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

### Full Imagined Model

Train the normal SS/DAgger imagined model through the shared R2R launcher:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash imagined_dagger 2333
```

This trains the navigation model, the instruction-to-map predictor, and the
PriorGT map encoder together. The launcher uses:

```text
--exp_name release_r2r_imagined_dagger
--run-type dagger
TRAINER_NAME SS-ETP-Imagined
MODEL.policy_name ImaginedPolicy
MODEL.MAP_ENCODER.enabled True
MODEL.MAP_ENCODER.precomputed_dir data/cognitive_maps
MODEL.pretrained_path pretrained/r2r_rxr_ce/prior_gt/store2/reuse+full_192500.pt
```

Evaluate the SS checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash imagined_eval_ss 2333
```

The default eval checkpoint path is:

```text
data/logs/checkpoints/release_r2r_imagined_dagger/store/ckpt.iter30000.pth
```

Update `IMAGINED_DAGGER_CKPT` in `run_r2r/main_server.bash` if the saved
checkpoint name differs.

### Predictor-Only Probe

For quick verification, train only the instruction-to-map predictor while the
base VLN model and PriorGT map encoder stay frozen:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash imagined_predictor_probe 2333
```

This mode sets:

```text
MODEL.MAP_ENCODER.freeze_base True
MODEL.MAP_ENCODER.freeze_map_encoder True
IL.iters 3000
IL.lr 1e-4
```

`freeze_base=True` freezes the base VLN stack. `freeze_map_encoder=True` also
freezes the fixed CLIP-category map encoder, leaving `map_predictor` as the only
trainable imagined-map module.

## Scope

Current implementation covers SS/DAgger and predictor-only probe training. GRPO,
GRPO eval, and inference launcher paths currently print warnings because they
are not yet migrated to the imagined-map path.
