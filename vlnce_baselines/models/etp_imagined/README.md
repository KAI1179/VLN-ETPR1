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

### Predictor-Only Training

Predictor-only training should not use the VLN rollout trainer. The predictor
only needs paired instructions and ground-truth cognitive maps:

```text
instruction text -> frozen VLN language encoder -> txt_embeds/txt_masks
txt_embeds/txt_masks -> InstructionCognitiveMapPredictor -> map logits
map logits + GT cognitive map -> BCEWithLogitsLoss
```

That path avoids Habitat envs, waypoint prediction, navigation loss, and DAgger
rollout. It should save predictor weights that the full imagined policy can load
before SS/DAgger fine-tuning.

A standalone predictor trainer is the intended entry point, for example:

```bash
python -m vlnce_baselines.models.etp_imagined.train_map_predictor \
  --exp-config run_r2r/iter_train.yaml \
  --cognitive-map-dir data/cognitive_maps \
  --output data/logs/checkpoints/release_r2r_imagined_predictor/store/predictor.pt
```

This standalone trainer is not implemented yet. Until it exists, use
`imagined_dagger` for end-to-end training.

## Scope

Current implementation covers SS/DAgger. Predictor-only offline training, GRPO,
GRPO eval, and inference launcher paths are not yet migrated to the imagined-map
path.
