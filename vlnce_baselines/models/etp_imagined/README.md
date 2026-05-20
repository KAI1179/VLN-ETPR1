# ETP Imagined

`etp_imagined` is the inferred cognitive-map variant of PriorGT.

PriorGT consumes precomputed ground-truth cognitive maps. ETP Imagined predicts a
soft cognitive map from the instruction text embeddings plus inference-safe start
pose metadata, then feeds that predicted map through the existing PriorGT
map-token encoder and graph-map cross-attention.

## Architecture

```text
instruction -> VLN text embeddings
episode start_rotation/start_position -> start metadata
text embeddings + start metadata -> InstructionCognitiveMapPredictor
predicted grid logits -> sigmoid -> soft cognitive grid (B,37,100,100)
predicted direction vectors (B,5,2)
soft grid + predicted directions + real start metadata -> PriorGT map_encoding
map tokens -> GraphMapCrossAttention -> navigation logits
```

`InstructionCognitiveMapPredictor` uses an OccWorld-style latent map prior:

```text
VLN text embeddings
start_direction_vector + start_position
-> learned 10x10 latent map tokens
-> repeated cross-attention to text + latent self-attention + FFN blocks
-> latent-to-CNN projection
-> progressive upsample decoder
-> 37-channel 100x100 map logits
-> pooled latent direction head -> 5 route direction vectors
```

The grid output is still logits, not probabilities. Callers use
`sigmoid(logits)` before passing the soft map to `EmbeddingGridMapEncoder`.
`direction_vectors` are predicted by the predictor. `start_direction_vector` and
`start_position` are real episode-start metadata and do not use the reference
path.

## Training

Use:

- `TRAINER_NAME SS-ETP-Imagined`
- `MODEL.policy_name ImaginedPolicy`

During SS training, ground-truth cognitive maps are loaded only as supervision
targets for grid BCE and direction-vector MSE. They are not passed to navigation.
During eval or inference, no cognitive-map file is required; the trainer derives
start metadata from the current episode.

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

Pretrain the imagined model on the joint R2R/RxR pretraining data:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash pretrain_src/run_pt/run_mix_server.bash 2333 \
  --use_imagined \
  --checkpoint pretrained/r2r_rxr_ce/baseline/ckpts/model_step_367500.pt
```

The helper writes checkpoints under:

```text
pretrained/r2r_rxr_ce/imagined/ckpts/
```

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
map logits + predicted directions + GT cognitive map metadata -> weighted BCE/focal loss + direction MSE
```

That path avoids Habitat envs, waypoint prediction, navigation loss, and DAgger
rollout. It should save predictor weights that the full imagined policy can load
before SS/DAgger fine-tuning.

A standalone predictor script is provided for this. In its default `train` mode,
it reads the VLN dataset episodes and generates cognitive-map targets on the fly
from `reference_path`, matching PriorGT training.

```bash
python -m vlnce_baselines.models.etp_imagined.train_map_predictor \
  --opts MODEL.task_type r2r MODEL.pretrained_path pretrained/r2r_rxr_ce/prior_gt/store2/try-5-vlnce_step_462500.pt
```

The trainer freezes the same VLN language encoder used by the navigation model,
so the predictor sees text embeddings aligned with `ImaginedPolicy`. When
multiple GPUs are visible, it wraps the frozen text encoder and predictor in one
`DataParallel` module. Restrict cards with `CUDA_VISIBLE_DEVICES`, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vlnce_baselines.models.etp_imagined.train_map_predictor \
  --epochs 3 --batch-size 32 --limit 1024 --val-limit 256 \
  --opts MODEL.task_type r2r MODEL.pretrained_path pretrained/r2r_rxr_ce/prior_gt/store2/try-5-vlnce_step_462500.pt
```

The trainer reports sparse-map metrics across configurable thresholds:
`pred_pos@t`, `iou@t`, `precision@t`, `recall@t`, `top1pct_recall`, and
`top5pct_recall`, plus `direction_mae` and `direction_cos`. Best checkpoint
selection uses the best validation IoU across thresholds, not loss, because loss
can improve while the predictor over-expands positive cells.

It saves:

```text
map_predictor       # raw InstructionCognitiveMapPredictor state dict
state_dict          # same weights with map_predictor.* keys for policy loading
optimizer
metrics
args
```

Use `--limit N` and `--val-limit N` for quick smoke runs.

### Predictor Visualization

The same script can load a predictor checkpoint and visualize one generated map
without running Habitat rollout:

```bash
CUDA_VISIBLE_DEVICES=1 python -m vlnce_baselines.models.etp_imagined.train_map_predictor \
  --mode visualize \
  --predictor-checkpoint data/logs/checkpoints/release_r2r_imagined_predictor/store/predictor.best.pt \
  --visualize-dataset data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz \
  --episode-index 0 \
  --visualize-output-dir data/visualizations/map_predictor \
  --opts MODEL.task_type r2r MODEL.pretrained_path pretrained/r2r_rxr_ce/prior_gt/store2/try-5-vlnce_step_462500.pt
```

Use `nvidia-smi` first and bind `CUDA_VISIBLE_DEVICES` to a vacant card. The
visualizer loads the frozen VLN text encoder plus the map predictor, so it can
use several GB of GPU memory even for one example.

Selection options:

- `--episode-index N` selects by dataset order.
- `--episode-id ID` selects a specific episode.
- `--visualize-dataset PATH` overrides the default first validation dataset.
- `--skip-ground-truth` saves only `predicted.png`; otherwise it also saves
  `ground_truth.png` from the on-the-fly cognitive-map generator.

The generated images are written to:

```text
<visualize-output-dir>/episode_<episode_id>/predicted.png
<visualize-output-dir>/episode_<episode_id>/ground_truth.png
```

The script uses `typed-argument-parser` (`Tap`) field inference with dashed CLI
names, so Python fields such as `predictor_checkpoint` are exposed as
`--predictor-checkpoint`.

## Scope

Current implementation covers SS/DAgger and offline predictor-only training.
GRPO, GRPO eval, and inference launcher paths are not yet migrated to the
imagined-map path.
