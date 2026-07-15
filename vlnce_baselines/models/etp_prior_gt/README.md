# ETP PriorGT

ETP PriorGT extends ETP-R1 by adding cognitive map features into the navigation branch.

## What Is Added

- Cognitive-map encoder: `EmbeddingGridMapEncoder`
- Map loading utilities for precomputed episode maps
- Token-level bidirectional map-token fusion in navigation forward
- New policy: `PriorGTPolicy`
- New trainers:
  - `SS-ETP-PriorGT`
  - `GRPO-ETP-PriorGT`

## Current Map Encoder

The PriorGT map encoder outputs spatial map tokens, not a single pooled vector:

- `map_tokens`: `(B, 101, hidden_size)`
- `map_token_masks`: `(B, 101)`, bool, `True` means valid

The 101 tokens are 100 spatial tokens from a fixed `10x10` grid over the `100x100`
cognitive map plus one metadata token from `trajectory_keypoints`, `start_direction_vector`,
and `start_position`.

Architecture:

1. Category projection
   - Input grid shape: `(B, 37, 100, 100)`
   - A `1x1` conv projects the 37 semantic channels into CLIP text space `(512)`
   - The `1x1` conv weights are initialized from CLIP text embeddings of the fixed
     37 object + region labels

2. Spatial tokenizer
   - `Conv2d(512, hidden_size, kernel_size=10, stride=10)`
   - Produces `10x10 = 100` spatial tokens
   - Applies token `LayerNorm`

3. Metadata branch
   - Extra map metadata is encoded alongside the grid:
     - `trajectory_keypoints`: shape `(B, 5, 2)`
     - `start_direction_vector`: shape `(B, 2)`
     - `start_position`: shape `(B, 2)`
   - These are flattened to `(B, 14)` and passed through a small MLP to produce
     one metadata token

4. Token encoder
   - Concatenates 100 spatial tokens plus metadata token
   - Runs a 2-layer Transformer encoder
   - Applies final `LayerNorm`

Notes:

- The old weighted-average / pooled map embedding path has been removed from PriorGT
  policy and trainers.
- The CLIP text encoder is used only once during initialization to build the category
  projection weights. It is not used in the map forward pass.

## Navigation Fusion

`GlocalTextPathNavCMT.forward_navigation()` now accepts:

- `map_tokens`
- `map_token_masks`

The shared `BidirectionalMapTokenFusion` module updates cognitive-map tokens from
valid global graph nodes first, excluding the STOP pseudo-node, then updates graph
node embeddings from those updated map tokens before the existing global encoder.
Both residual projections are zero-initialized, so initial behavior is identity
when map tokens are present and also no-op when map tokens are absent.

## Data Requirement

### Dynamic cognitive maps

PriorGT no longer loads precomputed cognitive-map `.npz` files during navigation
training/evaluation. Instead, it builds maps on the fly:

- `SceneSemanticBoxes.from_scene_id(scene_id)` builds/caches MP3D semantic boxes.
- The first level encountered by the dense ground-truth trajectory is selected.
- `SceneSemanticBoxes.to_cognitive_map(...)` creates the instruction cognitive map
  from trajectory-near boxes, scaling unmentioned categories below mentioned ones.
- The resulting `grid`, `trajectory_keypoints`, `start_direction_vector`, and
  `start_position` are passed directly to the map encoder.

This path reuses classes, constants, and map construction logic from `prior`.

## Checkpoint Compatibility

Both PriorGT trainers load checkpoints with `strict=False`, so **existing R1 checkpoints can be
loaded directly**. New `map_encoder.*` and `graph_map_attention.*` keys will be absent
and are initialised from defaults.

`BidirectionalMapTokenFusion` has zero-initialized residual projections, so
map-token fusion starts as an identity operation. This keeps initial navigation
behavior aligned with the R1 baseline while still allowing gradients to train map
fusion.

The map encoder initializes its category projection from built-in CLIP text
embeddings for the fixed 37 object+region labels.

For evaluation, use checkpoints produced by `SS-ETP-PriorGT` or `GRPO-ETP-PriorGT`.

## Quick Verification (Probe Mode)

You can verify whether cognitive maps improve performance **without full retraining** (~3 k
steps vs 30 k) by freezing the base model and training only the map encoder.

**Why this works:**
- Zero-init guarantees the model is exactly R1 at step 0.
- Only the map encoder parameters are trained instead of the full model.
- 3 k IL steps give a clear signal whether the map encoder adds value.

### Step 1 — Probe training (load R1 GRPO checkpoint, freeze base)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_r2r/main_server.bash priorgt_probe
```

This sets:
- `IL.ckpt_to_load data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth`
- `MODEL.MAP_ENCODER.freeze_base True`
- `IL.iters 3000`

### Step 2 — Evaluate the probe checkpoint

You may have to move `data/logs/checkpoints/release_r2r_priorgt_probe/ckpt.iter3000.pth` to `data/logs/checkpoints/release_r2r_priorgt_probe/store/ckpt.iter3000.pth` first.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_r2r/main_server.bash priorgt_probe_eval
```

Compare `SR` / `SPL` against the R1 GRPO baseline. If the probe is better, proceed with full
pipeline: `priorgt_dagger` → `priorgt_grpo` → `priorgt_eval_grpo`.

---

## Quick Start (R2R)

The launcher now supports dedicated PriorGT modes in `run_r2r/main_server.bash`.

### 1) Train IL (DAgger) PriorGT

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_r2r/main_server.bash priorgt_dagger
```

This runs with:

- `TRAINER_NAME SS-ETP-PriorGT`
- `MODEL.policy_name PriorGTPolicy`
- `MODEL.MAP_ENCODER.enabled True`

### 2) Train GRPO PriorGT

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_r2r/main_server.bash priorgt_grpo
```

By default this mode loads IL checkpoint:

- `GRPO.ckpt_to_load data/logs/checkpoints/release_r2r_priorgt_dagger/store/ckpt.iter28800.pth`

Update this path if your IL checkpoint name differs.

### 3) Evaluate PriorGT

Evaluate SS checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_r2r/main_server.bash priorgt_eval_ss
```

Evaluate GRPO checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_r2r/main_server.bash priorgt_eval_grpo
```

## Manual Override Command (Optional)

If you want to run without the bash mode, use direct overrides:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc-per-node=4 run.py \
  --exp_name release_r2r_priorgt_eval \
  --run-type eval \
  --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS [0,1,2,3] \
  TORCH_GPU_IDS [0,1,2,3] \
  GPU_NUMBERS 4 \
  NUM_ENVIRONMENTS 8 \
  TRAINER_NAME SS-ETP-PriorGT \
  MODEL.policy_name PriorGTPolicy \
  MODEL.MAP_ENCODER.enabled True \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_priorgt_dagger/store/ckpt.iter28800.pth
```

## Troubleshooting

- Error: trainer not found
  - Ensure imports are present in `vlnce_baselines/__init__.py`.
- Error: policy not found
  - Ensure `MODEL.policy_name` is `PriorGTPolicy`.
- Error: map construction fails
  - Check MP3D scene assets, connectivity data, and instruction metadata.
- Eval exits early with checkpoint issues
  - Confirm checkpoint was trained by PriorGT trainer/policy, not plain R1.
- Cuda out of memory!
  - Change `NUM_ENVIRONMENTS` in `main_server.bash` as needed.

## Pretraining Support

PriorGT maps are integrated into pretraining through the same token map interface used
by navigation training.

To run pretraining with PriorGT maps:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash pretrain_src/run_pt/run_mix_server.bash pretrained/r2r_rxr_ce/prior_gt \
    --use_prior_gt \
    --checkpoint pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt
```

When `--use_prior_gt` is enabled:

- The dataset loads precomputed maps from `data/cognitive_maps_etp_r1`.
- Annotation entries without a corresponding cache are removed during dataset
  construction and reported once as `skipped_missing`.
- The collate path stacks `grid`, `trajectory_keypoints`, `start_direction_vector`, and
  `start_position`.
- `EmbeddingGridMapEncoder` emits `map_tokens` and `map_token_masks`.
- Pretraining `GlocalTextPathCMT` uses the same shared bidirectional map-token fusion
  module as finetuning.

The loader does not substitute zero metadata for missing maps. A partially complete
cache skips missing annotation entries; a completely absent cache fails before
training starts.

PriorGT and LLM finetuning apply the same policy before constructing vector
environments: episodes without the required cache are excluded and summarized as
`skipped_missing`. A completely absent split cache fails before rollout starts.
