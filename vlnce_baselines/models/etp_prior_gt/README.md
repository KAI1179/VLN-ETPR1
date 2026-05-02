# ETP PriorGT

ETP PriorGT extends ETP-R1 by adding cognitive map features into the navigation branch.

## What Is Added

- Cognitive-map encoder: `EmbeddingGridMapEncoder`
- Map loading utilities for precomputed episode maps
- Additive fusion into global map embeddings in navigation forward
- New policy: `PriorGTPolicy`
- New trainers:
  - `SS-ETP-PriorGT`
  - `GRPO-ETP-PriorGT`

## Current Map Encoder

The current PriorGT map encoder outputs a single `(B, 768)` vector per episode map,
which is then broadcast-added to all global-map node embeddings in the navigation
branch.

Architecture:

1. Category projection
   - Input grid shape: `(B, 37, H, W)`
   - A `1x1` conv projects the 37 semantic channels into CLIP text space `(512)`
   - The `1x1` conv weights are initialized from CLIP text embeddings of the fixed
     37 object + region labels

2. Visual map backbone
   - `1x1 conv -> GroupNorm -> GELU`
   - 5 custom residual CNN blocks
   - two stride-2 downsampling stages
   - global average pooling

3. Metadata branch
   - Extra map metadata is encoded alongside the grid:
     - `direction_vectors`: shape `(B, 5, 2)`
     - `start_position`: shape `(B, 2)`
   - These are flattened to `(B, 12)` and passed through a small MLP

4. Fusion and output
   - Concatenate pooled visual feature and metadata feature
   - `Linear -> LayerNorm -> 768`
   - The final linear layer is zero-initialized so `map_embeds == 0` at step 0

Notes:

- The old explicit weighted-average matmul is now implemented as the CLIP-initialized
  `1x1` convolution over category channels.
- The CLIP text encoder is used only once during initialization to build the category
  projection weights. It is not used in the map forward pass.

## Data Requirement

### Cognitive maps

You must have precomputed cognitive maps at:

- `data/cognitive_maps/<scene_id>/R2R_<episode_id>.npz`
- `data/cognitive_maps/<scene_id>/RxR_<episode_id>.npz`

Each file is a single compressed NumPy archive (`np.savez_compressed`) containing:
- `grid` — shape `(37, H, W)` float32 semantic grid
- `offset_x`, `offset_z` — world-space origin of the map grid
- `direction_vectors` — shape `(5, 2)` float32 unit vectors
- `start_position` — shape `(2,)` float32 normalized continuous grid coordinates

Default config path is `MODEL.MAP_ENCODER.precomputed_dir = data/cognitive_maps`.

### Cognitive maps for pretraining

You must have precomputed cognitive maps at `data/cognitive_maps_etp_r1/<scene_id>/<instr_id>.npz`, with the same file structure.

## Checkpoint Compatibility

Both PriorGT trainers load checkpoints with `strict=False`, so **existing R1 checkpoints can be
loaded directly** — `map_encoder.*` keys will be absent and are initialised from the new map-encoder defaults.

The map-encoder output linear layer is zero-initialised, so at step 0 `map_embeds ≡ 0` and
the model behaves identically to the R1 baseline. Gradients teach the map encoder from there.

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
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash priorgt_probe 2333
```

This sets:
- `IL.ckpt_to_load data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth`
- `MODEL.MAP_ENCODER.freeze_base True`
- `IL.iters 3000`

### Step 2 — Evaluate the probe checkpoint

You may have to move `data/logs/checkpoints/release_r2r_priorgt_probe/ckpt.iter3000.pth` to `data/logs/checkpoints/release_r2r_priorgt_probe/store/ckpt.iter3000.pth` first.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash priorgt_probe_eval 2333
```

Compare `SR` / `SPL` against the R1 GRPO baseline. If the probe is better, proceed with full
pipeline: `priorgt_dagger` → `priorgt_grpo` → `priorgt_eval_grpo`.

---

## Quick Start (R2R)

The launcher now supports dedicated PriorGT modes in `run_r2r/main_server.bash`.

### 1) Train IL (DAgger) PriorGT

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_r2r/main_server.bash priorgt_dagger 2333
```

This runs with:

- `TRAINER_NAME SS-ETP-PriorGT`
- `MODEL.policy_name PriorGTPolicy`
- `MODEL.MAP_ENCODER.enabled True`

### 2) Train GRPO PriorGT

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_r2r/main_server.bash priorgt_grpo 2333
```

By default this mode loads IL checkpoint:

- `GRPO.ckpt_to_load data/logs/checkpoints/release_r2r_priorgt_dagger/store/ckpt.iter28800.pth`

Update this path if your IL checkpoint name differs.

### 3) Evaluate PriorGT

Evaluate SS checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_r2r/main_server.bash priorgt_eval_ss 2333
```

Evaluate GRPO checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_r2r/main_server.bash priorgt_eval_grpo 2333
```

## Manual Override Command (Optional)

If you want to run without the bash mode, use direct overrides:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=4 --master_port 2333 run.py \
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
  MODEL.MAP_ENCODER.precomputed_dir data/cognitive_maps \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_priorgt_dagger/store/ckpt.iter28800.pth
```

## Troubleshooting

- Error: trainer not found
  - Ensure imports are present in `vlnce_baselines/__init__.py`.
- Error: policy not found
  - Ensure `MODEL.policy_name` is `PriorGTPolicy`.
- Error: map files missing
  - Check `MODEL.MAP_ENCODER.precomputed_dir` and per-episode map files.
- Eval exits early with checkpoint issues
  - Confirm checkpoint was trained by PriorGT trainer/policy, not plain R1.
- Cuda out of memory!
  - Change `NUM_ENVIRONMENTS` in `main_server.bash` as needed.

## Pretraining Support

Prior GT maps are also integrated into the pretraining phase for learning continuous-level visual map priors offline.

To run pretraining incorporating PriorGT maps:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash pretrain_src/run_pt/run_mix_server.bash 2333 \
    --use_prior_gt \
    --checkpoint pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt
```

- When enabled via `--use_prior_gt`, the dataloader will fetch map contexts matching the target scans and forward them through the map encoder, fusing `map_embeds` into the `GlocalTextPathCMTPreTraining` architecture.
- If the map file is missing, the loader falls back to an all-zero cognitive map and zero metadata for that sample.
