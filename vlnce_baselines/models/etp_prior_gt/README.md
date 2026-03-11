# ETP PriorGT

ETP PriorGT extends ETP-R1 by adding cognitive map features into the navigation branch.

## What Is Added

- Cognitive-map encoder: `EmbeddingGridMapEncoder`
- Map loading and crop utilities for precomputed episode maps
- Additive fusion into global map embeddings in navigation forward
- New policy: `PriorGTPolicy`
- New trainers:
  - `SS-ETP-PriorGT`
  - `GRPO-ETP-PriorGT`

## Data Requirement

You must have precomputed cognitive maps at:

- `data/cognitive_maps/<scene_id>/episode_<episode_id>.npy`
- optional meta: `data/cognitive_maps/<scene_id>/episode_<episode_id>_meta.npz`

Default config path is `MODEL.MAP_ENCODER.precomputed_dir = data/cognitive_maps`.

## Important Checkpoint Note

Do not evaluate ETP PriorGT with a checkpoint trained by plain ETP-R1 trainer/policy.

Reason: PriorGT has additional map-encoder parameters and uses different trainer/policy wiring.
Use checkpoints produced by `SS-ETP-PriorGT` or `GRPO-ETP-PriorGT`.

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

- `GRPO.ckpt_to_load data/logs/checkpoints/release_r2r_priorgt_dagger/store/ckpt.iter25000.pth`

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
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_priorgt_dagger/store/ckpt.iter25000.pth
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
