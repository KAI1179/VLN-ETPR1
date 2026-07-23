#!/bin/bash
#SBATCH --job-name=llm-grid-oracle-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONHASHSEED=42

run_dir=outputs/llm_grid/r2r-rxr-oracle-t0-v1-s2-e2-seed42
mkdir "$run_dir"

torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  -m vlnce_baselines.models.etp_llm.llm_grid_train \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --gradient-checkpointing \
  --device-map none \
  --max-input-length 3072 \
  --max-new-tokens 3072 \
  --max-sequence-length 4096 \
  --cuda-cache-clear-min-sequence-length 3072 \
  --epochs 2 \
  --seed 42 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --cognitive-map-namespace gt.legacy.r1p5.direction5.v1 \
  --evidence-root data/llm_grid_oracle_evidence \
  --evidence-key oracle-t0-v1 \
  --output-dir "$run_dir"
