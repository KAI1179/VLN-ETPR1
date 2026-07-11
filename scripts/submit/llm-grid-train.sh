#!/bin/bash
#SBATCH --job-name=llm-grid-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=6
#SBATCH -p vip_gpu_scze096
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv

set -u

python -m vlnce_baselines.models.etp_llm.train_llm_grid train \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --gradient-checkpointing \
  --max-new-tokens 2048 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --cognitive-map-namespace gt.legacy.r1p5.direction5.v1 \
  --output-dir outputs/llm_grid/r2r-legacy-r1p5-direction5-scale2
