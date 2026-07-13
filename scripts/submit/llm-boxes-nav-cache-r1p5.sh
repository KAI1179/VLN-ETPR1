#!/bin/bash
#SBATCH --job-name=llm-boxes-nav-r1p5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
python -m vlnce_baselines.models.etp_llm.llm_boxes_navigation_cache \
  --model-name-or-path ./outputs/llm_boxes/r2r-bbox-r1p5-path5/checkpoints/final/ \
  --cache-model-key llm-boxes-r1p5-path5 \
  --batch-size 8
