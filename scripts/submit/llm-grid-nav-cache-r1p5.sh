#!/bin/bash
#SBATCH --job-name=llm-grid-nav-r1p5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv

set -u

python -m vlnce_baselines.models.etp_llm.llm_grid_navigation_cache \
  --model-name-or-path outputs/llm_grid/r2r-legacy-r1p5-direction5-scale2/checkpoints/final \
  --cache-model-key llm-grid-r2r-legacy-r1p5-direction5-scale2 \
  --scale 2 \
  --batch-size 8 \
  --max-new-tokens 2048
