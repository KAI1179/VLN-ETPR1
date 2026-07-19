#!/bin/bash
#SBATCH --job-name=llm-grid-nav-r1p5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv

set -u

python -m vlnce_baselines.models.etp_llm.llm_grid_navigation_cache \
  --model-name-or-path outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag/checkpoints/final \
  --cache-model-key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
  --scale 2 \
  --batch-size 8 \
  --max-new-tokens 4096
