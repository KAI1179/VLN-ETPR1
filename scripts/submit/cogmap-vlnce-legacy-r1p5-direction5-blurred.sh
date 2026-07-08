#!/bin/bash
#SBATCH --job-name=cogmap-vlnce-blur
#SBATCH -p vip_gpu_scze096
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv

python -m prior.blur_cognitive_maps \
  --cache-root data/cognitive_maps \
  --source-namespace gt.legacy.r1p5.direction5.v1 \
  --target-namespace gt.legacy.r1p5.direction5.blurred.v1 \
  --scale 2
