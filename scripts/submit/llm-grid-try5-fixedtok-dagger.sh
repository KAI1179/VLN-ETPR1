#!/bin/bash
#SBATCH --job-name=llm-grid-try5-fixedtok-dagger
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
# DAgger from the fixed LLM-Grid Try5 pretraining, loading its map encoder and
# fusion (the first DAgger whose pretrained raster path is actually trained).
# LLM_GRID_TRY5_FIXEDTOK_PRETRAINED_CKPT selects the pretraining step.
# The env's conda activate.d hooks read unset variables, so turn on nounset only afterwards.
eval "$(conda shell.bash hook)" && conda activate etpr1-uv || { echo "conda activate etpr1-uv failed" >&2; exit 2; }
set -euo pipefail
bash run_r2r/main_server.bash llm_grid_try5_fixedtok_dagger
