#!/bin/bash
#SBATCH --job-name=llm-smoke-eval
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash scripts/tries/smoke-llm-map-try7.sh llm5 2334
