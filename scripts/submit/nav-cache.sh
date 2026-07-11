#!/bin/bash
#SBATCH --job-name=nav-cache
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
python -m vlnce_baselines.models.etp_llm.generate_navigation_cache \
  --model-name-or-path ./data/logs/llm/checkpoints/final/ \
  --cache-model-key llama-3.1-8b-instruct \
  --batch-size 8

