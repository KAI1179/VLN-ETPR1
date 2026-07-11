#!/bin/bash
#SBATCH --job-name=llm-grid-sample-s1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=1
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
python -m prior.llm_grid_samples \
  --namespace gt.legacy.r1p5.direction5.v1 \
  --count 100 \
  --scale 1 \
  --seed 708 \
  --tokenizer-path data/models/Llama-3.1-8B-Instruct \
  --max-new-tokens 5120
