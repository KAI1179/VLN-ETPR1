#!/bin/bash
#SBATCH --job-name=pose-gated-pt
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096

eval "$(conda shell.bash hook)" && conda activate etpr1-uv

bash scripts/llm-grid-pose-gated-pretrain-8gpu.sh \
  pretrained/r2r_rxr_ce/llm_grid_pose_gated_from367k_8gpu_n40
