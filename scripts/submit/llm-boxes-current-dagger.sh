#!/bin/bash
#SBATCH --job-name=llm-boxes-current-dagger
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash run_r2r/main_server.bash llm_boxes_current_dagger
