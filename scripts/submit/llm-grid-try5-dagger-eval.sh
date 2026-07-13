#!/bin/bash
#SBATCH --job-name=llm-grid-try5-eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash run_r2r/main_server.bash llm_grid_try5_eval_dagger 2333
