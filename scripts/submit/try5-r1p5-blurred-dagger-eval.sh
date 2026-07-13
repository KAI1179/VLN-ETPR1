#!/bin/bash
#SBATCH --job-name=try5-r1p5-blur-eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash run_r2r/main_server.bash prior_gt_try5_blurred_eval_dagger 2332
