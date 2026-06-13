#!/bin/bash
#SBATCH --job-name=try7
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash pretrain_src/run_pt/run_mix_server.bash 2333 --use_prior_gt # --n_workers 4
