#!/bin/bash
#SBATCH --job-name=nav1
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash pretrain_src/run_pt/run_mix_server.bash 2334 --use_llm --n_workers 4
