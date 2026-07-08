#!/bin/bash
#SBATCH --job-name=try5-r1p5-dagger
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash run_r2r/main_server.bash priorgt_try5_dagger 2333
