#!/bin/bash
#SBATCH --job-name=cogmap-vlnce-d5
#SBATCH --gpus=1
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
python -m prior --map-source legacy --radius-m 1.5 --metadata-schema direction5
