#!/bin/bash
#SBATCH --job-name=gt-cognitive-map-legacy-r2p5
#SBATCH --gpus=1
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
python -m prior.etp_r1 --map-source legacy --radius-m 2.5
python -m prior --map-source legacy --radius-m 2.5
