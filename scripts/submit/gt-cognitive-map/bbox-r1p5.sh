#!/bin/bash
#SBATCH --job-name=gt-cognitive-map-bbox-r1p5
#SBATCH --gpus=1
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
python -m prior.etp_r1 --map-source bbox --radius-m 1.5
python -m prior --map-source bbox --radius-m 1.5
