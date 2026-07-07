#!/bin/bash
#SBATCH --job-name=llm-train-r2p5
#SBATCH --gpus=6
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
python -m vlnce_baselines.models.etp_llm.train_llm_boxes train --batch-size 2 --cognitive-map-namespace bbox_r2p5
