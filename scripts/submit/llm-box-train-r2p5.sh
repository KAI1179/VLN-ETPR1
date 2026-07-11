#!/bin/bash
#SBATCH --job-name=llm-box-train-r2p5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=6
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
python -m vlnce_baselines.models.etp_llm.train_llm_boxes train \
  --batch-size 2 \
  --max-new-tokens 2048 \
  --cognitive-map-namespace gt.bbox.r2p5.path5.v1
