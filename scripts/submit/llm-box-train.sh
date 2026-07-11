#!/bin/bash
#SBATCH --job-name=llm-box-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=6
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
python -m vlnce_baselines.models.etp_llm.train_llm_boxes train \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --gradient-checkpointing \
  --max-new-tokens 2048
