#!/bin/bash
#SBATCH --job-name=llm-grid-oracle
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-5
#SBATCH --gpus=1
#SBATCH -p vip_gpu_scze096
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u

datasets=(R2R R2R R2R RxR RxR RxR)
splits=(train val_seen val_unseen train val_seen val_unseen)
dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
split="${splits[$SLURM_ARRAY_TASK_ID]}"

python -m vlnce_baselines.models.etp_llm.llm_grid_oracle_cache \
  --output-dir data/llm_grid_oracle_evidence \
  --evidence-key oracle-t0-v1 \
  --datasets "$dataset" \
  --splits "$split" \
  --gpu-device-id 0
