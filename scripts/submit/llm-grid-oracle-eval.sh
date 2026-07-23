#!/bin/bash
#SBATCH --job-name=llm-grid-oracle-eval
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-3
#SBATCH --gpus=2
#SBATCH -p vip_gpu_scze096
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u

assignments=(matched null within-scene global)
assignment="${assignments[$SLURM_ARRAY_TASK_ID]}"
cache_key="llm-grid-oracle-t0-v1-e2-${assignment}"

python -m vlnce_baselines.models.etp_llm.llm_grid_navigation_cache \
  --model-name-or-path \
    outputs/llm_grid/r2r-rxr-oracle-t0-v1-s2-e2-seed42/checkpoints/final \
  --cache-model-key "$cache_key" \
  --scope predictor-eval \
  --scale 2 \
  --batch-size 1 \
  --max-input-length 3072 \
  --max-new-tokens 3072 \
  --evidence-root data/llm_grid_oracle_evidence \
  --evidence-key oracle-t0-v1 \
  --evidence-assignment "$assignment" \
  --evidence-assignment-seed 42

python -m vlnce_baselines.models.etp_llm.llm_grid_eval \
  --cache-model-key "$cache_key" \
  --evidence-root data/llm_grid_oracle_evidence \
  --evidence-key oracle-t0-v1 \
  --output-dir "outputs/llm_grid_eval/oracle-t0-v1-e2-${assignment}"
