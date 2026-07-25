#!/bin/bash
#SBATCH --job-name=llm-grid-r2r-eval-sweep
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=1
#SBATCH -p vip_gpu_scze096
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u

python -m vlnce_baselines.models.etp_llm.llm_grid_eval_sweep \
  --run-manifest scripts/submit/llm-grid-r2r-only-eval-sweep-r1p5.json \
  --output-dir outputs/llm_grid_eval/r2r-only-checkpoint-sweep-r1p5
