#!/bin/bash
#SBATCH --job-name=llm-grid-eval-sweep-r1p5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv

set -u

checkpoint_root=outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag/checkpoints
cache_key_prefix=llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch

for epoch in {1..9}; do
  python -m vlnce_baselines.models.etp_llm.llm_grid_navigation_cache \
    --model-name-or-path "${checkpoint_root}/epoch-${epoch}" \
    --cache-model-key "${cache_key_prefix}-${epoch}" \
    --scope predictor-eval \
    --scale 2 \
    --batch-size 8 \
    --max-new-tokens 4096
done

python -m vlnce_baselines.models.etp_llm.llm_grid_eval_sweep \
  --run-manifest scripts/submit/llm-grid-eval-sweep-r1p5.json \
  --output-dir outputs/llm_grid_eval/checkpoint-sweep-r1p5
