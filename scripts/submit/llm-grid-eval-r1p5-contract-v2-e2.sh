#!/bin/bash
#SBATCH --job-name=llm-grid-contract-v2-eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u

run_dir=outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag-contract-v2-e2
cache_key=llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-contract-v2-epoch-2
cache_root="data/llm_navigation/$cache_key"
eval_dir=outputs/llm_grid_eval/contract-v2-epoch-2-r1p5

mkdir "$cache_root"
RUN_DIR="$run_dir" python -c \
  'import os; from pathlib import Path; from vlnce_baselines.models.etp_llm.llm_grid_train import load_system_prompt; assert (Path(os.environ["RUN_DIR"]) / "artifacts/system_prompt.md").read_text(encoding="utf-8").rstrip("\n") == load_system_prompt(scale=2)'

python -m vlnce_baselines.models.etp_llm.llm_grid_navigation_cache \
  --model-name-or-path "$run_dir/checkpoints/epoch-2" \
  --cache-model-key "$cache_key" \
  --scope predictor-eval \
  --scale 2 \
  --batch-size 8 \
  --max-input-length 1024 \
  --max-new-tokens 4096

mkdir "$eval_dir"
python -m vlnce_baselines.models.etp_llm.llm_grid_eval \
  --cache-model-key "$cache_key" \
  --output-dir "$eval_dir"
