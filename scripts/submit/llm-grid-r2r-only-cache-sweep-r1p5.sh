#!/bin/bash
#SBATCH --job-name=llm-grid-r2r-cache-sweep
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=1-10%10
#SBATCH --gpus=1
#SBATCH -p vip_gpu_scze096
set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u

run_dir=outputs/llm_grid/r2r-legacy-r1p5-direction5-scale2
epoch="$SLURM_ARRAY_TASK_ID"
cache_key="llm-grid-r2r-legacy-r1p5-direction5-scale2-epoch-${epoch}"
system_prompt_path="${run_dir}/artifacts/system_prompt.md"

SYSTEM_PROMPT_PATH="$system_prompt_path" python -c \
  'import hashlib, os; from pathlib import Path; prompt = Path(os.environ["SYSTEM_PROMPT_PATH"]).read_text(encoding="utf-8").strip(); assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == "6de023cedc1409ce82c5d93d52083b8b980f788b380ea475c69451370cf28409"'

python -m vlnce_baselines.models.etp_llm.llm_grid_navigation_cache \
  --model-name-or-path "${run_dir}/checkpoints/epoch-${epoch}" \
  --cache-model-key "$cache_key" \
  --scope predictor-eval \
  --prompt-contract r2r-legacy-v1 \
  --system-prompt-path "$system_prompt_path" \
  --scale 2 \
  --batch-size 8 \
  --max-input-length 1024 \
  --max-new-tokens 4096
