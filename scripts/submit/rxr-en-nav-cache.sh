#!/usr/bin/env bash
#SBATCH --job-name=rxr-en-nav-cache
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=1
#SBATCH -p vip_gpu_scze096
# RxR-English LLM navigation cache (train / val_seen / val_unseen), one shard
# per job.  Sharding is a deterministic hash of the cache id (empty shard
# seed), so N jobs with WORKER_INDEX 0..N-1 and the same WORKER_COUNT cover
# every episode exactly once and can be submitted at the same time:
#
#   for k in 0 1 2 3; do
#     sbatch --export=ALL,WORKER_COUNT=4,WORKER_INDEX=$k scripts/submit/rxr-en-nav-cache.sh
#   done
#
# Default model: the LoRA the R2R cache under the same model key was generated
# from on this machine (scripts/submit/llm-grid-nav-cache-r1p5.sh).  Re-running
# a shard skips episodes whose cache is already complete.
set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
MODEL="${LLM_LORA:-outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag/checkpoints/final}"
[[ -d "${MODEL}" ]] || { echo "LLM checkpoint not found: ${MODEL} (set LLM_LORA)" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1)}"
PYTHONPATH="${REPO_ROOT}" python scripts/distill/cache_rxr_en_llm_grid.py \
  --model-name-or-path "${MODEL}" \
  --cache-model-key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
  --scale 2 --batch-size "${BATCH_SIZE:-8}" --max-new-tokens 4096 \
  --worker-count "${WORKER_COUNT:-1}" --worker-index "${WORKER_INDEX:-0}"
