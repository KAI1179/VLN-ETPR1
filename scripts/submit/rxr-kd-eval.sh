#!/usr/bin/env bash
#SBATCH --job-name=rxr-kd-eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096
# Evaluation of RxR KD checkpoints in one 8-GPU job.  Default: one checkpoint
# at a time over all 8 GPUs.  To fill the GPUs, run several at once:
#   GPUS_PER_EVAL=1 EVALS_PER_GPU=2 NUM_ENVS=8  -> 16 checkpoints concurrently
# (see eval_rxr_kd_iters.sh; NUM_ENVS <= 11 on val_unseen).
#   sbatch --export=ALL,SETTING=en,ITERS=all      -J rxr-kd-eval-en  scripts/submit/rxr-kd-eval.sh
#   sbatch --export=ALL,SETTING=all,ITERS=every:1000 -J rxr-kd-eval-all scripts/submit/rxr-kd-eval.sh
#   ITERS="4000 5000 5200"  explicit list;  SPLIT, EVAL_LANGUAGES, NUM_ENVS,
#   GPUS_PER_EVAL, EVALS_PER_GPU pass through (--export=ALL,...).
set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/run.py" ]] || { echo "submit from the repo root (got ${REPO_ROOT})" >&2; exit 1; }
export TORCHRUN="$(command -v torchrun)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
# shellcheck disable=SC2086
bash "${REPO_ROOT}/scripts/distill/eval_rxr_kd_iters.sh" "${SETTING:?SETTING=en or all}" ${ITERS:?ITERS=all | every:N | list}
