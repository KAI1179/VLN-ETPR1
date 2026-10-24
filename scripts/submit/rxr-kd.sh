#!/usr/bin/env bash
#SBATCH --job-name=rxr-kd
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
# RxR-CE final KD student on ln01; SETTING=en (English guide) or all (all languages):
#   for s in en all; do sbatch --export=ALL,SETTING=$s -J rxr-kd-$s scripts/submit/rxr-kd.sh; done
set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
# sbatch runs a spooled copy of this file: submit from the repo root.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f "${REPO_ROOT}/run.py" ]] || { echo "submit from the repo root (got ${REPO_ROOT})" >&2; exit 1; }
export TORCHRUN="$(command -v torchrun)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
bash "${REPO_ROOT}/scripts/distill/run_rxr_kd.sh" "${SETTING:?SETTING=en or all}"
