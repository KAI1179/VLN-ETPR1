#!/usr/bin/env bash
#SBATCH --job-name=d8-dump
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH -p vip_gpu_scze096
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/xukai/code/ETP-R1-snapshot/ETP-R1}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:?Slurm must expose allocated GPUs}}"
else
    export GPUS="${GPUS:-4,5,6,7}"
fi
# run_all.sh launches one model inference at a time; shared states are
# validated before starting any own-rollout dump.
bash "${REPO_ROOT}/scripts/eval_d8/run_all.sh"
