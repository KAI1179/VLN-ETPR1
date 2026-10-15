#!/usr/bin/env bash
#SBATCH --job-name=d9-dump
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH -p vip_gpu_scze096
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:?Slurm must expose allocated GPUs}}"
else
    export GPUS="${GPUS:-4,5,6,7}"
fi
bash "$ROOT/scripts/eval_d9/run_all.sh"
