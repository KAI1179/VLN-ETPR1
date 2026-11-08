#!/usr/bin/env bash
#SBATCH --job-name=cogmap-g1-eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
# Evaluate G1 checkpoints on ln01: one job = one run, 4 GPUs, the listed iters
# dealt round-robin onto the GPUs (each GPU works through its share serially).
# Usage: sbatch scripts/submit/g1-eval-ln01.sh RUN_NAME ITER [ITER ...]
#   e.g. sbatch scripts/submit/g1-eval-ln01.sh dagger_g1_box_noreg 16000 19200 24000 30000
# Env: PRETRAINED_CKPT (default: the no-Gaussian PriorGT 440000 used by the G1 arms),
#      SPLIT (default val_unseen).
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run.py" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
cd "${REPO_ROOT}"

export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/data/home/scze096/run/ETP-R1/pretrained/r2r_rxr_ce/prior_gt_r1p5_nogaussian/ckpts/model_step_440000.pt}"
[[ -f "${PRETRAINED_CKPT}" ]] || { echo "PRETRAINED_CKPT not found: ${PRETRAINED_CKPT}" >&2; exit 1; }
# Slurm exposes the allocated GPUs through CUDA_VISIBLE_DEVICES; eval_iters.sh wants them as GPUS.
export GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export PYTHON="${PYTHON:-python}"

exec bash scripts/distill/eval_iters.sh "$@"
