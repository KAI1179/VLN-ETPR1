#!/usr/bin/env bash
#SBATCH --job-name=d10-map-attn
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH -p vip_gpu_scze096
set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate "${D10_CONDA_ENV:-etpr1-uv}"
set -u
export D10_ROOT="${D10_ROOT:-$SLURM_SUBMIT_DIR}"
export D10_PYTHON="$(command -v python)"
# On a remote server, export D10_ZERO_CKPT, D10_PRETRAINED, and D10_M_CKPT
# to their uploaded locations. D8 replay and D9 M/S data must also be present.
# Keep the scheduler's CUDA_VISIBLE_DEVICES rather than choosing another GPU.
export GPUS="${CUDA_VISIBLE_DEVICES:?Slurm must provide a GPU allocation}"
bash "$D10_ROOT/scripts/eval_d10/run.sh"
