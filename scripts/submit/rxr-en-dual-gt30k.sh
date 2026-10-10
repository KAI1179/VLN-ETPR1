#!/usr/bin/env bash
#SBATCH --job-name=rxr-en-dual-gt30k
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH -p vip_gpu_scze096
set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TORCHRUN="$(command -v torchrun)"
# These are cloud paths: supply both with sbatch --export=ALL,...
: "${GT_TEACHER_CKPT:?Provide uploaded joint GT 30000 checkpoint}"
: "${PRETRAINED_CKPT:?Provide uploaded LLM 460000 checkpoint}"
bash "${REPO_ROOT}/scripts/distill/run_rxr_en_dual_gt30k.sh"
