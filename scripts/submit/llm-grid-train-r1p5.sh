#!/bin/bash
#SBATCH --job-name=llm-grid-train-r1p5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/gpu-detection.bash"

eval "$(conda shell.bash hook)"
conda activate etpr1-uv
configure_distributed_gpu_vars
set -u

torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node="${NPROC_PER_NODE}" \
  -m vlnce_baselines.models.etp_llm.llm_grid_train train \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --gradient-checkpointing \
  --device-map none \
  --max-input-length 1152 \
  --max-new-tokens 4096 \
  --epochs 10 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --cognitive-map-namespace gt.legacy.r1p5.direction5.v1 \
  --output-dir outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag
