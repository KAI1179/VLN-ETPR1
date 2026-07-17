#!/bin/bash
#SBATCH --job-name=llm-boxes-train-r2p5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/gpu-detection.bash"
configure_exact_distributed_gpu_vars 8

eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u

torchrun --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  -m vlnce_baselines.models.etp_llm.llm_boxes_train train \
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
  --cognitive-map-namespace gt.bbox.r2p5.path5.v1 \
  --output-dir outputs/llm_boxes/r2r-rxr-bbox-r2p5-path5-no-dataset-tag
