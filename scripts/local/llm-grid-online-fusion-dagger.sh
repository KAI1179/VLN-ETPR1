#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 PRETRAINED_CHECKPOINT EXP_NAME" >&2
  exit 2
fi

PRETRAINED_CHECKPOINT="$1"
EXP_NAME="$2"
MODEL_KEY="llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
CACHE_ROOT="data/llm_navigation/${MODEL_KEY}/r2r/train/cognitive_maps/raster"
TARGET_ROOT="data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster"
OUTPUT_ROOT="data/logs/checkpoints/${EXP_NAME}"

case "${EXP_NAME}" in
  ""|*[!a-zA-Z0-9._-]*)
    echo "EXP_NAME must contain only letters, digits, dot, underscore or hyphen" >&2
    exit 2
    ;;
esac

if [ ! -f "${PRETRAINED_CHECKPOINT}" ]; then
  echo "Missing OnlineFusion pretraining checkpoint: ${PRETRAINED_CHECKPOINT}" >&2
  exit 2
fi
if [ ! -d "${CACHE_ROOT}" ]; then
  echo "Missing LLM-Grid DAgger cache: ${CACHE_ROOT}" >&2
  exit 2
fi
if [ ! -d "${TARGET_ROOT}" ]; then
  echo "Missing cognitive-map GT targets: ${TARGET_ROOT}" >&2
  exit 2
fi
if [ -d "${OUTPUT_ROOT}" ] && [ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "DAgger output directory is not empty: ${OUTPUT_ROOT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export LLM_GRID_ONLINE_FUSION_PRETRAINED_CKPT="${PRETRAINED_CHECKPOINT}"
export LLM_GRID_ONLINE_FUSION_EXP_NAME="${EXP_NAME}"
exec bash run_r2r/main_server.bash llm_grid_online_fusion_dagger
