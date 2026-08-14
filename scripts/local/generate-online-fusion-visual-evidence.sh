#!/bin/bash
set -euo pipefail

OUTPUT_FILE="${1:-data/online_fusion/panorama_semantics_v1.hdf5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

exec python -m vlnce_baselines.models.etp_prior_gt.panorama_semantic_cache \
  --output-file "${OUTPUT_FILE}" \
  --gpu-device-id 0
