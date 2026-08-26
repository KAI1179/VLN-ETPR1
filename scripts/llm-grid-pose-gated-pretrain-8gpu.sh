#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/gpu-detection.bash"

if [ "$#" -ne 1 ]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir="$1"
checkpoint="${repo_root}/pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt"
checkpoint_sha256="203fe62cc22c63261a5c5b6a3638bc52fd3b08a7f09dd31d8539bf2beab6c3cf"
spatial_cache="data/pose_gated/spatial_semantics_v2.hdf5"
target_namespace="gt.legacy.r1p5.direction5.v1"
llm_cache_key="llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"

cd "${repo_root}"

gpu_count="$(detect_gpu_count)"
if [ "${gpu_count}" -ne 8 ]; then
  echo "Expected exactly 8 visible GPUs, found ${gpu_count}. Set CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 2
fi
if [ -e "${output_dir}" ]; then
  echo "Output path already exists: ${output_dir}" >&2
  exit 2
fi
if [ ! -f "${checkpoint}" ]; then
  echo "Missing base checkpoint: ${checkpoint}" >&2
  exit 2
fi
if [ ! -f "${spatial_cache}" ]; then
  echo "Missing spatial visual cache: ${spatial_cache}" >&2
  exit 2
fi
if [ ! -d "data/cognitive_maps/${target_namespace}" ]; then
  echo "Missing target cognitive-map namespace: ${target_namespace}" >&2
  exit 2
fi
if [ ! -d "data/llm_navigation/${llm_cache_key}" ]; then
  echo "Missing LLM cognitive-map cache: ${llm_cache_key}" >&2
  exit 2
fi

export NUM_GPUS=8
bash pretrain_src/run_pt/run_mix_server.bash \
  "${output_dir}" \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --navigation-architecture try5 \
  --cognitive-map-source llm_grid \
  --llm-cache-model-key "${llm_cache_key}" \
  --pose-gated-map \
  --spatial-visual-cache "${spatial_cache}" \
  --target-cognitive-map-namespace "${target_namespace}" \
  --pose-gated-visual-loss-weight 1.0 \
  --pose-gated-route-loss-weight 1.0 \
  --pose-gated-seen-loss-weight 1.0 \
  --pose-gated-fix-loss-weight 1.0 \
  --pose-gated-keep-loss-weight 1.0 \
  --checkpoint "${checkpoint}" \
  --checkpoint-sha256 "${checkpoint_sha256}"
