#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 OUTPUT_DIR [BASE_CHECKPOINT]" >&2
  exit 2
fi

OUTPUT_DIR="$1"
BASE_CHECKPOINT="${2:-pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt}"
MODEL_KEY="llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
TARGET_NAMESPACE="gt.legacy.r1p5.direction5.v1"
VISUAL_EVIDENCE_CACHE="${ONLINE_FUSION_VISUAL_EVIDENCE_CACHE:-data/online_fusion/panorama_semantics_v1.hdf5}"
CACHE_ROOT="data/llm_navigation/${MODEL_KEY}/pretrain/mixed/cognitive_maps/raster"
TARGET_ROOT="data/cognitive_maps/${TARGET_NAMESPACE}"

if [ ! -f "${BASE_CHECKPOINT}" ]; then
  echo "Missing base checkpoint: ${BASE_CHECKPOINT}" >&2
  exit 2
fi
if [ ! -d "${CACHE_ROOT}" ]; then
  echo "Missing LLM-Grid pretraining cache: ${CACHE_ROOT}" >&2
  exit 2
fi
if [ ! -d "${TARGET_ROOT}" ]; then
  echo "Missing cognitive-map target namespace: ${TARGET_ROOT}" >&2
  exit 2
fi
if [ ! -f "${VISUAL_EVIDENCE_CACHE}" ]; then
  echo "Missing panorama semantic evidence: ${VISUAL_EVIDENCE_CACHE}" >&2
  echo "Generate it first with: bash scripts/local/generate-online-fusion-visual-evidence.sh ${VISUAL_EVIDENCE_CACHE}" >&2
  exit 2
fi
if [ -d "${OUTPUT_DIR}" ] && [ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "Output directory is not empty: ${OUTPUT_DIR}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
exec bash pretrain_src/run_pt/run_mix_server.bash \
  "${OUTPUT_DIR}" \
  --navigation-architecture online_fusion \
  --cognitive-map-source llm_grid \
  --llm-cache-model-key "${MODEL_KEY}" \
  --cognitive-map-target-namespace "${TARGET_NAMESPACE}" \
  --visual-evidence-cache "${VISUAL_EVIDENCE_CACHE}" \
  --online-visual-loss-weight 0.2 \
  --optimizer-profile online_fusion \
  --checkpoint "${BASE_CHECKPOINT}"
