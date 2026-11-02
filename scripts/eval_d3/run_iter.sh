#!/usr/bin/env bash
# Manually evaluate one checkpoint on own, cross-scene donor, and zero raster.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
[[ $# -eq 2 ]] || { echo "Usage: $0 {region|full} ITER" >&2; exit 2; }
ARM="$1"; ITER="$2"
if [[ "${ARM}" == region ]]; then
  SOURCE=llm.region_only.s2tagfree.v1
elif [[ "${ARM}" == full ]]; then
  SOURCE=llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
else
  echo 'ARM must be region or full' >&2; exit 2
fi
DONOR="${ROOT}/data/llm_navigation/${SOURCE}__raster-cross_scene-s0/r2r/val_unseen/manifest.json"
if [[ ! -f "${DONOR}" ]]; then
  cd "${ROOT}"
  /home/xukai/anaconda3/envs/etpr1-py38/bin/python \
    "${ROOT}/scripts/distill/make_raster_donor_cache.py" \
    --mode cross_scene --seed 0 --source-model-key "${SOURCE}" --split val_unseen
fi
for variant in own cross_scene zero_grid; do
  bash "${ROOT}/scripts/eval_d3/eval_one.sh" "${ARM}" "${ITER}" "${variant}"
done
