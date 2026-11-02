#!/usr/bin/env bash
# User-run watcher; serial evaluations only. No GPU process starts until invoked.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
[[ $# -ge 2 ]] || { echo "Usage: $0 {region|full} ITER [ITER ...]" >&2; exit 2; }
ARM="$1"; shift
case "${ARM}" in
  region) TRAIN=dagger_llm_region_only ;;
  full) TRAIN=dagger_llm_full_control ;;
  *) echo 'ARM must be region or full' >&2; exit 2 ;;
esac
for ITER in "$@"; do
  [[ "${ITER}" =~ ^[0-9]+$ ]] || { echo "Invalid iteration: ${ITER}" >&2; exit 2; }
done
while true; do
  DONE=true
  for ITER in "$@"; do
    CKPT="${ROOT}/data/logs/checkpoints/${TRAIN}/ckpt.iter${ITER}.pth"
    OUT="${ROOT}/data/logs/checkpoints/d3_region_eval"
    if [[ -f "${OUT}/${TRAIN}_iter${ITER}_own/done" && \
          -f "${OUT}/${TRAIN}_iter${ITER}_cross_scene/done" && \
          -f "${OUT}/${TRAIN}_iter${ITER}_zero_grid/done" ]]; then
      continue
    fi
    DONE=false
    [[ -f "${CKPT}" ]] || continue
    BEFORE="$(stat -c %s "${CKPT}")"
    sleep 120
    AFTER="$(stat -c %s "${CKPT}")"
    [[ "${BEFORE}" == "${AFTER}" ]] || continue
    bash "${ROOT}/scripts/eval_d3/run_iter.sh" "${ARM}" "${ITER}"
  done
  [[ "${DONE}" == true ]] && exit 0
  sleep 600
done
