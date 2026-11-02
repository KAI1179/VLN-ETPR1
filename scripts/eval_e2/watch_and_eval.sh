#!/usr/bin/env bash
# User-run watcher: sequentially evaluate stable E2 checkpoints.
set -euo pipefail

ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
TRAIN="${ROOT}/data/logs/checkpoints/dagger_coordfusion_llm"
OUT="${ROOT}/data/logs/checkpoints/e2_coordfusion_val_unseen"
EVAL="${ROOT}/scripts/eval_e2/eval_one.sh"
TARGETS=(8000 12000 16000 20000 24000 28000 30000)
if [[ -f "${TRAIN}/ckpt.iter4000.pth" ]]; then
  TARGETS=(4000 "${TARGETS[@]}")
else
  echo "iter4000 is already past and absent; not substituting iter1200."
fi
mkdir -p "${OUT}"

while true; do
  for iter in "${TARGETS[@]}"; do
    ckpt="${TRAIN}/ckpt.iter${iter}.pth"
    [[ -f "${ckpt}" ]] || continue
    [[ -f "${OUT}/iter${iter}.done" ]] && continue
    before="$(stat -c %s "${ckpt}")"
    sleep 120
    after="$(stat -c %s "${ckpt}")"
    [[ "${before}" == "${after}" ]] || { echo "Still writing ${ckpt}"; continue; }
    bash "${EVAL}" "${iter}" own
    bash "${EVAL}" "${iter}" cross_scene
    touch "${OUT}/iter${iter}.done"
  done

  all_done=true
  for iter in "${TARGETS[@]}"; do
    [[ -f "${OUT}/iter${iter}.done" ]] || all_done=false
  done
  if [[ "${all_done}" == true || -f "${OUT}/iter30000.done" ]]; then
    echo "Target evaluations complete (or final iter30000 evaluated)."
    exit 0
  fi
  sleep 600
done
