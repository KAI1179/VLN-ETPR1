#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
: "${GPUS:?Set GPUS to the GPUs authorized for evaluation}"
IFS=, read -ra IDS <<< "$GPUS"
[[ ${#IDS[@]} -eq 4 ]] || { echo 'Use four GPUs to match the final evaluation rank layout' >&2; exit 2; }
export GPUS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for split in val_unseen val_seen; do
    for model in ours zeromap; do
        bash "$ROOT/scripts/analysis/paper_cases/eval_one.sh" "$model" "$split"
    done
done
echo 'Batch complete. Confirm the attn_case.pdf episode IDs, then run select_cases.py --exclude ID1 ID2.'
