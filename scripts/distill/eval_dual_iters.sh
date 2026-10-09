#!/usr/bin/env bash
# Evaluate several checkpoints of a dual-branch distillation run on R2R
# val_unseen in both views -- the deployment view (raster zeroed, metadata
# kept: map_ablation metadata_only) and the map view (map_ablation none) --
# and print one table with the paired delta.
#
# Usage: bash scripts/distill/eval_dual_iters.sh RUN_NAME ITER [ITER ...]
# Env: GPUS            comma-separated physical GPU ids (default 4,5,6,7); the
#                      (iter, view) jobs are dealt round-robin, each GPU works
#                      through its share one after another (one process, 4 envs,
#                      ~15 min per job);
#      VIEWS           space-separated subset of "metadata_only none"
#                      (default both);
#      PRETRAINED_CKPT the student's pretraining checkpoint (shapes only; the
#                      DAgger checkpoint overrides every weight) -- default the
#                      460000 LLM-grid pretraining the dual run starts from.
# Already-evaluated (iter, view) pairs are skipped.
# Results: data/logs/checkpoints/<RUN_NAME>_eval_iter<ITER>_abl_<VIEW>/eval_results/
# (eval_student_ablation.sh layout; paired_analysis.py reads it).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/xukai/code/ETP-R1-snapshot/ETP-R1}"
PYTHON="${PYTHON:-/home/xukai/anaconda3/envs/etpr1-py38/bin/python}"
EVAL_ONE="${REPO_ROOT}/scripts/distill/eval_student_ablation.sh"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt}"

[[ $# -ge 2 ]] || { echo "Usage: bash scripts/distill/eval_dual_iters.sh RUN_NAME ITER [ITER ...]" >&2; exit 2; }
RUN_NAME="$1"; shift
RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
[[ -d "${RUN_DIR}" ]] || { echo "Run directory not found: ${RUN_DIR}" >&2; exit 1; }
[[ -f "${PRETRAINED_CKPT}" ]] || { echo "PRETRAINED_CKPT not found: ${PRETRAINED_CKPT}" >&2; exit 1; }
IFS=',' read -ra GPU_LIST <<< "${GPUS:-4,5,6,7}"
read -ra VIEW_LIST <<< "${VIEWS:-metadata_only none}"

result_json() { echo "${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}_eval_iter$1_abl_$2/eval_results/stats_ckpt_$1_val_unseen.json"; }

# Build the job list (iter view), skipping finished ones and missing ckpts.
JOBS=()
for ITER in "$@"; do
    [[ -f "${RUN_DIR}/ckpt.iter${ITER}.pth" ]] || { echo "skip iter ${ITER}: no checkpoint" >&2; continue; }
    for VIEW in "${VIEW_LIST[@]}"; do
        if [[ -f "$(result_json "${ITER}" "${VIEW}")" ]]; then echo "skip iter ${ITER} ${VIEW}: already evaluated"; continue; fi
        JOBS+=("${ITER} ${VIEW}")
    done
done
echo "$(date -Is) run=${RUN_NAME} gpus=${GPU_LIST[*]} jobs=${#JOBS[@]}"

# Deal the jobs round-robin onto the GPUs; each GPU runs its share sequentially.
PIDS=()
for g in "${!GPU_LIST[@]}"; do
    (
        for j in "${!JOBS[@]}"; do
            (( j % ${#GPU_LIST[@]} == g )) || continue
            read -r ITER VIEW <<< "${JOBS[$j]}"
            echo "[gpu ${GPU_LIST[$g]}] $(date -Is) start iter ${ITER} ${VIEW}"
            if CUDA_VISIBLE_DEVICES="${GPU_LIST[$g]}" bash "${EVAL_ONE}" "${RUN_NAME}" "${ITER}" "${VIEW}" >/dev/null 2>&1; then
                echo "[gpu ${GPU_LIST[$g]}] $(date -Is) done  iter ${ITER} ${VIEW}"
            else
                echo "[gpu ${GPU_LIST[$g]}] $(date -Is) FAILED iter ${ITER} ${VIEW} (see ${RUN_DIR}/eval_iter${ITER}_abl_${VIEW}.log)"
            fi
        done
    ) &
    PIDS+=($!)
done
wait "${PIDS[@]}"

# Summary: one row per iter, both views, paired SR delta (bootstrap CI) when both exist.
SUMMARY="${RUN_DIR}/eval_dual_summary.md"
{
echo "| iter | deploy SR | SPL | OSR | NE | map SR | SPL | OSR | NE | map - deploy SR [95% CI] |"
echo "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
for ITER in "$@"; do
    "${PYTHON}" - "${ITER}" "$(result_json "${ITER}" metadata_only)" "$(result_json "${ITER}" none)" <<'PY'
import json, os, sys
it, dep, mp = sys.argv[1:4]
def cell(p):
    if not os.path.isfile(p): return "missing | | |"
    m = json.load(open(p))
    return f"{m['success']*100:.2f} | {m['spl']*100:.2f} | {m['oracle_success']*100:.2f} | {m['distance_to_goal']:.2f}"
print(f"| {it} | {cell(dep)} | {cell(mp)} | ", end="")
PY
    if [[ -f "$(result_json "${ITER}" metadata_only)" && -f "$(result_json "${ITER}" none)" ]]; then
        "${PYTHON}" "${REPO_ROOT}/scripts/distill/paired_analysis.py" \
            --dir-a "$(dirname "$(result_json "${ITER}" metadata_only)")" \
            --dir-b "$(dirname "$(result_json "${ITER}" none)")" 2>/dev/null \
            | grep -m1 -E '^\| *SR ' | awk -F'|' '{gsub(/^ +| +$/,"",$5); gsub(/^ +| +$/,"",$6); print $5 " " $6 " |"}' || echo "n/a |"
    else
        echo "|"
    fi
done
} | tee "${SUMMARY}"
echo "Summary: ${SUMMARY}"
