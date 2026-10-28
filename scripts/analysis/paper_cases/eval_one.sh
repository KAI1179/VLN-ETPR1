#!/usr/bin/env bash
# Logged own-rollout evaluation; model/trainer source remains untouched.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
MODEL="${1:?Usage: eval_one.sh ours|zeromap val_unseen|val_seen}"
SPLIT="${2:?Supply split}"
: "${GPUS:?Set GPUS to the GPUs authorized for this evaluation}"
[[ "$SPLIT" == val_unseen || "$SPLIT" == val_seen ]] || exit 2
case "$MODEL" in
    ours)
        LAUNCH=kd
        PORT=29910
        EXTRA=(MODEL.MAP_ENCODER.llm_cache_dir "$ROOT/outputs/kd27000_missing_zero_cache"
               MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__missing-all-zero) ;;
    zeromap)
        LAUNCH=try5
        PORT=29912
        CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$ROOT/scripts/analysis/paper_cases/make_zero_cache.py"
        EXTRA=(EVAL.CKPT_PATH_DIR /data/xukai/etp-r1-snapshot/checkpoints/llm-grid-000-baseline-dagger-28k/ckpt.iter28000.pth
               MODEL.pretrained_path "$ROOT/pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt"
               MODEL.MAP_ENCODER.llm_cache_dir "$ROOT/outputs/paper_case_zero_cache"
               MODEL.MAP_ENCODER.llm_cache_model_key paper.case.zero_all_inputs.v1) ;;
    *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
[[ "$SPLIT" == val_seen ]] && PORT=$((PORT + 1))
NAME="paper_case_${MODEL}_${SPLIT}"
DIRECTORY="$ROOT/data/logs/checkpoints/$NAME/eval_results"
if compgen -G "$DIRECTORY/decisions_*.jsonl" >/dev/null; then
    echo "Existing decision logs: $DIRECTORY; refusing overwrite" >&2
    exit 1
fi
EXP_NAME="$NAME" MASTER_PORT="${PAPER_CASE_PORT:-$PORT}" \
    bash "$ROOT/scripts/eval_d7/d7d_eval.sh" "$LAUNCH" \
    EVAL.SPLIT "$SPLIT" EVAL.d7d_logging False EVAL.decision_dump True \
    MODEL.MAP_ENCODER.input_form raster MODEL.MAP_ENCODER.map_ablation none "${EXTRA[@]}"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
    "$ROOT/scripts/analysis/paper_cases/collect.py" \
    --model "$MODEL" --split "$SPLIT" --eval-dir "$DIRECTORY"
