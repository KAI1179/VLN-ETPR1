#!/usr/bin/env bash
# Same KD FULL evaluation; missing maps are zero inputs, not zero-score episodes.
set -euo pipefail
REPO=/home/xukai/code/ETP-R1-snapshot/ETP-R1
export GPUS="${GPUS:-0,1,2,3}"
port=29852
for split in val_seen val_unseen; do
    EXP_NAME="kd_full27000_control_${split}_missing_zero" MASTER_PORT="$port" \
        bash "$REPO/scripts/eval_d7/d7d_eval.sh" kd \
        EVAL.SPLIT "$split" EVAL.d7d_logging False \
        MODEL.MAP_ENCODER.llm_cache_dir "$REPO/outputs/kd27000_missing_zero_cache" \
        MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__missing-all-zero
    port=$((port + 1))
done
