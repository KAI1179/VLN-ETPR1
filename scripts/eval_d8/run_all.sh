#!/usr/bin/env bash
# Strictly serial: never overlap model inference jobs on the selected GPUs.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
export GPUS="${GPUS:-4,5,6,7}"
for SOURCE in S O; do
    for MODEL in base try5 kd; do
        echo "D8 $MODEL $SOURCE"
        if [[ -f "$ROOT/data/logs/d8_dump/$MODEL/$SOURCE/.complete.json" ]]; then
            echo 'Already complete; validating existing files'
        else
            bash "$ROOT/scripts/eval_d8/dump.sh" "$MODEL" "$SOURCE"
        fi
        if [[ "$SOURCE" == O ]]; then
            "$PYTHON" "$ROOT/scripts/eval_d8/verify.py" \
                --reference "$ROOT/data/logs/checkpoints/d7d_${MODEL}_control_val_unseen/eval_results" \
                --candidate "$ROOT/data/logs/checkpoints/d8_${MODEL}_O_control_val_unseen/eval_results" \
                --output "$ROOT/reports/d8/${MODEL}_own_equivalence.json"
        fi
        "$PYTHON" "$ROOT/scripts/eval_d8/validate_features.py" \
            --directory "$ROOT/data/logs/d8_dump/$MODEL/$SOURCE" --model "$MODEL" --source "$SOURCE"
    done
    if [[ "$SOURCE" == S ]]; then
        "$PYTHON" "$ROOT/scripts/distill/progress_probe.py" --check-shared-only
    fi
done
