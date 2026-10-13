#!/usr/bin/env bash
# One model inference at a time. Shared states first; no background jobs.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
[[ $# -ge 2 ]] || { echo 'Usage: dump.sh base|try5|kd S|O [config overrides...]' >&2; exit 2; }
MODEL="$1"; SOURCE="$2"; shift 2
DUMP_ROOT="${DUMP_ROOT:-$ROOT/data/logs/d8_dump}"
DUMP_DIR="$DUMP_ROOT/$MODEL/$SOURCE"
case "$SOURCE" in
    S)
        SHARED=True; REPLAY=''
        [[ "$MODEL" == base ]] || REPLAY="$DUMP_ROOT/base/S"
        IDS="$($PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print(repr([int(k) for k,v in d["per_episode"].items() if v["ours_category"] != "U"]))' "$ROOT/reports/d7d_kd_control.json")"
        ;;
    O) SHARED=False; REPLAY=''; IDS='[]' ;;
    *) echo "Unknown source: $SOURCE" >&2; exit 2 ;;
esac
if [[ -n "${D8_EPISODES_JSON:-}" ]]; then
    IDS="$($PYTHON -c 'import json,sys; print(repr(json.load(open(sys.argv[1]))))' "$D8_EPISODES_JSON")"
fi
case "$MODEL" in base) PORT=29790 ;; try5) PORT=29791 ;; kd) PORT=29792 ;; *) exit 2 ;; esac
EXP_NAME="${EXP_NAME:-d8_${MODEL}_${SOURCE}_control_val_unseen}"
export EXP_NAME
MASTER_PORT="${MASTER_PORT:-$PORT}" GPUS="${GPUS:-4,5,6,7}" \
    bash "$ROOT/scripts/eval_d7/d7d_eval.sh" "$MODEL" \
    EVAL.dump_decision_features "$DUMP_DIR" EVAL.d8_shared_states "$SHARED" \
    EVAL.d8_replay_dir "$REPLAY" EVAL.EPISODE_ID "$IDS" "$@"
