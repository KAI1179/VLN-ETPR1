#!/usr/bin/env bash
# D9 eval-only extraction, one foreground inference at a time.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
[[ $# -ge 2 ]] || { echo 'Usage: dump.sh base|try5|kd S|O [config overrides...]' >&2; exit 2; }
MODEL="$1"; SOURCE="$2"; shift 2
export D9_DUMP_ROOT="${D9_DUMP_ROOT:-$ROOT/data/logs/d9_dump}"
D8_ROOT="$ROOT/data/logs/d8_dump"
case "$SOURCE" in
    S)
        SHARED=True; REPLAY="$D8_ROOT/base/S"
        IDS="$("$PYTHON" -c 'import pathlib,sys; print(repr(sorted(int(p.stem) for p in pathlib.Path(sys.argv[1]).glob("*.npz"))))' "$REPLAY")" ;;
    O) SHARED=False; REPLAY=''; IDS='[]' ;;
    *) echo "Unknown source: $SOURCE" >&2; exit 2 ;;
esac
if [[ -n "${D9_EPISODES_JSON:-}" ]]; then
    IDS="$("$PYTHON" -c 'import json,sys; print(repr(json.load(open(sys.argv[1]))))' "$D9_EPISODES_JSON")"
fi
case "$MODEL" in base) PORT=29830 ;; try5) PORT=29831 ;; kd) PORT=29832 ;; *) exit 2 ;; esac
export EXP_NAME="${EXP_NAME:-d9_${MODEL}_${SOURCE}_control_val_unseen}"
MASTER_PORT="${MASTER_PORT:-$PORT}" GPUS="${GPUS:-4,5,6,7}" \
    bash "$ROOT/scripts/eval_d7/d7d_eval.sh" "$MODEL" \
    EVAL.dump_attention "$D9_DUMP_ROOT/$MODEL/$SOURCE" \
    EVAL.d8_shared_states "$SHARED" EVAL.d8_replay_dir "$REPLAY" \
    EVAL.EPISODE_ID "$IDS" "$@"
