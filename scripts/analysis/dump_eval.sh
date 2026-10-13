#!/usr/bin/env bash
# Decision dump for the paired-episode analysis, R2R val_unseen.
#
# Usage: bash scripts/analysis/dump_eval.sh MODEL PROTOCOL
#   MODEL     base | try5 | kd -- checkpoints and every evaluation setting come
#             from scripts/eval_d7/d7d_eval.sh unchanged (baseline iter25000 no
#             map; try5 no-KD iter28000 and KD FULL iter27000 with their own
#             LLM map; control, 4 envs/rank, 150 tokens, 15 high-level steps).
#   PROTOCOL  own     the policy drives (protocol 1).  A normal evaluation with
#                     the read-only dump; afterwards checked episode by episode
#                     against the existing D7-d evaluation of the same model.
#             shared  every model sees the same states (protocol 2), via the D8
#                     mechanism: base walks the SPL expert and saves its states,
#                     try5 and kd replay those actions with a per-step check of
#                     position and candidate graph.  Run base first.  Episode
#                     set = D7-d non-U episodes (1833), as in D8.
# Env: GPUS (default 4,5,6,7), MASTER_PORT.  One model job at a time.
# Output: data/logs/checkpoints/pa_<MODEL>_<PROTOCOL>_val_unseen/eval_results/
#           decisions_val_unseen_r*_w*.jsonl, stats_ep_*.json, stats_ckpt_*.json
#         shared protocol also: data/logs/pa_shared/<MODEL>/*.npz (D8 format)
# Nothing else is touched; existing d7d_*/d8_* results are only read.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python

[[ $# -eq 2 ]] || { echo "Usage: bash scripts/analysis/dump_eval.sh base|try5|kd own|shared" >&2; exit 2; }
MODEL="$1"; PROTOCOL="$2"
case "$MODEL" in base) PORT=29820 ;; try5) PORT=29821 ;; kd) PORT=29822 ;;
    *) echo "MODEL must be base, try5 or kd" >&2; exit 2 ;; esac

EXP_NAME="pa_${MODEL}_${PROTOCOL}_val_unseen"
RESULT_DIR="$ROOT/data/logs/checkpoints/$EXP_NAME/eval_results"
SHARED_ROOT="$ROOT/data/logs/pa_shared"
if compgen -G "$RESULT_DIR/decisions_val_unseen_r*.jsonl" >/dev/null; then
    echo "Dump already exists in $RESULT_DIR; delete that directory to redo." >&2; exit 1
fi

EXTRA=()
case "$PROTOCOL" in
    own) ;;
    shared)
        PORT=$((PORT + 3))
        IDS="$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(repr([int(k) for k,v in d["per_episode"].items() if v["ours_category"] != "U"]))' "$ROOT/reports/d7d_kd_control.json")"
        FEATURE_DIR="$SHARED_ROOT/$MODEL"
        if compgen -G "$FEATURE_DIR/*.npz" >/dev/null; then
            echo "$FEATURE_DIR already holds state files; delete it to redo." >&2; exit 1
        fi
        REPLAY=""
        if [[ "$MODEL" != base ]]; then
            REPLAY="$SHARED_ROOT/base"
            compgen -G "$ROOT/data/logs/checkpoints/pa_base_shared_val_unseen/eval_results/stats_ckpt_*_val_unseen.json" >/dev/null \
                || { echo "Run 'dump_eval.sh base shared' to completion first (it creates the shared states)." >&2; exit 1; }
        fi
        EXTRA=(EVAL.d8_shared_states True EVAL.d8_replay_dir "$REPLAY"
               EVAL.dump_decision_features "$FEATURE_DIR" EVAL.EPISODE_ID "$IDS")
        ;;
    *) echo "PROTOCOL must be own or shared" >&2; exit 2 ;;
esac

echo "$(date -Is) model=$MODEL protocol=$PROTOCOL -> $RESULT_DIR"
EXP_NAME="$EXP_NAME" MASTER_PORT="${MASTER_PORT:-$PORT}" GPUS="${GPUS:-4,5,6,7}" \
    bash "$ROOT/scripts/eval_d7/d7d_eval.sh" "$MODEL" \
    EVAL.d7d_logging False EVAL.decision_dump True ${EXTRA[@]+"${EXTRA[@]}"}

"$PYTHON" - "$RESULT_DIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "decisions_val_unseen_r*.jsonl")))
steps = eps = 0
for f in files:
    for line in open(f):
        k = json.loads(line)["kind"]
        steps += k == "step"; eps += k == "episode"
print(f"dump: {len(files)} file(s), {eps} episodes, {steps} steps in {d}")
PY

if [[ "$PROTOCOL" == own ]]; then
    REFERENCE="$ROOT/data/logs/checkpoints/d7d_${MODEL}_control_val_unseen/eval_results"
    if [[ -d "$REFERENCE" ]]; then
        mkdir -p "$ROOT/reports/analysis"
        # Exact per-episode identity of success / OSR / SPL / steps / NE with
        # the D7-d run: proves the dump does not change the rollout.
        "$PYTHON" "$ROOT/scripts/eval_d8/verify.py" --reference "$REFERENCE" --candidate "$RESULT_DIR" \
            --output "$ROOT/reports/analysis/pa_${MODEL}_own_equivalence.json"
    else
        echo "No D7-d reference at $REFERENCE; equivalence check skipped."
    fi
fi
