#!/usr/bin/env bash
# Unattended D9: fail immediately on a mismatch; never overlap inference jobs.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
export GPUS="${GPUS:-4,5,6,7}"
export D9_DUMP_ROOT="${D9_DUMP_ROOT:-$ROOT/data/logs/d9_dump}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$ROOT/reports/d9"
exec 9>"$ROOT/reports/d9/run.lock"
flock -n 9 || { echo 'Another D9 pipeline owns run.lock' >&2; exit 1; }
STAGE=initializing
write_status() {
    "$PYTHON" -c 'import datetime,json,os,pathlib,sys
pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(status=sys.argv[2],stage=sys.argv[3],gpus=sys.argv[4],pid=os.getppid(),updated_at=datetime.datetime.now().astimezone().isoformat(),exit_code=int(sys.argv[5])),indent=2)+"\n")' \
        "$ROOT/reports/d9/status.json" "$1" "$STAGE" "$GPUS" "${2:-0}"
}
trap 'code=$?; if [[ $code == 0 ]]; then write_status completed "$code"; else write_status failed "$code"; fi; echo "$(date -Is) D9 pipeline exited with status $code"; exit "$code"' EXIT
write_status running

wait_for_gpus() {
    while ! nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
        "$PYTHON" -c 'import sys
requested={int(x) for x in sys.argv[1].split(",")}
rows={int(v[0]): (int(v[1]),int(v[2])) for v in (line.strip().split(",") for line in sys.stdin) if v[0].strip()}
if requested-rows.keys():
    raise ValueError("Requested GPU does not exist")
sys.exit(0 if all(rows[g][0]>=16000 and rows[g][1]<10 for g in requested) else 1)' "$GPUS"; do
        echo "$(date -Is) Waiting for GPUs $GPUS: each needs >=16000 MiB free and <10% utilization"
        write_status waiting_for_gpus
        sleep 60
    done
    write_status running
}

SMOKE_IDS="$ROOT/scripts/eval_d8/smoke_episodes.json"
for MODEL in base try5 kd; do
    STAGE="smoke_${MODEL}_O"
    REPORT="$ROOT/reports/d9/${MODEL}_smoke_equivalence.json"
    if [[ ! -f "$REPORT" ]]; then
        for HOOK in off on; do
            wait_for_gpus
            EXTRA=()
            [[ "$HOOK" == on ]] || EXTRA=(EVAL.dump_attention '')
            D9_EPISODES_JSON="$SMOKE_IDS" \
            D9_DUMP_ROOT="$ROOT/data/logs/d9_smoke/$HOOK" \
            EXP_NAME="d9_smoke_${MODEL}_${HOOK}" \
                bash "$ROOT/scripts/eval_d9/dump.sh" "$MODEL" O "${EXTRA[@]}"
        done
    fi
    "$PYTHON" "$ROOT/scripts/eval_d9/verify.py" --mode metrics --expected 4 \
        --reference-metrics "$ROOT/data/logs/checkpoints/d9_smoke_${MODEL}_off/eval_results" \
        --candidate-metrics "$ROOT/data/logs/checkpoints/d9_smoke_${MODEL}_on/eval_results" \
        --output "$REPORT"
    "$PYTHON" "$ROOT/scripts/eval_d9/verify.py" --mode dump --model "$MODEL" --source O \
        --directory "$ROOT/data/logs/d9_smoke/on/$MODEL/O" --episodes-json "$SMOKE_IDS"
    STAGE="smoke_${MODEL}_S"
    if [[ ! -f "$ROOT/data/logs/d9_smoke/on/$MODEL/S/.complete.json" ]]; then
        wait_for_gpus
        D9_EPISODES_JSON="$SMOKE_IDS" \
        D9_DUMP_ROOT="$ROOT/data/logs/d9_smoke/on" EXP_NAME="d9_smoke_${MODEL}_S" \
            bash "$ROOT/scripts/eval_d9/dump.sh" "$MODEL" S
    fi
    "$PYTHON" "$ROOT/scripts/eval_d9/verify.py" --mode dump --model "$MODEL" --source S \
        --directory "$ROOT/data/logs/d9_smoke/on/$MODEL/S" --episodes-json "$SMOKE_IDS" \
        --reference-dump "$ROOT/data/logs/d8_dump/$MODEL/S"
done

for SOURCE in S O; do
    for MODEL in base try5 kd; do
        STAGE="dump_${MODEL}_${SOURCE}"
        DIRECTORY="$D9_DUMP_ROOT/$MODEL/$SOURCE"
        if [[ ! -f "$DIRECTORY/.complete.json" ]]; then
            wait_for_gpus
            echo "$(date -Is) Starting D9 $MODEL/$SOURCE on GPUs $GPUS"
            bash "$ROOT/scripts/eval_d9/dump.sh" "$MODEL" "$SOURCE"
        fi
        REFERENCE=()
        [[ "$SOURCE" != S ]] || REFERENCE=(--reference-dump "$ROOT/data/logs/d8_dump/$MODEL/S")
        "$PYTHON" "$ROOT/scripts/eval_d9/verify.py" --mode dump --model "$MODEL" --source "$SOURCE" \
            --directory "$DIRECTORY" "${REFERENCE[@]}"
        if [[ "$SOURCE" == O ]]; then
            "$PYTHON" "$ROOT/scripts/eval_d9/verify.py" --mode metrics --expected 1839 \
                --reference-metrics "$ROOT/data/logs/checkpoints/d8_${MODEL}_O_control_val_unseen/eval_results" \
                --candidate-metrics "$ROOT/data/logs/checkpoints/d9_${MODEL}_O_control_val_unseen/eval_results" \
                --output "$ROOT/reports/d9/${MODEL}_own_equivalence.json"
        fi
    done
done
STAGE=cpu_attention_statistics
write_status running
"$PYTHON" "$ROOT/scripts/distill/attention_stats.py" --dump-root "$D9_DUMP_ROOT"
echo "$(date -Is) D9 completed: $ROOT/reports/d9/README.md"
