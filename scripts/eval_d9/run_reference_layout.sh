#!/usr/bin/env bash
# Restore original D8 logical rank batches; at most one inference per GPU.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
export D9_DUMP_ROOT="${D9_DUMP_ROOT:-$ROOT/data/logs/d9_reference_layout}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
GPUS="${GPUS:-4,5}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
[[ ${#GPU_LIST[@]} == 2 ]] || { echo 'Exactly two GPUs required' >&2; exit 2; }
mkdir -p "$ROOT/reports/d9"
exec 9>"$ROOT/reports/d9/run.lock"
flock -n 9 || { echo 'Another D9 pipeline owns run.lock' >&2; exit 1; }
status() {
    "$PYTHON" -c 'import datetime,json,os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(status=sys.argv[2],stage=sys.argv[3],pid=os.getppid(),gpus=sys.argv[4],num_environments_per_rank=4,reference_world_size=4,updated_at=datetime.datetime.now().astimezone().isoformat()),indent=2)+"\n")' \
        "$ROOT/reports/d9/status.json" "$1" "${STAGE:-initializing}" "$GPUS"
}
trap 'code=$?; if [[ $code == 0 ]]; then status completed; else status failed; fi; exit "$code"' EXIT
wait_gpu() {
    while ! nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
        "$PYTHON" -c 'import sys; rows={int(a):int(b) for a,b in (line.strip().split(",") for line in sys.stdin)}; sys.exit(0 if rows[int(sys.argv[1])] >= 10000 else 1)' "$1"; do
        echo "$(date -Is) Waiting for GPU $1: >=10000 MiB free"
        sleep 60
    done
}
run_rank() {
    local model="$1" source="$2" rank="$3" gpu="$4"
    local directory="$D9_DUMP_ROOT/$model/$source"
    if [[ -f "$directory/.rank$rank.complete.json" ]]; then
        D9_REFERENCE_RANK="$rank" "$PYTHON" "$ROOT/scripts/eval_d9/run_shard.py" --verify-rank "$model" "$source" "$directory" \
            "$ROOT/data/logs/checkpoints/d9_reference_${model}_${source}_rank${rank}/eval_results"
        return
    fi
    wait_gpu "$gpu"
    bash "$ROOT/scripts/eval_d9/dump_shard.sh" "$model" "$source" "$rank" "$gpu"
}
for SOURCE in S O; do
    for MODEL in base try5 kd; do
        STAGE="reference_${MODEL}_${SOURCE}"
        status running
        for FIRST in 0 2; do
            run_rank "$MODEL" "$SOURCE" "$FIRST" "${GPU_LIST[0]}" & PID0=$!
            run_rank "$MODEL" "$SOURCE" "$((FIRST+1))" "${GPU_LIST[1]}" & PID1=$!
            RESULT=0
            wait "$PID0" || RESULT=$?
            wait "$PID1" || RESULT=$?
            [[ "$RESULT" == 0 ]] || exit "$RESULT"
        done
        REFERENCE=()
        [[ "$SOURCE" != S ]] || REFERENCE=(--reference-dump "$ROOT/data/logs/d8_dump/$MODEL/S")
        "$PYTHON" "$ROOT/scripts/eval_d9/verify.py" --mode dump --model "$MODEL" --source "$SOURCE" \
            --directory "$D9_DUMP_ROOT/$MODEL/$SOURCE" "${REFERENCE[@]}"
    done
done
STAGE=cpu_attention_statistics
status running
"$PYTHON" "$ROOT/scripts/distill/attention_stats.py" --dump-root "$D9_DUMP_ROOT"
