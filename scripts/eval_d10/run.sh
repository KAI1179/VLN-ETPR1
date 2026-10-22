#!/usr/bin/env bash
# User-run only: strictly serial smoke checks, own evaluation, and S extraction.
set -euo pipefail
ROOT="${D10_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export D10_ROOT="$ROOT"
export D10_PYTHON="${D10_PYTHON:-/home/xukai/anaconda3/envs/etpr1-py38/bin/python}"
PYTHON="$D10_PYTHON"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export D10_DUMP_ROOT="${D10_DUMP_ROOT:-$ROOT/data/logs/d10_dump}"
REPORT="$ROOT/reports/d10"
M="$ROOT/data/logs/d9_reference_layout/try5/S"
IFS=, read -ra GPU_LIST <<< "${GPUS:-${CUDA_VISIBLE_DEVICES:-7}}"
mkdir -p "$REPORT"
exec 9>"$REPORT/run.lock"
flock -n 9 || { echo 'Another D10 run owns run.lock' >&2; exit 1; }

for mode in smoke_off smoke_on; do
    bash "$ROOT/scripts/eval_d10/dump_rank.sh" "$mode" 0 "${GPU_LIST[0]}"
done
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$ROOT/scripts/eval_d10/verify.py" --mode smoke \
    --reference-metrics "$ROOT/data/logs/checkpoints/d10_zero_smoke_off_rank0/eval_results" \
    --candidate-metrics "$ROOT/data/logs/checkpoints/d10_zero_smoke_on_rank0/eval_results" \
    --output "$REPORT/hook_equivalence.json"
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$ROOT/scripts/eval_d10/verify.py" --mode alignment \
    --directory "$D10_DUMP_ROOT/smoke/on/attention" --reference-dump "$M" --partial \
    --output "$REPORT/smoke_alignment.json"

for rank in 0 1 2 3; do
    gpu="${GPU_LIST[$((rank % ${#GPU_LIST[@]}))]}"
    bash "$ROOT/scripts/eval_d10/dump_rank.sh" O "$rank" "$gpu"
done
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$ROOT/scripts/eval_d10/summarize_own.py"

for rank in 0 1 2 3; do
    gpu="${GPU_LIST[$((rank % ${#GPU_LIST[@]}))]}"
    bash "$ROOT/scripts/eval_d10/dump_rank.sh" S "$rank" "$gpu"
    CUDA_VISIBLE_DEVICES='' "$PYTHON" "$ROOT/scripts/eval_d10/verify.py" --mode alignment \
        --directory "$D10_DUMP_ROOT/zero/S" --reference-dump "$M" --partial \
        --output "$REPORT/alignment_partial.json"
done
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$ROOT/scripts/eval_d10/verify.py" --mode alignment \
    --directory "$D10_DUMP_ROOT/zero/S" --reference-dump "$M" --output "$REPORT/alignment.json"

# Attention analysis is CPU-only and stays single-threaded.
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$ROOT/scripts/eval_d10/checks.py" \
    --zero-root "$D10_DUMP_ROOT/zero" --map-root "$ROOT/data/logs/d9_reference_layout/try5" \
    --checkpoint "${D10_M_CKPT:-/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth}" \
    --output "$REPORT"
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$ROOT/scripts/eval_d10/stats.py" \
    --zero-dir "$D10_DUMP_ROOT/zero/S" --map-dir "$M" \
    --checks "$REPORT/checks.json" --output "$REPORT"
