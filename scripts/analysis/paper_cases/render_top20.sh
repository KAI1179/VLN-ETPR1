#!/usr/bin/env bash
# Foreground, serial rendering of both splits' top20; no policy inference.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/paper-case-mpl
cd "$ROOT"
exec /home/xukai/anaconda3/envs/etpr1-py38/bin/python \
    "$ROOT/scripts/analysis/paper_cases/top20.py" --render
