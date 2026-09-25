#!/usr/bin/env bash
# Does the distilled student still read a map? Evaluate one distilled
# checkpoint with its map input swapped at eval time (S5-D1 protocol):
#   p0      : the LLM map it was trained on (control; plain code path)
#   gt_full : the whole GT map (gt.legacy.r1p5.direction5.v1)
#   gt_seen : GT only in observed cells, LLM elsewhere (needs semantic sensors)
#
# Usage: bash scripts/distill/eval_student_gt_full.sh [MODE] [ITER] [RUN_NAME]
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
MODE="${1:-gt_full}"
ITER="${2:-10000}"
RUN_NAME="${3:-dagger_distill_gt_teacher}"
case "${MODE}" in p0|gt_full|gt_seen) ;; *) echo "Unknown MODE ${MODE}" >&2; exit 2 ;; esac

cd "${REPO_ROOT}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"
export LD_PRELOAD="${LD_PRELOAD:-/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0}"
# The distill runs were initialised from the GT-line pretraining checkpoint;
# keep pretrained_path consistent (it only seeds shapes before ckpt load).
env PRETRAINED_CKPT="${PRETRAINED_CKPT:-/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt}" \
    LOAD_PRETRAINED_MAP_MODULES=False \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    bash scripts/refiner/eval_s4.sh "${RUN_NAME}" "${ITER}" "${MODE}"
