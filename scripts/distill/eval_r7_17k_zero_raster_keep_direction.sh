#!/usr/bin/env bash
# Full val_unseen evaluation of R7 DAgger iter 17000 on GPUs 0,2,3,4.
# The built-in metadata_only ablation zeros only the cognitive-map raster.
# It keeps direction_vectors, start_direction_vector, and start_position.
# Usage: bash scripts/distill/eval_r7_17k_zero_raster_keep_direction.sh
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
PYTHON="/home/xukai/anaconda3/envs/etpr1-py38/bin/python"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
CKPT="${REPO_ROOT}/data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter17000.pth"
PRETRAINED="/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt"
EVAL_NAME="dagger_distill_gt_teacher_llmpt_eval_iter17000_raster_zero_keep_direction"
EVAL_DIR="${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}"
LOG="${EVAL_DIR}/eval.log"
RESULT="${EVAL_DIR}/eval_results/stats_ckpt_17000_val_unseen.json"

[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }
[[ -f "${PRETRAINED}" ]] || { echo "Missing pretraining checkpoint: ${PRETRAINED}" >&2; exit 1; }
if [[ -e "${RESULT}" ]]; then
    echo "Result already exists; refusing to overwrite: ${RESULT}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${EVAL_DIR}"
export CUDA_VISIBLE_DEVICES=0,2,3,4
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export LD_PRELOAD="${LD_PRELOAD:-/lib/x86_64-linux-gnu/libEGL_nvidia.so.0:/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0}"
RDZV_PORT="${MASTER_PORT:-$("${PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("localhost", 0)); print(s.getsockname()[1]); s.close()')}"

echo "Evaluating ${CKPT} on GPUs ${CUDA_VISIBLE_DEVICES}; log: ${LOG}"
GLOG_minloglevel=2 MAGNUM_LOG=quiet PYTHONPATH="${REPO_ROOT}" \
    "${TORCHRUN}" --rdzv_backend=c10d --rdzv_endpoint="localhost:${RDZV_PORT}" \
    --nproc_per_node=4 "${REPO_ROOT}/run.py" \
    --exp_name "${EVAL_NAME}" --run-type eval \
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS '[0,1,2,3]' TORCH_GPU_IDS '[0,1,2,3]' GPU_NUMBERS 4 \
    TASK_CONFIG.SEED 100 \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    NUM_ENVIRONMENTS 4 \
    EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 \
    EVAL.CKPT_PATH_DIR "${CKPT}" IL.back_algo control \
    TRAINER_NAME SS-ETP-LLM MODEL.policy_name LLMGridTry5Policy \
    MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source llm_grid \
    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
    MODEL.MAP_ENCODER.eval_map_source p0 MODEL.MAP_ENCODER.refiner_ckpt "" \
    MODEL.MAP_ENCODER.load_pretrained_map_modules False \
    MODEL.MAP_ENCODER.map_ablation metadata_only \
    MODEL.elevation_axis y MODEL.pretrained_path "${PRETRAINED}" \
    2>&1 | tee "${LOG}"

"${PYTHON}" - "${RESULT}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as result_file:
    metrics = json.load(result_file)
print(
    f"SR {metrics['success'] * 100:.2f} | "
    f"SPL {metrics['spl'] * 100:.2f} | "
    f"OSR {metrics['oracle_success'] * 100:.2f} | "
    f"NE {metrics['distance_to_goal']:.3f}"
)
print(f"Result: {sys.argv[1]}")
PY
