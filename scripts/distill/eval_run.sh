#!/usr/bin/env bash
# Evaluate one DAgger checkpoint of any LLM-Grid Try5 run on R2R val_unseen.
#
# Usage: bash scripts/distill/eval_run.sh RUN_NAME ITER [PRETRAINED_CKPT]
# Env: CUDA_VISIBLE_DEVICES (default 0; several ids = several processes),
#      NUM_ENVS (4), BACK_ALGO (control), SPLIT (val_unseen).
# Results: data/logs/checkpoints/<RUN_NAME>_eval_iter<ITER>_<SPLIT>/eval_results/
# (the layout summarize_eval.py and paired_analysis.py read).
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
PYTHON="/home/xukai/anaconda3/envs/etpr1-py38/bin/python"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
# torchrun --standalone binds a fixed rendezvous port on torch 2.1, so concurrent
# evals collide; pick a free port instead (override with MASTER_PORT).
free_port() { "${PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1]); s.close()'; }
RDZV_PORT="${MASTER_PORT:-$(free_port)}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash scripts/distill/eval_run.sh RUN_NAME ITER [PRETRAINED_CKPT]" >&2
    exit 2
fi
RUN_NAME="$1"; ITER="$2"
PRETRAINED_CKPT="${3:-/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
GPU_NUMBERS="${#_gpu_list[@]}"
GPU_IDS="[$(seq -s, 0 $((GPU_NUMBERS - 1)))]"
NUM_ENVS="${NUM_ENVS:-4}"
BACK_ALGO="${BACK_ALGO:-control}"
SPLIT="${SPLIT:-val_unseen}"

RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
CKPT_PATH="${RUN_DIR}/ckpt.iter${ITER}.pth"
EVAL_NAME="${RUN_NAME}_eval_iter${ITER}_${SPLIT}"
EVAL_LOG="${RUN_DIR}/eval_iter${ITER}_${SPLIT}.log"
RESULT_JSON="${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}/eval_results/stats_ckpt_${ITER}_${SPLIT}.json"
[[ -f "${CKPT_PATH}" ]] || { echo "Checkpoint not found: ${CKPT_PATH}" >&2; exit 1; }

MAP_LOAD_ARGS=()
if grep -q "load_pretrained_map_modules" "${REPO_ROOT}/vlnce_baselines/config/default.py"; then
    MAP_LOAD_ARGS=(MODEL.MAP_ENCODER.load_pretrained_map_modules False)
fi

cd "${REPO_ROOT}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
    PYTHONPATH="${REPO_ROOT}" "${TORCHRUN}" --rdzv_backend=c10d --rdzv_endpoint="localhost:${RDZV_PORT}" --nproc_per_node="${GPU_NUMBERS}" \
    "${REPO_ROOT}/run.py" \
    --exp_name "${EVAL_NAME}" \
    --run-type eval \
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS "${GPU_IDS}" TORCH_GPU_IDS "${GPU_IDS}" GPU_NUMBERS "${GPU_NUMBERS}" \
    TASK_CONFIG.SEED 100 \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    NUM_ENVIRONMENTS "${NUM_ENVS}" \
    EVAL.SPLIT "${SPLIT}" EVAL.EPISODE_COUNT -1 \
    EVAL.CKPT_PATH_DIR "${CKPT_PATH}" \
    IL.back_algo "${BACK_ALGO}" \
    TRAINER_NAME SS-ETP-LLM \
    MODEL.policy_name LLMGridTry5Policy \
    MODEL.MAP_ENCODER.enabled True \
    MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source llm_grid \
    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
    MODEL.MAP_ENCODER.refiner_ckpt "" \
    "${MAP_LOAD_ARGS[@]}" \
    MODEL.elevation_axis "${ELEVATION_AXIS:-y}" \
    MODEL.pretrained_path "${PRETRAINED_CKPT}" 2>&1 | tee "${EVAL_LOG}"

"${PYTHON}" - "${RESULT_JSON}" "${RUN_NAME}" "${ITER}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"\n{sys.argv[2]} iter {sys.argv[3]} | SR {m['success']*100:.2f} | SPL {m['spl']*100:.2f} | "
      f"OSR {m['oracle_success']*100:.2f} | NE {m['distance_to_goal']:.3f} | "
      f"stop_err {(m['oracle_success']-m['success'])*100:.2f} | cache_missing {int(m.get('llm_cache_missing_count',0))}")
PY
echo "Metrics: ${RESULT_JSON}"
