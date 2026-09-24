#!/usr/bin/env bash
# Evaluate one DAgger distillation checkpoint on R2R val_unseen.
#
# Usage:
#   bash scripts/distill/eval_dagger_distill.sh [--dry-run] [ITER]
#
# Defaults: ITER=6000, GPUs 1 and 6 (override with CUDA_VISIBLE_DEVICES),
# one environment per process (smallest batch), IL.back_algo=control.
# Student-side settings mirror scripts/distill/run_dagger_distill.sh; the
# frozen GT teacher is never built in eval, so no IL.gt_teacher_* keys are set.
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
PYTHON="/home/xukai/anaconda3/envs/etpr1-py38/bin/python"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"

DRY_RUN_ARGS=()
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_ARGS=(--dry-run)
    shift
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/distill/eval_dagger_distill.sh [--dry-run] [ITER]" >&2
    exit 2
fi
ITER="${1:-6000}"

# GPU selection: physical ids in CUDA_VISIBLE_DEVICES map to logical 0..N-1.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,6}"
IFS=',' read -ra _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
GPU_NUMBERS="${#_gpu_list[@]}"
GPU_IDS="[$(seq -s, 0 $((GPU_NUMBERS - 1)))]"
NUM_ENVS="${NUM_ENVS:-1}"          # per-process environments; 1 = smallest batch
BACK_ALGO="${BACK_ALGO:-control}"  # matches run_r2r/main_server.bash eval modes
SPLIT="${SPLIT:-val_unseen}"

TRAIN_RUN_NAME="dagger_distill_gt_teacher"
TRAIN_RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${TRAIN_RUN_NAME}"
CKPT_PATH="${TRAIN_RUN_DIR}/ckpt.iter${ITER}.pth"
EVAL_NAME="${TRAIN_RUN_NAME}_eval_iter${ITER}_${SPLIT}"
EVAL_LOG="${TRAIN_RUN_DIR}/eval_iter${ITER}_${SPLIT}.log"
RESULT_JSON="${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}/eval_results/stats_ckpt_${ITER}_${SPLIT}.json"

# Same pretrained backbone as run_dagger_distill.sh; the DAgger checkpoint
# above is loaded on top of it through EVAL.CKPT_PATH_DIR.
PRETRAINED_CKPT="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt"
LLM_CACHE_MODEL_KEY="llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"

if [[ ${#DRY_RUN_ARGS[@]} -eq 0 && ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint not found: ${CKPT_PATH}" >&2
    echo "Available checkpoints:" >&2
    ls -1 "${TRAIN_RUN_DIR}"/ckpt.iter*.pth 2>/dev/null >&2 || true
    exit 1
fi

COMMAND=(
    "${TORCHRUN}" --standalone --nproc_per_node="${GPU_NUMBERS}" "${REPO_ROOT}/run.py"
    --exp_name "${EVAL_NAME}"
    --run-type eval
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml"
    "${DRY_RUN_ARGS[@]}"
    SIMULATOR_GPU_IDS "${GPU_IDS}"
    TORCH_GPU_IDS "${GPU_IDS}"
    GPU_NUMBERS "${GPU_NUMBERS}"
    TASK_CONFIG.SEED 100
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
    NUM_ENVIRONMENTS "${NUM_ENVS}"
    EVAL.SPLIT "${SPLIT}"
    EVAL.EPISODE_COUNT -1
    EVAL.CKPT_PATH_DIR "${CKPT_PATH}"
    IL.back_algo "${BACK_ALGO}"
    TRAINER_NAME SS-ETP-LLM
    MODEL.policy_name LLMGridTry5Policy
    MODEL.MAP_ENCODER.enabled True
    MODEL.MAP_ENCODER.architecture try5
    MODEL.MAP_ENCODER.source llm_grid
    MODEL.MAP_ENCODER.llm_cache_model_key "${LLM_CACHE_MODEL_KEY}"
    MODEL.MAP_ENCODER.refiner_ckpt ""
    MODEL.pretrained_path "${PRETRAINED_CKPT}"
)

mkdir -p "${TRAIN_RUN_DIR}"
cd "${REPO_ROOT}"

echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${GPU_NUMBERS} processes x ${NUM_ENVS} env)"
echo "Checkpoint: ${CKPT_PATH}"
echo "Log: ${EVAL_LOG}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}" \
    PYTHONPATH="${REPO_ROOT}" "${COMMAND[@]}" 2>&1 | tee "${EVAL_LOG}"

if [[ ${#DRY_RUN_ARGS[@]} -ne 0 ]]; then
    echo "Dry run complete"
    echo "Config: ${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}/config.yaml"
    exit 0
fi

if [[ ! -f "${RESULT_JSON}" ]]; then
    echo "Result file not found: ${RESULT_JSON}" >&2
    exit 1
fi

"${PYTHON}" - "${RESULT_JSON}" "${ITER}" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1]))
iteration = sys.argv[2]

def pct(key):
    return f"{metrics[key] * 100:.2f}" if key in metrics else "n/a"

def raw(key, fmt="{:.3f}"):
    return fmt.format(metrics[key]) if key in metrics else "n/a"

print()
print(f"iter {iteration} | SR {pct('success')} | SPL {pct('spl')} | "
      f"OSR {pct('oracle_success')} | NE {raw('distance_to_goal')} | "
      f"nDTW {pct('ndtw')} | SDTW {pct('sdtw')}")
if "llm_cache_missing_count" in metrics:
    print(f"llm_cache_missing: {metrics['llm_cache_missing_count']} "
          f"({metrics.get('llm_cache_missing_rate', 0.0) * 100:.2f}%)")
PY

echo "Metrics: ${RESULT_JSON}"
