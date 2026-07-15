#!/bin/bash
set -euo pipefail

export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export LD_PRELOAD="${LD_PRELOAD:-/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/gpu-detection.bash"

CACHE_ALIAS="${1:-llm5}"

case "${CACHE_ALIAS}" in
      llm4)
      CACHE_MODEL_KEY="${LLM4_CACHE_MODEL_KEY:-llm4}"
      ;;
      llm5)
      CACHE_MODEL_KEY="${LLM5_CACHE_MODEL_KEY:-llm5}"
      ;;
      *)
      CACHE_MODEL_KEY="${CACHE_ALIAS}"
      ;;
esac

configure_distributed_gpu_vars

EXP_CONFIG="run_r2r/iter_train.yaml"
MAP_NUM_ENVS=4
GT_PRETRAINED_CKPT="pretrained/r2r_rxr_ce/prior_gt/store2/try7_step_435000.pt"
GT_DAGGER_CKPT="data/logs/checkpoints/release_r2r_priorgt_dagger/store/try7.iter29600.pth"
EXP_NAME="smoke_r2r_llm_map_try7_${CACHE_MODEL_KEY}"

COMMON_ARGS="--exp-config ${EXP_CONFIG}
      SIMULATOR_GPU_IDS ${GPU_IDS}
      TORCH_GPU_IDS ${GPU_IDS}
      GPU_NUMBERS ${GPU_NUMBERS}
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True"

LLM_TRY7_ARGS="TRAINER_NAME SS-ETP-LLM
      MODEL.policy_name LLMPolicy
      MODEL.MAP_ENCODER.enabled True
      MODEL.MAP_ENCODER.llm_cache_model_key ${CACHE_MODEL_KEY}
      MODEL.pretrained_path ${GT_PRETRAINED_CKPT}"

echo "###### LLM map + Try7 smoke eval ######"
echo "cache_model_key=${CACHE_MODEL_KEY}"
echo "pretrained_ckpt=${GT_PRETRAINED_CKPT}"
echo "dagger_ckpt=${GT_DAGGER_CKPT}"
echo "exp_name=${EXP_NAME}"

cd "${REPO_ROOT}"
torchrun --standalone \
      --nproc-per-node="${NPROC_PER_NODE}" \
      run.py \
      --exp_name "${EXP_NAME}" \
      --run-type eval \
      ${COMMON_ARGS} \
      NUM_ENVIRONMENTS ${MAP_NUM_ENVS} \
      ${LLM_TRY7_ARGS} \
      EVAL.CKPT_PATH_DIR "${GT_DAGGER_CKPT}" \
      IL.back_algo control
