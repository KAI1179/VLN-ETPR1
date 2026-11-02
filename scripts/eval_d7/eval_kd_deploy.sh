#!/usr/bin/env bash
# D7: D5 deployment protocol, four ranks, val_unseen.
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
CKPT="${REPO_ROOT}/data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter17000.pth"
EVAL_NAME="d7_kd_deploy_val_unseen"
EVAL_DIR="${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}"
GPUS="${GPUS:-4,5,6,7}"
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
[[ ${#GPU_LIST[@]} -eq 4 ]] || { echo "D7 requires four GPUs" >&2; exit 2; }
[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }

cd "${REPO_ROOT}"
mkdir -p "${EVAL_DIR}"
env CUDA_VISIBLE_DEVICES="${GPUS}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
    PYTHONPATH="${REPO_ROOT}" "${TORCHRUN}" --master_port 29680 --nproc_per_node=4 \
    "${REPO_ROOT}/run.py" \
    --exp_name "${EVAL_NAME}" \
    --run-type eval \
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS '[0,1,2,3]' \
    TORCH_GPU_IDS '[0,1,2,3]' \
    GPU_NUMBERS 4 \
    NUM_ENVIRONMENTS 4 \
    TRAINER_NAME SS-ETP-LLM \
    MODEL.policy_name LLMGridTry5Policy \
    MODEL.MAP_ENCODER.enabled True \
    MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source llm_grid \
    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__deploy-zero \
    MODEL.MAP_ENCODER.refiner_ckpt "" \
    MODEL.MAP_ENCODER.load_pretrained_map_modules False \
    MODEL.pretrained_path "/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt" \
    EVAL.SPLIT val_unseen \
    EVAL.CKPT_PATH_DIR "${CKPT}" \
    IL.back_algo teleport \
    2>&1 | tee "${EVAL_DIR}/eval.log"
