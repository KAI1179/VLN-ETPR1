#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
CKPT="${REPO_ROOT}/data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/store/ckpt.iter28000.pth"

cd "${REPO_ROOT}"
env CUDA_VISIBLE_DEVICES=0 GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
    PYTHONPATH="${REPO_ROOT}" "${TORCHRUN}" --standalone --nproc_per_node=1 \
    "${REPO_ROOT}/run.py" \
    --exp_name llm_try5_control_val_unseen \
    --run-type eval \
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS '[0]' \
    TORCH_GPU_IDS '[0]' \
    GPU_NUMBERS 1 \
    NUM_ENVIRONMENTS 4 \
    TRAINER_NAME SS-ETP-LLM \
    MODEL.policy_name LLMGridTry5Policy \
    MODEL.MAP_ENCODER.enabled True \
    MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source llm_grid \
    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
    MODEL.MAP_ENCODER.refiner_ckpt "" \
    MODEL.pretrained_path /data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt \
    EVAL.SPLIT val_unseen \
    EVAL.CKPT_PATH_DIR "${CKPT}" \
    IL.back_algo teleport
