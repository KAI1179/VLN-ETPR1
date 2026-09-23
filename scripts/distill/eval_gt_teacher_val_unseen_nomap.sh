#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
RUN_NAME="gt_teacher_try5_val_unseen_nomap"
TEACHER_PRETRAINED="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt"
TEACHER_DAGGER="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth"
TEACHER_NAMESPACE="gt.legacy.r1p5.direction5.v1"

cd "${REPO_ROOT}"
env CUDA_VISIBLE_DEVICES=0 GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
    PYTHONPATH="${REPO_ROOT}" "${TORCHRUN}" --standalone --nproc_per_node=1 \
    "${REPO_ROOT}/run.py" \
    --exp_name "${RUN_NAME}" \
    --run-type eval \
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS '[0]' \
    TORCH_GPU_IDS '[0]' \
    GPU_NUMBERS 1 \
    NUM_ENVIRONMENTS 4 \
    TRAINER_NAME SS-ETP-PriorGT \
    MODEL.policy_name PriorGTTry5Policy \
    MODEL.MAP_ENCODER.enabled False \
    MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source prior_gt \
    MODEL.MAP_ENCODER.cache_namespace "${TEACHER_NAMESPACE}" \
    MODEL.MAP_ENCODER.refiner_ckpt "" \
    MODEL.pretrained_path "${TEACHER_PRETRAINED}" \
    EVAL.SPLIT val_unseen \
    EVAL.CKPT_PATH_DIR "${TEACHER_DAGGER}" \
    IL.back_algo teleport
