#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
RUN_NAME="dagger_distill_gt_teacher_smoke"
RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
LOG_PATH="${RUN_DIR}/train.log"
CONFIG_PATH="${RUN_DIR}/config.yaml"
PRETRAINED_CKPT="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt"
GT_TEACHER_CKPT="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth"
GT_TEACHER_NAMESPACE="gt.online121c369.r1p5.direction5.v1"
GT_TEACHER_POLICY_NAME="PriorGTTry5Policy"

DRY_RUN_ARGS=()
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_ARGS=(--dry-run)
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/distill/run_dagger_distill.sh [--dry-run]" >&2
    exit 2
fi

COMMAND=(
    "${TORCHRUN}" --standalone --nproc_per_node=1 "${REPO_ROOT}/run.py"
    --exp_name "${RUN_NAME}"
    --run-type dagger
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml"
    "${DRY_RUN_ARGS[@]}"
    SIMULATOR_GPU_IDS '[0]'
    TORCH_GPU_IDS '[0]'
    GPU_NUMBERS 1
    TASK_CONFIG.SEED 100
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
    NUM_ENVIRONMENTS 1
    CHECKPOINT_INTERVAL 20
    ONLY_LAST_SAVEALL False
    IL.iters 20
    IL.lr 1e-5
    IL.log_every 5
    IL.ml_weight 1.0
    IL.sample_ratio 0.75
    IL.decay_interval 2000
    IL.warmup_iters 500
    IL.min_lr_ratio 1.0
    IL.load_from_ckpt False
    IL.is_requeue False
    IL.waypoint_aug True
    TASK_CONFIG.DATASET.SUFFIX _90
    TRAINER_NAME SS-ETP-LLM
    MODEL.policy_name LLMGridTry5Policy
    MODEL.MAP_ENCODER.enabled True
    MODEL.MAP_ENCODER.architecture try5
    MODEL.MAP_ENCODER.source llm_grid
    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
    MODEL.MAP_ENCODER.refiner_ckpt ""
    MODEL.pretrained_path "${PRETRAINED_CKPT}"
    IL.gt_teacher_enabled True
    IL.distill_weight 1.0
    IL.distill_temperature 1.0
    IL.gt_teacher_ckpt "${GT_TEACHER_CKPT}"
    IL.gt_teacher_map_namespace "${GT_TEACHER_NAMESPACE}"
    IL.gt_teacher_policy_name "${GT_TEACHER_POLICY_NAME}"
)

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

echo "Log: ${LOG_PATH}"
echo "Config: ${CONFIG_PATH}"
env CUDA_VISIBLE_DEVICES=0 GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
    LD_PRELOAD=/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0 \
    PYTHONPATH="${REPO_ROOT}" "${COMMAND[@]}" 2>&1 | tee "${LOG_PATH}"

if [[ ${#DRY_RUN_ARGS[@]} -ne 0 ]]; then
    echo "Dry run complete"
fi
