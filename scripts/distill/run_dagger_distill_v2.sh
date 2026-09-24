#!/usr/bin/env bash
# Distillation follow-up runs. Identical to run_dagger_distill.sh except:
#   z        : MODEL.elevation_axis z for the student (world-frame dz compass);
#              pretrained map modules NOT loaded (historical behaviour)
#   z_loader : same plus MODEL.MAP_ENCODER.load_pretrained_map_modules True
#              (pretrained map_encoder + graph_map_attention actually land)
#   loader   : elevation y (current) plus the loader fix only
# ITERS defaults to 12000 (the previous run peaked at 10k); checkpoints every
# 1000 iters. Run scripts/distill/check_pretrained_map_loading.py first.
#
# Usage: bash scripts/distill/run_dagger_distill_v2.sh <z|z_loader|loader> [--dry-run]
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
PRETRAINED_CKPT="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt"
GT_TEACHER_CKPT="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth"
GT_TEACHER_NAMESPACE="gt.online121c369.r1p5.direction5.v1"
GT_TEACHER_POLICY_NAME="PriorGTTry5Policy"

VARIANT="${1:-}"
case "${VARIANT}" in
    z)        ELEVATION_AXIS=z; LOAD_MAP=False ;;
    z_loader) ELEVATION_AXIS=z; LOAD_MAP=True ;;
    loader)   ELEVATION_AXIS=y; LOAD_MAP=True ;;
    *) echo "Usage: bash scripts/distill/run_dagger_distill_v2.sh <z|z_loader|loader> [--dry-run]" >&2; exit 2 ;;
esac
DRY_RUN_ARGS=()
if [[ "${2:-}" == "--dry-run" ]]; then DRY_RUN_ARGS=(--dry-run); fi

RUN_NAME="dagger_distill_gt_teacher_${VARIANT}"
RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
LOG_PATH="${RUN_DIR}/train.log"

COMMAND=(
    "${TORCHRUN}" --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:${MASTER_PORT:-29500} "${REPO_ROOT}/run.py"
    --exp_name "${RUN_NAME}"
    --run-type dagger
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml"
    "${DRY_RUN_ARGS[@]}"
    SIMULATOR_GPU_IDS '[0,1,2,3,4,5,6,7]'
    TORCH_GPU_IDS '[0,1,2,3,4,5,6,7]'
    GPU_NUMBERS 8
    TASK_CONFIG.SEED 100
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
    NUM_ENVIRONMENTS 4
    CHECKPOINT_INTERVAL 1000
    ONLY_LAST_SAVEALL False
    IL.iters "${ITERS:-12000}"
    IL.lr 1e-5
    IL.log_every 200
    IL.ml_weight 1.0
    IL.sample_ratio 0.75
    IL.decay_interval 1400
    IL.warmup_iters 300
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
    MODEL.MAP_ENCODER.load_pretrained_map_modules "${LOAD_MAP}"
    MODEL.elevation_axis "${ELEVATION_AXIS}"
    MODEL.pretrained_path "${PRETRAINED_CKPT}"
    IL.gt_teacher_enabled True
    IL.distill_weight 1.0
    IL.distill_temperature 1.0
    IL.gt_teacher_ckpt "${GT_TEACHER_CKPT}"
    IL.gt_teacher_map_namespace "${GT_TEACHER_NAMESPACE}"
    IL.gt_teacher_policy_name "${GT_TEACHER_POLICY_NAME}"
    IL.gt_teacher_elevation_axis z
)

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"
echo "Variant: ${VARIANT} (elevation_axis=${ELEVATION_AXIS}, load_pretrained_map_modules=${LOAD_MAP}, iters=${ITERS:-12000})"
echo "Log: ${LOG_PATH}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    PYTHONPATH="${REPO_ROOT}" "${COMMAND[@]}" 2>&1 | tee "${LOG_PATH}"
if [[ ${#DRY_RUN_ARGS[@]} -ne 0 ]]; then echo "Dry run complete"; fi
