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
# Default to every GPU nvidia-smi can see instead of assuming eight cards.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${_n}" -gt 0 ]]; then
        CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_n - 1)))"
    else
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    fi
fi
# GPU layout is derived from CUDA_VISIBLE_DEVICES: N visible cards -> N processes
# with logical ids 0..N-1. NUM_ENVS is the per-process environment count.
IFS=',' read -ra _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
GPU_NUMBERS="${#_gpu_list[@]}"
GPU_IDS="[$(seq -s, 0 $((GPU_NUMBERS - 1)))]"
NUM_ENVS="${NUM_ENVS:-4}"
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
    "${TORCHRUN}" --nproc_per_node="${GPU_NUMBERS}" --rdzv_backend=c10d --rdzv_endpoint=localhost:${MASTER_PORT:-29500} "${REPO_ROOT}/run.py"
    --exp_name "${RUN_NAME}"
    --run-type dagger
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml"
    "${DRY_RUN_ARGS[@]}"
    SIMULATOR_GPU_IDS "${GPU_IDS}"
    TORCH_GPU_IDS "${GPU_IDS}"
    GPU_NUMBERS "${GPU_NUMBERS}"
    TASK_CONFIG.SEED 100
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
    NUM_ENVIRONMENTS "${NUM_ENVS}"
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
# Habitat needs the NVIDIA EGL vendor file; a shell without it fails at env
# construction. Same default as the eval launchers.
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"

# Name the first module that emits NaN/+inf during the first N policy forwards
# (the hooks remove themselves afterwards; 0 disables them).
ETP_NAN_DEBUG_STEPS="${ETP_NAN_DEBUG_STEPS:-1000}"
echo "Variant: ${VARIANT} (elevation_axis=${ELEVATION_AXIS}, load_pretrained_map_modules=${LOAD_MAP}, iters=${ITERS:-12000})"
echo "Log: ${LOG_PATH}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    ETP_NAN_DEBUG_STEPS="${ETP_NAN_DEBUG_STEPS}" \
    PYTHONPATH="${REPO_ROOT}" "${COMMAND[@]}" 2>&1 | tee "${LOG_PATH}"
if [[ ${#DRY_RUN_ARGS[@]} -ne 0 ]]; then echo "Dry run complete"; fi
