#!/usr/bin/env bash
# GT-teacher distillation with the student initialised from the LLM-Grid
# pretraining checkpoint (model_step_460000.pt), with automatic resume.
#
# Usage: bash scripts/distill/run_dagger_distill_llmpt.sh [--dry-run]
#
# Resume: if data/logs/checkpoints/<RUN_NAME>/ckpt.iter*.pth exists the run
# continues from the latest one (IL.load_from_ckpt True + IL.is_requeue True
# restores weights, optimizer, scheduler and iteration); otherwise it starts
# from the pretraining checkpoint. Re-running this script after a crash is
# therefore always safe.
#
# Env overrides: ITERS (20000), NUM_ENVS (4 per process), LOAD_MAP (auto|True|False; auto = True when the
# MODEL.MAP_ENCODER.load_pretrained_map_modules key exists), ELEVATION_AXIS (y),
# CUDA_VISIBLE_DEVICES (0-7), MASTER_PORT (29500).
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# GPU layout is derived from CUDA_VISIBLE_DEVICES: N visible cards -> N processes
# with logical ids 0..N-1. NUM_ENVS is the per-process environment count.
IFS=',' read -ra _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
GPU_NUMBERS="${#_gpu_list[@]}"
GPU_IDS="[$(seq -s, 0 $((GPU_NUMBERS - 1)))]"
NUM_ENVS="${NUM_ENVS:-4}"
RUN_NAME="dagger_distill_gt_teacher_llmpt"
RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
LOG_PATH="${RUN_DIR}/train.log"
PRETRAINED_CKPT="/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt"
GT_TEACHER_CKPT="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth"
GT_TEACHER_NAMESPACE="gt.online121c369.r1p5.direction5.v1"
GT_TEACHER_POLICY_NAME="PriorGTTry5Policy"

DRY_RUN_ARGS=()
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_ARGS=(--dry-run)
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/distill/run_dagger_distill_llmpt.sh [--dry-run]" >&2
    exit 2
fi

cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}"

# Resume decision.
RESUME_ARGS=(IL.load_from_ckpt False IL.is_requeue False)
LATEST_CKPT="$(ls -1 "${RUN_DIR}"/ckpt.iter*.pth 2>/dev/null | sed -E 's/.*ckpt\.iter([0-9]+)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)"
if [[ -n "${LATEST_CKPT}" ]]; then
    RESUME_ARGS=(IL.load_from_ckpt True IL.is_requeue True)
    echo "Resuming from ${LATEST_CKPT}"
else
    echo "Fresh start from ${PRETRAINED_CKPT}"
fi

# Pretrained map-module loading: only pass the key when the config defines it
# (patch 0001 applied); LOAD_MAP=auto -> True in that case.
MAP_LOAD_ARGS=()
if grep -q "load_pretrained_map_modules" vlnce_baselines/config/default.py; then
    LOAD_MAP_VALUE="${LOAD_MAP:-True}"
    [[ "${LOAD_MAP_VALUE}" == "auto" ]] && LOAD_MAP_VALUE=True
    MAP_LOAD_ARGS=(MODEL.MAP_ENCODER.load_pretrained_map_modules "${LOAD_MAP_VALUE}")
else
    LOAD_MAP_VALUE="unavailable"
fi

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
    IL.iters "${ITERS:-20000}"
    IL.lr 1e-5
    IL.log_every 200
    IL.ml_weight 1.0
    IL.sample_ratio 0.75
    IL.decay_interval 1400
    IL.warmup_iters 300
    IL.min_lr_ratio 1.0
    "${RESUME_ARGS[@]}"
    IL.waypoint_aug True
    TASK_CONFIG.DATASET.SUFFIX _90
    TRAINER_NAME SS-ETP-LLM
    MODEL.policy_name LLMGridTry5Policy
    MODEL.MAP_ENCODER.enabled True
    MODEL.MAP_ENCODER.architecture try5
    MODEL.MAP_ENCODER.source llm_grid
    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
    MODEL.MAP_ENCODER.refiner_ckpt ""
    "${MAP_LOAD_ARGS[@]}"
    MODEL.elevation_axis "${ELEVATION_AXIS:-y}"
    MODEL.pretrained_path "${PRETRAINED_CKPT}"
    IL.gt_teacher_enabled True
    IL.distill_weight 1.0
    IL.distill_temperature 1.0
    IL.gt_teacher_ckpt "${GT_TEACHER_CKPT}"
    IL.gt_teacher_map_namespace "${GT_TEACHER_NAMESPACE}"
    IL.gt_teacher_policy_name "${GT_TEACHER_POLICY_NAME}"
    IL.gt_teacher_elevation_axis z
)

{
    echo "launch_time=$(date -Is)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "resume_from=${LATEST_CKPT:-none}"
    echo "load_pretrained_map_modules=${LOAD_MAP_VALUE}"
    echo "elevation_axis=${ELEVATION_AXIS:-y}"
    echo "iters=${ITERS:-20000}"
    echo "gpus=${CUDA_VISIBLE_DEVICES} (${GPU_NUMBERS} procs x ${NUM_ENVS} envs)"
} >> "${RUN_DIR}/launch_info.txt"

echo "Log: ${LOG_PATH}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    PYTHONPATH="${REPO_ROOT}" "${COMMAND[@]}" 2>&1 | tee -a "${LOG_PATH}"

if [[ ${#DRY_RUN_ARGS[@]} -ne 0 ]]; then
    echo "Dry run complete"
fi
