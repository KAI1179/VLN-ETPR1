#!/usr/bin/env bash
# E1 / E2 of the coordinate-fusion plan (docs/daily/2026-09-30.md): plain
# try5 DAgger (no GT teacher, no refiner) with MODEL.MAP_ENCODER.coordinate_fusion,
# release try5 DAgger schedule (30000 iters, decay 2000, warmup 500).
#
# Usage: bash scripts/distill/run_dagger_coord_fusion.sh ARM [--dry-run]
#   ARM gt   E1: GT maps gt.online121c369 (start_position in cells -> 0.5),
#            SS-ETP-PriorGT / PriorGTTry5Policy, from try-5-r1p5_step_387500.
#            Reference: the dz GT teacher, 73.84 on val_unseen.
#   ARM llm  E2: LLM-Grid maps (start_position in metres -> 1.0),
#            SS-ETP-LLM / LLMGridTry5Policy, from model_step_460000, map
#            modules from init exactly like the historical try5 run.
#            Reference: historical try5 iter28000, 65.09 in the same harness.
# Both arms: elevation axis y (the physically correct one), map modules not
# loaded from pretraining (both pretraining checkpoints have a dead tokenizer).
#
# Env: COORD (True; False gives the same run without coordinate fusion as a
#      control, in its own directory), CUDA_VISIBLE_DEVICES (all visible;
#      the historical run used 4 cards x 4 envs), NUM_ENVS (4), ITERS (30000),
#      MASTER_PORT (29500; use distinct ports for concurrent runs),
#      PRETRAINED_CKPT (arm default), RUN_NAME (dagger_coordfusion_<arm>).
# Checkpoint every CKPT_INTERVAL iters (400).
# Resume: re-running continues from the latest ckpt.iter*.pth in the run dir.
# The run dir also gets eval_args.txt, which eval_run.sh / eval_iters.sh /
# eval_raster_donor.sh append, so checkpoints are always evaluated with the
# arm's map source and coordinate-fusion setting.
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"

ARM="${1:-}"
DRY_RUN_ARGS=()
if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN_ARGS=(--dry-run)
elif [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/distill/run_dagger_coord_fusion.sh {gt|llm} [--dry-run]" >&2
    exit 2
fi
COORD="${COORD:-True}"
case "${COORD}" in True|False) ;; *) echo "COORD must be True or False" >&2; exit 2 ;; esac

case "${ARM}" in
    gt)
        PRETRAINED_CKPT="${PRETRAINED_CKPT:-/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt}"
        ARM_ARGS=(
            TRAINER_NAME SS-ETP-PriorGT
            MODEL.policy_name PriorGTTry5Policy
            MODEL.MAP_ENCODER.source prior_gt
            MODEL.MAP_ENCODER.cache_namespace gt.online121c369.r1p5.direction5.v1
            MODEL.MAP_ENCODER.start_position_meters_per_unit 0.5
        ) ;;
    llm)
        PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt}"
        ARM_ARGS=(
            TRAINER_NAME SS-ETP-LLM
            MODEL.policy_name LLMGridTry5Policy
            MODEL.MAP_ENCODER.source llm_grid
            MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
            MODEL.MAP_ENCODER.start_position_meters_per_unit 1.0
        ) ;;
    *) echo "ARM must be gt or llm" >&2; exit 2 ;;
esac
[[ -f "${PRETRAINED_CKPT}" ]] || { echo "Missing pretraining checkpoint ${PRETRAINED_CKPT}" >&2; exit 1; }
grep -q "coordinate_fusion" "${REPO_ROOT}/vlnce_baselines/config/default.py" \
    || { echo "coordinate_fusion config key missing: apply patch 0025" >&2; exit 1; }

MODEL_ARGS=(
    "${ARM_ARGS[@]}"
    MODEL.MAP_ENCODER.enabled True
    MODEL.MAP_ENCODER.architecture try5
    MODEL.MAP_ENCODER.refiner_ckpt ""
    MODEL.MAP_ENCODER.load_pretrained_map_modules False
    MODEL.MAP_ENCODER.coordinate_fusion "${COORD}"
    MODEL.elevation_axis y
    MODEL.pretrained_path "${PRETRAINED_CKPT}"
)

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((${_n:-8} - 1)))"
fi
IFS=',' read -ra _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
GPU_NUMBERS="${#_gpu_list[@]}"
GPU_IDS="[$(seq -s, 0 $((GPU_NUMBERS - 1)))]"
NUM_ENVS="${NUM_ENVS:-4}"
ITERS="${ITERS:-30000}"

if [[ -z "${RUN_NAME:-}" ]]; then
    RUN_NAME="dagger_coordfusion_${ARM}"
    [[ "${COORD}" == "False" ]] && RUN_NAME="dagger_nocoord_${ARM}"
fi
RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
LOG_PATH="${RUN_DIR}/train.log"
mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"
ETP_NAN_DEBUG_STEPS="${ETP_NAN_DEBUG_STEPS:-1000}"

# Model/map options every evaluation of this run must use (one "KEY VALUE" per line).
printf '%s %s\n' "${MODEL_ARGS[@]}" | grep -v '^MODEL.MAP_ENCODER.refiner_ckpt ' \
    > "${RUN_DIR}/eval_args.txt"

RESUME_ARGS=(IL.load_from_ckpt False IL.is_requeue False)
LATEST_CKPT="$(ls -1 "${RUN_DIR}"/ckpt.iter*.pth 2>/dev/null | sed -E 's/.*ckpt\.iter([0-9]+)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 || true)"
if [[ -n "${LATEST_CKPT}" ]]; then
    RESUME_ARGS=(IL.load_from_ckpt True IL.is_requeue True)
    echo "Resuming from ${LATEST_CKPT}"
else
    echo "Fresh start from ${PRETRAINED_CKPT}"
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
    CHECKPOINT_INTERVAL "${CKPT_INTERVAL:-400}"
    ONLY_LAST_SAVEALL False
    IL.iters "${ITERS}"
    IL.lr 1e-5
    IL.log_every 200
    IL.ml_weight 1.0
    IL.sample_ratio 0.75
    IL.decay_interval 2000
    IL.warmup_iters 500
    IL.min_lr_ratio 1.0
    "${RESUME_ARGS[@]}"
    IL.waypoint_aug True
    TASK_CONFIG.DATASET.SUFFIX _90
    IL.gt_teacher_enabled False
    "${MODEL_ARGS[@]}"
)

{
    echo "launch_time=$(date -Is)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "arm=${ARM} coordinate_fusion=${COORD}"
    echo "resume_from=${LATEST_CKPT:-none}"
    echo "pretrained=${PRETRAINED_CKPT}"
    echo "iters=${ITERS}"
    echo "gpus=${CUDA_VISIBLE_DEVICES} (${GPU_NUMBERS} procs x ${NUM_ENVS} envs)"
} >> "${RUN_DIR}/launch_info.txt"

echo "Log: ${LOG_PATH}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    ETP_NAN_DEBUG_STEPS="${ETP_NAN_DEBUG_STEPS}" \
    PYTHONPATH="${REPO_ROOT}" "${COMMAND[@]}" 2>&1 | tee -a "${LOG_PATH}"

if [[ ${#DRY_RUN_ARGS[@]} -ne 0 ]]; then
    echo "Dry run complete"
fi
