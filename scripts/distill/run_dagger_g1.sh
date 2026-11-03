#!/usr/bin/env bash
# G1: exact saved dagger_coordfusion_gt_legacy recipe, varying raster/head only.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
ARM="${1:-}"
DRY_RUN=false
if [[ $# == 2 && "$2" == --dry-run ]]; then DRY_RUN=true
elif [[ $# != 1 ]]; then echo "Usage: $0 ARM [--dry-run]" >&2; exit 2; fi
case "${ARM}" in
    corridor_noreg|corridor_reg|corridor_reg_fb) NAMESPACE=gt.graph.r1p5.direction5.v1 ;;
    box_noreg|box_reg|box_reg_fb) NAMESPACE=gt.graph.box.direction5.v1 ;;
    *) echo "ARM must be corridor_noreg, corridor_reg, corridor_reg_fb, box_noreg, box_reg or box_reg_fb" >&2; exit 2 ;;
esac
REG=False; FEEDBACK=False
[[ "${ARM}" == *_noreg ]] || REG=True
[[ "${ARM}" != *_reg_fb ]] || FEEDBACK=True
case "${ARM}" in
    corridor_noreg) DEFAULT_PORT=29540 ;; corridor_reg) DEFAULT_PORT=29541 ;;
    corridor_reg_fb) DEFAULT_PORT=29542 ;; box_noreg) DEFAULT_PORT=29543 ;;
    box_reg) DEFAULT_PORT=29544 ;; box_reg_fb) DEFAULT_PORT=29545 ;;
esac
PYTHON="${PYTHON:-python}"
if [[ -z "${PRETRAINED_CKPT:-}" ]]; then
    # Read the translated V1 artifact path if available; do not embed a machine path.
    TEMPLATE_ARGS="${REPO_ROOT}/data/logs/checkpoints/dagger_coordfusion_gt_legacy/eval_args.txt"
    if [[ -f "${TEMPLATE_ARGS}" ]]; then
        while read -r key value; do
            if [[ "${key}" == MODEL.pretrained_path && -f "${value}" ]]; then PRETRAINED_CKPT="${value}"; fi
        done < "${TEMPLATE_ARGS}"
    fi
    for candidate in \
        "${REPO_ROOT}/pretrained/r2r_rxr_ce/prior_gt_r1p5_nogaussian/ckpts/model_step_440000.pt" \
        "${REPO_ROOT}/pretrained/pretrained/pretrained/r2r_rxr_ce/prior_gt_r1p5_nogaussian/ckpts/model_step_440000.pt"; do
        if [[ -z "${PRETRAINED_CKPT:-}" && -f "${candidate}" ]]; then PRETRAINED_CKPT="${candidate}"; fi
    done
fi
[[ -f "${PRETRAINED_CKPT:-}" ]] || { echo "Set PRETRAINED_CKPT to the template's prior_gt_r1p5_nogaussian model_step_440000.pt" >&2; exit 1; }
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data_v2/runs}"
OUTPUT_ROOT="$(${PYTHON} -c 'from pathlib import Path; import sys; from prior.cogmap_v2.graph import output_path; print(output_path(Path(sys.argv[1])))' "${OUTPUT_ROOT}")"
RUN_NAME="${RUN_NAME:-dagger_g1_${ARM}}"
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid RUN_NAME" >&2; exit 2; }
RUN_DIR="${OUTPUT_ROOT}/checkpoints/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
MODEL_ARGS=(
    TRAINER_NAME SS-ETP-PriorGT MODEL.policy_name PriorGTTry5Policy
    MODEL.MAP_ENCODER.source prior_gt MODEL.MAP_ENCODER.architecture try5
    MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.coordinate_fusion True
    MODEL.MAP_ENCODER.cache_dir "${REPO_ROOT}/data_v2/cognitive_maps"
    MODEL.MAP_ENCODER.cache_namespace "${NAMESPACE}"
    MODEL.MAP_ENCODER.start_position_meters_per_unit 1.0
    MODEL.MAP_ENCODER.start_position_scale 1.0
    MODEL.MAP_ENCODER.load_pretrained_map_modules True
    MODEL.MAP_ENCODER.refiner_ckpt "" MODEL.MAP_ENCODER.eval_map_source p0
    MODEL.elevation_axis y MODEL.pretrained_path "${PRETRAINED_CKPT}"
    IL.reg_enabled "${REG}" IL.reg_feedback "${FEEDBACK}"
    IL.reg_weight "${REG_WEIGHT:-0.1}" IL.reg_n_max 16 IL.reg_max_distance 2.0
    IL.reg_label_dir "${REPO_ROOT}/data_v2/cogmap_v2/registration"
    IL.back_algo teleport IL.tryout True IL.max_text_len 150 IL.max_traj_len 15
)
printf '%s %s\n' "${MODEL_ARGS[@]}" > "${RUN_DIR}/eval_args.txt"
RESUME_ARGS=(IL.load_from_ckpt False IL.is_requeue False)
LATEST_CKPT="$(find "${RUN_DIR}" -maxdepth 1 -name 'ckpt.iter*.pth' -printf '%f\n' | sort -V | tail -1)"
if [[ -n "${LATEST_CKPT}" ]]; then RESUME_ARGS=(IL.load_from_ckpt True IL.is_requeue True); fi
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPU_LIST <<< "${VISIBLE_GPUS}"
[[ ${#GPU_LIST[@]} == 4 ]] || { echo "The template uses four ranks; provide four CUDA_VISIBLE_DEVICES" >&2; exit 2; }
ARGS=(
    "${REPO_ROOT}/run.py" --exp_name "${RUN_NAME}" --run-type dagger
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml" --output-root "${OUTPUT_ROOT}"
)
OPTS=(
    SIMULATOR_GPU_IDS '[0,1,2,3]' TORCH_GPU_IDS '[0,1,2,3]' GPU_NUMBERS 4
    TASK_CONFIG.SEED 100 TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
    NUM_ENVIRONMENTS 4 CHECKPOINT_INTERVAL 800 ONLY_LAST_SAVEALL False
    IL.iters 30000 IL.lr 1e-5 IL.log_every 200 IL.ml_weight 1.0
    IL.sample_ratio 0.75 IL.decay_interval 2000 IL.warmup_iters 500 IL.min_lr_ratio 1.0
    IL.max_text_len 150 IL.max_traj_len 15 IL.back_algo teleport IL.tryout True
    "${RESUME_ARGS[@]}" IL.waypoint_aug True TASK_CONFIG.DATASET.SUFFIX _90
    IL.gt_teacher_enabled False "${MODEL_ARGS[@]}"
)
printf 'arm=%s\nseed=100\niters=30000\ncheckpoint_interval=800\nranks=4\nenvs_per_rank=4\npretrained=%s\n' \
    "${ARM}" "${PRETRAINED_CKPT}" > "${RUN_DIR}/launch_info.txt"
TORCHRUN_COMMAND=(
    "${PYTHON}" -m torch.distributed.run --nproc_per_node=4 --rdzv_backend=c10d
    --rdzv_endpoint="localhost:${MASTER_PORT:-${DEFAULT_PORT}}" "${ARGS[@]}" "${OPTS[@]}"
)
{
    printf 'env CUDA_VISIBLE_DEVICES=%q' "${VISIBLE_GPUS}"
    printf ' %q' "${TORCHRUN_COMMAND[@]}"
    printf '\n'
} > "${RUN_DIR}/torchrun_command.sh"
printf 'Resolved torchrun (training is not started by --dry-run):\n'
cat "${RUN_DIR}/torchrun_command.sh"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_CACHE_HOME="${REPO_ROOT}/data_v2/cache"
mkdir -p "${XDG_CACHE_HOME}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"
export GLOG_minloglevel=2 MAGNUM_LOG=quiet
if [[ "${DRY_RUN}" == true ]]; then
    CUDA_VISIBLE_DEVICES='' "${PYTHON}" "${ARGS[@]}" --dry-run "${OPTS[@]}" 2>&1 | tee "${RUN_DIR}/dry_run.log"
else
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" "${TORCHRUN_COMMAND[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
fi
