#!/usr/bin/env bash
# Distil the GT-legacy teacher (try5, dy, legacy r1.5 m GT map, no coordinate
# fusion; val_unseen SR 68.41) into an isomorphic LLM-Grid try5 student.
#
# Teacher and student share the architecture, the dy position features and the
# legacy map format; only the raster content differs (GT corridor vs LLM
# prediction).  That is what makes the counterfactual map-effect matching and
# the reachability gate well defined: the teacher can read the student's map.
#
# Usage: bash scripts/distill/run_dagger_distill_legacy.sh MODE [--dry-run]
#
#   MODE  kl      plain action KL (historical distillation, the control arm);
#                 JS_LOG=True additionally logs the teacher's GT-vs-LLM-map JS
#                 per iteration (gate_js_mean / gate_js_median) without
#                 weighting anything: that is where GATE_TAU comes from.
#         gated   action KL with the reachability gate (GATE_TAU required)
#         full    STOP-factorised KL + gate + counterfactual effect matching
#                 (GATE_TAU required)
#
# Resume: if data/logs/checkpoints/<RUN_NAME>/ckpt.iter*.pth exists the run
# continues from the latest one; re-running after a crash is always safe.
#
# Env overrides: RUN_NAME (dagger_distill_legacy18600_<MODE>), ITERS (20000),
# NUM_ENVS (4 per process; batch_donor needs >= 2), PRETRAINED_CKPT (460000),
# LOAD_MAP (False: map modules train from initialisation, as the teacher's
# did), GT_TEACHER_CKPT, EFFECT_WEIGHT (1.0), EFFECT_CF (batch_donor|zero),
# GATE_TAU (absolute JS scale, no default: take gate_js_median from a kl
# JS_LOG=True run), GATE_MIN (0.2), GATE_NORMALIZE (True), EFFECT_REDUCTION
# (sum|mean), JS_LOG (False), STOP_WEIGHT (1.0), TEMP (1.0),
# DISTILL_WEIGHT (1.0), CKPT_INTERVAL (1000), CUDA_VISIBLE_DEVICES, MASTER_PORT.
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"

MODE="${1:-}"
case "${MODE}" in
    kl|gated|full) ;;
    *)
        echo "Usage: bash scripts/distill/run_dagger_distill_legacy.sh {kl|gated|full} [--dry-run]" >&2
        exit 2
        ;;
esac
DRY_RUN_ARGS=()
if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN_ARGS=(--dry-run)
elif [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/distill/run_dagger_distill_legacy.sh {kl|gated|full} [--dry-run]" >&2
    exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${_n}" -gt 0 ]]; then
        CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_n - 1)))"
    else
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    fi
fi
IFS=',' read -ra _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
GPU_NUMBERS="${#_gpu_list[@]}"
GPU_IDS="[$(seq -s, 0 $((GPU_NUMBERS - 1)))]"
NUM_ENVS="${NUM_ENVS:-4}"

RUN_NAME="${RUN_NAME:-dagger_distill_legacy18600_${MODE}}"
RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
LOG_PATH="${RUN_DIR}/train.log"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt}"
GT_TEACHER_CKPT="${GT_TEACHER_CKPT:-/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth}"
GT_TEACHER_NAMESPACE="gt.legacy.r1p5.direction5.v1"
GT_TEACHER_POLICY_NAME="PriorGTTry5Policy"
LOAD_MAP="${LOAD_MAP:-False}"

EFFECT_WEIGHT="${EFFECT_WEIGHT:-1.0}"
EFFECT_CF="${EFFECT_CF:-batch_donor}"
GATE_TAU="${GATE_TAU:-0.0}"
GATE_MIN="${GATE_MIN:-0.2}"
GATE_NORMALIZE="${GATE_NORMALIZE:-True}"
EFFECT_REDUCTION="${EFFECT_REDUCTION:-sum}"
JS_LOG="${JS_LOG:-False}"
STOP_WEIGHT="${STOP_WEIGHT:-1.0}"
TEMP="${TEMP:-1.0}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-1.0}"

# Mode -> distillation switches.  Every key defaults to the historical plain
# KL, so the kl arm passes the defaults explicitly for the record.
_need_tau() {
    if ! awk -v t="${GATE_TAU}" 'BEGIN { exit !(t > 0) }'; then
        echo "GATE_TAU must be set to a positive absolute JS scale for MODE=${MODE}: run" >&2
        echo "  JS_LOG=True bash scripts/distill/run_dagger_distill_legacy.sh kl" >&2
        echo "for a few hundred iters and use its gate_js_median" >&2
        exit 2
    fi
}
case "${MODE}" in
    kl)
        if [[ "${JS_LOG}" == "True" ]]; then
            # gate_min_weight 1.0 makes every weight 1: the KL is unchanged,
            # only the teacher's GT-vs-LLM-map JS is logged.
            DISTILL_ARGS=(
                IL.distill_stop_factorized False
                IL.gate_enabled True
                IL.gate_tau 1.0
                IL.gate_min_weight 1.0
                IL.effect_match_weight 0.0
            )
        else
            DISTILL_ARGS=(
                IL.distill_stop_factorized False
                IL.gate_enabled False
                IL.effect_match_weight 0.0
            )
        fi
        ;;
    gated)
        _need_tau
        DISTILL_ARGS=(
            IL.distill_stop_factorized False
            IL.gate_enabled True
            IL.gate_tau "${GATE_TAU}"
            IL.gate_min_weight "${GATE_MIN}"
            IL.gate_normalize "${GATE_NORMALIZE}"
            IL.effect_match_weight 0.0
        )
        ;;
    full)
        _need_tau
        if [[ "${EFFECT_CF}" == "batch_donor" && "${NUM_ENVS}" -lt 2 ]]; then
            echo "EFFECT_CF=batch_donor needs NUM_ENVS >= 2 (got ${NUM_ENVS}); set EFFECT_CF=zero" >&2
            exit 2
        fi
        DISTILL_ARGS=(
            IL.distill_stop_factorized True
            IL.distill_stop_weight "${STOP_WEIGHT}"
            IL.gate_enabled True
            IL.gate_tau "${GATE_TAU}"
            IL.gate_min_weight "${GATE_MIN}"
            IL.gate_normalize "${GATE_NORMALIZE}"
            IL.effect_match_weight "${EFFECT_WEIGHT}"
            IL.effect_counterfactual "${EFFECT_CF}"
            IL.effect_reduction "${EFFECT_REDUCTION}"
        )
        ;;
esac

for key in distill_stop_factorized effect_match_weight gate_enabled gate_normalize effect_reduction; do
    if ! grep -q "_C.IL.${key}" "${REPO_ROOT}/vlnce_baselines/config/default.py"; then
        echo "IL.${key} missing from vlnce_baselines/config/default.py: apply patches 0050 and 0053 first" >&2
        exit 2
    fi
done
[[ -f "${GT_TEACHER_CKPT}" ]] || { echo "Teacher checkpoint not found: ${GT_TEACHER_CKPT}" >&2; exit 1; }
[[ -d "${REPO_ROOT}/data/cognitive_maps/${GT_TEACHER_NAMESPACE}/raster" ]] \
    || { echo "Teacher namespace not found: ${REPO_ROOT}/data/cognitive_maps/${GT_TEACHER_NAMESPACE}/raster" >&2; exit 1; }

cd "${REPO_ROOT}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"
ETP_NAN_DEBUG_STEPS="${ETP_NAN_DEBUG_STEPS:-1000}"
mkdir -p "${RUN_DIR}"

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
    CHECKPOINT_INTERVAL "${CKPT_INTERVAL:-1000}"
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
    MODEL.MAP_ENCODER.load_pretrained_map_modules "${LOAD_MAP}"
    MODEL.MAP_ENCODER.coordinate_fusion False
    MODEL.elevation_axis y
    MODEL.pretrained_path "${PRETRAINED_CKPT}"
    IL.gt_teacher_enabled True
    IL.distill_weight "${DISTILL_WEIGHT}"
    IL.distill_temperature "${TEMP}"
    IL.gt_teacher_ckpt "${GT_TEACHER_CKPT}"
    IL.gt_teacher_map_namespace "${GT_TEACHER_NAMESPACE}"
    IL.gt_teacher_policy_name "${GT_TEACHER_POLICY_NAME}"
    IL.gt_teacher_elevation_axis y
    IL.gate_teacher_start_position_scale 1.0
    "${DISTILL_ARGS[@]}"
)

{
    echo "launch_time=$(date -Is)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "resume_from=${LATEST_CKPT:-none}"
    echo "run_name=${RUN_NAME}"
    echo "mode=${MODE}"
    echo "teacher=${GT_TEACHER_CKPT} (${GT_TEACHER_NAMESPACE}, dy)"
    echo "pretrained=${PRETRAINED_CKPT} load_pretrained_map_modules=${LOAD_MAP}"
    echo "distill_args=${DISTILL_ARGS[*]}"
    echo "iters=${ITERS:-20000}"
    echo "gpus=${CUDA_VISIBLE_DEVICES} (${GPU_NUMBERS} procs x ${NUM_ENVS} envs)"
} >> "${RUN_DIR}/launch_info.txt"

echo "Log: ${LOG_PATH}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    ETP_NAN_DEBUG_STEPS="${ETP_NAN_DEBUG_STEPS}" \
    PYTHONPATH="${REPO_ROOT}" "${COMMAND[@]}" 2>&1 | tee -a "${LOG_PATH}"

if [[ ${#DRY_RUN_ARGS[@]} -ne 0 ]]; then
    echo "Dry run complete"
fi
