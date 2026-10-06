#!/usr/bin/env bash
# Evaluate a PriorGT try5 DAgger checkpoint on R2R val_unseen with any GT map
# namespace (training-free map ablations on e.g. the GT-legacy 18600 policy).
#
# Usage: bash scripts/distill/eval_gt_namespace.sh TAG NAMESPACE
#   TAG        result-directory prefix, e.g. gt18600
#   NAMESPACE  MODEL.MAP_ENCODER.cache_namespace, e.g.
#              gt.legacy.r1p5.direction5.v1.abl_occupancy_only
# Env: CKPT_PATH (required), CUDA_VISIBLE_DEVICES (required, one card),
#      MAP_ABLATION (none | metadata_only | raster_only | no_direction),
#      NUM_ENVS (4), BACK_ALGO (control), ELEVATION_AXIS (y),
#      PRETRAINED_CKPT (shapes only; the DAgger ckpt overrides every weight).
# Results: data/logs/checkpoints/<TAG>_ns_<NAMESPACE>[_abl_<MAP_ABLATION>]/eval_results/
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
PYTHON="/home/xukai/anaconda3/envs/etpr1-py38/bin/python"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
free_port() { "${PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1]); s.close()'; }
RDZV_PORT="${MASTER_PORT:-$(free_port)}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt}"

if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/distill/eval_gt_namespace.sh TAG NAMESPACE" >&2
    exit 2
fi
[[ -n "${CKPT_PATH:-}" && -f "${CKPT_PATH}" ]] || { echo "Set CKPT_PATH to the DAgger checkpoint" >&2; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Set CUDA_VISIBLE_DEVICES to one free card" >&2; exit 2; }
TAG="$1"; NAMESPACE="$2"
MAP_ABLATION="${MAP_ABLATION:-none}"
case "${MAP_ABLATION}" in none|metadata_only|raster_only|no_direction) ;; *)
    echo "Unknown MAP_ABLATION ${MAP_ABLATION}" >&2; exit 2 ;; esac
[[ -d "${REPO_ROOT}/data/cognitive_maps/${NAMESPACE}/raster" ]] \
    || { echo "Namespace not found: ${REPO_ROOT}/data/cognitive_maps/${NAMESPACE}/raster" >&2; exit 1; }

EVAL_NAME="${TAG}_ns_${NAMESPACE}"
[[ "${MAP_ABLATION}" == none ]] || EVAL_NAME="${EVAL_NAME}_abl_${MAP_ABLATION}"
EVAL_DIR="${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}"
mkdir -p "${EVAL_DIR}"
EVAL_LOG="${EVAL_DIR}/eval.log"
echo "ckpt=${CKPT_PATH} namespace=${NAMESPACE} map_ablation=${MAP_ABLATION}" | tee "${EVAL_DIR}/launch_info.txt"

cd "${REPO_ROOT}"
env GLOG_minloglevel=2 MAGNUM_LOG=quiet PYTHONPATH="${REPO_ROOT}" \
    "${TORCHRUN}" --rdzv_backend=c10d --rdzv_endpoint="localhost:${RDZV_PORT}" --nproc_per_node=1 "${REPO_ROOT}/run.py" \
    --exp_name "${EVAL_NAME}" \
    --run-type eval \
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS '[0]' TORCH_GPU_IDS '[0]' GPU_NUMBERS 1 \
    TASK_CONFIG.SEED 100 \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    NUM_ENVIRONMENTS "${NUM_ENVS:-4}" \
    EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 \
    EVAL.CKPT_PATH_DIR "${CKPT_PATH}" \
    IL.back_algo "${BACK_ALGO:-control}" \
    TRAINER_NAME SS-ETP-PriorGT MODEL.policy_name PriorGTTry5Policy \
    MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source prior_gt \
    MODEL.MAP_ENCODER.cache_namespace "${NAMESPACE}" \
    MODEL.MAP_ENCODER.eval_map_source p0 \
    MODEL.MAP_ENCODER.refiner_ckpt "" \
    MODEL.MAP_ENCODER.load_pretrained_map_modules False \
    MODEL.MAP_ENCODER.map_ablation "${MAP_ABLATION}" \
    MODEL.pretrained_path "${PRETRAINED_CKPT}" \
    MODEL.elevation_axis "${ELEVATION_AXIS:-y}" >"${EVAL_LOG}" 2>&1

RESULT_JSON="$(ls "${EVAL_DIR}"/eval_results/stats_ckpt_*_val_unseen.json | head -n 1)"
"${PYTHON}" - "${RESULT_JSON}" "${NAMESPACE}" "${MAP_ABLATION}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]} ablation={sys.argv[3]} | SR {m['success']*100:.2f} | SPL {m['spl']*100:.2f} | "
      f"OSR {m['oracle_success']*100:.2f} | NE {m['distance_to_goal']:.3f} | nDTW {m['ndtw']*100:.2f}")
PY
echo "Log: ${EVAL_LOG}"
echo "Results: ${EVAL_DIR}/eval_results"
