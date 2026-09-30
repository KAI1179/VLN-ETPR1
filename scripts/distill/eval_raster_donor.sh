#!/usr/bin/env bash
# Episode-level raster information test for an LLM-grid try5 policy
# (training-free, R2R val_unseen).
#
# The raster of every episode is replaced by another episode's LLM raster while
# the episode keeps its own metadata token inputs, so the input stays in
# distribution (unlike zeroing). If SR(cross_scene) ~= SR(control), the policy
# uses no episode-specific raster information and the drop seen when zeroing
# the raster is a distribution-shift cost.
#
# Usage: bash scripts/distill/eval_raster_donor.sh MODE
#   MODE: control       the unmodified LLM cache (same harness, required baseline)
#         cross_scene   raster from a random episode in another scene
#         within_scene  raster from the same scene, different trajectory_id
#         block_permute own raster, 10x10 token blocks permuted (sanity check;
#                       the encoder has no positional encoding, expect == control)
# Env:
#   CKPT_PATH  DAgger checkpoint (default: historical LLM-Grid try5 iter28000)
#   TAG        result-directory prefix (default: llm_grid_try5_iter28000)
#   SEED       donor seed (default 0)
#   CUDA_VISIBLE_DEVICES (required, one free card), NUM_ENVS (default 4)
#   BACK_ALGO  (default control), ELEVATION_AXIS (default y: trained after 375b443)
# Results: data/logs/checkpoints/<TAG>_raster_<MODE>/eval_results/
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
PYTHON="/home/xukai/anaconda3/envs/etpr1-py38/bin/python"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
SOURCE_KEY="llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
CKPT_PATH="${CKPT_PATH:-/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth}"
TAG="${TAG:-llm_grid_try5_iter28000}"
SEED="${SEED:-0}"
# Only seeds tensor shapes; the DAgger checkpoint overrides every weight.
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt}"
free_port() { "${PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1]); s.close()'; }
RDZV_PORT="${MASTER_PORT:-$(free_port)}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"

if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/distill/eval_raster_donor.sh MODE" >&2
    exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to one free card." >&2
    exit 2
fi
MODE="$1"
[[ -f "${CKPT_PATH}" ]] || { echo "Missing checkpoint ${CKPT_PATH}" >&2; exit 1; }

cd "${REPO_ROOT}"
case "${MODE}" in
    control) CACHE_KEY="${SOURCE_KEY}" ;;
    cross_scene|within_scene|block_permute)
        CACHE_KEY="$(PYTHONPATH="${REPO_ROOT}" "${PYTHON}" scripts/distill/make_raster_donor_cache.py \
            --mode "${MODE}" --seed "${SEED}" --source-model-key "${SOURCE_KEY}" | tail -n 1)" ;;
    *) echo "Unknown MODE ${MODE}" >&2; exit 2 ;;
esac

EVAL_NAME="${TAG}_raster_${MODE}"
[[ "${MODE}" == control ]] || EVAL_NAME="${EVAL_NAME}_s${SEED}"
EVAL_DIR="${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}"
mkdir -p "${EVAL_DIR}"
EVAL_LOG="${EVAL_DIR}/eval.log"
echo "mode=${MODE} cache_key=${CACHE_KEY} ckpt=${CKPT_PATH}" | tee "${EVAL_DIR}/launch_info.txt"

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
    TRAINER_NAME SS-ETP-LLM MODEL.policy_name LLMGridTry5Policy \
    MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source llm_grid \
    MODEL.MAP_ENCODER.llm_cache_model_key "${CACHE_KEY}" \
    MODEL.MAP_ENCODER.eval_map_source p0 \
    MODEL.MAP_ENCODER.refiner_ckpt "" \
    MODEL.MAP_ENCODER.load_pretrained_map_modules False \
    MODEL.MAP_ENCODER.map_ablation none \
    MODEL.pretrained_path "${PRETRAINED_CKPT}" \
    MODEL.elevation_axis "${ELEVATION_AXIS:-y}" >"${EVAL_LOG}" 2>&1

RESULT_DIR="${EVAL_DIR}/eval_results"
RESULT_JSON="$(ls "${RESULT_DIR}"/stats_ckpt_*_val_unseen.json | head -n 1)"
"${PYTHON}" - "${RESULT_JSON}" "${MODE}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"raster={sys.argv[2]} | SR {m['success']*100:.2f} | SPL {m['spl']*100:.2f} | "
      f"OSR {m['oracle_success']*100:.2f} | NE {m['distance_to_goal']:.3f}")
PY
grep -h "cache_missing" "${EVAL_LOG}" | tail -n 1 || true
echo "Log: ${EVAL_LOG}"
echo "Paired vs control:"
echo "  ${PYTHON} scripts/distill/paired_analysis.py --dir-a data/logs/checkpoints/${TAG}_raster_control/eval_results --dir-b ${RESULT_DIR}"
