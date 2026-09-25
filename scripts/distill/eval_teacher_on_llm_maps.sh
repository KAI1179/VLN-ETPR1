#!/usr/bin/env bash
# Evaluate the GT-trained teacher checkpoint on inputs it never saw:
#   llm   : LLM-Grid maps (cache llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree),
#           start_position scaled x2 because the teacher was trained on
#           online maps that store grid cells (= 2 x metres, see R4).
#   nomap : map encoder disabled.
# Both use MODEL.elevation_axis z (the teacher's dz position features).
#
# Usage: bash scripts/distill/eval_teacher_on_llm_maps.sh [llm|nomap]
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
PYTHON="/home/xukai/anaconda3/envs/etpr1-py38/bin/python"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
# torchrun --standalone binds a fixed rendezvous port on torch 2.1, so concurrent
# evals collide; pick a free port instead (override with MASTER_PORT).
free_port() { "${PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1]); s.close()'; }
RDZV_PORT="${MASTER_PORT:-$(free_port)}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"
TEACHER_PRETRAINED="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt"
TEACHER_DAGGER="/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth"
LLM_CACHE_MODEL_KEY="llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"

MODE="${1:-llm}"
case "${MODE}" in
    llm)
        RUN_NAME="gt_teacher_try5_val_unseen_z_on_llm_maps"
        MODE_ARGS=(TRAINER_NAME SS-ETP-LLM MODEL.policy_name LLMGridTry5Policy
                   MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5
                   MODEL.MAP_ENCODER.source llm_grid
                   MODEL.MAP_ENCODER.llm_cache_model_key "${LLM_CACHE_MODEL_KEY}"
                   MODEL.MAP_ENCODER.start_position_scale "${START_POSITION_SCALE:-2.0}")
        ;;
    nomap)
        RUN_NAME="gt_teacher_try5_val_unseen_z_nomap"
        MODE_ARGS=(TRAINER_NAME SS-ETP-PriorGT MODEL.policy_name PriorGTTry5Policy
                   MODEL.MAP_ENCODER.enabled False MODEL.MAP_ENCODER.architecture try5
                   MODEL.MAP_ENCODER.source prior_gt)
        ;;
    *) echo "Unknown MODE ${MODE}" >&2; exit 2 ;;
esac
RESULT_JSON="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}/eval_results/stats_ckpt_16000_val_unseen.json"
LOG_PATH="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}/eval.log"
mkdir -p "$(dirname "${LOG_PATH}")"

cd "${REPO_ROOT}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
    PYTHONPATH="${REPO_ROOT}" "${TORCHRUN}" --rdzv_backend=c10d --rdzv_endpoint="localhost:${RDZV_PORT}" --nproc_per_node=1 \
    "${REPO_ROOT}/run.py" \
    --exp_name "${RUN_NAME}" \
    --run-type eval \
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS '[0]' TORCH_GPU_IDS '[0]' GPU_NUMBERS 1 \
    NUM_ENVIRONMENTS "${NUM_ENVS:-4}" \
    TASK_CONFIG.SEED 100 \
    "${MODE_ARGS[@]}" \
    MODEL.MAP_ENCODER.refiner_ckpt "" \
    MODEL.MAP_ENCODER.load_pretrained_map_modules False \
    MODEL.pretrained_path "${TEACHER_PRETRAINED}" \
    MODEL.elevation_axis z \
    EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 \
    EVAL.CKPT_PATH_DIR "${TEACHER_DAGGER}" \
    IL.back_algo control 2>&1 | tee "${LOG_PATH}"

"${PYTHON}" - "${RESULT_JSON}" "${MODE}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"\nteacher on {sys.argv[2]} | SR {m['success']*100:.2f} | SPL {m['spl']*100:.2f} | "
      f"OSR {m['oracle_success']*100:.2f} | NE {m['distance_to_goal']:.3f} | stop_err {(m['oracle_success']-m['success'])*100:.2f}"
      + (f" | cache_missing {int(m.get('llm_cache_missing_count', 0))}" if 'llm_cache_missing_count' in m else ""))
PY
echo "Metrics: ${RESULT_JSON}"
