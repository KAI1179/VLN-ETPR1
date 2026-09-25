#!/usr/bin/env bash
# Teacher channel ablation on R2R val_unseen (closes R5 §6 and tests which map
# channel carries the teacher's edge).
#
# Usage: bash scripts/distill/eval_teacher_ablation.sh [MODE]
#   MODE: none (default, = dz teacher, expected ~74 SR)
#         metadata_only  (raster zeroed; only the 14-dim metadata token carries map info)
#         raster_only    (all metadata zeroed; only the raster carries map info)
#         no_direction   (only the five route direction vectors zeroed)
# GPU: CUDA_VISIBLE_DEVICES (default 0), NUM_ENVS (default 4).
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
TEACHER_NAMESPACE="gt.online121c369.r1p5.direction5.v1"

MODE="${1:-none}"
case "${MODE}" in none|metadata_only|raster_only|no_direction) ;; *)
    echo "Unknown MODE ${MODE}" >&2; exit 2 ;; esac
RUN_NAME="gt_teacher_try5_val_unseen_z_${MODE}"
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
    TRAINER_NAME SS-ETP-PriorGT MODEL.policy_name PriorGTTry5Policy \
    MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source prior_gt \
    MODEL.MAP_ENCODER.cache_namespace "${TEACHER_NAMESPACE}" \
    MODEL.MAP_ENCODER.refiner_ckpt "" \
    MODEL.MAP_ENCODER.load_pretrained_map_modules False \
    MODEL.MAP_ENCODER.map_ablation "${MODE}" \
    MODEL.pretrained_path "${TEACHER_PRETRAINED}" \
    MODEL.elevation_axis z \
    EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 \
    EVAL.CKPT_PATH_DIR "${TEACHER_DAGGER}" \
    IL.back_algo control 2>&1 | tee "${LOG_PATH}"

"${PYTHON}" - "${RESULT_JSON}" "${MODE}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"\nteacher ablation={sys.argv[2]} | SR {m['success']*100:.2f} | SPL {m['spl']*100:.2f} | "
      f"OSR {m['oracle_success']*100:.2f} | NE {m['distance_to_goal']:.3f} | stop_err {(m['oracle_success']-m['success'])*100:.2f}")
PY
echo "Metrics: ${RESULT_JSON}"
