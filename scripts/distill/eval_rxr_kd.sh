#!/usr/bin/env bash
# Evaluate one checkpoint of an RxR KD run (kd_rxr_{en,all}_gt15000_kl) on an
# RxR split with the exact map/config the run trained with (rxr_kd_env.sh).
#
# Usage: bash scripts/distill/eval_rxr_kd.sh SETTING ITER
#   SETTING  en | all   (languages, LLM cache key and run name follow it)
# Env: CUDA_VISIBLE_DEVICES (default 0; several ids = one process per GPU),
#      NUM_ENVS (4), SPLIT (val_unseen), BACK_ALGO (teleport, the RxR default),
#      EVAL_LANGUAGES (override the setting's languages, e.g. "en-US,en-IN"
#      to score the all-language model on English only), RUN_NAME, MASTER_PORT.
# Results: data/logs/checkpoints/<RUN>_eval_iter<ITER>_<SPLIT>[_<langtag>]/eval_results/
#          stats_ckpt_<ITER>_<SPLIT>.json   (summarize_eval.py layout)
# Console: data/logs/checkpoints/<RUN>/eval_iter<ITER>_<SPLIT>[_<langtag>].log
#
# RxR eval reads its reference paths from IL.RECOLLECT_TRAINER.gt_file, whose
# default points at a dataset directory this machine does not have
# (RxR_VLNCE_v0); it is set here to the enc_xlmr files training uses.
#
# Episodes without an LLM map are pre-filled as failures by the trainer and
# counted in SR (llm_cache_missing_count in the json says how many).
set -euo pipefail
SETTING="${1:-}"; ITER="${2:-}"
[[ "${SETTING}" =~ ^(en|all)$ && "${ITER}" =~ ^[0-9]+$ ]] || { echo "Usage: $0 {en|all} ITER" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT SETTING
source "${REPO_ROOT}/scripts/distill/rxr_kd_env.sh"
TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"
[[ -x "${TORCHRUN}" ]] || { echo "torchrun not found; set TORCHRUN" >&2; exit 1; }
PYTHON="$(dirname "${TORCHRUN}")/python"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
GPU_NUMBERS="${#_gpu_list[@]}"
GPU_IDS="[$(seq -s, 0 $((GPU_NUMBERS - 1)))]"
NUM_ENVS="${NUM_ENVS:-4}"
SPLIT="${SPLIT:-val_unseen}"
BACK_ALGO="${BACK_ALGO:-teleport}"

LANG_ARGS=(); LANG_TAG=""
if [[ -n "${EVAL_LANGUAGES:-}" ]]; then
    IFS=',' read -ra _langs <<< "${EVAL_LANGUAGES}"
    LANG_LIST="[$(printf '%s,' "${_langs[@]}" | sed 's/,$//')]"
    LANG_ARGS=(EVAL.LANGUAGES "${LANG_LIST}")
    LANG_TAG="_$(printf '%s' "${EVAL_LANGUAGES}" | tr -c 'A-Za-z0-9' '-' | sed 's/-*$//')"
fi

RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
CKPT_PATH="${RUN_DIR}/ckpt.iter${ITER}.pth"
EVAL_NAME="${RUN_NAME}_eval_iter${ITER}_${SPLIT}${LANG_TAG}"
EVAL_LOG="${RUN_DIR}/eval_iter${ITER}_${SPLIT}${LANG_TAG}.log"
RESULT_JSON="${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}/eval_results/stats_ckpt_${ITER}_${SPLIT}.json"
[[ -f "${CKPT_PATH}" ]] || { echo "Checkpoint not found: ${CKPT_PATH}" >&2; exit 1; }

free_port() { "${PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1]); s.close()'; }
RDZV_PORT="${MASTER_PORT:-$(free_port)}"

cd "${REPO_ROOT}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}" \
    PYTHONPATH="${REPO_ROOT}" "${TORCHRUN}" --rdzv_backend=c10d --rdzv_endpoint="localhost:${RDZV_PORT}" \
    --nproc_per_node="${GPU_NUMBERS}" "${REPO_ROOT}/run.py" \
    --exp_name "${EVAL_NAME}" \
    --run-type eval \
    --exp-config "${EXP_CONFIG}" \
    SIMULATOR_GPU_IDS "${GPU_IDS}" TORCH_GPU_IDS "${GPU_IDS}" GPU_NUMBERS "${GPU_NUMBERS}" \
    TASK_CONFIG.SEED 100 \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    NUM_ENVIRONMENTS "${NUM_ENVS}" \
    EVAL.SPLIT "${SPLIT}" EVAL.EPISODE_COUNT -1 \
    EVAL.CKPT_PATH_DIR "${CKPT_PATH}" \
    "${LANG_ARGS[@]}" \
    IL.back_algo "${BACK_ALGO}" \
    IL.RECOLLECT_TRAINER.gt_file "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz" \
    TRAINER_NAME SS-ETP-LLM \
    MODEL.policy_name LLMGridTry5Policy \
    MODEL.MAP_ENCODER.enabled True \
    MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source llm_grid \
    MODEL.MAP_ENCODER.llm_cache_model_key "${LLM_CACHE_KEY}" \
    MODEL.MAP_ENCODER.llm_cache_dir "${LLM_CACHE_DIR}" \
    MODEL.MAP_ENCODER.refiner_ckpt "" \
    MODEL.MAP_ENCODER.load_pretrained_map_modules False \
    MODEL.MAP_ENCODER.coordinate_fusion False \
    MODEL.elevation_axis y \
    MODEL.pretrained_path "${PRETRAINED_CKPT}" \
    IL.gt_teacher_enabled False 2>&1 | tee "${EVAL_LOG}"

[[ -f "${RESULT_JSON}" ]] || { echo "No result written: ${RESULT_JSON}" >&2; exit 1; }
"${PYTHON}" - "${RESULT_JSON}" "${EVAL_NAME}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"\n{sys.argv[2]} | SR {m['success']*100:.2f} | SPL {m['spl']*100:.2f} | OSR {m['oracle_success']*100:.2f} | "
      f"nDTW {m.get('ndtw',0)*100:.2f} | SDTW {m.get('sdtw',0)*100:.2f} | NE {m['distance_to_goal']:.3f} | "
      f"cache_missing {int(m.get('llm_cache_missing_count',0))}")
PY
echo "Metrics: ${RESULT_JSON}"
