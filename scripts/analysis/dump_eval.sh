#!/usr/bin/env bash
# Evaluate one checkpoint on R2R val_unseen with the per-step decision dump
# (EVAL.decision_dump), in one of two protocols:
#   free    the policy drives (protocol 1): normal evaluation + dump;
#   forced  the DAgger expert drives (protocol 2): every policy sees the same
#           states, the dump records what each would have chosen.
#
# Usage: bash scripts/analysis/dump_eval.sh MODEL PROTOCOL RUN_NAME ITER
#   MODEL     baseline  mapless ETP-R1 (TRAINER SS-ETP-R1); the checkpoint is
#                       CKPT (default: the release_r2r_dagger iter25000 path
#                       below), RUN_NAME/ITER only name the output;
#             try5      LLM-grid try5 DAgger run (same arguments as
#                       scripts/distill/eval_run.sh, reads the run's
#                       eval_args.txt when present);
#             kd        dual-distillation student, same as try5 plus
#                       MODEL.MAP_ENCODER.map_ablation ${MAP_VIEW:-none}
#                       (none = map view, metadata_only = deployment view).
# Env: CUDA_VISIBLE_DEVICES (default 0; several ids = several ranks),
#      NUM_ENVS (4), BACK_ALGO (control), SPLIT (val_unseen), MASTER_PORT,
#      PRETRAINED_CKPT (shapes only), ELEVATION_AXIS (y), CKPT (baseline),
#      MAP_VIEW (kd), REPO_ROOT, PYTHON, TORCHRUN.
# Output: data/logs/checkpoints/<RUN_NAME>_dump_<PROTOCOL>_iter<ITER>_<SPLIT>/eval_results/
#   stats_ckpt_<ITER>_<SPLIT>.json        aggregate metrics (free protocol)
#   stats_ep_ckpt_<ITER>_<SPLIT>_r*_w*.json per-episode metrics
#   decisions_<SPLIT>_r*_w*.jsonl          per-step decisions (one file per rank)
# Then: python scripts/analysis/episode_features.py --split val_unseen
#       python scripts/analysis/paired_report.py --help
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python)}"
TORCHRUN="${TORCHRUN:-$(dirname "${PYTHON}")/torchrun}"
free_port() { "${PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("localhost",0)); print(s.getsockname()[1]); s.close()'; }
RDZV_PORT="${MASTER_PORT:-$(free_port)}"
export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"

if [[ $# -ne 4 ]]; then
    echo "Usage: bash scripts/analysis/dump_eval.sh MODEL(baseline|try5|kd) PROTOCOL(free|forced) RUN_NAME ITER" >&2
    exit 2
fi
MODEL="$1"; PROTOCOL="$2"; RUN_NAME="$3"; ITER="$4"
case "${PROTOCOL}" in free) FORCING=False ;; forced) FORCING=True ;; *)
    echo "PROTOCOL must be free or forced, got '${PROTOCOL}'" >&2; exit 2 ;; esac

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
GPU_NUMBERS="${#_gpu_list[@]}"
GPU_IDS="[$(seq -s, 0 $((GPU_NUMBERS - 1)))]"
NUM_ENVS="${NUM_ENVS:-4}"
BACK_ALGO="${BACK_ALGO:-control}"
SPLIT="${SPLIT:-val_unseen}"
RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
EVAL_NAME="${RUN_NAME}_dump_${PROTOCOL}_iter${ITER}_${SPLIT}"
EVAL_LOG="${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}.log"
RESULT_DIR="${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}/eval_results"

MODEL_ARGS=()
case "${MODEL}" in
    baseline)
        CKPT_PATH="${CKPT:-/data/xukai/etp-r1-prior/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth}"
        PRETRAINED_CKPT="${PRETRAINED_CKPT:-${REPO_ROOT}/pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt}"
        MODEL_ARGS=(TRAINER_NAME SS-ETP-R1 MODEL.pretrained_path "${PRETRAINED_CKPT}")
        ;;
    try5|kd)
        CKPT_PATH="${CKPT:-${RUN_DIR}/ckpt.iter${ITER}.pth}"
        PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt}"
        MODEL_ARGS=(
            TRAINER_NAME SS-ETP-LLM
            MODEL.policy_name LLMGridTry5Policy
            MODEL.MAP_ENCODER.enabled True
            MODEL.MAP_ENCODER.architecture try5
            MODEL.MAP_ENCODER.source llm_grid
            MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
            MODEL.MAP_ENCODER.refiner_ckpt ""
            MODEL.MAP_ENCODER.load_pretrained_map_modules False
            MODEL.elevation_axis "${ELEVATION_AXIS:-y}"
            MODEL.pretrained_path "${PRETRAINED_CKPT}"
        )
        if [[ "${MODEL}" == "kd" ]]; then
            MODEL_ARGS+=(MODEL.MAP_ENCODER.eval_map_source p0 MODEL.MAP_ENCODER.map_ablation "${MAP_VIEW:-none}")
        fi
        # Runs started by run_dagger_coord_fusion.sh record the model/map
        # options their checkpoints need; they override the defaults above.
        if [[ -f "${RUN_DIR}/eval_args.txt" ]]; then
            while read -r key value; do MODEL_ARGS+=("${key}" "${value}"); done < "${RUN_DIR}/eval_args.txt"
            echo "Using ${RUN_DIR}/eval_args.txt"
        fi
        ;;
    *) echo "MODEL must be baseline, try5 or kd, got '${MODEL}'" >&2; exit 2 ;;
esac
[[ -f "${CKPT_PATH}" ]] || { echo "Checkpoint not found: ${CKPT_PATH}" >&2; exit 1; }
[[ -f "${PRETRAINED_CKPT}" ]] || { echo "Pretrained checkpoint not found: ${PRETRAINED_CKPT}" >&2; exit 1; }
if ls "${RESULT_DIR}"/decisions_"${SPLIT}"_r*.jsonl >/dev/null 2>&1; then
    echo "Dump exists in ${RESULT_DIR}; delete it to redo." >&2; exit 1
fi

cd "${REPO_ROOT}"
echo "model=${MODEL} protocol=${PROTOCOL} ckpt=${CKPT_PATH} gpus=${CUDA_VISIBLE_DEVICES} -> ${RESULT_DIR}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    PYTHONPATH="${REPO_ROOT}" "${TORCHRUN}" --rdzv_backend=c10d --rdzv_endpoint="localhost:${RDZV_PORT}" --nproc_per_node="${GPU_NUMBERS}" \
    "${REPO_ROOT}/run.py" \
    --exp_name "${EVAL_NAME}" \
    --run-type eval \
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS "${GPU_IDS}" TORCH_GPU_IDS "${GPU_IDS}" GPU_NUMBERS "${GPU_NUMBERS}" \
    TASK_CONFIG.SEED 100 \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    NUM_ENVIRONMENTS "${NUM_ENVS}" \
    EVAL.SPLIT "${SPLIT}" EVAL.EPISODE_COUNT -1 \
    EVAL.CKPT_PATH_DIR "${CKPT_PATH}" \
    EVAL.decision_dump True EVAL.teacher_forcing "${FORCING}" \
    IL.back_algo "${BACK_ALGO}" \
    "${MODEL_ARGS[@]}" 2>&1 | tee "${EVAL_LOG}"

"${PYTHON}" - "${RESULT_DIR}" "${SPLIT}" <<'PY'
import glob, json, os, sys
d, split = sys.argv[1:3]
files = sorted(glob.glob(os.path.join(d, f"decisions_{split}_r*.jsonl")))
steps = eps = 0
for f in files:
    for line in open(f):
        k = json.loads(line)["kind"]
        steps += k == "step"; eps += k == "episode"
print(f"\ndump: {len(files)} file(s), {eps} episodes, {steps} steps in {d}")
PY
