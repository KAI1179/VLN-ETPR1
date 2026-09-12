#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
PYTHON="/home/xukai/anaconda3/envs/etpr1-py38/bin/python"
TORCHRUN="/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun"
PRETRAINED_CKPT="/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt"
REPORT_PATH="${REPO_ROOT}/reports/refiner/S4_progress.md"

DRY_RUN_ARGS=()
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_ARGS=(--dry-run)
    shift
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash scripts/refiner/eval_s4.sh [--dry-run] <run_name> <iter> [map_source]" >&2
    exit 2
fi

RUN_NAME="$1"
ITER="$2"
MAP_SOURCE="${3:-refiner}"
RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
CONFIG_PATH="${RUN_DIR}/config.yaml"
CKPT_PATH="${RUN_DIR}/ckpt.iter${ITER}.pth"
EVAL_NAME="${RUN_NAME}_eval_iter${ITER}"
EVAL_LOG="${RUN_DIR}/eval_iter${ITER}.log"
RESULT_JSON="${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}/eval_results/stats_ckpt_${ITER}_val_unseen.json"

REFINER_CKPT="$(${PYTHON} -c \
    'import sys, yaml; config = yaml.safe_load(open(sys.argv[1])); print(config["MODEL"]["MAP_ENCODER"].get("refiner_ckpt", ""))' \
    "${CONFIG_PATH}")"
REFINER_ARGS=(MODEL.MAP_ENCODER.eval_map_source "${MAP_SOURCE}")
if [[ -n "${REFINER_CKPT}" && "${MAP_SOURCE}" == "refiner" ]]; then
    REFINER_ARGS=(MODEL.MAP_ENCODER.refiner_ckpt "${REFINER_CKPT}")
fi

COMMAND=(
    "${TORCHRUN}" --standalone --nproc_per_node=1 "${REPO_ROOT}/run.py"
    --exp_name "${EVAL_NAME}"
    --run-type eval
    --exp-config "${REPO_ROOT}/run_r2r/iter_train.yaml"
    "${DRY_RUN_ARGS[@]}"
    SIMULATOR_GPU_IDS '[0]'
    TORCH_GPU_IDS '[0]'
    GPU_NUMBERS 1
    TASK_CONFIG.SEED 100
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
    NUM_ENVIRONMENTS 4
    EVAL.SPLIT val_unseen
    EVAL.EPISODE_COUNT -1
    EVAL.CKPT_PATH_DIR "${CKPT_PATH}"
    IL.back_algo control
    TRAINER_NAME SS-ETP-LLM
    MODEL.policy_name LLMGridTry5Policy
    MODEL.MAP_ENCODER.enabled True
    MODEL.MAP_ENCODER.architecture try5
    MODEL.MAP_ENCODER.source llm_grid
    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
    "${REFINER_ARGS[@]}"
    MODEL.pretrained_path "${PRETRAINED_CKPT}"
)

cd "${REPO_ROOT}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" \
    GLOG_minloglevel=2 MAGNUM_LOG=quiet PYTHONPATH="${REPO_ROOT}" \
    "${COMMAND[@]}" >"${EVAL_LOG}" 2>&1

if [[ ${#DRY_RUN_ARGS[@]} -ne 0 ]]; then
    echo "Dry run complete"
    echo "Log: ${EVAL_LOG}"
    echo "Config: ${REPO_ROOT}/data/logs/checkpoints/${EVAL_NAME}/config.yaml"
    echo "Checkpoint: ${CKPT_PATH}"
    echo "Refiner: ${REFINER_CKPT:-disabled}"
    exit 0
fi

"${PYTHON}" -c '
import json
import sys

metrics = json.load(open(sys.argv[1]))
success = metrics["success"] * 100
spl = metrics["spl"] * 100
distance_to_goal = metrics["distance_to_goal"]
oracle_success = metrics["oracle_success"] * 100
row = (
    f"| {sys.argv[3]} | {sys.argv[4]} | "
    f"{success:.2f} | "
    f"{spl:.2f} | "
    f"{distance_to_goal:.3f} | "
    f"{oracle_success:.2f} |\n"
)
with open(sys.argv[2], "a") as report:
    report.write(row)
' "${RESULT_JSON}" "${REPORT_PATH}" "${RUN_NAME}" "${ITER}"

echo "Evaluation log: ${EVAL_LOG}"
echo "Metrics: ${RESULT_JSON}"
echo "Progress report: ${REPORT_PATH}"
