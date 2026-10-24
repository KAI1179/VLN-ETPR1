#!/usr/bin/env bash
# Serially evaluate several checkpoints of one RxR KD run, each as a single
# torchrun over every visible GPU; then print the summary table.
#
# Usage: bash scripts/distill/eval_rxr_kd_iters.sh SETTING [ITER ...]
#   no ITER     list checkpoints and which already have a result, exit
#   ITER ...    explicit iterations
#   all         every checkpoint in the run directory
#   every:N     every checkpoint whose iteration is a multiple of N
# Env: CUDA_VISIBLE_DEVICES (all GPUs of the job), SPLIT (val_unseen), NUM_ENVS,
#      EVAL_LANGUAGES, SKIP_DONE (1: skip iters with a result, default 1).
set -euo pipefail
SETTING="${1:-}"; [[ "${SETTING}" =~ ^(en|all)$ ]] || { echo "Usage: $0 {en|all} [ITER ...|all|every:N]" >&2; exit 2; }
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT SETTING
source "${REPO_ROOT}/scripts/distill/rxr_kd_env.sh"
TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"
[[ -x "${TORCHRUN}" ]] || { echo "torchrun not found; set TORCHRUN" >&2; exit 1; }
PYTHON="$(dirname "${TORCHRUN}")/python"
SPLIT="${SPLIT:-val_unseen}"
LANG_TAG=""
[[ -n "${EVAL_LANGUAGES:-}" ]] && LANG_TAG="_$(printf '%s' "${EVAL_LANGUAGES}" | tr -c 'A-Za-z0-9' '-' | sed 's/-*$//')"
RUN_DIR="${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}"
[[ -d "${RUN_DIR}" ]] || { echo "Run directory not found: ${RUN_DIR}" >&2; exit 1; }

result_json() { echo "${REPO_ROOT}/data/logs/checkpoints/${RUN_NAME}_eval_iter$1_${SPLIT}${LANG_TAG}/eval_results/stats_ckpt_$1_${SPLIT}.json"; }
all_iters() { ls -1 "${RUN_DIR}"/ckpt.iter*.pth 2>/dev/null | sed -E 's/.*ckpt\.iter([0-9]+)\.pth/\1/' | sort -n; }

if [[ $# -eq 0 ]]; then
    echo "Checkpoints of ${RUN_NAME} (${SPLIT}${LANG_TAG}):"
    for it in $(all_iters); do
        if [[ -f "$(result_json "${it}")" ]]; then echo "  iter ${it}  evaluated"; else echo "  iter ${it}"; fi
    done
    exit 0
fi

ITERS=()
if [[ "$1" == "all" ]]; then
    mapfile -t ITERS < <(all_iters)
elif [[ "$1" =~ ^every:([0-9]+)$ ]]; then
    step="${BASH_REMATCH[1]}"
    mapfile -t ITERS < <(all_iters | awk -v s="${step}" '$1 % s == 0')
else
    ITERS=("$@")
fi
[[ ${#ITERS[@]} -gt 0 ]] || { echo "No checkpoints selected" >&2; exit 1; }

DRIVER_LOG="${RUN_DIR}/eval_iters_${SPLIT}${LANG_TAG}_$(date +%Y%m%d-%H%M%S).log"
{
    echo "run=${RUN_NAME} split=${SPLIT} languages=${EVAL_LANGUAGES:-<setting>} gpus=${CUDA_VISIBLE_DEVICES:-?} iters=${ITERS[*]}"
    for it in "${ITERS[@]}"; do
        if [[ "${SKIP_DONE:-1}" == "1" && -f "$(result_json "${it}")" ]]; then
            echo "iter ${it}: result exists, skipping"; continue
        fi
        echo "iter ${it}: start $(date +%F' '%T)"
        if bash "${REPO_ROOT}/scripts/distill/eval_rxr_kd.sh" "${SETTING}" "${it}" > /dev/null 2>&1; then
            echo "iter ${it}: done  $(date +%F' '%T)"
        else
            echo "iter ${it}: FAILED, see ${RUN_DIR}/eval_iter${it}_${SPLIT}${LANG_TAG}.log"
        fi
    done
    cd "${REPO_ROOT}"
    "${PYTHON}" scripts/distill/summarize_eval.py --run-name "${RUN_NAME}" --split "${SPLIT}" \
        --glob "data/logs/checkpoints/${RUN_NAME}_eval_iter*_${SPLIT}${LANG_TAG}/eval_results/stats_ckpt_*_${SPLIT}.json"
} 2>&1 | tee "${DRIVER_LOG}"
echo "Driver log: ${DRIVER_LOG}"
