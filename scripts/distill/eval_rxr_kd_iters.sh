#!/usr/bin/env bash
# Evaluate several checkpoints of one RxR KD run, then print the summary table.
#
# Usage: bash scripts/distill/eval_rxr_kd_iters.sh SETTING [ITER ...]
#   no ITER     list checkpoints and which already have a result, exit
#   ITER ...    explicit iterations
#   all         every checkpoint in the run directory
#   every:N     every checkpoint whose iteration is a multiple of N
# Env: CUDA_VISIBLE_DEVICES (all GPUs of the job), SPLIT (val_unseen), NUM_ENVS,
#      EVAL_LANGUAGES, SKIP_DONE (1: skip iters with a result, default 1),
#      GPUS_PER_EVAL  GPUs one checkpoint's torchrun uses (default: all, i.e.
#                     one checkpoint at a time over every GPU)
#      EVALS_PER_GPU  checkpoint evaluations sharing each GPU group (default 1)
#   Slots = (#GPUs / GPUS_PER_EVAL) * EVALS_PER_GPU checkpoints run at once;
#   the iterations are dealt round-robin to the slots, each slot runs its share
#   serially.  E.g. 8 GPUs, GPUS_PER_EVAL=1 EVALS_PER_GPU=2 -> 16 at once.
#   Per-episode results do not depend on the slot layout (episodes are split by
#   scene over envs; the policy is greedy).
#   NUM_ENVS is capped by the scenes of the split (habitat splits scenes over
#   envs): val_unseen has 11, so NUM_ENVS <= 11.
#   GPU memory / utilisation and host load are logged every MONITOR_SEC (600).
set -euo pipefail
SETTING="${1:-}"; [[ "${SETTING}" =~ ^(en|all)$ ]] || { echo "Usage: $0 {en|all} [ITER ...|all|every:N]" >&2; exit 2; }
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT SETTING
source "${REPO_ROOT}/scripts/distill/rxr_kd_env.sh"
TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"
[[ -x "${TORCHRUN}" ]] || { echo "torchrun not found; set TORCHRUN" >&2; exit 1; }
export TORCHRUN
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
TODO=()
for it in "${ITERS[@]}"; do
    [[ -f "${RUN_DIR}/ckpt.iter${it}.pth" ]] || { echo "Checkpoint not found: ${RUN_DIR}/ckpt.iter${it}.pth" >&2; exit 1; }
    if [[ "${SKIP_DONE:-1}" == "1" && -f "$(result_json "${it}")" ]]; then
        echo "iter ${it}: result exists, skipping"
    else
        TODO+=("${it}")
    fi
done

# Slot layout
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"
GPUS_PER_EVAL="${GPUS_PER_EVAL:-${#GPUS[@]}}"
EVALS_PER_GPU="${EVALS_PER_GPU:-1}"
(( ${#GPUS[@]} % GPUS_PER_EVAL == 0 )) || { echo "#GPUs (${#GPUS[@]}) not divisible by GPUS_PER_EVAL=${GPUS_PER_EVAL}" >&2; exit 1; }
GPU_GROUPS=()
for ((g = 0; g < ${#GPUS[@]}; g += GPUS_PER_EVAL)); do
    GPU_GROUPS+=("$(IFS=,; echo "${GPUS[*]:g:GPUS_PER_EVAL}")")
done
SLOTS=()
for ((k = 0; k < EVALS_PER_GPU; k++)); do SLOTS+=("${GPU_GROUPS[@]}"); done

# NUM_ENVS cap: habitat gives every env at least one scene of the split
NUM_ENVS="${NUM_ENVS:-4}"; export NUM_ENVS
N_SCENES="$("${PYTHON}" - "${REPO_ROOT}/data/datasets/RxR_VLNCE_v0_enc_xlmr/${SPLIT}/${SPLIT}_guide.json.gz" <<'PY'
import gzip, json, sys
print(len({e["scene_id"] for e in json.load(gzip.open(sys.argv[1], "rt"))["episodes"]}))
PY
)"
(( NUM_ENVS <= N_SCENES )) || { echo "NUM_ENVS=${NUM_ENVS} > ${N_SCENES} scenes in ${SPLIT}; habitat would refuse" >&2; exit 1; }

DRIVER_LOG="${RUN_DIR}/eval_iters_${SPLIT}${LANG_TAG}_$(date +%Y%m%d-%H%M%S).log"
run_slot() {   # slot_id gpus iter...
    local sid="$1" gpus="$2"; shift 2
    sleep $(( sid * 20 ))   # stagger start-up (dataset / scene loading)
    for it in "$@"; do
        echo "[slot ${sid} gpu ${gpus}] iter ${it}: start $(date +%F' '%T)"
        local out
        if out="$(CUDA_VISIBLE_DEVICES="${gpus}" bash "${REPO_ROOT}/scripts/distill/eval_rxr_kd.sh" "${SETTING}" "${it}" 2>&1 | tail -n 3)"; then
            echo "[slot ${sid} gpu ${gpus}] iter ${it}: done  $(date +%F' '%T)  $(grep -o '| SR .*' <<< "${out}")"
        else
            echo "[slot ${sid} gpu ${gpus}] iter ${it}: FAILED, see ${RUN_DIR}/eval_iter${it}_${SPLIT}${LANG_TAG}.log"
        fi
    done
}
monitor() {   # sleeps in 10 s steps so it stops promptly when killed
    local t=0
    while sleep 10; do
        (( (t += 10) % ${MONITOR_SEC:-600} == 0 )) || continue
        echo "[monitor $(date +%T)] load $(cut -d' ' -f1-3 /proc/loadavg) / $(nproc) cpus, mem used $(free -g | awk '/Mem:/{print $3"/"$2}') GiB"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/[monitor]   gpu /'
    done
}

{
    echo "run=${RUN_NAME} split=${SPLIT} languages=${EVAL_LANGUAGES:-<setting>} num_envs=${NUM_ENVS} scenes=${N_SCENES}"
    echo "gpus=${CUDA_VISIBLE_DEVICES} gpus_per_eval=${GPUS_PER_EVAL} evals_per_gpu=${EVALS_PER_GPU} slots=${#SLOTS[@]} cpus=$(nproc) mem=$(free -g | awk '/Mem:/{print $2}')GiB"
    echo "to evaluate: ${TODO[*]:-<none>}"
    if (( ${#TODO[@]} > 0 )); then
        monitor & MON=$!
        pids=()
        for ((s = 0; s < ${#SLOTS[@]}; s++)); do
            mine=()
            for ((i = s; i < ${#TODO[@]}; i += ${#SLOTS[@]})); do mine+=("${TODO[i]}"); done
            (( ${#mine[@]} > 0 )) || continue
            run_slot "${s}" "${SLOTS[s]}" "${mine[@]}" & pids+=($!)
        done
        wait "${pids[@]}" || true
        kill "${MON}" 2>/dev/null || true
    fi
    cd "${REPO_ROOT}"
    "${PYTHON}" scripts/distill/summarize_eval.py --run-name "${RUN_NAME}" --split "${SPLIT}" \
        --glob "data/logs/checkpoints/${RUN_NAME}_eval_iter*_${SPLIT}${LANG_TAG}/eval_results/stats_ckpt_*_${SPLIT}.json"
} 2>&1 | tee "${DRIVER_LOG}"
echo "Driver log: ${DRIVER_LOG}"
