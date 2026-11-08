#!/usr/bin/env bash
#SBATCH --job-name=cogmap-g1-perturb
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH -p vip_gpu_scze096
# Training-free map-error probe for a G1 checkpoint.
#   1. Render the perturbed whole-box namespaces for val_unseen (CPU, in-job).
#   2. Evaluate one checkpoint on each namespace, 4 GPUs, one eval per GPU,
#      each GPU working through its share serially.
#   3. Print a table against the unperturbed result.
# Usage: sbatch scripts/submit/g1-perturb-ln01.sh [RUN_NAME] [ITER]
#        (default dagger_g1_box_noreg 30000)
# Env: PRETRAINED_CKPT (default: G1 arms' 440000), MODES (override the list below).
set -eo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run.py" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
cd "${REPO_ROOT}"

RUN_NAME="${1:-dagger_g1_box_noreg}"
ITER="${2:-30000}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/data/home/scze096/run/ETP-R1/pretrained/r2r_rxr_ce/prior_gt_r1p5_nogaussian/ckpts/model_step_440000.pt}"
export OUTPUT_ROOT="${REPO_ROOT}/data_v2/runs"
export PYTHON="${PYTHON:-python}"
export SPLIT=val_unseen
# mode[:magnitude]
MODES="${MODES:-cross_scene mentioned_only shift:0.5 shift:1 shift:2 rotate:15 rotate:30 scale:0.8 scale:1.2 jitter:0.5 jitter:1}"

namespace_of() {  # mode[:magnitude] -> namespace
    local mode="${1%%:*}" mag=""
    [[ "$1" == *:* ]] && mag="${1#*:}"
    "${PYTHON}" -c "from prior.cogmap_v2.perturb import namespace_for; import sys; print(namespace_for(sys.argv[1], float(sys.argv[2]) if sys.argv[2] else None))" "${mode}" "${mag}"
}

echo "== unit tests"
CUDA_VISIBLE_DEVICES='' "${PYTHON}" -m pytest -q tests/test_cogmap_v2_perturb.py

echo "== render perturbed namespaces (${SPLIT})"
declare -a NAMESPACES=()
for spec in ${MODES}; do
    mode="${spec%%:*}"; mag=""; [[ "${spec}" == *:* ]] && mag="${spec#*:}"
    ns="$(namespace_of "${spec}")"
    NAMESPACES+=("${ns}")
    if [[ -f "data_v2/cognitive_maps/${ns}/perturb_summary_${SPLIT}.json" ]]; then
        echo "   ${ns}: exists, skipping render"; continue
    fi
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "${PYTHON}" -m prior.cogmap_v2.perturb \
        --split "${SPLIT}" --mode "${mode}" ${mag:+--magnitude "${mag}"} --seed 0
done

echo "== evaluate ${RUN_NAME} iter ${ITER} on ${#NAMESPACES[@]} namespaces"
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
declare -a SHARES
for ((i = 0; i < ${#GPU_LIST[@]}; i++)); do SHARES[i]=""; done
for ((i = 0; i < ${#NAMESPACES[@]}; i++)); do
    g=$((i % ${#GPU_LIST[@]})); SHARES[g]="${SHARES[g]} ${NAMESPACES[i]}"
done
result_json() { echo "${OUTPUT_ROOT}/checkpoints/${RUN_NAME}_eval_iter${ITER}_${SPLIT}_$1/eval_results/stats_ckpt_${ITER}_${SPLIT}.json"; }
worker() {
    local gpu="$1"; shift
    for ns in "$@"; do
        local tag="${ns#gt.graph.box.}"; tag="${tag%.direction5.v1}"; tag="${tag//./_}"
        if [[ -f "$(result_json "${tag}")" ]]; then echo "[gpu ${gpu}] ${tag}: already evaluated"; continue; fi
        echo "[gpu ${gpu}] ${tag}: start $(date +%H:%M:%S)"
        if CUDA_VISIBLE_DEVICES="${gpu}" EVAL_TAG="${tag}" \
            EXTRA_ARGS="MODEL.MAP_ENCODER.cache_namespace ${ns}" \
            bash scripts/distill/eval_run.sh "${RUN_NAME}" "${ITER}" "${PRETRAINED_CKPT}" >/dev/null 2>&1; then
            echo "[gpu ${gpu}] ${tag}: done  $(date +%H:%M:%S)"
        else
            echo "[gpu ${gpu}] ${tag}: FAILED (see ${OUTPUT_ROOT}/checkpoints/${RUN_NAME}/eval_iter${ITER}_${SPLIT}_${tag}.log)"
        fi
    done
}
for ((g = 0; g < ${#GPU_LIST[@]}; g++)); do
    [[ -n "${SHARES[g]}" ]] && worker "${GPU_LIST[g]}" ${SHARES[g]} &
done
wait

echo
echo "# ${RUN_NAME} iter ${ITER} on ${SPLIT}: map-error probe"
"${PYTHON}" - "${OUTPUT_ROOT}/checkpoints" "${RUN_NAME}" "${ITER}" "${SPLIT}" "${NAMESPACES[@]}" <<'PY'
import json, sys, pathlib
root, run, it, split, *namespaces = sys.argv[1:]
root = pathlib.Path(root)
def load(d):
    p = root / d / "eval_results" / f"stats_ckpt_{it}_{split}.json"
    return json.load(open(p)) if p.exists() else None
base = load(f"{run}_eval_iter{it}_{split}")
rows = [("unperturbed (gt.graph.box)", base)]
for ns in namespaces:
    tag = ns[len("gt.graph.box."):-len(".direction5.v1")].replace(".", "_")
    rows.append((tag, load(f"{run}_eval_iter{it}_{split}_{tag}")))
print("| condition | SR | SPL | OSR | NE | ΔSR |")
print("|---|---:|---:|---:|---:|---:|")
for name, m in rows:
    if m is None:
        print(f"| {name} | — | — | — | — | — |"); continue
    d = "" if base is None else f"{(m['success']-base['success'])*100:+.2f}"
    print(f"| {name} | {m['success']*100:.2f} | {m['spl']*100:.2f} | {m['oracle_success']*100:.2f} | {m['distance_to_goal']:.2f} | {d} |")
PY
