#!/usr/bin/env bash
# Run the whole training-free GT map ablation set on one PriorGT try5
# checkpoint, dealing the jobs round-robin onto GPUS (each GPU works through
# its share sequentially), then print a summary table against the own-map
# reference and the paired CI of every ablation vs own.
#
# Usage: bash scripts/distill/eval_gt_ablations.sh [TAG]
#   TAG  result-directory prefix (default gt18600)
# Env: CKPT_PATH (default: the GT-legacy 18600 ckpt), GPUS (default 1,5,6,7),
#      NAMESPACE (default gt.legacy.r1p5.direction5.v1),
#      OWN_DIR (eval_results dir of the own-map run used as the paired
#      reference; default data/logs/checkpoints/<TAG>_legacy_val_unseen_own/eval_results),
#      JOBS (override the job list, space separated "namespace[:map_ablation]").
# Jobs whose result json already exists are skipped, so re-running after a
# failure only does the missing ones.
set -euo pipefail

REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
PYTHON="/home/xukai/anaconda3/envs/etpr1-py38/bin/python"
EVAL="${REPO_ROOT}/scripts/distill/eval_gt_namespace.sh"
TAG="${1:-gt18600}"
export CKPT_PATH="${CKPT_PATH:-/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth}"
NS="${NAMESPACE:-gt.legacy.r1p5.direction5.v1}"
OWN_DIR="${OWN_DIR:-${REPO_ROOT}/data/logs/checkpoints/${TAG}_legacy_val_unseen_own/eval_results}"
IFS=',' read -ra GPU_LIST <<< "${GPUS:-1,5,6,7}"

# "namespace[:map_ablation]"
DEFAULT_JOBS=(
    "${NS}.abl_occupancy_only"
    "${NS}.abl_cell_shuffle_s0"
    "${NS}.abl_regions_only"
    "${NS}.abl_objects_only"
    "${NS}.abl_category_scramble_s0"
    "${NS}.abl_unmentioned_only"
    "${NS}.mentioned_only"
    "${NS}:metadata_only"
    "${NS}:raster_only"
    "${NS}:no_direction"
)
read -ra JOB_LIST <<< "${JOBS:-${DEFAULT_JOBS[*]}}"

cd "${REPO_ROOT}"
[[ -f "${CKPT_PATH}" ]] || { echo "Missing CKPT_PATH ${CKPT_PATH}" >&2; exit 1; }
DRIVER_DIR="${REPO_ROOT}/data/logs/checkpoints/${TAG}_ablations"
mkdir -p "${DRIVER_DIR}"
DRIVER_LOG="${DRIVER_DIR}/driver.log"
echo "$(date -Is) ckpt=${CKPT_PATH} gpus=${GPU_LIST[*]} jobs=${#JOB_LIST[@]}" | tee -a "${DRIVER_LOG}"

job_dir() {  # -> eval dir of a job
    local ns="${1%%:*}" abl=""
    [[ "$1" == *:* ]] && abl="${1##*:}"
    local name="${TAG}_ns_${ns}"
    [[ -z "${abl}" ]] || name="${name}_abl_${abl}"
    echo "${REPO_ROOT}/data/logs/checkpoints/${name}"
}
job_done() { ls "$(job_dir "$1")"/eval_results/stats_ckpt_*_val_unseen.json >/dev/null 2>&1; }

# Preflight: every namespace must exist before any GPU time is spent.
for job in "${JOB_LIST[@]}"; do
    ns="${job%%:*}"
    [[ -d "${REPO_ROOT}/data/cognitive_maps/${ns}/raster" ]] \
        || { echo "Namespace missing: ${ns} (run make_ablated_gt_cache.py / make_mentioned_only_gt_cache.py)" >&2; exit 1; }
done

run_share() {  # gpu, jobs...
    local gpu="$1"; shift
    for job in "$@"; do
        if job_done "${job}"; then
            echo "[gpu ${gpu}] ${job}: already evaluated" | tee -a "${DRIVER_LOG}"
            continue
        fi
        local ns="${job%%:*}" abl=none
        [[ "${job}" == *:* ]] && abl="${job##*:}"
        echo "[gpu ${gpu}] $(date -Is) start ${job}" | tee -a "${DRIVER_LOG}"
        if CUDA_VISIBLE_DEVICES="${gpu}" MAP_ABLATION="${abl}" bash "${EVAL}" "${TAG}" "${ns}" >>"${DRIVER_DIR}/gpu${gpu}.log" 2>&1; then
            echo "[gpu ${gpu}] $(date -Is) done  ${job}" | tee -a "${DRIVER_LOG}"
        else
            echo "[gpu ${gpu}] $(date -Is) FAILED ${job}, see $(job_dir "${job}")/eval.log" | tee -a "${DRIVER_LOG}"
        fi
    done
}

# Deal jobs round-robin onto the GPUs and run each share in the background.
declare -a SHARES
for i in "${!JOB_LIST[@]}"; do
    g=$(( i % ${#GPU_LIST[@]} ))
    SHARES[g]="${SHARES[g]:-} ${JOB_LIST[i]}"
done
for g in "${!GPU_LIST[@]}"; do
    [[ -n "${SHARES[g]:-}" ]] || continue
    # shellcheck disable=SC2086
    run_share "${GPU_LIST[g]}" ${SHARES[g]} &
done
wait

# Summary: metrics table plus paired SR delta vs own with scene-bootstrap CI.
"${PYTHON}" - "${OWN_DIR}" "${TAG}" "${REPO_ROOT}" "${JOB_LIST[@]}" <<'PY' | tee "${DRIVER_DIR}/summary.md"
import glob, json, subprocess, sys
own_dir, tag, root, *jobs = sys.argv[1:]

def stats(d):
    files = glob.glob(f"{d}/stats_ckpt_*_val_unseen.json")
    return json.load(open(files[0])) if files else None

def paired(d):
    try:
        out = subprocess.run([sys.executable, f"{root}/scripts/distill/paired_analysis.py",
                              "--dir-a", own_dir, "--dir-b", d, "--reps", "5000"],
                             capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if line.startswith("| SR "):
                cells = [c.strip() for c in line.strip("|").split("|")]
                return f"{cells[3]} {cells[4]}"
        return "n/a"
    except subprocess.CalledProcessError:
        return "n/a"

rows = [("own (reference)", stats(own_dir), "")]
for job in jobs:
    ns, _, abl = job.partition(":")
    name = f"{tag}_ns_{ns}" + (f"_abl_{abl}" if abl else "")
    d = f"{root}/data/logs/checkpoints/{name}/eval_results"
    label = ns.split(".")[-1] + (f" + {abl}" if abl else "")
    m = stats(d)
    rows.append((label, m, paired(d) if m else ""))
print("| variant | SR | SPL | OSR | nDTW | NE | PL | SR - own [95% CI] |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for label, m, ci in rows:
    if m is None:
        print(f"| {label} | missing | | | | | | |")
        continue
    print(f"| {label} | {m['success']*100:.2f} | {m['spl']*100:.2f} | {m['oracle_success']*100:.2f} | "
          f"{m['ndtw']*100:.2f} | {m['distance_to_goal']:.2f} | {m['path_length']:.1f} | {ci} |")
PY
echo "Summary: ${DRIVER_DIR}/summary.md"
