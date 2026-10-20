#!/usr/bin/env bash
# Read-only readiness check for both RxR KD settings; CPU only (login node is
# fine, nothing touches a GPU).  Run after prepare_rxr_kd.sh, before sbatch.
#
#   bash scripts/distill/check_rxr_kd.sh            # en and all, full map scan
#   SETTINGS=en bash scripts/distill/check_rxr_kd.sh
#   MAX_LOAD=2000 bash scripts/distill/check_rxr_kd.sh   # sample the map loads
#
# Writes the full report to data/logs/check_rxr_kd_<time>.log; exit code 0
# only when there is no FAIL.
set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT
cd "${REPO_ROOT}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

ARGS=()
for s in ${SETTINGS:-en all}; do
    # Exact torchrun command the job will run (launcher stops before torchrun).
    (
        export SETTING="${s}" CUDA_VISIBLE_DEVICES=0,1,2,3 TORCHRUN="$(command -v torchrun)"
        source scripts/distill/rxr_kd_env.sh
        env | grep -E '^(GT_TEACHER_|TEACHER_NS_SRC|PRETRAINED_CKPT|LLM_|DATASET_SUFFIX|EXP_CONFIG|RUN_NAME|ITERS|CKPT_INTERVAL|NUM_ENVS|MASTER_PORT)=' \
            | sort > "${TMP}/${s}.env"
        EMIT_OPTS="${TMP}/${s}.cmd" bash scripts/distill/run_dagger_distill_legacy.sh kl > "${TMP}/${s}.emit.log" 2>&1 \
            || { cat "${TMP}/${s}.emit.log" >&2; exit 1; }
    )
    ARGS+=(--setting "${s}" "${TMP}/${s}.cmd" "${TMP}/${s}.env")
done

mkdir -p data/logs
LOG="data/logs/check_rxr_kd_$(date +%Y%m%d_%H%M%S).log"
set +e
CUDA_VISIBLE_DEVICES='' PYTHONPATH="${REPO_ROOT}" python scripts/distill/check_rxr_kd.py \
    "${ARGS[@]}" --max-load "${MAX_LOAD:-0}" 2>&1 | tee "${LOG}"
status="${PIPESTATUS[0]}"
echo "report: ${REPO_ROOT}/${LOG}"
exit "${status}"
