#!/usr/bin/env bash
# One-time CPU setup for both RxR KD settings (login node is fine, no GPU):
#   1. link the teacher GT namespace into <repo>/data/cognitive_maps/
#   2. build the all-language LLM cache key by symlinks
#      (en-* -> English-instruction cache, other languages -> translated cache)
# Idempotent; the source caches are never modified.  Run check_rxr_kd.sh after.
set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT
cd "${REPO_ROOT}"
SETTING=all source "${REPO_ROOT}/scripts/distill/rxr_kd_env.sh"

NS_DST="${REPO_ROOT}/data/cognitive_maps/${GT_TEACHER_NAMESPACE}"
if [[ -e "${NS_DST}" ]]; then
    echo "teacher namespace present: ${NS_DST} -> $(readlink -f "${NS_DST}")"
else
    [[ -d "${TEACHER_NS_SRC}/raster" ]] || { echo "teacher namespace not found: ${TEACHER_NS_SRC}" >&2; exit 1; }
    ln -s "${TEACHER_NS_SRC}" "${NS_DST}"
    echo "linked ${NS_DST} -> ${TEACHER_NS_SRC}"
fi

# ln01 kills login-node processes above 4 cores; the merge is I/O-bound.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="${REPO_ROOT}" \
    python "${REPO_ROOT}/scripts/distill/merge_rxr_llm_caches.py" --workers "${MERGE_WORKERS:-8}" \
    --cache-dir "${LLM_CACHE_DIR}" --en-key "${LLM_EN_KEY}" \
    --translated-key "${LLM_TRANSLATED_KEY}" --merged-key "${LLM_MERGED_KEY}"
