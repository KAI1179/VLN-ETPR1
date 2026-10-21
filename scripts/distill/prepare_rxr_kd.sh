#!/usr/bin/env bash
# One-time CPU setup for both RxR KD settings (login node is fine, no GPU):
#   1. link the teacher GT namespace into <repo>/data/cognitive_maps/
#   2. repair key: maps rejected only for non-unit direction vectors, re-parsed
#      with renormalisation (verified against stored rasters first)
#   3. the en and all LLM cache keys the student reads, by symlinks
#      (en-* -> English-instruction cache, other languages -> translated cache,
#      repair key where those have no complete map)
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

# ln01 kills login-node processes above 4 cores: one compute thread.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="${REPO_ROOT}"

# 2. re-parse maps rejected only for non-unit direction vectors (repair key)
python "${REPO_ROOT}/scripts/distill/repair_rxr_llm_direction_norm.py" --cache-dir "${LLM_CACHE_DIR}" \
    --en-key "${LLM_EN_KEY}" --translated-key "${LLM_TRANSLATED_KEY}" --repair-key "${LLM_REPAIR_KEY}"

# 3. the two symlink keys the student reads (I/O-bound, 8 threads)
for spec in "${LLM_EN_MERGED_KEY}:en-US,en-IN" "${LLM_MERGED_KEY}:*"; do
    key="${spec%%:*}"; IFS=',' read -ra langs <<< "${spec#*:}"
    echo "== merged key ${key} (languages ${langs[*]})"
    python "${REPO_ROOT}/scripts/distill/merge_rxr_llm_caches.py" --workers "${MERGE_WORKERS:-8}" \
        --cache-dir "${LLM_CACHE_DIR}" --en-key "${LLM_EN_KEY}" --translated-key "${LLM_TRANSLATED_KEY}" \
        --repair-key "${LLM_REPAIR_KEY}" --merged-key "${key}" --languages "${langs[@]}"
done
