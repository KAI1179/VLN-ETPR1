#!/usr/bin/env bash
# RxR-CE final student: DAgger from the LLM-map try5 pretrain (460000), plain
# action KL from the frozen RxR GT teacher (iter 15000), no self-distillation.
# Discrete (legacy) LLM map, no coordinate fusion / positional encoding, dy.
# The student executes the actions (DAgger mixes expert and student actions
# with IL.sample_ratio); the teacher only supplies the KL target.
#
# Usage: bash scripts/distill/run_rxr_kd.sh {en|all} [--dry-run]
#   en   RxR guide, en-US + en-IN, LLM maps from the English-instruction cache
#   all  RxR guide, all languages; non-English episodes read the LLM maps
#        generated from translated instructions (merged key, symlinks)
#
# Env overrides (ln01 defaults): GT_TEACHER_CKPT, TEACHER_NS_SRC (directory of
# the teacher GT namespace), PRETRAINED_CKPT, LLM_CACHE_DIR, LLM_EN_KEY,
# LLM_TRANSLATED_KEY, RUN_NAME, ITERS (30000), CKPT_INTERVAL (1000),
# MASTER_PORT (en 29710 / all 29720), NUM_ENVS (4), DATASET_SUFFIX (_90).
set -euo pipefail
SETTING="${1:-}"
case "${SETTING}" in en|all) ;; *) echo "Usage: $0 {en|all} [--dry-run]" >&2; exit 2 ;; esac
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set the four allocated GPU ids explicitly}"
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
[[ ${#GPUS[@]} -eq 4 ]] || { echo 'This experiment requires four GPUs' >&2; exit 1; }

export GT_TEACHER_CKPT="${GT_TEACHER_CKPT:-${HOME}/run/ETP-R1/data/logs/checkpoints/priorgt_rxr_native_policy_translated_map_dagger/ckpt.iter15000.pth}"
export GT_TEACHER_NAMESPACE="gt.legacy.r1p5.direction5.rxr-translated-mention-97525891.v1"
TEACHER_NS_SRC="${TEACHER_NS_SRC:-${HOME}/run/ETP-R1/data/cognitive_maps/${GT_TEACHER_NAMESPACE}}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${HOME}/run/ETP-R1/pretrained/r2r_rxr_ce/llm_grid_try5/store2/model_step_460000.pt}"
export LLM_CACHE_DIR="${LLM_CACHE_DIR:-${HOME}/ETP-R1/data/llm_navigation}"
LLM_EN_KEY="${LLM_EN_KEY:-llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree}"
LLM_TRANSLATED_KEY="${LLM_TRANSLATED_KEY:-llm-grid-rxr-translated-hi-te-grid2-epoch10-v1}"
LLM_MERGED_KEY="llm-grid-rxr-all-en-tagfree-hite-translated-v1"

# Teacher GT maps are read from <repo>/data/cognitive_maps/<namespace>.
NS_DST="${REPO_ROOT}/data/cognitive_maps/${GT_TEACHER_NAMESPACE}"
if [[ ! -e "${NS_DST}" ]]; then
    [[ -d "${TEACHER_NS_SRC}/raster" ]] || { echo "Teacher namespace not found: ${TEACHER_NS_SRC}" >&2; exit 1; }
    ln -s "${TEACHER_NS_SRC}" "${NS_DST}"
    echo "Linked ${NS_DST} -> ${TEACHER_NS_SRC}"
fi

PY="$(dirname "${TORCHRUN}")/python"
if [[ "${SETTING}" == "en" ]]; then
    export LLM_CACHE_KEY="${LLM_EN_KEY}"
    export MASTER_PORT="${MASTER_PORT:-29710}"
else
    export LLM_CACHE_KEY="${LLM_MERGED_KEY}"
    export MASTER_PORT="${MASTER_PORT:-29720}"
    "${PY}" "${REPO_ROOT}/scripts/distill/merge_rxr_llm_caches.py" --cache-dir "${LLM_CACHE_DIR}" \
        --en-key "${LLM_EN_KEY}" --translated-key "${LLM_TRANSLATED_KEY}" --merged-key "${LLM_MERGED_KEY}"
fi

export DATASET_SUFFIX="${DATASET_SUFFIX:-_90}"
export EXP_CONFIG="${REPO_ROOT}/run_rxr/iter_train.yaml,${REPO_ROOT}/run_rxr/kd_${SETTING}.yaml"
export GT_TEACHER_POLICY_NAME="PriorGTTry5Policy"
export RUN_NAME="${RUN_NAME:-kd_rxr_${SETTING}_gt15000_kl}"
export ITERS="${ITERS:-30000}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-1000}"
export NUM_ENVS="${NUM_ENVS:-4}"
export LOAD_MAP=False
export DISTILL_WEIGHT=1.0 TEMP=1.0

CUDA_VISIBLE_DEVICES='' "${PY}" "${REPO_ROOT}/scripts/distill/check_rxr_kd.py" --setting "${SETTING}" --require
bash "${REPO_ROOT}/scripts/distill/run_dagger_distill_legacy.sh" kl "$@"
