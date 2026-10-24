#!/usr/bin/env bash
# RxR-CE final student: DAgger from the LLM-map try5 pretrain (460000), plain
# action KL from the frozen RxR GT teacher (iter 15000), no self-distillation.
# Discrete (legacy) LLM map, no coordinate fusion / positional encoding, dy.
# The student executes the actions (DAgger mixes expert and student actions
# with IL.sample_ratio); the teacher only supplies the KL target.
#
# Order on ln01: prepare_rxr_kd.sh (once) -> check_rxr_kd.sh -> sbatch
# scripts/submit/rxr-kd.sh.  This script only launches; all paths and settings
# come from rxr_kd_env.sh.
#
# Usage: bash scripts/distill/run_rxr_kd.sh {en|all} [--dry-run]
#   en   RxR guide, en-US + en-IN, LLM maps from the English-instruction cache
#   all  RxR guide, all languages, merged LLM cache key (see prepare_rxr_kd.sh)
set -euo pipefail
SETTING="${1:-}"
case "${SETTING}" in en|all) ;; *) echo "Usage: $0 {en|all} [--dry-run]" >&2; exit 2 ;; esac
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set the four allocated GPU ids explicitly}"
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
[[ ${#GPUS[@]} -eq 4 ]] || { echo 'This experiment requires four GPUs' >&2; exit 1; }

source "${REPO_ROOT}/scripts/distill/rxr_kd_env.sh"
export TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"
[[ -x "${TORCHRUN}" ]] || { echo "torchrun not found; set TORCHRUN" >&2; exit 1; }
bash "${REPO_ROOT}/scripts/distill/run_dagger_distill_legacy.sh" kl "$@"
