#!/usr/bin/env bash
# Same dual objective and zero-branch rollout as kd_dual; RxR-English + joint GT teacher.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set the four allocated GPU ids explicitly}"
export GT_TEACHER_CKPT="${GT_TEACHER_CKPT:?Set path to joint ckpt.iter30000.pth}"
export PRETRAINED_CKPT="${PRETRAINED_CKPT:?Set path to LLM model_step_460000.pt}"
export EXP_CONFIG="${REPO_ROOT}/run_rxr/iter_train.yaml,${REPO_ROOT}/run_rxr/dual_gt30k_en.yaml"
export RUN_NAME="${RUN_NAME:-kd_dual_rxr_en_gt30k}"
export ITERS="${ITERS:-30000}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-200}"
export MASTER_PORT="${MASTER_PORT:-29710}"
export NUM_ENVS="${NUM_ENVS:-4}"
export LOAD_MAP=False
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
[[ ${#GPUS[@]} -eq 4 ]] || { echo 'This experiment requires four GPUs' >&2; exit 1; }
"$(dirname "${TORCHRUN}")/python" "${REPO_ROOT}/scripts/distill/check_rxr_en_dual.py" --require-cache
bash "${REPO_ROOT}/scripts/distill/run_dagger_distill_legacy.sh" dual "$@"
