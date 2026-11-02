#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/home/xukai/code/ETP-R1-snapshot/ETP-R1"
export GPUS="${GPUS:-4,5,6,7}"
bash "${REPO_ROOT}/scripts/eval_d7/eval_kd_deploy.sh"
bash "${REPO_ROOT}/scripts/eval_d7/eval_remove_kd.sh"
bash "${REPO_ROOT}/scripts/eval_d7/eval_remove_28000.sh"
