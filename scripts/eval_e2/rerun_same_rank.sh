#!/usr/bin/env bash
# Manual launcher: evaluate the 21600 checkpoint twice with identical rank settings.
set -euo pipefail

ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
export E2_GPUS=5,6,7
export E2_RUN_SUFFIX=_w3

E2_PORT=29620 bash "${ROOT}/scripts/eval_e2/eval_one.sh" 21600 own
E2_PORT=29621 bash "${ROOT}/scripts/eval_e2/eval_one.sh" 21600 cross_scene
