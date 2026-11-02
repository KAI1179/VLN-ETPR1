#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
SCRIPTS=(
  "${ROOT}/scripts/distill/eval_vs_28000_own.sh"
  "${ROOT}/scripts/distill/eval_vs_28000_cross.sh"
  "${ROOT}/scripts/eval_gt18600/eval_vs_18600_gtown.sh"
  "${ROOT}/scripts/eval_gt18600/eval_vs_18600_gtcross.sh"
  "${ROOT}/scripts/eval_gt18600/eval_vs_18600_llm.sh"
)
RUNS=(
  d4_vs_28000_own
  d4_vs_28000_cross
  d4_vs_18600_gtown
  d4_vs_18600_gtcross
  d4_vs_18600_llm
)

for index in "${!SCRIPTS[@]}"; do
  bash "${SCRIPTS[index]}"
  "${PYTHON}" -c '
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
files = sorted((run / "eval_results").glob("stats_ep_ckpt_*_val_seen_r*_w4.json"))
assert len(files) == 4, (run, files)
episodes = {}
for file in files:
    part = json.loads(file.read_text())
    assert not (part.keys() & episodes.keys()), file
    episodes.update(part)
assert len(episodes) == 778, (run, len(episodes))
print(f"{run.name}: {len(episodes)} episodes, {len(files)} rank files")
' "${ROOT}/data/logs/checkpoints/${RUNS[index]}"
done
