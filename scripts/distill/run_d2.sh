#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
SCRIPTS=(
  "${ROOT}/scripts/distill/eval_probe_llm_zero_grid.sh"
  "${ROOT}/scripts/distill/eval_probe_llm_gtlegacy_grid.sh"
  "${ROOT}/scripts/distill/eval_probe_llm_mean_grid.sh"
  "${ROOT}/scripts/distill/eval_probe_llm_region_only.sh"
  "${ROOT}/scripts/eval_gt18600/eval_gt18600_region_only.sh"
  "${ROOT}/scripts/eval_gt18600/eval_gt18600_object_only.sh"
)
RUNS=(
  probe_llm_zero_grid_val_unseen
  probe_llm_gtlegacy_grid_val_unseen
  probe_llm_mean_grid_val_unseen
  probe_llm_region_only_val_unseen
  gt18600_region_only_val_unseen
  gt18600_object_only_val_unseen
)

for index in "${!SCRIPTS[@]}"; do
  bash "${SCRIPTS[index]}"
  "${PYTHON}" -c '
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
files = sorted((run / "eval_results").glob("stats_ep_ckpt_*_val_unseen_r*_w*.json"))
assert files, run
episodes = {}
for file in files:
    part = json.loads(file.read_text())
    assert not (part.keys() & episodes.keys()), file
    episodes.update(part)
assert len(episodes) == 1839, (run, len(episodes))
if run.name.startswith("probe_llm_"):
    missing = sum(row.get("llm_cache_missing", 0) for row in episodes.values())
    assert missing == 6, (run, missing)
print(f"{run.name}: {len(episodes)} episodes, {len(files)} rank files")
' "${ROOT}/data/logs/checkpoints/${RUNS[index]}"
done
