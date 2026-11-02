#!/usr/bin/env bash
set -euo pipefail
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/train_llm_d3.sh \
  dagger_llm_region_only llm.region_only.s2tagfree.v1 29660
