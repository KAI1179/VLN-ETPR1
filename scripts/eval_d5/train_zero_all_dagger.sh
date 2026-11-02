#!/usr/bin/env bash
# Matched HEAD DAgger control for scripts/distill/train_llm_full_dagger_control.sh.
set -euo pipefail

: "${D3_GPUS:?Set D3_GPUS to four GPU IDs before running}"
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/train_llm_d3.sh \
    dagger_llm_zero_all_control \
    llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__zero-all \
    29664
