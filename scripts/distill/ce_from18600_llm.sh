#!/usr/bin/env bash
# CE-only control: same 18600 student initialization, no GT teacher.
set -euo pipefail

D0_ARM=ce bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/distill_from18600_llm.sh "$@"
