#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${1:?usage: $0 JOB_NAME submission.sh [sbatch args...]}"
SCRIPT="${2:?usage: $0 JOB_NAME submission.sh [sbatch args...]}"
shift 2

mapfile -t JOB_IDS < <(squeue -u "$USER" --name "$JOB_NAME" -a -h -o "%A")

if ((${#JOB_IDS[@]} == 0)); then
    echo "Error: no running/pending job named '$JOB_NAME' found for user '$USER'." >&2
    exit 1
fi

DEP=$(IFS=:; echo "${JOB_IDS[*]}")

echo "Found job(s): ${JOB_IDS[*]}"
echo "Submitting '$SCRIPT' after dependency $JOB_NAME."

sbatch --dependency="afterany:$DEP" "$@" "$SCRIPT"
