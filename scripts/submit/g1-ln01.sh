#!/usr/bin/env bash
#SBATCH --job-name=cogmap-g1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --nodelist=ln01
#SBATCH -p vip_gpu_scze096
set -eo pipefail
# Under sbatch the script runs from Slurm's spool copy, so resolve the repo from
# the submission directory first and from BASH_SOURCE only for direct runs.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run.py" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ "${2:-}" != --dry-run ]]; then
    eval "$(conda shell.bash hook)"
    conda activate etpr1-uv
fi
set -u
exec bash "${REPO_ROOT}/scripts/distill/run_dagger_g1.sh" "$@"
