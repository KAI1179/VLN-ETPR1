#!/usr/bin/env bash
#SBATCH --job-name=cogmap-v2-gen
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH -p vip_gpu_scze096
# Regenerate the V2 GT data (graphs, both raster namespaces, registration labels)
# on the training host instead of transferring data_v2/ from WZ.  CPU only; the
# single GPU is requested only because the partition is GPU-scheduled.
# Usage: sbatch scripts/submit/cogmap-v2-gen-data.sh
set -eo pipefail
# Under sbatch the script runs from Slurm's spool copy, so resolve the repo from
# the submission directory first and from BASH_SOURCE only for direct runs.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run.py" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
eval "$(conda shell.bash hook)"
conda activate etpr1-uv
set -u
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p data_v2
if [[ ! -s data_v2/features_val_unseen.json ]]; then
    git show origin/data/pa-dumps:features_val_unseen.json > data_v2/features_val_unseen.json \
        || { git fetch origin data/pa-dumps && git show origin/data/pa-dumps:features_val_unseen.json > data_v2/features_val_unseen.json; }
fi

echo "== graphs"
python -m prior.cogmap_v2.graph --split train --calibrate-distance-bins
python -m prior.cogmap_v2.graph --split val_seen
python -m prior.cogmap_v2.graph --split val_unseen

for split in train val_seen val_unseen; do
    echo "== render corridor ${split}"
    python -m prior.cogmap_v2.render --split "${split}" --output-dir data_v2/cognitive_maps
    echo "== render whole-box ${split}"
    python -m prior.cogmap_v2.render --split "${split}" --whole-boxes --output-dir data_v2/cognitive_maps
    echo "== registration ${split}"
    python -m prior.cogmap_v2.registration --split "${split}"
done

echo "== checks"
python scripts/cogmap_v2/check_graph_vs_features.py
python scripts/cogmap_v2/audit_registration.py --split val_unseen
echo "== distance bins"
cat data_v2/cogmap_v2/graphs/train/distance_bins.json
echo
echo "== pretraining checkpoint candidates for G1 (set PRETRAINED_CKPT to one of these)"
find -L "${REPO_ROOT}/pretrained" "${HOME}/run/ETP-R1/pretrained" -name model_step_440000.pt 2>/dev/null | grep -i nogaussian || true
echo "== done"
