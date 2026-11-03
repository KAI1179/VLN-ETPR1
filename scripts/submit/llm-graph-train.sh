#!/usr/bin/env bash
#SBATCH --job-name=llm-graph-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH -p vip_gpu_scze096
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
DRY_RUN=false
if [[ $# == 1 && "$1" == --dry-run ]]; then DRY_RUN=true
elif [[ $# != 0 ]]; then echo "Usage: $0 [--dry-run]" >&2; exit 2; fi
if [[ "${DRY_RUN}" == false && -n "${SLURM_JOB_ID:-}" && -z "${PYTHON:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate etpr1-uv
fi
PYTHON="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONHASHSEED=42
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${REPO_ROOT}/data/models/Llama-3.1-8B-Instruct}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC="${#VISIBLE_GPUS[@]}"
elif [[ "${DRY_RUN}" == true ]]; then
    NPROC=8  # CPU-only preview of the intended eight-GPU allocation.
else
    NPROC="$(${PYTHON} -c 'import torch; print(torch.cuda.device_count())')"
fi
[[ "${NPROC}" =~ ^[1-9][0-9]*$ && "${CUDA_VISIBLE_DEVICES:-0}" != -1 ]] \
    || { echo "No visible GPUs; set CUDA_VISIBLE_DEVICES to the intended allocation." >&2; exit 1; }
ARGS=(
    -m vlnce_baselines.models.etp_llm.llm_graph_train
    --model-name-or-path "${MODEL_NAME_OR_PATH}"
    --training-data data_v2/llm_graph/train.jsonl
    --per-device-batch-size 1 --gradient-accumulation-steps 1
    --gradient-checkpointing --device-map none
    --max-input-length 1152 --max-new-tokens 640 --max-sequence-length 1792
    --max-dropped-fraction 0.005 --cuda-cache-clear-min-sequence-length 1024
    --epochs 10 --seed 42 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05
    --output-dir "${OUTPUT_DIR:-data_v2/llm_graph/runs/r2r}"
)
COMMAND=("${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NPROC}" "${ARGS[@]}")
printf 'Resolved torchrun (training is not started by --dry-run):\n'
printf 'env'
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then printf ' CUDA_VISIBLE_DEVICES=%q' "${CUDA_VISIBLE_DEVICES}"; fi
printf ' %q' "${COMMAND[@]}"; printf '\n'
if [[ "${DRY_RUN}" == true ]]; then
    export CUDA_VISIBLE_DEVICES=""
    exec "${PYTHON}" "${ARGS[@]}" --dry-run \
        --tokenizer-name-or-path "${TOKENIZER_NAME_OR_PATH:-${REPO_ROOT}/data_v2/tokenizers/llama-3.1-8b}"
fi
if [[ "${MODEL_NAME_OR_PATH}" == /* && ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
    echo "Set MODEL_NAME_OR_PATH to the Llama-3.1-8B-Instruct base weights; this copy contains tokenizer files only." >&2
    exit 1
fi
exec "${COMMAND[@]}"
