SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/gpu-detection.bash"
configure_pretrain_gpu_vars
OUTPUT_DIR=$1
shift

if [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 OUTPUT_DIR [training arguments...]" >&2
    exit 2
fi

echo "Output dir: $OUTPUT_DIR"
echo "Args: $*"

PYTHONPATH=$PYTHONPATH:. torchrun --standalone --nproc-per-node="${NUM_GPUS}" \
    pretrain_src/pretrain_src/train_r2r.py \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/mix_model_config_dep.json \
    --config pretrain_src/run_pt/mix_pretrain_server.json \
    --output_dir "$OUTPUT_DIR" \
    "$@"
