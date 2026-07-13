NODE_RANK=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/gpu-detection.bash"
configure_pretrain_gpu_vars
PORT=$1
OUTPUT_DIR=$2
shift 2

if [ -z "$PORT" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 PORT OUTPUT_DIR [training arguments...]" >&2
    exit 2
fi

echo "Using port: $PORT"
echo "Output dir: $OUTPUT_DIR"
echo "Args: $*"

PYTHONPATH=$PYTHONPATH:. python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank "$NODE_RANK" --master_port="$PORT" \
    pretrain_src/pretrain_src/train_r2r.py --world_size "${NUM_GPUS}" \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/mix_model_config_dep.json \
    --config pretrain_src/run_pt/mix_pretrain_server.json \
    --output_dir "$OUTPUT_DIR" \
    "$@"
