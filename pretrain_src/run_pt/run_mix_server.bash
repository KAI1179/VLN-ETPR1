NODE_RANK=0
NUM_GPUS=4
PORT=$1
shift

outdir=pretrained/r2r_rxr_ce/baseline
args=()
for arg in "$@"; do
    case "$arg" in
        --use_prior_gt)
            outdir=pretrained/r2r_rxr_ce/prior_gt
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

echo "Using port: $PORT"
echo "Output dir: $outdir"
echo "Args: ${args[*]}"

PYTHONPATH=$PYTHONPATH:. python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank "$NODE_RANK" --master_port="$PORT" \
    pretrain_src/pretrain_src/train_r2r.py --world_size "${NUM_GPUS}" \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/mix_model_config_dep.json \
    --config pretrain_src/run_pt/mix_pretrain_server.json \
    --output_dir "$outdir" \
    "${args[@]}"
