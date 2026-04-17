#!/bin/bash

# Source - https://stackoverflow.com/a/56431189
# Posted by Lyn, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-17, License - CC BY-SA 4.0

has_param() {
    local term="$1"
    shift
    for arg; do
        if [[ $arg == "$term" ]]; then
            return 0
        fi
    done
    return 1
}

NODE_RANK=0
NUM_GPUS=4
PORT=$1
if has_param '--use_prior_gt' "$@"; then
    outdir=pretrained/r2r_rxr_ce/prior_gt
else
    outdir=pretrained/r2r_rxr_ce/mlm.sap_habitat_depth
fi
echo "Output dir: $outdir"
shift

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port=$PORT \
    pretrain_src/pretrain_src/train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/mix_model_config_dep.json \
    --config pretrain_src/run_pt/mix_pretrain_server.json \
    --output_dir $outdir \
    "$@"

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash pretrain_src/run_pt/run_mix_server.bash 2333
