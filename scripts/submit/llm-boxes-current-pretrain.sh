#!/bin/bash
#SBATCH --job-name=llm-boxes-current-pt
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash pretrain_src/run_pt/run_mix_server.bash \
  pretrained/r2r_rxr_ce/llm_boxes_current \
  --navigation-architecture current \
  --cognitive-map-source llm_boxes \
  --llm-cache-model-key llm-boxes-r1p5-path5 \
  --checkpoint pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt
