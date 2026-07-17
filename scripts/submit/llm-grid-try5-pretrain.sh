#!/bin/bash
#SBATCH --job-name=llm-grid-try5-pt
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash pretrain_src/run_pt/run_mix_server.bash \
  pretrained/r2r_rxr_ce/llm_grid_try5 \
  --navigation-architecture try5 \
  --cognitive-map-source llm_grid \
  --llm-cache-model-key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
  --checkpoint pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt
