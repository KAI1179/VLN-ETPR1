#!/bin/bash
#SBATCH --job-name=try5-r1p5-blur-pt
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash pretrain_src/run_pt/run_mix_server.bash \
  2333 pretrained/r2r_rxr_ce/prior_gt_try5_r1p5_blurred \
  --navigation-architecture try5 \
  --cognitive-map-source prior_gt \
  --cognitive-map-namespace gt.legacy.r1p5.direction5.blurred.v1 \
  --checkpoint pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt
