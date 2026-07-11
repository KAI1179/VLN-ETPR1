#!/bin/bash
#SBATCH --job-name=try5-r1p5-blur-pt
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash pretrain_src/run_pt/run_mix_server.bash 2333 --use_prior_gt --checkpoint pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt --cognitive_map_namespace gt.legacy.r1p5.direction5.blurred.v1 --cognitive_map_metadata_schema direction5
