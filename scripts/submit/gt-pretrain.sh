#!/bin/bash
#SBATCH --job-name=gt-pretrain
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
eval "$(conda shell.bash hook)" && conda activate etpr1-uv
bash pretrain_src/run_pt/run_mix_server.bash 2333 --use_prior_gt --checkpoint pretrained/r2r_rxr_ce/baseline/ckpts/model_step_367500.pt # --n_workers 4
