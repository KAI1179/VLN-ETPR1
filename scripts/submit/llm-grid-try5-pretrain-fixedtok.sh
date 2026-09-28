#!/bin/bash
#SBATCH --job-name=llm-grid-try5-pt-fixedtok
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gpus=4
#SBATCH -p vip_gpu_scze096
# LLM-Grid Try5 pretraining, re-run from the ETP-R1 release weights with the
# spatial-tokenizer initialisation fix (patch 0017) and the start-up guard.
# Submit from the checkout that carries the fix (the exp/refiner worktree).
# Extra arguments are passed to train_r2r.py, e.g. --num_train_steps 200000.
set -euo pipefail
eval "$(conda shell.bash hook)" && conda activate etpr1-uv

grep -q "def reinit_spatial_tokenizer" vlnce_baselines/models/etp_prior_gt/map_encoder.py \
  || { echo "checkout lacks the tokenizer init fix (patch 0017)" >&2; exit 2; }
grep -q "assert_conv_modules_initialised" pretrain_src/pretrain_src/train_r2r.py \
  || { echo "checkout lacks the start-up guard (patch 0017)" >&2; exit 2; }
python -c "import transformers; print('transformers', transformers.__version__)"
START_CKPT="pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt"
[ -f "${START_CKPT}" ] || { echo "missing ETP-R1 release checkpoint ${START_CKPT}" >&2; exit 2; }
OUTPUT_DIR="${OUTPUT_DIR:-pretrained/r2r_rxr_ce/llm_grid_try5_fixedtok}"

bash pretrain_src/run_pt/run_mix_server.bash "${OUTPUT_DIR}" \
  --navigation-architecture try5 \
  --cognitive-map-source llm_grid \
  --llm-cache-model-key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
  --checkpoint "${START_CKPT}" \
  --log_steps 1000 \
  "$@"
