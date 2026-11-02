#!/usr/bin/env bash
# Training-free ablation: retain the GT teacher map branch, zero all its inputs.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
TORCHRUN=/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun
CKPT=/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth
PRETRAIN=/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/model_step_440000.pt
NAME=gt18600_legacy_val_unseen_all_map_zero
OUT="$ROOT/data/logs/checkpoints/$NAME"
[[ -f "$CKPT" && -f "$PRETRAIN" ]]
[[ ! -e "$OUT/eval_results/stats_ckpt_18600_val_unseen.json" ]]
mkdir -p "$OUT"
cd "$ROOT"
env CUDA_VISIBLE_DEVICES="${GPUS:-7}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
  PYTHONPATH="$ROOT" __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
  "$TORCHRUN" --master_port "${MASTER_PORT:-29857}" --nproc_per_node=1 \
  "$ROOT/scripts/distill/run_all_map_zero_eval.py" \
  --exp_name "$NAME" --run-type eval --exp-config "$ROOT/run_r2r/iter_train.yaml" \
  SIMULATOR_GPU_IDS '[0]' TORCH_GPU_IDS '[0]' GPU_NUMBERS 1 \
  NUM_ENVIRONMENTS 4 TASK_CONFIG.SEED 100 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  TRAINER_NAME SS-ETP-PriorGT MODEL.policy_name PriorGTTry5Policy \
  MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5 \
  MODEL.MAP_ENCODER.source prior_gt \
  MODEL.MAP_ENCODER.cache_namespace gt.legacy.r1p5.direction5.v1 \
  MODEL.MAP_ENCODER.refiner_ckpt '' MODEL.MAP_ENCODER.eval_map_source p0 \
  MODEL.MAP_ENCODER.coordinate_fusion False \
  MODEL.MAP_ENCODER.load_pretrained_map_modules False \
  MODEL.MAP_ENCODER.map_ablation none \
  MODEL.pretrained_path "$PRETRAIN" MODEL.elevation_axis y \
  EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 EVAL.SAVE_RESULTS True \
  EVAL.USE_CKPT_CONFIG False EVAL.CKPT_PATH_DIR "$CKPT" \
  IL.back_algo teleport IL.tryout True IL.max_text_len 150 IL.max_traj_len 15 \
  2>&1 | tee "$OUT/eval.log"
