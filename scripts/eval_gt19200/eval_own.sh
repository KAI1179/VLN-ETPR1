#!/usr/bin/env bash
# One HEAD-harness evaluation; match the transferred training map settings.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
TORCHRUN=/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun
CKPT=/data/xukai/checkpoint/etp-r1-snapshot-data-logs/logs/checkpoints/dagger_coordfusion_gt_legacy/ckpt.iter19200.pth
PRETRAIN=/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/model_step_440000.pt
NAME=gt_coordfusion_legacy_iter19200_val_unseen_teleport_w4
OUT="$ROOT/data/logs/checkpoints/$NAME"
[[ -f "$CKPT" && -f "$PRETRAIN" ]]
cd "$ROOT"
mkdir -p "$OUT"
env CUDA_VISIBLE_DEVICES=0,1,2,3 GLOG_minloglevel=2 MAGNUM_LOG=quiet \
  PYTHONPATH="$ROOT" __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
  "$TORCHRUN" --master_port 29820 --nproc_per_node=4 "$ROOT/run.py" \
  --exp_name "$NAME" --run-type eval --exp-config "$ROOT/run_r2r/iter_train.yaml" \
  SIMULATOR_GPU_IDS '[0,1,2,3]' TORCH_GPU_IDS '[0,1,2,3]' GPU_NUMBERS 4 \
  NUM_ENVIRONMENTS 4 TASK_CONFIG.SEED 100 \
  TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 5000 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  TRAINER_NAME SS-ETP-PriorGT MODEL.policy_name PriorGTTry5Policy \
  MODEL.task_type r2r MODEL.elevation_axis y MODEL.pretrained_path "$PRETRAIN" \
  MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5 \
  MODEL.MAP_ENCODER.source prior_gt \
  MODEL.MAP_ENCODER.cache_namespace gt.legacy.r1p5.direction5.v1 \
  MODEL.MAP_ENCODER.start_position_scale 1.0 \
  MODEL.MAP_ENCODER.start_position_meters_per_unit 1.0 \
  MODEL.MAP_ENCODER.coordinate_fusion True MODEL.MAP_ENCODER.refiner_ckpt '' \
  MODEL.MAP_ENCODER.eval_map_source p0 MODEL.MAP_ENCODER.map_ablation none \
  MODEL.MAP_ENCODER.load_pretrained_map_modules False \
  IL.gt_teacher_enabled False IL.back_algo teleport IL.tryout True \
  IL.max_text_len 150 IL.max_traj_len 15 \
  EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 EVAL.EPISODE_ID '[]' \
  EVAL.SAVE_RESULTS True EVAL.USE_CKPT_CONFIG False EVAL.CKPT_PATH_DIR "$CKPT" \
  2>&1 | tee "$OUT/eval.log"
