#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
TORCHRUN=/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun
CKPT=/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth
PRETRAIN=/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/model_step_440000.pt
NAMESPACE=/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.donor_cross_scene.v1
NAME=gt18600_legacy_val_unseen_cross_scene
OUT="${ROOT}/data/logs/checkpoints/${NAME}"

test -d "${NAMESPACE}/raster"
mkdir -p "${OUT}"
cd "${ROOT}"
env CUDA_VISIBLE_DEVICES=1,5,6,7 GLOG_minloglevel=2 MAGNUM_LOG=quiet \
  PYTHONPATH="${ROOT}" __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
  "${TORCHRUN}" --master_port 29602 --nproc_per_node=4 "${ROOT}/run.py" \
  --exp_name "${NAME}" --run-type eval --exp-config "${ROOT}/run_r2r/iter_train.yaml" \
  SIMULATOR_GPU_IDS '[0,1,2,3]' TORCH_GPU_IDS '[0,1,2,3]' GPU_NUMBERS 4 \
  NUM_ENVIRONMENTS 4 TASK_CONFIG.SEED 100 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  TRAINER_NAME SS-ETP-PriorGT MODEL.policy_name PriorGTTry5Policy \
  MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5 \
  MODEL.MAP_ENCODER.source prior_gt \
  MODEL.MAP_ENCODER.cache_namespace "${NAMESPACE}" \
  MODEL.MAP_ENCODER.refiner_ckpt '' MODEL.MAP_ENCODER.eval_map_source p0 \
  MODEL.MAP_ENCODER.coordinate_fusion False \
  MODEL.MAP_ENCODER.load_pretrained_map_modules False \
  MODEL.pretrained_path "${PRETRAIN}" MODEL.elevation_axis y \
  EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 EVAL.SAVE_RESULTS True \
  EVAL.CKPT_PATH_DIR "${CKPT}" IL.back_algo teleport IL.tryout True \
  2>&1 | tee "${OUT}/eval.log"
