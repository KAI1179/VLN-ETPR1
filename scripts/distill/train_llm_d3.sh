#!/usr/bin/env bash
# Shared D3 training command; the two entry points differ only in key/run/port.
set -euo pipefail

ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
TORCHRUN=/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun
PRETRAIN=/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt
RUN_NAME="$1"
MODEL_KEY="$2"
PORT="$3"
: "${D3_GPUS:?Set D3_GPUS to four GPU IDs, e.g. 1,5,6,7}"
IFS=, read -r -a GPU_LIST <<< "${D3_GPUS}"
[[ "${#GPU_LIST[@]}" -eq 4 ]] || { echo "D3_GPUS must list four GPUs" >&2; exit 2; }
RUN_DIR="${ROOT}/data/logs/checkpoints/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
if [[ "${MODEL_KEY}" == llm.region_only.s2tagfree.v1 ]]; then
  CACHE_MANIFEST="${ROOT}/data/llm_navigation/${MODEL_KEY}/manifest.json"
else
  CACHE_MANIFEST="${ROOT}/data/llm_navigation/${MODEL_KEY}/r2r/train/manifest.json"
fi
{
  echo "head_commit=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "pretrained=${PRETRAIN}"
  echo "model_key=${MODEL_KEY}"
  echo "gpus=${D3_GPUS}"
  sha256sum "${CACHE_MANIFEST}"
} > "${RUN_DIR}/provenance.txt"

cd "${ROOT}"
env CUDA_VISIBLE_DEVICES="${D3_GPUS}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
  PYTHONPATH="${ROOT}" __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
  "${TORCHRUN}" --master_port "${PORT}" --nproc_per_node=4 \
  "${ROOT}/run.py" --exp_name "${RUN_NAME}" --run-type dagger \
  --exp-config "${ROOT}/run_r2r/iter_train.yaml" \
  SIMULATOR_GPU_IDS '[0,1,2,3]' TORCH_GPU_IDS '[0,1,2,3]' GPU_NUMBERS 4 \
  TASK_CONFIG.SEED 100 TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  NUM_ENVIRONMENTS 4 CHECKPOINT_INTERVAL 400 ONLY_LAST_SAVEALL False \
  IL.iters 30000 IL.lr 1e-5 IL.batch_size 1 IL.log_every 200 \
  IL.ml_weight 1.0 IL.sample_ratio 0.75 IL.decay_interval 2000 \
  IL.warmup_iters 500 IL.min_lr_ratio 1.0 IL.load_from_ckpt False \
  IL.is_requeue False IL.waypoint_aug True IL.back_algo teleport IL.tryout True \
  TASK_CONFIG.DATASET.SUFFIX _90 TRAINER_NAME SS-ETP-LLM \
  MODEL.policy_name LLMGridTry5Policy MODEL.MAP_ENCODER.enabled True \
  MODEL.MAP_ENCODER.architecture try5 MODEL.MAP_ENCODER.source llm_grid \
  MODEL.MAP_ENCODER.llm_cache_model_key "${MODEL_KEY}" \
  MODEL.MAP_ENCODER.load_pretrained_map_modules False \
  MODEL.MAP_ENCODER.coordinate_fusion False MODEL.MAP_ENCODER.refiner_ckpt '' \
  MODEL.elevation_axis y MODEL.pretrained_path "${PRETRAIN}" \
  IL.gt_teacher_enabled False \
  2>&1 | tee "${RUN_DIR}/train.log"
