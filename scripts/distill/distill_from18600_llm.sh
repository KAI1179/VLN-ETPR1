#!/usr/bin/env bash
# Initialise an LLM-Grid student from the complete 18600 DAgger checkpoint.
set -euo pipefail

ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
TORCHRUN=/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun
PRETRAIN=/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt
CKPT=/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth
GPUS=1,5,6,7
ARM="${D0_ARM:-distill}"
case "${ARM}" in
  distill) RUN=d0_distill_from18600_llm; PORT=29631; TEACHER=True ;;
  ce) RUN=d0_ce_from18600_llm; PORT=29632; TEACHER=False ;;
  *) echo "D0_ARM must be distill or ce" >&2; exit 2 ;;
esac
OUT="${ROOT}/data/logs/checkpoints/${RUN}"

[[ -f "${PRETRAIN}" && -f "${CKPT}" ]] || { echo "Missing checkpoint" >&2; exit 1; }
COMMAND=(
  "${TORCHRUN}" --master_port "${PORT}" --nproc_per_node=4 "${ROOT}/run.py"
  --exp_name "${RUN}" --run-type dagger --exp-config "${ROOT}/run_r2r/iter_train.yaml"
  SIMULATOR_GPU_IDS '[0,1,2,3]' TORCH_GPU_IDS '[0,1,2,3]' GPU_NUMBERS 4
  NUM_ENVIRONMENTS 4 TASK_CONFIG.SEED 100
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
  CHECKPOINT_FOLDER "${ROOT}/data/logs/checkpoints/"
  RESULTS_DIR "${ROOT}/data/logs/checkpoints/"
  TENSORBOARD_DIR "${ROOT}/data/logs/checkpoints/"
  CHECKPOINT_INTERVAL 1000 ONLY_LAST_SAVEALL False
  IL.iters 20000 IL.lr 1e-5 IL.log_every 200 IL.ml_weight 1.0
  IL.sample_ratio 0.75 IL.decay_interval 1400 IL.warmup_iters 300
  IL.min_lr_ratio 1.0 IL.waypoint_aug True TASK_CONFIG.DATASET.SUFFIX _90
  IL.load_from_ckpt True IL.is_requeue False IL.ckpt_to_load "${CKPT}"
  TRAINER_NAME SS-ETP-LLM MODEL.policy_name LLMGridTry5Policy
  MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5
  MODEL.MAP_ENCODER.source llm_grid
  MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
  MODEL.MAP_ENCODER.refiner_ckpt '' MODEL.MAP_ENCODER.eval_map_source p0
  MODEL.MAP_ENCODER.load_pretrained_map_modules False
  MODEL.MAP_ENCODER.coordinate_fusion False
  MODEL.MAP_ENCODER.start_position_meters_per_unit 1.0
  MODEL.elevation_axis y MODEL.pretrained_path "${PRETRAIN}"
  IL.gt_teacher_enabled "${TEACHER}"
  IL.distill_weight 1.0 IL.distill_temperature 1.0
  IL.gt_teacher_ckpt "${CKPT}"
  IL.gt_teacher_policy_name PriorGTTry5Policy
  IL.gt_teacher_map_namespace gt.legacy.r1p5.direction5.v1
  IL.gt_teacher_elevation_axis y
)

if [[ "${1:-}" == --print-command ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPUS}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
[[ $# -eq 0 ]] || { echo "Usage: bash $0 [--print-command]" >&2; exit 2; }
[[ ! -e "${OUT}" ]] || { echo "Output directory already exists: ${OUT}" >&2; exit 1; }
mkdir -p "${OUT}"
{
  echo "head_commit=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "ckpt_18600_sha256=$(sha256sum "${CKPT}" | cut -d' ' -f1)"
  echo "student_init_ckpt=${CKPT}"
  echo "constructor_pretrain=${PRETRAIN}"
  echo "arm=${ARM} teacher_enabled=${TEACHER} teacher_elevation_axis=y"
  echo "gpus=${GPUS} world_size=4 port=${PORT}"
} > "${OUT}/provenance.txt"

cd "${ROOT}"
env CUDA_VISIBLE_DEVICES="${GPUS}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
  PYTHONPATH="${ROOT}" __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
  "${COMMAND[@]}" 2>&1 | tee "${OUT}/train.log"
