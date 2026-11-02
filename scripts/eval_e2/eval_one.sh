#!/usr/bin/env bash
# Evaluate one E2 checkpoint on own or the existing seed-0 cross-scene raster.
set -euo pipefail

ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
TORCHRUN=/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun
TRAIN_DIR="${ROOT}/data/logs/checkpoints/dagger_coordfusion_llm"
BASE="${ROOT}/data/logs/checkpoints/e2_coordfusion_val_unseen"
PRETRAIN=/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt
SOURCE_KEY=llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree
DONOR_KEY=llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__raster-cross_scene-s0

if [[ $# -ne 2 ]]; then
  echo "Usage: bash ${ROOT}/scripts/eval_e2/eval_one.sh <iter> {own|cross_scene}" >&2
  exit 2
fi
ITER="$1"
VARIANT="$2"
[[ "${ITER}" =~ ^[0-9]+$ ]] || { echo "iter must be numeric" >&2; exit 2; }
case "${VARIANT}" in
  own) KEY="${SOURCE_KEY}"; PORT=29610 ;;
  cross_scene) KEY="${DONOR_KEY}"; PORT=29611 ;;
  *) echo "variant must be own or cross_scene" >&2; exit 2 ;;
esac
PORT="${E2_PORT:-${PORT}}"
CKPT="${TRAIN_DIR}/ckpt.iter${ITER}.pth"
[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }
if [[ "${VARIANT}" == cross_scene ]]; then
  [[ -f "${ROOT}/data/llm_navigation/${DONOR_KEY}/r2r/val_unseen/manifest.json" ]] || {
    echo "Missing existing seed-0 donor cache: ${DONOR_KEY}" >&2; exit 1;
  }
fi

GPUS="${E2_GPUS:-5,6,7}"
IFS=, read -r -a GPU_LIST <<< "${GPUS}"
WORLD="${#GPU_LIST[@]}"
[[ "${WORLD}" -eq 3 || "${WORLD}" -eq 4 ]] || { echo "E2_GPUS must list 3 or 4 cards" >&2; exit 2; }
for gpu in "${GPU_LIST[@]}"; do
  case "${gpu}" in
    1)
      FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')"
      [[ "${FREE_MIB}" =~ ^[0-9]+$ && "${FREE_MIB}" -ge 12288 ]] || {
        echo "GPU 1 has less than 12 GiB free or nvidia-smi is unavailable" >&2; exit 1;
      } ;;
    4|5|6|7) ;;
    *) echo "GPU ${gpu} is outside the allowed set 1,4,5,6,7" >&2; exit 2 ;;
  esac
done
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" -eq "${WORLD}" ]] || {
  echo "E2_GPUS contains duplicates" >&2; exit 2;
}
LOGICAL="$(seq -s, 0 "$((WORLD - 1))")"
GPU_IDS="[${LOGICAL}]"
RUN="iter${ITER}_${VARIANT}${E2_RUN_SUFFIX:-}"
OUT="${BASE}/${RUN}"
if [[ -f "${OUT}/done" ]]; then
  echo "Already complete: ${OUT}"
  exit 0
fi
mkdir -p "${OUT}"

{
  echo "head_commit=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "training_commit=4fd623142f4c727a33e8c3a42e2352281b2572db"
  echo "training_config_sha256=$(sha256sum "${TRAIN_DIR}/config.yaml" | cut -d' ' -f1)"
  echo "checkpoint=${CKPT}"
  echo "variant=${VARIANT} cache_key=${KEY}"
  echo "gpus=${GPUS} world_size=${WORLD} port=${PORT}"
  echo "tracked_diff_stat:"
  git -C "${ROOT}" diff --stat
  echo "worktree_status:"
  git -C "${ROOT}" status --short
} > "${OUT}/provenance.txt"

cd "${ROOT}"
env CUDA_VISIBLE_DEVICES="${GPUS}" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
  PYTHONPATH="${ROOT}" __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
  "${TORCHRUN}" --master_port "${PORT}" --nproc_per_node="${WORLD}" \
  "${ROOT}/run.py" --exp_name "${RUN}" --run-type eval \
  --exp-config "${ROOT}/run_r2r/iter_train.yaml" \
  CHECKPOINT_FOLDER "${BASE}/" RESULTS_DIR "${BASE}/" TENSORBOARD_DIR "${BASE}/" \
  SIMULATOR_GPU_IDS "${GPU_IDS}" TORCH_GPU_IDS "${GPU_IDS}" GPU_NUMBERS "${WORLD}" \
  NUM_ENVIRONMENTS 4 TASK_CONFIG.SEED 100 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  TRAINER_NAME SS-ETP-LLM MODEL.policy_name LLMGridTry5Policy \
  MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5 \
  MODEL.MAP_ENCODER.source llm_grid MODEL.MAP_ENCODER.llm_cache_model_key "${KEY}" \
  MODEL.MAP_ENCODER.coordinate_fusion True \
  MODEL.MAP_ENCODER.start_position_scale 1.0 \
  MODEL.MAP_ENCODER.start_position_meters_per_unit 1.0 \
  MODEL.MAP_ENCODER.refiner_ckpt '' MODEL.MAP_ENCODER.eval_map_source p0 \
  MODEL.MAP_ENCODER.load_pretrained_map_modules False \
  MODEL.MAP_ENCODER.map_ablation none \
  MODEL.pretrained_path "${PRETRAIN}" MODEL.elevation_axis y \
  EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 EVAL.SAVE_RESULTS True \
  EVAL.CKPT_PATH_DIR "${CKPT}" IL.back_algo teleport IL.tryout True \
  2>&1 | tee "${OUT}/eval.log"

"${PYTHON}" - "${OUT}/eval_results" "${ITER}" "${WORLD}" <<'PY'
import glob
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
iteration, world = sys.argv[2], int(sys.argv[3])
files = sorted(glob.glob(str(result_dir / f"stats_ep_ckpt_{iteration}_val_unseen_r*_w{world}.json")))
if len(files) != world:
    raise RuntimeError(f"Expected {world} rank files; got {len(files)}: {files}")
seen = set()
for path in files:
    episodes = json.load(open(path))
    if seen.intersection(episodes):
        raise RuntimeError(f"Duplicate episode in {path}")
    seen.update(episodes)
if len(seen) != 1839:
    raise RuntimeError(f"Expected 1839 val_unseen episodes; got {len(seen)}")
print(f"Validated {len(seen)} episodes across {world} ranks")
PY
touch "${OUT}/done"
echo "Complete: ${OUT}"
