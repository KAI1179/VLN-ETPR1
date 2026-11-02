#!/usr/bin/env bash
# D3 val_unseen evaluation. Usage: eval_one.sh {region|full} ITER {own|cross_scene|zero_grid}
set -euo pipefail

ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
TORCHRUN=/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun
PRETRAIN=/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt
[[ $# -eq 3 ]] || { echo "Usage: $0 {region|full} ITER {own|cross_scene|zero_grid}" >&2; exit 2; }
ARM="$1"; ITER="$2"; VARIANT="$3"
[[ "${ITER}" =~ ^[0-9]+$ ]] || { echo 'ITER must be numeric' >&2; exit 2; }
case "${ARM}" in
  region) TRAIN=dagger_llm_region_only; SOURCE=llm.region_only.s2tagfree.v1; PORT_BASE=29662 ;;
  full) TRAIN=dagger_llm_full_control; SOURCE=llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree; PORT_BASE=29665 ;;
  *) echo 'ARM must be region or full' >&2; exit 2 ;;
esac
case "${VARIANT}" in
  own) KEY="${SOURCE}"; PORT="${PORT_BASE}" ;;
  cross_scene) KEY="${SOURCE}__raster-cross_scene-s0"; PORT="$((PORT_BASE + 1))" ;;
  zero_grid) KEY=probe.llm_zero_grid.v1; PORT="$((PORT_BASE + 2))" ;;
  *) echo 'VARIANT must be own, cross_scene, or zero_grid' >&2; exit 2 ;;
esac
PORT="${D3_PORT:-${PORT}}"
GPUS="${D3_EVAL_GPUS:-1,5,6,7}"
IFS=, read -r -a GPU_LIST <<< "${GPUS}"
WORLD="${#GPU_LIST[@]}"
[[ "${WORLD}" -eq 3 || "${WORLD}" -eq 4 ]] || { echo 'D3_EVAL_GPUS requires 3 or 4 cards' >&2; exit 2; }
LOGICAL="$(seq -s, 0 "$((WORLD - 1))")"
GPU_IDS="[${LOGICAL}]"
CKPT="${ROOT}/data/logs/checkpoints/${TRAIN}/ckpt.iter${ITER}.pth"
[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }
CACHE="${ROOT}/data/llm_navigation/${KEY}/r2r/val_unseen/manifest.json"
if [[ "${KEY}" == llm.region_only.s2tagfree.v1 ]]; then
  CACHE="${ROOT}/data/llm_navigation/${KEY}/manifest.json"
fi
[[ -f "${CACHE}" ]] || { echo "Missing cache: ${CACHE}" >&2; exit 1; }
BASE="${ROOT}/data/logs/checkpoints/d3_region_eval"
RUN="${TRAIN}_iter${ITER}_${VARIANT}"
OUT="${BASE}/${RUN}"
if [[ -f "${OUT}/done" ]]; then echo "Already complete: ${OUT}"; exit 0; fi
mkdir -p "${OUT}"
{
  echo "head_commit=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "checkpoint=${CKPT}"
  echo "cache_key=${KEY}"
  echo "cache_manifest_sha256=$(sha256sum "${CACHE}" | cut -d' ' -f1)"
  echo "gpus=${GPUS} world_size=${WORLD} port=${PORT}"
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
  MODEL.MAP_ENCODER.coordinate_fusion False \
  MODEL.MAP_ENCODER.refiner_ckpt '' MODEL.MAP_ENCODER.eval_map_source p0 \
  MODEL.MAP_ENCODER.load_pretrained_map_modules False MODEL.MAP_ENCODER.map_ablation none \
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
    episodes = json.loads(Path(path).read_text())
    if seen.intersection(episodes):
        raise RuntimeError(f"Duplicate episode in {path}")
    seen.update(episodes)
if len(seen) != 1839:
    raise RuntimeError(f"Expected 1839 val_unseen episodes; got {len(seen)}")
print(f"Validated {len(seen)} episodes across {world} ranks")
PY
touch "${OUT}/done"
echo "Complete: ${OUT}"
