#!/usr/bin/env bash
# One physical inference process, retaining the original D9 logical-rank batch.
set -euo pipefail
ROOT="${D10_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${D10_PYTHON:-/home/xukai/anaconda3/envs/etpr1-py38/bin/python}"
MODE="${1:?Usage: dump_rank.sh S|O|smoke_on|smoke_off LOGICAL_RANK GPU}"
RANK="${2:?Set logical rank}"
GPU="${3:?Set GPU}"
CKPT="${D10_ZERO_CKPT:-/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-000-baseline-dagger-28k/ckpt.iter28000.pth}"
PRETRAIN="${D10_PRETRAINED:-$ROOT/pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt}"
DATA="${D10_DUMP_ROOT:-$ROOT/data/logs/d10_dump}"
REPLAY="$ROOT/data/logs/d8_dump/base/S"
SHARED=True
ATTENTION="$DATA/zero/S"
FEATURES=''
COUNT=-1
TOKEN_DIR="$DATA/token_checks"
case "$MODE" in
    S) ;;
    O) SHARED=False; ATTENTION='' ;;
    smoke_on) ATTENTION="$DATA/smoke/on/attention"; TOKEN_DIR="$DATA/smoke/on/token_checks"; COUNT=4 ;;
    smoke_off) ATTENTION=''; FEATURES="$DATA/smoke/off/features"; COUNT=4 ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
IDS='[]'
if [[ "$SHARED" == True ]]; then
    IDS="$("$PYTHON" -c 'import pathlib,sys; print(repr(sorted(int(p.stem) for p in pathlib.Path(sys.argv[1]).glob("*.npz"))))' "$REPLAY")"
else
    REPLAY=''
fi
NAME="d10_zero_${MODE}_rank${RANK}"
OUT="$ROOT/data/logs/checkpoints/$NAME"
[[ -f "$CKPT" && -f "$PRETRAIN" ]]
mkdir -p "$OUT"
cd "$ROOT"
env CUDA_VISIBLE_DEVICES="$GPU" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT" \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d \
    LOCAL_RANK=0 MASTER_PORT="${D10_PORT:-29870}" D9_REFERENCE_RANK="$RANK" \
    D10_TOKEN_DIR="$TOKEN_DIR" \
    "$PYTHON" "$ROOT/scripts/eval_d10/run_dump.py" \
    --exp_name "$NAME" --run-type eval --exp-config "$ROOT/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS '[0]' TORCH_GPU_IDS '[0]' GPU_NUMBERS 1 NUM_ENVIRONMENTS 4 \
    TASK_CONFIG.SEED 100 TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 5000 \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT "$COUNT" EVAL.USE_CKPT_CONFIG False \
    EVAL.CKPT_PATH_DIR "$CKPT" EVAL.d7d_logging False \
    TRAINER_NAME SS-ETP-LLM MODEL.policy_name LLMGridTry5Policy \
    MODEL.task_type r2r MODEL.elevation_axis y MODEL.pretrained_path "$PRETRAIN" \
    MODEL.MAP_ENCODER.enabled True MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source llm_grid \
    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
    MODEL.MAP_ENCODER.refiner_ckpt '' MODEL.MAP_ENCODER.coordinate_fusion False \
    MODEL.MAP_ENCODER.load_pretrained_map_modules False \
    MODEL.MAP_ENCODER.start_position_scale 1.0 MODEL.MAP_ENCODER.start_position_meters_per_unit 1.0 \
    MODEL.MAP_ENCODER.eval_map_source p0 MODEL.MAP_ENCODER.map_ablation none \
    IL.back_algo control IL.tryout True IL.max_text_len 150 IL.max_traj_len 15 \
    EVAL.dump_attention "$ATTENTION" EVAL.dump_decision_features "$FEATURES" \
    EVAL.d8_shared_states "$SHARED" EVAL.d8_replay_dir "$REPLAY" EVAL.EPISODE_ID "$IDS" \
    2>&1 | tee "$OUT/eval.log"
