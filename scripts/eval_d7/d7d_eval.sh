#!/usr/bin/env bash
# D7-d eval-only logging; foreground execution for tmux.
set -euo pipefail
REPO_ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
TORCHRUN=/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun
[[ $# -ge 1 ]] || { echo 'Usage: bash d7d_eval.sh base|try5|kd [config overrides...]' >&2; exit 2; }
NAME="$1"; shift
PRETRAINED=/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt
TRAINER=SS-ETP-LLM; POLICY=LLMGridTry5Policy; MAP=True
case "$NAME" in
    base)
        CKPT=/data/xukai/etp-r1-prior/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth
        PORT=29610; TRAINER=SS-ETP-R1; POLICY=R1Policy; MAP=False
        PRETRAINED="$REPO_ROOT/pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt" ;;
    try5)
        CKPT=/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth
        PORT=29611 ;;
    kd)
        CKPT="$REPO_ROOT/data/logs/checkpoints/kd_dual/ckpt.iter27000.pth"
        PORT=29612 ;;
    *) echo "Unknown model: $NAME" >&2; exit 2 ;;
esac
GPUS="${GPUS:-4,5,6,7}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
WORLD=${#GPU_LIST[@]}
GPU_IDS="$(/home/xukai/anaconda3/envs/etpr1-py38/bin/python -c "print(list(range($WORLD)))")"
EXP_NAME="${EXP_NAME:-d7d_${NAME}_control_val_unseen}"
EVAL_DIR="$REPO_ROOT/data/logs/checkpoints/$EXP_NAME"
[[ -f "$CKPT" && -f "$PRETRAINED" ]]
cd "$REPO_ROOT"
mkdir -p "$EVAL_DIR"
env CUDA_VISIBLE_DEVICES="$GPUS" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d PYTHONPATH="$REPO_ROOT" \
    "$TORCHRUN" --master_port "${MASTER_PORT:-$PORT}" --nproc_per_node="$WORLD" \
    "$REPO_ROOT/run.py" --exp_name "$EXP_NAME" --run-type eval \
    --exp-config "$REPO_ROOT/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS "$GPU_IDS" TORCH_GPU_IDS "$GPU_IDS" GPU_NUMBERS "$WORLD" \
    NUM_ENVIRONMENTS 4 TASK_CONFIG.SEED 100 \
    TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS 5000 \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 EVAL.USE_CKPT_CONFIG False \
    EVAL.CKPT_PATH_DIR "$CKPT" EVAL.d7d_logging True \
    TRAINER_NAME "$TRAINER" MODEL.policy_name "$POLICY" \
    MODEL.task_type r2r MODEL.elevation_axis y MODEL.pretrained_path "$PRETRAINED" \
    MODEL.MAP_ENCODER.enabled "$MAP" MODEL.MAP_ENCODER.architecture try5 \
    MODEL.MAP_ENCODER.source llm_grid \
    MODEL.MAP_ENCODER.llm_cache_model_key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
    MODEL.MAP_ENCODER.refiner_ckpt '' MODEL.MAP_ENCODER.coordinate_fusion False \
    MODEL.MAP_ENCODER.load_pretrained_map_modules False \
    MODEL.MAP_ENCODER.eval_map_source p0 MODEL.MAP_ENCODER.map_ablation none \
    IL.back_algo control IL.tryout True IL.max_text_len 150 IL.max_traj_len 15 \
    "$@" 2>&1 | tee "$EVAL_DIR/eval.log"
