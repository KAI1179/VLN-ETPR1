#!/usr/bin/env bash
# One original four-environment D8 rank on one physical GPU.
set -euo pipefail
ROOT=/home/xukai/code/ETP-R1-snapshot/ETP-R1
PYTHON=/home/xukai/anaconda3/envs/etpr1-py38/bin/python
[[ $# -ge 4 ]] || { echo 'Usage: dump_shard.sh base|try5|kd S|O logical_rank gpu [overrides...]' >&2; exit 2; }
MODEL="$1"; SOURCE="$2"; export D9_REFERENCE_RANK="$3"; GPU="$4"; shift 4
PRETRAINED=/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt
TRAINER=SS-ETP-LLM; POLICY=LLMGridTry5Policy; MAP=True
case "$MODEL" in
    base)
        CKPT=/data/xukai/etp-r1-prior/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth
        TRAINER=SS-ETP-R1; POLICY=R1Policy; MAP=False
        PRETRAINED="$ROOT/pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt" ;;
    try5) CKPT=/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth ;;
    kd) CKPT="$ROOT/data/logs/checkpoints/kd_dual/ckpt.iter27000.pth" ;;
    *) exit 2 ;;
esac
case "$SOURCE" in
    S)
        SHARED=True; REPLAY="$ROOT/data/logs/d8_dump/base/S"
        IDS="$("$PYTHON" -c 'import pathlib,sys; print(repr(sorted(int(p.stem) for p in pathlib.Path(sys.argv[1]).glob("*.npz"))))' "$REPLAY")" ;;
    O) SHARED=False; REPLAY=''; IDS='[]' ;;
    *) exit 2 ;;
esac
D9_DUMP_ROOT="${D9_DUMP_ROOT:-$ROOT/data/logs/d9_reference_layout}"
EXP_NAME="${EXP_NAME:-d9_reference_${MODEL}_${SOURCE}_rank${D9_REFERENCE_RANK}}"
PORT="${MASTER_PORT:-$((29840 + D9_REFERENCE_RANK))}"
EVAL_DIR="$ROOT/data/logs/checkpoints/$EXP_NAME"
[[ -f "$CKPT" && -f "$PRETRAINED" ]]
cd "$ROOT"
mkdir -p "$EVAL_DIR"
env CUDA_VISIBLE_DEVICES="$GPU" GLOG_minloglevel=2 MAGNUM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d PYTHONPATH="$ROOT" \
    MASTER_PORT="$PORT" LOCAL_RANK=0 \
    "$PYTHON" "$ROOT/scripts/eval_d9/run_shard.py" --exp_name "$EXP_NAME" --run-type eval \
    --exp-config "$ROOT/run_r2r/iter_train.yaml" \
    SIMULATOR_GPU_IDS '[0]' TORCH_GPU_IDS '[0]' GPU_NUMBERS 1 \
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
    EVAL.dump_attention "$D9_DUMP_ROOT/$MODEL/$SOURCE" \
    EVAL.d8_shared_states "$SHARED" EVAL.d8_replay_dir "$REPLAY" \
    EVAL.EPISODE_ID "$IDS" "$@" 2>&1 | tee "$EVAL_DIR/eval.log"
"$PYTHON" "$ROOT/scripts/eval_d9/run_shard.py" --verify-rank "$MODEL" "$SOURCE" "$D9_DUMP_ROOT/$MODEL/$SOURCE" "$EVAL_DIR/eval_results"
