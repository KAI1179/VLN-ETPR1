# Sourced (not executed) by prepare_rxr_kd.sh, check_rxr_kd.sh and run_rxr_kd.sh
# so the three always agree.  Needs SETTING=en|all and REPO_ROOT.
#
# Every value can be overridden from the environment; defaults are ln01's.
case "${SETTING:-}" in en|all) ;; *) echo "SETTING must be en or all" >&2; return 2 ;; esac
: "${REPO_ROOT:?REPO_ROOT not set}"

# Host defaults.  ln01: the RxR GT teacher lives in the partner repo's worktree
# (read-only use).  WZ: the same files, copied from ln01 into
# /data/xukai/checkpoint/From_ln01 (tar bundles) and unpacked next to the other
# GT namespaces / into the repo's data/llm_navigation.
if [[ -d /home/xukai/code/ETP-R1-snapshot ]]; then   # WZ
    export GT_TEACHER_CKPT="${GT_TEACHER_CKPT:-/data/xukai/checkpoint/From_ln01/ckpt.iter15000.pth}"
    export GT_TEACHER_NAMESPACE="${GT_TEACHER_NAMESPACE:-gt.legacy.r1p5.direction5.rxr-translated-mention-97525891.v1}"
    export TEACHER_NS_SRC="${TEACHER_NS_SRC:-/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/${GT_TEACHER_NAMESPACE}}"
    export PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt}"
    export LLM_CACHE_DIR="${LLM_CACHE_DIR:-${REPO_ROOT}/data/llm_navigation}"
    export TORCHRUN="${TORCHRUN:-/home/xukai/anaconda3/envs/etpr1-py38/bin/torchrun}"
else                                                   # ln01
    TEACHER_WT="${TEACHER_WT:-${HOME}/run/ETP-R1/.worktrees/priorgt-rxr-native-policy-translated-map-dagger}"
    export GT_TEACHER_CKPT="${GT_TEACHER_CKPT:-${TEACHER_WT}/data/logs/checkpoints/priorgt_rxr_native_policy_translated_map_dagger/ckpt.iter15000.pth}"
    export GT_TEACHER_NAMESPACE="${GT_TEACHER_NAMESPACE:-gt.legacy.r1p5.direction5.rxr-translated-mention-97525891.v1}"
    export TEACHER_NS_SRC="${TEACHER_NS_SRC:-${TEACHER_WT}/data/cognitive_maps/${GT_TEACHER_NAMESPACE}}"
    export PRETRAINED_CKPT="${PRETRAINED_CKPT:-${HOME}/run/ETP-R1/pretrained/r2r_rxr_ce/llm_grid_try5/store2/model_step_460000.pt}"
    export LLM_CACHE_DIR="${LLM_CACHE_DIR:-${HOME}/run/ETP-R1/data/llm_navigation}"
fi
export GT_TEACHER_POLICY_NAME="${GT_TEACHER_POLICY_NAME:-PriorGTTry5Policy}"

export LLM_EN_KEY="${LLM_EN_KEY:-llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree}"
export LLM_TRANSLATED_KEY="${LLM_TRANSLATED_KEY:-llm-grid-rxr-translated-hi-te-grid2-epoch10-v1}"
# Episodes rejected only for non-unit direction vectors, re-parsed with
# renormalisation (repair_rxr_llm_direction_norm.py); used as fallback.
export LLM_REPAIR_KEY="${LLM_REPAIR_KEY:-llm-grid-rxr-dirnorm-repair-v1}"
# Symlink keys the student reads (prepare_rxr_kd.sh builds both).
export LLM_EN_MERGED_KEY="${LLM_EN_MERGED_KEY:-llm-grid-rxr-en-tagfree-dirnorm-v1}"
export LLM_MERGED_KEY="${LLM_MERGED_KEY:-llm-grid-rxr-all-tagfree-hite-dirnorm-v1}"

if [[ "${SETTING}" == "en" ]]; then
    export LLM_CACHE_KEY="${LLM_EN_MERGED_KEY}"
    export MASTER_PORT="${MASTER_PORT:-29710}"
else
    export LLM_CACHE_KEY="${LLM_MERGED_KEY}"
    export MASTER_PORT="${MASTER_PORT:-29720}"
fi

export DATASET_SUFFIX="${DATASET_SUFFIX-_90}"
export EXP_CONFIG="${REPO_ROOT}/run_rxr/iter_train.yaml,${REPO_ROOT}/run_rxr/kd_${SETTING}.yaml"
export RUN_NAME="${RUN_NAME:-kd_rxr_${SETTING}_gt15000_kl}"
export ITERS="${ITERS:-30000}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-200}"
# older checkpoints keep weights only (~2 GiB instead of ~5 GiB with Adam state)
export CKPT_OPTIM_LATEST_ONLY="${CKPT_OPTIM_LATEST_ONLY:-True}"
export NUM_ENVS="${NUM_ENVS:-4}"
export LOAD_MAP=False
export DISTILL_WEIGHT="${DISTILL_WEIGHT:-1.0}"
export TEMP="${TEMP:-1.0}"
