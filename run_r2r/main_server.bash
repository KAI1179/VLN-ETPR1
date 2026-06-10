export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/gpu-detection.bash"

# 代码运行架构：
# 节点：指的是一台独立的物理计算机或服务器，内部可能包含多个gpu。若节点为1则是单机多卡训练，若为1以上，则是多机多卡训练
# 进程(即nproc_per_node的值)：一个“进程”是一个正在运行的程序实例。在 PyTorch 分布式训练（尤其是使用 DistributedDataParallel）中，通常每个 GPU 会由一个独立的 Python 进程来控制和运行训练代码。
# local_rank：当前节点（电脑）下的所有进程（gpu）编号
# world_size/rank：整个分布式训练任务中，参与训练的全局总进程（gpu）数量/编号。对于单机多卡而言，rank就等于local_rank
# 全部代码都运行在一个master节点上；在节点中含有多个进程，每个进程使用一个单独的gpu；
# 每个进程中创建了多个habitat的envs，每个habitat的env包含多个scenes，负责运行该gpu需要运行的episodes中对应scenes的子集
# （torch.distributed.launch默认只能处理一个进程一张gpu的问题,因为训练需要保证每个gpu都至少能单独运行全部可训练权重的网络）


# NUM_ENVIRONMENTS：num_envs_per_gpu
# 只有当IL.load_from_ckpt为True时,IL.is_requeue为True才会起作用;而后者的作用是,改写IL.ckpt_to_load,将其设置为权重文件夹下iteration次数最多的那个权重

# Continue from previous checkpoints: Set IL.load_from_ckpt and IL.is_requeue to True

configure_distributed_gpu_vars
MASTER_PORT=${2:-2333}

EXP_CONFIG="run_r2r/iter_train.yaml"
BASE_NUM_ENVS=8
MAP_NUM_ENVS=4

BASE_PRETRAINED_CKPT="pretrained/r2r_rxr_ce/baseline/ckpts/model_step_367500.pt"
BASE_DAGGER_CKPT="data/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth"
BASE_GRPO_CKPT="data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth"

GT_PRETRAINED_CKPT="pretrained/r2r_rxr_ce/prior_gt/store2/try6_step_415000.pt"
GT_DAGGER_CKPT="data/logs/checkpoints/release_r2r_priorgt_dagger/store/try-5-vlnce.iter27800.pth"
GT_GRPO_CKPT="data/logs/checkpoints/release_r2r_priorgt_grpo/store/try-5-vlnce.iter350.pth"
GT_PROBE_CKPT="data/logs/checkpoints/release_r2r_priorgt_probe/store/ckpt.iter3000.pth"

IMAGINED_PREDICTOR_CKPT="${IMAGINED_PREDICTOR_CKPT:-}"
IMAGINED_PRETRAINED_CKPT="pretrained/r2r_rxr_ce/imagined/ckpts/model_step_100000.pt"
IMAGINED_DAGGER_CKPT="data/logs/checkpoints/release_r2r_imagined_dagger/store/ckpt.iter30000.pth"
IMAGINED_GRPO_CKPT="data/logs/checkpoints/release_r2r_imagined_grpo/store/ckpt.iter270.pth"

LLM_PRETRAINED_CKPT="pretrained/r2r_rxr_ce/llm/ckpts/model_step_100000.pt"
LLM_DAGGER_CKPT="data/logs/checkpoints/release_r2r_llm_dagger/store/ckpt.iter30000.pth"
LLM_GRPO_CKPT="data/logs/checkpoints/release_r2r_llm_grpo/store/ckpt.iter270.pth"

COMMON_ARGS="--exp-config ${EXP_CONFIG}
      SIMULATOR_GPU_IDS ${GPU_IDS}
      TORCH_GPU_IDS ${GPU_IDS}
      GPU_NUMBERS ${GPU_NUMBERS}
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True"

DAGGER_ARGS="IL.iters 30000
      IL.lr 1e-5
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 2000
      IL.warmup_iters 500
      IL.min_lr_ratio 1.0
      IL.load_from_ckpt False
      IL.is_requeue False
      IL.waypoint_aug True
      TASK_CONFIG.DATASET.SUFFIX _90"

GRPO_ARGS="ONLY_LAST_SAVEALL True
      GRPO.iters 500
      GRPO.lr 2e-5
      GRPO.warmup_iters 0
      GRPO.min_lr_ratio 0.25
      GRPO.log_every 100
      GRPO.load_from_ckpt True
      GRPO.is_requeue False
      GRPO.waypoint_aug True
      GRPO.sample_num 8
      GRPO.update_epochs 1
      GRPO.grpo_beta 0.04
      GRPO.grpo_epsilon 0.2
      GRPO.enable_amp False
      GRPO.enable_all_dropouts True
      GRPO.dropout_in_sampling True
      GRPO.dropout_rate 0.10
      GRPO.max_grad_norm 2.0
      TASK_CONFIG.DATASET.SUFFIX _10"

BASE_MODEL_ARGS="MODEL.pretrained_path ${BASE_PRETRAINED_CKPT}"

GT_MODEL_ARGS="TRAINER_NAME SS-ETP-PriorGT
      MODEL.policy_name PriorGTPolicy
      MODEL.MAP_ENCODER.enabled True
      MODEL.pretrained_path ${GT_PRETRAINED_CKPT}"

GT_GRPO_MODEL_ARGS="TRAINER_NAME GRPO-ETP-PriorGT
      MODEL.policy_name PriorGTPolicy
      MODEL.MAP_ENCODER.enabled True
      MODEL.pretrained_path ${GT_PRETRAINED_CKPT}"

IMAGINED_PREDICTOR_ARG=""
if [ -n "${IMAGINED_PREDICTOR_CKPT}" ]; then
      IMAGINED_PREDICTOR_ARG="MODEL.MAP_ENCODER.predictor_checkpoint ${IMAGINED_PREDICTOR_CKPT}"
fi

IMAGINED_MODEL_ARGS="TRAINER_NAME SS-ETP-Imagined
      MODEL.policy_name ImaginedPolicy
      MODEL.MAP_ENCODER.enabled True
      ${IMAGINED_PREDICTOR_ARG}
      MODEL.pretrained_path ${IMAGINED_PRETRAINED_CKPT}"

IMAGINED_GRPO_MODEL_ARGS="TRAINER_NAME GRPO-ETP-Imagined
      MODEL.policy_name ImaginedPolicy
      MODEL.MAP_ENCODER.enabled True
      ${IMAGINED_PREDICTOR_ARG}
      MODEL.pretrained_path ${IMAGINED_PRETRAINED_CKPT}"

LLM_MODEL_ARGS="TRAINER_NAME SS-ETP-LLM
      MODEL.policy_name LLMPolicy
      MODEL.MAP_ENCODER.enabled True
      MODEL.pretrained_path ${LLM_PRETRAINED_CKPT}"

LLM_GRPO_MODEL_ARGS="TRAINER_NAME GRPO-ETP-LLM
      MODEL.policy_name LLMPolicy
      MODEL.MAP_ENCODER.enabled True
      MODEL.pretrained_path ${LLM_PRETRAINED_CKPT}"

launch() {
      python -m torch.distributed.launch --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" run.py $1
}

warn_unimplemented() {
      echo "WARNING: $1 is not implemented yet." >&2
      exit 2
}

mode=$1
case $mode in
      dagger)
      echo "###### dagger train mode ######"
      launch "--exp_name release_r2r_dagger --run-type dagger ${COMMON_ARGS} NUM_ENVIRONMENTS ${BASE_NUM_ENVS} ${DAGGER_ARGS} ${BASE_MODEL_ARGS}"
      ;;
      grpo)
      echo "###### grpo train mode ######"
      launch "--exp_name release_r2r_grpo --run-type grpo ${COMMON_ARGS} NUM_ENVIRONMENTS ${BASE_NUM_ENVS} TRAINER_NAME GRPO-R1 ${GRPO_ARGS} GRPO.ckpt_to_load ${BASE_DAGGER_CKPT} ${BASE_MODEL_ARGS}"
      ;;
      eval)
      echo "###### eval mode ######"
      launch "--exp_name release_r2r_grpo --run-type eval ${COMMON_ARGS} NUM_ENVIRONMENTS ${BASE_NUM_ENVS} EVAL.CKPT_PATH_DIR ${BASE_GRPO_CKPT} IL.back_algo control ${BASE_MODEL_ARGS}"
      ;;
      infer)
      echo "###### infer mode ######"
      launch "--exp_name release_r2r_grpo --run-type inference ${COMMON_ARGS} NUM_ENVIRONMENTS ${BASE_NUM_ENVS} INFERENCE.CKPT_PATH ${BASE_GRPO_CKPT} INFERENCE.PREDICTIONS_FILE preds.json IL.back_algo control ${BASE_MODEL_ARGS}"
      ;;
      priorgt_dagger)
      echo "###### priorgt dagger train mode ######"
      launch "--exp_name release_r2r_priorgt_dagger --run-type dagger ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${GT_MODEL_ARGS} ${DAGGER_ARGS}"
      ;;
      priorgt_grpo)
      echo "###### priorgt grpo train mode ######"
      launch "--exp_name release_r2r_priorgt_grpo --run-type grpo ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${GT_GRPO_MODEL_ARGS} ${GRPO_ARGS} GRPO.log_every 10 GRPO.ckpt_to_load ${GT_DAGGER_CKPT}"
      ;;
      priorgt_eval_ss)
      echo "###### priorgt eval mode (SS ckpt) ######"
      launch "--exp_name release_r2r_priorgt_dagger --run-type eval ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${GT_MODEL_ARGS} EVAL.CKPT_PATH_DIR ${GT_DAGGER_CKPT} IL.back_algo control"
      ;;
      priorgt_eval_grpo)
      echo "###### priorgt eval mode (GRPO ckpt) ######"
      launch "--exp_name release_r2r_priorgt_grpo --run-type eval ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${GT_MODEL_ARGS} EVAL.CKPT_PATH_DIR ${GT_GRPO_CKPT} IL.back_algo control"
      ;;
      priorgt_probe)
      echo "###### priorgt probe: freeze base, train map encoder only ######"
      launch "--exp_name release_r2r_priorgt_probe --run-type dagger ${COMMON_ARGS} NUM_ENVIRONMENTS ${BASE_NUM_ENVS} ${GT_MODEL_ARGS} MODEL.MAP_ENCODER.freeze_base True IL.iters 3000 IL.lr 1e-4 IL.log_every 100 IL.ml_weight 1.0 IL.sample_ratio 0.75 IL.decay_interval 1000 IL.warmup_iters 200 IL.min_lr_ratio 0.1 IL.load_from_ckpt True IL.is_requeue False IL.waypoint_aug True IL.ckpt_to_load ${BASE_GRPO_CKPT} TASK_CONFIG.DATASET.SUFFIX _90"
      ;;
      priorgt_probe_eval)
      echo "###### priorgt probe eval ######"
      launch "--exp_name release_r2r_priorgt_probe --run-type eval ${COMMON_ARGS} NUM_ENVIRONMENTS ${BASE_NUM_ENVS} ${GT_MODEL_ARGS} EVAL.CKPT_PATH_DIR ${GT_PROBE_CKPT} IL.back_algo control"
      ;;
      imagined_dagger)
      echo "###### imagined dagger train mode ######"
      launch "--exp_name release_r2r_imagined_dagger --run-type dagger ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${IMAGINED_MODEL_ARGS} ${DAGGER_ARGS}"
      ;;
      imagined_eval_ss)
      echo "###### imagined eval mode (SS ckpt) ######"
      launch "--exp_name release_r2r_imagined_dagger --run-type eval ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${IMAGINED_MODEL_ARGS} EVAL.CKPT_PATH_DIR ${IMAGINED_DAGGER_CKPT} IL.back_algo control"
      ;;
      imagined_grpo)
      echo "###### imagined grpo train mode ######"
      launch "--exp_name release_r2r_imagined_grpo --run-type grpo ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${IMAGINED_GRPO_MODEL_ARGS} ${GRPO_ARGS} GRPO.log_every 10 GRPO.ckpt_to_load ${IMAGINED_DAGGER_CKPT}"
      ;;
      imagined_eval_grpo)
      echo "###### imagined eval mode (GRPO ckpt) ######"
      launch "--exp_name release_r2r_imagined_grpo --run-type eval ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${IMAGINED_MODEL_ARGS} EVAL.CKPT_PATH_DIR ${IMAGINED_GRPO_CKPT} IL.back_algo control"
      ;;
      llm_dagger)
      echo "###### llm dagger train mode ######"
      launch "--exp_name release_r2r_llm_dagger --run-type dagger ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${LLM_MODEL_ARGS} ${DAGGER_ARGS}"
      ;;
      llm_eval_ss)
      echo "###### llm eval mode (SS ckpt) ######"
      launch "--exp_name release_r2r_llm_dagger --run-type eval ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${LLM_MODEL_ARGS} EVAL.CKPT_PATH_DIR ${LLM_DAGGER_CKPT} IL.back_algo control"
      ;;
      llm_grpo)
      echo "###### llm grpo train mode ######"
      launch "--exp_name release_r2r_llm_grpo --run-type grpo ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${LLM_GRPO_MODEL_ARGS} ${GRPO_ARGS} GRPO.log_every 10 GRPO.ckpt_to_load ${LLM_DAGGER_CKPT}"
      ;;
      llm_eval_grpo)
      echo "###### llm eval mode (GRPO ckpt) ######"
      launch "--exp_name release_r2r_llm_grpo --run-type eval ${COMMON_ARGS} NUM_ENVIRONMENTS ${MAP_NUM_ENVS} ${LLM_MODEL_ARGS} EVAL.CKPT_PATH_DIR ${LLM_GRPO_CKPT} IL.back_algo control"
      ;;
      imagined_infer)
      warn_unimplemented "inference path for imagined cognitive maps"
      ;;
      *)
      echo "Usage: $0 {dagger|grpo|eval|infer|priorgt_dagger|priorgt_grpo|priorgt_eval_ss|priorgt_eval_grpo|priorgt_probe|priorgt_probe_eval|imagined_dagger|imagined_eval_ss|imagined_grpo|imagined_eval_grpo|imagined_infer|llm_dagger|llm_eval_ss|llm_grpo|llm_eval_grpo} [master_port]" >&2
      exit 1
      ;;
esac

# 命令行运行：
# Uses all visible GPUs by default. Set CUDA_VISIBLE_DEVICES first to restrict cards.
# bash run_r2r/main_server.bash dagger 2333
# bash run_r2r/main_server.bash grpo 2333
# bash run_r2r/main_server.bash eval 2333
# bash run_r2r/main_server.bash infer 2333
# bash run_r2r/main_server.bash priorgt_dagger 2333
# bash run_r2r/main_server.bash priorgt_grpo 2333
# bash run_r2r/main_server.bash priorgt_eval_ss 2333
# bash run_r2r/main_server.bash priorgt_eval_grpo 2333
# bash run_r2r/main_server.bash priorgt_probe 2333       # quick verify: freeze base, ~3k steps
# bash run_r2r/main_server.bash priorgt_probe_eval 2333  # eval after probe
# bash run_r2r/main_server.bash imagined_dagger 2333
# bash run_r2r/main_server.bash imagined_eval_ss 2333
# bash run_r2r/main_server.bash imagined_grpo 2333
# bash run_r2r/main_server.bash imagined_eval_grpo 2333
# bash run_r2r/main_server.bash llm_dagger 2333
# bash run_r2r/main_server.bash llm_eval_ss 2333
# bash run_r2r/main_server.bash llm_grpo 2333
# bash run_r2r/main_server.bash llm_eval_grpo 2333
