export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/gpu-detection.bash"
configure_distributed_gpu_vars

flag_template=" --exp_name release_r2r_dagger
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS ${GPU_IDS}
      TORCH_GPU_IDS ${GPU_IDS}
      GPU_NUMBERS ${GPU_NUMBERS}
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      IL.back_algo control
      MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dagger/ckpt.iter{ckpt_num}.pth
      "

# 初始 ckpt 迭代数字，最终停止的数字
start_ckpt=35000
end_ckpt=25000
step=-200

# 日志文件
log_file="data/logs/checkpoints/release_r2r_dagger/eval_${start_ckpt}-${end_ckpt}.txt"

# 开始循环
for ((ckpt_num=$start_ckpt; ckpt_num>=$end_ckpt; ckpt_num+=$step)); do
    echo "###### eval mode with ckpt.iter${ckpt_num}.pth ######" | tee -a $log_file
    # 将 ckpt_num 替换到 flag 中
    flag=$(echo "$flag_template" | sed "s/{ckpt_num}/$ckpt_num/")
    # 运行 eval 模式
    torchrun --standalone --nproc-per-node="${NPROC_PER_NODE}" run.py $flag 2>&1 | stdbuf -oL grep -v 'it/s' | tee -a $log_file

done

# Uses all visible GPUs by default. Set CUDA_VISIBLE_DEVICES first to restrict cards.
# bash run_r2r/auto_eval_server.bash
