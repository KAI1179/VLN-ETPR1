# RxR-CE English dual student with joint GT 30K teacher

## 设置与检查

实验在当前 /home/xukai/code/ETP-R1-snapshot/ETP-R1 编写，不修改 /data 仓库。保持 kd_dual 的双分支 CE、两路教师 KL、map→zero KL；实际 rollout 使用 zero 分支，教师冻结。lr=1e-5、seed=100、30K iter、4 rank×4 env、200 iter 保存、load_pretrained_map_modules=False。学生仍从 LLM 460000 开始，不从教师权重初始化。

教师换为联合 R2R+RxR English 的 ckpt.iter30000.pth，GT namespace 仍是 gt.legacy.r1p5.direction5.v1。CPU 对比教师与现有 D17K：980 个参数键，missing/unexpected=0/0、shape mismatch=0。训练版本 3bdad49 的 graph_utils.py:47 使用 arcsin(dy/distance)，故教师与学生均用 y。

数据使用 RxR English guide（en-US/en-IN），train_90 共17972个英语 episode；val_seen2255、val_unseen3669。RxR 原生视角63°、高层上限25、文本上限250，使用已有 hfov63 waypoint 权重。没有排除99个历史不可达 val_unseen episode。

最小共享修改：两处 cache ID 的 RxR 前缀由错误 RXR 改为 RxR；cache_report 同样规范化名称；旧 launcher 仅新增 EXP_CONFIG 环境覆盖，默认仍使用原 R2R 配置。既有 R2R 行为不变，未修改训练损失、模型结构或默认配置。

CPU 配置检查通过，RxR/R2R 回归测试8项通过；未启动 GPU、torchrun 或 run.py。**本地 RxR LLM navigation cache 为0，训练尚未端到端验证。** GT train_90缺38个、val_seen缺18个、val_unseen缺0个；启动检查会对“有学生缓存但缺教师GT”的样本直接报错，不回退、不偷偷跳过。需补齐这部分 GT 或明确修复其生成原因。

## 云端准备

上传联合教师30000、学生预训练460000、GT namespace、RxR dataset与GT轨迹、MP3D场景、hfov63 waypoint及RGB/depth模型。缓存预训练 mixed 下的数值 instruction-ID 文件不是 RxR episode 导航缓存，不能直接代替。

若已有 RxR navigation cache，按同 model_key 的 rxr/{train,val_seen,val_unseen} 布局上传；若没有，以下入口只生成 RxR English，复用原生成方法，不改变原 R2R 缓存生成默认设置。需要另行上传原 LLM LoRA checkpoint，并由用户启动 GPU：

```bash
cd ~/run/ETP-R1-snapshot/ETP-R1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python scripts/distill/cache_rxr_en_llm_grid.py \
  --model-name-or-path /云端/原LLM-LoRA-checkpoint \
  --cache-model-key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
  --scale 2 --batch-size 8 --max-new-tokens 4096
```

此命令只给出，不由CLI执行；使用原训练LLM checkpoint，不建议换预测器导致额外实验变量。

## 云端训练

实际路径以云端上传位置为准；Slurm 分配 CUDA_VISIBLE_DEVICES，脚本要求4卡，不自动使用全部服务器GPU。

```bash
cd ~/run/ETP-R1-snapshot/ETP-R1
export GT_TEACHER_CKPT=/云端/联合GT/ckpt.iter30000.pth
export PRETRAINED_CKPT=/云端/LLM预训练/model_step_460000.pt
sbatch --export=ALL scripts/submit/rxr-en-dual-gt30k.sh
```

输出 data/logs/checkpoints/kd_dual_rxr_en_gt30k/，train.log、launch_info.txt及checkpoint独立；重跑沿用旧launcher的断点恢复机制。默认端口29710，不与本地当前训练冲突。

本地只读CPU检查命令：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /home/xukai/anaconda3/envs/etpr1-py38/bin/python scripts/distill/check_rxr_en_dual.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /home/xukai/anaconda3/envs/etpr1-py38/bin/python -m pytest tests/etp_llm/test_rxr_distill_paths.py -q
```
