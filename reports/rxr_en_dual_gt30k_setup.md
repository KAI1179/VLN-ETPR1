# RxR-CE English dual student with joint GT 30K teacher

## 设置与检查

实验在 WZ 的 /home/xukai/code/ETP-R1-snapshot/ETP-R1 编写，不修改 /data 仓库。保持 kd_dual 的双分支 CE、两路教师 KL、map→zero KL；实际 rollout 使用 zero 分支，教师冻结。lr=1e-5、seed=100、30K iter、4 rank×4 env、200 iter 保存、load_pretrained_map_modules=False。学生仍从 LLM 460000 开始，不从教师权重初始化。

教师换为联合 R2R+RxR English 的 ckpt.iter30000.pth，GT namespace 仍是 gt.legacy.r1p5.direction5.v1。CPU 对比教师与现有 D17K：980 个参数键，missing/unexpected=0/0、shape mismatch=0。教师训练版本 3bdad49（partner 仓库的提交，不在 KAI1179/VLN-ETPR1 内）的 graph_utils.py:47 使用 arcsin(dy/distance)，与本仓库 HEAD 一致，故教师与学生均用 y。

数据使用 RxR English guide（en-US/en-IN），train_90 共 17972 个英语 episode；val_seen 2255、val_unseen 3669。RxR 原生视角 63°、高层上限 25、文本上限 250，使用已有 hfov63 waypoint 权重（data/wp_pred/check_cwp_bestdist_hfov63，启动检查会 assert）。没有排除 99 个历史不可达 val_unseen episode。

最小共享修改：两处 cache ID 的 RxR 前缀由错误 RXR 改为 RxR；cache_report 同样规范化名称；旧 launcher 仅新增 EXP_CONFIG 环境覆盖，默认仍使用原 R2R 配置。既有 R2R 行为不变，未修改训练损失、模型结构或默认配置。

CPU 配置检查通过，RxR/R2R 回归测试 8 项通过；未启动 GPU、torchrun 或 run.py。**WZ 本地 RxR LLM navigation cache 为 0，训练尚未端到端验证。** WZ 上 GT train_90 缺 38 个、val_seen 缺 18 个、val_unseen 缺 0 个；启动检查会对"有学生缓存但缺教师 GT"的样本直接报错，不回退、不偷偷跳过。缺失的 38 个若是英语 episode，云端生成完学生缓存后就会被拦住，需先补齐这部分 GT（或核对 ln01 同一 namespace 下 RxR 的 GT 是否本来就齐——该 namespace 在 ln01 之前只验证过 R2R）。

## 云端准备（ln01）

云端仓库是 `~/run/ETP-R1-snapshot/ETP-R1-refiner`（`~/run/ETP-R1` 是 partner 的只读仓库，不要用）。该 clone 当前是 `git am` 过若干 patch 的 exp/refiner，SHA 与本分支不同，直接整体切到本分支即可；data/、pretrained/ 的软链是未跟踪文件，不受 checkout 影响：

```bash
cd ~/run/ETP-R1-snapshot/ETP-R1-refiner
git fetch origin exp/rxr-en-dual-gt30k && git checkout exp/rxr-en-dual-gt30k
```

需要上传/确认的数据：联合教师 ckpt.iter30000.pth、学生预训练 model_step_460000.pt（ln01 已有：`~/run/ETP-R1/pretrained/r2r_rxr_ce/llm_grid_try5/store2/model_step_460000.pt`）、gt.legacy.r1p5.direction5.v1 的 RxR 部分、RxR dataset 与 GT 轨迹、MP3D 场景、hfov63 waypoint 及 RGB/depth 模型。预训练 mixed 下的数值 instruction-ID 缓存不是 RxR episode 导航缓存，不能直接代替。

RxR navigation cache：若已有，按同 model_key 的 rxr/{train,val_seen,val_unseen} 布局放到 data/ 下即可；若没有，用下面的 sbatch 生成。LLM LoRA 不需要另外上传——R2R 缓存当初就是在 ln01 用 `outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag/checkpoints/final` 生成的（见 scripts/submit/llm-grid-nav-cache-r1p5.sh），脚本默认指向它。生成器只做 RxR、只做英语（prior/vlnce.py 过滤 en-*），覆盖 train 全量英语（约 25920，train_90 的 17972 是其子集）加 val_seen/val_unseen。单卡串行会很慢，按 cache id 的确定性哈希分片，N 个作业同时提交互不重叠：

```bash
cd ~/run/ETP-R1-snapshot/ETP-R1-refiner
for k in 0 1 2 3; do sbatch --export=ALL,WORKER_COUNT=4,WORKER_INDEX=$k scripts/submit/rxr-en-nav-cache.sh; done
```

重跑同一分片会跳过已完成的 episode。生成完成后先跑只读检查，看 llm_available 和 teacher_missing_for_available_students：

```bash
cd ~/run/ETP-R1-snapshot/ETP-R1-refiner && PYTHONPATH=. CUDA_VISIBLE_DEVICES='' python scripts/distill/check_rxr_en_dual.py
```

## 云端训练

实际路径以云端上传位置为准；Slurm 导出 CUDA_VISIBLE_DEVICES（未导出时 sbatch 脚本从 nvidia-smi 取），launcher 要求恰好 4 卡，不自动使用全部服务器 GPU。

```bash
cd ~/run/ETP-R1-snapshot/ETP-R1-refiner
export GT_TEACHER_CKPT=/云端/联合GT/ckpt.iter30000.pth
export PRETRAINED_CKPT=$HOME/run/ETP-R1/pretrained/r2r_rxr_ce/llm_grid_try5/store2/model_step_460000.pt
sbatch --export=ALL scripts/submit/rxr-en-dual-gt30k.sh
```

输出 data/logs/checkpoints/kd_dual_rxr_en_gt30k/，train.log、launch_info.txt 及 checkpoint 独立；重跑沿用旧 launcher 的断点恢复机制。默认端口 29710，不与本地当前训练冲突。

WZ 本地只读 CPU 检查命令：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /home/xukai/anaconda3/envs/etpr1-py38/bin/python scripts/distill/check_rxr_en_dual.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' /home/xukai/anaconda3/envs/etpr1-py38/bin/python -m pytest tests/etp_llm/test_rxr_distill_paths.py -q
```
