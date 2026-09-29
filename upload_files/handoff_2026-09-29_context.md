# 交接文档：LLM-Grid Try5 栅格通路修复与 R11′/R16 实验（截至 2026-09-29）

用途：在新的对话中继续讨论时，把本文件整体贴入或 `@` 引用即可。本文件只写已验证的事实和当前状态；推测处均标明。

## 1. 环境与代码

| 项 | 内容 |
|---|---|
| 仓库 | GitHub `KAI1179/VLN-ETPR1`。代码分支 `exp/refiner`（实验实现）；Claude 的工作分支 `claude/awesome-ride-axuad6`（只放补丁、笔记、任务书，不含代码） |
| 补丁机制 | Claude 在 `scripts/distill/patches/0001–0022` 提交 `git format-patch` 文件；用户在 exp/refiner 上 `git am` 应用。同步命令：`git fetch -q origin claude/awesome-ride-axuad6 && git checkout origin/claude/awesome-ride-axuad6 -- scripts/distill/patches && git reset -q -- scripts/distill/patches`，然后 `git am scripts/distill/patches/00NN-*.patch` |
| 本地服务器 WZ | `/home/xukai/code/ETP-R1-snapshot/ETP-R1`，conda `etpr1-py38`（transformers 4.28.1），8 卡；训练 R11′ 占 0–3，评测用 4–7 |
| 云登录节点 ln01 | 用户 `scze096`；原始 checkout `~/run/ETP-R1/`（数据都在这里）；exp/refiner worktree `~/run/ETP-R1-snapshot/ETP-R1-refiner`（Deploy Key，权限限于 `~/run/ETP-R1-snapshot/`）；conda `etpr1-uv`（transformers 4.43.4，torch 2.1.2+cu118）；slurm 分区 `vip_gpu_scze096`，节点 8 卡按卡号独占分配，不支持跨作业共卡 |
| worktree 数据链接 | `data`、`pretrained`、`pretrain_src/datasets`、`pretrain_src/img_features` 均为指向 `~/run/ETP-R1/` 的符号链接（`ETP-R1-try5-r1p5` 下没有 R2R 预训练标注） |
| 分支状态 | 远端 exp/refiner 最新为 `084aeb3`（0020）。WZ 已 rebase 本地文档提交到其上并打 0021+0022（待推送）。ln01 worktree 有自己打的 0021（哈希不同），WZ 推送后需 `git pull --rebase origin exp/refiner` |
| 文档约定 | `docs/NOTE.md`、`docs/daily/` 用中文；代码、提交信息英文。逐日记录在 `docs/daily/2026-09-24 … 09-29.md` |

## 2. 模型与任务背景

- ETP-R1：图式 VLN-CE 策略，流程 预训练（MLM+SAP）→ DAgger → GRPO。
- Try5 = `GraphMapCrossAttention` 单向融合：认知地图 37×100×100 栅格 + 元数据 token（5 个方向向量、起始方向、起始位置）。
- 地图编码器 `EmbeddingGridMapEncoder`（`vlnce_baselines/models/etp_prior_gt/map_encoder.py`）：`category_projection` Conv2d 37→512（CLIP 文本嵌入初始化）→ `spatial_tokenizer` Conv2d 512→768，k=10,s=10 → LayerNorm → metadata_encoder → 2 层 transformer → output_norm。无位置编码，100 个空间 token 进 cross-attention 时是无序集合。
- 两条数据线：GT 认知地图（`gt.legacy.r1p5.direction5.v1`）与 LLM-Grid 地图（`llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree`）。
- 教师–学生蒸馏：冻结 GT 教师 `try-5-r1p5-dagger.iter16000.pth`（dz 位置特征，val_unseen SR 73.84），对 LLM-Grid 学生 `LLMGridTry5Policy` 做动作层 KL。

## 3. 已确认的核心发现

### 3.1 死的预训练 spatial_tokenizer（根因已确认）

- 预训练 checkpoint `pretrained/r2r_rxr_ce/llm_grid_try5/ckpts/model_step_460000.pt` 中 `spatial_tokenizer.weight` 全零、bias 为垃圾值（范数约 55 至 1e19 量级）。
- 根因：transformers ≥ 4.31（`etpr1-uv` 为 4.43.4）的 `from_pretrained()` 在 `no_init_weights()` 下构造模型，`torch.nn.init.*` 被空操作；BERT 的 `_init_weights` 只重初始化 Linear/Embedding/LayerNorm；起始 checkpoint（ETP-R1 原版 367500）里没有的 Conv2d 保留 `torch.empty` 内存。`nn.MultiheadAttention` 幸免（模块导入时绑定了 xavier 本地名）。复现脚本 `scripts/distill/repro_hf_no_init_conv.py`（4.43.4 复现，4.28.1 不复现）。
- 后果：巨大 bias 经 LayerNorm 后梯度消失，栅格通路不可训；fp16 下溢出 → R7 早期 NaN。
- 附带发现：历史 loader 缺陷——预训练把 fusion 存在 `bert.global_encoder.graph_map_attention.*`、map encoder 存在 `map_encoder.*`，导航模型从未加载（09-24 `ec5fde0` 才加入 `load_pretrained_map_modules`）。因此历史 try5 DAgger（65.42 SR）和 S4 refiner 实验的 map 模块都是 DAgger 从初始化训的，栅格通路是活的但很弱。

### 3.2 各策略对栅格的依赖（R2R val_unseen，SR）

| 策略 | 基线 p0 | 去掉栅格（metadata_only） | 去掉元数据（raster_only） | 说明 |
|---|---:|---:|---:|---|
| GT 教师 iter16000 | 73.84 | 56.93 | 69.33 | 真正依赖栅格 |
| R5 学生 10k（387500 起步，不加载 map 模块） | 65.63 | 63.02（−2.61，CI [+0.21, +4.50]） | 65.74 | 栅格贡献小；gt_full 65.52 无增益 |
| R7 学生 17k（460000 起步，加载了死 map 模块） | 66.72 | 66.72（逐 episode MD5 相同） | 65.25（metadata +1.47） | 栅格盲；p0 与 gt_full 字节相同 |
| S4 refiner 策略 15k | 64.27 | 未测 | 未测 | gt_seen 65.31，refiner 64.76，gt_full 64.60 |

结论：LLM-Grid 线从未有过"在预训练规模上训过栅格通路"的策略；R7 的 66.72 完全来自骨干 + 预训练 fusion + 元数据。

### 3.3 refiner 线（S3–S5）失败与死 tokenizer 无直接因果

- S4（09-05 至 09-10）虽以 460000 为 `PRETRAINED_CKPT`，但当时 loader 不加载 map 模块，tokenizer 是普通 `nn.Conv2d`，初始化正常；S5-D1 换栅格源结果会变，证明栅格活着。
- 真正瓶颈：策略对栅格依赖太弱（喂 GT 栅格只多 1 分），refiner 已拿到 gt_seen 的大部分（差 0.55）。
- refiner 实现：`vlnce_baselines/models/refiner/model.py` `CognitiveMapRefiner`（4 层 UNet，67→37 通道，7.8M 参数）；证据通道来自 Habitat GT 语义传感器 + 深度（不是检测器）；接入点 `ss_trainer_ETP_PriorGT.py` `_update_refined_cognitive_maps`，只替换 grid，不动元数据；权重 `data/refiner/checkpoints_aug/best.pt`。
- 未补的测量：`s4_try5_refiner` 的 checkpoint 没直接扫过，可用 `map_encoder_weight_sweep.py --checkpoint-globs 'data/logs/checkpoints/s4_try5_refiner/ckpt.iter15000.pth'` 补上。

## 4. 修复与工具（补丁 0009–0022 摘要）

| 补丁 | 内容 |
|---|---|
| 0009/0010 | map 路径 fp32（禁 autocast）；`nan_guard.py`（`ETP_NAN_DEBUG_STEPS`）|
| 0011–0013 | `map_sensitivity_probe.py`（LLM/GT/零/无地图 logits 对比）、`map_encoder_stage_probe.py`（逐级 rel diff）、`map_encoder_weight_sweep.py`（跨 ckpt 的 tokenizer 范数）|
| 0014/0015 | 启动器 `RUN_NAME`/`LOAD_MAP` 可覆盖；EGL 变量 |
| 0016 | `eval_student_ablation.sh RUN ITER MODE`（`metadata_only`=栅格清零，`raster_only`=元数据清零）|
| 0017/0018 | map_encoder 显式初始化 + `assert_initialised`；预训练启动时 `assert_conv_modules_initialised`；只查二维以上张量 |
| 0019 | `MODEL.MAP_ENCODER.reinit_spatial_tokenizer`；`REINIT_TOKENIZER=True` → RUN_NAME 后缀 `_reinit_tok`；加载到死 tokenizer 且未开重初始化则直接报错 |
| 0020 | 预训练每 `log_steps` 记录 tokenizer 权重/偏置/梯度范数（TB `map/*`）；启动 guard 扩展到全部参数；新启动器 `scripts/submit/llm-grid-try5-pretrain-fixedtok.sh`、`llm-grid-try5-fixedtok-dagger.sh`；`main_server.bash` 新模式 `llm_grid_try5_fixedtok_dagger`（`load_pretrained_map_modules True`）|
| 0021 | 两个 fixedtok 启动器先激活 conda 再 `set -u`（`etpr1-uv` 的 activate.d 钩子读未定义变量）|
| 0022 | `scripts/distill/eval_iters.sh RUN [ITER ...]`：多 ckpt 并行评测（`GPUS` 默认 4,5,6,7），已有结果跳过，末尾 `summarize_eval.py` 出表；结果目录 `<RUN>_eval_iter<N>_val_unseen`，与 `paired_analysis.py --run-a/--run-b` 兼容 |

## 5. 正在运行的实验

### R11′（WZ，卡 0–3）

- 启动：`REINIT_TOKENIZER=True CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/distill/run_dagger_distill_llmpt.sh`。
- 含义：460000 骨干 + 加载预训练 fusion（6 张量）+ 加载 map encoder（37 张量）但 **tokenizer 重初始化**（权重范数 15.97，偏置 0.0712），DAgger 从头训栅格；蒸馏教师同 R7。
- 目录：`data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_reinit_tok/`，ckpt 每 1000 iter，共 20000。
- 进度：iter 11200 于 09-28 21:32，约 506 iter/h，预计 09-29 15 点前后结束。student_ce 3.03→0.74，distill_loss 2.66→0.235。
- 待做：`bash scripts/distill/eval_iters.sh dagger_distill_gt_teacher_llmpt_reinit_tok 5000 7000 9000 11000`（之后加 13000–20000）；与 R7 同 iter 配对 `paired_analysis.py --run-a dagger_distill_gt_teacher_llmpt_reinit_tok --iter-a N --run-b dagger_distill_gt_teacher_llmpt --iter-b N`；栅格敏感度 `map_sensitivity_probe.py --run-name dagger_distill_gt_teacher_llmpt_reinit_tok --iter N`（期望 tok_rel>0、flip_gt>0）。

### R16（ln01，作业 1311965，g0042 卡 4–7）

- 启动：worktree 内 `sbatch scripts/submit/llm-grid-try5-pretrain-fixedtok.sh --num_train_steps 150000 --warmup_steps 10000`。
- 含义：从 ETP-R1 原版权重 `model_step_367500.pt` 重做 LLM-Grid try5 预训练，带 0017 修复；学习率线性预热 10k 再线性衰减到 0；batch 16×4；fp32；每 2500 步验证并存 ckpt（2.5 GB/个，共约 150 GB）。
- 输出：`~/run/ETP-R1/pretrained/r2r_rxr_ce/llm_grid_try5_fixedtok/ckpts/model_step_<N>.pt`；日志 `~/run/ETP-R1-snapshot/ETP-R1-refiner/slurm-llm-grid-try5-pt-fixedtok-1311965.out`。
- 进度：1.40 it/s，每 2500 步 31 分钟，预计 09-29 晚 20 点左右完成。tokenizer 健康：step 7000→8000 权重范数 16.67→16.79（初始化理论值 16.0），偏置 0.087→0.093，梯度范数约 1.8e-3。
- 完成后：先 `python scripts/distill/map_encoder_stage_probe.py --checkpoints pretrained/r2r_rxr_ce/llm_grid_try5_fixedtok/ckpts/model_step_150000.pt`（期望 rel(gt) 远大于 0），再 `LLM_GRID_TRY5_FIXEDTOK_PRETRAINED_CKPT=pretrained/r2r_rxr_ce/llm_grid_try5_fixedtok/ckpts/model_step_150000.pt sbatch scripts/submit/llm-grid-try5-fixedtok-dagger.sh`（原版 try5 DAgger 配置，30000 iter，无教师，与 65.42 那次直接对照；默认 ckpt 路径写的是 500000，必须覆盖）。蒸馏版可在 WZ 用 `run_dagger_distill_llmpt.sh` 换 `PRETRAINED_CKPT`。

### 三条线的对照设计

| 线 | 骨干 | fusion | 栅格通路 |
|---|---|---|---|
| R7 | 460000 | 预训练 | 死 |
| R11′ | 460000 | 预训练 | DAgger 从头训 |
| R16 + DAgger | 修复后重新预训练 | 预训练（活） | 预训练时训好 |

判据：R16 的 DAgger 若 gt_full 明显高于 p0、去掉栅格明显掉分，则 LLM-Grid 线的栅格通路终于可用，refiner 值得在其上重测；若仍不敏感，问题转向架构（无位置编码等）。

## 6. 参考数值（R2R val_unseen）

| 行 | SR | SPL | OSR |
|---|---:|---:|---:|
| Baseline DAgger（作者） | 63.13 | 54.23 | 68.52 |
| 历史 LLM-Grid try5 DAgger iter28000（460000 起步，未加载 map 模块） | 65.42 | 55.25 | 70.58 |
| LLM-Grid GRPO matched nav4 | 66.50 | 55.31 | 73.30 |
| R5 10k | 65.63 | 54.14 | 72.16 |
| R7 17k | 66.72 | 56.02 | 72.43 |
| GT 教师 iter16000（dz） | 73.84 | — | — |
| GT 教师 iter16000 @121c369 | 74.23 | 63.38 | 77.98 |

## 7. 与用户的协作约定

- 中文对话；命令一次只给一两条，避免长链和 `$VAR` 循环（用户明确要求）。
- Claude 不启动、重启、终止训练，不碰 tmux；实验由用户决定并亲自执行。
- 出现 bug 早说，不反复打补丁；补丁应用失败先看是否已在远端。
- 云端 `etpr1-uv` 环境未被修改（已确认）；Deploy Key 只覆盖 `~/run/ETP-R1-snapshot/`。

## 8. 待办清单

1. WZ：`git push origin exp/refiner`（0021+0022 已打）；跑 `eval_iters.sh` 评 R11′ 5000/7000/9000/11000，贴表。
2. R11′ 跑完后评 13000–20000，与 R7 配对分析，并做 `map_sensitivity_probe.py` / `eval_student_ablation.sh metadata_only` 看栅格是否被用上。
3. R16 完成后按 §5 做 probe + DAgger；DAgger 结束后做同样的栅格敏感度与消融。
4. ln01 worktree `git pull --rebase origin exp/refiner` 同步（R16 结束后再做）。
5. 可选：扫 `s4_try5_refiner` checkpoint 的 tokenizer 范数；提交 WZ 上未提交的 `reports/refiner/S4_progress.md`。

## 9. 可讨论的开放问题

- R16 预训练 150k 步是否足够让栅格通路学到可用的表征（对照 460000 用了 500k 中的 460k）。
- map encoder 缺位置编码是否是 LLM-Grid 线栅格贡献弱的结构性原因；若是，改法与重训成本。
- R11′（DAgger 阶段从头训栅格）与 R16（预训练阶段训栅格）若结果接近，说明栅格信息本身对 R2R 的价值有限，应回到地图质量或任务设计层面。
- refiner 在"栅格真正被用上"的策略上的重测方案。
