# 栅格认知地图：是否被加载、是否有效 —— 实验、实现与结果整合报告

日期：2026-09-30。范围：ETP-R1 / LLM-Grid Try5 线上关于 37×100×100 栅格认知地图的全部核查与实验（R5–R16、S3–S5、GRPO）。除特别标注外，指标均为 R2R val_unseen（1839 episode）SR，单 seed。

## 1. 一页结论

1. **两个预训练 checkpoint（Prior-GT 387500、LLM-Grid 460000）的栅格编码器都是死的**：`spatial_tokenizer.weight` 精确为 0，偏置是未初始化内存（387500 范数 55，460000 范数 2.4e36）。栅格信号在 10×10 卷积处消失，编码器只剩元数据 token 起作用。
2. **根因是 HuggingFace `from_pretrained` 与 transformers ≥ 4.31 的初始化行为**：模型在 `no_init_weights()` 下构造，`torch.nn.init.*` 被空操作；事后 `_init_weights` 只重初始化 Linear / Embedding / LayerNorm，`nn.Conv2d` 不在其列，起始 checkpoint 里没有的卷积保持 `torch.empty` 原始内存。云端预训练环境 `etpr1-uv` 的 transformers 为 4.43.4，本地 `etpr1-py38` 为 4.28.1 不复现。已用独立脚本复现。
3. **历史上所有 DAgger 都没有加载预训练的 map 模块**（loader 缺陷，09-24 才修）。因此历史 try5（65.42）和 refiner 线 S4（64.76）的栅格通路都是 DAgger 从初始化训的，是活的但很弱；它们反而躲开了死编码器。
4. **加载了死编码器的 R7 是"栅格盲"策略**：p0 与 gt_full 逐 episode 字节相同，栅格置零结果不变。它的 66.72 来自 460000 骨干 + 预训练 fusion + 元数据 token（元数据贡献 +1.47，CI [+0.07, +2.78]）。
5. **用 LLM 栅格在 DAgger 阶段从头训栅格通路，收益 ≤ +2.6 SR，且训完仍用不上 GT 栅格**（R5：栅格置零 −2.61，CI [−4.50, −0.21]；gt_full 对 p0 无增益）。把 R7 的 tokenizer 重初始化再 DAgger（R11′）与 R7 训练损失逐 iter 重合，评测差在噪声内。
6. **但"DAgger 学不动栅格通路"不成立。** GT 教师的 map 模块同样是 DAgger 从初始化训出来的（其预训练 387500 的 tokenizer 也是死的，当时 loader 不加载，教师 ckpt 的 tokenizer 范数 17.85），却从栅格拿到约 17 分（none 73.84 / raster_only 69.33 / metadata_only 56.93）。学不动的是 **LLM 栅格**：LLM 地图喂给 GT 教师只有 49.48，低于无地图的 57.04。教师与学生的差别在于地图真假，以及节点能否定位到地图上（教师用 dz 位置特征，恰是地图 x/z 轴的分量；学生用 dy，节点特征完全自我中心）。
7. **跨 episode 置换测试（patch 0024）证明历史 try5 策略不使用栅格中任何 episode 或场景特异信息**：把栅格换成别场景的 LLM 栅格，SR 不掉反升（control 65.09，cross_scene 66.01 / 65.91 / 65.96 三个 donor seed，配对 +0.9，CI [+0.11, +1.86]）；同场景他人栅格 65.20 ≈ control。全零置零掉的 1.4 是分布外代价，不是信息损失。
8. **修复后重做 LLM-Grid try5 预训练（R16，150k 步）得到活的编码器**：tokenizer 权重范数 17.08、偏置 0.28，信号/偏置比 83（教师 84.7），输出对 GT 栅格的相对变化 0.36。其 DAgger 正在云端运行；判定改用 control vs cross_scene 配对（不用 metadata_only）。
9. **方案修正为坐标融合（patch 0025/0026，默认关）**：节点按与位置特征相同的换算映射到栅格坐标，cross-attention 加逐 head 固定距离偏置，map encoder 加固定 2D 位置码，不增加参数；计划 E1（GT 在线地图，387500 起步，对照 dz 教师 73.84）与 E2（LLM 地图，460000 起步，对照 control 65.09）并行。
10. GRPO 接 R7 17k 无增益（最高 66.29 对起点 66.72）。

## 2. 模块与数据

### 2.1 认知地图

- 栅格 `grid`：37 通道 × 100 × 100，0.5 m/格；27 个物体类别 + 10 个区域类别。
- 元数据：`map_trajectory_metadata`（5 个方向关键点，5×2）、`start_direction_vector`（2）、`start_position`（2）。
- 两条数据线：GT 认知地图 `gt.legacy.r1p5.direction5.v1`（由 Habitat GT 语义投影生成）；LLM-Grid 地图 `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree`（LLM 从指令推断，缓存于 `data/llm_navigation/`）。LLM 栅格与 GT 的 val_unseen 物体 mIoU：100×100 约 4–5%，10×10 约 12–14%。

### 2.2 地图编码器 `EmbeddingGridMapEncoder`（`vlnce_baselines/models/etp_prior_gt/map_encoder.py`）

```
grid (B,37,100,100)
  → category_projection  Conv2d 37→512, k=1（CLIP 文本嵌入初始化）
  → spatial_tokenizer    Conv2d 512→768, k=10, s=10   → 100 个空间 token
  → LayerNorm
  → metadata_encoder(MLP) → 1 个元数据 token
  → 2 层 TransformerEncoder（无位置编码，101 个 token 为无序集合）
  → output_norm → map_tokens (B,101,768), map_token_masks
```

### 2.3 融合 `GraphMapCrossAttention`（`map_fusion.py`，try5）

图节点作为 query 对 101 个 map token 做 cross-attention，经 `residual_projection`（零初始化）残差加回节点表征；共 6 个张量。单向融合，地图不反向读图。

### 2.4 权重在 checkpoint 里的位置

| 模块 | 预训练 checkpoint 键前缀 | 导航模型键 |
|---|---|---|
| map encoder（37 张量） | `map_encoder.*` | `net.map_encoder.*` |
| fusion（6 张量） | `bert.global_encoder.graph_map_attention.*` | `net.vln_bert.graph_map_attention.*` |

09-24 之前的 loader 只按关键字加载导航头，这 43 个张量被当作"unused"丢弃：**所有历史 DAgger 都从初始化训 map 模块**（R13 审计确认历史 try5 run 亦如此）。patch 0001/0006 增加 `MODEL.MAP_ENCODER.load_pretrained_map_modules`（默认 True）和 fusion 键重映射，并在加载后校验张量确已落位。

### 2.5 预训练配方

`pretrain_src/run_pt/run_mix_server.bash`，`mix_pretrain_server.json`：AdamW，lr 5e-5，wd 0.01，梯度裁剪 5，线性预热后线性衰减到 0，batch 16 × 4 卡，fp32，MLM + SAP，起始权重 ETP-R1 原版 `model_step_367500.pt`。原 LLM-Grid try5 预训练 500k 步，DAgger 取 460000。

## 3. 问题发现的时间线

| 步骤 | 观察 | 结论 |
|---|---|---|
| R7 首启 | `Categorical(nav_probs)` NaN | 先以 fp32 map 路径绕过（patch 0009）；真实来源后来确认为死 tokenizer 的巨大偏置在 fp16 溢出 |
| R8 | R7 5000 的 p0 与 gt_full 评测逐 episode MD5 相同 | 换栅格不改变任何输出：模型侧栅格盲，不是评测脚本问题（`trace_gt_full_swap.py` 证实 swap 生效） |
| R9 | 逐级探针 `map_encoder_stage_probe.py` | `category_projection` 层 rel(gt)=0.97，`spatial_tokenizer` 层起为 0：信号死于 10×10 卷积 |
| R10 | `map_encoder_weight_sweep.py` 扫所有 checkpoint | 387500 与 460000 的 tokenizer 权重全 0，偏置 55 / 2.4e36；从预训练第一个存档起就是如此 |
| R12 | 学生 / 教师通道消融 | 见第 5 节 |
| R13 | 历史 try5 加载审计 | 历史 DAgger 未加载 map 模块，tokenizer 是普通 `nn.Conv2d` 初始化，活着 |
| R15 | 根因排查（CLI 自主，GPU 0–3） | 复现 HF `no_init_weights` 行为，见第 4 节 |

## 4. 根因与机制

### 4.1 机制

预训练用 `GlocalTextPathNavCMT.from_pretrained(None, config=..., state_dict=...)` 构造模型。transformers ≥ 4.31 在 `from_pretrained` 期间进入 `no_init_weights()` 上下文，把 `torch.nn.init.*` 全部替换为空操作；构造完成后按 `_init_weights` 只对 Linear / Embedding / LayerNorm 重新初始化。`nn.Conv2d` 的 `reset_parameters` 调用的是被空操作的 `kaiming_uniform_`，因此：

- `spatial_tokenizer.weight`（768×512×10×10，3930 万参数）保持 `torch.empty` 分配的内存，恰好为全 0；
- `spatial_tokenizer.bias` 保持垃圾值，巨大；
- 起始 checkpoint（ETP-R1 原版 367500）里没有这两个张量，`state_dict` 覆盖不到。

`nn.MultiheadAttention` 幸免（PyTorch 在模块导入时把 `xavier_uniform_` 绑定为本地名，不受上下文影响）。`category_projection` 的 CLIP 初始化在构造后显式赋值，不受影响，但由于下游权重为 0，它在整个预训练期间收不到梯度，保持 CLIP 初始值。

### 4.2 为什么训不动

偏置范数 55 至 1e36 经 LayerNorm 后，每个位置的输出是同一个常数向量；LayerNorm 对输入的雅可比按 1/std 缩放，std 极大导致回传到 W 的梯度趋近 0。AdamW 也救不了：梯度本身在数值上消失。结果：预训练 460k 步，W 始终为 0。

### 4.3 影响范围

同一预训练模型里所有"不是 Linear/Embedding/LayerNorm、且不在起始 checkpoint 里"的参数都受影响（卷积、裸 `nn.Parameter`）。imagined 线的 `map_decoder`/`map_predictor` 若含卷积，其预训练权重同样可疑，重用前需用 `map_encoder_weight_sweep.py` 核查。

### 4.4 复现

`scripts/distill/repro_hf_no_init_conv.py`：玩具 `PreTrainedModel` 含 Conv2d / Linear / MHA，`from_pretrained` 后 transformers 4.43.4 下 Conv2d 权重为 0，4.28.1 下正常。云端 `etpr1-uv`：transformers 4.43.4，torch 2.1.2+cu118（已确认环境未被修改）。

## 5. 消融与探针数据

### 5.1 教师通道消融（GT 教师 `try-5-r1p5-dagger.iter16000.pth`，dz）

| 条件 | SR |
|---|---:|
| none（完整） | 73.84 |
| no_direction | 71.02 |
| raster_only（元数据置零） | 69.33 |
| metadata_only（栅格置零） | 56.93 |
| nomap | 57.04 |
| 喂 LLM 栅格 | 49.48 |

教师的栅格价值约 17 分；LLM 栅格对 GT 读图器是负信息。

### 5.2 学生通道消融（R12）

| 学生 | 条件 | SR | SPL | OSR | 说明 |
|---|---|---:|---:|---:|---|
| R5 10k（387500 起步，未加载 map 模块） | p0 | 65.63 | 54.14 | 72.16 | 基线 |
| | metadata_only | 63.02 | 53.25 | 67.86 | 栅格贡献 +2.61，CI [+0.21, +4.50]，主要作用于 OSR |
| | raster_only | 65.74 | 54.24 | 72.21 | 元数据贡献 ≈ 0 |
| | no_direction | 65.69 | 54.23 | 72.16 | 方向向量 ≈ 0 |
| | gt_full | 65.52 | — | — | GT 栅格无增益 |
| R7 17k（460000 起步，加载死 map 模块） | p0 | 66.72 | 56.02 | 72.43 | 基线 |
| | metadata_only | 66.72 | 56.02 | 72.43 | 逐 episode MD5 相同：栅格盲 |
| | raster_only | 65.25 | 55.35 | 71.29 | 元数据贡献 +1.47，CI [+0.07, +2.78] |
| | no_direction | 66.67 | 56.00 | 72.43 | ≈ 0 |

### 5.3 逐级探针（`map_encoder_stage_probe.py`，5 个 val_unseen episode）

| checkpoint | tokenizer 权重范数 | 偏置范数 | ‖W·x‖/‖b‖ | output_norm rel(gt) |
|---|---:|---:|---:|---:|
| LLM-Grid 预训练 460000（R7 起点） | 0 | inf（2.4e36） | 0 | 0 |
| Prior-GT 预训练 387500（R5 起点） | 0 | 55.18 | 0 | 0 |
| R7 DAgger 5000 | 0 | inf | 0 | 0 |
| R5 DAgger 10000 | 17.16 | 0.072 | 39.6 | 0.64 |
| GT 教师 DAgger 16000 | 17.85 | 0.075 | 84.7 | 0.78 |
| 历史 try5 DAgger iter28000 | 19.1 | — | — | — |
| **R16 预训练 142500** | **17.08** | **0.28** | **83.0** | **0.36** |

注意：R5、GT 教师、历史 try5 三行的 map 模块都是 DAgger 从初始化训出来的（当时 loader 不加载预训练 map 模块），tokenizer 活着；GT 教师的这条 DAgger 训出的通路值 17 分。

rel(gt) = ‖stage(GT grid) − stage(LLM grid)‖ / ‖stage(LLM grid)‖。R16 各级：category_projection 0.970、spatial_tokenizer 0.882、spatial_token_norm 0.386、token_transformer 0.357、output_norm 0.363；rel(zero) 0.436。

## 6. 实验矩阵

| 实验 | 骨干 | map 模块来源 | tokenizer 状态 | 训练 | 最好 SR（iter） | 栅格是否有效 |
|---|---|---|---|---|---:|---|
| 历史 try5 DAgger | 460000 | DAgger 从初始化 | 活（DAgger 训） | 30k iter，无教师 | 65.42（28000） | 弱（未消融） |
| 历史 try5 GRPO nav4 | 上者 28000 | 冻结 | 同上 | 500 iter | 66.50（500） | — |
| S4 refiner DAgger | 460000 | DAgger 从初始化 | 活（DAgger 训） | 30k，refiner 在线精炼栅格 | 64.76（15000） | gt_seen 仅 +1.0 |
| R5 蒸馏 | 387500 | DAgger 从初始化 | 活（DAgger 训） | 20k，GT 教师 KL | 65.63（10000） | +2.6（OSR） |
| R7 蒸馏 | 460000 | 加载预训练 | **死** | 20k，GT 教师 KL | 66.72（17000） | 0（盲）；元数据 +1.5 |
| R11′ 蒸馏 | 460000 | 加载预训练，tokenizer 重初始化 | 活（DAgger 训） | 20k，GT 教师 KL | 65.09（13000） | 训练损失与 R7 重合 → ≈ 0 |
| GRPO on R7 17k | R7 17k | 冻结 | 死 | 500 iter nav4 | 66.29（300） | — |
| R16 预训练（修复后） | 367500 → 150k 步 | — | **活** | MLM+SAP | 见 §9 | 编码器活；策略层待 DAgger |
| R16 DAgger | R16 142500 | 加载预训练（活） | 活（预训练） | 30k，无教师 | 进行中 | 待测 |

## 7. R7 与 R11′ 的对比（重初始化 tokenizer 是否有用）

### 7.1 评测逐 iter 对齐

| iter | R7 | R11′ |
|---:|---:|---:|
| 7000 | 63.02 | 63.08 |
| 9000–11000 | 63.30（10k） | 63.46 / 64.17 |
| 13000 | 65.31 | 65.09 |
| 15000 | 65.20 | 64.82 |
| 17000 | **66.72** | 64.60 |
| 18000–20000 | 65.25（20k） | 63.68（18k） |

三个点重合在 0.4 以内；"低 1.6"几乎全来自 R7 17k 的尖峰（前后点 65.20、65.25）。平台期均值 R7 65.62（去尖峰 65.25）对 R11′ 64.71。

### 7.2 训练损失逐 iter 对齐（student_ce / distill_loss）

| iter | R7 | R11′ |
|---:|---|---|
| 200 | 3.121 / — | 3.025 / 2.663 |
| 2000 | 1.037 / 0.583 | 1.007 / 0.580 |
| 5000 | 0.886 / 0.393 | 0.822 / 0.377 |
| 8000 | 0.814 / 0.281 | 0.842 / 0.290 |
| 11000 | 0.690 / 0.220 | 0.719 / 0.232 |
| 14000 | 0.621 / 0.186 | 0.672 / 0.194 |
| 17000 | 0.560 / 0.161 | 0.556 / 0.161 |

完全重合。结论：活的栅格通路在 DAgger 规模下既没有让学生更接近重度依赖栅格的教师，也没有造成扰动成本；R7 与 R11′ 是同一实验的两次噪声采样。结合 GT 教师的证据（同样 DAgger 从初始化训出栅格通路，却拿到 17 分）和置换测试（第 10b 节），"通路在 DAgger 里学不动"可以排除，问题在 LLM 栅格本身的信息与坐标对齐。

### 7.3 R11′ 实现

patch 0019：`MODEL.MAP_ENCODER.reinit_spatial_tokenizer`；加载 37 个预训练 map 张量后对 tokenizer 做 `uniform_(±1/√51200)`（权重范数 15.97，偏置 0.071）；未开此开关而加载到死 tokenizer 时直接报错。启动：`REINIT_TOKENIZER=True CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/distill/run_dagger_distill_llmpt.sh`，输出 `data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_reinit_tok/`。

## 8. refiner 线（S3–S5）的重新解读

- refiner：`vlnce_baselines/models/refiner/model.py` `CognitiveMapRefiner`，4 层 UNet，输入 67 通道（P0 的 37 + 证据 30），输出 37，7.8M 参数；证据来自 Habitat GT 语义传感器 + 深度的 12 视角投影（不是检测器）；训练目标 GT 栅格，BCE；接入点 `ss_trainer_ETP_PriorGT.py` `_update_refined_cognitive_maps`，只替换 `grid`，不动元数据。
- 地图质量确有改善：val_unseen 物体 mIoU 100×100 从 3.99 到 17.23，10×10 从 11.8 到 32.7。
- S5-D1 在同一 S4 策略上换栅格源：p0 64.27 / refiner 64.76 / gt_seen 65.31 / gt_full 64.60。refiner 离 gt_seen 只差 0.55，但 gt_seen 本身只比 p0 高 1 分。
- **与死 tokenizer 无直接因果**：S4 期间 loader 不加载 map 模块，栅格通路是活的（S5-D1 换栅格结果会变，死栅格的 R7 则字节相同）。真正的瓶颈是策略对栅格依赖太弱，根源同第 7 节：栅格通路从未在预训练规模上训过。
- 未补的测量：`s4_try5_refiner` checkpoint 的 tokenizer 范数扫描（Step-0 检查当时崩溃）。

## 9. R16：修复后重做预训练

### 9.1 修复

- patch 0017：`EmbeddingGridMapEncoder` 构造时显式调用 `reinit_spatial_tokenizer()`（不依赖 `torch.nn.init` 上下文），`assert_initialised` 拒绝二维以上全零或非有限参数；预训练 `train_r2r.py` 在 `from_pretrained` 后 `assert_conv_modules_initialised(model)`；测试 `tests/etp_prior_gt/test_map_encoder_init_under_hf.py` 在 HF 上下文下验证。
- patch 0020：预训练每 `log_steps` 记录 tokenizer 权重/偏置/梯度范数（TensorBoard `map/*`）；启动 guard 扩展到全部参数；启动器 `scripts/submit/llm-grid-try5-pretrain-fixedtok.sh`、`llm-grid-try5-fixedtok-dagger.sh`；`main_server.bash` 新模式 `llm_grid_try5_fixedtok_dagger`（`load_pretrained_map_modules True`）。
- patch 0021：启动器先激活 conda 再 `set -u`（`etpr1-uv` 的 activate.d 钩子读未定义变量）。

### 9.2 运行

登录节点 worktree `~/run/ETP-R1-snapshot/ETP-R1-refiner`，作业 1311965，g0042 卡 4–7，`--num_train_steps 150000 --warmup_steps 10000`，1.40 it/s，每 2500 步验证并存 ckpt（2.5 GB）。

### 9.3 日志

- tokenizer：权重范数 16.0 → 17.36（25k）→ 17.12（150k，后段由 weight decay 缓慢拉低），偏置 0.07 → 0.28，梯度范数约 1e-4，随学习率归零。
- SAP 验证准确率自约 90k 起平台：R2R unseen 78.3–80.7（噪声约 ±0.7），RxR 77–79.8。
- 选 **142500**（R2R 80.63 / RxR 79.31，两项之和在后 1/3 段最高，学习率已充分衰减）。

### 9.4 probe

见 §5.3：信号/偏置比 83.0，output rel(gt) 0.363。编码器确认为活。rel(gt) 低于 R5 与教师，但那两者是 DAgger 微调后的值，待 R16-DAgger 的 ckpt 再扫。

### 9.5 DAgger

`LLM_GRID_TRY5_FIXEDTOK_PRETRAINED_CKPT=pretrained/r2r_rxr_ce/llm_grid_try5_fixedtok/ckpts/model_step_142500.pt sbatch scripts/submit/llm-grid-try5-fixedtok-dagger.sh`，输出 `data/logs/checkpoints/release_r2r_llm_grid_try5_fixedtok_dagger/`，原版 try5 DAgger 配置（30000 iter，无教师，4 卡 × 4 环境），与历史 65.42 直接对照。已提交，进行中。

判定（按 patch 0024 的方法修正）：DAgger 后做 control vs cross_scene 配对（`eval_raster_donor.sh`，栅格换成别场景的 LLM 栅格，元数据不变），不用 metadata_only（全零输入有分布外代价，会把 OOD 损失误判为信息损失）。cross_scene 明显低于 control → 预训练规模训出的通路确实用上了 LLM 栅格的场景特异信息；cross_scene ≈ control（如历史 try5）→ 预训练也没让 LLM 栅格产生可用信息，转向坐标融合（E1/E2）与地图坐标系。

## 10. GRPO

`scripts/distill/run_grpo_from_dagger.sh`（patch 0023）接 R7 17k：nav4 profile（234 张量，map encoder 与 fusion 冻结），`load_pretrained_map_modules False`（全部权重来自 DAgger ckpt，`require_complete_checkpoint True`），LR 2e-5 余弦到 0.25×，500 iter，G=8，β 0.04，ε 0.2，train_10。结果 iter 100/200/300/400/450：66.18 / 65.58 / 66.29 / 65.47 / 65.58，均低于起点 66.72。与历史 +1.1 合看，GRPO 对该族模型效应约 0 ± 1。已停。

## 10b. 跨 episode 置换测试与坐标融合方案（patch 0024–0027，另一会话完成）

### 置换测试

- 历史 try5（LLM-Grid 2，iter28000）training-free 通道消融：Full 65.42 / No-object 65.80 / No-region 65.31 / Joint-zero 64.06 / Direction-zero 65.25 / All-three-zero 64.17。置零是训练中从未出现的输入，分不清"信息损失"与"分布外代价"。
- 新工具 `make_raster_donor_cache.py` + `eval_raster_donor.sh`：每个 episode 保留自己的元数据，只把栅格换成别的 episode 的 LLM 栅格（cross_scene，或 within_scene 且排除同 trajectory），派生为新的 cache key，评测零代码改动。
- 结构性事实：map encoder 无位置编码、mask 全真，fusion 是普通 cross-attention，因此策略对 10×10 块的置换严格不变（随机权重数值验证输出差 2e-7）；`gmap_pos_fts` 以当前位姿为参照，是自我中心坐标。
- 结果（同一 harness，val_unseen 1839）：control 65.09 / within_scene 65.20 / cross_scene 66.01；cross_scene 三个 donor seed 66.01 / 65.91 / 65.96，配对 +0.82～+0.92，CI [+0.11, +1.86]，增益主要在可达性。**该策略不使用栅格中任何 episode 或场景特异信息**；Joint-zero 掉的 1.4 是全零输入的分布外代价。

### 方案修正：坐标融合

- 重新判断：GT 教师的 map 模块同样由 DAgger 从初始化训出（387500 tokenizer 死、loader 未加载，教师 ckpt tokenizer 范数 17.85），却从栅格拿到 17 分，因此"DAgger 学不动栅格通路"不成立。教师与学生的差别是地图真假，以及节点能否定位到地图上：教师用 dz，地图就是 Habitat x/z 轴（`meters_to_grid(x, z)`），dz 恰给节点一个沿地图轴的分量；学生 dy 下节点特征完全自我中心。LLM 地图唯一不与指令重复的信息是布局，需要编码器保留布局、节点知道自己在图上的位置。
- `MODEL.MAP_ENCODER.coordinate_fusion`（默认关，仅 try5）：节点按与 `get_pos_fts` 相同的位置换算映射到栅格坐标（原点 = 起点世界 x,z − 缓存 start_position；`start_position_meters_per_unit` LLM/legacy 1.0、gt.online 0.5）；cross-attention 加固定逐 head 距离偏置（1/3 head 纯内容，其余 σ = 2.5/5/10/20 m），STOP/填充/元数据 token 不加偏置；map encoder 加固定 2D sin/cos 位置码。全为非持久 buffer，不增参数、不改 state_dict 键；单元测试 9/9 通过。
- `check_map_frame.py`：把参考路径按训练同一变换映射进栅格，统计落在有区域标注格子上的比例，与错误单位、x/z 互换对照。
- 计划：先对 gt.online121c369（0.5）与 LLM（1.0）跑 `check_map_frame`；通过后并行 E1（GT 在线地图，387500 起步，dy + 坐标融合，对照 dz 教师 73.84）与 E2（LLM 地图，460000 起步、map 模块从初始化，dy + 坐标融合，对照 control 65.09，并做 control vs cross_scene 配对）。E1 过而 E2 不过 → 改 LLM 地图为起点坐标系输出。启动器 `run_dagger_coord_fusion.sh {gt|llm}`（`COORD=False` 为同臂对照）；加载 checkpoint 时核对 `coordinate_fusion` 与训练一致，不一致直接报错。

## 11. 结论与未决问题

**已确定**

- 预训练 map encoder 因 HF 初始化行为而死，机制、范围、复现、修复齐备。
- 历史 DAgger 从未加载预训练 map 模块，因而未受影响，但也从未有过"预训练规模训好的栅格通路"。
- 用 LLM 栅格在 DAgger 里从头训栅格通路的价值 ≤ 2.6 SR，且无法利用 GT 栅格；R7 的优势来自元数据 token 而非栅格。历史 try5 策略不使用栅格中任何 episode/场景特异信息（置换测试）。
- "DAgger 学不动栅格通路"不成立：GT 教师同样在 DAgger 里从初始化训出栅格通路并拿到 17 分。差别在地图真假与节点—地图坐标对齐。
- refiner 线失败不是死 tokenizer 所致，而是策略对栅格依赖弱。
- 修复后的预训练得到活编码器（R16）。

**未决**

- R16-DAgger 的 control vs cross_scene 配对结果（进行中）。
- 坐标融合 E1/E2 的结果；`check_map_frame` 对两种缓存的坐标系核查。
- LLM 栅格本身的信息量是否足以支撑任何策略（教师喂 LLM 栅格 49.48；置换测试提示当前策略下为零）。
- R11′ 13k 消融、R11′ vs R7 配对 CI、`s4_try5_refiner` tokenizer 扫描（均为收尾，不改变结论）。

## 12. 附录：工具与命令

| 工具 | 用途 |
|---|---|
| `scripts/distill/map_encoder_weight_sweep.py --checkpoint-globs ...` | 跨 checkpoint 的 tokenizer 权重/偏置范数 |
| `scripts/distill/map_encoder_stage_probe.py --checkpoints ... [--device cpu]` | 逐级 rel(gt)/rel(zero)、‖W·x‖/‖b‖ |
| `scripts/distill/map_sensitivity_probe.py --run-name RUN --iter N` | DAgger ckpt 的 logits 对 LLM/GT/零/无地图的敏感度 |
| `scripts/distill/eval_student_ablation.sh RUN ITER {none,metadata_only,raster_only,no_direction}` | 评测时通道消融 |
| `scripts/distill/eval_student_gt_full.sh {p0,gt_full,gt_seen} ITER RUN` | 评测时换栅格源 |
| `scripts/distill/eval_iters.sh RUN [ITER ...]`（`GPUS=4,5,6,7`） | 多 ckpt 并行评测 + 汇总表 |
| `scripts/distill/paired_analysis.py --run-a/--iter-a --run-b/--iter-b` 或 `--dir-a/--dir-b` | 逐 episode 配对、场景聚类 bootstrap、McNemar |
| `scripts/distill/repro_hf_no_init_conv.py` | HF 初始化行为复现 |
| `scripts/distill/run_dagger_distill_llmpt.sh`（`REINIT_TOKENIZER`、`LOAD_MAP`） | 蒸馏 DAgger 启动器 |
| `scripts/distill/run_grpo_from_dagger.sh`（`DAGGER_RUN`、`DAGGER_ITER`、`PROFILE`） | GRPO 续训启动器 |
| `scripts/submit/llm-grid-try5-pretrain-fixedtok.sh`、`llm-grid-try5-fixedtok-dagger.sh` | 云端修复后预训练 / DAgger |
| `scripts/distill/make_raster_donor_cache.py`、`eval_raster_donor.sh`（patch 0024） | 跨 episode 栅格置换测试（cross_scene / within_scene） |
| `scripts/distill/check_map_frame.py`（patch 0025） | 离线核查节点→栅格坐标系（单位、x/z 轴） |
| `scripts/distill/run_dagger_coord_fusion.sh {gt\|llm}`（patch 0026，`COORD=False` 为对照） | 坐标融合 DAgger E1/E2 启动器 |

关键路径：预训练 checkpoint `pretrained/r2r_rxr_ce/llm_grid_try5/ckpts/model_step_460000.pt`（死）、`pretrained/r2r_rxr_ce/llm_grid_try5_fixedtok/ckpts/model_step_142500.pt`（活）；GT 教师 `try-5-r1p5-dagger.iter16000.pth`；补丁序列 `scripts/distill/patches/0001–0027`（分支 `claude/awesome-ride-axuad6`）；逐日记录 `docs/daily/2026-09-24 … 09-30.md`。
