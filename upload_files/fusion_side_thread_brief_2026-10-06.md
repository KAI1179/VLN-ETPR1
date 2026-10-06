# 旁支讨论简报：ETP-R1 / Try5 的地图融合（fusion）

用途：新会话讨论 fusion 相关问题时作为起点。更完整的背景见 `upload_files/handoff_2026-09-29_context.md`（环境、分支、补丁机制）与 `upload_files/raster_map_report_2026-09-30.md`（栅格通路全部实验）。逐日记录在 `docs/daily/2026-09-24 … 10-06.md`。本文只列与 fusion 直接相关的事实。

## 1. fusion 是什么、在哪

- 类：`GraphMapCrossAttention`（`vlnce_baselines/models/etp_prior_gt/map_fusion.py`），由 `build_map_token_fusion` 在 `vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py:855` 构造，挂在 `GlocalTextPathNavCMT.graph_map_attention`。
- 结构：图节点表征作 query，对 101 个 map token（100 个空间 token + 1 个元数据 token）做一次 `nn.MultiheadAttention` cross-attention，经 `residual_projection`（Linear，代码里零初始化）残差加回节点表征。单向：地图不反向读图节点。共 6 个张量（MHA 的 in_proj/out_proj 权重与偏置、residual_projection 权重与偏置）。
- 输入侧 map token 来自 `EmbeddingGridMapEncoder`（`map_encoder.py`）：category_projection（CLIP 初始化）→ spatial_tokenizer（10×10 卷积）→ LayerNorm → 元数据 MLP → 2 层 transformer（无位置编码）→ output_norm。mask 全真，100 个空间 token 对 fusion 来说是无序集合（对 10×10 块置换严格不变，数值验证输出差 2e-7）。
- 节点侧位置特征 `gmap_pos_fts`（`graph_utils.get_pos_fts`）是以当前位姿为参照的自我中心坐标（heading、elevation、距离），与地图的世界系（层内 x/z）之间没有显式对齐。

## 2. 权重从哪来

| 阶段 | fusion 权重 | 备注 |
|---|---|---|
| 预训练 checkpoint | 存在 `bert.global_encoder.graph_map_attention.*` | 09-24 之前导航 loader 从不加载它（键名不匹配被当 unused 丢弃） |
| 历史所有 DAgger（含 65.42 的 try5、S4 refiner） | DAgger 从初始化训 | residual_projection 实测范数约 15.3，相当于 N(0,0.02)，说明 HF `from_pretrained` 跳过了零初始化 |
| R7 / R11′（patch 0001/0006 之后，`load_pretrained_map_modules True`） | 加载 460000 的 6 个张量 | 460000 的 fusion 是和**死栅格**（100 个常数 token）一起预训练的，学到的是只取元数据 token |
| R16-DAgger | 加载修复后预训练 142500 的 fusion | 第一次与活栅格一起预训练的 fusion |

加载后用 `_verify_fusion_transfer` 校验张量确已落位（`vlnbert_init.py`）。

## 3. 与 fusion 相关的数据

- GRPO `trainable_profile`：`nav4` 冻结 fusion；`nav4-fusion` 额外训 fusion 的 6 个张量，历史上 65.42 对 66.50 输给 nav4，已退役。GRPO 训练器在 nav4-fusion 下校验 fusion 梯度非零、非有限则报错，并记录 `fusion_grad_norm`。
- 教师（GT 地图、dz 位置特征）：none 73.84 / raster_only 69.33 / metadata_only 56.93 / nomap 57.04。fusion 在教师里承载约 17 分。
- LLM 学生：R5（fusion 从零训）栅格 +2.6、元数据 0；R7（加载死栅格 fusion）栅格 0、元数据 +1.47；R16-DAgger 20400 置换测试 cross_scene 三 seed 均值 −0.60（历史 try5 为 +0.9）。
- 教师喂 LLM 栅格只有 49.48，低于 nomap：fusion 对栅格内容是"信任"的，内容错了会被误导。
- 教师的 dz 位置特征（Habitat 的 dz，`base_elevation` 恒为 0）实际是世界系南北分量，恰好给节点一个沿地图轴的坐标，被解释为"罗盘"；学生用 dy 则节点特征完全自我中心。这是另一会话提出"坐标融合"方案的出发点。

## 4. 已有的 fusion 改动方案

- 坐标融合（patch 0025/0026，`MODEL.MAP_ENCODER.coordinate_fusion`，默认关）：节点按与 `get_pos_fts` 相同的换算映射到栅格坐标；cross-attention 加逐 head 固定距离偏置（1/3 head 纯内容，其余 σ = 2.5/5/10/20 m）；map encoder 加固定 2D sin/cos 位置码；全为非持久 buffer，不增参数、不改 state_dict 键。计划 E1（GT 在线地图，387500 起步）与 E2（LLM 地图，460000 起步）。
- `check_map_frame.py`：离线核查节点→栅格坐标系（单位、x/z 轴）。

## 5. 可讨论的开放问题

- 单向 cross-attention 是否够：地图 token 不读图节点，节点对地图的"注意"只能基于内容相似度；要不要双向或在 map encoder 侧注入节点位置。
- residual_projection 的实际幅度与贡献：把它置零是否等价于 nomap（可用现有消融框架做 training-free 测）。
- fusion 预训练目标：MLM+SAP 下 fusion 收到的梯度是否足以学会读栅格；是否需要地图相关的辅助目标。
- 位置编码缺失：100 个空间 token 无序，布局信息只能通过内容间接表达；坐标融合是否足够，还是需要显式 2D 位置码加到 token 上。
- 教师 dz "罗盘"效应与 fusion 的关系：教师依赖栅格的 17 分里有多少来自节点能定位到地图上。

## 6. 新会话的约定

- 分支 `claude/awesome-ride-axuad6` 上同时有多个会话在提交补丁，**补丁编号已出现冲突**（0030、0032 各有两份）。新会话请用独立编号段（建议从 0050 起）或子目录 `scripts/distill/patches/fusion/`，并在提交前 `git pull --rebase`。
- 代码改动仍以 exp/refiner 为基础做补丁；`docs/daily/` 用中文，新开小节注明会话主题。
- 不启动、不中止任何训练；实验由用户决定并执行。
