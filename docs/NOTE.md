# 基于指令的先验认知地图 Instruction-based prior

## 结果对比

### 地图编码器 GT

| Method                     | SR         | OSR        | SPL        | ckpt    |
| -------------------------- | ---------- | ---------- | ---------- | ------- |
| Baseline (Dagger)          | 0.6313     | 0.6852     | 0.5423     |         |
| Baseline (GRPO)            | 0.6536     | 0.7151     | 0.5582     |         |
| Baseline (Partial, dagger) | 0.5568     | 0.6253     | 0.4728     |         |
| Ours (FT only, GRPO)       | 0.6487     | 0.7151     | 0.5519     |         |
| Try 3 (Dagger)             | 0.5905     | 0.6542     | 0.4985     |         |
| Try 3 (Partial, dagger)    | 0.5525     | 0.6183     | 0.4667     |         |
| Try 4 (Dagger)             | 0.6444     | 0.7172     | 0.5302     | 27800   |
| Try 4 (GRPO)               | 0.6455     | 0.7080     | 0.5542     | 270     |
| Try 5 (VLNCE, Dagger)      | 0.7304     | 0.7602     | 0.6498     | 27800   |
| Try 5 (VLNCE, GRPO)        | 0.7401     | 0.7809     | 0.6506     | 350     |
| Try 5 (Dagger)             | 0.7265     | 0.7591     | 0.6305     | 27400   |
| Try 5 (GRPO)               | 0.7417     | 0.7830     | 0.6304     | 420     |
| Try 6 (Dagger)             | 0.6449     | 0.6835     | 0.5609     | 27800   |
| Try 7 (Dagger)             | 0.6520     | 0.7230     | 0.5450     | 29600   |
| Try 8 (Dagger)             | 0.6400     | 0.7025     | 0.5382     | 26800   |
| Try 9 (Dagger)             | 0.6378     | 0.6965     | 0.5394     | 28800   |
| Try 10 (Dagger)            |            |            |            |         |
| Try 5 (Dagger, r=1.5)      | **0.7656** | **0.7896** | **0.6682** | 25600   |
| Try 5 Like 2x (Dagger, r=1.5) | 0.6917  | 0.7221     | 0.5988     | 29800   |

- Try 4 (reuse+full, new-model-full)
    - [新的地图编码器结构](https://github.com/PRO-2684/ETP-R1/commit/8b7c5763f06156581b59d871e2afaaf4131aad3a) V2 (ResNet-like)
    - [新增 `direction_vectors` & `start_position`](https://github.com/PRO-2684/ETP-R1/commit/c445ac01a7ee5256f168dafbcc6d929027e30524)
    - 复用作者权重
    - 完整数据
- Try 5 (try5)
    - 新的地图编码器结构 V3 (参考 Imagine Before Go)
        - Starting from `7de636b53f02a8d96e2457f03742ea5370527944`
    - 新增 [`start_direction_vector`](https://github.com/PRO-2684/ETP-R1/commit/e8a4d1210b7b815104dfd66dd25c2f12183c296c)
    - VLNCE variant: 预训练只使用 VLNCE 标注数据，未使用作者的增广数据
        - (new-vlnce-only, try-5-vlnce)
    - 实现有问题：没有单词向量相似度匹配，于 [`8189627`](https://github.com/PRO-2684/ETP-R1/commit/81896270673eb6ccd67ea375c97375d9878b7863) 修复
    - **Lesson: 多参考已有工作**
- Try 6 (try6, [`705dd7b`](https://github.com/PRO-2684/ETP-R1/commit/705dd7bd5977238d4b7da02c35a6235189932ec5))
    - 基于 bounding box 的 gt 认知地图
    - 半径：2.5m -> 1.5m
    - Waypoint：GT (稠密) -> Reference (稀疏)
    - *数据增强：随机旋转 (90° 整)*
    - `direction_vectors` -> `reference_path`
- Try 7 (try7, `d7f681b`, @超算)
    - 重新使用 GT 路径
    - 如果第一个遇到的楼层路径点不足，使用其它楼层
- Try 8 (`2b3d97d`, @VIPL&超算)
    - 禁用随机旋转
    - 加入拓扑地图 -> 认知地图的 attn
- Try 9 (`9caeb61`, @VIPL)
    - 简易的认知地图 Decoder
- Try 10 (`0a31d47`, @VIPL)
    - DETR-style Decoder
- Try 5, r=1.5 (`121c369`)
- Try 5 Like, 2x blurred (`6ce1e4a`)
    - 模型：带双向 attn 和 DETR-style Decoder
    - GT: legacy, direction-vectors, 1.5m

### 基于 LLM 的预测器

| Method | Commit | E-valid | IoU | Cat Precision | Cat-F1 |
| - | - | - | - | - | - |
| LLM 1       | `1e8b031` | 99.98%  | 0.121% | 1.28%  | 0.34%  |
| LLM 2       | `4ec2b81` | 99.42%  | 1.42%  | 67.53% | 56.26% |
| LLM 3       | `92d2476` | 94.17%  | 2.22%  | 26.18% | 33.90% |
| LLM 4       | `4e9a456` | 94.22%  | 1.31%  | 18.72% | 23.43% |
| LLM 5       | `42c16c8` | 95.24%  | 1.76%  | 23.06% | 29.42% |
| LLM-Grid 1  | `6a2623b` | -       | -      | -      | -      |
| LLM-Boxes 1 | `d7a0205` | -       | -      | -      | -      |
| LLM-Grid 2  | `ad1049f` | -       | -      | -      | -      |

- LLM 1: Tell2Design-style
    - e.g. `[ object appliances | center x = 14.0 | center z = 26.9 | half x = 1.2 | half z = 0.3 | rotation = -2.55 ]`
    - 问题：过长导致截断
- LLM 2: 更简短的输出格式
    - e.g. `obj door 25.1 29.0 1.1 0.2 -2.87`, `reg circulation 26.5 22.2 35.0 37.6`
    - 仅考虑 mentioned
- LLM 3: Llama-based
    - rank = 8
- LLM 4: Use GT path
    - rank = 32
    - 如果第一个遇到的楼层路径点不足，使用其它楼层
    - 随机旋转
    - -> 生成的预训练 navigation cache 仅 8621/109507
- LLM 5: 禁用随机旋转
    - -> ?
- LLM-Grid 1: 逐网格预测，JSON 格式
- LLM-Boxes 1: 逐 bbox 预测，JSON 格式
    - Train: 28,713 / 11,802 (70.9%)
    - R2R val-unseen: 2,089 / 260 (88.9%)
    - RxR val-unseen, English: 2,932 / 1,620 (64.4%)
    - Total: 33,734 / 13,682 (71.1%)
- LLM-Grid 2: 分布式训练，加上 RxR 数据

### 基于 LLM 的 pipeline

- Smoke: LLM-derived cognitive-map cache + Try 7 navigation checkpoint
    - Script: `scripts/tries/smoke-llm-map-try7.sh`
    - Matrix: LLM4 cache, LLM5 cache
    - Checkpoint: `data/logs/checkpoints/release_r2r_priorgt_dagger/store/try7.iter29600.pth`
    - R2R evaluation only for the first smoke pass
    - Shared VLN-CE episode iteration filters to English instructions
    - Training skips missing LLM-Navigation `.npz` cache entries
    - Evaluation treats a missing `.npz` as an LLM generation failure:
        - keep the episode in the denominator
        - assign zero navigation metrics
        - report `llm_cache_missing_count` / `llm_cache_missing_rate`
    - SR 46.43%; Missing cache 26.48%; SR in entries with cache 63.17%
- Nav 1: Try 8 + LLM 5
    - 未完整运行 - 检查点丢失
- Nav 2: Try 9 + LLM 5
    - 未完整运行 - 中断，运行 try5-r1p5
- LLM-Grid 1，Try 5 架构
- LLM-Boxes 1，Try 10 架构

# 讨论与结果

## 04/30

### VLN 训练范式

- [x] 认知地图网格数据形式
    - [x] 网格
        - 直接给模型 (计算量？)
        - 只取最大置信度的
        - 加权：需要检查 embedding 加权结果对应的词汇 (`Conv2d(37, 512, kernel_size=1)`)
    - 起始位置
    - 方向向量
- [x] map encoder 模型结构？(寻找已有类 pointnet 的 backbone)
    - CLIP embedding
    - ResNet-Like
- [x] 训练
    1. 利用已有权重
    2. 预训练 map encoder
        - 部分数据 (R2R, RxR)
        - 全部数据 (+annotation)
    3. Dagger
    4. GRPO

TODO

- [x] 修改地图编码器的模型
- [x] 生成带有起始位置和方向向量的认知地图
- [x] 将起始位置和方向向量接入 pipeline
- [x] 从作者的权重开始预训练
- [x] 微调

### 认知地图生成器的训练 (指令 -> 地图)

- 结构
    - **多模态大模型微调**
    - 世界模型的训练 (大模型常识的利用？)

### Online 双地图更新

视觉观察的物体映射至网格，加入认知地图/删除错误部分

### 认知地图轨迹半径

预测置信度网格->生成 embedding 网格

## 05/06

- [x] 传入朝向
- [ ] paper 1 ([Imagine Before Go](https://github.com/sx-zhang/SGM)) 参考 pipeline 结构
    - [ ] tokenize (encoder decoder)
    - [ ] LLM 常识注入与融合
    - [x] map encoder 参考：文本 RoBERTa 编码，图像通过 ViT 多个 Transformer block 编码 (只编码可见部分)
- [x] paper 2 ([OccWorld](https://github.com/wzzheng/OccWorld)) 参考：
    - (delta x, delta y) -> 方向向量
    - 衡量是否去除时间融合 self-attention
    - 模型结构对保留同一场景其它指令的认知地图是否有影响？(有利于持续进化)
- [x] 生成认知地图：传入所有信息
    - 认知地图
    - ...
- [ ] 是否可以收敛？无法收敛则考虑修改损失函数。

## 05/08

- [x] * 原文 grpo 为什么只更新两个模块？grpo 是否需要冻结地图编码器？
    - GRPO_trainer_ETP_R1.py
    - `def setup_training_parts(self):`
        - 冻结 `self.policy`
    - `self.trainable_parts`
        - vln_bert_module.global_encoder
        - vln_bert_module.graph_query_text
        - vln_bert_module.graph_attentioned_txt_embeds_transform
        - vln_bert_module.global_sap_head
    - 目前 map encoder 在 GRPO 已经是冻结的 - 作者先冻结所有权重，再启用上面的模块
- [x] 涉及多楼层的数据比例？(先仅统计 R2R, RxR)
- 若多楼层，方向向量加一维？

### 多楼层调研

| Dataset                         | Entries | Up/down stairs  | Avg. positions |
| ------------------------------- | ------- | --------------- | -------------- |
| VLNCE-R2R                       | 13436   | 1024 (7.62%)    | 38.31          |
| VLNCE-RxR                       | 51139   | 6643 (12.99%)   | 68.32          |
| ETP-R1-R2R_Prevalent            | 1069620 | 131658 (12.31%) | 6.06           |
| ETP-R1-R2R_Prevalent_gemini_aug | 1046280 | 122647 (11.72%) | 6.06           |
| ETP-R1-R2R_train                | 14039   | 2672 (19.03%)   | 6.00           |
| ETP-R1-R2R_val_unseen           | 2349    | 516 (21.97%)    | 5.97           |
| ETP-R1-rxr_marky                | 1001331 | 244833 (24.45%) | 10.50          |
| ETP-R1-rxr_train_guide          | 79467   | 17681 (22.25%)  | 9.08           |
| ETP-R1-rxr_val_unseen_guide     | 13652   | 3474 (25.45%)   | 8.52           |

### 微调阶段可训练/冻结的模块

#### Author ETP-R1

In DAgger (SS-ETP-R1), almost the whole policy is trainable. The trainer freezes only the waypoint predictor explicitly, then builds the optimizer
over self.policy.named_parameters() vlnce_baselines/ss_trainer_ETP_R1.py:224 vlnce_baselines/ss_trainer_ETP_R1.py:237. Inside the model, RGB CLIP is
frozen, depth encoder is frozen by default, and fix_lang_embedding / fix_pano_embedding can freeze VLN-BERT language or pano embeddings, but in
run_r2r/iter_train.yaml both are False, so they train.

In GRPO (GRPO-R1), the trainer first freezes the whole policy, then unfreezes only these VLN-BERT navigation modules vlnce_baselines/
GRPO_trainer_ETP_R1.py:229:

- vln_bert.global_encoder
- vln_bert.graph_query_text
- vln_bert.graph_attentioned_txt_embeds_transform
- vln_bert.global_sap_head

Everything else stays frozen: language encoder, panorama/image embedding path, RGB/depth encoders, waypoint predictor, etc. The optimizer is built
only from requires_grad=True params vlnce_baselines/GRPO_trainer_ETP_R1.py:319.

#### Our ETP-PriorGT

In DAgger (SS-ETP-PriorGT) with normal config, it trains the same broad policy surface as ETP-R1 DAgger, plus the new map_encoder. The optimizer
filters to p.requires_grad, and map_encoder params are trainable by default vlnce_baselines/ss_trainer_ETP_PriorGT.py:244. If
MODEL.MAP_ENCODER.freeze_base=True, PriorGT DAgger freezes every policy parameter whose name does not include map_encoder, so only the map encoder
trains vlnce_baselines/ss_trainer_ETP_PriorGT.py:232.

In GRPO (GRPO-ETP-PriorGT), current code follows the author GRPO setup: it freezes the whole policy, then unfreezes only the same four VLN-BERT
navigation modules vlnce_baselines/GRPO_trainer_ETP_PriorGT.py:219 vlnce_baselines/GRPO_trainer_ETP_PriorGT.py:286. That means our map_encoder is not
trained during normal PriorGT GRPO. It is used to produce map_embeds, but its weights stay fixed from the loaded DAgger checkpoint.

One important caveat: GRPO-ETP-PriorGT has a freeze_base probe block, but it runs after the optimizer was already built and after
setup_training_parts() froze the map encoder. As written, freeze_base=True in GRPO does not actually make map_encoder trainable. The DAgger probe path
is correct; the GRPO probe path would need a small fix if we want “train map encoder only” during GRPO too.

## 05/13

- ~~统计涉及楼层数量分布~~ / 统计各场景楼层数量
    1. 29
    2. 34
    3. 18
    4. 7
    5. 1
    6. 1
- 认知地图 3 个楼层（最底下） (upd 05/16)
    - **参考现有工作的多楼层处理**
    - 起始楼层开始
    - 空间顺序排列
    - 单层：一层为空？上/下？
    - 层数 F=3 的处理？B x F x N x 100 x 100
        - **放到 embedding 里 (B x FN x 100 x 100)**?
            - 可能的问题：多层被压到一层里了 (FD -> D)，无法确认出现的物体在哪一层
        - 层数放到 Batch Size 里？(BF x N x 100 x 100)

## 认知地图生成

- predicts grid, but metadata?
    - zero

### Option 1: Simple Direct Decoder

Fastest to implement. Architecture:

VLN text embeddings or CLIP text embedding + learned 10x10 query tokens + metadata query
-> 2-4 Transformer decoder layers
-> upsample CNN head
-> logits (37,100,100)

Train against PrecomputedCognitiveMap.grid. Loss:

- BCEWithLogitsLoss if grid is multi-hot
- CrossEntropyLoss if one category per cell
- weighted/focal loss for rare objects
- optional Dice/IoU loss for spatial quality

Then:

```python
pred_grid = torch.sigmoid(logits)
map_tokens, map_masks = map_encoder(
    pred_grid,
    direction_vectors,
    start_direction_vectors,
    start_positions,
)
```

### Option 2: Better, OccWorld-Style Latent Map Prior

Use OccWorld idea: VQ-VAE compresses maps, transformer predicts latent codes.

Stage 1:

grid (37,100,100)
-> VQ-VAE encoder
-> discrete latent grid, e.g. (10,10)
-> VQ-VAE decoder reconstructs grid

Stage 2:

instruction tokens
-> transformer
-> predict VQ code IDs for 10x10 latent map
-> VQ decoder
-> predicted grid

Why better:

- avoids blurry map logits
- predicts structured layouts
- cheap autoregressive or masked-token training
- aligns well with our 10x10 map-token encoder

This mirrors OccWorld-main/model/TransVQVAE.py: encode occupancy to VQ codes, transformer predicts future codes, decoder reconstructs occupancy.

### Option 3: Best If Partial Map Exists

> “持续进化/学习”

If agent has observed partial map, use SGM-style masked reconstruction:

partial semantic map + mask
-> ViT/MAE map encoder
instruction/category text tokens
-> cross-attention
-> reconstruct unknown map cells

This matches SGM-main/SGM_train/models_mae_cross.py:

- ViT patch tokens from semantic map
- RoBERTa/category context
- Cross_model cross-attention
- decoder reconstructs masked map patches

For VLN this is stronger than instruction-only because partial observation anchors layout.

## 05/16

- ~~训练时 GT 路径添加随机偏移？？~~
    - `start_position`?
- [x] **生成时输入 metadata** (方向、位置)
- [x] 生成认知地图：考虑最先经过的一层，而非经过的最下面一层
- 循环式生成认知地图：借鉴 Imagine before go
    - 生成认知地图的代码搬过来 (`prior`)，实现在线生成？
    - 两个地图每格取 max？楼层？
- 新的认知地图生成流程
    - [x] 建模 90 个场景的 GT 语义地图 (0-1 格式)
        - [ ] 预生成
        - [x] **on the fly**
    - [x] 根据轨迹+阈值取出局部语义地图 (0-1 格式) (**训练过程中**)
        - 方便调阈值
        - 若空，取底层 `gt_maps[0]`
    - [x] 高斯模糊局部语义地图，生成认知地图 (**训练过程中**)
        - 先 max 后高斯 vs 先高斯后 max
    - [ ] 输出格式 F x N x 100 x 100（参考语义分割 decoder?）

## 05/20

- 流程
    - 训练认知地图生成器
    - **使用认知地图生成器的结果训练 GT 模型 (预训练开始)**
    - 测试
- 指标
    - [x] **可视化**？
    - 方向向量的指标？
    - top5pct_recall 90% 如何计算？
- 生成
    - LLM 常识注入？
    - 微调大模型？
        - 输出：
    - 问题
        - 绝对位置
        - 稀疏
- 路线
    - 保留当前 GT 结构，处理稀疏问题
    - 修改 GT 结构，从根源避免稀疏问题
- **损失函数**
    - 目标：物体相对位置、方向
    - 参考语义分割的 loss 方法

## 05/21

- 给模型输入位置取整？
- Sample 10; npz -> json (砍掉模糊，polygon); LLM prompt & generate
- pipeline
    - [x] **预计算所有认知地图**?
    - [ ] grid -> json (polygon-based?)
- 预测 direction vectors -> 预测关键的 waypoint? 维度不用改。
    - [ ] prompt engineering
    - [ ] *较大转向的 waypoints，角度 ~36.7*
- 如何评价相对位置的正确性？（绝对位置无所谓）量化指标？
- Q:
    - ~~预计算所有认知地图~~ -> 预计算语义地图，动态生成认知地图
        - 生成确实快
            - 2-3min for all r2r/rxr, 202M
            - ??? for annotations, ???G
        - 修改简单
    - `reference_path` or `locations`?
        - `reference_path`: Only key waypoints
        - `locations`: Dense, step-level, gt
    - RxR 不使用 follower?
        - 缺 `instruction_tokens`
        - 语义不同?
    - 支持所有 `language`？
    - 物体也使用 aabb？

## 05/28

- LLM 微调
    - 8B 4090 ~2 天
    - 损失函数：token-level 交叉熵，和 gt json 比较
        - 参考 [aclanthology.org/2023.acl-long.820.pdf](https://aclanthology.org/2023.acl-long.820.pdf#page=6.86) 代码、参数与环境
    - 确定 prompt 模板？
- map predictor: 候选
- 参考
    - [LengSicong/Tell2Design](https://github.com/LengSicong/Tell2Design) | [paper](https://aclanthology.org/2023.acl-long.820.pdf)：基于 T5，seq-to-seq，没有 prompt
        - treat the instructions as the input sequence and consider bounding boxes of rooms as the target sequence
    - [HOUSETUNE paper](https://arxiv.org/pdf/2411.12279)：
        - no code
    - [ChatHouseDiffusion paper](https://arxiv.org/pdf/2410.11908)：
        - no code


- Questions
    - 两阶段
        - LLM-Boxes: train/evaluate T5 JSON generation for object/region boxes and ref path
        - LLM-Navigation: integrate into nav pipeline
    - T5 输入是否包含 dataset flag (r2r / rxr)? 包含。
    - LLM 训练时随机旋转：先不添加，能 work 后再添加。
    - **仅考虑 mentioned???** (与 GT 实验不同)
-

## 06/05

- [x] `reference_path` (x) -> gt path 计算出转折点/key waypoint/bbox
- 半径：查看当前 LLM 微调结果
- eval：部分数据
- 提升 rank：32/64？
- 第二阶段预训练不用增广数据
- 数据增强
    - GT：验证上限，无需增强
    - 训练：需要数据增强，学习方向无关性
- Questions
    - trajectory_keypoints 包含最后一个位置
    - trajectory_keypoint_mask
    - 地图编码器的 start_position - 预测路径还是 episode?
    - 预训练数据集的 path 是稀疏的，不是 ground_truth_trajectory
    - Rank-Stabilized LoRA？
    - 第一个遇到的楼层 ground_truth_trajectory 路径点个数 < 1
        - If another level has >=2 dense GT points: use that level.
        - If every level has <2 points: generation warns/skips that entry.
        - Training/direct cache loading can still fail unless explicitly in skip-generation mode.
        - -> generated=91479 skipped=9 for VLNCE
        - -> generated=3226159 skipped=579 for pretrain

## 06/11

- 评价指标？
- [x] **可视化 sample**
- 生成的地图 <-> 视觉构建的地图
- **确认当前只有一个认知地图解码器，是否可以说成视觉地图和认知地图有交互；没有的话加交互？**
    - step 0: 大模型预测的地图 0 -> action, 预测的认知地图 1
    - step 1: 预测的认知地图 1 -> action, 预测的认知地图 2
    - decoder 损失 参考现有工作（网格 vs 网格的稀疏问题；）
- Claims:
    - 🟢 feature-level interaction/fusion
        - We augment ETP-R1’s online topological map with an external cognitive map by encoding the cognitive map as spatial-semantic tokens and letting topological graph nodes attend to those tokens. This enables feature-level interaction between the original topological planning representation and our semantic cognitive-map prior.
    - 🔴 interactively update each other or co-evolve

## 06/22

- LLM 生成的 nav cache 合法的数量少
    - 预训练 8621/109507
    - 可能原因：**语言不是英语**
    - [x] 剔除非英语数据
- fail?
    - fallback 到原模型？
    - [x] 认为 LLM generation failure，记录 `llm_cache_missing_count/rate`
- 认知地图/拓扑地图交互
    - 之前：已有单向特征融合 (GraphMapCrossAttention)，可以表述为 A + B => A + B*
    - Implemented (try 8)：拓扑地图 -> 认知地图的 attn（双向 attn）
        - A: 认知地图 embedding；B: 拓扑地图 embedding
        - A* = A + map_residual(A <- B)
            - `map_from_graph_attention` + `map_residual_projection`
        - B* = B + graph_residual(B <- A*)
            - `graph_from_map_attention` + `graph_residual_projection`
        - 后续流程使用 B*
- 工作
    1. 认知地图训练的并行实验
        - LLM 微调结果 (llm 4/5?) -> try 7 dagger, 直接测试
    2. 加入认知地图的更新
        - 更新后的认知地图 embedding A* 接入解码器，解码成更新后的认知地图，用 GT loss 约束（每一步都约束并更新）
    3. 视觉拓扑图的前瞻更新
        - 确认拓扑图构造方法：每一步更新？
            - Imagine Before Go：网格地图的前瞻
            - 迁移到当前工作：拓扑图的前瞻
            - 构造方法：在线更新
            - 存储细节
                - `node_pos[vp]`: 3D simulator position (x, y, z)
                - `node_embeds[vp]`: panoramic embedding for that visited location, computed from avg_pano_embeds[i]
                - `node_stepId[vp]`: rollout step when the node was inserted
                - `node_stop_scores[vp]`: model’s STOP probability when at that node
                - `graph_nx`: NetworkX graph with weighted edges
                - `shortest_path`, `shortest_dist`: all-pairs shortest path/distance caches
        - 确认拓扑图是否有前瞻（预测，例如 Imagine Before Go）？没有前瞻则考虑加前瞻
            - 局部的单步前瞻/预测；ghost node
            - `vlnce_baselines/models/R1Policy.py:187`, `vlnce_baselines/models/graph_utils.py:69`
            - From the current panoramic RGB/depth observation, it predicts candidate waypoint directions/distances.
            - Internally this is a heatmap over 120 angle bins x 12 distance bins.
            - NMS keeps up to 5 candidate waypoints.
            - Each candidate becomes a possible ghost/frontier node with an estimated 3D position from current pose + predicted angle/distance.

## 06/23

- 认知地图 Decoder 路径
    - [Masked-attention Mask Transformer for Universal Image Segmentation](http://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html): 直接生成 mask
        - 重点：Transformer Decoder
        - feature 作为 k, v, N queries 解码出来 N 个特征 token，认为是 N 个 mask (可以为空)
        - MLP 向右预测 mask 类别，向下 mask embedding
        - pixel-level 特征和 mask embedding 相乘，得 N x H x W mask 形状
        - 丢弃空集
    - [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://proceedings.neurips.cc/paper/2021/hash/950a4152c2b4aa3ad78bdd6b366cc179-Abstract.html): mask query 分割
        - feature 作为 k, v, N queries 解码出来 N 个特征 token，认为各对应一个候选 mask
        - 实际输出是类别 logits 和二值 mask logits，不预测 bbox
        - Hungarian matching 基于类别、focal mask loss、dice mask loss
        - 若迁移到认知地图，需要 per-entity raster mask 作为监督目标
    - ⭐ [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

## 06/24

- [x] Cache failure rate?
    - Episodes that fail pre-generation item construction
    - 更新输出结构以方便诊断
- [x] Nav 1: Try 8 (加 decoder 之前) 使用 LLM5 Nav Cache 训练并验证
- [x] 添加认知地图 Decoder

决策：

- MaskFormer 论文不是基于 bbox 计算损失，而是基于 mask query 和匹配后的 mask loss。
- 真正的 MaskFormer-style 监督需要 per-entity raster masks；当前认知地图只有 category-first dense grid，没有保存不同 entity 的独立 mask。
- per-entity raster masks 会改变认知地图存储格式，先记录为后续方向。
- 当前先实现简单替代方案：从 updated map tokens 解码完整 updated cognitive map logits，使用 dense BCE loss 对齐现有 `(37, 100, 100)` 认知地图。

## 06/26

- Decoder 可能是三条路：
    - 改变认知地图存储格式，重跑出来，用 DETR
    - 想办法生成 entity mask，用 MaskFormer
    - 保留认知地图存储形式，换成语义分割范式，例如 Segmenter
- 当前流程更像指称分割 (Referring Expression Segmentation)，解码的时候语言指令要不要送入到 Decoder?
    - Referring Expression Segmentation (RES) is a vision-language task: given an image and a natural-language phrase such as “the man in the red shirt” or “the dog under the table,” the model must output a pixel-level mask for the referred object.
    - A typical RES model has three parts: an image encoder, a text encoder, and a fusion/segmentation decoder that combines visual and linguistic features to produce a binary mask. Recent systems increasingly use transformer vision-language backbones and sometimes large multimodal language models. For example, LISA extends this idea to “reasoning segmentation,” where the query can be implicit and require world knowledge.

## 07/03

- Nav cache failure 常见的失败原因
    - unknown region category: 'structure';
    - unknown region category: 'void';
    - unknown region category: 'free-space';
    - unknown object category: 'free-space 2.1 2.0 1.9 1.4 4.1'
    - missing_trajectory_keypoints: refusing to generate cache without trajectory keypoints
- 其它观察到的问题
    - **过度重复**
        - `data/llm_navigation/llama-3.1-8b-instruct/pretrain/mixed/predictions/5q7pvUzZiYa/2675_0.txt`
        - `data/llm_navigation/llama-3.1-8b-instruct/pretrain/mixed/predictions/ur6pFq6Qu1A/827_0.txt`
- [x] 确认 try7 测试是否有旋转，开了的话重新测试 - 测试没有旋转
- Try 5 变量
    - 半径 2.5 -> 1.5
    - path-mask -> bounding box
    - `direction_vectors` -> `reference_path`?
    - 如果第一个遇到的楼层路径点不足，使用其它楼层的 fallback?
- 先保持模型结构，不要加 attn 和 decoder
- 8 卡比 4 卡慢？
- 实验：GT + LLM 训练 + Nav
- 并行实验与模型结构改进
- GT 为空：跳过？
- 实验
    1. 去掉“拓扑地图 -> 认知地图的 attn”，仅 GT+VLN
    2. ~~GT 认知地图生成：Try 5（2.5m 半径+按路径切）~~ -> on-the-fly
    3. ~~GT 认知地图生成：Try 5（1.5m 半径+按路径切）~~ -> on-the-fly
    4. GT 认知地图生成：bbox, 2.5m 半径

## 07/08

### 改进 LLM 预测？

> https://chatgpt.com/g/g-6a3dee67d7d88191b59bb4ccb82063b7-paper-chat/c/6a48f3a4-1f58-83ec-bc7f-32748234dfe7

- 部分数据期望输出过长导致截断，没有终止符，可能会让模型倾向于不输出终止符，从而导致重复
- 查看期望输出中是否有重复
- 施加重复惩罚
- 对比实验
    - rank:           8, 16, 32
    - target modules: q/v vs. q/k/v/o
    - format:         free text vs. navigation DSL
    - decoding:       unconstrained vs. grammar-constrained
    - termination:    normal EOS vs. EOS + record-repeat stop
- Reasoning/CoT?
- 参考文献
    - [Dense Coordinate-List Fine-Tuning Induces a Controllable Interference Surface in Vision-Language Models](https://arxiv.org/html/2606.14507v1): 微调 LLM 输出结构化信息，缓解输出尾部重复
    - [NaviLLM: Towards Learning a Generalist Model for Embodied Navigation](https://arxiv.org/html/2312.02010v3): VLN 任务中微调 LLM 输出结构化信息
    - [Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks](https://arxiv.org/html/2412.06224v2): 有限长度的输出可能更容易学习 (3–6 actions / waypoints)
    - [NavGPT](https://ar5iv.labs.arxiv.org/html/2305.16986) / [NavGPT-2](https://arxiv.org/abs/2407.12366): Reasoning
    - [UIE: Unified Structure Generation for Universal Information Extraction](https://arxiv.org/abs/2203.12277): 统一的结构化输出框架
    - [GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction](https://arxiv.org/abs/2310.03668): Prompt 中包含清楚的定义/规则很重要（角度、坐标、STOP...）
    - [PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models](https://arxiv.org/abs/2109.05093): 渐进式解析输出，拒绝让输出无效的 token (SQL)
    - [Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning](https://arxiv.org/abs/2305.13971): 约束输出的格式，格式可根据输入的不同动态调整（输入依赖型语法）
- 想法：让预测结果与 try5 类型的认知地图相匹配
    - 预测 bbox 后根据路径点切出来（仅保留路径点附近格子）
    - 问题：要不要在关键路径点之间采样模拟连续轨迹

### 讨论

- keypoints 不输出？改提示词？
- 考虑调大 rank，影响不会特别大
- 查看 GT 是否有终止符
- Try5 GT 让 LLM 预测网格物体种类+占用位置，顺序：
    1. 找 20 个 GT 网格（注意别同一个场景+不同指令，跨场景比较好），不压缩直接生成 GT 文本，看长度和琐碎程度（比如一般有几个物体，平均每个物体有多少网格、整个GT文本Tokenizer‌之后长度）
    2. 2 倍压缩之后，还是查看上述信息，然后决定要不要继续压缩
    3. 然后进行非压缩 or 2x 压缩的 LLM 微调实验 和 VLN+GT 上限实验（其中非压缩的 VLN+GT 实验已经做过了）

### 跟进 `a467f7e`

- 格式：`g 4 18 38 0.6` (grid category axis1 axis2 <confidence>)
- 分析结果
    - Scale 1 output: outputs/llm_grid_samples/20260708-053628-818288-gt.legacy.r1p5.direction5.v1-n20-s1
        - mean target tokens: 6029.4
        - max target tokens: 10347
    - Scale 2 output: outputs/llm_grid_samples/20260708-053628-924893-gt.legacy.r1p5.direction5.v1-n20-s2
        - mean target tokens: 2441.2
        - max target tokens: 3455
- 是否保留置信度？（未提及但是在附近的物体）
- 是否提升 max_tokens？
- Downsample 必要性？不如直接降低 VLN 模型的期望分辨率，而非预测低分辨率的然后缩放？

## 07/09

LLM-Grid token stats (`d071fe5`, 2x maxpool):

- Mean: 1084
- Median: 1076
- P95: 1637
- P99: 1957
- Max: 2536

![llm_grid_token_distribution_blurred](images/llm_grid_token_distribution_blurred.png)

## 07/10

LLM-Boxes token stats:

| split      | n      | mean | p50 | p90 | p95  | p98  | p99  | p99.5 | max  |
| ---------- | ------ | ---- | --- | --- | ---- | ---- | ---- | ----- | ---- |
| train      | 10,786 | 525  | 447 | 991 | 1218 | 1552 | 1764 | 1996  | 3090 |
| val_seen   | 778    | 512  | 432 | 997 | 1170 | 1467 | 1645 | 1873  | 2380 |
| val_unseen | 1,839  | 547  | 488 | 983 | 1178 | 1464 | 1618 | 1712  | 2689 |
| all        | 13,403 | 527  | 453 | 991 | 1207 | 1530 | 1750 | 1981  | 3090 |

Drop Rate By `max_new_tokens`:

| max_new_tokens | dropped | rate  |
| -------------- | ------- | ----- |
| 1024           | 1203    | 8.98% |
| 1280           | 546     | 4.07% |
| 1536           | 262     | 1.95% |
| 1792           | 110     | 0.82% |
| 2048           | 46      | 0.34% |
| 2304           | 20      | 0.15% |
| 2560           | 7       | 0.05% |
| 3072           | 1       | 0.01% |

## 07/15

- 实验
    - 输入：是否足够？动态输入（拓扑图）？
    - 预测器的问题：多模态大模型？是否过拟合（训练集/验证集的可行比例，手动比较）？
    - 输出：问题是否可解？量化哪些能预测，哪些不能预测？
    - 需要了解为什么可行或不行
- 思路
    - 认知地图 - 情境地图的不同/作用，如何融合才能起作用？
    - 证据/实验证明认知地图的作用（去除情境地图？）
        - "认知地图是更好的媒介" - 证明
- 优先
    - 分析样本，哪些可以预测 - 今天
        - 哪些类别好或差
        - 训练集 vs 验证集的区别
        - 预测好的 vs 预测差的，为什么会预测好/差
            - 可能语言指令的问题（模糊）
            - 可能预测的问题
    - 确定性的能预测的，预测出来多少
    - 不确定的没法预测的，其它办法解决
- 分工
    - 我：预测
    - 学长：融合
- 分析
    - 采样 (`5dbad61`): `python -m prior.analyze.batch_vis --count 20 --seed 0`; `python -m prior.analyze.batch_vis --prediction-root data/llm_navigation/llm-grid-r2r-legacy-r1p5-direction5-scale2/r2r/train/cognitive_maps/raster --count 20 --seed 0`
        - [`R2R_train_4715`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/29hnd4uzFmX/R2R_train_4715.png): 预测的路径方向不正确（指令中确实没明说）；正确推测出初始房间的类型，未能推测出 `sofa`
        - [`R2R_val_unseen_1756`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/2azQ1b91cZZ/R2R_val_unseen_1756.png): 明显质量差
        - [`R2R_train_4823`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/759xd9YjKW5/R2R_train_4823.png): 完全一致
        - [`R2R_train_7354`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/8WUmhLawc2A/R2R_train_7354.png): 很接近 GT，甚至预测到了 `bathroom`
        - [`R2R_train_7433`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/8WUmhLawc2A/R2R_train_7433.png): 只预测了一半，预测的也没有很好地和轨迹对齐；能预测到 `bathroom`，学习到 `bathroom` -> `towel`/`bathtub`/`sink` 等的关联，但是 GT 中没有出现后两者
        - [`R2R_val_unseen_1275`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/EU6Fwq7SyZv/R2R_val_unseen_1275.png): 指令本身没有场景信息；上楼
        - [`R2R_train_3215`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/JeFG25nYj2p/R2R_train_3215.png): 门、区域预测基本正确
        - [`R2R_train_6444`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/PX4nDJXEHrG/R2R_train_6444.png): 有 dining room，但是方向错误；区域和 物体预测总体不好
        - [`R2R_train_636`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/PuKPg4mmafe/R2R_train_636.png): 指令过于简单，信息量不足；只预测了一小部分，但是预测的正确
        - [`R2R_val_unseen_1095`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/QUCTc6BB5sX/R2R_val_unseen_1095.png): 预测初始房间就是 bathroom, 可以认为预测错误
        - [`R2R_val_unseen_176`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/QUCTc6BB5sX/R2R_val_unseen_176.png): 偏移；预测到了 bathroom 和 recreation，相对方位错误
        - [`R2R_val_unseen_786`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/QUCTc6BB5sX/R2R_val_unseen_786.png): 偏移；bathroom & bedroom 基本正确
        - [`R2R_train_5262`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/SN83YJsR3w2/R2R_train_5262.png): 预测错误
        - [`R2R_val_unseen_1409`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/TbHJrupSAjP/R2R_val_unseen_1409.png): 预测错误；指令仅包含区域信息
        - [`R2R_val_unseen_309`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/TbHJrupSAjP/R2R_val_unseen_309.png): 指令方向信息不足；预测错误
        - [`R2R_val_unseen_581`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/TbHJrupSAjP/R2R_val_unseen_581.png): 预测错误
        - [`R2R_val_unseen_792`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/TbHJrupSAjP/R2R_val_unseen_792.png): regions 有前进离开房间进入 bedroom，并离开 bedroom 的意思；door 疑似太大了
        - [`R2R_train_1582`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/Uxmj2M2itWa/R2R_train_1582.png): 预测错误
        - [`R2R_train_5570`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/Vvot9Ly1tCj/R2R_train_5570.png): 可以预测到起始/结束的区域，但是预测总体较差
        - [`R2R_val_unseen_372`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/X7HyMhZNoso/R2R_val_unseen_372.png): 方向错误（指令未提及）；未能预测到提及的 dining room；预测错误
        - [`R2R_val_unseen_599`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/X7HyMhZNoso/R2R_val_unseen_599.png): 路径方向正确（指令提及）；能预测到末尾的 bedroom 以及 door；关键部分可以认为正确
        - [`R2R_val_unseen_927`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/X7HyMhZNoso/R2R_val_unseen_927.png): 起始位置偏移；预测的是可行解（bathroom 向下右转进入 bedroom），但是方向错误；不符合 GT
        - [`R2R_val_unseen_1164`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/Z6MFQCViBuw/R2R_val_unseen_1164.png): 预测错误；指令未指明方向与物体/区域，信息量少
        - [`R2R_train_528`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/ZMojNkEp431/R2R_train_528.png): 起始位置偏移；物体/区域类别基本正确（只缺少 circulation）；预测类似 GT
        - [`R2R_train_1390`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/ac26ZMwG7aT/R2R_train_1390.png): 旋转 90° 后房间布局类似（均有 `outdoor`/`dining`），物体分布类似；指令本身不精确
        - [`R2R_train_492`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/b8cTxDM8gDG/R2R_train_492.png): 涉及上楼；预测较为精确，只是缺少了 painting
        - [`R2R_train_9133`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/jh4fc5c5qoQ/R2R_train_9133.png): 预测正确
        - [`R2R_train_6850`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/kEZ7cmS4wCh/R2R_train_6850.png): 未能捕捉到 "turn around" 含义；旋转后布局类似（bed, bathtub...）
        - [`R2R_train_7976`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/mJXqzFtmKg4/R2R_train_7976.png): 偏移；旋转 180°后区域布局类似；bed 不在 bedroom 内
        - [`R2R_val_unseen_1047`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/oLBMNvg9in8/R2R_val_unseen_1047.png): 预测错误
        - [`R2R_train_454`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/p5wJjkQkbXX/R2R_train_454.png): 预测不完整；指令中场景信息不足
        - [`R2R_train_6808`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/qoiz87JEwZ2/R2R_train_6808.png): 偏移；预测错误；可以认为初始从左上角 bathroom 向右，出房间右转（方向错误）
        - [`R2R_train_9263`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/s8pcmisQ38h/R2R_train_9263.png): 预测不完整
        - [`R2R_train_6840`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/uNb9QFRL6hY/R2R_train_6840.png): 指令信息量少；预测基本正确
        - [`R2R_val_unseen_1440`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/zsNo4HB9uLZ/R2R_val_unseen_1440.png): 可以预测到 bed 和 bedroom 的大致位置（指令未提及方向）；终点预测错误
        - [`R2R_val_unseen_1685`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/zsNo4HB9uLZ/R2R_val_unseen_1685.png): 轨迹方向大致正确；sofa counter 相对位置正确，甚至能预测到终点附近的 bathroom；预测质量较好
        - [`R2R_val_unseen_387`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/zsNo4HB9uLZ/R2R_val_unseen_387.png): 轨迹方向大致正确；可以预测到中间的 kitchen，未能预测到起点的 bathroom；物体预测不好
        - [`R2R_val_unseen_441`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/zsNo4HB9uLZ/R2R_val_unseen_441.png): 能预测到终点 bathroom，但是多预测了一个；区域分布合理，物体预测不好（指令缺少物体描述）
        - [`R2R_val_unseen_924`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/zsNo4HB9uLZ/R2R_val_unseen_924.png): 起点偏移；指令信息过少
        - [`R2R_val_unseen_963`](./images/llm-grid-s2.legacy.r1p5.direction5.comparison/zsNo4HB9uLZ/R2R_val_unseen_963.png): 起点偏移，预测错误；指令信息过少
    - 总结
        - 可能有过拟合现象
        - 信息确实不足

## 07/17

- 实验过程中的问题
    - grid ​预训练缓存只生成了一半（可能一部分 worker 中途 oom 了），预训练结束了，微调还没开始
    - boxes 开始预训练了，使用当前架构，预训练缓存生成率 ~70%
    - 读论文？
- 后续
    - [x] bbox sample 比对
    - [x] 用上 RxR 数据 (数据不够 (~1w) -> 过拟合)
        - [x] RxR 会不会爆 OOM? 统计 max token 后调整
        - [x] 加快训练速度？多卡调小 rank 调大并行度？
    - [x] 去除 dataset 标记
    - [x] 采样前几个 epoch 生成的数据（epoch 过多 -> 过拟合）
    - [x] LLM-Grid predictor 定量评估：IoU、cell/category precision/recall/F1、mentioned/unmentioned category/spatial quality、方向向量与格式合法率
    - [x] 重要：除了跑实验外，还需要分析成功/失败结果；周报加上思考过程
    - [x] 验证工具网站数据的全面性

加上 RxR 后的 token distribution：

| RxR train target  | Examples | Mean  | P95   | P99   | Max   | Over 2048 | Over 3072 | Over 4096 |
| ----------------- | -------- | ----- | ----- | ----- | ----- | --------- | --------- | --------- |
| LLM-Boxes         | 19,954   | 1,234 | 2,818 | 3,958 | 7,173 | 14.83%    | 3.37%     | 0.81%     |
| LLM-Grid, scale 2 | 19,954   | 1,388 | 2,595 | 3,219 | 4,969 | 15.80%    | 1.59%     | 0.15%     |

## 07/18

- 分析 LLM-Boxes 1 的结果 (`e9b6cbc`): `python -m prior.analyze.batch_vis --prediction-root data/llm_navigation/llm-boxes-r1p5-path5-r2r-only/r2r/train/cognitive_maps/raster/ --ground-truth-root data/cognitive_maps/gt.bbox.r1p5.path5.v1 --count 20 --seed 0`, `python -m prior.analyze.batch_vis --prediction-root data/llm_navigation/llm-boxes-r1p5-path5-r2r-only/r2r/val_unseen/cognitive_maps/raster/ --ground-truth-root data/cognitive_maps/gt.bbox.r1p5.path5.v1 --count 20 --seed 0`
    - 同样过拟合
        - 部分训练数据上关键路径点基本一致 (`R2R_train_2883`, `R2R_train_3356`)
        - 物体基本乱预测
        - 区域/物体有时候预测为空 (`R2R_val_unseen_162`, `R2R_val_unseen_802`)
    - 模型更容易关注细节信息 (token 占比大)

## 07/21

- [详细记录：LLM-Grid Epoch 5/10 泛化分析](daily/2026-07-21.md)
- 早期 epoch 5/10 抽样用于发现 train/unseen 差距；正式 checkpoint 结论已由 07/22 的完整固定分母 sweep 取代，逐样本清单仍保留作历史证据。
- GT namespace 名称差异不是混淆因素：训练会对 non-blurred cache 做 scale-2 max pooling，与 blurred cache 的有效 50×50 target 相同。system prompt 的 row/column 与 direction vector 语义错误作为独立实现混淆因素处理，不与 input insufficiency 合并归因。
- 该日提出的 predictor 成因与可预测边界问题已由 [07/23 的今日结论](daily/2026-07-23.md#今日结论) 回答。
- `llm-grid-try5-pt` 暂时保留到下一个可用 checkpoint，作为 integration datapoint；不要默认跑满 500k steps。

## 07/22（续 07/21：checkpoint 泛化曲线）

- [详细记录：LLM-Grid Checkpoint 泛化曲线](daily/2026-07-22.md)
- 先在原 prompt contract 下公平比较 epoch 1–10，再把 prompt 修正作为新的训练实验线；否则无法判断现有 run 从何时开始过拟合。
- 正式比较需要完整且相同的 R2R validation denominator；现有 epoch 5/10 抽样 cache 不能用于完整曲线。
- evaluator、validation-only cache scope、显式多 cache sweep 和折线图均已完成；epoch 1–10 都覆盖相同的 778 个 `val_seen`、1839 个 `val_unseen` episode，raw prediction 缺失数为 0。
- 按预先固定的 `val_unseen` raster IoU 主指标，epoch 2 为候选（0.135701）；epoch 8 的 cell F1 最高且 IoU 仅低 0.000496，epoch 4 的 direction cosine 最高，三者差异需 scene-bootstrap CI 才能判断显著性。
- unseen IoU 在 epoch 2 后进入平台，seen IoU 却继续升至 epoch 9 的 0.299，seen–unseen gap 同期扩大到 0.167，清楚支持继续训练增强 scene memorization 而非 unseen 空间泛化。epoch 5 是 schema validity 明显下降的独立格式异常点。
- episode 级分析进一步显示 instruction 长度与 unseen IoU 几乎无关（`r=-0.028`），target 空间密度有中等负相关（`r=-0.280`），而类别 F1 与 unseen IoU 的相关性很弱；scene novelty 与输入不可辨识性仍是主要解释。

## 07/24（起点观测边界 oracle evidence）

- [详细记录：LLM-Grid 起点观测边界 oracle evidence](daily/2026-07-24.md)
- 已选择 causal-diagnostic first：用受 `t=0` 视野与遮挡限制、且与 instruction/goal/GT trajectory 无关的 semantic+depth oracle evidence 检验 input insufficiency；oracle 标签只作为 predictor 诊断上界，不作为导航候选。
- 保持 full-grid target、LLM JSON、navigation-ready `.npz`、Try5 与 navigation trainer 不变；第一轮只改变 predictor input/cache generation，并对 matched/null/within-scene shuffle/global shuffle 做显式 manifest。
- Uncertainty-Aware VLN 论文建模的是 observation-built online Gaussian map 上的局部 perceptual reliability，不区分 free/unknown、没有 multiple full-layout hypotheses，也未用 matched controls 隔离 information gain 与 architecture gain；第一轮只借鉴 explicit observed/unknown 表示，不引入 Gaussian map、uncertainty output 或 online map fusion。
- 当前 frozen waypoint sidecar 可低成本复用，但实际为 depth-only，保留作 deployable geometry control；Habitat semantic+depth renderer 可作为 oracle 起点，但 37-channel projection、free/unknown mask 与 vocabulary coverage 仍需原型验证。
- 起点 oracle 原型、strict artifact/index、四类 observation-level assignment、train/cache/eval wiring 已完成。R2R `val_unseen` 50-observation sample 对应 222 examples，生成约 1 分 50 秒，artifact 平均 3.5 KB；t=0 observed cells 覆盖 full target semantic support 的 64.1%，直接 evidence 在 observed target 上 recall 67.2%，但对 route-relevant target precision 仅 17.1%，因此必须由 instruction-conditioned predictor筛选，不能直接复制。
- 原始 evidence JSON 严重超 prompt budget；exact compact row-run grammar 并去除与 observed/free mask 重复的 broad environment labels 后，sample median/P90/max prompt 为 1,229/2,049/2,580 tokens，3,072 prompt + 4,096 sequence budget 下 sample 无 truncation。下一步先在 login node 并行生成六个 R2R/RxR split caches，再跑 seed 42 的 2-epoch matched screen 与 matched/null/within/global controls。
- 完整 R2R `val_unseen` 的 prompt median/P90/max 为 1,246/1,986/2,727；仅 1/1,839（0.054%）因 combined sequence 4,118 超预算。Observed mask 覆盖 70.2% target semantic cells；实际 prompt semantic evidence precision/recall 为 16.1%/25.5%。Evaluator controls 已禁止读取 prompt 未包含的 raw environmental labels，否则 recall 会虚增到 48.2%。
- RxR `val_unseen` 的 fixed raster corpus 为 evidence index 11,006 episodes 中的 3,669 examples；prompt median/P90/max 为 1,225/2,013/2,800。原 4,096 sequence budget 会丢 4.69%，因此 screen 改为 completion 4,096、sequence 5,120；该 split 只剩 0.136% 超预算，train corpus 仍由 2% guard fail-fast。
- Remote cache array `1182682`（六个 split task，各 1 GPU）已完成四个 validation tasks；R2R `val_unseen` 为 393 observations/1,839 examples、9 分 30 秒，manifest/index 与抽查 artifact hash 一致。R2R/RxR train tasks 继续运行。
- Matched training `1182802`（8 GPU、seed 42、2 epochs）依赖完整 cache；四条件 cache/eval array `1182803`（matched/null/within-scene/global，各 2 GPUs）再依赖 training。旧 pending jobs `1182689`/`1182694` 的 Slurm snapshots 保留旧 budget，均未运行即取消并替代。

## 07/23（mentioned/unmentioned 空间与类别）

- [详细记录：LLM-Grid mentioned/unmentioned 空间与类别分析](daily/2026-07-23.md)
- 使用 instruction-derived category partition 重跑相同的 epoch 1–10 R2R sweep；不采用模型自报的 `mentioned` flag，category presence 与 spatial quality 都由同一组完整词表 channel mask 划分。
- 在 instruction-derived partition 的 pooled presence 指标下，unmentioned category 在 10/10 个 epoch、seen/unseen 和 object/region 中均更低。epoch 2 的 `val_unseen` mentioned/unmentioned F1 为 object 0.873/0.636、region 0.948/0.529；precision 和 recall 都同步下降。
- `val_unseen` mentioned−unmentioned F1 gap 跨 epoch 为 object 0.237–0.264、region 0.359–0.432。unmentioned target support 反而更多，差距不是小样本造成。
- 新增 spatial 指标使用同一互补 channel mask，但按 split 汇总 intersection/union/cell count 后计算 pooled IoU/P/R/F1；原 overall spatial 指标仍是 episode-macro，两者 aggregation 不同，不能直接相减或加权还原。empty/empty 不产生伪零分，prediction-only 计 false positive，invalid-with-target 计 false negative，并另报 target-bearing episode support。
- 空间结果没有复现 category presence 的大幅 mentioned 优势：`val_unseen` unmentioned IoU/F1 在 10/10 epoch 略高，但差值仅为 `+0.0002`–`+0.0104`/`+0.0003`–`+0.0167`；epoch 2 两组几乎相同（IoU 0.1181/0.1182，F1 0.2112/0.2115）。seen 的差值方向随 epoch 改变。
- [2×2 空间与类别曲线](images/llm_grid_mentioned_spatial_category_sweep_r1p5.png) 显示两组共同保留 seen 持续改善、unseen 早期平台化的 spatial gap。近似持平或轻微反向的 M/U spatial gap 受 category composition、cell density 与 prevalence 混淆，不能解释为 unmentioned layout 更好。
- category 结果与 full-grid target 要求恢复输入未提供的 scene context 这一假设一致，但 category composition/base rate 仍是混淆因素；本轮也不能直接证明 mentioned-only target 会改善 predictor 或下游导航。
- 今日结论：主要瓶颈是 input/target 信息不匹配；当前输入不含 scene observation，却要求恢复 unseen-house 精确 full-grid 几何与 unmentioned content。scene-specific overfit 会放大差距，但不是首要根因；错误 prompt 是单独的实现混淆因素。
- prompt contract 已在 `a0bb323` 修正并完成 matched 2-epoch control。`val_unseen` Raster IoU 为 0.1357→0.1340，delta -0.0017 `[-0.0080, 0.0051]`；Cell F1 同样不能与零区分，该 checkpoint 的 direction 与 object/region F1 下降。代码链与 raw-output 变换诊断确认旧 row/column 和 `[dx,dz]` 定义确实错误，不能因本次下降而恢复；但训练 seed 只固定 sampler、未统一 LoRA/Torch/CUDA/dropout RNG，因此单次 run 只能说明修正未解决低 IoU，不能证明正确 contract 导致性能下降。v2 仍遗漏 level-local origin/resolution 与 direction change-point 选择规则，详见[今日记录](daily/2026-07-23.md#单独的实现混淆因素)。
- 旧 prompt epoch 2 的 [input-dependence controls](daily/2026-07-23.md#input-dependence-controls) 已完成：`val_unseen` binary raster IoU 为 matched 0.136、within-scene permutation 0.061、global permutation 0.036；同路径 paraphrase prediction IoU 为 0.284。输出含 episode-specific signal，但不足以稳定恢复唯一 full-grid layout；这里是 cache reassignment，不是真正的 shuffled-input model run。
- paired scene-cluster bootstrap 已完成：epoch 4/8 相对 epoch 2 的 `val_unseen` Raster IoU delta 均不能区分于零；contract-v2 相对旧 epoch 2 也不能区分于零。详细 matched 指标见[今日记录](daily/2026-07-23.md#matched-result)。
- 增加输入的首选最小改动是只改 cache generation：把起点 `t=0` panorama 投影成带 observed/unknown mask 的稀疏观测证据，逐级比较 free-space、semantic inventory 与 spatial semantic cells，保持 `.npz`、Try5 和 navigation 接口不变。详见[方案与防泄漏约束](daily/2026-07-23.md#增加输入但保持导航接口不变)。
- 下一步：先统一固定训练全部 RNG，再用 paired run seeds 依次比较 full-grid、mentioned-only absolute grid、mentioned-only start-centered/heading-normalized route corridor；若仍不足，再在不改 navigation 接口的前提下逐级加入 `t=0` 观测。只有超过 prior、对输入 shuffle 敏感且缩小 seen/unseen gap 的方案才进入下游导航实验。

# 实验

- [x] 去除随机旋转，加入 cross attn 的 GT 实验 (try 8@超算)
- [x] 去除随机旋转的 LLM 评估 (llm 5@超算)
- [x] 去除随机旋转的 LLM 导航缓存生成 (llm 5@超算)
    - R2R train: 10494/10819
    - R2R val_unseen: 1351/1839
- ~~Nav 1: Try 8 + LLM 5~~
- [x] Try 9: 简易的认知地图 Decoder @ VIPL
- [x] LLM 5 nav cache (structured, `c0ed1e9`) @ 超算
- ~~Nav 2: Try 9 + LLM 5 @ 超算~~
- ~~Try 10: DETR-style decoder @ VIPL~~
- [x] Try 5 repro (半径 1.5, on-the-fly) @ 超算
- [x] 2x "Blurred", r=1.5 Try5-like (try5-r1p5-blurred) @ 超算
- [x] 2x "Blurred", r=1.5 LLM-Grid 微调 (r2r-legacy-r1p5-direction5-scale2) @ 超算
- [x] 2x "Blurred", r=1.5 LLM-Grid 预训练导航缓存生成 @ 超算
- [ ] 2x "Blurred", r=1.5 LLM-Grid, Try 5 架构 导航 @ 超算
- [x] LLM-Boxes 微调 @ 超算
- [x] LLM-Boxes 导航缓存生成 @ 超算
- ~~LLM-Boxes, Try 10 架构 导航 @ 超算~~
- [ ] LLM-Grid 2 大模型微调 @ 超算
