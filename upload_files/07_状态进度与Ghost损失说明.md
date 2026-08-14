# 07｜状态、进度与 Ghost 损失说明

日期：2026-08-13

## 1. 已采纳：DAgger 冻结 VisualEvidenceHead

- 预训练：`VisualEvidenceHead` 以 `1× LR` 训练，使用离线 37 类 panorama GT，`λ_visual=0.2`。
- DAgger：严格加载预训练权重后冻结该 head，`λ_visual=0`，不创建 Semantic sensors。
- 固定的 37 类输出仍参与地图更新；后续 `visual_category_projection` 和其余融合模块继续训练。
- 该 head 只读取现有 `gmap_img_fts` panorama 节点特征，不读取 graph step/task/global-position embedding 或指令。

## 2. L_state 是什么

```text
L_state = mean((Z_t-Z_0)²)
        + 0.1·mean(write_gate)
        + max(0, 0.05-map_token_spread)
```

- 第一项限制更新地图相对初始 LLM 地图变化过大。
- 第二项所谓“写门预算”其实只是**平均写门惩罚**：门开得越多，惩罚越大；它不是固定额度。
- 第三项防止所有地图 token 变得过于相似。
- 它没有 GT，整体再乘 `λ_state=0.05`。

评价：它能防止地图乱写和坍塌，但也可能阻碍修正错误的初始地图。当前方案明确保留，后续只对其权重和组成做消融。

## 3. L_progress 与 L_recovery 的最终划分

```text
L_progress = CE(phase)
           + CE(route_state)
           + SmoothL1(remaining)

L_recovery = class-balanced BCE(recovery)
```

拆分已实施：`recovery` 是 off-route 时的逐 ghost 候选动作监督，以独立 `λ_recovery=0.1` 加权；它不再影响 `L_progress` 的量级。`phase` 目前仍保留，作为 `remaining` 的五档离散监督，是后续可消融项。

各预测值本身不硬编码回 action head；它们共享的 `progress hidden` 会条件化 map→ghost 查询，并进入 ghost/STOP 独立残差分支。

## 4. map→ghost 查询与残差边界

```text
graph query = graph node + instruction pool + progress hidden
graph context = CrossAttention(graph query, updated map tokens)
```

- 只有 ghost 获得 map-conditioned **graph embedding residual**；visited 节点、padding 和 STOP 保持原 graph embedding。
- ghost 和 STOP 的 action logit 由两个独立 head 修正：`ghost_residual_head` 只写 ghost，`stop_residual_head` 只写 STOP。
- 这样既让认知地图影响 frontier 排序，又不污染 visited 表示，STOP 也不被当作空间节点。

## 5. L_ghost 是什么

`L_ghost` 只在 expert action 指向某个 ghost 时计算：

```text
用 instruction/progress 查询得到的地图上下文 + ghost 表示
        → ghost residual logits
        → 对 expert ghost 做 CrossEntropy
```

它不监督 visited 节点，也不监督 STOP；预测 residual 最终会加到原 action logits 上。

主 action loss 已经监督最终 logits，因此 `L_ghost` 有一定重复；但它能防止强 baseline 独自完成决策，直接迫使“认知地图→ghost”新分支学习候选排序。当前建议保留，确认新分支有效后再考虑退火。

## 6. phase、remaining 与连续环境中的 p

### 6.1 phase 的五类

phase 不是五种语义事件，只是路线比例的五个区间：

| 类别 | p 区间 | 含义 |
|---:|---:|---|
| 0 | `[0, 0.2)` | 起始段 |
| 1 | `[0.2, 0.4)` | 前段 |
| 2 | `[0.4, 0.6)` | 中段 |
| 3 | `[0.6, 0.8)` | 后段 |
| 4 | `[0.8, 1]` | 接近终点 |

`finished` 是单独的 route state；此时 phase 被 mask。

### 6.2 remaining

```text
remaining = 1-p
```

它表示 GT 路径还剩多少比例，不是剩余米数、动作数或语义子任务数。`finished` 时为 0；`off-route` 时不计算该损失。

### 6.3 连续环境如何计算 p

DAgger 中的当前位置虽然连续，但 GT trajectory 是一串离散坐标：

1. 计算当前位置到每个 GT 路径点的 geodesic distance；
2. 只在不早于历史进度的路径点中选择最近点；
3. 保存其索引 `i`，并强制索引不回退；
4. 计算 `p=i/(N-1)`，其中 `N` 是 GT 路径点数量。

所以它本质是“连续位置投影到最近的未来 GT 点”，不是连续积分距离。GT 点近似均匀时才近似真实路程比例；更准确的做法是使用累计 geodesic 路径长度比例。

另外，预训练的 `p` 使用拓扑 viewpoint 索引，DAgger 使用连续 GT trajectory 点索引，两阶段定义并不完全一致。

## 7. 六项辅助 loss 与训练契约

OnlineFusion 共有 `grid/state/visual/progress/recovery/ghost` 6 项辅助 loss。预训练全部启用；DAgger 令 `visual=0`，因此是 `L_action` 加其余 5 项辅助 loss。

- 预训练：Map Encoder 与 OnlineFusion 用 `1× LR`，现有 graph-language 后端和 MLM/SAP head 用 `0.1× LR`，language/panorama 前端冻结。基础 checkpoint 只可缺少明确新增的 Map Encoder/OnlineFusion 子树，若任一子树已存在就必须完整加载。
- DAgger：OnlineFusion 仍用 `1× LR`，但 `VisualEvidenceHead` 强制冻结；Map Encoder 与现有后端用 `0.1× LR`，其余前端冻结。
- DAgger 必须使用 `optimizer_profile=online_fusion`、`freeze_base=False`、`require_complete_pretrained_modules=True`，且从完整 OnlineFusion 预训练 checkpoint 启动。Map Encoder 或 VLN-BERT/OnlineFusion 子树的键名/shape 不完整时直接报错。
- 参数组必须完整覆盖 trainable tensors 且不重复。
