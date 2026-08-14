# 07｜状态、进度与 Ghost 损失说明

日期：2026-08-13

## 1. 已采纳：DAgger 冻结 VisualEvidenceHead

- 预训练：`VisualEvidenceHead` 以 `1× LR` 训练，使用离线 37 类 panorama GT，`λ_visual=0.2`。
- DAgger：严格加载预训练权重后冻结该 head，`λ_visual=0`，不创建 Semantic sensors。
- 固定的 37 类输出仍参与地图更新；后续 `visual_category_projection` 和其余融合模块继续训练。

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

评价：它能防止地图乱写和坍塌，但也可能阻碍修正错误的初始地图；必要性较低，建议后续删除或显著减弱，而不是作为核心监督。

## 3. L_progress 是否过于复杂

当前形式：

```text
L_progress = CE(phase)
           + CE(route_state)
           + SmoothL1(remaining)
           + BCE(recovery)
```

这些预测值本身不直接送入动作 head；它们主要监督共享的 `progress hidden`，后者才进入 ghost/STOP 分支。

必要性判断：

- `route_state`：保留，用于区分 `on-route/off-route/finished`。
- `remaining`：保留，提供连续的剩余路线比例。
- `phase`：建议删除；它只是 `remaining` 的五档离散化，信息重复。
- `recovery`：不是进度，而是 off-route 时的候选动作监督；建议从 `L_progress` 拆出，之后决定与 `L_ghost` 合并或删除。

建议最终简化为：

```text
L_progress = CE(route_state) + SmoothL1(remaining)
```

本次只做必要性评估，尚未修改这部分代码。

## 4. L_ghost 是什么

`L_ghost` 只在 expert action 指向某个 ghost 时计算：

```text
更新地图 + progress hidden + ghost 表示
        → ghost residual logits
        → 对 expert ghost 做 CrossEntropy
```

它不监督 visited 节点，也不监督 STOP；预测 residual 最终会加到原 action logits 上。

主 action loss 已经监督最终 logits，因此 `L_ghost` 有一定重复；但它能防止强 baseline 独自完成决策，直接迫使“认知地图→ghost”新分支学习候选排序。当前建议保留，确认新分支有效后再考虑退火。

## 5. phase、remaining 与连续环境中的 p

### 5.1 phase 的五类

phase 不是五种语义事件，只是路线比例的五个区间：

| 类别 | p 区间 | 含义 |
|---:|---:|---|
| 0 | `[0, 0.2)` | 起始段 |
| 1 | `[0.2, 0.4)` | 前段 |
| 2 | `[0.4, 0.6)` | 中段 |
| 3 | `[0.6, 0.8)` | 后段 |
| 4 | `[0.8, 1]` | 接近终点 |

`finished` 是单独的 route state；此时 phase 被 mask。

### 5.2 remaining

```text
remaining = 1-p
```

它表示 GT 路径还剩多少比例，不是剩余米数、动作数或语义子任务数。`finished` 时为 0；`off-route` 时不计算该损失。

### 5.3 连续环境如何计算 p

DAgger 中的当前位置虽然连续，但 GT trajectory 是一串离散坐标：

1. 计算当前位置到每个 GT 路径点的 geodesic distance；
2. 只在不早于历史进度的路径点中选择最近点；
3. 保存其索引 `i`，并强制索引不回退；
4. 计算 `p=i/(N-1)`，其中 `N` 是 GT 路径点数量。

所以它本质是“连续位置投影到最近的未来 GT 点”，不是连续积分距离。GT 点近似均匀时才近似真实路程比例；更准确的做法是使用累计 geodesic 路径长度比例。

另外，预训练的 `p` 使用拓扑 viewpoint 索引，DAgger 使用连续 GT trajectory 点索引，两阶段定义并不完全一致。
