# 14｜oracle-seen-only 零训练归因评测 Task Book

日期：2026-08-26
执行方式：Claude Code CLI，在本地 Try5 仓库中执行
前置文档：12（融合方案）、13（六项风险回应）

---

## 0. 目标与决策规则

用**固定的 Try5 checkpoint**、不训练任何模块，在 val_unseen 上跑四组地图配置，回答一个问题：

> 若 agent 亲眼看过的区域被完美修正到 GT，其余保持 LLM 先验，SR 能到多少？

四组（同一 checkpoint、同一评测集、同一 endpoint 逻辑、同一随机种子）：

| 组名 | 已覆盖区域 | 未覆盖区域 | 作用 |
|---|---|---|---|
| `llm_full` | LLM | LLM | 复现基线 65.42 / 55.25 |
| `gt_seen` | GT | LLM | **核心诊断** |
| `gt_unseen` | LLM | GT | 量化远端 gap |
| `gt_full` | GT | GT | 复现上界 ~72 |

决策规则（文档 13 §1 + 修正余量）：

```text
gt_seen SR ≥ 69.0        → GO，按文档 12/13 修改设计后训练
67.6 ≤ gt_seen SR < 69.0 → 可做，但需压缩范围，先讨论
gt_seen SR < 67.6        → NO-GO，停止"收益主要来自已见纠错"叙事
```

同时报告回收率 `(SR_gt_seen − SR_llm_full) / (SR_gt_full − SR_llm_full)`。

---

## 1. 硬约束

1. **不修改任何模型权重、训练代码、配置默认值。** 所有改动放在新文件或以 `--oracle_mode` 开关注入的评测路径中，默认关闭时行为与原评测完全一致。
2. **覆盖必须来自该组自己走出的轨迹。** 四组各自 rollout，各自累积覆盖；禁止用 GT path 预计算覆盖，禁止跨组复用轨迹。
3. **覆盖只来自几何。** 用真实 Depth + 真实 pose 做射线投射，不使用 Semantic sensor、不使用任何学习模块。
4. `llm_full` 和 `gt_full` 必须先复现出与已知数字一致的结果（±0.3 SR），否则后续两组不可信，先排查。
5. 不做任何"顺手优化"。发现原代码 bug（如文档 13 提到的 `route_map_progress` argmin 问题）只记录，不在本轮修。

---

## 2. Phase 0 — 仓库勘察（先做，写入 `reports/oracle_seen/phase0_discovery.md`）

Claude Code 需要先弄清并记录以下事实，**不确定的项写"未确认"，不要猜**：

- [ ] Try5 评测入口脚本、checkpoint 路径、val_unseen 配置文件
- [ ] LLM-Grid 地图 `P0` 的存储位置、格式、shape（预期 `[C, 100, 100]`，C = 通道数：region/object 如何分层）
- [ ] GT 地图的存储位置、格式、shape；确认与 `P0` **同一坐标系、同一原点、同一朝向、同一通道顺序**（跑 gt_full 复现 72 即验证）
- [ ] 地图坐标系定义：起点格索引、0.5 m/cell、初始 heading 对齐方式（对齐到 +x 还是 +y）、Habitat Y-up → 地图 2D 的投影轴
- [ ] 每个高层 step 可获取的：agent `(x,y,z)`、heading、12-view Depth（分辨率、fov、每个 view 的相对方位角）、`GraphMap.node_pos`
- [ ] 地图在评测路径中被读入的确切位置（哪一行把 `P0` 送进 Map Encoder）——这是注入点
- [ ] 高层 step 与 waypoint 的对应关系：地图更新应在每个高层 step 决策**之前**完成
- [ ] Depth 有效范围（min/max，是否 normalize 到 [0,1]）
- [ ] 评测中 endpoint / STOP 逻辑是否有随机性；固定 seed 的方法
- [ ] 多层场景处理：`P0`/GT 是否只表示起点楼层；若是，覆盖时需按高度过滤

Phase 0 完成后**暂停并汇报**，确认注入点无误再继续。

---

## 3. Phase 1 — 几何覆盖模块 `oracle_seen/coverage.py`

### 3.1 接口

```python
class CoverageAccumulator:
    def __init__(self, grid_size=100, cell_m=0.5, origin_xy, heading0,
                 max_range_m=10.0, floor_z=None, floor_tol_m=1.0):
        """
        origin_xy: episode 起点在世界坐标中的 (x, y)（地图 2D 平面）
        heading0:  起点朝向，用于把世界坐标旋转到地图坐标
        floor_z:   起点高度，用于多层过滤（Phase 0 决定是否需要）
        """
        self.C = np.zeros((grid_size, grid_size), dtype=bool)   # 累计覆盖
        self.hit = np.zeros((grid_size, grid_size), dtype=bool) # 累计击中（可选统计）

    def world_to_grid(self, xy_world) -> (row, col)
    def update(self, pose, depth_views, view_yaws, depth_meta) -> None
    def mask(self) -> np.ndarray   # 返回 self.C 副本
```

### 3.2 `update` 的射线投射逻辑

对每个 view、每列像素（可按列下采样，如每 4 列取 1 列）：

1. 取该列在**agent 高度附近**的深度值（取中间若干行的 min，避免地板/天花板）。
2. 若深度无效（0 / 超出 max_range），沿该射线只标记到 `max_range` 为止的自由格子为覆盖（可选，默认**不标记**——无效深度不产生覆盖，保守）。
3. 若深度有效：世界坐标下从 agent 位置出发，沿该列对应的方位角走到击中点；用 Bresenham 或 0.25 m 步长采样，把经过的格子写入 `C`，击中格子同时写入 `hit`。
4. 多层过滤：若 `floor_z` 给定，击中点高度与 `floor_z` 差超过 `floor_tol_m` 的射线整条丢弃。

方位角计算：`yaw_world = agent_heading + view_yaw + pixel_angle`，`pixel_angle` 由列索引与 fov 线性映射。**先在 Phase 0 确认 Habitat 的 heading 正方向和旋转方向，写一个 3 点单元测试（前方 / 左侧 / 右侧各放一个已知深度）验证投影落到正确格子。**

### 3.3 保守性说明

覆盖是"上界诊断"的一部分：**宁少勿多**。任何不确定的像素（深度无效、边缘、跨层）都不标记覆盖。少标记只会让 `gt_seen` 偏低，不影响 NO-GO 结论的可靠性；多标记会虚高，是不可接受的方向。

---

## 4. Phase 2 — 地图混合注入 `oracle_seen/mixer.py`

```python
def mix_map(P_llm, P_gt, C, mode):
    # P_*: [Ch, 100, 100] float; C: [100,100] bool
    if mode == "llm_full":  return P_llm
    if mode == "gt_full":   return P_gt
    if mode == "gt_seen":   return np.where(C[None], P_gt,  P_llm)
    if mode == "gt_unseen": return np.where(C[None], P_llm, P_gt)
```

注入点：在 Phase 0 确认的那一行之前，每个高层 step：

```text
pose, depth = simulator 读取
acc.update(pose, depth, ...)
P_t = mix_map(P0, GT, acc.mask(), mode)
→ 原 Map Encoder → 原 Try5
```

要求：

- 通过评测脚本的新参数 `--oracle_mode {llm_full,gt_seen,gt_unseen,gt_full}` 选择；不给参数时走原路径（不实例化 accumulator，不调用 mix）。
- `P_t` 的 dtype、shape、device 与原 `P0` 完全一致；混合在 CPU numpy 完成后再转 tensor，避免改动 Map Encoder。
- 第 0 个高层 step（agent 未动）：`C` 已包含起点的第一次观测，`gt_seen` 应在起点周围就有 GT。

---

## 5. Phase 3 — 评测与逐步日志

### 5.1 运行

```bash
for mode in llm_full gt_seen gt_unseen gt_full; do
  python <eval_entry> --oracle_mode $mode --seed 0 \
      --log_dir reports/oracle_seen/$mode
done
```

先用 **50 个 episode** 的子集跑通四组（子集 id 固定并保存），确认 `llm_full` 与 `gt_full` 与原数字方向一致，再跑全量 val_unseen。

### 5.2 每 episode 每高层 step 记录（jsonl）

```json
{"ep": 362, "step": 3,
 "pos": [x, y, z], "heading": h,
 "cov_cells": 412, "cov_frac": 0.041,
 "dist_to_gt_suffix": 1.7,
 "on_route": true,
 "map_l1_seen_vs_gt": 0.0,      // gt_seen 组应恒为 0（sanity）
 "map_l1_unseen_vs_gt": 0.23,
 "action": "...", "stopped": false}
```

`dist_to_gt_suffix`：当前位置到**剩余** GT path 的最近距离。进度指针用 `suffix_start + argmin(suffix_distances)` 且只允许前移（文档 13 §2），不要复用原 `route_map_progress`。

### 5.3 每 episode 汇总

`success, spl, path_len, nav_error, n_steps, final_cov_frac, never_offroute, first_offroute_step, recovered`

---

## 6. Phase 4 — 统计与报告 `reports/oracle_seen/summary.md`

### 6.1 主表

| mode | SR | SPL | NE | 回收率 | mean final_cov_frac |
|---|---|---|---|---|---|

### 6.2 路线统计（文档 13 §6，四组都报）

- `never_offroute_rate`：全程 `dist_to_gt_suffix ≤ 3 m` 的 episode 比例
- `first_offroute_step`：首次 > 3 m 的高层 step 分布（median / 直方图）
- `recovery_rate`：首次偏离后重新回到 ≤ 3 m 的比例
- 按 `never_offroute` 分层的 SR：看 `gt_seen` 相对 `llm_full` 的增益是来自"不走错"还是"走错后恢复"

### 6.3 覆盖分析

- `final_cov_frac` 分布；与 GT path 长度的相关性
- `gt_seen` 的增益 vs `final_cov_frac` 分桶：若增益集中在高覆盖 episode，说明方案依赖看到足够多；若与覆盖无关，怀疑增益来自起点附近少量格子（这本身是个重要发现）

### 6.4 sanity 检查（必须全部通过才出结论）

1. `llm_full` SR/SPL 在 65.42/55.25 ± 0.3
2. `gt_full` SR 在 72 ± 0.5
3. `gt_seen` 组 `map_l1_seen_vs_gt` 全程为 0
4. `gt_unseen` 组 `map_l1_unseen_vs_gt` 全程为 0
5. 覆盖单元测试（§3.2）通过
6. 四组 episode 数、seed、checkpoint hash 一致，写入报告头

### 6.5 结论段

按 §0 决策规则给出 GO / 讨论 / NO-GO，并附一句话说明 `gt_unseen` 组暗示的远端 gap 大小。

---

## 7. 交付物

```text
oracle_seen/coverage.py
oracle_seen/mixer.py
oracle_seen/test_coverage.py
<eval_entry> 的 --oracle_mode 补丁（diff 单独保存为 oracle_seen/eval.patch）
reports/oracle_seen/phase0_discovery.md
reports/oracle_seen/{mode}/steps.jsonl, episodes.jsonl
reports/oracle_seen/summary.md
reports/oracle_seen/subset50_ids.json
```

---

## 8. 执行顺序与暂停点

1. Phase 0 → **暂停汇报**
2. Phase 1 + 单元测试 → 继续
3. Phase 2 + 50-episode 四组 → **暂停汇报**（重点看 sanity 1、2）
4. 全量四组 → Phase 4 报告 → 交付

遇到 Phase 0 中任何"未确认"项影响注入点或坐标系时，先停下来问，不要按假设推进。
