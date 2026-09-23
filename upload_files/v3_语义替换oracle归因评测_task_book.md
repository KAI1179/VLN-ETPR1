# 14 v3｜固定策略下的语义替换 oracle 归因评测 Task Book

日期：2026-08-26
执行方式：Claude Code CLI，在本地 Try5 仓库中执行
前置：文档 12、13，14 v1/v2 及两轮 CLI 反馈

**v3 变更**：主诊断改为 `sem_hit`（Depth 定覆盖、Semantic sensor 定内容）；`scene_seen` 降为对照；region 通道禁止回退到路线 GT；删除保守覆盖版本，改为覆盖召回率验证；NO-GO 改为双 CI 上界规则；术语改为"固定策略下的语义替换 oracle"；rollout 次数修正为 6。

---

## 0. 目标、术语与决策规则

用产生 65.42 / 55.25 的 LLM-Try5 checkpoint 及其 metadata，不训练任何模块，在 val_unseen 上跑六组地图配置。

**术语**：这些数字是"固定策略下的语义替换 oracle"，不是理论上界。冻结的 Try5 未见过混合地图，分布偏移会压低所有替换组；结论方向据此解释。

### 0.1 六组配置

| 组名 | 已覆盖区域写入内容 | 未覆盖区域 | 作用 |
|---|---|---|---|
| `llm_full` | — | LLM | 基线复现 |
| `sem_hit` | **Semantic sensor 真实 hit 像素投影的类别**（只写 hit 格子；region 通道保持 LLM） | LLM | **主诊断**：完美 SpatialVisualHead 能提供的全部信息 |
| `scene_seen` | 全场景 Semantic GT raster（object 通道；region 通道保持 LLM） | LLM | 对照：含"同格未见"语义，预期 ≥ sem_hit |
| `gt_seen` | 路线裁剪 GT `gt.legacy.r1p5.direction5.v1`（全部通道） | LLM | path-conditioned GT 对照 |
| `gt_unseen` | LLM | 路线裁剪 GT | 远端 gap |
| `gt_full` | 路线裁剪 GT | 路线裁剪 GT | 同 checkpoint 参考；回收率分母 |

所有组：只混合 grid，direction5 等其他输入保持 LLM 版。覆盖只一个版本（§3）。共 **6 次** rollout。

### 0.2 决策规则

主指标：`sem_hit` 相对 `llm_full` 的逐 episode 配对 ΔSR，bootstrap 95% CI。

```text
GO      ：SR(sem_hit) ≥ 69.0 且 ΔSR CI 下界 > 0
讨论    ：67.6 ≤ SR(sem_hit) < 69.0
分支    ：SR(sem_hit) 低但 SR(gt_seen) 的 CI 上界 ≥ 67.6
          → 不停线；结论改为"关键在语义筛选 / 路线门，而非视觉纠错本身"，另议
强 NO-GO：SR(sem_hit) 与 SR(gt_seen) 的 CI 上界均 < 67.6
```

回收率 `(SR_sem_hit − SR_llm_full) / (SR_gt_full − SR_llm_full)`，分母用同 checkpoint 的 `gt_full`；约 72 只作外部参考。

`gt_seen − scene_seen` 与 `scene_seen − sem_hit` 各自报告配对差，术语为 **"path-conditioned GT 性能影响"** 和 **"同格未见语义影响"**，不称"泄漏量"（两组轨迹、覆盖、地图分布均不同）。

---

## 1. 硬约束

1. 不修改模型权重、训练代码、配置默认值；改动放 `oracle_seen/`，以 `--oracle_mode` 注入，不给参数时逐位一致。
2. 覆盖与 hit 内容来自**该组自己 rollout 的轨迹**；禁止用 GT path 预计算；禁止跨组复用。
3. 覆盖只由 Depth 几何决定；Semantic sensor **只**在 `sem_hit` 中提供 hit 像素的类别，不参与判断覆盖。
4. **region 通道在 `sem_hit`/`scene_seen` 中保持 LLM 不变**；若代码路径试图用路线 GT 填 region，直接报错退出。
5. 原代码 bug 只记录不修。
6. 50-episode 阶段只验证逐 episode 行为一致性（同 episode 在 llm_full 下与原评测输出逐步一致）；65.42 ± 0.3 的复现在全量阶段验证。

---

## 2. Phase 0 — 仓库勘察（`reports/oracle_seen/phase0_discovery.md`）

不确定写"未确认"。

**checkpoint 与地图**
- [ ] 65.42 checkpoint 路径、hash、metadata（LLM grid 缓存 key、direction5 来源）
- [ ] `P0` 位置、shape `[Ch,100,100]`、通道语义、object 通道与 mpcat40 的映射
- [ ] `gt.legacy.r1p5.direction5.v1` 位置、shape、通道顺序，与 `P0` 一致性
- [ ] 坐标定义：`level_origin_xz = episode_start_xz − cache.start_position`；`grid_rc = (world_xz − level_origin_xz) / 0.5`；row/col 与 x/z 对应、翻转、`cache.start_position` 含义
- [ ] 楼层判定方式、多层 episode 处理

**Semantic sensor**
- [ ] 评测 simulator 能否同时开 Semantic sensor（与 Depth 同分辨率、同 fov、同 12-view 布局）
- [ ] Semantic 像素值 → instance id → mpcat40 类别的映射路径（house file / scene semantic annotations）
- [ ] `P0` object 通道的取值语义（0/1 占据？置信度？）：`sem_hit` 写入时用相同语义

**scene GT（scene_seen 需要）**
- [ ] 复用 `gt.legacy` 生成代码去掉 1.5 m 裁剪、保留楼层过滤，生成 `gt.scene.level.v1`；只生成 object 通道
- [ ] 全场景 region 标注是否可用：**可用则用，不可用则 region 保持 LLM**

**观测与注入点**
- [ ] 每高层 step 可得：pose、heading、12-view Depth / Semantic、各 view 相对 yaw、相机内参
- [ ] `llm_grid_oracle_cache.py` 中的完整 Depth hit 投影逻辑：签名、坐标系、高度处理；**直接复用，不重写**
- [ ] `P0` 送入 Map Encoder 的注入点；混合在每高层 step 决策前完成
- [ ] endpoint/STOP 随机性与 seed

Phase 0 后**暂停汇报**。

---

## 3. Phase 1 — 覆盖与 hit 投影 `oracle_seen/coverage.py`

### 3.1 接口

```python
class HitAccumulator:
    def __init__(self, level_origin_xz, cell_m=0.5, grid_size=100, n_classes=37):
        self.C   = np.zeros((grid_size, grid_size), bool)          # 射线经过或击中
        self.hit = np.zeros((grid_size, grid_size), bool)          # 击中
        self.sem = np.zeros((n_classes, grid_size, grid_size), bool)  # 击中格子的类别（sem_hit 用）

    def update(self, agent_pos, agent_heading, depth_views, sem_views, view_yaws, intrinsics)
    def mask(self) -> C
    def sem_map(self) -> sem
```

### 3.2 逻辑

1. 复用 `llm_grid_oracle_cache.py` 的反投影：整幅 Depth → 世界点云。高度范围沿用该文件的做法（**不用** `|y−floor_y|≤1 m` 这种会漏高处物体的规则；若该文件用楼层区间 `[floor, floor+h]`，照抄）。
2. 每个点：agent→点做 2D 射线采样写 `C`；终点写 `hit`；若有 `sem_views`，终点格子对应类别写 `sem[c]`。
3. 像素下采样 stride 由 3 个 episode 上 stride 1/2/4 的覆盖率对比决定，差异 > 2% 则减小。
4. `sem_hit` 的 mix 只替换 `hit` 为 True 的格子（不是整个 `C`）：hit 格子的 object 通道 ← `sem`；`C` 中非 hit 格子保持 LLM。这一区分要在报告中说明。

### 3.3 验证 `oracle_seen/test_coverage.py`（全部通过）

1. **合成测试**：假 Depth（前 3 m 墙、左 2 m 墙）落到正确格子。
2. **hit 精度**：3 个 episode，`hit` 格子落在 `gt.scene.level.v1` 占据格或其 8 邻域的比例 > 80%。
3. **覆盖召回率**：对 `gt.legacy` 中距 agent 实际轨迹 ≤ 2 m 的非零格，被 `C` 覆盖的比例 > 85%；被 `hit` 覆盖的比例单独报告。低于阈值 → 坐标/内参/高度范围有误，停下排查。
4. **sem 一致性**：`sem_hit` 组，`sem` 中非零类别与 `gt.scene` 同格类别的一致率 > 85%（Semantic 映射正确性）。
5. **单调性**：`C`、`hit` 随 step 单调不减。
6. **可视化**：5 个 episode，`C`/`hit`/`sem` 叠加 `P0` 的 png。

Phase 1 后**暂停汇报**（附 png 与四个比例）。

---

## 4. Phase 2 — scene GT 与混合

### 4.1 `oracle_seen/build_scene_gt.py`

生成 `gt.scene.level.v1`（object 通道；region 通道仅当全场景 region 可用时生成）。检查：GT path 1.5 m 内与 `gt.legacy` IoU > 0.9；非零格数倍数记录。

### 4.2 `oracle_seen/mixer.py`

```python
def mix_map(P_llm, P_route_gt, P_scene_gt, C, hit, sem, mode, obj_ch, reg_ch):
    out = P_llm.copy()
    if mode == "llm_full":  return out
    if mode == "gt_full":   return P_route_gt
    if mode == "sem_hit":
        out[obj_ch][:, hit] = sem[:, hit]                 # 只写 hit 格子，只写 object 通道
        return out
    if mode == "scene_seen":
        out[obj_ch][:, C] = P_scene_gt[obj_ch][:, C]
        if scene_region_available: out[reg_ch][:, C] = P_scene_gt[reg_ch][:, C]
        return out                                        # 否则 region 保持 LLM
    if mode == "gt_seen":   return np.where(C[None], P_route_gt, P_llm)
    if mode == "gt_unseen": return np.where(C[None], P_llm, P_route_gt)
```

`sem` 的取值语义与 `P0` object 通道一致（Phase 0 确认）。任何把路线 GT 写入 `sem_hit`/`scene_seen` region 通道的路径 → `raise`。

### 4.3 注入

`--oracle_mode {llm_full,sem_hit,scene_seen,gt_seen,gt_unseen,gt_full}`。每高层 step 决策前：读 pose/depth/sem → `acc.update` → `mix_map` → 原 Map Encoder。dtype/shape/device 与 `P0` 一致。

---

## 5. Phase 3 — 评测与日志

### 5.1 运行

6 次 rollout，同 seed。先 50-episode 子集（`subset50_ids.json`）跑通 6 组：`llm_full` 与原评测逐 episode 逐步一致；其余组 sanity 见 §6.5。再全量。

### 5.2 `steps.jsonl`

```json
{"ep":362,"step":3,"pos":[x,y,z],"heading":h,
 "cov_cells":412,"hit_cells":130,"sem_cells":41,
 "dist_to_gt_suffix":1.7,"on_route":true,
 "map_l1_written_vs_target":0.0,"action":"...","stopped":false}
```

`map_l1_written_vs_target`：该组实际写入区域（sem_hit → hit 格子；其余 → C）与应写入内容的 L1，应为 0。进度指针 `suffix_start + argmin(suffix_distances)`，只前移，不复用 `route_map_progress`。

### 5.3 `episodes.jsonl`

`success, spl, path_len, nav_error, n_steps, final_cov_frac, final_hit_frac, final_sem_cells, never_offroute, first_offroute_step, recovered`

---

## 6. Phase 4 — 报告 `reports/oracle_seen/summary.md`

### 6.1 主表

| mode | SR | SPL | NE | 配对 ΔSR vs llm_full [95% CI] | 配对 ΔSPL [95% CI] | 回收率 | mean cov / hit frac |
|---|---|---|---|---|---|---|---|

bootstrap 10000 次，episode 级。禁止 `gt_seen`/`gt_unseen` 相加分解 gap。

### 6.2 对照差（配对 + CI）

- 同格未见语义影响：`scene_seen − sem_hit`
- path-conditioned GT 性能影响：`gt_seen − scene_seen`
- 远端 gap：`gt_unseen − llm_full`

### 6.3 路线统计（文档 13 §6，每组）

`never_offroute_rate`、`first_offroute_step`、`recovery_rate`、按 `never_offroute` 分层 SR。

### 6.4 覆盖分析

`sem_hit` 增益按 `final_hit_frac` 分桶；增益与覆盖无关时记录。

### 6.5 sanity（全部通过才出结论）

1. 全量 `llm_full` SR/SPL 在 65.42/55.25 ± 0.3
2. 各组 `map_l1_written_vs_target` 全程 0
3. §3.3 六项通过
4. §4.1 检查通过
5. 6 次 rollout 的 checkpoint hash、metadata、seed、episode 集一致
6. `gt_full` 与外部参考 72 的差写入并解释（不作通过条件）
7. `sem_hit`/`scene_seen` 的 region 通道与 `P0` region 通道逐位相同（除非全场景 region 可用并已注明）

### 6.6 结论段

按 §0.2 给 GO / 讨论 / 分支 / 强 NO-GO，附三项对照差的解读。

---

## 7. 交付物

```text
oracle_seen/{coverage.py, test_coverage.py, build_scene_gt.py, mixer.py, eval.patch}
reports/oracle_seen/phase0_discovery.md
reports/oracle_seen/coverage_vis/*.png
reports/oracle_seen/{mode}/steps.jsonl, episodes.jsonl
reports/oracle_seen/summary.md
reports/oracle_seen/subset50_ids.json
caches/gt.scene.level.v1/...
```

## 8. 暂停点

Phase 0 → 暂停；Phase 1 验证 → 暂停；50-episode 6 组 → 暂停；全量 → 报告。任何"未确认"影响坐标、Semantic 映射、注入点时先问。
