# 14 v2｜oracle-seen-only 零训练归因评测 Task Book

日期：2026-08-26
执行方式：Claude Code CLI，在本地 Try5 仓库中执行
前置文档：12（融合方案）、13（六项风险回应）、14 v1 及 CLI 六条反馈
本版相对 v1 的变更：坐标系改为楼层局部世界轴对齐；取消 gt_full≈72 sanity；新增 `scene_seen` 组作为主诊断；覆盖改为完整视锥反投影并报告双版本；统计改为逐 episode 配对差 + bootstrap CI。

---

## 0. 目标与决策规则

用**产生 65.42 / 55.25 的那一个 LLM-Try5 checkpoint**，不训练任何模块，在 val_unseen 上跑五组地图配置，回答：

> 若 agent 亲眼看过的区域被完美修正到**场景语义真值**（不含路线信息），其余保持 LLM 先验，SR 能到多少？

### 0.1 五组配置

| 组名 | 已覆盖区域 | 未覆盖区域 | 作用 |
|---|---|---|---|
| `llm_full` | LLM | LLM | 复现基线 65.42 / 55.25 |
| `scene_seen` | **全场景 Semantic GT** | LLM | **主诊断**：完美视觉模块的真实上界 |
| `gt_seen` | 路线裁剪 GT（`gt.legacy.r1p5.direction5.v1`） | LLM | 泄漏对照：与 scene_seen 的差 = 路线信息泄漏量 |
| `gt_unseen` | LLM | 路线裁剪 GT | 量化远端 gap |
| `gt_full` | 路线裁剪 GT | 路线裁剪 GT | 同 checkpoint 上的参考上界；回收率分母 |

所有组：同 checkpoint、同 metadata、同评测集、同 endpoint 逻辑、同 seed；**只混合 grid**，direction5 等其他输入保持 LLM 版本不变。

### 0.2 覆盖双版本

每组跑两个覆盖版本（见 §3）：

- `cov_frustum`：完整视锥反投影（主版本）
- `cov_conserv`：保守版本（丢弃边缘/无效/跨层射线）

### 0.3 决策规则

以 `scene_seen`（`cov_frustum`）相对 `llm_full` 的**逐 episode 配对 ΔSR** 及 bootstrap 95% CI 判定：

```text
SR(scene_seen) ≥ 69.0 且 ΔSR 的 CI 下界 > 0      → GO
67.6 ≤ SR(scene_seen) < 69.0                      → 讨论，压缩范围
SR(scene_seen) < 67.6 在 cov_frustum 和 cov_conserv 两个版本下均成立 → NO-GO
```

回收率：`(SR_scene_seen − SR_llm_full) / (SR_gt_full − SR_llm_full)`，分母用**同 checkpoint** 的 `gt_full`。约 72 的 GT 专用 checkpoint 数字只在报告中作外部参考，不参与计算。

泄漏量：`SR_gt_seen − SR_scene_seen`（配对），单独报告。

---

## 1. 硬约束

1. 不修改任何模型权重、训练代码、配置默认值。改动放在 `oracle_seen/` 新目录或以 `--oracle_mode` 注入的评测路径；不给参数时行为与原评测逐位一致。
2. 覆盖来自**该组自己 rollout 出的轨迹**，每组每覆盖版本独立累积；禁止用 GT path 预计算覆盖，禁止跨组复用轨迹。
3. 覆盖只来自几何：真实 Depth + 真实 pose + 相机内参反投影。不使用 Semantic sensor 决定覆盖；Semantic GT 只在 `scene_seen` 中作为**被替换进去的内容**，不参与判断哪些格子被覆盖。
4. `llm_full` 必须复现 65.42 / 55.25（±0.3），否则先排查注入路径。
5. 发现原代码 bug（含文档 13 的 `route_map_progress` argmin 问题）只记录，不修。

---

## 2. Phase 0 — 仓库勘察（写入 `reports/oracle_seen/phase0_discovery.md`）

不确定的项写"未确认"，不猜。

**地图与坐标**
- [ ] 产生 65.42 的 checkpoint 路径、hash、metadata（含使用的 LLM grid 缓存 key、direction5 来源）
- [ ] LLM grid `P0` 存储位置、格式、shape `[Ch,100,100]`、通道语义（region/object 分层方式、mpcat40 索引）
- [ ] 路线裁剪 GT `gt.legacy.r1p5.direction5.v1` 的位置、shape、通道顺序，与 `P0` 一致性
- [ ] 坐标定义确认：地图为楼层局部、世界轴对齐；
  ```text
  level_origin_xz = episode_start_xz − cache.start_position
  grid_rc = (world_xz − level_origin_xz) / 0.5
  ```
  确认 `cache.start_position` 的来源与含义、row/col 对应 x/z 的哪一个、是否有翻转
- [ ] 楼层判定：`P0`/GT 覆盖哪一层；多层 episode 如何处理；起点高度阈值

**全场景 Semantic GT（scene_seen 需要）**
- [ ] MP3D 场景语义源（house file / semantic mesh / 现有缓存）能否直接生成"该楼层全场景、不依赖 path 的 grid"，通道格式与 `P0` 相同
- [ ] 若需新建：复用生成 `gt.legacy` 的代码，去掉 1.5 m trajectory 裁剪，保留同一楼层过滤；输出为新缓存 key `gt.scene.level.v1`
- [ ] region 通道：全场景 region 标注是否可用；若不可用，`scene_seen` 的 region 通道退化为使用路线裁剪 GT 并在报告中注明

**观测与注入点**
- [ ] 每高层 step 可得：agent 位置、heading、12-view Depth、各 view 相对 yaw、相机内参（fov / 宽高 / near-far / 归一化方式）
- [ ] `llm_grid_oracle_cache.py` 中的反投影逻辑：函数签名、输入输出坐标系；能否直接复用
- [ ] `P0` 送入 Map Encoder 的确切代码位置（注入点）；地图混合必须在每个高层 step 决策之前完成
- [ ] endpoint / STOP 随机性与固定 seed 方法

Phase 0 完成后**暂停汇报**，确认坐标定义、注入点、scene GT 生成方案无误再继续。

---

## 3. Phase 1 — 几何覆盖模块 `oracle_seen/coverage.py`

### 3.1 接口

```python
class CoverageAccumulator:
    def __init__(self, level_origin_xz, cell_m=0.5, grid_size=100,
                 floor_y=None, floor_tol_m=1.0, variant="frustum"):
        self.C = np.zeros((grid_size, grid_size), bool)     # 累计覆盖（射线经过或击中）
        self.hit = np.zeros((grid_size, grid_size), bool)   # 累计击中

    def world_to_grid(self, xz_world) -> (row, col)
    def update(self, agent_pos, agent_heading, depth_views, view_yaws, intrinsics) -> None
    def mask(self) -> np.ndarray
```

### 3.2 `update`：完整视锥反投影

对每个 view：

1. 用相机内参把整幅 Depth 反投影为相机坐标点云（复用 `llm_grid_oracle_cache.py` 的逻辑，不要另写线性像素角）。
2. 用 `agent_heading + view_yaw` 和 agent 位置变换到世界坐标。heading **只**用于此处，不用于地图旋转。
3. 高度过滤：保留 `|y − floor_y| ≤ floor_tol_m` 的点（避免地板/天花板/跨层）。
4. 每个保留点：从 agent 位置到该点做 2D 射线采样（0.25 m 步长），经过格子写 `C`，终点格子写 `C` 和 `hit`。
5. 点数多时可对像素做均匀下采样（如 stride 4），但下采样后每 view 至少保留 2000 点；先在 3 个 episode 上比较 stride 1/2/4 的覆盖率差异，差异 > 2% 则减小 stride。

### 3.3 两个覆盖版本

- `frustum`：上述完整逻辑；深度值在有效范围 `[d_min, d_max]` 内即使用。
- `conserv`：在 frustum 基础上额外丢弃：图像边缘 10% 列、深度 > 8 m、以及高度过滤 tol 收紧到 0.5 m。

### 3.4 校验（`oracle_seen/test_coverage.py`，必须通过）

1. **合成测试**：构造一个 agent 前方 3 m 有墙、左侧 2 m 有墙的假 Depth，检查 `C`/`hit` 落在正确格子。
2. **实景一致性**：在 3 个 episode 上，把 frustum 覆盖的 `hit` 格子与 `gt.scene.level.v1` 中的占据格子做对比，`hit` 格子中落在场景占据/近墙格子的比例应 > 80%；低于此说明坐标或内参有误。
3. **单调性**：`C` 随 step 单调不减。
4. **可视化**：对 5 个 episode 输出 `C` 叠加在 `P0` 上的 png，人工核对方向是否正确。

---

## 4. Phase 2 — 场景语义 GT 与地图混合

### 4.1 `oracle_seen/build_scene_gt.py`

生成 `gt.scene.level.v1`：与 `gt.legacy.r1p5` 同格式、同坐标定义、同通道顺序，但**不做 trajectory 裁剪**，只做起点楼层过滤。输出后做两项检查：

- 在 GT path 1.5 m 范围内，`gt.scene` 与 `gt.legacy` 应几乎一致（IoU > 0.9）；
- `gt.scene` 的非零格数应显著多于 `gt.legacy`（记录倍数）。

### 4.2 `oracle_seen/mixer.py`

```python
def mix_map(P_llm, P_route_gt, P_scene_gt, C, mode):
    if mode == "llm_full":   return P_llm
    if mode == "gt_full":    return P_route_gt
    if mode == "scene_seen": return np.where(C[None], P_scene_gt, P_llm)
    if mode == "gt_seen":    return np.where(C[None], P_route_gt, P_llm)
    if mode == "gt_unseen":  return np.where(C[None], P_llm, P_route_gt)
```

若 Phase 0 确认 scene GT 的 region 通道不可用，`scene_seen` 的 region 通道单独用 `P_route_gt`，并在日志中标记。

### 4.3 注入

评测脚本新增参数：

```text
--oracle_mode {llm_full,scene_seen,gt_seen,gt_unseen,gt_full}
--cov_variant {frustum,conserv}
```

每高层 step、决策前：

```text
读取 pose / depth → acc.update(...) → P_t = mix_map(..., acc.mask(), mode) → 原 Map Encoder → 原 Try5
```

`P_t` 的 dtype/shape/device 与原 `P0` 一致；混合在 CPU numpy 完成再转 tensor。第 0 step 已含起点第一次观测。

---

## 5. Phase 3 — 评测与日志

### 5.1 运行矩阵

`llm_full` 与覆盖无关，只跑一次；其余四组 × 两个覆盖版本 = 8 次，共 9 次 rollout。

先用固定 50-episode 子集跑通全部 9 次（保存 `subset50_ids.json`），确认 `llm_full` 复现，再跑全量 val_unseen。

### 5.2 每 step 记录 `steps.jsonl`

```json
{"ep": 362, "step": 3, "pos": [x,y,z], "heading": h,
 "cov_cells": 412, "cov_frac": 0.041, "hit_cells": 130,
 "dist_to_gt_suffix": 1.7, "on_route": true,
 "map_l1_seen_vs_target": 0.0, "map_l1_unseen_vs_target": 0.23,
 "action": "...", "stopped": false}
```

`map_l1_seen_vs_target` 中的 target 是该组应写入已覆盖区域的地图（scene_seen → scene GT；gt_seen → route GT），用于 sanity。

`dist_to_gt_suffix`：到**剩余** GT path 的最近距离；进度指针 `suffix_start + argmin(suffix_distances)`，只允许前移。不复用原 `route_map_progress`。

### 5.3 每 episode 记录 `episodes.jsonl`

`success, spl, path_len, nav_error, n_steps, final_cov_frac, final_hit_frac, never_offroute, first_offroute_step, recovered, region_fallback(bool)`

---

## 6. Phase 4 — 统计与报告 `reports/oracle_seen/summary.md`

### 6.1 主表（每组 × 覆盖版本一行）

| mode | cov | SR | SPL | NE | 配对 ΔSR vs llm_full [95% CI] | 配对 ΔSPL [95% CI] | 回收率 | mean cov_frac |
|---|---|---|---|---|---|---|---|---|

配对差：同一 episode 在两组的 success 之差取均值；CI 用 episode 级 bootstrap（10000 次）。**不要**把 `gt_seen` 与 `gt_unseen` 的差相加去分解 gap，它们的覆盖来自不同轨迹。

### 6.2 泄漏与远端

- 泄漏量：`ΔSR(gt_seen − scene_seen)`，配对 + CI
- 远端 gap：`ΔSR(gt_unseen − llm_full)`，配对 + CI
- 覆盖版本差：`ΔSR(frustum − conserv)`，每组

### 6.3 路线统计（文档 13 §6，每组报告）

`never_offroute_rate`、`first_offroute_step` 分布、`recovery_rate`、按 `never_offroute` 分层的 SR。重点看 `scene_seen` 相对 `llm_full` 的增益来自"不走错"还是"走错后恢复"。

### 6.4 覆盖分析

`final_cov_frac` 分布；`scene_seen` 增益按 cov_frac 分桶；若增益与覆盖无关，记录并提示可能来自起点附近少量格子。

### 6.5 sanity（全部通过才出结论）

1. `llm_full` SR/SPL 在 65.42/55.25 ± 0.3
2. 各组 `map_l1_seen_vs_target` 全程为 0；`gt_unseen` 的 `map_l1_unseen_vs_target` 为 0
3. §3.4 四项覆盖校验通过
4. §4.1 两项 scene GT 检查通过
5. 9 次 rollout 的 checkpoint hash、metadata、seed、episode 集一致，写入报告头
6. `gt_full`（同 checkpoint）结果与外部参考 72 的差异写入报告并解释（不作为通过条件）

### 6.6 结论段

按 §0.3 给 GO / 讨论 / NO-GO；一句话说明泄漏量和远端 gap 的含义。

---

## 7. 交付物

```text
oracle_seen/coverage.py
oracle_seen/test_coverage.py
oracle_seen/build_scene_gt.py
oracle_seen/mixer.py
oracle_seen/eval.patch                     # 评测入口的 --oracle_mode/--cov_variant 补丁
reports/oracle_seen/phase0_discovery.md
reports/oracle_seen/coverage_vis/*.png
reports/oracle_seen/{mode}_{cov}/steps.jsonl, episodes.jsonl
reports/oracle_seen/summary.md
reports/oracle_seen/subset50_ids.json
caches/gt.scene.level.v1/...
```

---

## 8. 执行顺序与暂停点

1. Phase 0 → **暂停汇报**（坐标、注入点、scene GT 方案）
2. Phase 1 覆盖模块 + §3.4 校验 → **暂停汇报**（附可视化 png）
3. Phase 2 scene GT + mixer + 50-episode 9 次 rollout → **暂停汇报**（重点 sanity 1、2）
4. 全量 9 次 → Phase 4 报告 → 交付

任何"未确认"项影响坐标系、注入点或 scene GT 生成时，停下来问，不按假设推进。
