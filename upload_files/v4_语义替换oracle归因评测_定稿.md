# 14 v4（定稿）｜固定策略下的语义替换 oracle 归因评测

日期：2026-08-26
执行方式：Claude Code CLI，本地 Try5 仓库

## 执行约定

本文档只规定**不变量、验收标准、决策规则**。凡涉及通道数、数组布局、投影函数、sensor 参数、缓存格式等仓库事实，**一律以现有代码（尤其 `llm_grid_oracle_cache.py`）为准，CLI 自行决定，不必回报**。只有当某个不变量在仓库中无法满足时才暂停询问。本文档中任何具体数字或代码若与仓库冲突，以仓库为准。

---

## 0. 问题、术语、结论范围

**问题**：在固定的 LLM-Try5 策略下，把 agent 亲眼看过的格子替换为真实语义，SR 能到多少？

**术语**：结果称"固定策略下的语义替换 oracle"。不是理论上界；冻结策略未见过混合地图，分布偏移压低所有替换组。

**结论范围**：本评测只能肯定或否定**"直接纠正已见区域"**这一条路线。强 NO-GO 不否定路线门、语义筛选、未见区域推断等其他方向。

---

## 1. 六组配置

| 组 | 已覆盖区域写入 | 未覆盖 | 用途 |
|---|---|---|---|
| `llm_full` | — | LLM | 基线 |
| `sem_hit` | **加性写入**：Semantic sensor 真实 hit 像素对应的类别置 1，object 与 region 通道都写，其余值保持 LLM | LLM | **主诊断** |
| `sem_hit_replace` | hit 格子全向量替换为 sensor 观测（未命中类别清零） | LLM | 激进对照，可选 |
| `scene_seen` | 全场景 Semantic GT raster（object；region 仅当全场景 region 可用，否则 region 保持 LLM） | LLM | 对照 |
| `gt_seen` | 路线裁剪 GT `gt.legacy.r1p5.direction5.v1` 全通道 | LLM | path-conditioned 对照 |
| `gt_unseen` | LLM | 路线 GT | 远端 gap |
| `gt_full` | 路线 GT | 路线 GT | 同 checkpoint 参考，回收率分母 |

`sem_hit_replace` 可选；不跑则 6 次 rollout，跑则 7 次。

---

## 2. 不变量（必须成立，违反即暂停）

**I1 策略固定**：所有组使用产生 65.42/55.25 的同一 checkpoint 与 metadata；只混合 grid，direction5 等输入保持 LLM 版；同 seed、同 episode 集、同 endpoint 逻辑。

**I2 覆盖来源**：覆盖 `C` 与击中 `hit` 只由该组自身 rollout 的真实 Depth + 真实 sensor position/rotation 反投影决定；不用 GT path 预计算；不跨组复用；Semantic sensor 不参与判断覆盖。

**I3 hit 定义**：无效深度、达到最大量程的深度、超出现有投影代码高度范围的点，**不记为 hit**（可记为 C 中的射线经过格，按现有代码约定）。

**I4 sem_hit 写入规则**：只对 `hit` 格子写；只把该格 sensor 观测到的类别置为 `P0` 中"存在"对应的值；其他类别、非 hit 格子、所有未观测通道保持 LLM 原值。object 与 region 按仓库现有划分**分开存储、分开写入**，维度以 `llm_grid_oracle_cache.py` 为准。

**I5 region 通道不得来自路线 GT**：`sem_hit`/`sem_hit_replace`/`scene_seen` 的 region 通道来源只能是 Semantic sensor、全场景 region 标注、或 LLM 原值；代码中任何写入路线 GT 的路径 → `raise`。

**I6 坐标**：楼层局部、世界轴对齐；`level_origin_xz = episode_start_xz − cache.start_position`；`grid_rc = (world_xz − level_origin_xz) / cell`；heading/rotation 只用于射线方向。行列与 x/z 的对应照抄现有代码。

**I7 时序**：每高层 step 决策前完成 `update → mix`；第 0 step 已含起点观测。

**I8 隔离**：不改权重、训练代码、默认配置；`--oracle_mode` 不给时逐位等于原评测。原 bug 只记录。

---

## 3. 验收标准（全部通过才出结论）

**V1 覆盖召回率**（`test_coverage.py`）
- reference：stride=1 完整 Depth 投影（真实 sensor pose）得到的 `C_ref`/`hit_ref`；
- 实际使用的 stride 下，`C` 对 `C_ref` 的 recall ≥ 98%，`hit` 对 `hit_ref` 的 recall ≥ 98%；不满足则减小 stride。在 3 个 episode 上验证。

**V2 hit 精度**：3 个 episode，`hit` 格子落在 `gt.scene` 占据格或其 8 邻域的比例 ≥ 80%。

**V3 sem 一致性**：`sem_hit` 组，hit 格子写入的类别与 `gt.scene` 同格类别一致率 ≥ 85%。

**V4 scene GT 包含性**：`gt.legacy` 的正语义格子（object、region 分别统计）被 `gt.scene` 同格同类包含的比例 ≥ 95%。不用 IoU。

**V5 写入正确性**：每 step 记录该组"实际写入区域"与"应写入内容"的 L1，全程为 0；`sem_hit` 中非 hit 格子及未观测类别与 `P0` 逐位相同。

**V6 基线复现**：全量 `llm_full` SR/SPL 在 65.42/55.25 ± 0.3。50-episode 阶段只要求与原评测逐 episode 逐步一致。

**V7 单调性**：`C`、`hit` 随 step 单调不减。

**V8 一致性**：所有 rollout 的 checkpoint hash、metadata、seed、episode 集相同，写入报告头。

**V9 可视化**：5 个 episode 的 `C`/`hit`/写入内容叠加 `P0` 的 png，人工核对。

---

## 4. 统计

- 每组报 SR / SPL / NE；相对 `llm_full` 的**逐 episode 配对** ΔSR、ΔSPL，bootstrap 10000 次 95% CI；每组绝对 SR 的 bootstrap 95% CI。
- 回收率 `(SR_sem_hit − SR_llm_full) / (SR_gt_full − SR_llm_full)`，分母同 checkpoint；约 72 只作外部参考。
- 对照差（配对 + CI，不称"泄漏"）：`scene_seen − sem_hit`（同格未见语义影响）、`gt_seen − scene_seen`（path-conditioned GT 影响）、`gt_unseen − llm_full`（远端 gap）、`sem_hit_replace − sem_hit`（若跑）。
- 禁止 `gt_seen`/`gt_unseen` 相加分解。
- 路线统计（每组）：`never_offroute_rate`、`first_offroute_step`、`recovery_rate`、按 `never_offroute` 分层 SR。进度指针 `suffix_start + argmin(suffix_distances)` 只前移，不复用 `route_map_progress`。
- 覆盖分析：`sem_hit` 增益按 `final_hit_frac` 分桶。

---

## 5. 决策规则

以 `sem_hit` 为主：

```text
GO      ：SR(sem_hit) ≥ 69.0 且配对 ΔSR CI 下界 > 0
讨论    ：67.6 ≤ SR(sem_hit) < 69.0
分支    ：SR(sem_hit) < 67.6，但 scene_seen 或 gt_seen 的绝对 SR CI 上界 ≥ 67.6
          → 不停线；结论改为"关键在语义筛选/路线门/未见推断"，另议
强 NO-GO：sem_hit、scene_seen、gt_seen 三者绝对 SR 的 CI 上界均 < 67.6
          → 只否定"直接纠正已见区域"路线
```

---

## 6. 日志与交付

`steps.jsonl`：`ep, step, pos, heading, cov_cells, hit_cells, written_cells, dist_to_gt_suffix, on_route, map_l1_written_vs_target, action, stopped`
`episodes.jsonl`：`success, spl, path_len, nav_error, n_steps, final_cov_frac, final_hit_frac, never_offroute, first_offroute_step, recovered`

交付：
```text
oracle_seen/{coverage.py, test_coverage.py, build_scene_gt.py, mixer.py, eval.patch}
reports/oracle_seen/phase0_discovery.md        # 仓库事实清单，仅记录，不等待确认
reports/oracle_seen/coverage_vis/*.png
reports/oracle_seen/{mode}/{steps,episodes}.jsonl
reports/oracle_seen/summary.md                 # 主表、对照差、路线统计、V1–V9 结果、结论
reports/oracle_seen/subset50_ids.json
caches/gt.scene.level.v1/
```

## 7. 执行顺序

1. 勘察并写 `phase0_discovery.md`（不暂停，除非不变量无法满足）
2. 覆盖模块 + V1/V2/V3/V7/V9 → **暂停一次**，附 png 与比例
3. scene GT + V4；mixer + 50-episode 全组 + V5、逐步一致性
4. 全量 → V6、V8 → `summary.md` → 交付

只有第 2 步一个强制暂停点。
