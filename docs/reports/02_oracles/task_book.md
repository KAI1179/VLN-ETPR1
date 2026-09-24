# CLI 任务书 #2：O-进度 Oracle 构造与矛盾规则校准（零 API 费用）

日期：2026-09-24
前置：任务书 #1 已完成（报告 `docs/reports/01_mip_task1/task1_report.md`）。
对应：《研究提案：状态外置的 Agentic 导航》第 9.4 节（矛盾规则 v0）、第 10.1–10.2 节（oracle 与规则校准）。
本任务**不调用任何模型 API**，不需要 GPU，不需要启动仿真器（T2.6 除外，可选）。

---

## 0. 约定

```bash
export WORKDIR=/home/xukai/code/agentic-nav
export PY=$WORKDIR/MIP/envs/mip/bin/python
export OUT=$WORKDIR/oracles            # 本任务所有产物
export LOGDIR=$WORKDIR/logs/task2
# 数据（均取自任务 #1 报告 A5）
export CE_ORIG=/data/xukai/data/scene_datasets/datasets/R2R_VLNCE_v1-3_preprocessed   # 原版 R2R-CE
export FGR2R=$WORKDIR/Fine-Grained-R2R/data                                          # FGR2R_{train,val_seen,val_unseen}.json
export R2R_DISC=$WORKDIR/Fine-Grained-R2R/source/R2R-original                        # R2R_{train,val_seen,val_unseen}.json
export CONN=/data/xukai/VLN-GOAT/datasets/R2R/connectivity
export RAND100=$WORKDIR/MIP/splits/r2r/rand100
mkdir -p $OUT $LOGDIR
```

- 代码放在仓库 `tools/02_oracles/`，每个步骤一个脚本，可单独重跑；所有路径从上面的环境变量读，不写死。
- 坐标约定（任务 #1 B4 已确认）：`mp3d(x, y, z) = (hab_x, −hab_z, hab_y)`；**所有距离一律用水平距离**（habitat 的 x–z 平面），忽略竖直轴。
- 每步完成后在 `$OUT/task2_report.md` 追加：步骤号、命令、关键数字、PASS / FAIL，以及判据是否满足。

### T2.0 切换数据版本

把 `MIP/data/embodiedscore/datasets/vlnce/R2R_VLNCE_v1-3_preprocessed/` 下各 split 的软链从 `_xlmr` 改指向 `$CE_ORIG/<split>`（`rand100` 链接保持不变）。然后复跑任务 #1 的 A7-(1)（fake 智能体，episode 0），确认 exit 0 且 `summary.json` 与上次一致。

---

## T2.1 CE episode → R2R 指令序号 k

R2R-CE 没有 `instruction_id`，但 FGR2R 的 `new_instructions` 和 `chunk_view` 是按 R2R `instructions[k]` 的序号 k 组织的，所以需要先恢复 k。

- 对 split ∈ {train, val_unseen}：对每个 CE episode，在 `R2R_{split}.json` 中找到 `path_id == trajectory_id` 的条目，把它的 3 条 `instructions` 与 `instruction.instruction_text` 做匹配。
- 规范化：转小写、去首尾空白、连续空白合并为一个空格、去掉末尾的句号。先做精确匹配；匹配不上再退到字符级相似度（`difflib.SequenceMatcher` ratio 最大者，要求 ≥ 0.9），并记录用了退化匹配的条目。
- 输出：`$OUT/ce_instr_index_{split}.json` = `{episode_id: {"path_id": int, "k": int, "match": "exact"|"fuzzy", "ratio": float}}`
- 判据：val_unseen 1839 条中 exact + fuzzy 覆盖率 = 100%，fuzzy 占比写进报告；train 同理（预期 10819 条左右，以实际数为准）。

## T2.2 O-进度 oracle（子句分段）

对 split ∈ {rand100, val_unseen, train}，对每个 episode 生成：

```json
{
  "episode_id": 7,
  "path_id": 42,
  "k": 1,
  "clauses": [
    {"idx": 0, "text": "walk past the bed", "vp_range_1b": [1, 3],
     "ref_range_0b": [0, 2], "length_m": 4.87, "type": "move"}
  ],
  "ref_path_xz": [[x, z], ...],
  "cum_arclen_m": [0.0, ...],
  "max_single_point_offset_m": 0.03
}
```

构造规则：

1. 子句文本：`ast.literal_eval(new_instructions)[k]`，每个子句的 token 用空格拼接。
2. 区间：`chunk_view[k]` 的 1-based `[起, 止]` 减 1 得到 `reference_path` 的 0-based 索引区间（因为两者一一对应、同序）。
3. 单视点子句（起 == 止，段长为 0，多为转向或停止）：保留为独立子句，`length_m = 0`，它的"位置"定义为该视点对应的 `ref_path` 点。
4. `length_m`：区间内相邻 `ref_path` 点之间水平距离的累加。
5. `type`：按下列关键词规则分类，按顺序取**第一个命中**的类型：
   - `stop_at`：`stop | wait | halt`
   - `enter`：`enter | go into | walk into | into the`
   - `pass`：`pass | past | go by`
   - `turn`：含 `turn | veer | bear` 且不含 `walk | go | head | continue | proceed | move`
   - `move`：其余
6. `max_single_point_offset_m`：同序逐点水平误差的最大值（R2R 视点经 connectivity 转换后 vs `ref_path`），只用于记录。
7. 输出：`$OUT/oracle_progress_{split}.json`

判据：rand100 的 100 条全部生成；子句数分布（最小 / 中位 / 最大）写进报告；类型分布写进报告；随机抽 10 条，把子句文本与区间打印出来贴进报告供人工抽查。

## T2.3 "当前子句"函数与 GT 回放自检

实现 `current_clause(oracle_ep, agent_xz, prev_idx) -> idx`：

- 把智能体水平位置投影到 `ref_path` 折线上（取最近的线段投影点），得到投影弧长 s。
- idx = 满足 `cum_arclen[ref_range_0b[0]] ≤ s` 的最大子句序号；当多个子句覆盖同一点（单视点子句与相邻子句共用端点）时，取序号较大的那个。
- 单调约束：`idx = max(idx, prev_idx)`（与 oracle 语义一致：进度不回退）。另外单独输出一个不带单调约束的版本，用于统计。
- 偏离：到折线的水平距离 > 3.0 m 时返回 `(idx, offtrack=True)`。这就是 E0 中 O-偏离 的 v0 定义。

自检：用 `rand100_gt.json.gz` 的 `locations`（GT 动作逐步位置）回放 100 条 GT 轨迹，报告：

- 不带单调约束时，idx 序列单调不减的 episode 比例（目标 ≥ 95%）；
- 终点处 idx == 最后一个子句的比例（目标 100%）；
- 任一时刻 offtrack=True 的 episode 数（目标 0）。

不满足的逐条列出 episode_id 和原因。

## T2.4 子句预算统计（矛盾规则 R1 / R5 的参数）

在 **train** split 上（oracle 来自 T2.2，GT 来自 `$CE_ORIG/train/train_gt.json.gz`）：

- 每个 `type` 的 `length_m` 分布：n、P50、P90、P95、P99。
- 每个 `type` 对应的 GT 前进步数（forward 0.25 m 计 1 步；转向不计）分布：同上。
- 总路径长度（`info.geodesic_distance` 与 `ref_path` 水平弧长两者都报告）按子句数分组的中位数、P5、P95。
- 输出：`$OUT/clause_budget_train.json` 与报告中的 markdown 表。

## T2.5 规则误报率（GT 回放，val_unseen）

用 T2.4 得到的参数，在 val_unseen 的 GT 轨迹上回放下列规则。**GT 轨迹上的任何触发都算误报。**

| 规则 | 触发条件（v0） | 参数 |
|---|---|---|
| R1 预算超支 | 当前子句的已用水平路程 > `budget[type]`，已用路程从进入该子句时开始累计 | `budget[type]` 取 train 上该类型 `length_m` 的 P90 / P95 / P99 三档，分别报告 |
| R2 顺序违反 | GT 回放中不适用（没有 satisfy 调用），跳过，只在报告里标注 | — |
| R5 停止距离异常 | 在终点时，总水平路程 < a × median 或 > b × median，median 为 train 上同子句数的中位数 | (a, b) ∈ {(0.5, 2.0), (0.4, 2.5), (0.3, 3.0)} |
| R6 回环 | 同一子句内，智能体回到 ≥ 5 步之前经过的位置 1.0 m 以内 | 半径 ∈ {0.5, 1.0} m，间隔 ∈ {5, 10} 步 |

注意：R1 在 GT 上"当前子句"用 T2.3 的单调版本计算。

输出：每条规则、每档参数下的**按 episode 计误报率**（至少触发一次的 episode 比例）和**按步计触发率**。判据：每条规则至少有一档参数的 episode 误报率 < 5%；满足的最紧一档作为 v0 默认值，写进报告。R3 / R4 需要视觉验证器，本任务不做。

## T2.6（可选）R6 与 R1 在"已知失败"上的命中率

如果时间允许：MIP 仓库里若附带论文运行的 episode 日志（`scripts/`、`docs/` 或 release 附件中带位置信息的 `episode_*.jsonl`），用它们的轨迹做同样的回放，报告 R1 / R6 在成功与失败 episode 上的触发率差异。找不到这类日志就标 SKIP，**不要**为此调用任何模型。

---

## 交付物

- `tools/02_oracles/` 下的全部脚本
- `$OUT/ce_instr_index_{train,val_unseen}.json`
- `$OUT/oracle_progress_{rand100,val_unseen,train}.json`
- `$OUT/clause_budget_train.json`
- `$OUT/rule_calibration_val_unseen.json`
- `$OUT/task2_report.md`，结构如下：

```markdown
# Task 2 Report
## T2.0 数据切换：PASS/FAIL，A7-(1) 复跑结果
## T2.1 k 匹配：覆盖率、fuzzy 占比、异常条目
## T2.2 oracle：子句数分布、类型分布、10 条抽查
## T2.3 GT 回放自检：单调率、终点命中率、offtrack 数、不满足条目
## T2.4 预算表
## T2.5 误报率表与选定的 v0 参数
## T2.6（可选）
## 问题与需要人决定的事
```
