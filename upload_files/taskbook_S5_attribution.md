# Task Book：S5 — S4 结果归因（纯评测拆解 + 配对分析 + 对照）

分支 `exp/refiner`，基于 `a210fb3`。所有评测只读 checkpoint，不训练。
先答两个确认问题，再按 D1 → D2 → D3 顺序执行。每个 D 一份汇报，D1 做完先停一次。

---

## 0. 先确认（写在汇报开头）

1. S4 训练与评测使用的 region 合成规则：是 mask 版（observed 用 refiner、unseen 保 P0）还是全图替换版？给出对应代码行。
2. 训练日志里"每 200 iter 8–16 秒"与速度探针的 19.3 秒/iter 相差两个数量级。查明打印的是什么单位（单 iter？200 iter？只含前向？），报告真实的秒/iter 和 30k 总墙钟时间。

---

## D1：地图来源拆解（S4 iter15k 策略，val_unseen）

固定 policy checkpoint = `s4_try5_refiner` 的 `ckpt.iter15000.pth`。只改地图来源，其余评测配置与 S4 评测完全一致。

新增评测开关 `MODEL.MAP_ENCODER.eval_map_source`，取值：

| 值 | 地图 | 说明 |
|---|---|---|
| `refiner` | 现状 | 已有 64.76，不重跑，直接引用 |
| `p0` | 静态 P0，refiner 不加载 | 同 clean Try5 的地图路径 |
| `gt_seen` | observed cell 用 GT，unseen 保 P0 | 用 S4 同一套 OnlineEvidence 取 observed；GT 从 GT NPZ 读；object 与 region 都按 mask 合成 |
| `gt_full` | 整图 GT | 与 Try5 的 GT-map 评测路径一致 |

实现要求：`gt_seen` 与 `refiner` 共用同一 evidence 更新代码，只替换合成时的来源张量；不要另写投影。

汇报表：

| 来源 | SR | SPL | NE | OSR | 平均步数 | 平均路径长度 |
|---|---|---|---|---|---|---|
| p0 | | | | | | |
| refiner | 64.76 | 53.48 | 3.983 | 70.64 | | |
| gt_seen | | | | | | |
| gt_full | | | | | | |

外加一行 Try5 历史最佳（注明 checkpoint 路径与 seed）作参考。

**D1 汇报后停下，等下一步指令。**

---

## D2：episode 级配对分析

对比对象：`refiner`（S4-15k）与 `p0`（同一策略，D1 已得）。两次评测都保存每个 episode 的 `success / spl / steps / path_length / final_dist`（若 S4 评测脚本没有逐 episode 落盘，加一个 `--dump-episodes <path>` 输出 jsonl）。

统计：

1. 四格表：两者都成功 / 都失败 / 只 refiner 成功 / 只 p0 成功，各计数
2. 对每个 episode 计算 refiner 改动量 `delta = sum(|refined − P0| over observed cells)`，取末步值（评测时顺便落盘，每 episode 一个标量）
3. 把 episode 按 delta 分四分位，报每个分位内"只 p0 成功"和"只 refiner 成功"的比例
4. 报 delta 与 steps 的 Spearman 相关

汇报：四格表 + 分位表 + 相关系数，各一段一两句话的解读。

---

## D3：同 seed 对照（try5-control）

`scripts/refiner/run_s4_control.sh`（S4 已备好），GPU 0–3，seed 100，与 `s4_try5_refiner` 完全同配置，`refiner_ckpt` 为空。跑到 15k 即可（每 3k 存 checkpoint）。

评测 9k / 13k / 15k 三个点的 val_unseen，与 `s4_try5_refiner` 同 iter 并排：

| iter | control SR | refiner SR | control SPL | refiner SPL |
|---|---|---|---|---|
| 9k | | 61.94 | | 50.95 |
| 13k | | 64.00 | | 54.72 |
| 15k | | 64.76 | | 53.48 |

D3 可以在 D1 完成、等待指令期间启动，不冲突（D1 用单卡评测）。

---

## 不做

不改训练代码、不改 refiner、不改合成规则、不做停止/拓扑排序分析、不延长 S4 训练。D1–D3 之外的任何分析先不做。
