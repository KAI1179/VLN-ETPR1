# D7-d：四 rank 重评结果汇总

## 核验

三组均为 R2R val_unseen、control、4 rank × 4 env、150 token / 15 高层步 / 5000 低层步。每组 stats 覆盖 1839 个唯一 episode，rank 数量为 460 / 460 / 460 / 459。

baseline 有 1839 条完整高层轨迹；Try5 / KD FULL 各 1833 条完整轨迹，其余 6 条为 U 未运行：308、349、455、804、1318、1331。U 不进 C/D/E。

baseline 与旧 d7_base_control_val_unseen 的 1839 条 success / oracle_success / steps_taken / distance_to_goal 逐条全等；SR 变化 0.00 pp。

## 整体指标

百分比指标单位为 %；NE / PL 为米。主表按 1839 条计分，U 计零。

| 模型 | SR | SPL | OSR | nDTW | NE | PL | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline 25000 | 63.02 | 53.31 | 68.52 | 64.21 | 4.13 | 13.83 | 99.84 |
| Try5 无 KD 28000 | 65.04 | 55.01 | 70.36 | 64.06 | 3.99 | 14.03 | 99.48 |
| KD FULL 27000 | 67.26 | 56.48 | 72.10 | 65.41 | 3.78 | 13.54 | 94.71 |

| 相对 baseline | SR 差（pp） | 场景 bootstrap 95% CI |
|---|---:|---|
| Try5 无 KD | +2.01 | [−0.04, +4.72] |
| KD FULL | +4.24 | [+2.38, +6.68] |

## 终止标签（含成功样本）

| 模型 | stop | budget | no_candidate | U |
|---|---:|---:|---:|---:|
| baseline | 1564 | 275 | 0 | 0 |
| Try5 | 1566 | 267 | 0 | 6 |
| KD FULL | 1607 | 226 | 0 | 6 |

同时触发 budget 与 stop_requested 的次数为 baseline 28 / Try5 23 / KD FULL 21。统一标签优先级为 budget → no_candidate → stop，三个原始布尔条件均保存在轨迹。

## baseline C/D 的救回与反向丢失

仅比较双方实际运行的 1833 条；C=stop 且从未到目标范围，D=budget 且从未到目标范围，E=no_candidate 且从未到目标范围。

| 模型 | C→A | A→C | C 救回率（n=381） | D→A | A→D | D 救回率（n=193） |
|---|---:|---:|---:|---:|---:|---:|
| Try5 | 107 | 83 | 28.08% | 46 | 42 | 23.83% |
| KD FULL | 119 | 79 | 31.23% | 53 | 31 | 27.46% |

同次新评测中，按高层末步索引构建的代理分类与真实标签完全一致；历史 D7-b 分类对新标签的矩阵另列于分报告，包含 rank 重评带来的行为变化。

## 首次偏离：3m 阈值

以下每组取 baseline / 对应模型至少一方失败、且双方均运行的 episode。分岔比例的分母为该模型实际发生偏离的 episode；“从未偏离”的分母为整个子集。步索引从 0 开始。

| 对比 | 模型 | n | 首次偏离 P25/P50/P75 | 偏离发生在分岔节点 | 从未偏离 |
|---|---|---:|---|---:|---:|
| baseline vs Try5 | baseline | 845 | 1 / 3 / 6 | 77.91% | 49.11% |
| baseline vs Try5 | Try5 | 845 | 1 / 3 / 5 | 78.64% | 49.59% |
| baseline vs KD FULL | baseline | 826 | 1 / 3 / 6 | 77.52% | 48.31% |
| baseline vs KD FULL | KD FULL | 826 | 1 / 3 / 5 | 78.48% | 52.18% |

分岔基于原始 R2R 连通图的坐标匹配：连续位置映射到最近原始节点（≤3m），排除 GT 路径上的邻居与来向邻居后至少两个可达邻居。该映射检查在本次要求的分析子集上全部通过；坐标匹配规则不是动态拓扑节点 ID 与原始 viewpoint ID 的直接等同。

## 完整文件

- `reports/d7d_try5_control.md` / `.json`：Try5 vs baseline。
- `reports/d7d_kd_control.md` / `.json`：KD FULL vs baseline。

两份均含 D7 指标/原三层分层、ABCDE 转移、U、代理混淆、C 类分位数、McNemar、2m / 3m 分岔统计及配对场景 bootstrap CI。JSON 包含逐 episode 分类和偏离记录，MD 不作机制解释。
