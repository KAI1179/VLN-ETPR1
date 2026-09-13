# S5-D1 地图来源归因结果

## 评测设置

固定策略 checkpoint：`data/logs/checkpoints/s4_try5_refiner/ckpt.iter15000.pth`；val_unseen，共 1839 episodes。region 与 object 使用同一 mask 合成规则：observed 区域采用替换来源，unseen 区域保留 P0。实现位置为 `vlnce_baselines/ss_trainer_ETP_PriorGT.py:1220-1227`。

## 结果

| 地图来源 | SR | SPL | NE | OSR | 平均步数 | 平均路径长度 |
|---|---:|---:|---:|---:|---:|---:|
| p0 | 64.27 | 53.97 | 4.007 | 69.28 | 96.42 | 13.66 |
| refiner | 64.76 | 53.48 | 3.983 | 70.64 | — | — |
| gt_seen | **65.31** | **54.39** | **3.982** | **70.26** | 96.85 | 13.77 |
| gt_full | 64.60 | 54.28 | 3.969 | 68.90 | 94.13 | 13.39 |

refiner 行沿用既有 S4 iter15k 评测记录；gt_full 结果来自 `eval_iter15000_gt_full.log`。

## 归因

- p0 → gt_seen：SR 提升 1.04 个百分点，说明已观测区域的地图错误会影响导航，但可回收的收益有限。
- p0 → gt_full：SR 没有提升，反而为 64.60；因此“把整张图替换成 GT 就必然得到约 72%”并不成立。约 72% 很可能来自不同的 Try5 评测配置、checkpoint 或训练条件，不能直接作为本次策略的地图上界。
- refiner 与 gt_seen 的 SR 差距为 0.55 个百分点，说明 Refiner 已接近已观测 GT 的导航效果；主要差距不在视觉纠错是否发生，而在策略 checkpoint/评测条件及地图输入对决策的具体影响。
- gt_full 的 OSR 下降到 68.90，说明全图 GT 可能改变候选排序并损害当前策略的分布适配；地图更准确不等于当前策略能正确利用它。

## 训练耗时核对

训练日志最后记录到 iter 30000：`2026-09-10 13:14:23`；从首次初始化 `2026-09-05 20:38:43` 到结束约 4 天 16 小时 36 分。日志中的 `Train timing iterations=200 total_seconds=...` 是最近 200 个训练 iter 的总耗时，`seconds_per_iteration` 才是单 iter，末期约 8.58 秒/iter（其他窗口 8–16 秒/iter）。

## D1 结论

当前数据不支持将剩余约 7 个 SR 点归因于“未见区域的 LLM 地图错误”。已见区域 GT 只带来约 1 点 SR，且全图 GT 未复现 72%。下一步应先核对 72% 结果的 checkpoint、split、seed、地图 namespace 和评测配置，再进行 D2 episode 配对分析。
