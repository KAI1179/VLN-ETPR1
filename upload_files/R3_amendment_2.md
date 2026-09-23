# 任务书 R3 修正案 2：解决 A1′-1 与 A1′-2 的矛盾

A1′-1（考古：旧 PriorGTPolicy 为双向 fusion）与 A1′-2（结构：与当前 try5 零 missing / 零 unexpected / 零 shape 不匹配）互相矛盾。若 checkpoint 真由双向 fusion 训练，其 map←graph attention 权重应作为 unexpected key 出现。A1′-1 的隐含前提——`82a25d5` 时刻的代码即训练该 checkpoint 的代码——没有证据（checkpoint config 连 `cache_namespace` 都没有，早于 `82a25d5`；`pretrained_path` 名为 `try-5-r1p5_*`，说明"try 5"实验名先于仓库中的 `PriorGTTry5Policy` 类）。

按下列顺序执行，**不得在 C2 完成前停止**。

## C1. 参数集合对比（只读，无 GPU）

1. 在 `82a25d5` 上 `git show 82a25d5:vlnce_baselines/models/etp_prior_gt/map_fusion.py`，列出 `BidirectionalMapTokenFusion.__init__` 构造的全部子模块/参数名。
2. 当前代码中 try5 所用 fusion 类的 `__init__`，列出其构造的全部子模块/参数名。特别回答：当前 try5 类是否**仍构造**了 map←graph 方向的 attention 参数而仅在 forward 中不用。
3. 用 checkpoint `state_dict` 中前缀为 `graph_map_attention`（或实际模块名）的 key 列表，与上述两份参数名对照。结论三选一：
   - (a) checkpoint 只含单向参数，且旧双向类会产生额外 key → checkpoint **不是**双向代码训的，A1′-1 前提错误，作废。
   - (b) 当前 try5 类构造了双向全部参数、forward 只用一半 → key 匹配不能区分，交由 C2 裁决。
   - (c) 其他情况，原文列出。

## C2. 行为核对（GPU，决定性）

运行已生成的 `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_gt_teacher_val_unseen.sh`，跑完整 val_unseen，记录 SR / SPL / NE / OSR 与日志路径。

- SR 在 72 ± 1.5：checkpoint 即 72.0 教师，A1′ 整体通过，继续 A5 与 B1–B4。
- SR 明显偏低（< 68）且 C1 结论为 (b)：checkpoint 的融合语义与当前 try5 不同，**停止**并报告，等待人工决定（候选：在当前代码上补一个 `architecture=bidir` 分支给教师专用；或换教师 checkpoint）。
- SR 明显偏低且 C1 结论为 (a)：怀疑 namespace/schema 或 eval 配置，报告并停止。

## C3. 报告

追加 C1、C2 两节到 `reports/R3_smoke_report.md`，更新「异常与待决」。通过则直接续做 A5、B1–B4，同一报告内更新。
