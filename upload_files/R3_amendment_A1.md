# 任务书 R3 修正案：A1 判据替换

R3 报告已阅。A1 的判据依赖 checkpoint 自描述 config，而该 checkpoint 产生于 `MAP_ENCODER.architecture/source/cache_namespace` 三个键引入之前的代码版本，自描述核对不适用。以下用两条更强的判据替换 A1，通过后继续 R3 的 A2–A5 与 B1–B4，原文不变。

## A1′-1 代码考古（只读）

1. `git log -S "navigation_architecture" --oneline -- vlnce_baselines/models/` 与 `git log -S "MAP_ENCODER.architecture" --oneline -- vlnce_baselines/config/default.py`，找出这三个键与 `PriorGTTry5Policy` 类被引入的 commit。
2. 在该 commit 的**父提交**上 `git show <parent>:vlnce_baselines/models/etp_prior_gt/policy.py`，确认当时的 `PriorGTPolicy` 是否只有一种导航实现，以及该实现是否等于当前 `architecture="try5"` 分支（比对 `vilmodel_cmt.py` 中 try5 分支的关键结构：单向 attention、静态地图融合、`graph_map_attention` 的调用形式）。给出 `文件:行号` 级别的对照，结论写成一句：`旧 PriorGTPolicy ≡ 当前 try5` 或 `不等价，差异在 X`。
3. checkpoint 的 `iteration 16000`、`IL.iters 30000` 与目录名 `try-5-r1p5-dagger.iter16000.pth` 一致，记录即可。

## A1′-2 权重结构核对（只读实例化）

用当前代码构建教师（与 trainer §2 完全一致的 `t_config`：`policy_name=PriorGTTry5Policy, source=prior_gt, architecture=try5, cache_namespace=gt.legacy.r1p5.direction5.v1, refiner_ckpt=""`），不做 DDP，然后：

```python
sd = {k.replace("net.module.", "net.", 1): v for k, v in ckpt["state_dict"].items()}
model_sd = teacher.state_dict()
missing = sorted(set(model_sd) - set(sd))
unexpected = sorted(set(sd) - set(model_sd))
shape_mismatch = [(k, tuple(sd[k].shape), tuple(model_sd[k].shape))
                  for k in set(sd) & set(model_sd) if sd[k].shape != model_sd[k].shape]
print(len(missing), len(unexpected), len(shape_mismatch))
```

通过标准：三者全为 0。任一非零，把完整列表写进报告并**停止**。这一步等价于 trainer 里的 `assert missing == 0`，但提前到无 GPU 阶段，并额外查 shape。

## A1′-3 行为核对（GPU，可与 B1 前置执行）

用现有 eval 入口，以 `PriorGTTry5Policy` + `source=prior_gt` + `cache_namespace=gt.legacy.r1p5.direction5.v1` + 该 checkpoint，在 val_unseen 跑一次完整评测（现有 GT 地图 eval launcher 若存在则复用，只替换 ckpt 与 namespace；不存在则从 `run_r2r/main_server.bash` 的 eval 分支派生，写到 `scripts/distill/eval_gt_teacher_val_unseen.sh`）。

通过标准：SR 在 72 ± 1.5 范围内（历史记录"约 72%"）。这同时解决 R1-D.1 中"72.0 checkpoint 未唯一定位"的问题——若通过，报告里明确写：**该 checkpoint 即 72.0 教师**，并记录 SR/SPL/NE/OSR 四项。

若 SR 明显偏低（如 < 68）：优先怀疑 namespace/schema 错配，回到 A3 查 `metadata_schema` 与 npz 实际内容；其次怀疑 A1′-1 的等价性结论。报告并停止。

## 后续

A1′-1 至 A1′-3 全部通过后，按原 R3 执行 A2（补完 episode 级路径检查）、A3、A4、A5，再 B1–B4。报告仍写到 `reports/R3_smoke_report.md`，在原文上追加 A1′ 三节并更新各节状态。
