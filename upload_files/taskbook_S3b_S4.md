# Task Book：S3b（增强重训）+ S4（DAgger 接入与对照）

分支 `exp/refiner`，基于 `1f3fcc6`。两个阶段各一个 commit，S4 结束后停下汇报，不自动进入后续。

---

## S3b：dihedral 增强重训 refiner

### 改动

`vlnce_baselines/models/refiner/dataset.py`：

- 新增构造参数 `augment: bool = False`
- `augment=True` 时，`__getitem__` 对同一样本的 `x (67,100,100)`、`y (37,100,100)`、`p0 (37,100,100)`、`obs (100,100)`、`route (100,100)` 施加**同一个**随机 dihedral 变换：`k = rng.integers(4)` 次 `rot90`（最后两个轴），再以 0.5 概率沿最后一个轴翻转。随机数用 `np.random.default_rng(seed + epoch * len(samples) + index)`，保证可复现
- 验证集不变换

`scripts/refiner/train.py`：

- 新增 `--augment` 开关，只作用于训练集
- 新增 `--output-dir` 默认值不变；本次运行显式传 `data/refiner/checkpoints_aug`

### 训练

与 S3 完全相同的超参（batch 32、lr 3e-4、AdamW、cosine、10 epoch、6 step/轨迹、AMP、BCE with logits、8 workers、pos_weight 用同一 2000 条轨迹），仅加 `--augment`。

### 汇报

1. 每 epoch train loss、val_unseen route_seen object mIoU@100×100 与 @10×10，与 S3 无增强版并排（两列）
2. best epoch 编号
3. best epoch 的 val_unseen 两张表（100×100、10×10；行 seen/unseen/route/route_seen；列 P0 object / refined object / P0 region / refined region），与 S3 的表并排
4. ckpt 选择：若增强版 best 的 val_unseen route_seen object mIoU@10×10 ≥ 29.38 且 best epoch ≥ 3，选增强版；否则选 S3 原版。写明选择结果和依据

---

## S4：DAgger 接入 + 同 seed 对照

### 配置

以 clean Try5（`exp/try5-clean` @ `7535889`）的 DAgger 运行配置为唯一模板：同一 yaml、同一起始 checkpoint、同一 seed、同一迭代数、同一 GPU/env 数。

两组运行：

| 组 | 差异 |
|---|---|
| `try5-control` | 完全不改 |
| `try5-refiner` | 仅 `MODEL.MAP_ENCODER.refiner_ckpt` 指向 S3b 选定的 ckpt |

如果 clean Try5 在这个配置、这个 seed、这个迭代数下已有 val_unseen 结果，可以直接引用，但要给出其 checkpoint 路径和评测日志路径；否则必须重跑 `try5-control`。

### 执行

1. 先启动 `try5-refiner`，跑满 200 iteration 后记录单 iteration 平均耗时，与 `try5-control` 的同一指标对比。慢超过 30% 则暂停，报告 profile（投影、refiner 前向、其余各占多少）
2. 两组训练到相同迭代数
3. 两组各在 val_unseen 评测，报 SR / SPL / NE / OSR；同时报 val_seen
4. 评测时确认 `refiner_ckpt` 已生效：日志中打印 refiner 加载路径和参数量；随机抽 1 个 episode 打印第 0 步和最后一步 cognitive map grid 的非零 cell 数（P0 vs refined），两者不同才说明 refiner 在起作用

### 汇报

| 组 | val_unseen SR | SPL | NE | OSR | val_seen SR | SPL |
|---|---|---|---|---|---|---|
| try5-control | | | | | | |
| try5-refiner | | | | | | |

外加：单 iteration 耗时对比、训练 loss 曲线（每 1000 iteration 一个点）两组并排、所用 refiner ckpt 路径。

汇报后停下。

---

## 不做

不改 map encoder / fusion / head / 预训练；不做 region 合成消融、不做输出二值化（这两个作为 S4 结果出来后的备选）；不加本文以外的检查。
