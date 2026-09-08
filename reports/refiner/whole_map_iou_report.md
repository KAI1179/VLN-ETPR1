# 冻结 Refiner 整图 IoU 评估

## 1. 评估设置

- Refiner：`data/refiner/checkpoints_aug/best.pt`，全程冻结。
- 数据：`val_seen` 749 个 episode、`val_unseen` 1821 个 episode；每个 episode 只评估最后一步累积视觉证据对应的整张图。
- 合成：已观察格子采用 Refiner 输出，未观察格子保留原始 LLM 地图 P0；object 与 region 使用相同规则。
- 阈值：0.5；数值均为跨全部 episode、通道与格子累计交并集后的 Micro-IoU 百分比。
- object semantic 排除 `void / structure / other / free-space`；object occupancy 将其余物体类别合并为一个物体占用掩码；combined semantic 合并有效 object 通道与全部 region 通道。

## 2. val_unseen

| 分辨率 | 指标 | 原始 LLM P0 | Refined | 提升（百分点） |
|---|---|---:|---:|---:|
| 100×100 | Object semantic IoU | 1.60 | **9.64** | +8.04 |
| 100×100 | Object occupancy IoU | 7.29 | **14.47** | +7.18 |
| 100×100 | Region semantic IoU | 6.75 | **12.35** | +5.60 |
| 100×100 | Combined semantic IoU | 5.08 | **11.26** | +6.19 |
| 50×50 | Object semantic IoU | 2.85 | **16.35** | +13.49 |
| 50×50 | Object occupancy IoU | 12.07 | **25.86** | +13.78 |
| 50×50 | Region semantic IoU | 7.76 | **15.10** | +7.34 |
| 50×50 | Combined semantic IoU | 5.99 | **15.67** | +9.68 |
| 10×10 | Object semantic IoU | 11.47 | **27.47** | +16.00 |
| 10×10 | Object occupancy IoU | 37.70 | **52.08** | +14.38 |
| 10×10 | Region semantic IoU | 16.81 | **24.44** | +7.63 |
| 10×10 | Combined semantic IoU | 13.80 | **26.24** | +12.44 |

## 3. val_seen

| 分辨率 | 指标 | 原始 LLM P0 | Refined | 提升（百分点） |
|---|---|---:|---:|---:|
| 100×100 | Object semantic IoU | 6.96 | **14.27** | +7.32 |
| 100×100 | Object occupancy IoU | 12.08 | **18.82** | +6.74 |
| 100×100 | Region semantic IoU | 16.40 | **21.48** | +5.09 |
| 100×100 | Combined semantic IoU | 13.54 | **18.92** | +5.38 |
| 50×50 | Object semantic IoU | 13.13 | **23.94** | +10.81 |
| 50×50 | Object occupancy IoU | 20.74 | **32.37** | +11.63 |
| 50×50 | Region semantic IoU | 19.26 | **25.83** | +6.57 |
| 50×50 | Combined semantic IoU | 17.22 | **25.06** | +7.84 |
| 10×10 | Object semantic IoU | 21.56 | **35.43** | +13.88 |
| 10×10 | Object occupancy IoU | 43.00 | **55.99** | +12.99 |
| 10×10 | Region semantic IoU | 28.89 | **34.93** | +6.05 |
| 10×10 | Combined semantic IoU | 24.81 | **35.22** | +10.42 |

## 4. 结论

冻结 Refiner 在两个验证集、两种分辨率及四类整图指标上均优于原始 LLM 地图。`val_unseen` 的提升尤其明显，说明 Refiner 单独看具有有效的地图纠错能力；当前导航收益不足不能归因于 Refiner 完全没有改善地图，而应继续检查更新后地图与 Try5 决策之间的适配。
