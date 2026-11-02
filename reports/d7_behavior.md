# D7 T3 行为统计（CPU）

仅描述配对结果，不作机制推断。SR/OSR/SPL/nDTW及其差值按百分数/百分点；PL为米，steps为连续动作数。

特征定义：{"landmarks": "extract_categories: unique object indices plus unique region indices; separate category spaces", "directions": ["turn around", "on your left", "on your right", "to the left", "to the right", "straight ahead", "left", "right", "straight", "forward", "backward", "upstairs", "downstairs"], "junctions": "Original R2R path node occurrences of degree >=3; included unobstructed neighbors; endpoints included", "length": "Original R2R path length from 3D connectivity poses in metres", "bootstrap": "Scene-cluster resampling 2000 draws, seed0; episode-weighted mean; percentile95%CI"}

## baseline→B（n=1839，场景=11）

| 起点类别→终点类别 | 成功 | 曾到目标范围但失败 | 从未到达 |
|---|---:|---:|---:|
| 成功 | 991 | 35 | 128 |
| 曾到目标范围但失败 | 57 | 29 | 25 |
| 从未到达 | 134 | 21 | 419 |

| 指标 | 起点模型 | 终点模型 | 配对差 | 场景bootstrap95%CI |
|---|---:|---:|---:|---|
| success | 62.751 | 64.274 | +1.523 | [-0.358, +3.947] |
| oracle_success | 68.787 | 68.896 | +0.109 | [-0.823, +1.428] |
| spl | 53.564 | 53.456 | -0.107 | [-1.924, +2.465] |
| ndtw | 66.581 | 66.713 | +0.132 | [-1.157, +1.897] |
| path_length | 13.236 | 13.911 | +0.674 | [-0.075, +1.298] |
| steps_taken | 88.325 | 87.686 | -0.638 | [-3.372, +1.518] |

路径长度三分位切点：[8.236761794539953, 10.376267476756617] m。

| 分层 | n | 场景数 | 起点SR | 终点SR | 差值pp | 95%CI |
|---|---:|---:|---:|---:|---:|---|
| 地标数:0–1 | 80 | 11 | 57.50 | 62.50 | +5.00 | [-4.84, +14.67] |
| 地标数:2–3 | 764 | 11 | 66.23 | 65.71 | -0.52 | [-3.66, +2.44] |
| 地标数:≥4 | 995 | 11 | 60.50 | 63.32 | +2.81 | [-0.29, +7.64] |
| 岔路数:0–1 | 30 | 3 | 70.00 | 70.00 | +0.00 | [+0.00, +0.00] |
| 岔路数:2–3 | 192 | 11 | 64.06 | 69.27 | +5.21 | [-4.76, +13.07] |
| 岔路数:≥4 | 1617 | 11 | 62.46 | 63.57 | +1.11 | [-0.90, +3.80] |
| 路径长度:T1 | 612 | 11 | 77.94 | 81.05 | +3.10 | [-0.52, +7.38] |
| 路径长度:T2 | 612 | 10 | 61.27 | 65.52 | +4.25 | [+1.35, +7.92] |
| 路径长度:T3 | 615 | 10 | 49.11 | 46.34 | -2.76 | [-4.11, -0.24] |

## B→D（n=1839，场景=11）

| 起点类别→终点类别 | 成功 | 曾到目标范围但失败 | 从未到达 |
|---|---:|---:|---:|
| 成功 | 1071 | 38 | 73 |
| 曾到目标范围但失败 | 36 | 32 | 17 |
| 从未到达 | 120 | 18 | 434 |

| 指标 | 起点模型 | 终点模型 | 配对差 | 场景bootstrap95%CI |
|---|---:|---:|---:|---|
| success | 64.274 | 66.721 | +2.447 | [+0.331, +4.224] |
| oracle_success | 68.896 | 71.506 | +2.610 | [+1.002, +3.872] |
| spl | 53.456 | 56.232 | +2.776 | [+0.641, +4.943] |
| ndtw | 66.713 | 68.061 | +1.348 | [-0.065, +2.884] |
| path_length | 13.911 | 12.386 | -1.525 | [-2.989, -0.261] |
| steps_taken | 87.686 | 89.079 | +1.393 | [-0.423, +3.233] |

路径长度三分位切点：[8.236761794539953, 10.376267476756617] m。

| 分层 | n | 场景数 | 起点SR | 终点SR | 差值pp | 95%CI |
|---|---:|---:|---:|---:|---:|---|
| 地标数:0–1 | 80 | 11 | 62.50 | 63.75 | +1.25 | [-8.06, +8.49] |
| 地标数:2–3 | 764 | 11 | 65.71 | 67.67 | +1.96 | [-0.73, +4.19] |
| 地标数:≥4 | 995 | 11 | 63.32 | 66.23 | +2.91 | [+0.00, +5.30] |
| 岔路数:0–1 | 30 | 3 | 70.00 | 70.00 | +0.00 | [+0.00, +0.00] |
| 岔路数:2–3 | 192 | 11 | 69.27 | 69.27 | +0.00 | [-7.69, +4.98] |
| 岔路数:≥4 | 1617 | 11 | 63.57 | 66.36 | +2.78 | [+0.95, +4.49] |
| 路径长度:T1 | 612 | 11 | 81.05 | 80.72 | -0.33 | [-3.61, +3.31] |
| 路径长度:T2 | 612 | 10 | 65.52 | 68.30 | +2.78 | [-0.89, +5.76] |
| 路径长度:T3 | 615 | 10 | 46.34 | 51.22 | +4.88 | [+1.47, +8.38] |

## 输入状态

```json
{
  "baseline": {
    "status": "complete",
    "files": [
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_base_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r0_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_base_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r1_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_base_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r2_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_base_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r3_w4.json"
    ]
  },
  "B": {
    "status": "complete",
    "files": [
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d5_deploy_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r0_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d5_deploy_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r1_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d5_deploy_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r2_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d5_deploy_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r3_w4.json"
    ]
  },
  "D": {
    "status": "complete",
    "files": [
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_kd_deploy_val_unseen/eval_results/stats_ep_ckpt_17000_val_unseen_r0_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_kd_deploy_val_unseen/eval_results/stats_ep_ckpt_17000_val_unseen_r1_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_kd_deploy_val_unseen/eval_results/stats_ep_ckpt_17000_val_unseen_r2_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_kd_deploy_val_unseen/eval_results/stats_ep_ckpt_17000_val_unseen_r3_w4.json"
    ]
  }
}
```

缺失输入保留为待评测；不以历史汇总百分数替代逐 episode 数据。D默认读取新的teleport部署重评，未使用历史control结果作同协议对照。
