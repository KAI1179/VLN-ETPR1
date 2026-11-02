# KD FULL 27000 带图 vs ETP-R1 25000：行为归因

【口径不一致】baseline 25000 为 teleport；KD FULL 27000 带图为 control。以下仅为探索性配对描述，不能归因于模型改进。未启动 GPU；需要补评 baseline control 后重算。

1839 episodes、11 场景；沿用旧 D7 的原始 R2R 连通图特征。场景 bootstrap 2000 次，seed 0。ours 的 6 条缓存缺失仍计零。

## 失败类别转移（行 baseline，列 ours）

| 类别 | 成功 | 曾到目标范围但失败 | 从未到达 |
|---|---:|---:|---:|
| 成功 | 1001 | 37 | 116 |
| 曾到目标范围但失败 | 64 | 34 | 13 |
| 从未到达 | 172 | 19 | 383 |

## 配对指标

| 指标 | baseline | ours | 差值 | 场景 bootstrap 95% CI |
|---|---:|---:|---:|---|
| success | 62.751 | 67.265 | +4.513 | [+2.033, +7.180] |
| oracle_success | 68.787 | 72.159 | +3.371 | [+1.144, +5.547] |
| spl | 53.564 | 56.436 | +2.872 | [+0.309, +6.076] |
| ndtw | 66.581 | 65.404 | -1.177 | [-2.819, +1.016] |
| path_length | 13.236 | 13.538 | +0.302 | [-0.470, +1.008] |
| steps_taken | 88.325 | 94.678 | +6.353 | [+0.056, +15.338] |

## 分层 SR

路径长度三分位切点：[8.236761794539953, 10.376267476756617] m。

| 分层 | n | baseline SR | ours SR | 差值 pp | 95% CI |
|---|---:|---:|---:|---:|---|
| 地标数:0–1 | 80 | 57.50 | 68.75 | +11.25 | [-1.54, +21.98] |
| 地标数:2–3 | 764 | 66.23 | 69.37 | +3.14 | [+0.41, +5.84] |
| 地标数:≥4 | 995 | 60.50 | 65.53 | +5.03 | [+1.97, +8.39] |
| 岔路数:0–1 | 30 | 70.00 | 76.67 | +6.67 | [+0.00, +9.52] |
| 岔路数:2–3 | 192 | 64.06 | 69.79 | +5.73 | [-6.92, +14.10] |
| 岔路数:≥4 | 1617 | 62.46 | 66.79 | +4.33 | [+1.89, +7.14] |
| 路径长度:T1 | 612 | 77.94 | 81.86 | +3.92 | [-0.21, +9.15] |
| 路径长度:T2 | 612 | 61.27 | 68.30 | +7.03 | [+4.17, +10.88] |
| 路径长度:T3 | 615 | 49.11 | 51.71 | +2.60 | [-1.12, +6.71] |

## 数据来源

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
  "ours": {
    "status": "complete",
    "files": [
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual_eval_iter27000_abl_none/eval_results/stats_ep_ckpt_27000_val_unseen_r0_w1.json"
    ]
  }
}
```

注意：分层 CI 未做多重比较校正；指标差异不证明地标 grounding 等内部机制。
