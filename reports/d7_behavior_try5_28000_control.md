# Try5 无KD 28000 带图 vs ETP-R1 25000：行为归因

【执行口径已统一】baseline 25000 与 Try5 无KD 28000 带图均为 control、150 token、15 高层步。baseline 使用 4 rank，ours 已有结果使用 1 rank；rank 划分差异仍保留为限制。配对统计描述行为差异，不证明内部机制。

1839 episodes、11 场景；沿用旧 D7 的原始 R2R 连通图特征。场景 bootstrap 2000 次，seed 0。ours 的 6 条缓存缺失仍计零。

## 失败类别转移（行 baseline，列 ours）

| 类别 | 成功 | 曾到目标范围但失败 | 从未到达 |
|---|---:|---:|---:|
| 成功 | 988 | 45 | 126 |
| 曾到目标范围但失败 | 55 | 28 | 18 |
| 从未到达 | 154 | 24 | 401 |

## 配对指标

| 指标 | baseline | ours | 差值 | 场景 bootstrap 95% CI |
|---|---:|---:|---:|---|
| success | 63.023 | 65.090 | +2.066 | [+0.071, +4.728] |
| oracle_success | 68.515 | 70.364 | +1.849 | [+0.473, +3.372] |
| spl | 53.313 | 55.030 | +1.717 | [-0.640, +4.750] |
| ndtw | 64.205 | 64.070 | -0.136 | [-2.054, +2.392] |
| path_length | 13.832 | 14.024 | +0.191 | [-0.676, +1.135] |
| steps_taken | 99.843 | 99.404 | -0.439 | [-7.547, +6.855] |

## 分层 SR

路径长度三分位切点：[8.236761794539953, 10.376267476756617] m。

| 分层 | n | baseline SR | ours SR | 差值 pp | 95% CI |
|---|---:|---:|---:|---:|---|
| 地标数:0–1 | 80 | 61.25 | 67.50 | +6.25 | [-5.68, +17.95] |
| 地标数:2–3 | 764 | 66.36 | 67.80 | +1.44 | [-2.35, +5.37] |
| 地标数:≥4 | 995 | 60.60 | 62.81 | +2.21 | [-0.71, +6.76] |
| 岔路数:0–1 | 30 | 70.00 | 70.00 | +0.00 | [+0.00, +0.00] |
| 岔路数:2–3 | 192 | 64.58 | 66.67 | +2.08 | [-4.27, +7.94] |
| 岔路数:≥4 | 1617 | 62.71 | 64.81 | +2.10 | [-0.14, +5.41] |
| 路径长度:T1 | 612 | 78.27 | 79.90 | +1.63 | [-2.06, +5.58] |
| 路径长度:T2 | 612 | 61.93 | 66.01 | +4.08 | [+0.46, +8.48] |
| 路径长度:T3 | 615 | 48.94 | 49.43 | +0.49 | [-2.41, +5.66] |

## 数据来源

```json
{
  "baseline": {
    "status": "complete",
    "files": [
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_base_control_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r0_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_base_control_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r1_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_base_control_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r2_w4.json",
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7_base_control_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r3_w4.json"
    ]
  },
  "ours": {
    "status": "complete",
    "files": [
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/llm_grid_try5_iter28000_raster_control/eval_results/stats_ep_ckpt_28000_val_unseen_r0_w1.json"
    ]
  }
}
```

注意：分层 CI 未做多重比较校正；指标差异不证明地标 grounding 等内部机制。
