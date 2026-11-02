# KD FULL 27000 带图 vs ETP-R1 25000：行为归因

【执行口径已统一】baseline 25000 与 KD FULL 27000 带图均为 control、150 token、15 高层步。baseline 使用 4 rank，ours 已有结果使用 1 rank；rank 划分差异仍保留为限制。配对统计描述行为差异，不证明内部机制。

1839 episodes、11 场景；沿用旧 D7 的原始 R2R 连通图特征。场景 bootstrap 2000 次，seed 0。ours 的 6 条缓存缺失仍计零。

## 失败类别转移（行 baseline，列 ours）

| 类别 | 成功 | 曾到目标范围但失败 | 从未到达 |
|---|---:|---:|---:|
| 成功 | 1007 | 41 | 111 |
| 曾到目标范围但失败 | 58 | 33 | 10 |
| 从未到达 | 172 | 16 | 391 |

## 配对指标

| 指标 | baseline | ours | 差值 | 场景 bootstrap 95% CI |
|---|---:|---:|---:|---|
| success | 63.023 | 67.265 | +4.241 | [+2.352, +6.763] |
| oracle_success | 68.515 | 72.159 | +3.643 | [+2.127, +5.362] |
| spl | 53.313 | 56.436 | +3.123 | [+0.932, +6.084] |
| ndtw | 64.205 | 65.404 | +1.198 | [-0.676, +3.601] |
| path_length | 13.832 | 13.538 | -0.294 | [-1.127, +0.428] |
| steps_taken | 99.843 | 94.678 | -5.165 | [-11.432, +0.240] |

## 分层 SR

路径长度三分位切点：[8.236761794539953, 10.376267476756617] m。

| 分层 | n | baseline SR | ours SR | 差值 pp | 95% CI |
|---|---:|---:|---:|---:|---|
| 地标数:0–1 | 80 | 61.25 | 68.75 | +7.50 | [-5.71, +18.69] |
| 地标数:2–3 | 764 | 66.36 | 69.37 | +3.01 | [+0.79, +5.60] |
| 地标数:≥4 | 995 | 60.60 | 65.53 | +4.92 | [+2.78, +8.04] |
| 岔路数:0–1 | 30 | 70.00 | 76.67 | +6.67 | [+0.00, +9.52] |
| 岔路数:2–3 | 192 | 64.58 | 69.79 | +5.21 | [-2.33, +10.53] |
| 岔路数:≥4 | 1617 | 62.71 | 66.79 | +4.08 | [+2.27, +6.45] |
| 路径长度:T1 | 612 | 78.27 | 81.86 | +3.59 | [+0.35, +7.85] |
| 路径长度:T2 | 612 | 61.93 | 68.30 | +6.37 | [+3.93, +10.10] |
| 路径长度:T3 | 615 | 48.94 | 51.71 | +2.76 | [-0.66, +8.54] |

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
      "/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual_eval_iter27000_abl_none/eval_results/stats_ep_ckpt_27000_val_unseen_r0_w1.json"
    ]
  }
}
```

注意：分层 CI 未做多重比较校正；指标差异不证明地标 grounding 等内部机制。
