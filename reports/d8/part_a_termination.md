# D8 Part A — termination taxonomy (with-map)

Protocol: control / 4 ranks / 150 tokens / 15 high-level steps. Baseline has no map; Try5 and KD use their own LLM maps.

A: success; B: failed after reaching goal range; C: never reached, stop; D: never reached, budget; E: never reached, no candidate; U: missing cache, not run.

Path-length tercile cuts: 8.2367618 / 10.3762675 m. Rates include U in the denominator. Paired comparisons exclude U.

| Model | Group | n | A n (%) | B n (%) | C n (%) | D n (%) | E n (%) | U n (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | all | 1839 | 1159 (63.02) | 101 (5.49) | 382 (20.77) | 197 (10.71) | 0 (0.00) | 0 (0.00) |
| Try5 | all | 1839 | 1196 (65.04) | 98 (5.33) | 350 (19.03) | 189 (10.28) | 0 (0.00) | 6 (0.33) |
| KD | all | 1839 | 1237 (67.26) | 89 (4.84) | 335 (18.22) | 172 (9.35) | 0 (0.00) | 6 (0.33) |
| baseline | T1 | 612 | 479 (78.27) | 45 (7.35) | 49 (8.01) | 39 (6.37) | 0 (0.00) | 0 (0.00) |
| Try5 | T1 | 612 | 489 (79.90) | 42 (6.86) | 50 (8.17) | 31 (5.07) | 0 (0.00) | 0 (0.00) |
| KD | T1 | 612 | 500 (81.70) | 38 (6.21) | 48 (7.84) | 26 (4.25) | 0 (0.00) | 0 (0.00) |
| baseline | T2 | 612 | 379 (61.93) | 46 (7.52) | 127 (20.75) | 60 (9.80) | 0 (0.00) | 0 (0.00) |
| Try5 | T2 | 612 | 403 (65.85) | 39 (6.37) | 116 (18.95) | 53 (8.66) | 0 (0.00) | 1 (0.16) |
| KD | T2 | 612 | 419 (68.46) | 36 (5.88) | 109 (17.81) | 47 (7.68) | 0 (0.00) | 1 (0.16) |
| baseline | T3 | 615 | 301 (48.94) | 10 (1.63) | 206 (33.50) | 98 (15.93) | 0 (0.00) | 0 (0.00) |
| Try5 | T3 | 615 | 304 (49.43) | 17 (2.76) | 184 (29.92) | 105 (17.07) | 0 (0.00) | 5 (0.81) |
| KD | T3 | 615 | 318 (51.71) | 15 (2.44) | 178 (28.94) | 99 (16.10) | 0 (0.00) | 5 (0.81) |

| Group | Comparison | Paired n | C-rate difference (pp) | Scene-bootstrap 95% CI (pp) |
|---|---|---:|---:|---|
| all | KD - baseline | 1833 | -2.51 | [-4.03, -0.88] |
| all | KD - Try5 | 1833 | -0.82 | [-2.19, +0.53] |
| all | Try5 - baseline | 1833 | -1.69 | [-3.27, -0.22] |
| T1 | KD - baseline | 612 | -0.16 | [-2.08, +2.07] |
| T1 | KD - Try5 | 612 | -0.33 | [-2.09, +1.24] |
| T1 | Try5 - baseline | 612 | +0.16 | [-1.30, +1.78] |
| T2 | KD - baseline | 611 | -2.95 | [-6.05, -0.88] |
| T2 | KD - Try5 | 611 | -1.15 | [-2.76, +0.41] |
| T2 | Try5 - baseline | 611 | -1.80 | [-4.94, +0.75] |
| T3 | KD - baseline | 610 | -4.43 | [-7.19, -1.82] |
| T3 | KD - Try5 | 610 | -0.98 | [-3.71, +2.04] |
| T3 | Try5 - baseline | 610 | -3.44 | [-7.23, -0.36] |

| Group | Baseline A → KD C |
|---|---:|
| all | 79 |
| T1 | 13 |
| T2 | 27 |
| T3 | 39 |

## Input files

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_base_control_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r0_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_base_control_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r1_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_base_control_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r2_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_base_control_val_unseen/eval_results/stats_ep_ckpt_25000_val_unseen_r3_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_try5_control_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r0_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_try5_control_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r1_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_try5_control_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r2_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_try5_control_val_unseen/eval_results/stats_ep_ckpt_28000_val_unseen_r3_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_kd_control_val_unseen/eval_results/stats_ep_ckpt_27000_val_unseen_r0_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_kd_control_val_unseen/eval_results/stats_ep_ckpt_27000_val_unseen_r1_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_kd_control_val_unseen/eval_results/stats_ep_ckpt_27000_val_unseen_r2_w4.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/d7d_kd_control_val_unseen/eval_results/stats_ep_ckpt_27000_val_unseen_r3_w4.json`
