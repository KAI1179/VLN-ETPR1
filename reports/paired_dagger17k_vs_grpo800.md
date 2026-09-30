# Paired comparison on val_unseen: A (dagger_distill_gt_teacher_llmpt_eval_iter17000_p0) vs B (dagger_distill_gt_teacher_llmpt_grpo_i17000_eval_iter800_val_unseen)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 66.72 | 64.93 | -1.79 | [-3.32, -0.50] |
| SPL | 56.02 | 55.14 | -0.88 | [-2.38, +0.32] |
| OSR | 72.43 | 71.34 | -1.09 | [-2.77, +0.28] |
| nDTW | 65.63 | 65.33 | -0.29 | [-1.28, +0.60] |
| NE | 3.82 | 4.05 | +0.23 | [+0.03, +0.43] |
| PL | 12.83 | 11.76 | -1.07 | [-1.56, -0.75] |
| Steps | 91.53 | 82.09 | -9.44 | [-12.35, -7.56] |

success discordant pairs: A-only 125, B-only 92, exact McNemar p = 0.0296

stop error (OSR - SR): A 5.71 pp, B 6.42 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1102 | 46 | 79 |
| O | 30 | 63 | 12 |
| N | 62 | 9 | 436 |

net O->S (stop quality) = -16 episodes (-0.87 pp); net N->reach (reachability) = -20 episodes (-1.09 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1102 | -0.16 | -1.9 | +0.49 |
| N->N | 436 | -3.00 | -26.9 | +0.81 |
| S->N | 79 | -2.05 | +3.5 | -34.50 |
| O->O | 63 | -1.14 | -10.4 | -1.56 |
| N->S | 62 | -4.30 | -43.3 | +30.86 |
| S->O | 46 | +4.23 | +31.1 | -21.46 |
| O->S | 30 | -4.80 | -44.9 | +17.19 |
| O->N | 12 | -4.42 | -22.7 | -7.37 |
| N->O | 9 | +1.57 | -33.6 | +4.43 |
