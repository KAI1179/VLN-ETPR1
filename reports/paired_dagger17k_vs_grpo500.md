# Paired comparison on val_unseen: A (dagger_distill_gt_teacher_llmpt_eval_iter17000_p0) vs B (dagger_distill_gt_teacher_llmpt_grpo_i17000_eval_iter500_val_unseen)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 66.72 | 65.47 | -1.25 | [-3.16, +0.53] |
| SPL | 56.02 | 55.86 | -0.16 | [-1.89, +1.38] |
| OSR | 72.43 | 71.13 | -1.31 | [-3.48, +0.55] |
| nDTW | 65.63 | 65.94 | +0.32 | [-0.75, +1.19] |
| NE | 3.82 | 3.98 | +0.16 | [-0.03, +0.37] |
| PL | 12.83 | 11.92 | -0.91 | [-1.30, -0.57] |
| Steps | 91.53 | 84.17 | -7.36 | [-10.13, -4.81] |

success discordant pairs: A-only 99, B-only 76, exact McNemar p = 0.0960

stop error (OSR - SR): A 5.71 pp, B 5.66 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1128 | 29 | 70 |
| O | 24 | 69 | 12 |
| N | 52 | 6 | 449 |

net O->S (stop quality) = -5 episodes (-0.27 pp); net N->reach (reachability) = -24 episodes (-1.31 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1128 | -0.20 | -1.9 | +0.68 |
| N->N | 449 | -2.43 | -21.4 | +0.98 |
| S->N | 70 | -1.23 | +10.0 | -30.10 |
| O->O | 69 | -1.45 | -12.9 | +0.55 |
| N->S | 52 | -1.74 | -21.0 | +26.83 |
| S->O | 29 | +1.25 | +16.5 | -9.19 |
| O->S | 24 | -2.85 | -24.2 | +15.32 |
| O->N | 12 | -2.86 | -6.3 | -9.59 |
| N->O | 6 | -2.04 | -48.2 | +10.27 |
