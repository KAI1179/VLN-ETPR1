# Paired comparison on val_unseen: A (dagger_distill_gt_teacher_eval_iter10000_p0) vs B (dagger_distill_gt_teacher_llmpt_eval_iter17000_p0)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 65.63 | 66.72 | +1.09 | [+0.06, +2.27] |
| SPL | 54.14 | 56.02 | +1.89 | [+0.67, +3.41] |
| OSR | 72.16 | 72.43 | +0.27 | [-0.70, +1.49] |
| nDTW | 64.19 | 65.63 | +1.44 | [+0.30, +3.03] |
| NE | 3.89 | 3.82 | -0.07 | [-0.22, +0.06] |
| PL | 13.42 | 12.83 | -0.60 | [-1.38, +0.10] |
| Steps | 96.74 | 91.53 | -5.21 | [-10.68, +0.06] |

success discordant pairs: A-only 161, B-only 181, exact McNemar p = 0.3042

stop error (OSR - SR): A 6.53 pp, B 5.71 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1046 | 48 | 113 |
| O | 50 | 45 | 25 |
| N | 131 | 12 | 369 |

net O->S (stop quality) = +2 episodes (+0.11 pp); net N->reach (reachability) = +5 episodes (+0.27 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1046 | -0.44 | -4.6 | +1.12 |
| N->N | 369 | -0.00 | +1.4 | +1.22 |
| N->S | 131 | -2.78 | -38.3 | +33.56 |
| S->N | 113 | +0.88 | +24.5 | -35.22 |
| O->S | 50 | -5.00 | -39.8 | +25.58 |
| S->O | 48 | +3.35 | +25.4 | -20.62 |
| O->O | 45 | -5.36 | -39.8 | +8.47 |
| O->N | 25 | -2.97 | -16.6 | -8.28 |
| N->O | 12 | +2.34 | -2.5 | +12.00 |
