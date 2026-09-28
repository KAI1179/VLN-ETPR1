# Paired comparison on val_unseen: A (gt_teacher_try5_val_unseen_z_none) vs B (dagger_distill_gt_teacher_eval_iter10000_p0)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 73.84 | 65.63 | -8.21 | [-13.38, -1.51] |
| SPL | 63.38 | 54.14 | -9.25 | [-14.39, -2.79] |
| OSR | 77.49 | 72.16 | -5.33 | [-10.03, +1.16] |
| nDTW | 70.41 | 64.19 | -6.22 | [-9.69, -2.21] |
| NE | 3.29 | 3.89 | +0.60 | [+0.05, +1.09] |
| PL | 12.15 | 13.42 | +1.28 | [+0.28, +2.33] |
| Steps | 87.41 | 96.74 | +9.33 | [+1.27, +17.79] |

success discordant pairs: A-only 288, B-only 137, exact McNemar p = 0.0000

stop error (OSR - SR): A 3.64 pp, B 6.53 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1070 | 79 | 209 |
| O | 30 | 26 | 11 |
| N | 107 | 15 | 292 |

net O->S (stop quality) = -49 episodes (-2.66 pp); net N->reach (reachability) = -98 episodes (-5.33 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1070 | +0.96 | +7.4 | -3.29 |
| N->N | 292 | +1.90 | +15.7 | -2.18 |
| S->N | 209 | +1.80 | +22.9 | -40.59 |
| N->S | 107 | -2.39 | -40.8 | +38.07 |
| S->O | 79 | +9.26 | +61.9 | -36.55 |
| O->S | 30 | -2.47 | -22.3 | +10.66 |
| O->O | 26 | +0.53 | +8.0 | -14.29 |
| N->O | 15 | +2.62 | +20.1 | +11.36 |
| O->N | 11 | -5.90 | -44.5 | -9.50 |
