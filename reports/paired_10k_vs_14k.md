# Paired comparison on val_unseen: A (dagger_distill_gt_teacher_eval_iter10000_val_unseen) vs B (dagger_distill_gt_teacher_eval_iter14000_val_unseen)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 65.09 | 63.62 | -1.47 | [-2.28, -0.57] |
| SPL | 53.92 | 53.08 | -0.84 | [-1.78, +0.04] |
| OSR | 71.72 | 69.60 | -2.12 | [-3.19, -1.19] |
| nDTW | 63.84 | 63.85 | +0.01 | [-0.99, +0.94] |
| NE | 3.92 | 4.00 | +0.08 | [+0.00, +0.17] |
| PL | 13.42 | 13.14 | -0.28 | [-0.85, +0.22] |
| Steps | 96.81 | 95.88 | -0.93 | [-4.39, +2.47] |

success discordant pairs: A-only 115, B-only 88, exact McNemar p = 0.0678

stop error (OSR - SR): A 6.63 pp, B 5.98 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1082 | 30 | 85 |
| O | 37 | 71 | 14 |
| N | 51 | 9 | 460 |

net O->S (stop quality) = +7 episodes (+0.38 pp); net N->reach (reachability) = -39 episodes (-2.12 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1082 | -0.25 | -2.0 | +0.53 |
| N->N | 460 | +0.20 | +4.2 | -0.31 |
| S->N | 85 | +3.11 | +33.8 | -36.19 |
| O->O | 71 | -1.76 | -9.8 | +3.52 |
| N->S | 51 | -4.53 | -45.3 | +40.21 |
| O->S | 37 | -7.37 | -49.3 | +24.91 |
| S->O | 30 | +0.08 | +2.8 | -12.93 |
| O->N | 14 | -3.74 | -11.6 | -12.89 |
| N->O | 9 | +7.51 | +65.4 | -0.07 |
