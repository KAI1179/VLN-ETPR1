# Paired comparison on val_unseen: A (dagger_distill_gt_teacher_eval_iter8000_val_unseen) vs B (dagger_distill_gt_teacher_eval_iter10000_val_unseen)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 63.73 | 65.09 | +1.36 | [-0.58, +2.99] |
| SPL | 54.19 | 53.92 | -0.27 | [-2.18, +1.27] |
| OSR | 69.77 | 71.72 | +1.96 | [+0.27, +3.44] |
| nDTW | 64.42 | 63.84 | -0.58 | [-1.72, +0.41] |
| NE | 4.07 | 3.92 | -0.15 | [-0.22, -0.07] |
| PL | 12.39 | 13.42 | +1.03 | [+0.58, +1.58] |
| Steps | 88.14 | 96.81 | +8.68 | [+5.54, +12.79] |

success discordant pairs: A-only 92, B-only 117, exact McNemar p = 0.0967

stop error (OSR - SR): A 6.04 pp, B 6.63 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1080 | 37 | 55 |
| O | 24 | 75 | 12 |
| N | 93 | 10 | 453 |

net O->S (stop quality) = -13 episodes (-0.71 pp); net N->reach (reachability) = +36 episodes (+1.96 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1080 | +0.46 | +4.7 | -0.69 |
| N->N | 453 | +1.89 | +16.5 | -1.50 |
| N->S | 93 | -0.97 | -24.2 | +34.85 |
| O->O | 75 | +2.92 | +22.3 | -1.57 |
| S->N | 55 | +3.70 | +41.5 | -40.35 |
| S->O | 37 | +6.72 | +52.1 | -26.34 |
| O->S | 24 | -1.57 | -4.3 | +15.72 |
| O->N | 12 | -0.79 | -7.4 | -5.83 |
| N->O | 10 | +1.28 | -2.9 | +11.98 |
