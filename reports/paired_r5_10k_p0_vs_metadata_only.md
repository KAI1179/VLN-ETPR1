# Paired comparison on val_unseen: A (dagger_distill_gt_teacher_eval_iter10000_p0) vs B (dagger_distill_gt_teacher_eval_iter10000_abl_metadata_only)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 65.63 | 63.02 | -2.61 | [-4.50, -0.21] |
| SPL | 54.14 | 53.25 | -0.89 | [-2.70, +1.34] |
| OSR | 72.16 | 67.86 | -4.30 | [-5.84, -2.56] |
| nDTW | 64.19 | 64.63 | +0.45 | [-0.60, +1.82] |
| NE | 3.89 | 3.93 | +0.04 | [-0.17, +0.20] |
| PL | 13.42 | 13.33 | -0.09 | [-0.53, +0.44] |
| Steps | 96.74 | 97.23 | +0.48 | [-2.45, +3.59] |

success discordant pairs: A-only 124, B-only 76, exact McNemar p = 0.0008

stop error (OSR - SR): A 6.53 pp, B 4.84 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1083 | 22 | 102 |
| O | 42 | 60 | 18 |
| N | 34 | 7 | 471 |

net O->S (stop quality) = +20 episodes (+1.09 pp); net N->reach (reachability) = -79 episodes (-4.30 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1083 | -0.38 | -2.4 | +0.60 |
| N->N | 471 | +1.12 | +9.2 | -0.47 |
| S->N | 102 | +0.01 | +11.6 | -21.47 |
| O->O | 60 | -0.23 | +0.1 | +2.15 |
| O->S | 42 | -6.76 | -48.1 | +28.19 |
| N->S | 34 | -3.40 | -30.3 | +47.27 |
| S->O | 22 | +8.80 | +68.7 | -20.01 |
| O->N | 18 | -5.95 | -43.4 | +2.70 |
| N->O | 7 | +6.99 | +40.6 | +7.31 |
