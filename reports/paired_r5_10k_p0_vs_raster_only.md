# Paired comparison on val_unseen: A (dagger_distill_gt_teacher_eval_iter10000_p0) vs B (dagger_distill_gt_teacher_eval_iter10000_abl_raster_only)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 65.63 | 65.74 | +0.11 | [-0.44, +0.81] |
| SPL | 54.14 | 54.24 | +0.11 | [-0.35, +0.70] |
| OSR | 72.16 | 72.21 | +0.05 | [-0.31, +0.49] |
| nDTW | 64.19 | 64.08 | -0.11 | [-0.26, +0.08] |
| NE | 3.89 | 3.90 | +0.01 | [-0.02, +0.04] |
| PL | 13.42 | 13.42 | -0.00 | [-0.11, +0.07] |
| Steps | 96.74 | 96.82 | +0.08 | [-0.91, +0.73] |

success discordant pairs: A-only 8, B-only 10, exact McNemar p = 0.8145

stop error (OSR - SR): A 6.53 pp, B 6.47 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1199 | 3 | 5 |
| O | 4 | 115 | 1 |
| N | 6 | 1 | 505 |

net O->S (stop quality) = +1 episodes (+0.05 pp); net N->reach (reachability) = +1 episodes (+0.05 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1199 | +0.01 | -0.0 | -0.03 |
| N->N | 505 | +0.01 | +0.4 | -0.07 |
| O->O | 115 | -0.06 | +0.3 | -0.22 |
| N->S | 6 | -0.64 | -14.7 | +6.52 |
| S->N | 5 | -1.38 | -4.6 | -25.09 |
| O->S | 4 | +0.17 | +10.0 | +9.81 |
| S->O | 3 | +1.40 | +4.7 | -20.42 |
| O->N | 1 | -4.50 | -31.0 | +1.48 |
| N->O | 1 | +0.44 | -10.0 | +0.43 |
