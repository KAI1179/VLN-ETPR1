# Paired comparison on val_unseen: A (dagger_distill_gt_teacher_llmpt_eval_iter17000_p0) vs B (dagger_distill_gt_teacher_llmpt_eval_iter17000_abl_raster_only)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 66.72 | 65.25 | -1.47 | [-2.78, -0.07] |
| SPL | 56.02 | 55.35 | -0.68 | [-1.59, +0.37] |
| OSR | 72.43 | 71.29 | -1.14 | [-2.25, +0.05] |
| nDTW | 65.63 | 65.93 | +0.30 | [-0.18, +0.84] |
| NE | 3.82 | 3.84 | +0.02 | [-0.11, +0.11] |
| PL | 12.83 | 12.40 | -0.42 | [-0.86, -0.15] |
| Steps | 91.53 | 88.15 | -3.38 | [-6.87, -1.07] |

success discordant pairs: A-only 82, B-only 55, exact McNemar p = 0.0260

stop error (OSR - SR): A 5.71 pp, B 6.04 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1145 | 27 | 55 |
| O | 23 | 80 | 2 |
| N | 32 | 4 | 471 |

net O->S (stop quality) = -4 episodes (-0.22 pp); net N->reach (reachability) = -21 episodes (-1.14 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1145 | -0.16 | -1.5 | +0.35 |
| N->N | 471 | -0.93 | -8.0 | +0.43 |
| O->O | 80 | -0.41 | -2.7 | +0.71 |
| S->N | 55 | +1.68 | +33.0 | -22.71 |
| N->S | 32 | -5.23 | -59.7 | +34.24 |
| S->O | 27 | +0.83 | +7.3 | -12.08 |
| O->S | 23 | -4.20 | -42.7 | +17.77 |
| N->O | 4 | +3.79 | +17.8 | +1.89 |
| O->N | 2 | +4.12 | +117.0 | -23.25 |
