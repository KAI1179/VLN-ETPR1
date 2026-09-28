# Paired comparison on val_unseen: A (gt_teacher_try5_val_unseen_z_none) vs B (dagger_distill_gt_teacher_llmpt_eval_iter17000_p0)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 73.84 | 66.72 | -7.12 | [-11.80, -0.81] |
| SPL | 63.38 | 56.02 | -7.36 | [-11.70, -1.12] |
| OSR | 77.49 | 72.43 | -5.06 | [-9.93, +1.69] |
| nDTW | 70.41 | 65.63 | -4.78 | [-7.61, -0.96] |
| NE | 3.29 | 3.82 | +0.52 | [-0.04, +1.09] |
| PL | 12.15 | 12.83 | +0.68 | [-0.01, +1.40] |
| Steps | 87.41 | 91.53 | +4.12 | [-2.45, +9.82] |

success discordant pairs: A-only 275, B-only 144, exact McNemar p = 0.0000

stop error (OSR - SR): A 3.64 pp, B 5.71 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1083 | 68 | 207 |
| O | 34 | 23 | 10 |
| N | 110 | 14 | 290 |

net O->S (stop quality) = -34 episodes (-1.85 pp); net N->reach (reachability) = -93 episodes (-5.06 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1083 | +0.44 | +2.4 | -2.10 |
| N->N | 290 | +1.52 | +14.2 | -0.49 |
| S->N | 207 | +2.07 | +25.9 | -42.82 |
| N->S | 110 | -2.26 | -43.5 | +36.93 |
| S->O | 68 | +4.81 | +31.9 | -30.88 |
| O->S | 34 | -6.32 | -58.8 | +25.23 |
| O->O | 23 | +0.25 | +8.7 | -11.98 |
| N->O | 14 | +6.67 | +17.4 | +1.79 |
| O->N | 10 | -6.47 | -35.3 | -8.76 |
