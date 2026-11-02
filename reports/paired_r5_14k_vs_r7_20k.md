# Paired comparison on val_unseen: A (dagger_distill_gt_teacher_eval_iter14000_val_unseen) vs B (dagger_distill_gt_teacher_llmpt_eval_iter20000_p0)

episodes paired: 1839; scenes: 11; bootstrap reps: 10000

| metric | A | B | B - A | 95% CI (scene bootstrap) |
|---|---:|---:|---:|---:|
| SR | 63.62 | 65.25 | +1.63 | [-0.17, +4.00] |
| SPL | 53.08 | 55.56 | +2.49 | [+0.93, +4.62] |
| OSR | 69.60 | 70.91 | +1.31 | [-0.53, +3.37] |
| nDTW | 63.85 | 65.06 | +1.21 | [-0.22, +3.11] |
| NE | 4.00 | 4.03 | +0.03 | [-0.10, +0.13] |
| PL | 13.14 | 12.83 | -0.31 | [-1.00, +0.28] |
| Steps | 95.88 | 90.34 | -5.54 | [-11.88, -0.42] |

success discordant pairs: A-only 165, B-only 195, exact McNemar p = 0.1263

stop error (OSR - SR): A 5.98 pp, B 5.66 pp

## S/O/N transitions (rows = A state, cols = B state)

| A \ B | S | O | N |
|---|---:|---:|---:|
| S | 1005 | 48 | 117 |
| O | 47 | 43 | 20 |
| N | 148 | 13 | 398 |

net O->S (stop quality) = -1 episodes (-0.05 pp); net N->reach (reachability) = +24 episodes (+1.31 pp)

## Path-length / step deltas by transition

| transition | n | mean dPL (m) | mean dSteps | mean dnDTW |
|---|---:|---:|---:|---:|
| S->S | 1005 | -0.54 | -5.8 | +1.75 |
| N->N | 398 | +0.02 | -4.7 | +1.08 |
| N->S | 148 | -3.28 | -41.8 | +35.53 |
| S->N | 117 | +1.82 | +26.0 | -39.48 |
| S->O | 48 | +2.69 | +15.9 | -22.93 |
| O->S | 47 | -3.42 | -38.3 | +20.73 |
| O->O | 43 | +3.85 | +26.3 | -6.46 |
| O->N | 20 | -1.79 | -15.1 | -11.61 |
| N->O | 13 | +10.70 | +67.8 | +2.81 |
