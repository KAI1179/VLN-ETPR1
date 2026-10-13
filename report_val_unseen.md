# Paired-episode report (val_unseen)

Own rollouts: 1833 episodes common to all runs.

## 0. Feature distributions (common episodes)

| feature | n | mean | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|---|---|
| path_len_m | 1833 | 9.16 | 5.81 | 7.25 | 8.75 | 10.55 | 13.04 |
| n_turns | 1833 | 0.80 | 0.00 | 0.00 | 1.00 | 1.00 | 2.00 |
| n_forks_graph | 1833 | 4.53 | 2.00 | 3.00 | 4.00 | 6.00 | 7.00 |
| n_forks3_graph | 1833 | 1.91 | 0.00 | 1.00 | 2.00 | 3.00 | 4.00 |
| n_room_changes | 1833 | 2.13 | 0.00 | 1.00 | 2.00 | 3.00 | 4.00 |
| n_landmarks | 1833 | 1.27 | 0.00 | 1.00 | 1.00 | 2.00 | 3.00 |
| n_decoys | 1833 | 8.12 | 0.00 | 0.00 | 6.00 | 14.00 | 21.00 |
| n_decoy_categories | 1833 | 0.98 | 0.00 | 0.00 | 1.00 | 1.00 | 2.00 |
| n_spatial_words | 1833 | 5.22 | 2.00 | 3.00 | 5.00 | 7.00 | 9.00 |
| llm_region_hit | 1833 | 0.49 | 0.12 | 0.25 | 0.43 | 0.74 | 1.00 |

## 1. Own rollouts on common episodes

| run | n | SR | SPL | OSR | nDTW | OSR-SR | steps | PL (m) |
|---|---|---|---|---|---|---|---|---|
| base | 1833 | 63.18 | 53.43 | 68.69 | 64.32 | 5.51 | 99.3 | 13.76 |
| try5 | 1833 | 65.25 | 55.19 | 70.59 | 64.27 | 5.35 | 99.8 | 14.07 |
| kd | 1833 | 67.48 | 56.66 | 72.34 | 65.62 | 4.86 | 95.0 | 13.58 |

| pair | both succeed | base only | other only | neither | net |
|---|---|---|---|---|---|
| base vs try5 | 988 | 170 | 208 | 467 | 38 |
| base vs kd | 1007 | 151 | 230 | 445 | 79 |

## 2. Hard subsets (GT-derived strata, own rollouts)

| subset | n | base SR | try5 SR | kd SR | base SPL | try5 SPL | kd SPL | try5 - base SR | kd - base SR |
|---|---|---|---|---|---|---|---|---|---|
| all | 1833 | 63.18 | 65.25 | 67.48 | 53.43 | 55.19 | 56.66 | +2.07 | +4.31 |
| multi-fork (n_forks3_graph>=2) | 1057 | 61.87 | 63.39 | 65.56 | 52.62 | 53.47 | 55.09 | +1.51 | +3.69 |
| few forks (n_forks3_graph<2) | 776 | 64.95 | 67.78 | 70.10 | 54.54 | 57.53 | 58.80 | +2.84 | +5.15 |
| multi-turn (n_turns>=2) | 379 | 59.10 | 63.59 | 63.32 | 48.17 | 52.55 | 52.71 | +4.49 | +4.22 |
| few turns (n_turns<2) | 1454 | 64.24 | 65.68 | 68.57 | 54.80 | 55.88 | 57.69 | +1.44 | +4.33 |
| multi-room (n_room_changes>=2) | 1312 | 61.59 | 64.02 | 65.32 | 52.01 | 53.98 | 55.11 | +2.44 | +3.73 |
| decoys (n_decoy_categories>=2) | 414 | 59.18 | 58.94 | 64.25 | 50.71 | 50.74 | 54.54 | -0.24 | +5.07 |
| long path (>= 10.0 m) | 605 | 45.95 | 46.28 | 48.43 | 39.77 | 39.31 | 40.97 | +0.33 | +2.48 |
| short path (< 10.0 m) | 1228 | 71.66 | 74.59 | 76.87 | 60.16 | 63.02 | 64.39 | +2.93 | +5.21 |
| spatial language (words>=5) | 1014 | 62.82 | 65.58 | 67.75 | 52.79 | 55.99 | 57.24 | +2.76 | +4.93 |
| LLM map good (hit>=0.43) | 923 | 67.93 | 68.69 | 69.66 | 56.65 | 57.73 | 58.54 | +0.76 | +1.73 |
| LLM map poor (hit<0.43) | 910 | 58.35 | 61.76 | 65.27 | 50.17 | 52.62 | 54.75 | +3.41 | +6.92 |

## 3. Mean features of discordant episodes

| pair | succeeds | n | path_len_m | n_turns | n_forks_graph | n_forks3_graph | n_room_changes | n_landmarks | n_decoys | n_decoy_categories | n_spatial_words | llm_region_hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base vs try5 | base only | 170 | 9.55 | 0.66 | 4.44 | 1.92 | 2.14 | 1.47 | 9.22 | 1.12 | 5.32 | 0.45 |
| base vs try5 | try5 only | 208 | 9.24 | 0.79 | 4.50 | 1.83 | 2.20 | 1.29 | 7.59 | 1.02 | 5.48 | 0.45 |
| base vs try5 | both | 988 | 8.33 | 0.78 | 4.32 | 1.81 | 2.07 | 1.18 | 7.91 | 0.92 | 5.15 | 0.52 |
| base vs try5 | neither | 467 | 10.74 | 0.91 | 5.02 | 2.15 | 2.23 | 1.35 | 8.42 | 1.04 | 5.22 | 0.44 |
| base vs kd | base only | 151 | 9.79 | 0.72 | 4.54 | 1.99 | 2.22 | 1.33 | 8.77 | 1.06 | 5.25 | 0.45 |
| base vs kd | kd only | 230 | 9.34 | 0.77 | 4.52 | 1.87 | 2.12 | 1.30 | 7.68 | 1.02 | 5.26 | 0.43 |
| base vs kd | both | 1007 | 8.32 | 0.76 | 4.30 | 1.80 | 2.06 | 1.21 | 8.00 | 0.93 | 5.16 | 0.52 |
| base vs kd | neither | 445 | 10.77 | 0.93 | 5.04 | 2.15 | 2.27 | 1.35 | 8.41 | 1.04 | 5.32 | 0.45 |

## 4. Failure modes (own rollouts)

stop_error: reached the goal area but stopped elsewhere; wrong_turn: first disagreement with the expert at a turn step; off_path: first disagreement at a straight or backtrack step; other: never disagreed with the expert yet failed.

| run | success | stop_error | wrong_turn | off_path | other |
|---|---|---|---|---|---|
| base | 1158 | 101 | 285 | 287 | 2 |
| try5 | 1196 | 98 | 269 | 267 | 3 |
| kd | 1237 | 89 | 264 | 241 | 2 |

## 5. Efficiency on episodes every run solves

| run | n | low-level steps | PL (m) | backtracks | SPL |
|---|---|---|---|---|---|
| base | 924 | 68.85 | 10.18 | 0.38 | 86.05 |
| try5 | 924 | 67.40 | 10.20 | 0.34 | 87.10 |
| kd | 924 | 65.73 | 10.00 | 0.28 | 86.95 |

## 6a. Decision accuracy vs expert, own rollouts

| run | stop (n / acc%) | backtrack (n / acc%) | turn (n / acc%) | straight (n / acc%) | turn, >=3 options | all moves |
|---|---|---|---|---|---|---|
| base | 1143 / 58.53 | 4053 / 8.22 | 4136 / 63.68 | 6474 / 79.67 | 2308 / 62.00 | 14663 / 55.41 |
| try5 | 1229 / 60.70 | 3934 / 9.18 | 4096 / 64.55 | 6545 / 79.79 | 2271 / 61.96 | 14575 / 56.45 |
| kd | 1323 / 60.62 | 3698 / 9.41 | 4140 / 65.27 | 6596 / 81.08 | 2333 / 64.04 | 14434 / 58.18 |

## 6b. Decision accuracy vs expert, shared states (1833 episodes)

| run | stop (n / acc%) | backtrack (n / acc%) | turn (n / acc%) | straight (n / acc%) | turn, >=3 options | all moves |
|---|---|---|---|---|---|---|
| base | 1767 / 64.35 | 291 / 20.62 | 3940 / 69.29 | 7928 / 74.00 | 2198 / 67.88 | 12159 / 71.20 |
| try5 | 1767 / 64.18 | 291 / 20.96 | 3940 / 68.81 | 7928 / 74.26 | 2198 / 67.33 | 12159 / 71.21 |
| kd | 1767 / 63.50 | 291 / 22.34 | 3940 / 70.30 | 7928 / 75.63 | 2198 / 69.15 | 12159 / 72.63 |

Per-state 2x2 on identical states:

State check: 27984 step pairs compared, 0 episode(s) cut at a state mismatch (position, graph or reference action differ; expected 0 under D8 replay).

| pair | category | n | both right | base only | other only | neither | net |
|---|---|---|---|---|---|---|---|
| base vs try5 | backtrack | 291 | 39 | 21 | 22 | 209 | 1 |
| base vs try5 | turn | 3940 | 2431 | 299 | 280 | 930 | -19 |
| base vs try5 | straight | 7928 | 5343 | 524 | 544 | 1517 | 20 |
| base vs try5 | all moves | 12159 | 7813 | 844 | 846 | 2656 | 2 |
| base vs kd | backtrack | 291 | 40 | 20 | 25 | 206 | 5 |
| base vs kd | turn | 3940 | 2480 | 250 | 290 | 920 | 40 |
| base vs kd | straight | 7928 | 5413 | 454 | 583 | 1478 | 129 |
| base vs kd | all moves | 12159 | 7933 | 724 | 898 | 2604 | 174 |

## 7. First divergence between own rollouts

The first step where the two runs, on the same state, execute different actions.

| pair | who succeeds | step category | who agreed with expert there | n |
|---|---|---|---|---|
| base vs try5 | identical until one run ends |  |  | 457 |
| base vs try5 | both | straight | try5 = expert | 141 |
| base vs try5 | neither | backtrack | neither = expert | 134 |
| base vs try5 | both | straight | base = expert | 98 |
| base vs try5 | both | turn | base = expert | 96 |
| base vs try5 | both | turn | try5 = expert | 89 |
| base vs try5 | try5 only | straight | try5 = expert | 60 |
| base vs try5 | base only | straight | base = expert | 50 |
| base vs try5 | neither | turn | try5 = expert | 48 |
| base vs try5 | base only | turn | base = expert | 48 |
| base vs try5 | both | stop | base = expert | 47 |
| base vs try5 | neither | turn | base = expert | 45 |
| base vs try5 | try5 only | turn | try5 = expert | 43 |
| base vs try5 | neither | straight | try5 = expert | 42 |
| base vs try5 | both | stop | try5 = expert | 41 |
| base vs try5 | neither | turn | neither = expert | 37 |
| base vs try5 | neither | straight | base = expert | 29 |
| base vs try5 | both | stop | neither = expert | 25 |
| base vs try5 | try5 only | turn | neither = expert | 24 |
| base vs try5 | try5 only | straight | neither = expert | 22 |
| base vs try5 | neither | straight | neither = expert | 20 |
| base vs try5 | both | turn | neither = expert | 20 |
| base vs try5 | neither | backtrack | base = expert | 18 |
| base vs try5 | try5 only | turn | base = expert | 17 |
| base vs try5 | base only | turn | neither = expert | 17 |
| base vs try5 | base only | backtrack | neither = expert | 16 |
| base vs try5 | neither | backtrack | try5 = expert | 15 |
| base vs try5 | both | straight | neither = expert | 15 |
| base vs try5 | try5 only | straight | base = expert | 15 |
| base vs try5 | both | backtrack | neither = expert | 14 |
| base vs try5 | try5 only | backtrack | neither = expert | 13 |
| base vs try5 | try5 only | backtrack | try5 = expert | 12 |
| base vs try5 | base only | turn | try5 = expert | 11 |
| base vs try5 | base only | straight | try5 = expert | 10 |
| base vs try5 | base only | backtrack | base = expert | 8 |
| base vs try5 | both | backtrack | base = expert | 6 |
| base vs try5 | base only | straight | neither = expert | 5 |
| base vs try5 | both | backtrack | try5 = expert | 5 |
| base vs try5 | neither | turn | both = expert | 4 |
| base vs try5 | state differs first |  |  | 3 |
| base vs try5 | neither | cut | neither = expert | 3 |
| base vs try5 | base only | stop | base = expert | 2 |
| base vs try5 | neither | stop | neither = expert | 2 |
| base vs try5 | base only | backtrack | try5 = expert | 2 |
| base vs try5 | try5 only | backtrack | base = expert | 1 |
| base vs try5 | try5 only | stop | neither = expert | 1 |
| base vs try5 | both | turn | both = expert | 1 |
| base vs try5 | base only | stop | neither = expert | 1 |
| base vs kd | identical until one run ends |  |  | 465 |
| base vs kd | both | straight | kd = expert | 145 |
| base vs kd | neither | backtrack | neither = expert | 128 |
| base vs kd | both | straight | base = expert | 93 |
| base vs kd | both | turn | kd = expert | 91 |
| base vs kd | both | turn | base = expert | 83 |
| base vs kd | kd only | straight | kd = expert | 76 |
| base vs kd | base only | straight | base = expert | 54 |
| base vs kd | both | stop | base = expert | 51 |
| base vs kd | kd only | turn | kd = expert | 48 |
| base vs kd | both | stop | kd = expert | 47 |
| base vs kd | neither | turn | neither = expert | 40 |
| base vs kd | neither | turn | base = expert | 40 |
| base vs kd | neither | turn | kd = expert | 36 |
| base vs kd | neither | straight | kd = expert | 35 |
| base vs kd | base only | turn | base = expert | 34 |
| base vs kd | neither | straight | base = expert | 29 |
| base vs kd | neither | straight | neither = expert | 27 |
| base vs kd | both | turn | neither = expert | 27 |
| base vs kd | both | straight | neither = expert | 26 |
| base vs kd | kd only | turn | base = expert | 24 |
| base vs kd | kd only | straight | neither = expert | 18 |
| base vs kd | both | stop | neither = expert | 18 |
| base vs kd | kd only | backtrack | kd = expert | 18 |
| base vs kd | kd only | turn | neither = expert | 17 |
| base vs kd | neither | backtrack | base = expert | 16 |
| base vs kd | both | backtrack | neither = expert | 14 |
| base vs kd | neither | backtrack | kd = expert | 13 |
| base vs kd | kd only | backtrack | neither = expert | 13 |
| base vs kd | base only | turn | neither = expert | 13 |
| base vs kd | base only | backtrack | neither = expert | 13 |
| base vs kd | kd only | straight | base = expert | 11 |
| base vs kd | both | backtrack | base = expert | 8 |
| base vs kd | base only | straight | kd = expert | 7 |
| base vs kd | base only | backtrack | base = expert | 7 |
| base vs kd | base only | turn | kd = expert | 6 |
| base vs kd | base only | stop | neither = expert | 5 |
| base vs kd | both | backtrack | kd = expert | 5 |
| base vs kd | base only | straight | neither = expert | 5 |
| base vs kd | neither | stop | neither = expert | 4 |
| base vs kd | base only | stop | base = expert | 4 |
| base vs kd | kd only | stop | neither = expert | 4 |
| base vs kd | both | turn | both = expert | 3 |
| base vs kd | state differs first |  |  | 3 |
| base vs kd | base only | backtrack | kd = expert | 3 |
| base vs kd | neither | turn | both = expert | 2 |
| base vs kd | neither | cut | neither = expert | 2 |
| base vs kd | kd only | stop | kd = expert | 1 |
| base vs kd | both | straight | both = expert | 1 |
