# D7-case2 CPU selection

Global val_unseen length tertiles: [8.236761794539953, 10.376267476756617] m.

## Confirmed failure type × path-length tertile (remaining cases unknown)

| Type | T1 | T2 | T3 | Total |
|---|---:|---:|---:|---:|
| stop-3to6 | 6 | 34 | 27 | 67 |
| stop-over6 | 16 | 11 | 22 | 49 |
| timeout-highlevel | 12 | 13 | 13 | 38 |
| timeout-lowlevel | 0 | 0 | 0 | 0 |
| other | 0 | 1 | 0 | 1 |
| unknown-stop-type | 0 | 0 | 0 | 0 |

steps_taken is low-level. A value <5000 does not prove voluntary STOP: the policy has a separate 15 high-level-step cap. Only trajectory dumps distinguish argmax STOP from forced STOP.

baseline/D numeric metrics retain the original full1839 deploy-zero evaluations. Confirmed classification and selection use fresh module-removed D and baseline case trajectory dumps. These are different evaluations; reeval_valid explicitly checks the case outcomes.

## Validation examples

```json
{
  "1813": {
    "last_step": 8,
    "argmax_id": null,
    "forced_stop": false,
    "voluntary_stop_before_cap": true,
    "distance_to_goal_m": 3.3590826988220215,
    "divergence_decision_pair": [
      8,
      8
    ]
  },
  "7": {
    "last_step": 4,
    "argmax_id": null,
    "forced_stop": false,
    "voluntary_stop_before_cap": true,
    "distance_to_goal_m": 4.138576507568359,
    "divergence_decision_pair": [
      4,
      4
    ]
  },
  "978": {
    "last_step": 6,
    "argmax_id": null,
    "forced_stop": false,
    "voluntary_stop_before_cap": true,
    "distance_to_goal_m": 4.016384124755859,
    "divergence_decision_pair": [
      5,
      5
    ]
  },
  "1105": {
    "last_step": 14,
    "argmax_id": "g31",
    "forced_stop": true,
    "voluntary_stop_before_cap": false,
    "distance_to_goal_m": 20.197202682495117,
    "divergence_decision_pair": [
      0,
      0
    ]
  },
  "906": {
    "last_step": 8,
    "argmax_id": null,
    "forced_stop": false,
    "voluntary_stop_before_cap": true,
    "distance_to_goal_m": 3.42218017578125,
    "divergence_decision_pair": [
      0,
      0
    ]
  },
  "1134": {
    "last_step": 9,
    "argmax_id": null,
    "forced_stop": false,
    "voluntary_stop_before_cap": true,
    "distance_to_goal_m": 11.510783195495605,
    "divergence_decision_pair": [
      0,
      0
    ]
  }
}
```

## Provisional stats-only categories (NOT verified voluntary STOP)

| Category | T1 | T2 | T3 | Total |
|---|---:|---:|---:|---:|
| stop-3to6-candidate | 9 | 35 | 28 | 72 |
| stop-over6-candidate | 25 | 24 | 34 | 83 |
| timeout-lowlevel | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 0 |

## Selected cases

| Group | Episode | GT length | Junctions | D PL | D steps | Decision pair |
|---|---|---:|---:|---:|---:|---|
| start_grounding | 906 | 16.93 | 7 | 15.54 | 104 | [0, 0] |
| start_grounding | 872 | 13.82 | 7 | 13.42 | 91 | [0, 1] |
| start_grounding | 1225 | 10.52 | 5 | 10.77 | 73 | [1, 0] |
| start_grounding | 1134 | 13.06 | 4 | 14.25 | 83 | [0, 0] |
| start_grounding | 1105 | 10.08 | 5 | 7.50 | 45 | [0, 0] |
| mid_junction | 302 | 10.45 | 5 | 6.25 | 48 | [4, 4] |
| mid_junction | 978 | 10.16 | 4 | 9.07 | 94 | [6, 5] |
| mid_junction | 1033 | 12.02 | 4 | 11.85 | 110 | [3, 3] |
| early_stop | 1706 | 13.26 | 5 | 11.57 | 95 | [8, 6] |
| early_stop | 1813 | 15.55 | 7 | 15.14 | 72 | [8, 8] |
| early_stop | 1263 | 10.82 | 6 | 8.22 | 63 | [4, 4] |
| early_stop | 1709 | 10.31 | 5 | 9.76 | 93 | [6, 6] |
| early_stop | 740 | 14.47 | 4 | 13.51 | 83 | [8, 7] |

Pending slots: {'start_grounding': 0, 'mid_junction': 2, 'early_stop': 0}
Supplemental dump episode IDs: ['1648', '1224', '889', '1351', '125', '802', '1353', '1390', '996', '1318', '776', '1487', '579', '1152', '1731', '775', '1072', '1218', '1319', '1729', '803', '315', '1393', '1156', '1477', '876', '875', '69', '905', '643']

## All rescued episodes

| Episode | Type | Tertile | GT length | D PL | D steps |
|---|---|---:|---:|---:|---:|
| 7 | stop-3to6 | 3 | 10.55 | 9.75 | 45 |
| 16 | stop-over6 | 2 | 8.77 | 8.00 | 64 |
| 46 | stop-3to6 | 3 | 11.45 | 9.42 | 79 |
| 69 | timeout-highlevel | 3 | 13.49 | 26.15 | 172 |
| 85 | stop-over6 | 2 | 9.13 | 13.24 | 114 |
| 97 | stop-over6 | 1 | 7.76 | 52.95 | 156 |
| 107 | stop-3to6 | 2 | 10.07 | 5.65 | 67 |
| 112 | stop-3to6 | 2 | 10.20 | 10.27 | 87 |
| 125 | stop-3to6 | 3 | 12.68 | 11.44 | 117 |
| 131 | timeout-highlevel | 1 | 7.79 | 16.90 | 94 |
| 175 | timeout-highlevel | 1 | 7.75 | 10.96 | 64 |
| 177 | timeout-highlevel | 1 | 7.75 | 13.17 | 49 |
| 205 | stop-over6 | 3 | 11.51 | 11.06 | 81 |
| 237 | stop-3to6 | 2 | 9.94 | 8.00 | 64 |
| 265 | stop-over6 | 1 | 8.15 | 15.25 | 70 |
| 276 | stop-3to6 | 2 | 9.43 | 7.96 | 69 |
| 302 | timeout-highlevel | 3 | 10.45 | 6.25 | 48 |
| 315 | stop-3to6 | 3 | 10.40 | 17.22 | 108 |
| 316 | timeout-highlevel | 2 | 10.13 | 18.10 | 119 |
| 317 | timeout-highlevel | 2 | 10.13 | 18.10 | 119 |
| 318 | timeout-highlevel | 2 | 10.13 | 17.03 | 114 |
| 322 | stop-over6 | 1 | 8.23 | 8.46 | 40 |
| 323 | stop-over6 | 1 | 8.23 | 9.99 | 47 |
| 351 | stop-over6 | 3 | 14.67 | 18.66 | 136 |
| 356 | stop-over6 | 2 | 9.71 | 9.51 | 86 |
| 377 | stop-3to6 | 2 | 8.30 | 5.25 | 50 |
| 403 | stop-3to6 | 2 | 8.91 | 10.25 | 94 |
| 411 | stop-over6 | 1 | 7.69 | 8.48 | 57 |
| 436 | timeout-highlevel | 2 | 8.48 | 11.56 | 78 |
| 443 | stop-3to6 | 3 | 11.66 | 19.60 | 152 |
| 444 | stop-over6 | 3 | 11.66 | 17.72 | 151 |
| 460 | timeout-highlevel | 1 | 5.37 | 6.33 | 71 |
| 462 | timeout-highlevel | 1 | 5.37 | 8.34 | 77 |
| 532 | stop-3to6 | 2 | 9.65 | 13.11 | 106 |
| 533 | timeout-highlevel | 2 | 9.65 | 9.02 | 83 |
| 571 | stop-3to6 | 3 | 10.84 | 12.50 | 96 |
| 579 | stop-over6 | 3 | 15.93 | 17.24 | 102 |
| 592 | stop-over6 | 1 | 8.10 | 21.45 | 140 |
| 593 | timeout-highlevel | 1 | 8.10 | 21.69 | 143 |
| 594 | timeout-highlevel | 1 | 8.10 | 9.50 | 81 |
| 609 | timeout-highlevel | 2 | 10.15 | 11.24 | 62 |
| 616 | stop-3to6 | 2 | 9.54 | 10.55 | 52 |
| 624 | stop-3to6 | 3 | 11.65 | 11.12 | 98 |
| 643 | stop-over6 | 3 | 10.77 | 47.70 | 202 |
| 662 | stop-3to6 | 2 | 9.68 | 12.81 | 109 |
| 667 | timeout-highlevel | 3 | 13.90 | 13.63 | 92 |
| 670 | stop-3to6 | 2 | 9.51 | 7.80 | 66 |
| 672 | stop-3to6 | 2 | 9.51 | 7.80 | 66 |
| 706 | stop-3to6 | 2 | 9.29 | 10.49 | 90 |
| 717 | stop-3to6 | 3 | 12.49 | 11.75 | 94 |
| 725 | timeout-highlevel | 1 | 8.15 | 21.74 | 153 |
| 730 | stop-3to6 | 3 | 10.88 | 18.47 | 107 |
| 739 | timeout-highlevel | 3 | 14.47 | 13.74 | 79 |
| 740 | stop-3to6 | 3 | 14.47 | 13.51 | 83 |
| 768 | stop-3to6 | 2 | 9.30 | 7.79 | 68 |
| 775 | timeout-highlevel | 3 | 10.64 | 13.07 | 110 |
| 776 | timeout-highlevel | 3 | 10.64 | 15.80 | 122 |
| 778 | stop-3to6 | 2 | 10.09 | 18.25 | 116 |
| 780 | timeout-highlevel | 2 | 10.09 | 27.22 | 161 |
| 799 | stop-3to6 | 2 | 8.80 | 12.00 | 107 |
| 802 | stop-over6 | 3 | 12.94 | 10.25 | 98 |
| 803 | stop-over6 | 3 | 12.94 | 15.25 | 157 |
| 824 | stop-over6 | 1 | 6.54 | 6.57 | 28 |
| 825 | stop-over6 | 1 | 6.54 | 8.89 | 40 |
| 843 | stop-over6 | 3 | 12.92 | 10.57 | 106 |
| 846 | stop-over6 | 2 | 10.08 | 20.09 | 148 |
| 872 | stop-over6 | 3 | 13.82 | 13.42 | 91 |
| 875 | timeout-highlevel | 3 | 12.50 | 30.39 | 147 |
| 876 | timeout-highlevel | 3 | 12.50 | 16.76 | 95 |
| 881 | timeout-highlevel | 1 | 7.99 | 8.23 | 60 |
| 889 | stop-over6 | 3 | 15.46 | 16.23 | 121 |
| 905 | stop-3to6 | 3 | 16.93 | 19.97 | 133 |
| 906 | stop-3to6 | 3 | 16.93 | 15.54 | 104 |
| 919 | stop-3to6 | 2 | 9.15 | 9.98 | 92 |
| 978 | stop-3to6 | 2 | 10.16 | 9.07 | 94 |
| 980 | stop-over6 | 1 | 6.33 | 5.27 | 55 |
| 996 | stop-3to6 | 3 | 13.53 | 20.61 | 198 |
| 1005 | stop-over6 | 1 | 7.91 | 9.55 | 63 |
| 1027 | stop-over6 | 2 | 8.32 | 9.26 | 69 |
| 1033 | stop-3to6 | 3 | 12.02 | 11.85 | 110 |
| 1052 | stop-3to6 | 3 | 11.89 | 8.92 | 90 |
| 1053 | stop-3to6 | 3 | 11.89 | 8.92 | 90 |
| 1061 | timeout-highlevel | 2 | 9.33 | 7.75 | 64 |
| 1072 | stop-over6 | 3 | 10.60 | 15.87 | 136 |
| 1100 | stop-3to6 | 2 | 9.62 | 6.75 | 55 |
| 1105 | timeout-highlevel | 2 | 10.08 | 7.50 | 45 |
| 1133 | stop-over6 | 3 | 13.06 | 10.00 | 61 |
| 1134 | stop-over6 | 3 | 13.06 | 14.25 | 83 |
| 1148 | stop-3to6 | 2 | 8.40 | 17.75 | 103 |
| 1149 | stop-3to6 | 2 | 8.40 | 17.75 | 103 |
| 1152 | stop-3to6 | 3 | 13.22 | 24.70 | 123 |
| 1156 | stop-over6 | 3 | 18.07 | 21.47 | 181 |
| 1169 | timeout-highlevel | 2 | 9.43 | 6.50 | 65 |
| 1170 | stop-3to6 | 2 | 9.43 | 6.50 | 65 |
| 1189 | stop-over6 | 2 | 10.35 | 10.00 | 78 |
| 1203 | timeout-highlevel | 2 | 10.08 | 7.25 | 73 |
| 1218 | stop-over6 | 3 | 11.56 | 24.03 | 134 |
| 1224 | stop-over6 | 3 | 14.79 | 13.91 | 81 |
| 1225 | stop-over6 | 3 | 10.52 | 10.77 | 73 |
| 1229 | stop-over6 | 2 | 8.47 | 15.75 | 101 |
| 1263 | stop-3to6 | 3 | 10.82 | 8.22 | 63 |
| 1271 | stop-over6 | 1 | 8.13 | 22.51 | 97 |
| 1289 | stop-3to6 | 2 | 10.31 | 9.41 | 53 |
| 1299 | stop-over6 | 3 | 10.58 | 11.11 | 61 |
| 1318 | stop-3to6 | 3 | 10.44 | 7.53 | 86 |
| 1319 | stop-3to6 | 3 | 10.44 | 12.85 | 115 |
| 1321 | stop-over6 | 2 | 10.34 | 6.50 | 89 |
| 1328 | stop-3to6 | 2 | 10.23 | 8.25 | 72 |
| 1336 | stop-over6 | 1 | 7.97 | 5.50 | 62 |
| 1338 | stop-3to6 | 1 | 7.97 | 5.50 | 62 |
| 1342 | timeout-highlevel | 2 | 9.65 | 12.98 | 106 |
| 1351 | stop-over6 | 3 | 13.39 | 18.27 | 129 |
| 1353 | timeout-highlevel | 3 | 13.39 | 16.51 | 117 |
| 1375 | stop-3to6 | 2 | 9.13 | 9.25 | 52 |
| 1383 | stop-over6 | 2 | 8.85 | 6.75 | 68 |
| 1390 | stop-over6 | 3 | 16.22 | 14.86 | 115 |
| 1393 | timeout-highlevel | 3 | 16.11 | 18.25 | 173 |
| 1406 | timeout-highlevel | 2 | 8.30 | 24.50 | 116 |
| 1434 | stop-3to6 | 2 | 8.92 | 10.98 | 92 |
| 1438 | stop-over6 | 1 | 7.18 | 4.76 | 33 |
| 1439 | stop-3to6 | 1 | 7.18 | 4.76 | 33 |
| 1457 | stop-3to6 | 2 | 9.27 | 7.05 | 75 |
| 1463 | stop-3to6 | 2 | 9.06 | 14.76 | 87 |
| 1471 | timeout-highlevel | 3 | 11.26 | 11.00 | 99 |
| 1477 | timeout-highlevel | 3 | 11.33 | 25.36 | 145 |
| 1480 | stop-3to6 | 1 | 8.23 | 6.50 | 64 |
| 1487 | stop-3to6 | 3 | 10.38 | 14.48 | 97 |
| 1502 | stop-3to6 | 2 | 8.26 | 5.22 | 58 |
| 1503 | stop-3to6 | 2 | 8.26 | 6.47 | 70 |
| 1508 | stop-3to6 | 2 | 9.20 | 10.06 | 63 |
| 1531 | timeout-highlevel | 1 | 6.44 | 8.41 | 94 |
| 1533 | timeout-highlevel | 1 | 6.44 | 7.41 | 115 |
| 1552 | stop-3to6 | 1 | 6.08 | 7.43 | 75 |
| 1560 | stop-over6 | 1 | 5.04 | 5.38 | 41 |
| 1588 | stop-over6 | 2 | 9.69 | 9.82 | 70 |
| 1615 | stop-3to6 | 3 | 10.66 | 17.00 | 106 |
| 1621 | other | 2 | 8.66 | 9.73 | 104 |
| 1648 | stop-3to6 | 3 | 11.75 | 11.71 | 76 |
| 1682 | stop-3to6 | 3 | 10.84 | 17.26 | 91 |
| 1685 | stop-over6 | 3 | 15.09 | 12.78 | 94 |
| 1690 | stop-over6 | 2 | 8.88 | 18.65 | 100 |
| 1706 | stop-3to6 | 3 | 13.26 | 11.57 | 95 |
| 1709 | stop-3to6 | 2 | 10.31 | 9.76 | 93 |
| 1719 | stop-3to6 | 3 | 11.37 | 9.64 | 75 |
| 1729 | stop-over6 | 3 | 11.39 | 17.53 | 171 |
| 1731 | timeout-highlevel | 3 | 11.39 | 18.15 | 152 |
| 1766 | stop-3to6 | 2 | 9.38 | 8.00 | 61 |
| 1772 | stop-3to6 | 2 | 10.27 | 21.06 | 141 |
| 1792 | stop-3to6 | 2 | 9.12 | 9.21 | 90 |
| 1804 | stop-3to6 | 1 | 8.13 | 6.75 | 45 |
| 1813 | stop-3to6 | 3 | 15.55 | 15.14 | 72 |
| 1818 | stop-over6 | 1 | 5.41 | 5.50 | 41 |
| 1832 | stop-3to6 | 1 | 7.40 | 10.10 | 107 |
| 1835 | timeout-highlevel | 1 | 7.96 | 6.21 | 55 |
| 1836 | stop-over6 | 1 | 7.96 | 6.46 | 64 |
