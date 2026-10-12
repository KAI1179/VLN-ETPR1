# D7-d try5

## Definitions

```json
{
  "limits": {
    "text_tokens": 150,
    "high_level_steps": 15,
    "low_level_steps": 5000
  },
  "categories": {
    "A": "success",
    "B": "oracle success but failure",
    "C": "stop and never reached",
    "D": "budget and never reached",
    "E": "no_candidate and never reached",
    "U": "not executed"
  },
  "divergence": "first non-STOP selected target > threshold metres from all GT reference_path nodes; 3D Euclidean",
  "branch": "nearest original included connectivity node, <=3m; >=2 included unobstructed off-GT neighbors excluding logged incoming_position (nearest graph node), or previous distinct current node when incoming_position is null",
  "coordinate": "[pose[3], pose[11]-height, -pose[7]]; prior/etp_r1/__init__.py ConnectivityEntry.position",
  "termination_priority": "budget > no_candidate > stop; all simultaneous flags remain in trajectories",
  "proxy_confusion": "same new evaluation: zero-based final step +1=15 maps to D, earlier maps to C",
  "historical_confusion": "previous saved D7-b proxy categories vs new true categories; includes rerun/rank-induced behavior differences, not pure proxy error",
  "bootstrap": "2000 scene-cluster draws seed0; step mean difference uses matched finite episodes; branch conditional proportions and step quantiles use separate model denominators under shared scene resampling; never proportion over all subgroup episodes"
}
```

## Paired metrics

| Metric | Baseline | Ours | Difference | 95% CI |
| --- | --- | --- | --- | --- |
| success | 63.02338227297444 | 65.03534529635672 | 2.011963023382273 | [-0.04353884011043026, 4.723079041280542] |
| oracle_success | 68.51549755301795 | 70.36432843936922 | 1.8488308863512777 | [0.47343462024773914, 3.3717566919969855] |
| spl | 53.312785633214276 | 55.0116095097563 | 1.6988238765420223 | [-0.6728834349912812, 4.745708454198967] |
| ndtw | 64.20545006336708 | 64.05767961357984 | -0.14777044978723383 | [-2.0706211204594522, 2.3887564677021755] |
| path_length | 13.832127295738331 | 14.026809833955998 | 0.1946825382176659 | [-0.6753805919159586, 1.142629842689021] |
| steps_taken | 99.84339314845025 | 99.47852093529092 | -0.3648722131593257 | [-7.444298830504145, 6.902383437626365] |

## Baseline rank-aligned comparison

| Evaluation | Rank count | SR |
| --- | --- | --- |
| Previous baseline control | 4 | 63.02338227297444 |
| Rank-aligned baseline control | 4 | 63.02338227297444 |

## Stratified SR

| Stratum | n | Baseline | Ours | Difference pp | 95% CI |
| --- | --- | --- | --- | --- | --- |
| 地标数:0–1 | 80 | 61.25000000000001 | 67.5 | 6.25 | [-5.68262987012987, 17.94871794871795] |
| 地标数:2–3 | 764 | 66.36125654450262 | 67.80104712041884 | 1.4397905759162304 | [-2.345982945497509, 5.374995795663956] |
| 地标数:≥4 | 995 | 60.60301507537689 | 62.71356783919598 | 2.1105527638190953 | [-0.8339967866409, 6.729604292421192] |
| 岔路数:0–1 | 30 | 70.0 | 70.0 | 0.0 | [0.0, 0.0] |
| 岔路数:2–3 | 192 | 64.58333333333334 | 66.66666666666666 | 2.083333333333333 | [-4.274193548387097, 7.937370600414074] |
| 岔路数:≥4 | 1617 | 62.708719851577 | 64.7495361781076 | 2.0408163265306123 | [-0.24349194153266376, 5.3873567131994085] |
| 路径长度:T1 | 612 | 78.26797385620915 | 79.90196078431373 | 1.6339869281045754 | [-2.064927228845169, 5.582770647945263] |
| 路径长度:T2 | 612 | 61.9281045751634 | 65.84967320261438 | 3.9215686274509802 | [0.14942722594811964, 8.484942593638246] |
| 路径长度:T3 | 615 | 48.94308943089431 | 49.43089430894309 | 0.4878048780487805 | [-2.4123383395911553, 5.661024787273397] |

## True termination transition

| Baseline → ours | A | B | C | D | E | Total |
| --- | --- | --- | --- | --- | --- | --- |
| A | 988 | 45 | 83 | 42 | 0 | 1158 |
| B | 55 | 28 | 14 | 4 | 0 | 101 |
| C | 107 | 17 | 224 | 33 | 0 | 381 |
| D | 46 | 8 | 29 | 110 | 0 | 193 |
| E | 0 | 0 | 0 | 0 | 0 | 0 |
| Total | 1196 | 98 | 350 | 189 | 0 | 1833 |

## baseline proxy → true

| Proxy → true | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| A | 1159 | 0 | 0 | 0 | 0 |
| B | 0 | 101 | 0 | 0 | 0 |
| C | 0 | 0 | 382 | 0 | 0 |
| D | 0 | 0 | 0 | 197 | 0 |
| E | 0 | 0 | 0 | 0 | 0 |

## ours proxy → true

| Proxy → true | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| A | 1196 | 0 | 0 | 0 | 0 |
| B | 0 | 98 | 0 | 0 | 0 |
| C | 0 | 0 | 350 | 0 | 0 |
| D | 0 | 0 | 0 | 189 | 0 |
| E | 0 | 0 | 0 | 0 | 0 |

## baseline historical eval proxy → new eval true (includes rerun differences)

| Old proxy → new true | A | B | C | D | E | U |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1159 | 0 | 0 | 0 | 0 | 0 |
| B | 0 | 101 | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 382 | 0 | 0 | 0 |
| D | 0 | 0 | 0 | 197 | 0 | 0 |
| E | 0 | 0 | 0 | 0 | 0 | 0 |
| U | 0 | 0 | 0 | 0 | 0 | 0 |

## ours historical eval proxy → new eval true (includes rerun differences)

| Old proxy → new true | A | B | C | D | E | U |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1196 | 1 | 0 | 0 | 0 | 0 |
| B | 0 | 97 | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 350 | 0 | 0 | 0 |
| D | 0 | 0 | 0 | 189 | 0 | 0 |
| E | 0 | 0 | 0 | 0 | 0 | 0 |
| U | 0 | 0 | 0 | 0 | 0 | 6 |

## Rescue

| Category | n | To A | Rate |
| --- | --- | --- | --- |
| A | 1158 | 988 | 0.853195164075993 |
| B | 101 | 55 | 0.5445544554455446 |
| C | 381 | 107 | 0.28083989501312334 |
| D | 193 | 46 | 0.23834196891191708 |
| E | 0 | 0 | None |

## Loss

| Ours category | Baseline A → ours |
| --- | --- |
| B | 45 |
| C | 83 |
| D | 42 |
| E | 0 |

## Analysis E threshold 3m

| Subset | Model | n | Deviated n | Step P25/P50/P75 | Branch proportion | Never proportion |
| --- | --- | --- | --- | --- | --- | --- |
| at_least_one_failed | baseline | 845 | 430 | [1.0, 3.0, 6.0] | 0.7790697674418605 | 0.49112426035502954 |
| at_least_one_failed | ours | 845 | 426 | [1.0, 3.0, 5.0] | 0.7863849765258216 | 0.4958579881656805 |
| C_to_A | baseline | 107 | 47 | [2.0, 3.0, 5.0] | 0.7659574468085106 | 0.5607476635514019 |
| C_to_A | ours | 107 | 10 | [2.0, 3.0, 5.75] | 0.9 | 0.9065420560747663 |
| A_to_C | baseline | 83 | 14 | [1.25, 3.0, 3.75] | 0.7142857142857143 | 0.8313253012048193 |
| A_to_C | ours | 83 | 49 | [1.0, 2.0, 4.0] | 0.7551020408163265 | 0.40963855421686746 |

## Analysis E threshold 2m

| Subset | Model | n | Deviated n | Step P25/P50/P75 | Branch proportion | Never proportion |
| --- | --- | --- | --- | --- | --- | --- |
| at_least_one_failed | baseline | 845 | 577 | [1.0, 2.0, 4.0] | 0.8058925476603119 | 0.3171597633136095 |
| at_least_one_failed | ours | 845 | 573 | [1.0, 2.0, 4.0] | 0.7905759162303665 | 0.32189349112426036 |
| C_to_A | baseline | 107 | 62 | [1.0, 2.0, 4.0] | 0.7580645161290323 | 0.42056074766355145 |
| C_to_A | ours | 107 | 36 | [1.0, 4.0, 5.0] | 0.7222222222222222 | 0.6635514018691588 |
| A_to_C | baseline | 83 | 38 | [1.0, 1.5, 4.0] | 0.8157894736842105 | 0.5421686746987953 |
| A_to_C | ours | 83 | 67 | [1.0, 1.0, 3.0] | 0.7164179104477612 | 0.19277108433734935 |

## rank_alignment_baseline

```json
{
  "old_SR": 63.02338227297444,
  "new_SR": 63.02338227297444,
  "difference_pp": 0.0,
  "over_0_3pp": false
}
```

## unrun_ids

```json
[
  "308",
  "349",
  "455",
  "804",
  "1318",
  "1331"
]
```

## simultaneous_budget_stop

```json
{
  "baseline": 28,
  "ours": 23
}
```

## proxy_confusion

```json
{
  "baseline": [
    [
      1159,
      0,
      0,
      0,
      0
    ],
    [
      0,
      101,
      0,
      0,
      0
    ],
    [
      0,
      0,
      382,
      0,
      0
    ],
    [
      0,
      0,
      0,
      197,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0
    ]
  ],
  "ours": [
    [
      1196,
      0,
      0,
      0,
      0
    ],
    [
      0,
      98,
      0,
      0,
      0
    ],
    [
      0,
      0,
      350,
      0,
      0
    ],
    [
      0,
      0,
      0,
      189,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

## rescue

```json
[
  {
    "category": "A",
    "n": 1158,
    "to_A": 988,
    "rate": 0.853195164075993
  },
  {
    "category": "B",
    "n": 101,
    "to_A": 55,
    "rate": 0.5445544554455446
  },
  {
    "category": "C",
    "n": 381,
    "to_A": 107,
    "rate": 0.28083989501312334
  },
  {
    "category": "D",
    "n": 193,
    "to_A": 46,
    "rate": 0.23834196891191708
  },
  {
    "category": "E",
    "n": 0,
    "to_A": 0,
    "rate": null
  }
]
```

## C_distributions

```json
{
  "baseline_C_other_A_True": {
    "n": 107,
    "distance_to_goal": [
      3.636305570602417,
      4.958888530731201,
      9.262030601501465,
      14.602731704711925
    ],
    "steps_taken": [
      42.5,
      55.0,
      79.0,
      112.80000000000004
    ]
  },
  "baseline_C_other_A_False": {
    "n": 274,
    "distance_to_goal": [
      4.683213949203491,
      6.6608898639678955,
      11.546835899353027,
      19.634907531738286
    ],
    "steps_taken": [
      43.25,
      58.0,
      89.25,
      141.80000000000007
    ]
  },
  "ours_C_other_A_True": {
    "n": 83,
    "distance_to_goal": [
      4.159963846206665,
      6.364164352416992,
      11.017231941223145,
      19.253884887695307
    ],
    "steps_taken": [
      42.0,
      55.0,
      70.5,
      109.79999999999998
    ]
  },
  "ours_C_other_A_False": {
    "n": 267,
    "distance_to_goal": [
      4.629991769790649,
      8.04877758026123,
      11.65199327468872,
      18.52750473022461
    ],
    "steps_taken": [
      45.5,
      58.0,
      87.5,
      151.0
    ]
  }
}
```

## mcnemar

```json
{
  "B": {
    "to_A": 55,
    "A_to": 45,
    "chi2": 1.0
  },
  "C": {
    "to_A": 107,
    "A_to": 83,
    "chi2": 3.031578947368421
  },
  "D": {
    "to_A": 46,
    "A_to": 42,
    "chi2": 0.18181818181818182
  },
  "E": {
    "to_A": 0,
    "A_to": 0,
    "chi2": null
  }
}
```

## analysis_E

```json
{
  "3": {
    "at_least_one_failed": {
      "baseline": {
        "n": 845,
        "deviated_n": 430,
        "first_step_P25_P50_P75": [
          1.0,
          3.0,
          6.0
        ],
        "branch_given_deviation": 0.7790697674418605,
        "never_deviated": 0.49112426035502954
      },
      "ours": {
        "n": 845,
        "deviated_n": 426,
        "first_step_P25_P50_P75": [
          1.0,
          3.0,
          5.0
        ],
        "branch_given_deviation": 0.7863849765258216,
        "never_deviated": 0.4958579881656805
      },
      "paired_difference": {
        "branch": {
          "difference": 0.007315209083961105,
          "ci": [
            -0.034860212019145985,
            0.04877991874851757
          ],
          "valid_bootstrap_draws": 2000
        },
        "step": {
          "n": 313,
          "difference": -0.19808306709265175,
          "ci": [
            -0.4784778610725808,
            0.06686148869963969
          ]
        },
        "never": {
          "n": 845,
          "difference": 0.004733727810650888,
          "ci": [
            -0.04473191060147582,
            0.06911845264669328
          ]
        }
      },
      "paired_quantile_difference": {
        "difference_P25_P50_P75": [
          0.0,
          0.0,
          -1.0
        ],
        "ci_by_quantile": [
          [
            -1.0,
            0.0
          ],
          [
            -1.0,
            1.0
          ],
          [
            -1.0,
            0.0
          ]
        ],
        "valid_bootstrap_draws": 2000
      }
    },
    "C_to_A": {
      "baseline": {
        "n": 107,
        "deviated_n": 47,
        "first_step_P25_P50_P75": [
          2.0,
          3.0,
          5.0
        ],
        "branch_given_deviation": 0.7659574468085106,
        "never_deviated": 0.5607476635514019
      },
      "ours": {
        "n": 107,
        "deviated_n": 10,
        "first_step_P25_P50_P75": [
          2.0,
          3.0,
          5.75
        ],
        "branch_given_deviation": 0.9,
        "never_deviated": 0.9065420560747663
      },
      "paired_difference": {
        "branch": {
          "difference": 0.1340425531914894,
          "ci": [
            0.018157467532467542,
            0.29513391826368046
          ],
          "valid_bootstrap_draws": 1998
        },
        "step": {
          "n": 9,
          "difference": 0.8888888888888888,
          "ci": [
            0.0,
            5.25
          ]
        },
        "never": {
          "n": 107,
          "difference": 0.34579439252336447,
          "ci": [
            0.1855496837910422,
            0.5638429534442805
          ]
        }
      },
      "paired_quantile_difference": {
        "difference_P25_P50_P75": [
          0.0,
          0.0,
          0.75
        ],
        "ci_by_quantile": [
          [
            0.0,
            4.768750000000011
          ],
          [
            -2.5,
            5.0
          ],
          [
            -3.5,
            3.0
          ]
        ],
        "valid_bootstrap_draws": 1998
      }
    },
    "A_to_C": {
      "baseline": {
        "n": 83,
        "deviated_n": 14,
        "first_step_P25_P50_P75": [
          1.25,
          3.0,
          3.75
        ],
        "branch_given_deviation": 0.7142857142857143,
        "never_deviated": 0.8313253012048193
      },
      "ours": {
        "n": 83,
        "deviated_n": 49,
        "first_step_P25_P50_P75": [
          1.0,
          2.0,
          4.0
        ],
        "branch_given_deviation": 0.7551020408163265,
        "never_deviated": 0.40963855421686746
      },
      "paired_difference": {
        "branch": {
          "difference": 0.04081632653061218,
          "ci": [
            -0.09233504641909815,
            0.2297872340425532
          ],
          "valid_bootstrap_draws": 1998
        },
        "step": {
          "n": 12,
          "difference": -0.25,
          "ci": [
            -0.7777777777777778,
            0.08402777777777398
          ]
        },
        "never": {
          "n": 83,
          "difference": -0.42168674698795183,
          "ci": [
            -0.5813953488372093,
            -0.21614414414414454
          ]
        }
      },
      "paired_quantile_difference": {
        "difference_P25_P50_P75": [
          -0.25,
          -1.0,
          0.25
        ],
        "ci_by_quantile": [
          [
            -2.0,
            1.0
          ],
          [
            -2.0,
            0.5
          ],
          [
            -1.768749999999999,
            2.0
          ]
        ],
        "valid_bootstrap_draws": 1998
      }
    }
  },
  "2": {
    "at_least_one_failed": {
      "baseline": {
        "n": 845,
        "deviated_n": 577,
        "first_step_P25_P50_P75": [
          1.0,
          2.0,
          4.0
        ],
        "branch_given_deviation": 0.8058925476603119,
        "never_deviated": 0.3171597633136095
      },
      "ours": {
        "n": 845,
        "deviated_n": 573,
        "first_step_P25_P50_P75": [
          1.0,
          2.0,
          4.0
        ],
        "branch_given_deviation": 0.7905759162303665,
        "never_deviated": 0.32189349112426036
      },
      "paired_difference": {
        "branch": {
          "difference": -0.015316631429945438,
          "ci": [
            -0.04629894200652828,
            0.018039754050647692
          ],
          "valid_bootstrap_draws": 2000
        },
        "step": {
          "n": 464,
          "difference": -0.036637931034482756,
          "ci": [
            -0.19867594787811926,
            0.13262822464871432
          ]
        },
        "never": {
          "n": 845,
          "difference": 0.004733727810650888,
          "ci": [
            -0.0531573513520028,
            0.06602298541985026
          ]
        }
      },
      "paired_quantile_difference": {
        "difference_P25_P50_P75": [
          0.0,
          0.0,
          0.0
        ],
        "ci_by_quantile": [
          [
            0.0,
            0.0
          ],
          [
            -1.0,
            0.0
          ],
          [
            -1.0,
            0.0
          ]
        ],
        "valid_bootstrap_draws": 2000
      }
    },
    "C_to_A": {
      "baseline": {
        "n": 107,
        "deviated_n": 62,
        "first_step_P25_P50_P75": [
          1.0,
          2.0,
          4.0
        ],
        "branch_given_deviation": 0.7580645161290323,
        "never_deviated": 0.42056074766355145
      },
      "ours": {
        "n": 107,
        "deviated_n": 36,
        "first_step_P25_P50_P75": [
          1.0,
          4.0,
          5.0
        ],
        "branch_given_deviation": 0.7222222222222222,
        "never_deviated": 0.6635514018691588
      },
      "paired_difference": {
        "branch": {
          "difference": -0.03584229390681004,
          "ci": [
            -0.14551726564243914,
            0.08265582655826564
          ],
          "valid_bootstrap_draws": 2000
        },
        "step": {
          "n": 27,
          "difference": 0.4074074074074074,
          "ci": [
            0.125,
            0.8524305555555524
          ]
        },
        "never": {
          "n": 107,
          "difference": 0.24299065420560748,
          "ci": [
            0.0494928169287517,
            0.46808510638297873
          ]
        }
      },
      "paired_quantile_difference": {
        "difference_P25_P50_P75": [
          0.0,
          2.0,
          1.0
        ],
        "ci_by_quantile": [
          [
            0.0,
            3.0
          ],
          [
            0.0,
            3.0
          ],
          [
            0.0,
            2.0
          ]
        ],
        "valid_bootstrap_draws": 2000
      }
    },
    "A_to_C": {
      "baseline": {
        "n": 83,
        "deviated_n": 38,
        "first_step_P25_P50_P75": [
          1.0,
          1.5,
          4.0
        ],
        "branch_given_deviation": 0.8157894736842105,
        "never_deviated": 0.5421686746987953
      },
      "ours": {
        "n": 83,
        "deviated_n": 67,
        "first_step_P25_P50_P75": [
          1.0,
          1.0,
          3.0
        ],
        "branch_given_deviation": 0.7164179104477612,
        "never_deviated": 0.19277108433734935
      },
      "paired_difference": {
        "branch": {
          "difference": -0.09937156323644936,
          "ci": [
            -0.1860928497667628,
            0.012504960317460246
          ],
          "valid_bootstrap_draws": 2000
        },
        "step": {
          "n": 35,
          "difference": -0.4857142857142857,
          "ci": [
            -1.1538461538461537,
            -0.02775900900900911
          ]
        },
        "never": {
          "n": 83,
          "difference": -0.3493975903614458,
          "ci": [
            -0.4800253164556962,
            -0.19444444444444445
          ]
        }
      },
      "paired_quantile_difference": {
        "difference_P25_P50_P75": [
          0.0,
          -0.5,
          -1.0
        ],
        "ci_by_quantile": [
          [
            -0.75,
            1.0
          ],
          [
            -2.0,
            1.0
          ],
          [
            -2.0,
            1.0
          ]
        ],
        "valid_bootstrap_draws": 2000
      }
    }
  }
}
```
