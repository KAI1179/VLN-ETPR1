# D7-d kd

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
| success | 63.02338227297444 | 67.2648178357803 | 4.241435562805873 | [2.3776112952024206, 6.679524752781092] |
| oracle_success | 68.51549755301795 | 72.10440456769985 | 3.588907014681892 | [2.1000857681869567, 5.2400203962703955] |
| spl | 53.312785633214276 | 56.47577899000933 | 3.1629933567950577 | [0.9849803512382014, 6.063448368971367] |
| ndtw | 64.20545006336708 | 65.40593302717126 | 1.2004829638041838 | [-0.6813608324162969, 3.620344843696619] |
| path_length | 13.832127295738331 | 13.539913661457911 | -0.2922136342804221 | [-1.1189135722594323, 0.42332450359519774] |
| steps_taken | 99.84339314845025 | 94.71288743882545 | -5.130505709624796 | [-11.333183511088539, 0.23668107603091498] |

## Baseline rank-aligned comparison

| Evaluation | Rank count | SR |
| --- | --- | --- |
| Previous baseline control | 4 | 63.02338227297444 |
| Rank-aligned baseline control | 4 | 63.02338227297444 |

## Stratified SR

| Stratum | n | Baseline | Ours | Difference pp | 95% CI |
| --- | --- | --- | --- | --- | --- |
| 地标数:0–1 | 80 | 61.25000000000001 | 68.75 | 7.5 | [-5.714285714285714, 18.693049065420553] |
| 地标数:2–3 | 764 | 66.36125654450262 | 69.24083769633508 | 2.8795811518324608 | [0.7446098238525161, 5.288481404958677] |
| 地标数:≥4 | 995 | 60.60301507537689 | 65.62814070351759 | 5.025125628140704 | [2.881720470902462, 8.078099362765673] |
| 岔路数:0–1 | 30 | 70.0 | 76.66666666666667 | 6.666666666666667 | [0.0, 9.523809523809524] |
| 岔路数:2–3 | 192 | 64.58333333333334 | 69.27083333333334 | 4.6875 | [-2.3255813953488373, 9.401881720430106] |
| 岔路数:≥4 | 1617 | 62.708719851577 | 66.852195423624 | 4.1434755720470005 | [2.2849168469231014, 6.552757793764988] |
| 路径长度:T1 | 612 | 78.26797385620915 | 81.69934640522875 | 3.431372549019608 | [0.34183673469387754, 7.312247388088155] |
| 路径长度:T2 | 612 | 61.9281045751634 | 68.4640522875817 | 6.535947712418301 | [3.98507591442374, 10.38146846802262] |
| 路径长度:T3 | 615 | 48.94308943089431 | 51.707317073170735 | 2.7642276422764227 | [-0.6563750752086221, 8.535220002611304] |

## True termination transition

| Baseline → ours | A | B | C | D | E | Total |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1007 | 41 | 79 | 31 | 0 | 1158 |
| B | 58 | 33 | 9 | 1 | 0 | 101 |
| C | 119 | 9 | 220 | 33 | 0 | 381 |
| D | 53 | 6 | 27 | 107 | 0 | 193 |
| E | 0 | 0 | 0 | 0 | 0 | 0 |
| Total | 1237 | 89 | 335 | 172 | 0 | 1833 |

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
| A | 1237 | 0 | 0 | 0 | 0 |
| B | 0 | 89 | 0 | 0 | 0 |
| C | 0 | 0 | 335 | 0 | 0 |
| D | 0 | 0 | 0 | 172 | 0 |
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
| A | 1236 | 0 | 0 | 1 | 0 | 0 |
| B | 1 | 89 | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 335 | 0 | 0 | 0 |
| D | 0 | 0 | 0 | 171 | 0 | 0 |
| E | 0 | 0 | 0 | 0 | 0 | 0 |
| U | 0 | 0 | 0 | 0 | 0 | 6 |

## Rescue

| Category | n | To A | Rate |
| --- | --- | --- | --- |
| A | 1158 | 1007 | 0.8696027633851469 |
| B | 101 | 58 | 0.5742574257425742 |
| C | 381 | 119 | 0.3123359580052493 |
| D | 193 | 53 | 0.27461139896373055 |
| E | 0 | 0 | None |

## Loss

| Ours category | Baseline A → ours |
| --- | --- |
| B | 41 |
| C | 79 |
| D | 31 |
| E | 0 |

## Analysis E threshold 3m

| Subset | Model | n | Deviated n | Step P25/P50/P75 | Branch proportion | Never proportion |
| --- | --- | --- | --- | --- | --- | --- |
| at_least_one_failed | baseline | 826 | 427 | [1.0, 3.0, 6.0] | 0.775175644028103 | 0.48305084745762716 |
| at_least_one_failed | ours | 826 | 395 | [1.0, 3.0, 5.0] | 0.7848101265822784 | 0.5217917675544794 |
| C_to_A | baseline | 119 | 60 | [1.0, 2.0, 5.0] | 0.7666666666666667 | 0.4957983193277311 |
| C_to_A | ours | 119 | 15 | [1.5, 3.0, 5.5] | 0.7333333333333333 | 0.8739495798319328 |
| A_to_C | baseline | 79 | 12 | [1.0, 3.0, 4.25] | 0.8333333333333334 | 0.8481012658227848 |
| A_to_C | ours | 79 | 48 | [1.0, 2.0, 3.25] | 0.7708333333333334 | 0.3924050632911392 |

## Analysis E threshold 2m

| Subset | Model | n | Deviated n | Step P25/P50/P75 | Branch proportion | Never proportion |
| --- | --- | --- | --- | --- | --- | --- |
| at_least_one_failed | baseline | 826 | 568 | [1.0, 2.0, 4.0] | 0.8098591549295775 | 0.3123486682808717 |
| at_least_one_failed | ours | 826 | 540 | [1.0, 2.0, 4.0] | 0.8111111111111111 | 0.3462469733656174 |
| C_to_A | baseline | 119 | 75 | [1.0, 1.0, 3.0] | 0.7866666666666666 | 0.3697478991596639 |
| C_to_A | ours | 119 | 45 | [1.0, 2.0, 5.0] | 0.8 | 0.6218487394957983 |
| A_to_C | baseline | 79 | 35 | [1.0, 1.0, 4.0] | 0.9142857142857143 | 0.5569620253164558 |
| A_to_C | ours | 79 | 65 | [1.0, 1.0, 3.0] | 0.7692307692307693 | 0.17721518987341767 |

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
  "ours": 21
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
      1237,
      0,
      0,
      0,
      0
    ],
    [
      0,
      89,
      0,
      0,
      0
    ],
    [
      0,
      0,
      335,
      0,
      0
    ],
    [
      0,
      0,
      0,
      172,
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
    "to_A": 1007,
    "rate": 0.8696027633851469
  },
  {
    "category": "B",
    "n": 101,
    "to_A": 58,
    "rate": 0.5742574257425742
  },
  {
    "category": "C",
    "n": 381,
    "to_A": 119,
    "rate": 0.3123359580052493
  },
  {
    "category": "D",
    "n": 193,
    "to_A": 53,
    "rate": 0.27461139896373055
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
    "n": 119,
    "distance_to_goal": [
      3.6852335929870605,
      5.338990211486816,
      10.141214370727539,
      14.975960350036624
    ],
    "steps_taken": [
      42.5,
      59.0,
      82.0,
      115.2
    ]
  },
  "baseline_C_other_A_False": {
    "n": 262,
    "distance_to_goal": [
      4.495985269546509,
      6.471625328063965,
      11.645814657211304,
      19.27707424163819
    ],
    "steps_taken": [
      43.0,
      56.5,
      86.5,
      143.0
    ]
  },
  "ours_C_other_A_True": {
    "n": 79,
    "distance_to_goal": [
      3.983741044998169,
      6.583252906799316,
      11.67432975769043,
      17.719107055664065
    ],
    "steps_taken": [
      50.0,
      60.0,
      90.0,
      145.0
    ]
  },
  "ours_C_other_A_False": {
    "n": 256,
    "distance_to_goal": [
      4.462697982788086,
      7.164183616638184,
      11.789624452590942,
      18.09248638153076
    ],
    "steps_taken": [
      45.75,
      60.5,
      87.25,
      129.5
    ]
  }
}
```

## mcnemar

```json
{
  "B": {
    "to_A": 58,
    "A_to": 41,
    "chi2": 2.919191919191919
  },
  "C": {
    "to_A": 119,
    "A_to": 79,
    "chi2": 8.080808080808081
  },
  "D": {
    "to_A": 53,
    "A_to": 31,
    "chi2": 5.761904761904762
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
        "n": 826,
        "deviated_n": 427,
        "first_step_P25_P50_P75": [
          1.0,
          3.0,
          6.0
        ],
        "branch_given_deviation": 0.775175644028103,
        "never_deviated": 0.48305084745762716
      },
      "ours": {
        "n": 826,
        "deviated_n": 395,
        "first_step_P25_P50_P75": [
          1.0,
          3.0,
          5.0
        ],
        "branch_given_deviation": 0.7848101265822784,
        "never_deviated": 0.5217917675544794
      },
      "paired_difference": {
        "branch": {
          "difference": 0.009634482554175428,
          "ci": [
            -0.026255221143832153,
            0.03678245970461291
          ],
          "valid_bootstrap_draws": 2000
        },
        "step": {
          "n": 302,
          "difference": -0.04966887417218543,
          "ci": [
            -0.3087551074205878,
            0.16498902064119453
          ]
        },
        "never": {
          "n": 826,
          "difference": 0.0387409200968523,
          "ci": [
            0.002881880024737168,
            0.08081801629963585
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
            -1.25,
            0.0
          ]
        ],
        "valid_bootstrap_draws": 2000
      }
    },
    "C_to_A": {
      "baseline": {
        "n": 119,
        "deviated_n": 60,
        "first_step_P25_P50_P75": [
          1.0,
          2.0,
          5.0
        ],
        "branch_given_deviation": 0.7666666666666667,
        "never_deviated": 0.4957983193277311
      },
      "ours": {
        "n": 119,
        "deviated_n": 15,
        "first_step_P25_P50_P75": [
          1.5,
          3.0,
          5.5
        ],
        "branch_given_deviation": 0.7333333333333333,
        "never_deviated": 0.8739495798319328
      },
      "paired_difference": {
        "branch": {
          "difference": -0.03333333333333344,
          "ci": [
            -0.3059430184169612,
            0.12472363945578234
          ],
          "valid_bootstrap_draws": 2000
        },
        "step": {
          "n": 13,
          "difference": 1.3846153846153846,
          "ci": [
            0.2222222222222222,
            4.111458333333331
          ]
        },
        "never": {
          "n": 119,
          "difference": 0.37815126050420167,
          "ci": [
            0.24324324324324326,
            0.5370408631772269
          ]
        }
      },
      "paired_quantile_difference": {
        "difference_P25_P50_P75": [
          0.5,
          1.0,
          0.5
        ],
        "ci_by_quantile": [
          [
            -1.0,
            1.0
          ],
          [
            -1.5,
            3.0
          ],
          [
            -2.0,
            6.0
          ]
        ],
        "valid_bootstrap_draws": 2000
      }
    },
    "A_to_C": {
      "baseline": {
        "n": 79,
        "deviated_n": 12,
        "first_step_P25_P50_P75": [
          1.0,
          3.0,
          4.25
        ],
        "branch_given_deviation": 0.8333333333333334,
        "never_deviated": 0.8481012658227848
      },
      "ours": {
        "n": 79,
        "deviated_n": 48,
        "first_step_P25_P50_P75": [
          1.0,
          2.0,
          3.25
        ],
        "branch_given_deviation": 0.7708333333333334,
        "never_deviated": 0.3924050632911392
      },
      "paired_difference": {
        "branch": {
          "difference": -0.0625,
          "ci": [
            -0.21451863354037257,
            0.10528508771929826
          ],
          "valid_bootstrap_draws": 1998
        },
        "step": {
          "n": 11,
          "difference": -0.45454545454545453,
          "ci": [
            -0.8888888888888888,
            -0.1
          ]
        },
        "never": {
          "n": 79,
          "difference": -0.45569620253164556,
          "ci": [
            -0.6712623996221068,
            -0.2714225181598063
          ]
        }
      },
      "paired_quantile_difference": {
        "difference_P25_P50_P75": [
          0.0,
          -1.0,
          -1.0
        ],
        "ci_by_quantile": [
          [
            -2.0,
            1.0
          ],
          [
            -3.0,
            1.0
          ],
          [
            -2.5,
            1.0
          ]
        ],
        "valid_bootstrap_draws": 1998
      }
    }
  },
  "2": {
    "at_least_one_failed": {
      "baseline": {
        "n": 826,
        "deviated_n": 568,
        "first_step_P25_P50_P75": [
          1.0,
          2.0,
          4.0
        ],
        "branch_given_deviation": 0.8098591549295775,
        "never_deviated": 0.3123486682808717
      },
      "ours": {
        "n": 826,
        "deviated_n": 540,
        "first_step_P25_P50_P75": [
          1.0,
          2.0,
          4.0
        ],
        "branch_given_deviation": 0.8111111111111111,
        "never_deviated": 0.3462469733656174
      },
      "paired_difference": {
        "branch": {
          "difference": 0.00125195618153362,
          "ci": [
            -0.029664347401813597,
            0.03178688408517365
          ],
          "valid_bootstrap_draws": 2000
        },
        "step": {
          "n": 445,
          "difference": 0.06292134831460675,
          "ci": [
            -0.13731615837148625,
            0.2794429800221974
          ]
        },
        "never": {
          "n": 826,
          "difference": 0.03389830508474576,
          "ci": [
            -0.008973752416999218,
            0.07489704273684855
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
            -0.5,
            1.0
          ],
          [
            -1.0,
            1.0
          ]
        ],
        "valid_bootstrap_draws": 2000
      }
    },
    "C_to_A": {
      "baseline": {
        "n": 119,
        "deviated_n": 75,
        "first_step_P25_P50_P75": [
          1.0,
          1.0,
          3.0
        ],
        "branch_given_deviation": 0.7866666666666666,
        "never_deviated": 0.3697478991596639
      },
      "ours": {
        "n": 119,
        "deviated_n": 45,
        "first_step_P25_P50_P75": [
          1.0,
          2.0,
          5.0
        ],
        "branch_given_deviation": 0.8,
        "never_deviated": 0.6218487394957983
      },
      "paired_difference": {
        "branch": {
          "difference": 0.01333333333333342,
          "ci": [
            -0.1519638480392157,
            0.13469609582963615
          ],
          "valid_bootstrap_draws": 2000
        },
        "step": {
          "n": 37,
          "difference": 0.5945945945945946,
          "ci": [
            0.09090909090909091,
            1.5429220779220776
          ]
        },
        "never": {
          "n": 119,
          "difference": 0.25210084033613445,
          "ci": [
            0.12496778350515464,
            0.3892238423915528
          ]
        }
      },
      "paired_quantile_difference": {
        "difference_P25_P50_P75": [
          0.0,
          1.0,
          2.0
        ],
        "ci_by_quantile": [
          [
            0.0,
            1.0
          ],
          [
            -0.5,
            2.012499999999932
          ],
          [
            0.0,
            3.75
          ]
        ],
        "valid_bootstrap_draws": 2000
      }
    },
    "A_to_C": {
      "baseline": {
        "n": 79,
        "deviated_n": 35,
        "first_step_P25_P50_P75": [
          1.0,
          1.0,
          4.0
        ],
        "branch_given_deviation": 0.9142857142857143,
        "never_deviated": 0.5569620253164558
      },
      "ours": {
        "n": 79,
        "deviated_n": 65,
        "first_step_P25_P50_P75": [
          1.0,
          1.0,
          3.0
        ],
        "branch_given_deviation": 0.7692307692307693,
        "never_deviated": 0.17721518987341767
      },
      "paired_difference": {
        "branch": {
          "difference": -0.14505494505494498,
          "ci": [
            -0.24381011124931531,
            -0.028100023191444713
          ],
          "valid_bootstrap_draws": 2000
        },
        "step": {
          "n": 33,
          "difference": -0.9696969696969697,
          "ci": [
            -1.8464285714285715,
            -0.3599285714285718
          ]
        },
        "never": {
          "n": 79,
          "difference": -0.379746835443038,
          "ci": [
            -0.5168539325842697,
            -0.22535211267605634
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
            -2.5124999999999993,
            0.0
          ],
          [
            -3.0,
            1.0
          ]
        ],
        "valid_bootstrap_draws": 2000
      }
    }
  }
}
```
