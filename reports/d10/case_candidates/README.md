# D10 case candidates

CPU-only screening; no inference or rendering. Awaiting episode selection.

## Definitions

- **on_gt:** 3D Euclidean point-to-reference-polyline distance <=1m; off >1m
- **bearing:** signed degrees relative to forward; positive=left; wrapped separation >45deg; numerical boundary within0.001deg excluded conservatively (episode723 is ~45.000018deg)
- **lifts:** D10 conditional-spatial lift, inclusive IQR, all finite S frames (candidate excludes STOP)
- **corridor_hit:** existing dense_region_hit: fraction of dense GT path samples hitting any positive region channel 27:37; outside raster counts as miss; not hallway-channel-only
- **corridor_population:** all 1839 val_unseen episodes, including six missing-cache zero placeholders
- **preferred:** Z own decision pose within 1m in 3D, executed off-route target within 1m of Z S chosen target
- **probability:** not recorded by D10; recovered only from matching decision-log state and argmax (pose/ghost candidates atol1e-5m, heading1e-6rad); otherwise null; these are matched-state PA probabilities, not directly recorded D10 probabilities
- **rank:** preferred first, then minimum summed distance of lift percentiles from median; one frame per episode

## Population cutoffs

```json
{
  "lifts": {
    "route": [
      2.2371560113424005,
      4.390462211580449
    ],
    "candidate": [
      1.7581991718268926,
      4.220846470641756
    ],
    "agent": [
      1.92785913374258,
      4.686365586505774
    ]
  },
  "corridor": [
    0.25,
    0.7428571428571429
  ],
  "counts": {
    "S_frames": 13992,
    "own_Z_fail_M_success_frames": 1501,
    "nonstop_off_on_and_angle_frames": 109,
    "three_lifts_IQR_frames": 23,
    "all_required_frames": 10
  }
}
```

## Episode 1645, S step 0

Turn right and wait in the entryway of the bedroom.

Preferred: **True**; bearing separation 120.00°.

| Model | Candidate | Probability | Bearing (left +) | Distance to reference | Own SR | NE | PL |
|---|---|---|---|---|---|---|---|
| Z | g2 | 0.6464 | -90.00° | 1.500m | 0 | 13.60m | 20.00m |
| M | g4 | 0.7653 | 30.00° | 0.037m | 1 | 0.39m | 6.00m |

| M lift | Value | Percentile |
|---|---|---|
| route | 3.0495 | 49.12 |
| candidate | 3.3820 | 59.34 |
| agent | 3.3280 | 51.60 |

Corridor hit: 0.3200, percentile 36.49.

## Episode 1590, S step 0

Position yourself so that the red carpet leads to a bed with curtains.  Walk straight towards this bed and stop in the doorway of the room and then stop.

Preferred: **True**; bearing separation 162.00°.

| Model | Candidate | Probability | Bearing (left +) | Distance to reference | Own SR | NE | PL |
|---|---|---|---|---|---|---|---|
| Z | g0 | 0.8544 | 0.00° | 1.750m | 0 | 18.04m | 8.50m |
| M | g3 | 0.6928 | 162.00° | 0.613m | 1 | 1.57m | 8.32m |

| M lift | Value | Percentile |
|---|---|---|
| route | 2.8927 | 44.88 |
| candidate | 3.1592 | 54.26 |
| agent | 4.2805 | 69.64 |

Corridor hit: 0.5526, percentile 60.98.

## Episode 1392, S step 0

Following the red carpet, exit the room to the right. Continue straight passed the chest of drawers on the right and a series of 7 photo displays on the left. Still straight on the red carpet passed 2 sets of chairs and through the doorway stopping just through the threshold.

Preferred: **True**; bearing separation 171.00°.

| Model | Candidate | Probability | Bearing (left +) | Distance to reference | Own SR | NE | PL |
|---|---|---|---|---|---|---|---|
| Z | g4 | 0.7014 | 78.00° | 2.750m | 0 | 25.93m | 12.07m |
| M | g1 | 0.9951 | -93.00° | 0.095m | 1 | 0.56m | 16.58m |

| M lift | Value | Percentile |
|---|---|---|
| route | 2.3415 | 28.31 |
| candidate | 2.9782 | 49.82 |
| agent | 2.8085 | 40.53 |

Corridor hit: 0.3906, percentile 45.92.

## Episode 302, S step 3

Exit the entryway, turn left, walk to the study, turn right, enter the bathroom, wait at the sink.

Preferred: **True**; bearing separation 48.00°.

| Model | Candidate | Probability | Bearing (left +) | Distance to reference | Own SR | NE | PL |
|---|---|---|---|---|---|---|---|
| Z | g7 | 0.5238 | -45.00° | 1.172m | 0 | 3.56m | 9.51m |
| M | g9 | 0.5853 | 3.00° | 0.167m | 1 | 0.76m | 9.75m |

| M lift | Value | Percentile |
|---|---|---|
| route | 2.3681 | 29.18 |
| candidate | 2.7869 | 45.11 |
| agent | 2.3454 | 31.07 |

Corridor hit: 0.4054, percentile 46.98.

## Episode 83, S step 0

Walk past kitchen. Walk past thermostat. Walk into bedroom. Wait on rug.

Preferred: **True**; bearing separation 162.00°.

| Model | Candidate | Probability | Bearing (left +) | Distance to reference | Own SR | NE | PL |
|---|---|---|---|---|---|---|---|
| Z | g0 | 0.6846 | -90.00° | 1.750m | 0 | 21.85m | 11.76m |
| M | g2 | 0.8336 | 108.00° | 0.272m | 1 | 2.26m | 14.52m |

| M lift | Value | Percentile |
|---|---|---|
| route | 2.5708 | 35.30 |
| candidate | 4.1239 | 73.56 |
| agent | 3.8797 | 62.64 |

Corridor hit: 0.3846, percentile 45.13.

## Suggested choices

1645: short instruction, first-step split, all three lifts near the population median, 120-degree separation.
302: compact multi-stage instruction, later split at step3, with the preferred own-rollout branch condition satisfied.

## Next step

User selects one or two episodes. Then export raw map/attention/geometry/own trajectories and render via d7_case2; no assets or GPU jobs generated at this screening stage.
