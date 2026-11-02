# Paired differences — O

Own rollouts: secondary; visited-state difficulty differs across models

2000 scene-cluster bootstrap draws, seed 0, fixed out-of-fold predictions; no refitting within bootstrap.

|Comparison|Metric|Difference (pp)|95% scene-bootstrap CI (pp)|Status|
|---|---|---|---|---|
|base-step/O|balanced_accuracy|33.32|[30.59, 35.49]|final|
|base-step/O|accuracy|26.42|[25.16, 27.93]|final|
|try5-step/O|balanced_accuracy|32.50|[30.50, 34.64]|final|
|try5-step/O|accuracy|26.31|[24.87, 28.03]|final|
|kd-step/O|balanced_accuracy|31.00|[28.73, 33.26]|final|
|kd-step/O|accuracy|25.41|[23.74, 27.17]|final|
