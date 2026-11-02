# STOP probability — O

Own rollouts: secondary; visited-state difficulty differs across models

Only frames with geodesic distance to goal >3 m; O additionally requires distance to reference polyline <=1 m.
95% episode-bootstrap CIs, 2000 draws, seed 0. Bin 4 is tabulated but never plotted.

|Model|Bin|Frames|Mean STOP (%)|95% CI (%)|
|---|---|---|---|---|
|base|0|4106|0.12|[0.00, 0.29]|
|base|1|2237|0.57|[0.32, 0.86]|
|base|2|1816|2.80|[2.24, 3.42]|
|base|3|602|12.00|[10.05, 14.18]|
|base|4|2|0.07|[0.03, 0.10]|
|try5|0|4100|0.05|[0.01, 0.11]|
|try5|1|2202|0.41|[0.19, 0.67]|
|try5|2|1798|2.16|[1.68, 2.66]|
|try5|3|576|9.30|[7.54, 11.25]|
|try5|4|0|NA|NA|
|kd|0|4069|0.05|[0.01, 0.11]|
|kd|1|2210|0.53|[0.27, 0.85]|
|kd|2|1799|2.81|[2.29, 3.39]|
|kd|3|598|9.06|[7.45, 10.77]|
|kd|4|2|0.28|[0.20, 0.35]|
