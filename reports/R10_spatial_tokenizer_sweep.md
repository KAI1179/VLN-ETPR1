# R10 spatial-tokenizer sweep

The CPU sweep used `scripts/distill/map_encoder_weight_sweep.py` and `etpr1-py38`.

| step | st_w norm | st_w max | st_w 0% | st_b norm | st_b max | st_b 0% | source |
|---:|---:|---:|---:|---:|---:|---:|---|
|460000|0|0|1.000|2.430e36|1.718e36|0.990|llm-grid pretrain|
|387500|0|0|1.000|5.518e1|1.029e1|0.014|prior-GT pretrain|
|5000|1.639e1|1.206e-2|0|7.089e-2|4.970e-3|0|R5 DAgger|
|10000|1.716e1|1.560e-2|0|7.205e-2|5.420e-3|0|R5 DAgger|
|14000|1.776e1|1.704e-2|0|7.279e-2|5.958e-3|0|R5 DAgger|
|17000|0|0|1.000|2.430e36|1.718e36|0.990|R7 DAgger|
|20000|0|0|1.000|2.430e36|1.718e36|0.990|R7 DAgger|

The R7/LLM-grid spatial weight is already exactly zero at the earliest available checkpoint and remains zero; the bias is enormous. R5 has fresh-init-like, nonzero spatial weights through its available sweep, while R7 remains raster-blind. The exact death step before 460000 is unavailable because only that pretraining checkpoint is present.
