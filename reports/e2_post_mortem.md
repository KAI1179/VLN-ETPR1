# E2 post-mortem: coordinate scale, initialization, and evaluation control

## Verdict

**Coordinate scale is correct** for E2's LLM raster, not “should be 2 but used 1.” The cached `start_position` is in metres, while the raster cell is 0.5 m: the code's effective factor of two is already supplied by division by `CELL_SIZE=0.5`. On all 50 sampled `val_unseen` episodes, the computed start cell matches both GT representations exactly; the match extends to every available `val_unseen` cache. No validation start is out of bounds.

E2's map encoder and fusion attention were **initialized at DAgger**, rather than inherited from pretraining. That also describes the historical Try5 control, so it is not by itself an E2-specific explanation. Existing own/cross-scene results have a **3-rank versus 4-rank confound**; a same-rank rerun is prepared but has not been launched. The evidence does not establish a single cause for E2's lack of own-map advantage, and it does **not** justify declaring coordinate fusion intrinsically ineffective.

This report used source inspection and CPU-only cache/checkpoint reads. No GPU job was started. Results below refer to the E2 training run at commit `4fd623142f4c727a33e8c3a42e2352281b2572db` and its HEAD-harness evaluations at `2efac8518da80e6da80d942157bf85b8c231d047`.

## T1. Coordinate transform

The implemented transform is:

```text
s = npz.start_position                    # (row along world x, column along world z)
s_encoder = s * start_position_scale      # map-encoder input only
origin_xz = episode.start_world_xz - s * start_position_meters_per_unit
cell(world) = (world_xz - origin_xz) / CELL_SIZE
CELL_SIZE = 0.5 m
start_cell = s * start_position_meters_per_unit / 0.5
```

Evidence: `vlnce_baselines/ss_trainer_ETP_PriorGT.py:1373-1378,1422-1459`; `vlnce_baselines/models/etp_prior_gt/map_utils.py:94-130`. There is **no extra ×2, rounding, integer cast, axis flip, or clamp** in this path. The `start_position_scale` multiplies the map-encoder input but **does not** enter the origin calculation; this distinction matters for other configurations, although E2 sets both relevant scales to 1.0 (`data/logs/checkpoints/dagger_coordfusion_llm/config.yaml:222-223`; `scripts/eval_e2/eval_one.sh:89-90`).

LLM and legacy GT `start_position` are metres, so `meters_per_unit=1.0` yields `start_cell=2s`. The fine online GT cache stores the same start in 0.5-m cell units; its corresponding `meters_per_unit=0.5` yields `start_cell=s_fine`. Using 1.0 on **fine** would double the fine cell and would be the wrong comparison; E2 consumes the **LLM** cache, not the fine cache.

### First 50 val_unseen episodes

Continuous raster cells, rounded to 0.01 for display. Equality and zero differences were checked before rounding. The LLM cells also equal the two GT columns for all 50.

| Episode | cell_legacy | cell_fine | legacy − fine |
|---:|---:|---:|---:|
| 1 | (36.58, 11.10) | (36.58, 11.10) | (0, 0) |
| 2 | (36.58, 11.10) | (36.58, 11.10) | (0, 0) |
| 3 | (36.58, 11.10) | (36.58, 11.10) | (0, 0) |
| 4 | (9.88, 20.57) | (9.88, 20.57) | (0, 0) |
| 5 | (9.88, 20.57) | (9.88, 20.57) | (0, 0) |
| 6 | (9.88, 20.57) | (9.88, 20.57) | (0, 0) |
| 7 | (27.95, 31.79) | (27.95, 31.79) | (0, 0) |
| 8 | (27.95, 31.79) | (27.95, 31.79) | (0, 0) |
| 9 | (27.95, 31.79) | (27.95, 31.79) | (0, 0) |
| 10 | (54.00, 41.36) | (54.00, 41.36) | (0, 0) |
| 11 | (54.00, 41.36) | (54.00, 41.36) | (0, 0) |
| 12 | (54.00, 41.36) | (54.00, 41.36) | (0, 0) |
| 13 | (41.37, 20.50) | (41.37, 20.50) | (0, 0) |
| 14 | (41.37, 20.50) | (41.37, 20.50) | (0, 0) |
| 15 | (41.37, 20.50) | (41.37, 20.50) | (0, 0) |
| 16 | (62.65, 44.37) | (62.65, 44.37) | (0, 0) |
| 17 | (62.65, 44.37) | (62.65, 44.37) | (0, 0) |
| 18 | (62.65, 44.37) | (62.65, 44.37) | (0, 0) |
| 19 | (13.72, 15.78) | (13.72, 15.78) | (0, 0) |
| 20 | (13.72, 15.78) | (13.72, 15.78) | (0, 0) |
| 21 | (13.72, 15.78) | (13.72, 15.78) | (0, 0) |
| 22 | (11.53, 2.63) | (11.53, 2.63) | (0, 0) |
| 23 | (11.53, 2.63) | (11.53, 2.63) | (0, 0) |
| 24 | (11.53, 2.63) | (11.53, 2.63) | (0, 0) |
| 25 | (9.66, 12.42) | (9.66, 12.42) | (0, 0) |
| 26 | (9.66, 12.42) | (9.66, 12.42) | (0, 0) |
| 27 | (9.66, 12.42) | (9.66, 12.42) | (0, 0) |
| 28 | (3.08, 4.39) | (3.08, 4.39) | (0, 0) |
| 29 | (3.08, 4.39) | (3.08, 4.39) | (0, 0) |
| 30 | (3.08, 4.39) | (3.08, 4.39) | (0, 0) |
| 31 | (15.54, 10.00) | (15.54, 10.00) | (0, 0) |
| 32 | (15.54, 10.00) | (15.54, 10.00) | (0, 0) |
| 33 | (15.54, 10.00) | (15.54, 10.00) | (0, 0) |
| 34 | (11.72, 4.79) | (11.72, 4.79) | (0, 0) |
| 35 | (11.72, 4.79) | (11.72, 4.79) | (0, 0) |
| 36 | (11.72, 4.79) | (11.72, 4.79) | (0, 0) |
| 37 | (2.38, 12.18) | (2.38, 12.18) | (0, 0) |
| 38 | (2.38, 12.18) | (2.38, 12.18) | (0, 0) |
| 39 | (2.38, 12.18) | (2.38, 12.18) | (0, 0) |
| 40 | (35.22, 19.63) | (35.22, 19.63) | (0, 0) |
| 41 | (35.22, 19.63) | (35.22, 19.63) | (0, 0) |
| 42 | (35.22, 19.63) | (35.22, 19.63) | (0, 0) |
| 43 | (30.09, 18.89) | (30.09, 18.89) | (0, 0) |
| 44 | (30.09, 18.89) | (30.09, 18.89) | (0, 0) |
| 45 | (30.09, 18.89) | (30.09, 18.89) | (0, 0) |
| 46 | (13.90, 14.40) | (13.90, 14.40) | (0, 0) |
| 47 | (13.90, 14.40) | (13.90, 14.40) | (0, 0) |
| 48 | (13.90, 14.40) | (13.90, 14.40) | (0, 0) |
| 49 | (26.28, 24.33) | (26.28, 24.33) | (0, 0) |
| 50 | (26.28, 24.33) | (26.28, 24.33) | (0, 0) |

All 50 differences are **exactly zero**, not a constant factor of two or a variable offset. A CPU scan of the full split found the same legacy/fine cell for **1,839/1,839** episodes and the same legacy/LLM cell for **1,833/1,833** available LLM caches. The other six LLM caches are absent: 308, 349, 455, 804, 1318, 1331. LLM start-cell out-of-bounds count is **0/50** in the sample and **0/1,833 available** (or 0/1,839 total, with six undefined rather than counted in-bounds). GT legacy and fine are likewise **0/1,839** out of bounds. Inputs: `data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster/`, `data/cognitive_maps/gt.online121c369.r1p5.direction5.v1/raster/`, and `data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster/`; calculation uses the cited functions and cache-specific `meters_per_unit`.

### What “1,100 such episodes so far” counts

`_start_outside_raster` increments once per map attached to a rollout, **per rank/process** (`ss_trainer_ETP_PriorGT.py:1424-1447,1725-1726`). In training, each optimizer iteration calls one rollout, resets its environments, and initially attaches one map per environment (`:935-954,1580-1600`). The launcher records **4 ranks × 4 environments/rank** (`data/logs/checkpoints/dagger_coordfusion_llm/launch_info.txt:20-24`; `scripts/distill/run_dagger_coord_fusion.sh:84-86,114-125`). Thus the local denominator after approximately iteration 11,861 is `11,861 × 4 ≈ 47,444` episode starts **on that rank**, or about 2.32% for a local count of 1,100; aggregating 1,100 over all four ranks would be wrong. The interleaved progress output does not identify the precise iteration of each rank's warning, so 47,444 is an estimate, not an exact denominator. The log contains multiple launch blocks and the counter restarts with each process; the approximately 11,861-iteration comparison refers to the relevant uninterrupted launch. Independent CPU scanning of `train_90` found **198/9,639 (2.05%)** out-of-bounds starts in available LLM caches and **200/9,711 (2.06%)** in available legacy GT caches. That similarity supports the fixed 50-m window explanation over an LLM-only unit error; it does not prove every out-of-bounds training episode is harmless.

**T1 classification: scale correct.** In particular, the conditional warning requested by the task book—“E2 training and evaluation graph↔raster alignment was never correct”—does **not** apply.

## T2. Pretraining-weight transfer

`vlnce_baselines/models/etp_prior_gt/vlnbert_init.py:27-49,99-108` prints `graph_map_attention: no pretrained tensors loaded` when the remapped fusion dictionary is empty. E2 explicitly sets `MODEL.MAP_ENCODER.load_pretrained_map_modules=False` (`scripts/distill/run_dagger_coord_fusion.sh:73`; saved `config.yaml:215`), so this message is expected; it does **not** demonstrate a coordinate-fusion shape mismatch. The historical 28,000-step control's training code predates this message, so its old training log cannot contain the literal. Its modern evaluation log does print the message (`data/logs/checkpoints/llm_grid_try5_iter28000_raster_control/eval.log:53`). The 65.09 raster control and 64.33 HEAD control use the same historical 28,000-step checkpoint; historical code audit `reports/R13_try5_map_loading_audit.md:3-17,75-76` found neither a map-encoder loader nor a fusion-key remap at that training commit. A present-day evaluation message alone should not be misread as proof about historical training.

CPU comparison: pretrained `model_step_460000.pt` to E2's earliest retained `ckpt.iter1200.pth`. Keys were mapped `map_encoder.X → net.module.map_encoder.X` and `bert.global_encoder.graph_map_attention.X → net.module.vln_bert.graph_map_attention.X`; shapes matched. Cosines are flattened float-tensor cosine similarities. `>0.99` is the task-book's inheritance threshold; cosine is undefined when a norm is zero or nonfinite. This checkpoint is **1,200 steps after initialization**, so the explicit disabled-loading flag—not cosine alone—is decisive for initialization.

| Pretrained tensor suffix (prefix noted above) | Cosine |
|---|---:|
| graph_map_attention.attention.in_proj_bias | +0.033703 |
| graph_map_attention.attention.in_proj_weight | −0.000662 |
| graph_map_attention.attention.out_proj.bias | −0.011886 |
| graph_map_attention.attention.out_proj.weight | −0.001484 |
| graph_map_attention.residual_projection.bias | +0.037146 |
| graph_map_attention.residual_projection.weight | −0.000241 |
| map_encoder.category_projection.weight | 0.999995 |
| map_encoder.metadata_encoder.0.bias | −0.034531 |
| map_encoder.metadata_encoder.0.weight | −0.001053 |
| map_encoder.metadata_encoder.1.bias | +0.050444 |
| map_encoder.metadata_encoder.1.weight | 0.999984 |
| map_encoder.metadata_encoder.3.bias | +0.026129 |
| map_encoder.metadata_encoder.3.weight | +0.000649 |
| map_encoder.output_norm.bias | +0.039636 |
| map_encoder.output_norm.weight | 0.999190 |
| map_encoder.spatial_token_norm.bias | +0.018042 |
| map_encoder.spatial_token_norm.weight | 1.000000 |
| map_encoder.spatial_tokenizer.bias | undefined (pretrained norm infinite; E2 norm 0.0702) |
| map_encoder.spatial_tokenizer.weight | undefined (pretrained norm 0; E2 norm 15.9835) |
| map_encoder.token_transformer.layers.0.linear1.bias | +0.021582 |
| map_encoder.token_transformer.layers.0.linear1.weight | −0.000043 |
| map_encoder.token_transformer.layers.0.linear2.bias | −0.043791 |
| map_encoder.token_transformer.layers.0.linear2.weight | +0.000237 |
| map_encoder.token_transformer.layers.0.norm1.bias | +0.029545 |
| map_encoder.token_transformer.layers.0.norm1.weight | 0.999966 |
| map_encoder.token_transformer.layers.0.norm2.bias | +0.092778 |
| map_encoder.token_transformer.layers.0.norm2.weight | 0.999620 |
| map_encoder.token_transformer.layers.0.self_attn.in_proj_bias | −0.010043 |
| map_encoder.token_transformer.layers.0.self_attn.in_proj_weight | −0.000946 |
| map_encoder.token_transformer.layers.0.self_attn.out_proj.bias | +0.035187 |
| map_encoder.token_transformer.layers.0.self_attn.out_proj.weight | −0.001181 |
| map_encoder.token_transformer.layers.1.linear1.bias | +0.007292 |
| map_encoder.token_transformer.layers.1.linear1.weight | −0.000311 |
| map_encoder.token_transformer.layers.1.linear2.bias | +0.026947 |
| map_encoder.token_transformer.layers.1.linear2.weight | +0.000189 |
| map_encoder.token_transformer.layers.1.norm1.bias | +0.038476 |
| map_encoder.token_transformer.layers.1.norm1.weight | 0.999871 |
| map_encoder.token_transformer.layers.1.norm2.bias | +0.068247 |
| map_encoder.token_transformer.layers.1.norm2.weight | 0.999633 |
| map_encoder.token_transformer.layers.1.self_attn.in_proj_bias | +0.019855 |
| map_encoder.token_transformer.layers.1.self_attn.in_proj_weight | −0.000728 |
| map_encoder.token_transformer.layers.1.self_attn.out_proj.bias | +0.022888 |
| map_encoder.token_transformer.layers.1.self_attn.out_proj.weight | −0.000042 |

Only eight of the 35 map-encoder tensors with finite nonzero norms exceed 0.99; they are category-projection or normalization scale/affine weights, which can match because of shared initialization. The main learned matrices and all six fusion tensors are near zero cosine. **E2's map encoder/fusion started from DAgger initialization; the historical Try5 control also lacked that pretraining transfer.**

## T3. Controlled rerun prepared, not launched

Run manually when GPU 5,6,7 are available:

```bash
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_e2/rerun_same_rank.sh
```

The launcher evaluates iter 21600 own then cross_scene, both with `E2_GPUS=5,6,7`, world size 3, unchanged evaluation arguments, separate output names `iter21600_own_w3` and `iter21600_cross_scene_w3`, and ports 29620/29621. It reuses `eval_one.sh` with only optional output-suffix and port overrides; existing runs remain untouched. After both complete:

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_e2/summarize.py
```

The summary will print the w3 pair's SR/SPL/NE/OSR/nDTW and paired 2×2 success counts plus exact McNemar p. No rerun numbers are claimed in this report.

For context only, the existing **mixed-rank** iter-21600 pair is own SR/SPL 64.76/53.91 versus cross_scene 64.17/53.70; its paired success table is both 1,081, own-only 110, cross-only 99, neither 549 (exact McNemar p=0.489). Those numbers come from the saved per-episode JSON via `summarize.py`; they are not the controlled rerun. The summary script now discovers completed pairs from their variant-local `done` markers, matching what `eval_one.sh` actually writes.

## Remaining attribution limit

The val_unseen coordinate check rules out a start-position factor-of-two error **on the evaluated episodes**. The DAgger log's out-of-bounds count is compatible with the approximately 2% OOB rate shared by GT and LLM **training** caches; its precise rank-local denominator is not recoverable from the interleaved log alone. The initialization audit identifies a design choice shared with the historical control, not a unique E2 failure. The same-rank rerun addresses the known evaluation confound. Even if it again shows no own-map advantage, that would establish weak episode-specific raster sensitivity **for this trained checkpoint and test**, not prove every coordinate-fusion design ineffective.
