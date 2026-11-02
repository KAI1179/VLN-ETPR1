# D3: LLM region-channel quality and region-only DAgger preparation

**T1–T2 decision: do not launch D3 training yet.** On 1,833 shared `val_unseen` episodes, LLM-vs-legacy-GT region IoU is only **0.0445** at 100×100 and **0.1427** at 10×10, with a **4.53 m** median centroid offset. Both IoUs are below the task's stop thresholds (0.2 and 0.4). High region *presence* recall (0.684) does not rescue the spatial layout. Training a region-only policy on these maps is therefore likely to learn to ignore their layout; this is a prediction, not a measured D3 training result. The train/validation quality gap is large and its cause is not established.

## T1. CPU-only channel quality

Sources: LLM key `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree` under `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/llm_navigation/`; GT namespace `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster/`. The reproducible scorer is `scripts/distill/analyze_d3_region_quality.py`; exact values and sample counts are in `reports/d3_region_quality_metrics.json`. Files are paired by scene and `R2R_<split>_<episode>.npz`; `train_90` IDs come from `data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train_90.json.gz`. Counts: **1,833/1,839** `val_unseen`; **9,639/9,738** `train_90`.

GT grid values are binarized with `>0`, so 0.6 is positive. Each channel's IoU is computed per episode, then averaged across episode-channel pairs; both-empty pairs are skipped and counted. Presence precision/recall are per-episode group-category scores averaged over episodes with a nonempty prediction/GT denominator. Centroid distance uses 0.5 m per cell and only pairs present on both sides. Coarse IoU marks each 10×10 block positive if *any* underlying cell is positive. The environment group `{0,15,17}` is a subset of the object slice `0–26`, not an additional disjoint set.

This is a **raw cell-aligned** comparison; it does not optimize over rotations or translations. A read-only metadata spot check of 200 paired files per split found identical `start_position` and `start_direction_vector` in all 200 pairs for both splits, but only 0/200 identical `direction_vectors` and `range_y` in val_unseen (train: 29/200 and 0/200). Thus an alignment/metadata contribution to the low raster IoU is not excluded by this scorer; the reported score still describes the cached input in its native grid coordinates.

| Split | Group | 100×100 IoU mean / median | Presence precision / recall | Centroid median / P75 (m) | 10×10 IoU mean | Both-empty skipped |
|---|---|---:|---:|---:|---:|---:|
| val_unseen | region 27–36 | 0.0445 / 0.0000 | 0.6584 / 0.6840 | 4.53 / 6.99 | 0.1427 | 9,555 |
| val_unseen | object 0–26 | 0.0359 / 0.0000 | 0.7045 / 0.6669 | 4.41 / 6.83 | 0.1508 | 23,681 |
| val_unseen | environment 0,15,17 | 0.0851 / 0.0376 | 0.9571 / 0.8632 | 4.01 / 6.14 | 0.3073 | 555 |
| train_90 | region 27–36 | 0.3339 / 0.2993 | 0.9192 / 0.9128 | 0.92 / 3.87 | 0.5565 | 57,706 |
| train_90 | object 0–26 | 0.2093 / 0.1491 | 0.8785 / 0.8805 | 1.05 / 3.88 | 0.5103 | 147,782 |
| train_90 | environment 0,15,17 | 0.2967 / 0.2555 | 0.9766 / 0.9657 | 1.41 / 3.91 | 0.6173 | 5,394 |

Per-region-channel IoU is conditional on at least one side being nonempty; recall is the fraction of GT-present episodes where the LLM channel is also present. Channel names follow `prior/constants.py:254-265`.

| Channel | Region | val IoU | val recall | train_90 IoU | train_90 recall |
|---:|---|---:|---:|---:|---:|
| 27 | outdoor/semi-outdoor | 0.0080 | 0.4048 | 0.3477 | 0.8886 |
| 28 | living/social space | 0.0291 | 0.4996 | 0.3119 | 0.8719 |
| 29 | recreation/fitness | 0.0274 | 0.5586 | 0.4154 | 0.8878 |
| 30 | utility/service | 0.0072 | 0.3421 | 0.2579 | 0.8230 |
| 31 | work/study | 0.0055 | 0.1699 | 0.3493 | 0.8066 |
| 32 | circulation | 0.1101 | 0.9406 | 0.4110 | 0.9800 |
| 33 | private room | 0.0523 | 0.7263 | 0.3274 | 0.8941 |
| 34 | bathroom/sanitary | 0.0240 | 0.7980 | 0.2918 | 0.9443 |
| 35 | dining/food | 0.0391 | 0.6243 | 0.2939 | 0.8533 |
| 36 | other/miscellaneous | 0.0071 | 0.2041 | 0.2508 | 0.7502 |

## T2. Interpretation

The specified *go* condition is not met: `val_unseen` region IoU is far below 0.4, and its coarse IoU is far below 0.6 while centroid offset exceeds 2 m. The specified *stop* condition is met on both IoUs. Object quality is similarly low (0.0359/0.1508 fine/coarse), so D0's 18600+LLM result of 52.47 SR is consistent with a model trained on spatially accurate GT maps receiving spatially poor LLM maps. This does **not** prove the map mismatch is the only cause of D0 failure. The 0.3339 train-versus-0.0445 val region-IoU gap merits separate investigation before investing in DAgger: it may reflect distribution shift, cache generation differences, or coordinate/alignment differences; no cause is established here.

Recommended next action: improve/evaluate the LLM region prediction as its own low-dimensional task or revisit a visual-evidence refiner. T3 scripts and caches are preparation only, not approval to spend GPU time.

## T3. Training preparation

**Prepared only. Do not launch given the T2 stop result unless you explicitly choose to investigate the training failure mode.** The new model key `llm.region_only.s2tagfree.v1` contains only a 0–26 channel mask over the original LLM raster; region channels 27–36 and all non-grid fields are elementwise identical to the source arrays. Every written raster has a `status=complete` JSON. The first 20 available episodes of each split passed the actual cache loader and source-field equality checks (`scripts/distill/make_d3_region_only_cache.py`). The source cache was not modified.

| Split | Dataset episodes | New cache available | Missing | Source-equivalent note |
|---|---:|---:|---:|---|
| train_90, stored in `r2r/train` | 9,738 | 9,639 | 99 | Only train_90 IDs were copied; full-`train` `cache_report` sees 9,639 available / 1,180 missing across its larger denominator. |
| val_seen | 778 | 768 | 10 | Same missing count as source. |
| val_unseen | 1,839 | 1,833 | 6 | Same missing episode IDs as source. |

Root cache manifest: `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/llm_navigation/llm.region_only.s2tagfree.v1/manifest.json`, SHA256 `8c3660b488ec1126f8673d7d611ea8ef5bd6b58176c7680d0d19a7df1e3fe491`. Split-level manifests were added to the **new** key so the existing donor builder can consume it. The seed-0 region-only cross-scene donor key is `llm.region_only.s2tagfree.v1__raster-cross_scene-s0`: 1,833 episodes, same-scene rate 0, identical-grid rate 0, 1,151 distinct donor episodes (`scripts/distill/make_raster_donor_cache.py` output). The actual `llm_navigation_cache_report` returns **1,833 available / 6 missing** for both the own-region and donor keys. The original full-map seed-0 donor key remains separate.

### Two matched DAgger launchers

The two wrappers `scripts/distill/train_llm_region_only_dagger.sh` and `scripts/distill/train_llm_full_dagger_control.sh` call one identical command in `scripts/distill/train_llm_d3.sh`. Their **only three argument differences** are:

| Argument | Region-only | Full control |
|---|---|---|
| `MODEL.MAP_ENCODER.llm_cache_model_key` | `llm.region_only.s2tagfree.v1` | `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree` |
| Run/checkpoint directory | `data/logs/checkpoints/dagger_llm_region_only/` | `data/logs/checkpoints/dagger_llm_full_control/` |
| Port | 29660 | 29661 |

Both use four GPUs specified by `D3_GPUS`, four environments per rank, seed 100, `MODEL.policy_name=LLMGridTry5Policy`, Try5/LLM-Grid, 460000 pretraining checkpoint, `lr=1e-5`, batch size 1, 30,000 iterations, teleport/tryout, elevation `y`, coordinate fusion off, no refiner, no GT teacher, and the historical CE schedule (`ml_weight=1`, sample ratio 0.75, decay 2000, warmup 500). The original historical path `/home/xukai/code/ETP-R1-try5/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt` and this launcher's `/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt` have identical SHA256 `3ea8868fcb6d5d225b906cbe4ea6781fd7fc76697842ca5ed7a29761fce0fffb`.

The historical 28000 configuration is `/data/xukai/etp-r1-snapshot/checkpoints/data_try5_local/logs/checkpoints/release_r2r_llm_grid_try5_dagger/config.yaml`, corroborated by the 28000 checkpoint's embedded config. Relative to its effective training settings, the D3 command differs as follows; thus the requested "only key/output/port" historical diff is **not literally attainable on HEAD**:

| Key | Historical 28000 | Both D3 runs | Reason |
|---|---|---|---|
| Cache key | Original full key | Region-only or original full key | Experimental variable. |
| Output run and port | Historical run; historical port not recorded in config | Separate names; 29660/29661 | Isolation. |
| `MODEL.pretrained_path` | Try5-repo path | Snapshot-repo path | Same bytes by SHA256; path relocation only. |
| `MODEL.MAP_ENCODER.load_pretrained_map_modules` | Absent in old code/config (no such loading path) | `False` | Explicitly preserves historical no-transfer behavior; HEAD default is `True` and can reject the zero pretrained spatial tokenizer. |
| `MODEL.MAP_ENCODER.coordinate_fusion`, `refiner_ckpt`, `MODEL.elevation_axis`, `IL.gt_teacher_enabled` | New keys absent | `False`, empty, `y`, `False` | Explicit HEAD behavior; not extra D3 treatment differences. |
| `CHECKPOINT_INTERVAL`, `ONLY_LAST_SAVEALL` | `-1`, `True` | `400`, `False` | Required to keep intermediate checkpoints for the requested per-iter own/cross/zero evaluations. It changes checkpoint I/O, not model loss. |

All other listed training hyperparameters match the historical config; the two D3 arms are internally matched under the **same HEAD code**. This does not make either run identical to a historical-commit reproduction. Each launcher writes `provenance.txt` with HEAD hash and its source-cache manifest SHA256. `bash -n` passed; no dry-run or GPU process was started.

If you later authorize training, use separate tmux sessions and distinct four-card assignments (shown with example GPUs only):

```bash
D3_GPUS=1,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/train_llm_region_only_dagger.sh
D3_GPUS=1,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/train_llm_full_dagger_control.sh
```

Do **not** run these simultaneously on the same four cards. `scripts/eval_d3/eval_one.sh {region|full} ITER {own|cross_scene|zero_grid}` and `run_iter.sh {region|full} ITER` evaluate saved checkpoints. The optional `watch_and_eval.sh {region|full} ITER [ITER ...]` waits for explicit checkpoint iterations, then runs the three variants serially. Evaluations default to `D3_EVAL_GPUS=1,5,6,7` (three- or four-rank lists accepted), use ports 29662–29667, record cache-manifest SHA256 in `provenance.txt`, and require 1,839 distinct episode outcomes; no 12 GiB GPU-free gate is imposed. For a run's final or selected checkpoint, compare **own versus cross-scene per episode** as the primary test of whether it learned region-specific raster content. Zero-grid is an additional path-dependence control, not a substitute for the donor comparison.

Example manual evaluation **only if a D3 checkpoint is later trained**:

```bash
D3_EVAL_GPUS=1,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d3/run_iter.sh region 4000
D3_EVAL_GPUS=1,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d3/run_iter.sh full 4000
```
