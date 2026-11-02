# D2: map-path mechanism and region/object channel probes

**Status:** All six CPU caches and GPU evaluations are complete. Every run has 1,839 unique val_unseen episode results; the four LLM runs each retain the expected six cache-missing episodes. This report uses the saved per-episode JSON files and exact paired McNemar tests.

**Main result:** The 28000 policy retains approximately its own-map SR with an all-zero grid (64.06 vs 64.33), a GT-legacy grid (65.36), or LLM region channels alone (64.38). Thus its navigation success does not require the episode-specific object grid. The 18600 GT policy retains 67.32 SR with only region channels and 67.26 with only object/environment channels, versus 68.41 with both and 55.68 without the map pathway. Either retained group carries most of the GT-model advantage; the two effects are substantially redundant, not additive.

## T0. Channel contract — passed

`prior/constants.py:104-132,254-265` defines the names; `prior/bbox/_rasterize.py:96-105` maps object categories directly and regions at offset `OBJECT_CATEGORIES=27`. `prior/cognitive_map_generation.py:186-194` uses the same split. The refiner uses object `:27` with an observed mask and region `27:` with a full-map loss (`scripts/refiner/train.py:24,234-245`): **no index mismatch**.

| Index | Category | Group | Index | Category | Group |
|---:|---|---|---:|---|---|
| 0 | void | object | 19 | cabinet | object |
| 1 | chair | object | 20 | chest_of_drawers | object |
| 2 | door | object | 21 | stool | object |
| 3 | table | object | 22 | towel | object |
| 4 | cushion | object | 23 | fireplace | object |
| 5 | sofa | object | 24 | gym_equipment | object |
| 6 | bed | object | 25 | seating | object |
| 7 | plant | object | 26 | clothes | object |
| 8 | sink | object | 27 | outdoor/semi-outdoor | region |
| 9 | toilet | object | 28 | living/social space | region |
| 10 | tv_monitor | object | 29 | recreation/fitness | region |
| 11 | shower | object | 30 | utility/service | region |
| 12 | bathtub | object | 31 | work/study | region |
| 13 | counter | object | 32 | circulation | region |
| 14 | appliances | object | 33 | private room | region |
| 15 | structure | object | 34 | bathroom/sanitary | region |
| 16 | other | object | 35 | dining/food | region |
| 17 | free-space | object | 36 | other/miscellaneous | region |
| 18 | picture | object | — | — | — |

Try5's map encoder first applies a bias-free `nn.Conv2d(37,512,kernel_size=1)` (`vlnce_baselines/models/etp_prior_gt/map_encoder.py:169-176`), followed by spatial tokenization, normalization, positional/metadata embeddings, and transformer processing (`:327-355`). Zeroing a channel group removes its first-layer linear contribution, **not** the entire map-token path. Also, the object slice contains `void`, `structure`, and `free-space`; `region_only` removes these environmental channels as well as named object categories.

## T1–T2. CPU cache construction and checks

`scripts/distill/make_d2_probe_caches.py` generated the four LLM variants below in **new** model keys under `data/llm_navigation/`; each has **1,833 rasters and matching `status=complete` JSON files**. The actual `llm_navigation_cache_report` returns **available=1833, missing=6** for each, matching the six own-cache misses. For every variant, 20 samples passed its actual `llm_cached_cognitive_map_to_tensors` loader, exact non-grid-field equality to the episode's own source, exact expected grid transformation, and nonzero-channel support checks. Source caches were not modified.

| New key | Grid replacement | Non-grid fields |
|---|---|---|
| `probe.llm_zero_grid.v1` | All 37 channels zero | Own LLM |
| `probe.llm_gtlegacy_grid.v1` | Same-episode `gt.legacy.r1p5.direction5.v1` grid | Own LLM |
| `probe.llm_mean_grid.v1` | One fixed training-set mean grid for every episode | Own LLM |
| `probe.llm_region_only.v1` | Own LLM grid with object channels 0–26 zero | Own LLM |

For the mean, the script reads the exact `train_90.json.gz` R2R episode IDs, mapping each to the historical LLM `r2r/train/.../R2R_train_<id>.npz` path; **9,639** of those episodes have available LLM rasters. It accumulates in float64 and saves a float32 mean. Maximum value is **0.19078742**. Nonzero fraction is **0.419924** across all 37×100×100 channel-cells; the fraction of spatial cells with any nonzero channel is **0.8920**. The per-channel means below follow the T0 index table (source: `data/llm_navigation/probe.llm_mean_grid.v1/r2r/val_unseen/manifest.json`):

| Index | Mean | Index | Mean |
|---:|---:|---:|---:|
| 0 | 0.000881959 | 19 | 0.00158282 |
| 1 | 0.00168673 | 20 | 0.000218695 |
| 2 | 0.00520303 | 21 | 0.000148107 |
| 3 | 0.00145803 | 22 | 0.0000651105 |
| 4 | 0.000277083 | 23 | 0.000177985 |
| 5 | 0.000376720 | 24 | 0.0000553999 |
| 6 | 0.000488847 | 25 | 0.000208569 |
| 7 | 0.000585164 | 26 | 0.0000490922 |
| 8 | 0.000217325 | 27 | 0.00281581 |
| 9 | 0.000197033 | 28 | 0.00353265 |
| 10 | 0.0000682228 | 29 | 0.000708123 |
| 11 | 0.000195331 | 30 | 0.000537732 |
| 12 | 0.000159519 | 31 | 0.00131337 |
| 13 | 0.000437597 | 32 | 0.00984666 |
| 14 | 0.000217823 | 33 | 0.00365254 |
| 15 | 0.0167543 | 34 | 0.00294989 |
| 16 | 0.0195030 | 35 | 0.00357344 |
| 17 | 0.00865741 | 36 | 0.000703019 |
| 18 | 0.000579064 | — | — |

The two GT variants were written only to new namespaces under `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/`. Each has **1,839 raster files**; 20 samples per variant passed the actual `cached_cognitive_map_to_tensors` loader, exact expected grid transformation, and exact non-grid-field equality to their own GT source. The source namespace was not modified.

| New GT namespace | Grid replacement | Non-grid fields | Files / loader checks |
|---|---|---|---:|
| `probe.gt_region_only.v1` | GT object channels 0–26 zero; region 27–36 retained | Own GT | 1,839 / 20 passed |
| `probe.gt_object_only.v1` | GT region channels 27–36 zero; object 0–26 retained | Own GT | 1,839 / 20 passed |

## T3. Evaluation launchers — written, not run

Every launcher inherits the named source script verbatim except **three substitutions**: run name, cache key/namespace, and port. `diff -u` found **three removed plus three added lines** for each of the six scripts. The LLM source is `scripts/distill/eval_probe_llm_gtmeta.sh` (four ranks, GPUs 1,5,6,7, `load_pretrained_map_modules=False`); the GT source is `scripts/eval_gt18600/eval_own.sh`.

| Script under repo root | Run name | Key or absolute namespace | Port |
|---|---|---|---:|
| `scripts/distill/eval_probe_llm_zero_grid.sh` | `probe_llm_zero_grid_val_unseen` | `probe.llm_zero_grid.v1` | 29650 |
| `scripts/distill/eval_probe_llm_gtlegacy_grid.sh` | `probe_llm_gtlegacy_grid_val_unseen` | `probe.llm_gtlegacy_grid.v1` | 29651 |
| `scripts/distill/eval_probe_llm_mean_grid.sh` | `probe_llm_mean_grid_val_unseen` | `probe.llm_mean_grid.v1` | 29652 |
| `scripts/distill/eval_probe_llm_region_only.sh` | `probe_llm_region_only_val_unseen` | `probe.llm_region_only.v1` | 29653 |
| `scripts/eval_gt18600/eval_gt18600_region_only.sh` | `gt18600_region_only_val_unseen` | `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/probe.gt_region_only.v1` | 29654 |
| `scripts/eval_gt18600/eval_gt18600_object_only.sh` | `gt18600_object_only_val_unseen` | `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/probe.gt_object_only.v1` | 29655 |

Six `diff -u` checks (each has **exactly three `-` and three `+` changed lines**, no other changes):

| Source → target | `--master_port` | Run-name line | Cache line | `diff -u` count |
|---|---|---|---|---:|
| gtmeta → LLM zero | `29645 → 29650` | `probe_llm_gtmeta_val_unseen → probe_llm_zero_grid_val_unseen` | `probe.llm_gtmeta.v1 → probe.llm_zero_grid.v1` | 3−/3+ |
| gtmeta → LLM GT-legacy | `29645 → 29651` | `probe_llm_gtmeta_val_unseen → probe_llm_gtlegacy_grid_val_unseen` | `probe.llm_gtmeta.v1 → probe.llm_gtlegacy_grid.v1` | 3−/3+ |
| gtmeta → LLM mean | `29645 → 29652` | `probe_llm_gtmeta_val_unseen → probe_llm_mean_grid_val_unseen` | `probe.llm_gtmeta.v1 → probe.llm_mean_grid.v1` | 3−/3+ |
| gtmeta → LLM region | `29645 → 29653` | `probe_llm_gtmeta_val_unseen → probe_llm_region_only_val_unseen` | `probe.llm_gtmeta.v1 → probe.llm_region_only.v1` | 3−/3+ |
| GT own → GT region | `29600 → 29654` | `gt18600_legacy_val_unseen_own → gt18600_region_only_val_unseen` | `gt.legacy.r1p5.direction5.v1 → /data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/probe.gt_region_only.v1` | 3−/3+ |
| GT own → GT object | `29600 → 29655` | `gt18600_legacy_val_unseen_own → gt18600_object_only_val_unseen` | `gt.legacy.r1p5.direction5.v1 → /data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/probe.gt_object_only.v1` | 3−/3+ |

`scripts/distill/run_d2.sh` runs them in table order without a GPU-memory wait. After **each** run it merges rank JSON files and requires exactly 1,839 unique episode results (and exactly six expected LLM-cache misses for LLM probes). All paths are absolute in the launchers and runner. Shell syntax checks passed. Launch manually:

```bash
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/run_d2.sh
```

## T4. Are maps auxiliary supervised targets?

The requested `data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/config.yaml` is **not present** here. Two external `config.yaml` copies conflict: `/data/xukai/etp-r1-snapshot/checkpoints/data_try5_local/logs/checkpoints/release_r2r_llm_grid_try5_dagger/config.yaml:180-191` has two map-loss weights and no `online_*`; the other `/data/xukai/etp-r1-snapshot/checkpoints/data/logs/checkpoints/release_r2r_llm_grid_try5_dagger/config.yaml:181-200` additionally lists online weights. CPU inspection of the actual `ckpt.iter28000.pth` embedded config matches the **first** copy: Try5/LLM-Grid, map enabled, and **no `online_*` keys**. The second copy is not evidence of the checkpoint's training configuration.

| MAP_ENCODER key | Actual 28000 embedded config / matching copy | Conflicting copy |
|---|---:|---:|
| `map_loss_weight` | 0.1 | 0.1 |
| `trajectory_keypoint_loss_weight` | 0.001 | 0.001 |
| `online_ghost_loss_weight` | absent | 0.1 |
| `online_grid_loss_weight` | absent | 0.3 |
| `online_progress_loss_weight` | absent | 0.1 |
| `online_recovery_loss_weight` | absent | 0.1 |
| `online_state_loss_weight` | absent | 0.05 |
| `online_visual_loss_weight` | absent | 0.0 |

Those nonzero weights are **dormant for this candidate**: neither produces a loss or sends gradients to the language encoder, graph encoder, or map encoder in this run. The LLM DAgger trainer inherits PriorGT (`ss_trainer_ETP_LLM.py:32-34`); `_prepare_map_inputs` returns `None` as map auxiliary loss (`ss_trainer_ETP_PriorGT.py:1346-1407`), and DAgger sums action cross-entropy, not a map target loss (`:1873-1884,2172-2182`). `map_loss_weight`/`trajectory_keypoint_loss_weight` readers concern other candidates, notably Imagined, not this Try5/LLM-Grid path. No active `online_*` reader was found in the checked-out DAgger trainer. By contrast, action CE backpropagates through the graph/map fusion into trainable navigation and map modules; the actual config has `freeze_base=false`, `fix_lang_embedding=false`, and `fix_pano_embedding=false` (`.../data_try5_local/.../config.yaml:184,212-213`). The fixed CLIP category initialization is an exception (`map_encoder.py:33`).

Pretraining launchers select `architecture=try5, source=llm_grid` (`scripts/submit/llm-grid-try5-pretrain.sh:7-12`); `pretrain_src/run_pt/mix_pretrain_server.json:43-57` selects **MLM and SAP**. Map tokens enter both tasks as input (`pretrain_src/pretrain_src/model/pretrain_cmt.py:233-299,358-369`), while Try5 explicitly has no updated-map/box target loss (`:379-386`; `cognitive_map_candidate.py:63-64`). The preserved `model_step_460000.pt` directory contains the weight file but **no standalone run config**; this task-list attribution follows the maintained Try5 launcher and code, not a checkpoint-embedded config. **Conclusion:** in this training pipeline, the cognitive map is an input whose representation can receive MLM/SAP and DAgger action-CE gradients through shared modules; it is **not itself a supervised map target** in either phase.

## T5. Evaluation results and paired references

`scripts/distill/summarize_d1_probes.py` merges all rank episode files, requires 1,839 distinct episode IDs per run, and prints 2×2 counts with exact McNemar p. It includes every D2 variant against its primary and historical controls, plus the pre-existing 28000 grid-only cross-scene pair, 18600 own/cross and own/nomap, and **all 12 completed E2 own/cross pairs**. Six D2 result JSON files are under `data/logs/checkpoints/<run_name>/eval_results/`; no new evaluation is needed for the table below.

| Variant | SR | SPL | OSR | nDTW | Primary comparison: both / variant-only / control-only / neither | Exact McNemar p |
|---|---:|---:|---:|---:|---|---:|
| 28000 own LLM | 64.33 | 54.16 | 69.11 | 66.19 | reference | — |
| 28000 GT-metadata control | 64.27 | 54.11 | 69.00 | 66.16 | reference | — |
| 28000 zero grid | 64.06 | 53.27 | 68.68 | 66.53 | vs GT-metadata: 1118 / 60 / 64 / 597 | 0.788 |
| 28000 GT-legacy grid | 65.36 | 55.32 | 69.87 | 66.80 | vs GT-metadata: 1161 / 41 / 21 / 616 | 0.0151 |
| 28000 train-mean grid | 62.86 | 49.07 | 72.81 | 61.02 | vs GT-metadata: 1024 / 132 / 158 / 525 | 0.142 |
| 28000 LLM region only | 64.38 | 54.12 | 69.49 | 66.17 | vs GT-metadata: 1161 / 23 / 21 / 634 | 0.880 |
| 18600 no map | 55.68 | 46.86 | 59.27 | 65.75 | reference | — |
| 18600 own GT | 68.41 | 57.88 | 73.30 | 68.55 | reference | — |
| 18600 GT region only | 67.32 | 57.84 | 72.49 | 68.43 | vs own: 1182 / 56 / 76 / 525 | 0.0978 |
| 18600 GT object only | 67.26 | 57.33 | 71.78 | 68.62 | vs own: 1199 / 38 / 59 / 543 | 0.0417 |

Additional exact pairs: 28000 zero/GT-legacy/mean/region-only versus original own have p = 0.723/0.0226/0.126/1.000, respectively. GT region-only versus nomap has 951/287/73/528 (p = 4.06e-31); GT object-only versus nomap has 958/279/66/536 (p = 2.71e-32). These are exploratory multiple comparisons; the marginal object-only-vs-own p = 0.0417 should not be treated as a corrected discovery.

Relative to the 18600 own-minus-nomap SR gap of 12.73 points, region-only retains 11.64 points (**91.4%**) and object-only retains 11.58 points (**91.0%**). Their sum exceeding 100% indicates overlapping/redundant useful information, not separable causal shares. The result supports a region-only navigation path **when the region layout is GT**; it does not establish that the LLM-predicted region layout has sufficient quality. The 28000 mean grid has a substantial SPL and nDTW drop despite a smaller, nonsignificant SR drop; this fixed mean is also a diffuse, out-of-distribution raster, so it does not by itself establish a need for episode-specific map content.

| Existing pair | Both succeed | First only | Second only | Both fail | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| 28000 grid-only cross-scene s0 vs raster own | 1170 | 44 | 27 | 598 | 0.0568 |
| 18600 own vs cross-scene | 954 | 304 | 70 | 511 | 7.16e-36 |
| 18600 own vs nomap | 960 | 298 | 64 | 517 | 3.25e-37 |

| E2 iter | Own SR | Cross SR | Both succeed | Own only | Cross only | Both fail | Exact McNemar p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16000 | 63.84 | 65.42 | 1094 | 80 | 109 | 556 | 0.0414 |
| 16800 | 63.08 | 63.68 | 1074 | 86 | 97 | 582 | 0.460 |
| 17200 | 64.55 | 63.40 | 1076 | 111 | 90 | 562 | 0.158 |
| 17600 | 64.27 | 64.71 | 1100 | 82 | 90 | 567 | 0.594 |
| 18000 | 63.95 | 63.68 | 1088 | 88 | 83 | 580 | 0.760 |
| 19200 | 63.46 | 63.68 | 1070 | 97 | 101 | 571 | 0.831 |
| 20000 | 62.75 | 64.00 | 1061 | 93 | 116 | 569 | 0.128 |
| 21200 | 64.22 | 63.78 | 1075 | 106 | 98 | 560 | 0.624 |
| 21600 | 64.76 | 64.17 | 1081 | 110 | 99 | 549 | 0.489 |
| 22000 | 63.73 | 64.44 | 1093 | 79 | 92 | 575 | 0.359 |
| 22400 | 63.02 | 63.68 | 1057 | 102 | 114 | 566 | 0.454 |
| 22800 | 63.78 | 63.68 | 1087 | 86 | 84 | 582 | 0.939 |

Reproduce or refresh the CPU summary with:

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/summarize_d1_probes.py
```

## Interpretation and limits

- The measured 28000 zero grid has essentially the same SR as own/GT-metadata; this is evidence that episode-specific raster content is unnecessary for this checkpoint's SR **under retained metadata and the active map-token path**. The mean grid instead hurts SPL/nDTW, so a stronger claim that *any* fixed raster texture is equivalent is false. A constant-path/register effect remains plausible, not proven; metadata and distribution effects have not been eliminated.
- For 18600, `(region_only SR − nomap SR)/(own SR − nomap SR)` and its object-only analogue **descriptively** measure how much of the **12.73-point** own-vs-nomap gap each retained group recovers. They are not causal shares, because `nomap` disables the map pathway itself as well as content. The fractions need not sum to one: nonlinear interactions can make the groups complementary or redundant. Because `region_only` also removes `void`, `structure`, and `free-space`, it is a stricter “region only” test than removing only discrete object semantics.
- GT `region_only` reaches 67.32 SR, so region information alone preserves much of the observed advantage. The most direct next diagnostic is LLM-predicted **region-channel IoU** against GT, separately from whole-grid IoU; the current experiment cannot attribute the LLM shortfall specifically to region quality.
- T4 found no active map-target auxiliary supervision for Try5/LLM-Grid; the paper should not describe this checkpoint's map contribution as a training-time supervised map reconstruction effect.
