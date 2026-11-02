# D4: LLM-map generalization and val_seen map-use probes

**Status:** T1–T4 and the five user-run `val_seen` evaluations are complete. Map quality follows `train_90 > val_seen > val_unseen`, but this does **not** isolate instruction memorization from scene familiarity or the old prompt's spatial-contract error. The 28000 policy shows no significant own-map advantage on `val_seen`; the GT-trained 18600 policy uses GT map content strongly, but the LLM map does not significantly outperform a cross-scene GT donor.

## T1. Provenance and leakage audit

The maintained SFT launcher `scripts/submit/llm-grid-train-r1p5.sh:14-31` invokes `vlnce_baselines/models/etp_llm/llm_grid_train.py` for 10 epochs, LoRA rank/alpha/dropout 32/64/0.05, using only R2R `train` and English RxR `train` (`llm_grid_train.py:94,1119-1131`). The source corpus had **10,819 R2R** and **19,996 RxR** discovered English train episodes; historical records report **10,786 + 19,954 = 30,740** with required cached targets (`docs/daily/2026-08-07.md:380`, `CONTEXT.md:14-20`). The actual retained SFT-example count can differ after length filtering. The final adapter path recorded by the cache manifests is `outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag/checkpoints/final`, but neither it nor this run's `training_manifest.jsonl` is present locally. A read-only SSH check of the permitted remote `~/run/ETP-R1/` failed because `scze096-blsc` DNS resolution is unavailable in this environment. Thus its **actual retained count, exact scene set and exact membership of each train_90 instruction cannot be verified from the run artifact**; do not substitute the configured/raw corpus for the missing manifest.

The three original LLM cache manifests at `/data/xukai/etp-r1-snapshot/runtime-data/data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/{train,val_seen,val_unseen}/manifest.json` record the **same** final adapter path, scale 2, input limit 1,024, new-token limit 4,096, and system-prompt SHA256 `6de023cedc1409ce82c5d93d52083b8b980f788b380ea475c69451370cf28409`. `scripts/submit/llm-grid-nav-cache-r1p5.sh` is the generation launcher; `llm_grid_navigation_cache.py:244-249` uses the shared generation helper, whose `_generation_kwargs` sets `do_sample=False` (`llm_boxes_train.py:991-1000`). Temperature and top-p are not set; generation is greedy. The manifest creation times are:

| Split | Created UTC | Recorded LoRA / decode differences |
|---|---|---|
| train | 2026-07-20 22:00:15 | None found |
| val_seen | 2026-07-21 02:28:16 | None found |
| val_unseen | 2026-07-21 07:33:00 | None found |

Read-only dataset comparison: R2R train has **61 scenes**; val_seen has **53 scenes, all within those 61**; val_unseen has **11 unseen scenes**. Only **1/778** val_seen and **2/1,839** val_unseen instruction strings occur verbatim in raw R2R train. This rules out *verbatim R2R-train instruction reuse* as an explanation for val_seen quality, but not semantic paraphrase familiarity, RxR overlap, or scene familiarity. The requested `train_90` SFT-in/out split and its separate IoU/centroid metrics are **unavailable** without the actual SFT training manifest; no unverified membership labels were invented.

Historical same-checkpoint evidence in `docs/daily/2026-07-22.md:106-112` shows seen-scene raster IoU rising with more SFT epochs while unseen-scene IoU mostly plateaued; `docs/daily/2026-07-23.md:42-52` also documents an incorrect row/column and direction prompt contract in the *old* model used by this key. **One-sentence attribution:** generation-setting differences are ruled out by the recorded manifests; the gap is most consistent with scene-dependent spatial generalization failure, potentially amplified by the old prompt contract, but **scene memorization versus other scene-correlated difficulty is not causally identified**, and instruction-level SFT leakage remains unmeasured.

## T2. val_seen map quality, same D3 metric definitions

Reproducible CPU wrapper: `scripts/distill/analyze_d4_val_seen_quality.py`; full exact values and support counts: `reports/d4_region_quality_metrics.json`. It reuses the D3 scorer: GT 0.6 counts positive, per-episode/per-channel IoU skips both-empty pairs, presence P/R are per-episode group-category macro averages, coarse 10×10 uses any-positive blocks, and centroids use 0.5 m/cell. The LLM cache covers **9,639/9,738** `train_90`, **768/778** `val_seen`, and **1,833/1,839** `val_unseen`. Missing val_seen IDs are **11, 20, 99, 305, 365, 366, 419, 649, 736, 778**.

| Split | Group | 100×100 IoU mean / median | Presence P / R | Centroid median / P75 (m) | 10×10 IoU | Both-empty skipped |
|---|---|---:|---:|---:|---:|---:|
| train_90 | region 27–36 | 0.3339 / 0.2993 | 0.9192 / 0.9128 | 0.92 / 3.87 | 0.5565 | 57,706 |
| val_seen | region 27–36 | 0.1515 / 0.0000 | 0.7952 / 0.8065 | 3.46 / 6.53 | 0.2937 | 4,284 |
| val_unseen | region 27–36 | 0.0445 / 0.0000 | 0.6584 / 0.6840 | 4.53 / 6.99 | 0.1427 | 9,555 |
| train_90 | object 0–26 | 0.2093 / 0.1491 | 0.8785 / 0.8805 | 1.05 / 3.88 | 0.5103 | 147,782 |
| val_seen | object 0–26 | 0.0999 / 0.0000 | 0.7618 / 0.7802 | 3.49 / 6.38 | 0.2753 | 10,841 |
| val_unseen | object 0–26 | 0.0359 / 0.0000 | 0.7045 / 0.6669 | 4.41 / 6.83 | 0.1508 | 23,681 |
| train_90 | environment 0,15,17 | 0.2967 / 0.2555 | 0.9766 / 0.9657 | 1.41 / 3.91 | 0.6173 | 5,394 |
| val_seen | environment 0,15,17 | 0.1671 / 0.0876 | 0.9412 / 0.9403 | 3.55 / 5.99 | 0.4201 | 391 |
| val_unseen | environment 0,15,17 | 0.0851 / 0.0376 | 0.9571 / 0.8632 | 4.01 / 6.14 | 0.3073 | 555 |

The environment slice overlaps the object group. `val_seen` lies between train and val_unseen for both region and object layout, but its **region median IoU remains zero**. Because the metric compares cells in the native frames, possible alignment effects remain part of the measured shortfall.

| Region channel | train_90 IoU / recall | val_seen IoU / recall | val_unseen IoU / recall |
|---:|---:|---:|---:|
| 27 outdoor/semi-outdoor | 0.348 / 0.889 | 0.134 / 0.678 | 0.008 / 0.405 |
| 28 living/social space | 0.312 / 0.872 | 0.104 / 0.682 | 0.029 / 0.500 |
| 29 recreation/fitness | 0.415 / 0.888 | 0.136 / 0.592 | 0.027 / 0.559 |
| 30 utility/service | 0.258 / 0.823 | 0.122 / 0.641 | 0.007 / 0.342 |
| 31 work/study | 0.349 / 0.807 | 0.130 / 0.511 | 0.005 / 0.170 |
| 32 circulation | 0.411 / 0.980 | 0.223 / 0.964 | 0.110 / 0.941 |
| 33 private room | 0.327 / 0.894 | 0.174 / 0.811 | 0.052 / 0.726 |
| 34 bathroom/sanitary | 0.292 / 0.944 | 0.145 / 0.892 | 0.024 / 0.798 |
| 35 dining/food | 0.294 / 0.853 | 0.106 / 0.694 | 0.039 / 0.624 |
| 36 other/miscellaneous | 0.251 / 0.750 | 0.090 / 0.508 | 0.007 / 0.204 |

## T3. New val_seen probe caches

| Cache | Rule | Count | Checks |
|---|---|---:|---|
| LLM key `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__raster-cross_scene-s0`, `r2r/val_seen` | Existing donor assignment, seed 0, donors from other scenes, replace grid only; metadata from own episode. | 768 rasters, 768 new `status=complete` JSON; `cache_report` 768 available / 10 missing | `scripts/distill/verify_d4_val_seen_donor.py`: 20 actual LLM loader calls and non-grid field equality. Donor summary: same-scene rate 0, 490 distinct donors. |
| PriorGT namespace `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/llm.s2tagfree.as_priorgt.val_seen.v1` | 768 links to own LLM rasters; 10 missing use zero grid plus own GT metadata. | 778 rasters, 778 new `status=complete` JSON | `scripts/eval_gt18600/make_d4_val_seen_namespaces.py`: 20 actual PriorGT loader calls; missing IDs exactly those in T2. |
| PriorGT namespace `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.donor_cross_scene.val_seen.v1` | R6d seed-0 assignment over 778 GT episodes; donor from another scene, replace only grid; keep own GT metadata. | 778 rasters, 778 new `status=complete` JSON | Same script: 20 actual PriorGT loader calls and non-grid field equality. |

The donor builder initially linked `status/` to the source. `scripts/distill/materialize_d4_val_seen_donor_status.py` replaced **only that newly created donor-side link** with 768 independent status JSON files pointing at donor rasters; the original source status directory remains untouched. Cache-report and 20-loader checks passed again after this change. The two new PriorGT namespaces also have separate status JSON even though their loader does not require it. No original cache file was changed. **Do not compare LLM-donor and GT-donor episode assignments as if identical:** each donor pool has different size (768 vs 778), although both enforce cross-scene sampling with seed 0. The LLM-path policy counts its 10 missing cache episodes as zero-result failures; the 18600+LLM PriorGT probe instead navigates on 10 zero-grid placeholders with own GT metadata, as explicitly specified for this probe. For the 768 linked LLM episodes, the PriorGT probe loads the **entire LLM NPZ**, including predicted direction/height metadata; its comparison with GT-cross does not isolate the grid alone.

## T4. val_seen evaluation launchers — run by the user

All launchers use four ranks on GPUs **1,5,6,7** and unique ports. The two 28000 scripts are copies of `scripts/distill/eval_probe_llm_gtmeta.sh`; the three 18600 scripts are copies of `scripts/eval_gt18600/eval_own.sh`. Every launcher targets all 778 val_seen episodes, with unique output under `data/logs/checkpoints/<run_name>/` and per-episode `stats_ep_*` output. The source/namespace value is the **only model-input change** within each model pair/triple.

| Script under repo root | Run name | Map key / absolute namespace | Port |
|---|---|---|---:|
| `scripts/distill/eval_vs_28000_own.sh` | `d4_vs_28000_own` | `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree` | 29670 |
| `scripts/distill/eval_vs_28000_cross.sh` | `d4_vs_28000_cross` | Same key + `__raster-cross_scene-s0` | 29671 |
| `scripts/eval_gt18600/eval_vs_18600_gtown.sh` | `d4_vs_18600_gtown` | `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.v1` | 29672 |
| `scripts/eval_gt18600/eval_vs_18600_gtcross.sh` | `d4_vs_18600_gtcross` | `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.donor_cross_scene.val_seen.v1` | 29673 |
| `scripts/eval_gt18600/eval_vs_18600_llm.sh` | `d4_vs_18600_llm` | `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/llm.s2tagfree.as_priorgt.val_seen.v1` | 29674 |

`diff -u` against each required source showed **only these changed lines**:

| Source → target(s) | Changed original line → new value | Why |
|---|---|---|
| 28000 parent → own/cross | `--master_port 29645` → `29670`/`29671`; `--exp_name probe_llm_gtmeta_val_unseen` → `d4_vs_28000_own`/`d4_vs_28000_cross`; `llm_cache_model_key probe.llm_gtmeta.v1` → original full key / seed-0 donor key | Separate port/run and selected input. |
| 28000 parent → own/cross | `EVAL.SPLIT val_unseen` → `EVAL.SPLIT val_seen EVAL.EPISODE_COUNT -1 EVAL.SAVE_RESULTS True` | Change split and require full per-episode output (parent did not explicitly set the latter two). |
| 18600 parent → GT-own/GT-cross/LLM | `NAME=gt18600_legacy_val_unseen_own` → `d4_vs_18600_gtown`/`d4_vs_18600_gtcross`/`d4_vs_18600_llm`; `--master_port 29600` → `29672`/`29673`/`29674` | Separate port/run. |
| 18600 parent → GT-own/GT-cross/LLM | `cache_namespace gt.legacy.r1p5.direction5.v1` → the three absolute namespaces in the table; `EVAL.SPLIT val_unseen` → `val_seen` (rest of line unchanged) | Absolute map path as required; selected map and split. |

No checkpoint, architecture, map-loading flag, elevation convention or navigation setting was changed from each parent. `scripts/distill/run_d4.sh` executes the five scripts serially in table order, then requires **four rank files and exactly 778 unique episode IDs** after each run. `scripts/distill/summarize_d4_val_seen.py` is CPU-only; it reports SR/SPL/OSR/nDTW and the four requested 2×2 tables with exact McNemar p. All five user-run evaluations completed, and the CPU summarizer found 778 unique episode IDs per run.

User-run command (already completed):

```bash
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/run_d4.sh
```

To reproduce the numeric summary:

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/summarize_d4_val_seen.py
```

## T5. Completed evaluation results

Each result below covers the same 778 `val_seen` episode IDs. Values are percentages. Source: `data/logs/checkpoints/d4_vs_*/eval_results/stats_ep_*_val_seen_r*_w4.json`, four rank files per run, aggregated with `scripts/distill/summarize_d4_val_seen.py`. The 28000 scripts select `ckpt.iter28000.pth`; the 18600 scripts select `/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth`. The ten known missing LLM-cache episodes are counted as failures by the 28000 policy; the 18600+LLM probe uses the specified zero-grid placeholders instead.

| Model + input | SR | SPL | OSR | nDTW | Episodes |
|---|---:|---:|---:|---:|---:|
| 28000 + LLM own | 75.19 | 66.79 | 80.59 | 73.68 | 778 |
| 28000 + LLM cross-scene grid | 74.55 | 66.29 | 80.72 | 73.73 | 778 |
| 18600 + GT own | 79.18 | 69.95 | 83.42 | 75.46 | 778 |
| 18600 + GT cross-scene grid | 64.27 | 53.54 | 74.04 | 65.86 | 778 |
| 18600 + LLM NPZ | 66.20 | 54.68 | 77.76 | 65.29 | 778 |

Paired success comparison: `A only`/`B only` mean successful under just that arm. Exact two-sided McNemar p values use the discordant pairs; the p value tests SR, not SPL or other metrics.

| A vs B | SR difference (pp) | Both succeed | A only | B only | Both fail | Exact p |
|---|---:|---:|---:|---:|---:|---:|
| 28000 own vs cross | +0.64 | 573 | 12 | 7 | 186 | 0.359 |
| 18600 GT own vs GT cross | +14.91 | 472 | 144 | 28 | 134 | 5.21e-20 |
| 18600 LLM vs GT cross | +1.93 | 410 | 105 | 90 | 173 | 0.316 |
| 18600 LLM vs GT own | −12.98 | 492 | 23 | 124 | 139 | 6.08e-18 |

Excluding the ten LLM-missing episodes leaves 768 paired episodes. Then 28000 own/cross SR is 76.17/75.52 (p=0.359); 18600 GT own/cross is 79.17/64.45 (p=2.45e-19); 18600 LLM/GT-cross is 66.54/64.45 (p=0.281). The conclusions do not depend on the placeholders or missing-cache scoring.

## Interpretation and limits

The two proposed positive probes **both fail their stated paired tests**. The GT-trained 18600 checkpoint demonstrably uses GT episode-specific content on `val_seen` (+14.91 pp own over donor), but replacing that content with a val_seen LLM NPZ gives only +1.93 pp over GT donor, not statistically distinguishable here. The LLM-trained 28000 checkpoint likewise shows no significant advantage for its own map over a donor grid (+0.64 pp). Thus the claim “the policy reads the map whenever the LLM map is from a seen scene, so the bottleneck is entirely LLM unseen-scene generalization” is **not supported**.

This does not prove that 28000 cannot use *any* informative map: val_seen LLM region IoU is only 0.1515, with median per-channel IoU zero, and the train_90 subset was not evaluated. Nor does 18600+LLM isolate raster information: it loads the full LLM NPZ, including predicted direction/height metadata, while GT-cross retains GT metadata. The causal explanation for 28000's insensitivity therefore remains open. The actual SFT training manifest is unavailable, so instruction memorization versus scene familiarity also remains unresolved. A next decisive test would pair a demonstrably accurate map with 28000 under the same harness, or evaluate own/donor on a verified high-quality train subset; neither result is claimed here.
