# E2 own / cross-scene evaluation audit

## Verdict

The 12 checkpoint pairs finished and produced complete, episode-matched `val_unseen` results. The training and evaluation elevation convention is consistent (`y`); the coordinate-fusion map plane correctly uses world `x–z`. However, **own used three evaluation ranks while cross_scene used four**. Their observed SR/SPL differences are therefore descriptive, **not a clean map-content-only ablation**. Rerun both variants for a chosen checkpoint with identical rank count, GPU list, seed, and evaluation settings before making a causal claim.

This is a read-only audit of saved artifacts and source code; no evaluation was launched.

## What was actually evaluated

| Item | own | cross_scene | Evidence |
|---|---|---|---|
| Checkpoint | Same `dagger_coordfusion_llm/ckpt.iter<N>.pth` within each pair | Same | `data/logs/checkpoints/e2_coordfusion_val_unseen/iter21600_{own,cross_scene}/provenance.txt:4` |
| Cognitive-map key | `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree` | Same key with `__raster-cross_scene-s0` suffix | `scripts/eval_e2/eval_one.sh:13-25`; paired provenance `:5` |
| GPU / ranks | `5,6,7` / **3** | `1,5,6,7` / **4** | `iter21600_{own,cross_scene}/provenance.txt:6`; the same pattern holds for all 12 pairs |
| Other explicit evaluation settings | `LLMGridTry5Policy`, `try5`, `llm_grid`, `coordinate_fusion=True`, `elevation_axis=y`, `eval_map_source=p0`, no refiner, `back_algo=teleport`, `tryout=True`, `val_unseen` | Same | `scripts/eval_e2/eval_one.sh:77-98` |

The donor-cache manifest records `raster_donor_mode=cross_scene`, `raster_donor_seed=0`, and the own cache key as its source (`data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__raster-cross_scene-s0/r2r/val_unseen/manifest.json:2-12`). Donors are selected from **different scenes**. The generator replaces **only `grid`** and retains the target episode's other fields (`scripts/distill/make_raster_donor_cache.py:100-128,211-255`). The assignment file has 1,833 rows; all are cross-scene and all donor grids differ from the corresponding own grids. Direct checks of episodes 1–3 confirmed that the stored cross-scene grid equals its assigned donor grid while every non-grid field equals the own cache. Thus the test changes raster semantics, **not** the start-position or direction metadata. This is the intended grid-only donor control, not a full-map swap.

For every pair, the rank files contain **1,839 unique episode IDs**, and the own/cross_scene ID sets match exactly; all 24 variant directories have a `done` marker. The donor cache covers 1,833 episodes; the six LLM-cache misses are included in the 1,839-episode denominator rather than silently dropped (`scripts/eval_e2/eval_one.sh:102-120`; verified by merging the saved `stats_ep_*.json` files). Matching IDs permits paired statistics, but it does **not** remove the rank-count confound.

## Elevation and map coordinates

The E2 training config records `MODEL.elevation_axis: y` and `IL.gt_teacher_enabled: false` (`data/logs/checkpoints/dagger_coordfusion_llm/config.yaml:151-152,245`). The evaluation launcher explicitly sets `MODEL.elevation_axis y` (`scripts/eval_e2/eval_one.sh:94`). In `graph_utils.py:24-55`, `y` computes elevation from vertical `dy`, whereas `z` would use horizontal `dz`. Therefore **E2 DAgger and these evaluations both use `dy`**, with no train/eval switch to `dz`. The config's `IL.gt_teacher_elevation_axis: z` is inactive because the GT teacher is disabled. Historical GT-teacher `dz`/`dy` results refer to a different setup and should not be compared as if this E2 run changed its convention.

Separately, `map_utils.py:94-130` places the cognitive-map raster on world **`x–z`**, with 0.5 m cells. That use of `z` as a horizontal map coordinate does not imply `elevation_axis=z`. Both training and evaluation set `start_position_scale=1.0` and `start_position_meters_per_unit=1.0` (`config.yaml:222-223`; `eval_one.sh:89-90`). The cross-scene raster retains each target's own start-position metadata, so coordinate fusion is still computed in the target's coordinate frame.

## Code-version check

Training records commit `4fd623142f4c727a33e8c3a42e2352281b2572db`; all audited evaluation provenance files record HEAD `2efac8518da80e6da80d942157bf85b8c231d047` and the same training-config SHA-256, `0070c185e2484149fc7b74489d29fb57d48b7fba5fe38030bcf5970ae5aa375f` (`data/logs/checkpoints/e2_coordfusion_val_unseen/iter21600_{own,cross_scene}/provenance.txt:1-3`). `git diff 4fd6231 2efac85 --name-only` lists only four GT evaluation/cache scripts; it lists no model, trainer, or default-config files. The tracked working-tree edits recorded in provenance were also outside this evaluation path.

**Provenance limit:** `scripts/eval_e2/eval_one.sh` is currently untracked, so the Git hash alone cannot reconstruct its exact contents at evaluation time. The saved provenance does record each variant's checkpoint, cache key, GPUs, and world size; the current launcher agrees with those records. No stronger historical claim about launcher immutability is possible from these artifacts.

## Saved results (percentage points)

All rows are `val_unseen`, 1,839 episodes per variant. `Δ` is own minus cross_scene. Values were recomputed from the saved per-episode JSON files.

| Iter | own SR | cross SR | ΔSR | own SPL | cross SPL | ΔSPL |
|---:|---:|---:|---:|---:|---:|---:|
| 16000 | 63.84 | 65.42 | −1.58 | 52.17 | 53.63 | −1.46 |
| 16800 | 63.08 | 63.68 | −0.60 | 53.90 | 54.54 | −0.64 |
| 17200 | 64.55 | 63.40 | +1.14 | 52.79 | 52.24 | +0.55 |
| 17600 | 64.27 | 64.71 | −0.44 | 54.62 | 54.59 | +0.02 |
| 18000 | 63.95 | 63.68 | +0.27 | 54.45 | 54.23 | +0.23 |
| 19200 | 63.46 | 63.68 | −0.22 | 53.13 | 52.92 | +0.21 |
| 20000 | 62.75 | 64.00 | −1.25 | 53.67 | 53.99 | −0.33 |
| 21200 | 64.22 | 63.78 | +0.44 | 53.31 | 53.06 | +0.25 |
| 21600 | 64.76 | 64.17 | +0.60 | 53.91 | 53.70 | +0.21 |
| 22000 | 63.73 | 64.44 | −0.71 | 53.07 | 53.86 | −0.79 |
| 22400 | 63.02 | 63.68 | −0.65 | 53.60 | 53.98 | −0.38 |
| 22800 | 63.78 | 63.68 | +0.11 | 54.00 | 54.03 | −0.02 |

There is no consistent positive own-minus-cross_scene trend in this range. This is suggestive, **not decisive**, because world size differs. The minimum clean follow-up is to re-evaluate one selected checkpoint with **both variants on the same three GPUs and `world_size=3`**, using distinct new output directories so the existing results remain intact. No model retraining is needed for that check.
