# R6d: GT-discrete 18600 evaluation on the HEAD harness

Status: launchers and CPU summarizer are ready; the donor namespace was generated and checked. **No GPU evaluation was started.** Any subsequent metrics must be labeled **HEAD harness**, not results from training commit `8e73797`.

## T0: Defaults and coordinate units

`MODEL.MAP_ENCODER.eval_map_source` is read by the PriorGT trainer's `_refiner_enabled()` (`vlnce_baselines/ss_trainer_ETP_PriorGT.py:118-120`) and, when the refiner is enabled, by `_initialize_refiner_state` and `_update_refined_cognitive_maps` (`:1242,1301`). The ordinary map loader itself reads `cache_namespace` (`:1476-1490`). The launchers explicitly set `eval_map_source p0`, `refiner_ckpt ''`; consequently no online refiner/sensors are activated and the cached P0 is used.

`git log -S'start_position_scale' -- vlnce_baselines/config/default.py` finds `ec5fde0`, **2026-09-24**. `git log -S'start_position_meters_per_unit' -- vlnce_baselines/config/default.py` finds `8c9f8fc`, **2026-09-29**. Neither introduction date is later than 2026-09-29. HEAD defaults are **1.0** for both (`default.py:242,250`). The former multiplies the legacy `start_position` at `ss_trainer_ETP_PriorGT.py:1370-1378`, so its numerical value is unchanged. The latter is used only by coordinate fusion's map-origin path (`:1403-1406,1411-1432`); coordinate fusion is explicitly disabled here, so it has no effect. This is a date-based check, not an audit of the unavailable `8e73797` tree.

The checkpoint-to-HEAD CPU structural check from R6c remains valid: 980 policy parameters, **0 missing, 0 unexpected, 0 shape mismatch** after the saved `net.module.` to current `net.` prefix mapping. HEAD `MODEL.elevation_axis` defaults to `y` (`default.py:159`) and is explicitly set to `y`; training on `y` remains an inference from chronology, not independently confirmed from the unavailable historical commit.

## T1: own and nomap launchers

Run manually and sequentially when GPUs 4–7 are free:

```bash
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_own.sh
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_nomap.sh
```

The full commands are in `scripts/eval_gt18600/eval_own.sh` and `eval_nomap.sh`. Both use absolute repo, Python/torchrun, pretraining and DAgger paths; `CUDA_VISIBLE_DEVICES=4,5,6,7`, four ranks × four environments, ports **29600/29601**, `PriorGTTry5Policy` / `try5` / `prior_gt` / `gt.legacy.r1p5.direction5.v1`, no refiner or coordinate fusion, `IL.back_algo teleport`, `IL.tryout True`, full `val_unseen`, seed 100. Nomap differs in `MODEL.MAP_ENCODER.enabled False`. Output directories are respectively:

```text
/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt18600_legacy_val_unseen_own/
/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt18600_legacy_val_unseen_nomap/
```

`ss_trainer_ETP_PriorGT.py:1551-1578` writes `eval_results/stats_ep_ckpt_18600_val_unseen_r<rank>_w4.json` and aggregate `stats_ckpt_18600_val_unseen.json` when `EVAL.SAVE_RESULTS=True`. The launchers also tee stdout/stderr into each output's `eval.log`. `load_pretrained_map_modules=False` avoids transiently loading 440000 map modules at construction; the complete DAgger `state_dict` is subsequently loaded, and R6c's 0/0/0 structural check shows this does not omit trained tensors.

## T2: PriorGT donor namespace

E2's rule (`scripts/distill/make_raster_donor_cache.py:86-128,177-189,211-255`) was reused by importing `assign_donors` directly: val_unseen source-complete episodes, sorted by integer episode ID; `np.random.default_rng(0)`; for each episode choose uniformly with replacement among episodes from a **different scene**. GT has all 1839 source maps, whereas the LLM donor pool excludes incomplete LLM caches. The assignment *rule* is identical; the episode population is necessarily different.

| Field/rule | E2 LLM donor | R6d GT donor |
|---|---|---|
| split/seed/donor pool | val_unseen / 0 / source-complete episodes in different scenes | same; 1839 complete GT episodes |
| replaced | `grid` only (`:237-253`) | `grid` only |
| retained from target | `range_y`, `direction_vectors`, `start_direction_vector`, `start_position` and any other NPZ fields | same |
| storage | new LLM model key | new GT namespace under `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/` |

CPU generation was run once with:

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/make_gt_donor_namespace.py
```

It created **only** `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.donor_cross_scene.v1/`, leaving the original namespace untouched. Self-check: **1839** NPZ files; `cached_cognitive_map_to_tensors(..., metadata_schema='direction5')` loaded the first; 20/20 sampled files had exact donor `grid` and exact own `range_y/direction_vectors/start_direction_vector/start_position`. All sampled donor scenes differ:

| target→donor | target→donor | target→donor | target→donor |
|---|---|---|---|
| 1→1568 | 2→1176 | 3→943 | 4→497 |
| 5→569 | 6→77 | 7→140 | 8→35 |
| 9→332 | 10→1510 | 11→1202 | 12→1683 |
| 13→929 | 14→1123 | 15→1791 | 16→1347 |
| 17→1172 | 18→1019 | 19→1028 | 20→1722 |

Cross-scene launcher (run only after own/nomap, also manually):

```bash
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_cross_scene.sh
```

It uses port **29602** and output `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt18600_legacy_val_unseen_cross_scene/`; otherwise the full command in `eval_cross_scene.sh` matches own, with only `MODEL.MAP_ENCODER.cache_namespace` changed to the **absolute** new namespace path. `map_utils.py:166-178` composes the path with `pathlib`; an absolute namespace overrides the default cache root, so no repo-local symlink or existing namespace is modified.

## T3: summarize after all three runs

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/summarize.py
```

The CPU script requires each run's `stats_ep_*` files and the existing LLM HEAD control file. It prints episode count, GT-cache misses, SR/SPL/NE/OSR/nDTW for all variants, a reference table (LLM HEAD 64.33/54.16, historical LLM release 65.42/55.25, different-checkpoint fine teacher 73.84, teacher nomap 57.97), exact McNemar 2×2 tables for own vs LLM (all episodes and excluding LLM's six missing-cache episodes) and own vs cross_scene. It is intentionally **not run yet**. No result section should be filled before user-run evaluations complete.

## Verification and limits

`bash -n` passed for all three launchers; Python AST parsing passed for the donor and summary scripts. `ruff`/`ty` were not available on PATH or in `etpr1-py38`. No GPU process was started. This is a HEAD-harness comparison; original-commit result equivalence is unverified.
