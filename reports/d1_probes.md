# D1': three zero-training probes (setup; evaluations pending)

## Status

Six new `probe.*` caches are built and CPU-checked. Six evaluation launchers and a paired-summary script are ready. The first GT-metadata GPU evaluation ran on 2026-10-08 but was **invalid**: every episode was marked cache-missing because generated probe rasters lacked `status=complete` JSON files. The generator now writes these files; existing GT-metadata, metadata-only, and full probes have been repaired. The full and GT-metadata launchers use GPUs 1,5,6,7; their ports are 29641 and 29645 respectively.

## P1. Which LLM-map input does Try5 use?

Source policy: the historical LLM Try5 `ckpt.iter28000.pth`, with existing own-map HEAD-harness control **SR 64.33 / SPL 54.16**. The independent CPU generator [`scripts/distill/make_probe_metadata_donor.py`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/make_probe_metadata_donor.py) reuses the **exact** 1,833-episode, seed-0, different-scene donor assignment from `data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__raster-cross_scene-s0/r2r/val_unseen/donor_assignment.csv` (`scripts/distill/make_raster_donor_cache.py:99-128,211-255`). The original donor generator was not modified.

| New LLM model key | Donor fields | Own fields retained | Files |
|---|---|---|---:|
| `probe.llm_metadata_only_s0.v1` | `direction_vectors`, `start_direction_vector` | `grid`, `start_position`, `range_y` | 1833 |
| `probe.llm_full_s0.v1` | All five NPZ fields: `grid`, `range_y`, `direction_vectors`, `start_direction_vector`, `start_position` | None | 1833 |

Both keys live under `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/llm_navigation/`. The same six LLM-cache misses remain missing and are counted as zero by the LLM evaluation trainer, exactly as in own. The cache writer checked all 1,833 output files and verified every field of the first 20 against the assigned own/donor sources; both new model keys also passed 20 loader checks. The model's `direction5` path consumes `grid`, the two direction fields, and `start_position`; `range_y` is present for a faithful full-file swap but has no policy effect here (`map_utils.py:133-148`; `map_encoder.py:327-351`). A full swap makes donor position/direction metadata inconsistent with the recipient scene by design; interpret it separately from metadata-only.

Launchers: [`eval_probe_llm_metadata_only.sh`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_probe_llm_metadata_only.sh) (29640, one rank on GPU 0) and [`eval_probe_llm_full.sh`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_probe_llm_full.sh) (29641, four ranks on GPUs 1,5,6,7). The full launcher matches the GT-metadata launcher's evaluation settings apart from model key, run name, and port; it explicitly disables redundant pretraining-map loading before the DAgger checkpoint is loaded.

### P1 GT metadata oracle (added 2026-10-08)

`probe.llm_gtmeta.v1` preserves the episode's own LLM `grid`, `start_position`, and `range_y`; only `direction_vectors` and `start_direction_vector` come from `gt.legacy.r1p5.direction5.v1` for the **same episode**. The GT and LLM fields have the same schema and slot order, but **the coordinate frame is global display-frame `[right,up]=[-dz,-dx]`, not start-heading-relative**: see `prior/directions.py:24-29`, `prior/cognitive_map_generation.py:95-122`, and `vlnce_baselines/models/etp_llm/prompts/llm_grid_system.md:7-10`. GT vectors follow the actual path, while LLM vectors predict it. Hence this is an **oracle route-direction injection**, not a pure coordinate-alignment correction. The cache writer copies `start_direction_vector` from the source item rather than the LLM response (`llm_grid_navigation_cache.py:278-286,426-437`); all 1,833 paired start vectors are bitwise identical, so replacing that field is a no-op.

First 20 paired episodes below. Each direction list contains five ordered `(right,up)` vectors; `0` denotes `(0,0)`. Values are rounded to three decimals. The last column is the **shared** LLM/GT start vector.

| Episode | LLM directions | GT directions | Start vector |
|---:|---|---|---|
| 1 | (+.707,+.707);(-.259,+.966);0;0;0 | (0,+1);(-.866,+.500);(-.966,-.259);0;0 | (+.500,+.866) |
| 2 | (0,+1);(-.707,+.707);(-1,0);(-.707,-.707);0 | (0,+1);(-.866,+.500);(-.966,-.259);0;0 | (+.500,+.866) |
| 3 | (0,+1);(-.966,+.259);0;0;0 | (0,+1);(-.866,+.500);(-.966,-.259);0;0 | (+.500,+.866) |
| 4 | (+.966,-.259);(+.500,-.866);(-.259,-.966);(-.866,-.500);0 | (+1,0);(+.707,-.707);(0,-1);(+.966,-.259);(+.866,+.500) | (+.866,+.500) |
| 5 | (+.966,+.259);(+.500,+.866);(+.966,+.259);(+.259,+.966);0 | (+1,0);(+.707,-.707);(0,-1);(+.966,-.259);(+.866,+.500) | (+.866,+.500) |
| 6 | (+.966,-.259);(+.259,-.966);(-.500,-.866);(-.966,-.259);(-.500,-.866) | (+1,0);(+.707,-.707);(0,-1);(+.966,-.259);(+.866,+.500) | (+.866,+.500) |
| 7 | (+1,0);0;0;0;0 | (+1,0);0;0;0;0 | (+1,0) |
| 8 | (+.707,-.707);0;0;0;0 | (+1,0);0;0;0;0 | (+1,0) |
| 9 | (-.259,+.966);(-.966,+.259);(-.707,-.707);0;0 | (+1,0);0;0;0;0 | (+1,0) |
| 10 | (+.707,-.707);(-.500,-.866);0;0;0 | (+1,0);0;0;0;0 | (+1,0) |
| 11 | (+.966,-.259);0;0;0;0 | (+1,0);0;0;0;0 | (+1,0) |
| 12 | (+.500,-.866);(+.966,-.259);(+.259,-.966);(-.259,-.966);0 | (+1,0);0;0;0;0 | (+1,0) |
| 13 | (0,+1);(+.707,+.707);(+1,0);(+.707,-.707);0 | (0,+1);(+.707,+.707);(0,+1);0;0 | (-1,0) |
| 14 | (0,+1);(+.866,+.500);(+.966,-.259);0;0 | (0,+1);(+.707,+.707);(0,+1);0;0 | (-1,0) |
| 15 | (-.500,-.866);(+.259,-.966);0;0;0 | (0,+1);(+.707,+.707);(0,+1);0;0 | (-1,0) |
| 16 | (-.707,+.707);(+.966,+.259);0;0;0 | (+.361,+.933);0;0;0;0 | (+.500,-.866) |
| 17 | (-.259,-.966);(-.866,-.500);(-.966,+.259);(0,+1);(-.707,+.707) | (+.361,+.933);0;0;0;0 | (+.500,-.866) |
| 18 | (-.259,-.966);(-.966,-.259);(-.866,+.500);(-.966,-.259);(-.259,-.966) | (+.361,+.933);0;0;0;0 | (+.500,-.866) |
| 19 | (-.259,+.966);(-.866,+.500);(-.259,+.966);0;0 | (-.259,-.966);(-.866,-.500);(-.259,-.966);0;0 | (-1,0) |
| 20 | (0,+1);(-.866,+.500);0;0;0 | (-.259,-.966);(-.866,-.500);(-.259,-.966);0;0 | (-1,0) |

Across all 1,833 paired episodes, angle is computed slotwise as `acos(clamp(dot(LLM,GT),-1,1))` only when **both** vectors are nonzero; padded slots are excluded. The first-to-fifth slot sample counts are 1,831 / 1,422 / 739 / 221 / 38, with median angles **75° / 75° / 75° / 90° / 75°**. Pooled over 4,251 valid slots, the histogram for `[0,15), [15,30), [30,45), [45,60), [60,90), [90,120), [120,150), [150,180]` degrees is **523 / 507 / 434 / 332 / 650 / 656 / 505 / 644**. All 1,833 start-vector angles are **0°**. This is strong evidence of different route content rather than a constant coordinate rotation.

The CPU generator is [`make_probe_metadata_donor.py`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/make_probe_metadata_donor.py) with `--mode gt_metadata`; the launcher [`eval_probe_llm_gtmeta.sh`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_probe_llm_gtmeta.sh) uses four ranks on physical GPUs 1,5,6,7, port 29645. Its map/checkpoint/evaluation settings match the own launcher, while the GPU/world-size settings differ as requested. The paired summary compares this variant against LLM own 28000; a four-rank own rerun would remove rank count as a comparison variable.
The new cache contains **1,833 files**; all five fields in the first 20 were checked against their intended LLM/GT source, and all 20 passed the actual `direction5_cognitive_map_file_to_tensors` loader check (`grid=(37,100,100)`, directions `(5,2)`). The six original LLM-cache misses remain absent, matching the own control.

The loader also requires `status/<scene>/<episode>.json` with `status=complete` (`navigation.py:333-365`). The first GT-metadata run had no status files and reported `available=0`, so its all-zero metrics must be discarded. After repair, the **actual** `llm_navigation_cache_report` gives `available=1833, missing=6` for each of the three P1 namespaces; missing IDs are 308, 349, 455, 804, 1318, 1331. The generator has a `--status_only` repair mode that asserts each raster exists and writes the matching status; it does not rewrite raster content. The evaluation can now be rerun with the same launcher; the trainer's existing-output skip applies only when `EVAL.CKPT_PATH_DIR` is a directory, whereas this launcher passes one checkpoint file (`ss_trainer_ETP_PriorGT.py:1014-1023`), so the invalid stats files will be overwritten. The summary script now rejects a 100%-cache-missing result instead of reporting it as model performance.

On the next attempt, initialization stopped before loading the DAgger checkpoint: HEAD defaults `load_pretrained_map_modules=True`, so `policy.py:180-243` first tried the 460k pretraining map encoder and rejected its all-zero spatial tokenizer. This is not evidence of a dead **DAgger** map encoder: CPU inspection of `ckpt.iter28000.pth` found spatial-tokenizer weight norm **19.06** and fusion residual-projection weight norm **15.31**. The GT-metadata launcher now explicitly sets `MODEL.MAP_ENCODER.load_pretrained_map_modules False`, like the GT-own evaluator. The base pretraining path remains set, and `_initialize_policy` subsequently loads the DAgger `state_dict` (`ss_trainer_ETP_PriorGT.py:428-469`). No model source was changed. A strict comparison to the historical one-rank own control should rerun own under the same HEAD setting and four-rank harness.

The completed GT-metadata rerun evaluated all **1,839** val_unseen episodes, including the expected six missing caches scored zero: **SR 64.27, SPL 54.11, OSR 69.00, nDTW 66.16**. Historical own control: **64.33 / 54.16 / 69.11 / 66.19**. The per-episode pairing has 1,181 both successful, **1 GT-metadata-only success**, **2 own-only successes**, and 655 both failed; exact McNemar **p=1**. Thus oracle GT route directions produced no measurable benefit for this checkpoint in this probe. This does **not** prove the map encoder ignores all metadata; it isolates these direction fields under this checkpoint. The control was evaluated with one rank and the probe with four; only three success outcomes differ, but a matched four-rank own rerun remains the cleanest final control.

## P2. Does the GT-trained model expect soft positive values?

The CPU generator [`scripts/eval_gt18600/make_probe_soft06.py`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/make_probe_soft06.py) created `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/probe.llm_soft06.v1/raster/` with 1,839 files. It preserves every D0 LLM-map field and changes **only grid entries equal to 1.0 into float32 0.6**; zeros and the six D0 placeholder grids stay zero. Twenty samples passed loader and fieldwise reconstruction checks; their grid values are `{0, 0.6000000238}`. This tests value-scale sensitivity, not LLM semantic accuracy.

Launcher: [`eval_soft06.sh`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_soft06.sh), port 29642. It differs from the four-rank 18600 GT-own launcher in exactly three lines: run name, namespace, and port. Controls: D0 18600+LLM **52.47 SR** and 18600+GT-cross_scene **55.68 SR**. Use paired episodes, including the same six placeholders as D0.

## P3. Final-GT-trajectory observation-mask oracle

The S1/S2 corpus has **1,821 of 1,839** `val_unseen` teacher trajectories under `data/refiner/val_unseen/<scene>/<episode>_teacher.npz`. The final accumulated 100×100 observation mask is `np.unpackbits(steps[-1]['observed'], count=10000).reshape(100,100)` (`scripts/refiner/collect.py:145-152`; `vlnce_baselines/models/refiner/dataset.py:22-43`). The final mask covers **636.40 cells on average** (6.36% of the raster); LLM and GT start-position metadata match across all 1,821 covered episodes, supporting direct cell alignment.

The 18 missing masks are episodes **308, 349, 455, 463, 464, 564, 605, 634, 635, 636, 804, 987, 1021, 1022, 1087, 1253, 1318, 1331**: six LLM misses plus 12 zero-P0 episodes excluded during collection. They are treated as **all-unseen**, not silently dropped. The CPU generator [`scripts/eval_gt18600/make_probe_gtseen.py`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/make_probe_gtseen.py) created 1,839 files in each new namespace:

| Namespace | Observed cells | Unseen cells | Metadata |
|---|---|---|---|
| `probe.gtseen_llmunseen.v1` | Legacy GT grid | D0 LLM grid | D0 own-episode metadata |
| `probe.gtseen_zero_unseen.v1` | Legacy GT grid | All-zero grid | D0 own-episode metadata |

Both variants passed 20 sample loader and cellwise formula checks. The zero-unseen variant still retains metadata, so it is **not** the same intervention as disabling the map encoder (`nomap`). More importantly, the **final teacher mask contains future observations along the entire GT trajectory and is applied from the start**. P3 is a **non-causal oracle upper-bound probe**, not a deployable online refiner evaluation.

Launchers: [`eval_gtseen_llmunseen.sh`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_gtseen_llmunseen.sh) (29643) and [`eval_gtseen_zero_unseen.sh`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_gtseen_zero_unseen.sh) (29644). Each differs from the four-rank GT-own launcher in exactly three lines: run name, namespace, and port. Controls: GT own **68.41 SR**, nomap **55.68 SR**.

## Results table and paired analysis

Existing rows below are recomputed from saved per-episode files. The GT-metadata and full probes are complete; other probe rows remain **pending**. Full-swap interpretation and control limitations are detailed in [`d1_full_probe_result.md`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/d1_full_probe_result.md). After manual evaluations, run [`summarize_d1_probes.py`](/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/summarize_d1_probes.py) to refresh the table and paired 2×2 counts with exact McNemar p-values.

| Variant | SR | SPL | OSR | nDTW | Control | exact McNemar p |
|---|---:|---:|---:|---:|---|---:|
| LLM own 28000 | 64.33 | 54.16 | 69.11 | 66.19 | — | — |
| P1 metadata_only | pending | pending | pending | pending | LLM own | pending |
| P1 full | 65.42 | 54.64 | 69.93 | 66.65 | LLM own | 0.0169 |
| P1 GT metadata | 64.27 | 54.11 | 69.00 | 66.16 | LLM own | 1 |
| D0 18600+LLM | 52.47 | 41.13 | 67.65 | 57.22 | — | — |
| P2 soft06 | pending | pending | pending | pending | D0; GT cross | pending |
| GT cross 18600 | 55.68 | 44.44 | 65.09 | 61.74 | — | — |
| GT own 18600 | 68.41 | 57.88 | 73.30 | 68.55 | — | — |
| GT nomap 18600 | 55.68 | 46.86 | 59.27 | 65.75 | — | — |
| P3 GT-seen/LLM-unseen | pending | pending | pending | pending | GT own; nomap | pending |
| P3 GT-seen/zero-unseen | pending | pending | pending | pending | GT own; nomap | pending |

CPU summary command after runs:

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/summarize_d1_probes.py
```

## Manual evaluation commands (wait for free GPUs)

Run the remaining probes sequentially once GPU 0 has enough free memory for metadata-only and GPUs 1,5,6,7 have enough for full/P2/P3; GPU availability must be rechecked:

```bash
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/eval_probe_llm_metadata_only.sh
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_soft06.sh
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_gtseen_llmunseen.sh
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_gtseen_zero_unseen.sh
```

P1 metadata-only remains one-rank like its own baseline; P1 full, GT-metadata, and P2/P3 use four ranks. Full and GT-metadata have completed. The CLI did not start a GPU evaluation.
