# E2 coordinate-fusion evaluation setup (no GPU run by CLI)

## T0. Version and saved configuration

- HEAD: `2efac8518da80e6da80d942157bf85b8c231d047` (2026-10-06). The previous two commits are `c681822` and `b7eacb1`; `git log --grep=coordinate_fusion` finds the coordinate-fusion launcher commit `e6cffa8` and implementation commit `8c9f8fc`. `vlnce_baselines/config/default.py:247` has `coordinate_fusion=False` by default.
- E2 `launch_info.txt:2,9,16` records training code commit `4fd623142f4c727a33e8c3a42e2352281b2572db`. `git diff 4fd6231 HEAD --name-only` lists only four GT evaluation/cache scripts; **no model, trainer, or config source changed** between that commit and HEAD. The current worktree has unrelated tracked edits in `reports/refiner/S4_progress.md` and `scripts/distill/run_grpo_from_dagger.sh`, plus untracked files; `git diff --stat` is not empty, but neither tracked edit is in the E2 evaluation path.
- Training config SHA-256: `0070c185e2484149fc7b74489d29fb57d48b7fba5fe38030bcf5970ae5aa375f` (`data/logs/checkpoints/dagger_coordfusion_llm/config.yaml`). Saved values: `LLMGridTry5Policy`; `architecture=try5`, `source=llm_grid`, `coordinate_fusion=True`, key `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree`, `load_pretrained_map_modules=False`, `start_position_scale=1.0`, `start_position_meters_per_unit=1.0`, `refiner_ckpt=''`, `eval_map_source=refiner` (inactive because refiner checkpoint is empty), `MODEL.elevation_axis=y`; `IL.back_algo=teleport`, `tryout=True`, `lr=1e-5`, `iters=30000`; `MODEL.pretrained_path=/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt`. This matches `scripts/distill/run_dagger_coord_fusion.sh:43-82,132-164` plus HEAD defaults. Evaluation explicitly uses `eval_map_source=p0`; this also leaves the refiner inactive.
- `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv` failed in this container because the NVIDIA driver is inaccessible. GPU 1 headroom could not be measured; the launchers conservatively default to **5,6,7**, never use 0/2/3, and refuse GPU 1 unless a runtime `nvidia-smi` check shows at least 12 GiB free. Both three-rank and four-rank result files are supported.
- Each evaluation's `provenance.txt` records the then-current HEAD, training commit, training-config SHA-256, checkpoint/key/GPU/port, `git diff --stat`, and `git status --short`.

## T1-A. Map/fusion initialization

`vlnce_baselines/models/etp_prior_gt/vlnbert_init.py:27-49,99-108` emits “no pretrained tensors loaded” when the remapped fusion dictionary is empty. `policy.py:173-184` loads `map_encoder.*` only if `load_pretrained_map_modules` is true. E2's launcher explicitly sets **False** (`scripts/distill/run_dagger_coord_fusion.sh:73-81`), as does its saved config. The message is therefore caused by that choice, **not a coordinate-fusion parameter-shape mismatch**; `map_fusion.py:58-76` adds only non-persistent coordinate buffers.

CPU comparison of 460000 against E2's earliest retained `ckpt.iter1200.pth` (matching 37 `map_encoder.*` and six fusion tensor keys): all shapes matched. Fusion tensor cosines were about **−0.012 to +0.037**, with the main in-projection weight at **−0.00066**; thus they are unrelated to the pretrained fusion weights. Of 36 nonzero-norm map-encoder tensors, only eight had cosine >0.99; these were the fixed/initialization-aligned category projection and normalization weights, not evidence of transfer. For example, pretrained `spatial_tokenizer.weight` norm was **0**, E2 1200 norm **15.98**; `metadata_encoder.0.weight` cosine was **−0.0011**. **E2's map encoder and fusion trained from DAgger initialization**, while common navigation weights came from 460000.

The historical 28000 control training log predates the new explicit message; the current `llm_grid_try5_iter28000_raster_control/eval.log:53` prints the same message during model construction. The 65.09 raster control and 64.33 `llm_try5_control_val_unseen` both use the historical 28000 checkpoint. CPU comparison of that checkpoint's six fusion tensors against 460000 gave cosines **−0.026 to +0.119** (main in-projection **−0.00051**), again unrelated. The older DAgger code also did not transfer those tensors (see `docs/daily/2026-09-24.md:32-44`). **There is no E2-versus-control confound from pretrained map/fusion loading.**

## T1-B. Start positions outside the raster

`ss_trainer_ETP_PriorGT.py:1422-1450` computes map origin and logs out-of-bounds starts; `map_utils.py:94-133` defines the transform. It does **not clamp coordinates or zero the locality bias**. `map_fusion.py:90-120` computes a finite distance bias for out-of-range coordinates, with one-third of heads content-only; spatial heads therefore favor nearby edge tokens. The training log's “1100 such episodes” counts repeated episode encounters, not 1100 unique IDs.

CPU scan with the same `map_origin_xz`/`world_to_map_cells` transform and `meters_per_unit=1.0`:

| Source/split | Cached / total | OOB | OOB among cached | start row min / median / max | start col min / median / max |
|---|---:|---:|---:|---|---|
| LLM train | 10707 / 10819 | 217 | 2.03% | 0.25 / 23.98 / 158.28 | 1.15 / 23.69 / 126.14 |
| GT legacy train | 10786 / 10819 | 219 | 2.03% | 0.25 / 23.94 / 158.28 | 1.15 / 23.69 / 126.14 |
| LLM val_unseen | 1833 / 1839 | 0 | 0% | 2.04 / 20.30 / 79.05 | 1.86 / 19.18 / 83.67 |
| GT legacy val_unseen | 1839 / 1839 | 0 | 0% | 2.04 / 20.30 / 79.05 | 1.86 / 19.18 / 83.67 |

The near-identical GT and LLM train rates refute an **LLM-only** position error. They are consistent with the fixed 100×100, 0.5-m cell raster excluding starts on some large training levels; this scan alone does not prove the transform is correct. The requested `scripts/eval_e2/oob_val_unseen.txt` contains no IDs, so the val_unseen OOB stratum is **empty** and its SR/delta must be reported N/A. Six LLM val_unseen episodes lack cache (308, 349, 455, 804, 1318, 1331), distinct from OOB.

## T2. Checkpoints and schedule

At setup, retained checkpoint iterations were **1200; 8000, 8400, 8800, 9200, 9600, 10000, 10400, 10800, 11200, 11600, 12000, 12400, 12800**. `CHECKPOINT_INTERVAL=400` in the launcher (`run_dagger_coord_fusion.sh:143`); `ss_trainer_ETP_PriorGT.py:919-923` saves at divisible iterations or the final iteration. Requested 4000 is already past and absent; **1200 is not silently substituted for 4000**. The watcher targets 8000, 12000, 16000, 20000, 24000, 28000, 30000, adding 4000 only if its file exists when the watcher starts. All targets are divisible by 400, so no nearest-interval substitution is needed.

## T3. User-run evaluation scripts

```bash
E2_GPUS=5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_e2/eval_one.sh 12000 own
E2_GPUS=5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_e2/eval_one.sh 12000 cross_scene
mkdir -p /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/e2_coordfusion_val_unseen
nohup bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_e2/watch_and_eval.sh > /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/e2_coordfusion_val_unseen/watch.log 2>&1 &
```

Use either the two single-checkpoint examples **or** the watcher; do not run them concurrently. The watcher polls every ten minutes, requires unchanged checkpoint size for two minutes, evaluates own then cross_scene sequentially, writes a done marker after both pass, and skips completed checkpoints. `eval_one.sh` verifies 1839 unique episode results across **w3 or w4** rank files before marking a variant done. Ports 29610/29611 avoid training 29512 and the 29600–29603 GT evaluations. It never selects GPU 0/2/3; `E2_GPUS=4,5,6,7` is supported when available. The output layout is `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/e2_coordfusion_val_unseen/iter<iter>_<variant>/` with `eval.log`, `provenance.txt`, `eval_results/stats_ep_*`, and a variant `done` file.

The cross_scene variant reuses the **existing** seed-0 key `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree__raster-cross_scene-s0`; its 1833 val_unseen rasters already exist. `scripts/distill/make_raster_donor_cache.py:100-128,211-255` describes the same deterministic cross-scene, grid-only donor rule; no cache regeneration occurs.

After evaluations, CPU-only summary:

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_e2/summarize.py
```

This prints per-iter metrics, paired own/cross_scene and own/64.33-control McNemar tables (including the six-cache-miss-excluded version), and in-raster/OOB strata. For reference, the existing historical 28000 raster-control pair has own **65.09** vs seed-0 cross_scene **66.01**, paired own-only **27** vs cross-only **44**, exact McNemar **p=0.0568**. **Caution:** that pair's launcher defaults to `IL.back_algo=control`; the 64.33 `llm_try5_control_val_unseen` launcher explicitly uses `teleport`. E2 follows the user's requested teleport setting, so the 64.33 run is the closer action-mode comparator; the 65.09 pair is used only for its own-vs-donor sensitivity reference. GT legacy 18600 reference: own 68.41, cross_scene 55.68, nomap 55.68; GT fine teacher 73.84. These are reference rows, not E2 results.

## Verification / pending results

Only shell syntax, Python AST parsing, config-option parsing, cache/weight CPU reads, and CPU map-frame scanning were performed. The container's `nvidia-smi` cannot see the driver, so no local GPU availability claim is made. **No GPU evaluation was started.** The results section remains pending until the user runs the watcher/evaluations and then invokes `summarize.py`.
