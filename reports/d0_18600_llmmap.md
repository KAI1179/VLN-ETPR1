# D0: 18600 policy on LLM maps, plus paired distillation launchers

## Status and decisive caveats

The LLM-as-PriorGT namespace is ready and the **user completed the zero-training evaluation** on all 1,839 `val_unseen` episodes. The distillation and CE-only launchers remain **unrun**. This CLI's follow-up analysis used CPU only. GPU 1,5,6,7 were selected from `nvidia-smi` when the scripts were prepared (each then had 24,077 MiB free); availability must be rechecked before any later run. No model source file, Git state, existing cache namespace, or `dagger_coordfusion_llm` file was modified.

Two details prevent a misleading comparison:

1. PriorGT throws `FileNotFoundError` on a missing map. Its six zero-grid placeholders **will be navigated**, whereas the LLM control counts those six missing caches as zero without navigation. The summary therefore reports both all 1,839 episodes and the common 1,833 with those six excluded. Full-split scores are not perfectly identical in missing-cache treatment.
2. The 18600 DAgger checkpoint is a wrapped `state_dict` checkpoint, so setting `MODEL.pretrained_path` directly to it would not load it correctly. The launchers use the 460000 raw pretraining file only to construct the model, then load **all 18600 student weights** with `IL.load_from_ckpt=True`, `IL.is_requeue=False`, `IL.ckpt_to_load=<18600>`. The optimizer and iteration start fresh. This is the required loader correction, not a change of intended initial weights.

## T1. LLM map namespace

Destination: `/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/llm.s2tagfree.as_priorgt.v1/raster/`. `scripts/eval_gt18600/make_llmmap_namespace.py` created **1,833 absolute symlinks** to the LLM navigation rasters plus **six physical zero-grid NPZ placeholders** (episodes 308, 349, 455, 804, 1318, 1331). The placeholders copy each episode's own legacy-GT metadata, not its grid. The script refuses to overwrite an existing destination. No original cache files were changed.

PriorGT resolves `<namespace>/raster/<scene>/R2R_val_unseen_<episode_id>.npz` (`vlnce_baselines/models/etp_prior_gt/map_utils.py:166-179`; `vlnce_baselines/ss_trainer_ETP_PriorGT.py:1474-1491`). LLM and GT existing file names/scene directories match for all 1,833 available episodes; an absolute namespace path is accepted by `Path / absolute`. Missing files raise at `map_utils.py:250-260`, and PriorGT evaluation does not prefill missing episode scores (`ss_trainer_ETP_PriorGT.py:1493-1497`). In contrast, `vlnce_baselines/ss_trainer_ETP_LLM.py:91-124` handles its six misses as zero-score episodes.

The CPU self-check loaded episodes 1–20 through `cached_cognitive_map_to_tensors` and compared them with the corresponding legacy-GT files. Both have identical keys, shapes and dtypes:

| NPZ key | Shape | dtype |
|---|---|---|
| `grid` | `(37, 100, 100)` | `float32` |
| `range_y` | `(2,)` | `object` |
| `direction_vectors` | `(5, 2)` | `float32` |
| `start_direction_vector` | `(2,)` | `float32` |
| `start_position` | `(2,)` | `float32` |

The sampled LLM grid values are exactly **{0, 1}**; the GT legacy grid permits **{0, 0.6, 1}**. The loader returns matching tensor keys/shapes/dtypes, including `grid: float32[37,100,100]` and `map_trajectory_metadata: float32[5,2]`. Script output: `files=1839 links=1833 placeholders=6 sampled_grid_values=[0.0, 1.0]`.

## T2. Zero-training evaluation

`scripts/eval_gt18600/eval_llmmap.sh` is an exact copy of `eval_own.sh` except **three lines**: run name/output directory, port 29630, and the absolute LLM-as-PriorGT namespace. A `diff -u` check confirmed only those three substitutions. It retains `PriorGTTry5Policy`, `try5/prior_gt`, no refiner, `eval_map_source=p0`, `coordinate_fusion=False`, `elevation_axis=y`, `back_algo=teleport`, `tryout=True`, seed 100, and 4 ranks × 4 environments.

Manual command (only after checking GPU 1,5,6,7 are still free):

```bash
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/eval_llmmap.sh
```

The completed results are under `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt18600_llmmap_val_unseen/`. The CPU summary command is:

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt18600/summarize.py
```

The summary adds 18600+LLM versus 18600+GT own, 18600+cross_scene, and the LLM Try5 28000 control, each with a paired 2×2 success table and exact McNemar p; it prints each comparison both on all 1,839 episodes and on the common 1,833.

### Completed T2 results

The four `stats_ep_ckpt_18600_val_unseen_r*_w4.json` files merge to **1,839 unique episodes**. The user's aggregate log reports SR `0.524742`, SPL `0.411273`, NE `4.890942`, OSR `0.676455`, nDTW `0.572185`; recomputation from per-episode JSON agrees after rounding.

| Input / checkpoint (same HEAD harness unless noted) | N | SR | SPL | NE | OSR | nDTW |
|---|---:|---:|---:|---:|---:|---:|
| 18600 + GT own | 1839 | 68.41 | 57.88 | 3.738 | 73.30 | 68.55 |
| 18600 + GT cross_scene | 1839 | 55.68 | 44.44 | 4.478 | 65.09 | 61.74 |
| 18600 + no map | 1839 | 55.68 | 46.86 | 4.247 | 59.27 | 65.75 |
| **18600 + LLM map** (six placeholders) | **1839** | **52.47** | **41.13** | **4.891** | **67.65** | **57.22** |
| LLM Try5 28000 control | 1839 | 64.33 | 54.16 | 3.991 | 69.11 | 66.19 |

| Pair A vs B | N | both succeed | A only | B only | neither | exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| 18600+LLM vs 18600+GT own | 1839 | 894 | 71 | 364 | 510 | 1.67e-48 |
| 18600+LLM vs 18600+GT cross_scene | 1839 | 765 | 200 | 259 | 615 | 0.00672 |
| 18600+LLM vs LLM Try5 28000 | 1839 | 809 | 156 | 374 | 500 | 1.13e-21 |
| 18600+LLM vs 18600+GT own, common caches | 1833 | 892 | 71 | 361 | 509 | 7.83e-48 |
| 18600+LLM vs 18600+GT cross_scene, common caches | 1833 | 764 | 199 | 258 | 612 | 0.00660 |
| 18600+LLM vs LLM Try5 28000, common caches | 1833 | 809 | 154 | 374 | 496 | 3.85e-22 |

On the common 1,833 episodes, 18600+LLM is **52.54 SR / 41.18 SPL**; 18600+GT own is **68.36 / 57.86**, GT cross_scene **55.76 / 44.51**, and no-map **55.76 / 46.91**. The six placeholders produced **two actual successes**, illustrating why the all-episode 52.47 cannot be treated as the LLM control's zero-filled convention. The common-subset comparison barely changes the conclusion.

**Interpretation:** merely feeding the GT-trained 18600 policy LLM maps does not transfer its GT-map gain: SR falls **15.93 points** relative to its GT-own input and is **3.21 points below** its cross-scene input, or **11.86 points below** the separately trained LLM Try5 control. This is strong evidence of an input/model mismatch for this frozen checkpoint, not proof that LLM maps are intrinsically useless. Map quality, the GT-to-LLM value/content distribution shift (`{0,0.6,1}` versus `{0,1}`), and the policy's training adaptation are not separated by this one test. The paired distillation and CE-only runs are prepared to test adaptation, but have not been started.

## T3. Distillation and CE-only launchers (prepared, not run)

Scripts: `scripts/distill/distill_from18600_llm.sh` and `scripts/distill/ce_from18600_llm.sh`. The latter selects the `ce` arm of the former; every option is identical except teacher enablement, run name and port. Both require GPU 1,5,6,7 and use 4 ranks × 4 environments, seed 100, R2R `train_90`, 20,000 iterations. Run them **only after reviewing T2** and, if using the same GPUs, serially.

| Parameter | Distillation | CE-only |
|---|---|---|
| Run directory | `data/logs/checkpoints/d0_distill_from18600_llm/` | `data/logs/checkpoints/d0_ce_from18600_llm/` |
| Port | 29631 | 29632 |
| Trainer/policy | `SS-ETP-LLM` / `LLMGridTry5Policy` | Same |
| Student source | `llm_grid`, key `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree` | Same |
| Architecture / coordinate fusion / refiner | `try5` / `False` / empty | Same |
| Elevation / map units | `y` / `start_position_meters_per_unit=1.0` | Same |
| Construction-only `MODEL.pretrained_path` | `/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt` | Same |
| Actual student initialization | `IL.load_from_ckpt=True`, `is_requeue=False`, `ckpt_to_load=/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth` | Same |
| Optimizer/schedule | AdamW `lr=1e-5`, CE weight 1, sample ratio 0.75, decay interval 1400, warmup 300, min LR ratio 1, waypoint augmentation on | Same |
| Saving | Every 1,000 iterations, retain all | Same |
| GT teacher enabled | `True` | `False` |
| Teacher | Same 18600 checkpoint, `PriorGTTry5Policy`, `gt.legacy.r1p5.direction5.v1`, `gt_teacher_elevation_axis=y` | Configured but inactive |
| Distillation | Weight 1.0, temperature 1.0 | Inactive |

Both launchers write `provenance.txt` with the HEAD hash, 18600 checkpoint SHA-256, actual student init checkpoint, arm, GPU list and port. The checked 18600 SHA-256 is `43a06f31d3949f198beb50cfc04962c3f551c8263824014fd7ab43fbc1765ef6`. Each launcher refuses to reuse an existing output directory.

CPU key/shape validation: the HEAD `LLMGridTry5Policy` configured with `try5/llm_grid`, `coordinate_fusion=False`, and no map-pretraining transfer has **980 expected keys**. After stripping `net.module.` from the 18600 `PriorGTTry5Policy` checkpoint keys, it also has **980 keys**: **missing 0, unexpected 0, shape mismatch 0**. Both policies inherit the same underlying `PriorGTPolicy` network (`vlnce_baselines/models/etp_llm/policy.py:14-16`; `vlnce_baselines/models/etp_prior_gt/policy.py:102-104`). The loader's DDP-prefix handling and full-policy load are at `ss_trainer_ETP_PriorGT.py:426-470`. The wrapped-checkpoint/raw-pretraining distinction is at `vlnbert_init.py:25-55`.

Teacher positional features are independently recomputed with `elevation_axis=self.config.IL.gt_teacher_elevation_axis` and supplied to the teacher (`ss_trainer_ETP_PriorGT.py:1893-1900`); `graph_utils.py:49-52` selects the vertical `dy` calculation for `y`. Thus setting the teacher axis to **y** reaches the intended path, despite the default being `z`.

For the configured R2R `train_90` subset, a CPU file-presence scan found 9,738 episodes, 9,639 with the student LLM map, and **all 9,639** also with the teacher legacy-GT map. The student's normal available-cache filter (`ss_trainer_ETP_PriorGT.py:287-305`) therefore does not admit a teacher-cache miss in this subset.

Commands to run later, **not now**:

```bash
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/distill_from18600_llm.sh
bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/ce_from18600_llm.sh
```

For a CPU-only argument print (does not create output directories or start `torchrun`), append `--print-command` to either script.

## Validation performed

`nvidia-smi` was read once for GPU selection. Namespace creation and 20-loader validation ran on CPU. `diff -u` verified the evaluation launcher's three-line delta. `bash -n` passed for all three launchers; both training launchers' `--print-command` output included the intended port, init checkpoint, teacher flag and `y` elevation. The two Python scripts parsed successfully. The user subsequently ran T2; this CLI ran only the CPU summary and started no GPU process.
