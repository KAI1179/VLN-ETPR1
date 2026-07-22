# ETP LLM

ETP-LLM is the LLM-based variant of PriorGT.

## LLM-Boxes Training

LLM-Boxes and LLM-Grid always train on the fixed mixed corpus of R2R `train`
and English RxR `train`. Prompts contain the instruction and start metadata but
no source tag. Dataset provenance remains in example identifiers, metrics, and
artifacts. `--limit-per-dataset N` optionally limits each source independently
for a small debugging run.

The maintained Slurm launchers allocate one node, one task, and eight GPUs.
Slurm controls GPU visibility, and the launchers run
`torchrun --standalone --nnodes=1 --nproc-per-node=8`, with per-device batch
size 1, gradient accumulation 1, gradient checkpointing, `device_map=none`,
ten epochs, and LoRA rank/alpha/dropout 32/64/0.05. Accelerate FSDP uses
PEFT-aware full sharding across the eight ranks. Submit from the repository root:

```shell
sbatch scripts/submit/llm-boxes-train-r1p5.sh
sbatch scripts/submit/llm-boxes-train-r2p5.sh
sbatch scripts/submit/llm-grid-train-r1p5.sh
```

The corresponding output directories are:

```text
outputs/llm_boxes/r2r-rxr-bbox-r1p5-path5-no-dataset-tag
outputs/llm_boxes/r2r-rxr-bbox-r2p5-path5-no-dataset-tag
outputs/llm_grid/r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag
```

Both targets use a 1,152-token rendered prompt budget. LLM-Boxes keeps a
4,096-token completion budget. In the measured 19,954-example RxR training
target set, its completion lengths had mean/P95/P99/max
1,234/2,818/3,958/7,173 and 0.81% exceeded 4,096. LLM-Grid scale 2 uses a
3,072-token completion budget plus a 4,096-token total rendered-sequence
budget. Its RxR completion lengths were 1,388/2,595/3,219/4,969 and 1.59%
exceeded 3,072; no measured R2R completion exceeded that threshold. This Grid
completion limit retains about 30,423 of 30,740 combined targets (98.97%). The
total-sequence cap is enforced separately. Over-budget examples are reported
and dropped rather than silently truncated.

Before epoch one, every rank runs a backward preflight on the longest retained
sequence and releases its temporary CUDA allocations. Grid training groups
examples by rendered sequence length. Its maintained launcher sets
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`, and all ranks synchronously
clear their CUDA caches before a step if any rank has a sequence of at least
3,072 tokens. A training OOM fails immediately with epoch, step, rank, example,
sequence-width, and CUDA-memory context rather than retrying an unsafe partial
FSDP step. All ranks participate in FSDP state collection, then rank zero writes
metrics and exports portable PEFT adapters under `checkpoints/epoch-N` or
`checkpoints/final`; navigation-cache consumers load those directories without
FSDP. An epoch checkpoint is written after the epoch completes. Cancelling
mid-epoch does not create an interruption checkpoint or save optimizer state,
so wait for the desired epoch directory before cancelling.

Before model allocation, rank zero alone expands cached targets, filters
over-budget examples, and counts tokens. It atomically writes the per-run
`artifacts/training_manifest.jsonl`; every rank reads the same lightweight
text/token rows after a barrier, avoiding eight copies of heavy R2R/RxR
preprocessing.

## Predictor-Quality Evaluation

```shell
python -m vlnce_baselines.models.etp_llm.llm_boxes_eval \
  --cache-model-key llm-boxes-r2r-rxr-r1p5-path5-tagfree \
  --output-dir outputs/llm_boxes_eval/r1p5

python -m vlnce_baselines.models.etp_llm.llm_grid_eval \
  --cache-model-key llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree \
  --output-dir outputs/llm_grid_eval/r1p5
```

These reference diagnostics score cached raw predictions for R2R `val_seen`
and `val_unseen`; they do not load the trained LLM or run inference again.
Missing and invalid predictions remain in the denominator. RxR validation is
intentionally excluded because LLM-Navigation does not generate or consume an
RxR navigation cache. Use `--limit N` for a bounded check. Derived predictor
metrics are written to the selected output directory and do not overwrite the
cache generator's operational `metrics.json`.

Object/region category precision, recall, and F1 pool category-presence counts
over each split (micro averaging); missing and invalid rows act as empty
predictions. The sweep rejects incomplete prediction coverage and inconsistent
split denominators.

Compare an explicit ordered set of LLM-Grid caches with:

```shell
python -m vlnce_baselines.models.etp_llm.llm_grid_eval_sweep \
  --run-manifest scripts/submit/llm-grid-eval-sweep-r1p5.json \
  --output-dir outputs/llm_grid_eval/checkpoint-sweep-r1p5
```

The sweep writes per-run metrics plus aggregate JSON and CSV. Every listed
cache must already contain R2R `val_seen` and `val_unseen` manifests. Generate
only those predictor-evaluation splits with `llm_grid_navigation_cache --scope
predictor-eval`; `scripts/submit/llm-grid-eval-sweep-r1p5.sh` generates the
missing epoch 1-9 caches on eight GPUs and reuses the existing final/epoch-10
cache.

## LLM-Navigation Scaffold

The maintained LLM-Navigation paths register:

- Policies: `LLMBoxesCurrentPolicy` and `LLMGridTry5Policy`
- DAgger trainer: `SS-ETP-LLM`
- GRPO trainer: `GRPO-ETP-LLM`

Pretraining launchers:

```shell
sbatch scripts/submit/llm-boxes-current-pretrain.sh
sbatch scripts/submit/llm-grid-try5-pretrain.sh
```

Maintained navigation modes:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash llm_boxes_current_dagger
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash llm_boxes_current_grpo
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash llm_boxes_current_eval_dagger
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash llm_boxes_current_eval_grpo
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash llm_grid_try5_dagger
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash llm_grid_try5_grpo
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash llm_grid_try5_eval_dagger
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash llm_grid_try5_eval_grpo
```

These paths consume precomputed LLM-derived cognitive-map caches. They should not
run the LLM online during rollout.

## LLM-Navigation Cache

LLM-Navigation consumes precomputed LLM-derived cognitive maps. The cache layout is:

```text
data/llm_navigation/
  <cache-model-key>/
    r2r/
      train/
        predictions/<scene>/<cache_id>.txt
        cognitive_maps/<scene>/<cache_id>.npz
        status/<scene>/<cache_id>.json
        manifest.json
        metrics.json
        worker_metrics/worker_<index>.json
      val_seen/
      val_unseen/
    rxr/
      train/
      val_seen/
      val_unseen/
    pretrain/
      mixed/
        predictions/<scene>/<instr_id>.txt
        cognitive_maps/<scene>/<instr_id>.npz
        status/<scene>/<instr_id>.json
        manifest.json
        metrics.json
        worker_metrics/worker_<index>.json
```

VLN-CE navigation cache ids use the existing PriorGT convention:

```text
<DATASET>_<split>_<episode_id>
```

Pretraining cache ids use the pretraining annotation `instr_id` and are stored under
`pretrain/mixed` by default because pretraining annotations are not VLN-CE episodes.

Cache generation should warn and continue on malformed LLM output. Valid compact
entities from the output are still parsed and rasterized; invalid entities are recorded
in per-entry `status` JSON files, and aggregate failure rates are written to
`metrics.json`.

### Generate Navigation Cache

Generate all LLM-Navigation caches with one command:

```shell
sbatch scripts/submit/llm-boxes-nav-cache-r1p5.sh
sbatch scripts/submit/llm-grid-nav-cache-r1p5.sh
```

These launchers use the mixed, tag-free checkpoints and write under cache keys
`llm-boxes-r2r-rxr-r1p5-path5-tagfree` and
`llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree`. The matching navigation
pretraining consumers are `scripts/submit/llm-boxes-current-pretrain.sh` and
`scripts/submit/llm-grid-try5-pretrain.sh`. New keys intentionally prevent
resume from mixing these caches with older source-tagged or R2R-only artifacts.

By default, `--parallel-workers auto` launches one worker process per visible CUDA
device. With `CUDA_VISIBLE_DEVICES=4,5,6,7`, the parent process starts four
workers and gives each worker one logical GPU. Each worker loads its own LLM and
generates a deterministic shard of cache ids, so existing `.npz` files are still
skipped independently during resume.

Use a smaller per-worker batch size when a single model copy nearly fills a GPU:

```shell
python -m vlnce_baselines.models.etp_llm.llm_boxes_navigation_cache \
  --model-name-or-path outputs/llm_boxes/r2r-rxr-bbox-r1p5-path5-no-dataset-tag/checkpoints/final \
  --cache-model-key llm-boxes-r2r-rxr-r1p5-path5-tagfree \
  --batch-size 1
```

Force single-process generation when debugging or when only one model copy fits:

```shell
python -m vlnce_baselines.models.etp_llm.llm_boxes_navigation_cache \
  --model-name-or-path outputs/llm_boxes/r2r-rxr-bbox-r1p5-path5-no-dataset-tag/checkpoints/final \
  --cache-model-key llm-boxes-r2r-rxr-r1p5-path5-tagfree \
  --parallel-workers 1
```

Set `--parallel-workers N` to choose a fixed worker count. If `N` is larger than
the visible GPU count, workers are assigned to visible devices round-robin; this
usually only helps when the model is small enough for multiple workers per GPU.

The command writes:

```text
data/llm_navigation/<cache-model-key>/r2r/<split>/predictions/<scene>/<cache_id>.txt
data/llm_navigation/<cache-model-key>/r2r/<split>/cognitive_maps/<scene>/<cache_id>.npz
data/llm_navigation/<cache-model-key>/r2r/<split>/status/<scene>/<cache_id>.json
data/llm_navigation/<cache-model-key>/r2r/<split>/manifest.json
data/llm_navigation/<cache-model-key>/r2r/<split>/metrics.json
data/llm_navigation/<cache-model-key>/r2r/<split>/worker_metrics/worker_<index>.json
data/llm_navigation/<cache-model-key>/pretrain/mixed/predictions/<scene>/<instr_id>.txt
data/llm_navigation/<cache-model-key>/pretrain/mixed/cognitive_maps/<scene>/<instr_id>.npz
data/llm_navigation/<cache-model-key>/pretrain/mixed/status/<scene>/<instr_id>.json
```

It generates R2R `train`, `val_seen`, and `val_unseen`, plus the mixed pretraining
cache. Standalone RxR VLN-CE splits are excluded because navigation does not consume
them; `pretrain/mixed` still includes RxR-derived pretraining records. The cache
generator has no source, split, or annotation-file selector; a complete cache
should be generated as one reproducible artifact set.

Generation resumes by default. If the cognitive-map `.npz` already exists for an
item, that item is skipped before tokenization, LLM generation, and scene-box
preprocessing. The prediction text stores raw decoded model output for auditability,
but navigation consumes the `.npz`, so a map is sufficient for resume. To regenerate
existing cache entries, remove the target cache directory and run generation again.
The existing split manifest must match the requested model path, cache key,
prompt hash, token limits, dataset, split, generator, and scale where applicable;
a mismatch fails with instructions to use a new key or remove the stale cache.

Per-entry `status` JSON files record generation attempt outcomes only:
`complete`, `parse_failed`, `conversion_failed`, or `skipped_input`. Resume skips
do not write or overwrite status files. Failed entries also include a `failures`
array with the parse or conversion details for that entry.

Single-process generation writes split-level `metrics.json` directly. Parallel
generation writes per-worker metrics under `worker_metrics/` and aggregates them
into the split-level `metrics.json` after all workers finish.

Each source prints a resume summary before generation:

```text
cache_resume R2R/train: total=<seen> cached=<map_exists> pending=<to_generate>
```

VLN-CE `cache_id` is `<DATASET>_<split>_<episode_id>`, matching the loader used by
`SS-ETP-LLM` and `GRPO-ETP-LLM`. Pretraining cache ids are `instr_id`.

Malformed output does not abort the cache run. The generator uses strict JSON
parsing only; outputs with invalid JSON, trailing text, unknown categories, invalid
box geometry, or missing `keypoints` are recorded as `parse_failed` and do not
create cognitive-map caches. Other entries continue.

### Pretraining Cache

Pretraining does not use VLN-CE episodes. Cache generation reads ETP-R1 pretraining
annotations through `prior.etp_r1.AnnotationEntry`, which decodes `instr_encoding`,
resolves viewpoint paths through connectivity, and exposes `scan`, `instr_id`,
instruction text, positions, and start direction.

LLM-Navigation pretraining cache generation asks `AnnotationEntry.iter_from` for
English-like entries only. The filter is opt-in for `AnnotationEntry` callers and
keeps mostly ASCII decoded instructions while rejecting Hindi and Telugu script
ranges.

Pretraining consumers use this namespace:

```text
data/llm_navigation/<cache-model-key>/pretrain/mixed/predictions/<scene>/<instr_id>.txt
data/llm_navigation/<cache-model-key>/pretrain/mixed/cognitive_maps/<scene>/<instr_id>.npz
```

The pretraining loader does not use `VLNCEEpisodeEntry` or `AnnotationEntry`; it
loads raw JSONL rows by `item["scan"]` and `item["instr_id"]`. When `--use_llm`
is enabled, it applies the same English-like predicate before checking for
LLM-Navigation cognitive-map cache files.
