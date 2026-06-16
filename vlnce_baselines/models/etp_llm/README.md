# ETP LLM

ETP-LLM is the LLM-based variant of PriorGT.

## LLM-Boxes Training

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vlnce_baselines.models.etp_llm.train_llm_boxes train
```

## LLM-Boxes Evaluating

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vlnce_baselines.models.etp_llm.train_llm_boxes eval --model-name-or-path ./data/logs/llm/checkpoints/final/
```

## LLM-Navigation Scaffold

The LLM-Navigation scaffold registers:

- Policy: `LLMPolicy`
- DAgger trainer: `SS-ETP-LLM`
- GRPO trainer: `GRPO-ETP-LLM`
- Pretraining flag: `--use_llm`

Pretraining checkpoints are routed under:

```text
pretrained/r2r_rxr_ce/llm/
```

Launcher modes:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash pretrain_src/run_pt/run_mix_server.bash 2333 --use_llm
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash llm_dagger 2333
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash llm_grpo 2333
```

These paths consume precomputed LLM-derived cognitive-map caches. They should not
run the LLM online during rollout.

## LLM-Navigation Cache

LLM-Navigation consumes precomputed LLM-derived cognitive maps. The cache layout is:

```text
data/llm_navigation/
  llama-3.1-8b-instruct/
    r2r/
      train/
        predictions/<scene>/<cache_id>.txt
        cognitive_maps/<scene>/<cache_id>.npz
        failures.jsonl
        manifest.json
        metrics.json
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
        failures.jsonl
        manifest.json
        metrics.json
```

VLN-CE navigation cache ids use the existing PriorGT convention:

```text
<DATASET>_<split>_<episode_id>
```

Pretraining cache ids use the pretraining annotation `instr_id` and are stored under
`pretrain/mixed` by default because pretraining annotations are not VLN-CE episodes.

Cache generation should warn and continue on malformed LLM output. Valid compact
entities from the output are still parsed and rasterized; invalid entities are recorded
in `failures.jsonl`, and aggregate failure rates are written to `metrics.json`.

### Generate Navigation Cache

Generate all LLM-Navigation caches with one command:

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vlnce_baselines.models.etp_llm.generate_navigation_cache \
  --model-name-or-path ./data/logs/llm/checkpoints/final/ \
  --cache-model-key llama-3.1-8b-instruct \
  --batch-size 8
```

By default, `--parallel-workers auto` launches one worker process per visible CUDA
device. With `CUDA_VISIBLE_DEVICES=4,5,6,7`, the parent process starts four
workers and gives each worker one logical GPU. Each worker loads its own LLM and
generates a deterministic shard of cache ids, so existing `.npz` files are still
skipped independently during resume.

Use a smaller per-worker batch size when a single model copy nearly fills a GPU:

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vlnce_baselines.models.etp_llm.generate_navigation_cache \
  --model-name-or-path ./data/logs/llm/checkpoints/final/ \
  --cache-model-key llama-3.1-8b-instruct \
  --batch-size 1
```

Force single-process generation when debugging or when only one model copy fits:

```shell
CUDA_VISIBLE_DEVICES=4 python -m vlnce_baselines.models.etp_llm.generate_navigation_cache \
  --model-name-or-path ./data/logs/llm/checkpoints/final/ \
  --cache-model-key llama-3.1-8b-instruct \
  --parallel-workers 1
```

Set `--parallel-workers N` to choose a fixed worker count. If `N` is larger than
the visible GPU count, workers are assigned to visible devices round-robin; this
usually only helps when the model is small enough for multiple workers per GPU.

The command writes:

```text
data/llm_navigation/llama-3.1-8b-instruct/r2r/<split>/predictions/<scene>/<cache_id>.txt
data/llm_navigation/llama-3.1-8b-instruct/r2r/<split>/cognitive_maps/<scene>/<cache_id>.npz
data/llm_navigation/llama-3.1-8b-instruct/r2r/<split>/failures.jsonl
data/llm_navigation/llama-3.1-8b-instruct/r2r/<split>/manifest.json
data/llm_navigation/llama-3.1-8b-instruct/r2r/<split>/metrics.json
data/llm_navigation/llama-3.1-8b-instruct/rxr/<split>/predictions/<scene>/<cache_id>.txt
data/llm_navigation/llama-3.1-8b-instruct/rxr/<split>/cognitive_maps/<scene>/<cache_id>.npz
data/llm_navigation/llama-3.1-8b-instruct/pretrain/mixed/predictions/<scene>/<instr_id>.txt
data/llm_navigation/llama-3.1-8b-instruct/pretrain/mixed/cognitive_maps/<scene>/<instr_id>.npz
```

It generates `train`, `val_seen`, and `val_unseen` for both R2R and RxR, plus the
mixed pretraining cache. The cache generator intentionally has no `--dataset`,
`--split`, or `--annotation-file` selector; a complete cache should be generated as
one reproducible artifact set.

Generation resumes by default. If the cognitive-map `.npz` already exists for an
item, that item is skipped before tokenization, LLM generation, and scene-box
preprocessing. The prediction text is kept for auditability, but navigation consumes
the `.npz`, so a map is sufficient for resume. Use `--overwrite` to regenerate
existing cache entries.

Each source prints a resume summary before generation:

```text
cache_resume R2R/train: total=<seen> cached=<map_exists> pending=<to_generate>
```

VLN-CE `cache_id` is `<DATASET>_<split>_<episode_id>`, matching the loader used by
`SS-ETP-LLM` and `GRPO-ETP-LLM`. Pretraining cache ids are `instr_id`.

Malformed output does not abort the cache run. The generator first tries strict
parsing, then salvage parsing. Salvage parsing keeps valid `keypoints`, `obj`, and
`reg` entities even when other semicolon- or newline-separated entities are invalid.
Warnings are emitted per affected item, dropped entities are written to
`failures.jsonl`, and rates are summarized in `metrics.json`.

If the output omits `keypoints`, the generator warns, records the failure, and
refuses to create a cognitive-map cache for that entry. Other entries continue.

### Pretraining Cache

Pretraining does not use VLN-CE episodes. Cache generation reads ETP-R1 pretraining
annotations through `prior.etp_r1.AnnotationEntry`, which decodes `instr_encoding`,
resolves viewpoint paths through connectivity, and exposes `scan`, `instr_id`,
instruction text, positions, and start direction.

Pretraining consumers use this namespace:

```text
data/llm_navigation/llama-3.1-8b-instruct/pretrain/mixed/predictions/<scene>/<instr_id>.txt
data/llm_navigation/llama-3.1-8b-instruct/pretrain/mixed/cognitive_maps/<scene>/<instr_id>.npz
```

The pretraining loader does not use `VLNCEEpisodeEntry`; it loads by `item["scan"]`
and `item["instr_id"]`.
