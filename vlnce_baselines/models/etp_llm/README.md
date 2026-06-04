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

### Generate VLN-CE Navigation Cache

Generate one split at a time with the `cache` mode:

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vlnce_baselines.models.etp_llm.train_llm_boxes cache \
  --model-name-or-path ./data/logs/llm/checkpoints/final/ \
  --dataset R2R \
  --split train \
  --cache-model-key llama-3.1-8b-instruct \
  --batch-size 1 \
  --quiet
```

Repeat for the navigation splits you need:

```shell
for split in train val_seen val_unseen; do
  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vlnce_baselines.models.etp_llm.train_llm_boxes cache \
    --model-name-or-path ./data/logs/llm/checkpoints/final/ \
    --dataset R2R \
    --split "$split" \
    --cache-model-key llama-3.1-8b-instruct \
    --batch-size 1 \
    --quiet
done
```

The command writes:

```text
data/llm_navigation/llama-3.1-8b-instruct/r2r/<split>/predictions/<scene>/<cache_id>.txt
data/llm_navigation/llama-3.1-8b-instruct/r2r/<split>/cognitive_maps/<scene>/<cache_id>.npz
data/llm_navigation/llama-3.1-8b-instruct/r2r/<split>/failures.jsonl
data/llm_navigation/llama-3.1-8b-instruct/r2r/<split>/manifest.json
data/llm_navigation/llama-3.1-8b-instruct/r2r/<split>/metrics.json
```

`cache_id` is `<DATASET>_<split>_<episode_id>`, matching the loader used by
`SS-ETP-LLM` and `GRPO-ETP-LLM`.

Malformed output does not abort the cache run. The generator first tries strict
parsing, then salvage parsing. Salvage parsing keeps valid `path`, `obj`, and
`reg` entities even when other semicolon- or newline-separated entities are invalid.
Warnings are emitted per affected item, dropped entities are written to
`failures.jsonl`, and rates are summarized in `metrics.json`.

If the output omits `path`, the cache generator uses the prompt start position as a
single-point fallback path. It does not fall back to the ground-truth reference path.

### Pretraining Cache

Pretraining does not use VLN-CE episodes. The checked-in pretraining JSONL files are
annotation records keyed by `scan` and `instr_id`, and they contain tokenized
`instr_encoding` rather than raw instruction text. Because of that, pretraining cache
generation must be run from the raw pretraining annotation source before text is
discarded, or from an annotation file that has the original instruction text restored.

Pretraining consumers expect this namespace:

```text
data/llm_navigation/llama-3.1-8b-instruct/pretrain/mixed/predictions/<scene>/<instr_id>.txt
data/llm_navigation/llama-3.1-8b-instruct/pretrain/mixed/cognitive_maps/<scene>/<instr_id>.npz
```

The pretraining loader does not use `VLNCEEpisodeEntry`; it loads by `item["scan"]`
and `item["instr_id"]`.
