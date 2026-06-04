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
