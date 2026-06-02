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

These paths intentionally fail with `LLMReferencePathNotImplementedError` when
LLM-derived map construction needs `reference_path`. That prevents accidentally
training LLM-Navigation with ground-truth path metadata before reference-path
prediction or an explicit ablation is implemented.
