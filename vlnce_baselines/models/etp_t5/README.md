# ETP T5

ETP-T5 is the T5-based variant of PriorGT.

## T5-Boxes Training

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vlnce_baselines.models.etp_t5.train_t5_boxes train
```

## T5-Boxes Evaluating

```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vlnce_baselines.models.etp_t5.train_t5_boxes eval --model-name-or-path ./data/logs/t5/checkpoints/final/
```

## T5-Navigation Scaffold

The T5-Navigation scaffold registers:

- Policy: `T5Policy`
- DAgger trainer: `SS-ETP-T5`
- GRPO trainer: `GRPO-ETP-T5`
- Pretraining flag: `--use_t5`

Pretraining checkpoints are routed under:

```text
pretrained/r2r_rxr_ce/t5/
```

Launcher modes:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash pretrain_src/run_pt/run_mix_server.bash 2333 --use_t5
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash t5_dagger 2333
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash t5_grpo 2333
```

These paths intentionally fail with `T5ReferencePathNotImplementedError` when
T5-derived map construction needs `reference_path`. That prevents accidentally
training T5-Navigation with ground-truth path metadata before reference-path
prediction or an explicit ablation is implemented.
