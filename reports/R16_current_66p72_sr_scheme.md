# R7 LLM-Grid Try5 student: 66.72% SR

## Result and artifact

The reported **66.72% SR** is the `dagger_distill_gt_teacher_llmpt` student at **DAgger iteration 17000**, evaluated on R2R `val_unseen` with `eval_map_source=p0` (the precomputed LLM-derived cognitive map, not the GT teacher map). The checkpoint is `data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter17000.pth`; the aggregate result is `data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_eval_iter17000_p0/eval_results/stats_ckpt_17000_val_unseen.json`.

| Metric | Value |
|---|---:|
| Episodes in evaluation denominator | 1,839 |
| SR | 66.72% |
| SPL | 56.02% |
| OSR | 72.43% |
| Navigation error (NE) | 3.817 m |
| nDTW | 65.63% |
| LLM-Navigation cache misses / generation failures | 6 / 1,839 (0.326%) |

The six missing-cache episodes remain in the denominator with zero navigation metrics, as required by [ADR 0007](../docs/adr/0007-precompute-llm-navigation-caches.md). The evaluation log is `data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/eval_iter17000_p0.log`; its lines 9 and 63–81 record cache availability, checkpoint loading, and aggregate metrics.

## Student and teacher

The student is `LLMGridTry5Policy` with `MODEL.MAP_ENCODER.enabled=True`, `architecture=try5`, `source=llm_grid`, and LLM cache model key `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree`. Try5 is one-way graph-to-map attention with `direction5` metadata; the architecture/source distinction follows [ADR 0013](../docs/adr/0013-separate-navigation-architecture-from-cognitive-map-source.md).

The student's pretraining checkpoint was `/home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt`. The training launcher selected `MODEL.MAP_ENCODER.load_pretrained_map_modules=True`. The frozen teacher was `PriorGTTry5Policy`, loaded from `/data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth`, with GT cache namespace `gt.online121c369.r1p5.direction5.v1`. The teacher used legacy `z` elevation while the student used `y`. The training log records teacher checkpoint loading with `missing=0 unexpected=0`; teacher parameters were set non-trainable in `vlnce_baselines/ss_trainer_ETP_PriorGT.py:493-529`.

The teacher did **not** choose rollout actions. Student/expert-forced trajectories and candidate sets were shared, while the teacher independently encoded its language, panoramas, and GT cognitive maps. The student loss combined supervised action cross-entropy and teacher-logit KL with `IL.ml_weight=1`, `IL.distill_weight=1`, and temperature 1; see `vlnce_baselines/ss_trainer_ETP_PriorGT.py:1588-1658,1736-1830,2086-2089`.

## DAgger configuration

The stored configuration at `data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/config.yaml` records: 4 GPUs × 4 environments, `TASK_CONFIG.SEED=100`, R2R training suffix `_90`, 20,000 planned iterations, learning rate `1e-5`, sample ratio 0.75, 300 warmup iterations, decay interval 1400, waypoint augmentation on, and checkpoint interval 1000. The selected artifact is the 17,000-iteration checkpoint, not the final 20,000-iteration checkpoint. The launch script is `scripts/distill/run_dagger_distill_llmpt.sh`; `launch_info.txt` records the final fresh launch at 2026-09-25 00:03:42 local time, git commit `f84c8b1`, four processes × four environments, `load_pretrained_map_modules=True`, and student elevation axis `y`.

## Evaluation protocol

The p0 evaluation used `scripts/refiner/eval_s4.sh dagger_distill_gt_teacher_llmpt 17000 p0`: one evaluation process, four environments, `val_unseen`, all episodes, `IL.back_algo=control`, seed 100, `LLMGridTry5Policy`, Try5/LLM-Grid, and the same LLM cache key. `refiner_ckpt` was empty, so this is not a learned-refiner result. The evaluation loads the DAgger checkpoint above; its pretraining path only supplies construction shapes before the checkpoint weights are loaded.

## What contributes to 66.72%?

The R12 channel ablations on the same 17k checkpoint show:

| Map input | SR | SPL | OSR | NE |
|---|---:|---:|---:|---:|
| p0, unmodified | 66.72 | 56.02 | 72.43 | 3.817 |
| metadata only (raster zeroed) | 66.72 | 56.02 | 72.43 | 3.817 |
| raster only (metadata zeroed) | 65.25 | 55.35 | 71.29 | 3.836 |
| no direction vectors | 66.67 | 56.00 | 72.43 | 3.820 |

The p0 and metadata-only per-episode files have identical MD5 `6ad860ddbd699d04f1d5d4f4d61b7745`. Thus, for this checkpoint and evaluation, zeroing the raster changes no episode outcome. Removing metadata reduces SR by 1.47 percentage points (scene-bootstrap 95% CI for raster-only minus p0: `[-2.78, -0.07]`; exact McNemar `p=0.0260`). The weight sweep and stage probe explain the raster invariance: the loaded `spatial_tokenizer.weight` is exactly zero and remains zero in R7 DAgger checkpoints; its output contains no raster-dependent component. This result should therefore be described as an **LLM-Grid Try5 student whose measured navigation benefit comes from metadata, not from its raster channel**. Sources: `reports/R12_no_new_training_summary.md`, `reports/paired_r7_17k_p0_vs_raster_only.md`, and `reports/R9_map_encoder_stage_probe.md`.

## Comparisons and limits

Against the R5 387500-backbone student at iter 10000, R7 17k gains 1.09 pp SR (scene-bootstrap 95% CI `[+0.06,+2.27]`) and 1.89 pp SPL (`[+0.67,+3.41]`); see `reports/paired_r5_10k_vs_r7_17k.md`. Against the GT teacher's 73.84% SR, this student trails by 7.12 pp SR and 7.36 pp SPL on paired `val_unseen` episodes; most of the SR gap aligns with reachability rather than stopping (`net N→reach = -93` episodes versus `net O→S = -34`), per `reports/paired_teacher_vs_r7_17k.md`.

This is an observed single-checkpoint `val_unseen` result, not a claim about test-set performance or a raster-aware LLM-Grid model. No new training or evaluation was run for this report.
