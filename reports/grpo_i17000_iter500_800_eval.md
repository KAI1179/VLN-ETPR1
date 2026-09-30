# GRPO continuation of R7 DAgger iter 17000: checkpoint 500–800 evaluation

Evaluation date: 2026-09-30 (America/New_York). No training was started by this evaluation.

## Setup and provenance

- GRPO run: `dagger_distill_gt_teacher_llmpt_grpo_i17000`; parent DAgger checkpoint: `data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter17000.pth`.
- Evaluated checkpoints: `ckpt.iter500.pth`, `550`, `600`, `650`, `700`, `750`, and `800` in the GRPO run directory.
- Command: `GPUS=0,1,2,3 EGL_PLATFORM=surfaceless LD_PRELOAD=/lib/x86_64-linux-gnu/libEGL_nvidia.so.0:/lib/x86_64-linux-gnu/libGLX_nvidia.so.0:/lib/x86_64-linux-gnu/libGLdispatch.so.0 bash scripts/distill/eval_iters.sh dagger_distill_gt_teacher_llmpt_grpo_i17000 500 550 600 650 700 750 800`.
- One evaluation process per GPU, one checkpoint at a time on each card; R2R `val_unseen`, `IL.back_algo=control`, 1,839 episodes including six LLM cache misses in every run. The map source was LLM-Grid Try5 (`llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree`), not the GT teacher.
- Driver log: `data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_grpo_i17000/eval_iters_20260930-052903.log`. For each iteration `i`, the aggregate result is `data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_grpo_i17000_eval_iter{i}_val_unseen/eval_results/stats_ckpt_{i}_val_unseen.json`.

The GRPO run's `launch_info.txt` records that the initial segment to iter 500 used profile `nav4`, LR `2e-5`, seed 100. The resumed segment from iter 500 to 800 used profile `nav4`, LR `2e-6`, seed 10, `GRPO.is_requeue=True`. The launcher documents that a resumed GRPO run re-anchors its frozen KL reference to the resumed GRPO checkpoint, rather than the original DAgger checkpoint. Thus 550–800 are a continuation of the 500 checkpoint under changed settings, not a fresh replay of the initial 500-iteration recipe.

## Full val_unseen results

| GRPO iter | SR (%) | SPL (%) | OSR (%) | NE (m) | nDTW (%) | Cache misses |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | **65.47** | **55.86** | 71.13 | 3.978 | **65.94** | 6 |
| 550 | 65.25 | 55.56 | 71.34 | 3.975 | 65.77 | 6 |
| 600 | 64.55 | 55.01 | 71.13 | 3.978 | 65.40 | 6 |
| 650 | 65.31 | 54.49 | **72.32** | **3.937** | 64.74 | 6 |
| 700 | 64.33 | 54.48 | 71.02 | 4.020 | 64.74 | 6 |
| 750 | 64.87 | 54.63 | 71.78 | 4.000 | 64.62 | 6 |
| 800 | 64.93 | 55.14 | 71.34 | 4.050 | 65.33 | 6 |

The parent DAgger iter 17000 p0 evaluation was SR **66.72%**, SPL **56.02%**, OSR **72.43%**, NE **3.817 m**. Among GRPO 500–800, iter 500 has the highest SR (65.47%, −1.25 percentage points from the parent); iter 800 is 64.93% (−1.79 pp). Neither surpasses the DAgger parent. Earlier GRPO checkpoints 100–450 already had results; across all evaluated GRPO checkpoints 100–800, the highest SR remains iter 300 at 66.29%, still below 66.72%.

## Paired comparison with the parent

Both comparisons pair all 1,839 episodes and bootstrap by 11 scenes (10,000 replicates):

| Comparison, GRPO minus DAgger | ΔSR (pp) | 95% CI | ΔSPL (pp) | ΔOSR (pp) | McNemar p |
|---|---:|---:|---:|---:|---:|
| iter 500 | −1.25 | [−3.16, +0.53] | −0.16 | −1.31 | 0.0960 |
| iter 800 | −1.79 | [−3.32, −0.50] | −0.88 | −1.09 | 0.0296 |

At iter 800, path length is 1.07 m shorter and steps are 9.44 fewer than DAgger, but SR and NE are worse; shorter trajectories did not translate into better goal arrival. Detailed transition tables are in `reports/paired_dagger17k_vs_grpo500.md` and `reports/paired_dagger17k_vs_grpo800.md`.

## Conclusion

These seven full-split evaluations show no SR gain from continuing this DAgger student with the observed GRPO 500–800 run. The strongest statement supported by pairing is that iter 800 is worse than the DAgger parent in SR on this split; iter 500's SR difference is not resolved by the scene-bootstrap interval. This does not isolate whether the degradation comes from GRPO itself, the LR/seed change at resume, or KL-reference re-anchoring.
