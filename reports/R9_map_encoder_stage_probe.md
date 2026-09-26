# R9 map encoder stage probe

## Step 0: permissions

The requested whitelist was merged once into `.claude/settings.local.json` and approved before continuing. No GPU 0–3 was used. `nvidia-smi` showed GPUs 4–7 at 16 MiB each; GPU 4 was selected.

## Step 1: patch and validation

Before applying the patch, the only modified tracked file was `reports/refiner/S4_progress.md`; it was recorded and left untouched. Untracked pre-existing files were not added. Patch 0012 applied successfully:

```text
Applying: Add map-encoder stage probe; use the repo's Tap dashes convention
```

The last commits after applying the patch were:

```text
74bbc59 Record R8 probe and runtime trace diagnosis
fe95720 Record R8 stage 1 evaluation evidence
978d9ca Add offline map-sensitivity probe for distilled students
```

The requested help checks returned `2` and `2`, rather than the task book's expected `1` and `1`, because each option appears twice in the generated help text. Both options are present.

## Step 2: checkpoint paths

All five requested checkpoint paths existed:

```text
-rw-rw-r-- 1 xukai xukai 2492500010 Aug 18 06:43 /home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt
-rw-rw-r-- 1 xukai xukai 2492500010 Sep 17 09:26 /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt
-rw-rw-r-- 1 xukai xukai 5545623485 Sep 25 09:42 data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter5000.pth
-rw-rw-r-- 1 xukai xukai 5545625759 Sep 23 18:24 data/logs/checkpoints/dagger_distill_gt_teacher/ckpt.iter10000.pth
-rw-r--r-- 1 xukai xukai 5545475434 Sep 21 02:08 /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth
```

## Step 3: complete stage-probe output

The probe completed successfully on GPU 4. The raw output was:

```text
rel(gt) = ||stage(GT grid) - stage(LLM grid)|| / ||stage(LLM grid)||, rel(zero) likewise for an all-zero grid; mean over episodes.

=== /home/xukai/code/ETP-R1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt
    weight norms: category_projection.weight 5.3723 | spatial_tokenizer.weight 0.0000 bias inf | metadata_encoder.0.weight 2.2530
    spatial_tokenizer ||W*x|| / ||b|| (LLM grid): 0.000000
                   stage    rel(gt)  rel(zero)
     category_projection   0.968775   1.000000
       spatial_tokenizer   0.000000   0.000000
      spatial_token_norm   0.000000   0.000000
       token_transformer   0.000000   0.000000
             output_norm   0.000000   0.000000

=== /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5_step_387500.pt
    weight norms: category_projection.weight 6.0828 | spatial_tokenizer.weight 0.0000 bias 55.1822 | metadata_encoder.0.weight 2.0663
    spatial_tokenizer ||W*x|| / ||b|| (LLM grid): 0.000000
                   stage    rel(gt)  rel(zero)
     category_projection   0.968759   1.000000
       spatial_tokenizer   0.000000   0.000000
      spatial_token_norm   0.000000   0.000000
             token_transformer   0.000000   0.000000
             output_norm   0.000000   0.000000

=== data/logs/checkpoints/dagger_distill_gt_teacher_llmpt/ckpt.iter5000.pth
    weight norms: category_projection.weight 5.3692 | spatial_tokenizer.weight 0.0000 bias inf | metadata_encoder.0.weight 2.2500
    spatial_tokenizer ||W*x|| / ||b|| (LLM grid): 0.000000
                   stage    rel(gt)  rel(zero)
     category_projection   0.968775   1.000000
       spatial_tokenizer   0.000000   0.000000
      spatial_token_norm   0.000000   0.000000
      token_transformer   0.000000   0.000000
             output_norm   0.000000   0.000000

=== data/logs/checkpoints/dagger_distill_gt_teacher/ckpt.iter10000.pth
    weight norms: category_projection.weight 6.0527 | spatial_tokenizer.weight 17.1614 bias 0.0721 | metadata_encoder.0.weight 16.1073
    spatial_tokenizer ||W*x|| / ||b|| (LLM grid): 39.616801
                   stage    rel(gt)  rel(zero)
     category_projection   0.971602   1.000000
       spatial_tokenizer   0.931665   1.000672
      spatial_token_norm   0.466347   0.426202
       token_transformer   0.646189   0.890210
             output_norm   0.641792   0.975148

=== /data/xukai/etp-r1-snapshot/checkpoints/prior-gt-try5-r1p5/try-5-r1p5-dagger.iter16000.pth
    weight norms: category_projection.weight 6.0279 | spatial_tokenizer.weight 17.8533 bias 0.0749 | metadata_encoder.0.weight 16.1545
    spatial_tokenizer ||W*x|| / ||b|| (LLM grid): 84.703493
                   stage    rel(gt)  rel(zero)
     category_projection   0.968828   1.000000
      spatial_tokenizer   0.901909   1.000542
      spatial_token_norm   0.425154   0.426496
      token_transformer   0.794885   1.490474
             output_norm   0.778295   1.167001
```

## Step 4: decision

The R7 start checkpoint `model_step_460000.pt` already has `output_norm rel(gt)=0.000000`, and R7 iter 5000 is also `0.000000`. Therefore the early-checkpoint sweep was not required: the condition identifies the pretraining checkpoint as the source. No additional GPU probe was run.

Summary:

| checkpoint | spatial tokenizer weight | spatial tokenizer bias | ‖W·x‖/‖b‖ | output_norm rel(gt) | output_norm rel(zero) |
|---|---:|---:|---:|---:|---:|
| LLM-Grid pretrain 460000 | 0.0000 | inf | 0.000000 | 0.000000 | 0.000000 |
| Prior-GT pretrain 387500 | 0.0000 | 55.1822 | 0.000000 | 0.000000 | 0.000000 |
| R7 DAgger 5000 | 0.0000 | inf | 0.000000 | 0.000000 | 0.000000 |
| R5 DAgger 10000 | 17.1614 | 0.0721 | 39.616801 | 0.641792 | 0.975148 |
| GT teacher DAgger 16000 | 17.8533 | 0.0749 | 84.703493 | 0.778295 | 1.167001 |

The answer to question 1 is **pretraining**: the R7 starting checkpoint is already grid-blind, and R7 iter 5000 remains grid-blind. The answer to question 2 is **spatial_tokenizer**, immediately after `category_projection`: its weight is zero and the spatial signal is zero before normalization, so all later stages are identical for GT, LLM, and zero grids.

## Unfinished items

No requested diagnostic step remains unfinished. The two help-count checks differed from the task book's expected numeric output (`2` rather than `1`), and this discrepancy was recorded without modifying the diagnostic code. No evaluation was run and no GPU 0–3 was used.
