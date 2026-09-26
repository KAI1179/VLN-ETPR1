# R8 gt_full swap diagnosis

## Stage 1: preserved evaluation evidence

The four existing logs were copied to adjacent `.log.bak` files before the dry run. Start timestamps come from the first timestamped log entry; end timestamps are the backup file modification times. The MD5 hashes are for per-episode result files.

| Run | Mode | Start | End | Per-episode MD5 |
|---|---|---|---|---|
| R7 `dagger_distill_gt_teacher_llmpt` iter 5000 | p0 | 2026-09-25 11:51:00 | 2026-09-25 12:03:23 | `8c3c7dd255c331bafa3991d91783bb0c` |
| R7 `dagger_distill_gt_teacher_llmpt` iter 5000 | gt_full | 2026-09-25 22:13:13 | 2026-09-25 22:56:32 | `8c3c7dd255c331bafa3991d91783bb0c` |
| R5 `dagger_distill_gt_teacher` iter 10000 | p0 | 2026-09-24 20:20:43 | 2026-09-24 20:33:11 | `58cb18a9be1763c314ccdff5f8730877` |
| R5 `dagger_distill_gt_teacher` iter 10000 | gt_full | 2026-09-24 20:35:22 | 2026-09-24 21:18:18 | `269915735fdba4833a7e9093b1889603` |

The first sandboxed dry run failed before launch with `PermissionError: [Errno 1] Operation not permitted` while binding a local rendezvous port. Retrying outside the sandbox completed with terminal output `Refiner: disabled`. Its generated config contains `181:    eval_map_source: gt_full`, so the CLI override reached configuration. The R7 p0 and gt_full aggregate JSON files also share MD5 `5fa3222f62d10bba8f959f6d453320d8`.

## Stage 2: offline sensitivity probe

Patch 0011 was applied as commit `Add offline map-sensitivity probe for distilled students`. The documented `--run-name` option initially failed (`error: unrecognized arguments: --run-name dagger_distill_gt_teacher_llmpt`); the installed Tap accepted `--run_name`. Both corrected probe commands exited successfully on GPU 4.

R7 iter 5000 raw probe table and verdict:

```text
 episode        scene gt_cells llm_cells  tok_rel     d_gt   d_zero   d_none  flip_gt flip_zero flip_none
      53  zsNo4HB9uLZ      266       428   0.0000   0.0000   0.0000   1.2179     0.00      0.00      0.09
     992  TbHJrupSAjP      201       100   0.0000   0.0000   0.0000   1.2228     0.00      0.00      0.09
     444  EU6Fwq7SyZv      238       240   0.0000   0.0000   0.0000   1.2254     0.00      0.00      0.09
    1352  QUCTc6BB5sX      197       168   0.0000   0.0000   0.0000   1.2556     0.00      0.00      0.09
    1685  zsNo4HB9uLZ      242       128   0.0000   0.0000   0.0000   1.2359     0.00      0.00      0.09
VERDICT: the student's action never changes when the grid is swapped -> model side:
         the map channel is dead; a bit-identical p0/gt_full eval is expected.
```

R5 iter 10000 raw probe table and verdict:

```text
 episode        scene gt_cells llm_cells  tok_rel     d_gt   d_zero   d_none  flip_gt flip_zero flip_none
      53  zsNo4HB9uLZ      266       428   0.8113   0.1968   0.5046   0.5977     0.06      0.09      0.09
     992  TbHJrupSAjP      201       100   0.5443   0.2795   0.4641   0.5961     0.00      0.06      0.12
     444  EU6Fwq7SyZv      238       240   0.4745   0.1454   0.4670   0.5720     0.03      0.09      0.12
    1352  QUCTc6BB5sX      197       168   0.7116   0.1682   0.4967   0.6120     0.03      0.06      0.12
    1685  zsNo4HB9uLZ      242       128   0.6672   0.1947   0.4759   0.5629     0.06      0.09      0.09
VERDICT: the student reacts to the grid (6/160 argmax flips) -> a
         bit-identical p0/gt_full eval means the gt_full swap never reached the encoder.
```

Both probes reported checkpoint `missing=0 unexpected=0`. The full raw probe outputs are `reports/r8_probe_r7_5000.txt` and `reports/r8_probe_r5_10000.txt`.

## Stage 3: eight-episode runtime trace

The first trace invocation failed because this Tap version recognized underscore argument names, despite `underscores_to_dashes=True`; the new tracing script now accepts the requested dashed names by normalizing them before parsing. The successful runs used GPU 4 and `EVAL.EPISODE_COUNT=8`. The evaluator processed **10** episodes in each mode, apparently overshooting the requested cap at vector batch granularity. Both per-episode JSON dictionaries contain 10 entries.

All emitted trace lines for p0:

```text
[trace] map_tokens norm=353.1779
[trace] map_tokens norm=353.1779
[trace] map_tokens norm=353.1779
[trace] map_tokens norm=353.1779
```

All emitted trace lines for gt_full:

```text
[trace] swap changed=True active_before=100 active_after=201
[trace] map_tokens norm=353.1779
[trace] swap changed=False active_before=201 active_after=201
[trace] map_tokens norm=353.1779
[trace] swap changed=False active_before=201 active_after=201
[trace] map_tokens norm=353.1779
[trace] swap changed=False active_before=201 active_after=201
[trace] map_tokens norm=353.1779
[trace] swap changed=False active_before=201 active_after=201
[trace] map_tokens norm=353.1779
[trace] map_tokens norm=249.1193
```

The `_initialize_refiner_state` wrapper emitted no line in either mode. The gt_full swap wrapper did execute and saw a changed raster at its first call. Per-episode result MD5 values are identical:

```text
2b15768131e135bdd4d758e620fcc689  data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_trace_iter5000_p0/eval_results/stats_ep_ckpt_5000_val_unseen_r0_w1.json
2b15768131e135bdd4d758e620fcc689  data/logs/checkpoints/dagger_distill_gt_teacher_llmpt_trace_iter5000_gt_full/eval_results/stats_ep_ckpt_5000_val_unseen_r0_w1.json
```

Full runtime logs are `reports/r8_trace_5000_p0.log` and `reports/r8_trace_5000_gt_full.log`.

## Conclusion

**模型侧。** The direct evidence is R7 iter 5000's offline probe: swapping GT for LLM grid produced `tok_rel=0`, `d_gt=0`, and `flip_gt=0` across all five episodes, while removing the map changed logits (`d_none` about 1.2). The live gt_full trace confirms its grid was replaced (`active_before=100`, `active_after=201`), yet the per-episode results matched p0 byte for byte. R5 iter 10000 is a different case: its probe showed grid sensitivity and its full p0/gt_full per-episode MD5 values differed.

No requested diagnostic remains unrun. The trace emitted no `refiner_state` line, and the evaluator ran 10 rather than exactly 8 episodes under `EVAL.EPISODE_COUNT=8`; neither affects the direct R7 model-side evidence.
