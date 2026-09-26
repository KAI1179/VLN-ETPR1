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

Patch 0011 was applied as commit `Add offline map-sensitivity probe for distilled students`. Probe results pending.

## Stage 3: eight-episode runtime trace

Pending.

## Conclusion

Pending runtime evidence.
