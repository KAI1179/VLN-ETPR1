# GT coordinate-fusion iter19200: evaluation preparation

## Confirmed settings

Transferred run directory:
`/data/xukai/checkpoint/etp-r1-snapshot-data-logs/logs/checkpoints/dagger_coordfusion_gt_legacy/`.

The supplied `config.yaml`, `eval_args.txt` and `launch_info.txt` confirm
PriorGTTry5Policy / SS-ETP-PriorGT, GT legacy namespace
`gt.legacy.r1p5.direction5.v1`, coordinate_fusion=True, elevation_axis=y,
start_position_scale=1.0, start_position_meters_per_unit=1.0,
teleport/tryout, 150 tokens / 15 high-level steps.
Training commit: `55e2ddeaf705c8266554c66d81ab970e12e95598`.
Training initialization: model_step_440000.pt; seed 100; four ranks × four envs.

## Blocker: checkpoint unreadable

CPU `torch.load(map_location='cpu')` fails with:

```text
RuntimeError: PytorchStreamReader failed reading zip archive:
failed finding central directory
```

The file has a ZIP header but no readable central directory;
`zipfile.is_zipfile` is False. At 2026-10-13 10:54:27 -04:00,
size was 3,237,052,416 bytes and modification time was 10:48:21 -04:00.
This is consistent with an incomplete transfer or a truncated file, but the
cause has not been confirmed. Embedded configuration/weights cannot be checked.
No GPU evaluation was started and no performance result is available.
The earlier local search missed this newly transferred run; it is now located.

## Prepared launcher (not executed)

`/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_gt19200/eval_own.sh`
passes `bash -n`. The user authorized GPUs 0–3, so the launcher uses four
ranks on those GPUs, port 29820, R2R val_unseen, and the settings above.
It evaluates under the local HEAD harness, not the historical cloud commit.

The cloud-only pretraining path is replaced with local
`/data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/model_step_440000.pt`.
`load_pretrained_map_modules=False` applies only to initialization;
the intended evaluated map/fusion weights come from the complete iter19200
DAgger checkpoint. Its full loading/mismatch report must be checked after
the checkpoint is readable. Refiner is disabled; no map ablation is applied.

Planned output:
`/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/gt_coordfusion_legacy_iter19200_val_unseen_teleport_w4/`.

Do not launch until the checkpoint has been transferred completely and CPU
loading passes. No source file, trained weight, original namespace or other
process was modified by this preparation.
