# S4 Preparation

## Changes

- Removed per-step CUDA synchronization and the projection/forward profiler.
- Kept 200-iteration wall-time, IL-loss, and refiner-activation logging.
- Applied the observed mask to all 37 cognitive-map channels: observed cells use
  refiner output; unseen cells preserve P0.
- Decoupled logging and checkpoint cadence: logs every 200 iterations and
  checkpoints every 1,000 iterations.
- Added two background DAgger launchers and one single-GPU evaluation launcher.

## Launchers

| Script | Purpose | GPUs | Run name |
|---|---|---|---|
| [`scripts/refiner/run_s4_refiner.sh`](../../scripts/refiner/run_s4_refiner.sh) | Try5 + refiner, 30k DAgger | 0–3 | `s4_try5_refiner` |
| [`scripts/refiner/run_s4_control.sh`](../../scripts/refiner/run_s4_control.sh) | Try5 control, 30k DAgger | 0–3 | `s4_try5_control` |
| [`scripts/refiner/eval_s4.sh`](../../scripts/refiner/eval_s4.sh) | `val_unseen` evaluation | 4 by default | supplied run name |

All launchers contain an absolute repository path, Python environment path,
initial Try5 checkpoint path, and output paths. They change to the repository
root internally and do not depend on the caller's current directory.

## Output contract

| Artifact | Path |
|---|---|
| Training stdout/stderr | `data/logs/checkpoints/<run_name>/train.log` |
| Resolved training configuration | `data/logs/checkpoints/<run_name>/config.yaml` |
| Checkpoints | `data/logs/checkpoints/<run_name>/ckpt.iter1000.pth` through `ckpt.iter30000.pth` |
| Evaluation log | `data/logs/checkpoints/<run_name>/eval_iter<iter>.log` |
| Evaluation JSON | `data/logs/checkpoints/<run_name>_eval_iter<iter>/eval_results/stats_ckpt_<iter>_val_unseen.json` |
| Progress table | `reports/refiner/S4_progress.md` |

Checkpoints are not pruned or overwritten by a later iteration. At the measured
size of about 5.2 GiB per checkpoint, one 30-checkpoint run requires roughly
156 GiB.

## Dry-run verification

The three commands below completed outside the sandbox without constructing a
model, allocating a GPU, training, or evaluating:

| Command | Result |
|---|---|
| `bash scripts/refiner/run_s4_refiner.sh --dry-run` | Passed; 30k/200/1000 configuration written under `s4_try5_refiner` |
| `bash scripts/refiner/run_s4_control.sh --dry-run` | Passed; control has an empty `refiner_ckpt` |
| `bash scripts/refiner/eval_s4.sh --dry-run s4_try5_refiner 5000` | Passed; selected the expected policy and refiner checkpoints |

Dry-run verifies argument parsing, merged configuration, and output paths. It
does not load model weights or execute a rollout.

## Commands for the user

```bash
bash scripts/refiner/run_s4_refiner.sh
bash scripts/refiner/run_s4_control.sh
bash scripts/refiner/eval_s4.sh s4_try5_refiner 5000
```

The control launcher is prepared but should remain unstarted until requested.
