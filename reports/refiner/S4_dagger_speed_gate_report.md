# S4 DAgger Speed-Gate Report

## 1. Experiment contract

Both probes used the clean Try5 DAgger configuration with the same seed, data,
initial checkpoint, GPU count, environment count, and optimization settings.
The only model difference was `MODEL.MAP_ENCODER.refiner_ckpt`.

| Setting | Value |
|---|---|
| Seed | 100 |
| GPUs | 4 ranks, GPU 0–3 |
| Environments | 4 per rank |
| Probe length | 200 iterations |
| Initial checkpoint | `../checkpoints/llm-grid-try5-r1p5/model_step_460000.pt` |
| DAgger | lr `1e-5`, sample ratio `0.75`, teacher weight `1.0` |
| Refiner checkpoint | `data/refiner/checkpoints_aug/best.pt` |
| Refiner SHA-256 | `2a75a034b8db69b2673828285a43013bede5bd98eb04143a91f6f8e8d43a8f0d` |

The generated configurations are stored at:

- `data/logs/checkpoints/s4_try5_control_probe200/config.yaml`
- `data/logs/checkpoints/s4_try5_refiner_probe200/config.yaml`

## 2. Speed gate

| Group | Total time (s) | Seconds / iteration | Relative slowdown | Iter-200 IL loss |
|---|---:|---:|---:|---:|
| Try5 control | 2,444.040 | 12.220 | — | 3.204 |
| Try5 + refiner | 3,855.625 | 19.278 | **+57.76%** | 3.233 |

Both probes completed normally and wrote `ckpt.iter200.pth`. No traceback, NaN,
or out-of-memory error was found. The refiner slowdown exceeds the task-book
threshold of 30%, so S4 is paused here. The 30k DAgger runs and navigation
evaluations were not started.

## 3. Refiner profile

The profile was collected on rank 0 over the 200-iteration refiner probe.

| Component | Total (s) | Per iteration (s) | Share of refiner wall time | Share of added time |
|---|---:|---:|---:|---:|
| Evidence projection | 396.499 | 1.982 | 10.28% | 28.09% |
| Refiner input, forward, sigmoid, and composition | 30.149 | 0.151 | 0.78% | 2.14% |
| Remaining training path | 3,428.976 | 17.145 | 88.94% | 69.77%* |

`*` The added-time share subtracts the control wall time from the remaining
training path. It combines changed rollout trajectories, rank synchronization,
and profiler synchronization, so it cannot be attributed to one operation.
The dominant directly measured module cost is evidence projection, not the
refiner network forward.

The detailed probe uses CUDA synchronization around the refiner path. This
makes its timing conservative; it does not alter model outputs, gradients, the
optimizer, or random-number consumption.

## 4. Refiner activation check

The refiner probe log confirms:

- loaded checkpoint: `data/refiner/checkpoints_aug/best.pt`;
- frozen refiner parameter count: 7,827,749;
- episode 3267 at step 0: P0 active spatial cells = 372, refined active spatial
  cells = 349.

The differing cell counts confirm that the refiner was active. The task book
requests step-0/last diagnostics during evaluation; because evaluation was not
entered after the failed speed gate, the last-step evaluation diagnostic is not
available.

## 5. Navigation table status

| Group | val_unseen SR | SPL | NE | OSR | val_seen SR | SPL |
|---|---:|---:|---:|---:|---:|---:|
| Try5 control | not run in S4 | not run | not run | not run | not run | not run |
| Try5 + refiner | not run | not run | not run | not run | not run | not run |

The table is intentionally left unevaluated: proceeding would violate the
explicit speed-gate stop condition. Likewise, no 1,000-iteration loss curve
exists because both probes ended at iteration 200.

## 6. Logs and outputs

- Control log:
  `data/logs/checkpoints/s4_try5_control_probe200/s4_try5_control_probe200_train.log`
- Refiner log:
  `data/logs/checkpoints/s4_try5_refiner_probe200/s4_try5_refiner_probe200_train.log`
- Control probe checkpoint:
  `data/logs/checkpoints/s4_try5_control_probe200/ckpt.iter200.pth`
- Refiner probe checkpoint:
  `data/logs/checkpoints/s4_try5_refiner_probe200/ckpt.iter200.pth`

## 7. Decision

S4 is stopped at the prescribed speed gate. Before any 30k comparison, the
evidence projection path must be made materially cheaper and then re-benchmarked
with the same 200-iteration contract.
