# D10: model identity audit — stopped at Section 3.1

## Decision

**STOP: the two arms do not differ only in map input.** Z was initialized
from the baseline navigation pretraining checkpoint at step 367500; M was
initialized from LLM-Grid pretraining at step 460000. Both have the same
parameter keys and shapes, but that does not make their training initializations
identical. Therefore the task book's mandatory identity condition is not met.

No D10 launcher, sbatch job, GPU inference, attention-token check, or A1–A4/lift
analysis was executed. No training/model source or existing evaluation artifact
was modified. This report is not an attention-analysis result.

Task book:
`/home/xukai/code/ETP-R1-snapshot/ETP-R1/upload_files/d10_map_attention_ablation.md`,
Sections 3.1 and 4.1. Source HEAD at inspection:
`be078290ba9a88ac805919a0588a5daea3fcf21d`.

## 1. Checkpoints and architecture

| Item | Z: zero-map model | M: Try5, no KD |
|---|---|---|
| Checkpoint | `/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-000-baseline-dagger-28k/ckpt.iter28000.pth` | `/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth` |
| Original training run | `llm-grid-000-baseline-dagger-20261009T022556Z-dagger` | `release_r2r_llm_grid_try5_dagger` |
| Iteration | 28000 | 28000 |
| Policy | `LLMGridTry5Policy` | `LLMGridTry5Policy` |
| Map configuration | enabled, `try5`, `llm_grid` | enabled, `try5`, `llm_grid` |
| State-dict entries | 980 | 980 |

CPU checkpoint-metadata comparison found **0 Z-only keys, 0 M-only keys, and
0 shape mismatches**. Tensor storage was not needed for this structural check.
It does not establish equal weights or equal initialization.

## 2. Training configuration comparison

The following values were read from the checkpoints' embedded configurations,
not inferred from the current default configuration.

### Substantive differences

| Configuration key | Z | M |
|---|---|---|
| `MODEL.pretrained_path` | `/workspaces/ETP-R1/.worktrees/llm-grid-000-baseline-dagger/pretrained/r2r_rxr_ce/baseline/store2/model_step_367500.pt` | `pretrained/r2r_rxr_ce/llm_grid_try5/store2/model_step_460000.pt` |
| `MODEL.pretrained_initialization` | `baseline-nav` | not recorded |
| `MODEL.MAP_ENCODER.input_mask` | `zero-grid-and-directions` | not recorded |

The pretraining checkpoint is a **non-input confound**. Loading the complete
DAgger checkpoint for evaluation correctly restores each trained model; it does
not retrospectively remove this difference in how the models were trained.

The retained original M training configuration also records 460000 at line 216:
`/data/xukai/etp-r1-snapshot/checkpoints/data_try5_local/logs/checkpoints/release_r2r_llm_grid_try5_dagger/config.yaml`.
Its machine-specific path differs from the embedded relative path, but the
recorded pretraining checkpoint step is the same.

### Identical recorded settings

| Setting | Both arms |
|---|---|
| LLM cache key | `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree` |
| Seed / GPUs / environments per rank | 100 / 4 / 4 |
| Training dataset suffix | `_90` |
| Planned iterations / checkpoint examined | 30000 / 28000 |
| IL learning rate / batch size | `1e-5` / 1 |
| Warmup / decay interval / minimum LR ratio | 500 / 2000 / 1.0 |
| Sample ratio / ML weight | 0.75 / 1.0 |
| Logging interval / waypoint augmentation | 200 / enabled |
| Recorded training back algorithm / tryout | `teleport` / true |
| Maximum instruction length / high-level steps | 150 / 15 |
| Simulator step limit | 5000 |
| Agent height / radius | 1.5 / 0.1 m |
| `freeze_base` / `fix_lang_embedding` / `fix_pano_embedding` | false / false / false |
| Recorded map / trajectory-keypoint loss weights | 0.1 / 0.001 |

The presence of auxiliary-loss weights alone does not establish that those
losses were active on the Try5 path. Neither old configuration records
`coordinate_fusion`, `load_pretrained_map_modules`, or `elevation_axis`; their
historical values cannot be recovered simply by applying today's defaults.

### Other complete-config differences

| Key | Z | M |
|---|---|---|
| `GRPO.require_complete_checkpoint` | false | not recorded |
| `GRPO.resume` | empty string | not recorded |
| `GRPO.train_all` | not recorded | false |
| `GRPO.trainable_profile` | `nav4` | not recorded |
| `IL.resume` | empty string | not recorded |
| `MODEL.MAP_ENCODER.llm_train_reference_model_key` | empty string | not recorded |
| `MODEL.resume` | false | not recorded |
| `TRAINING_RECOVERY` | false | not recorded |
| `use_pbar` | true | false |

Five output/bookkeeping keys also differ: `CHECKPOINT_FOLDER`, `RESULTS_DIR`,
`TENSORBOARD_DIR`, `LOG_FILE`, and `VIDEO_DIR`, using the respective run names
above. `CMD_TRAILING_OPTS` records the launch-option differences and their
ordering, including Z's explicit initialization/mask/seed/resume options.
These account for the remaining differences in the flattened configurations.
Unrecorded keys are not silently equated with current defaults.

## 3. Which inputs were zero?

### Original Z training and archived evaluation

Both the embedded configuration and `CMD_TRAILING_OPTS` explicitly record
`MODEL.MAP_ENCODER.input_mask=zero-grid-and-directions`. However, that switch's
implementation and its original training/evaluation launcher were not found in
the current navigation source or local Git history. Consequently:

- The recorded mask value is confirmed.
- Whether that historical implementation also zeroed `start_position` and
  `start_direction_vector` is **not confirmed**.
- The archived 63.30/53.20 result is not proof that the start pose was zero in
  training. The model's directory name is not sufficient evidence either.

### Current Z own-rollout evaluation

The recent analysis launcher explicitly supplies a separate cache in which
`grid`, the five route direction vectors, `start_direction_vector`, and
`start_position` are zero. The actual loader outputs were CPU-checked:
`grid`, `map_trajectory_metadata`, `start_direction_vector`, and `start_position`
are all zero. The map branch remains enabled. Original caches are untouched.

Evidence:

- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/dump_eval.sh:46`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/make_zero_input_cache.py:35`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/pa_zero_own_val_unseen/verify.json`

Zero inputs do not imply a zero metadata **token** after the trained metadata
encoder. No D10 token-uniformity or metadata-attention check has been performed.

## 4. Existing own-rollout metrics, not D10 S reproduction

| Evaluation | Episodes | SR (%) | SPL (%) |
|---|---:|---:|---:|
| Z archived reference | 1839 | 63.30 | 53.20 |
| Z recent all-zero-input own dump | 1839 | 62.97 | 53.17 |
| M paired-analysis own dump | 1839 | 65.04 | 55.01 |

All three totals include six unavailable original LLM-cache episodes scored
zero. D10 shared-state frames cover the 1833 actually available episodes.
The Z reference comparison records 217 changed episodes / 575 changed metric
values. Its recent evaluation is therefore **not** an exact reproduction of
the archived evaluation. It has not been represented as one.

Metric sources:

- `/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-000-baseline-dagger-28k/eval_results/stats_ckpt_28000_val_unseen.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/pa_zero_own_val_unseen/eval_results/stats_ckpt_28000_val_unseen.json`
- `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/pa_try5_own_val_unseen/eval_results/stats_ckpt_28000_val_unseen.json`

An expert-replay S rollout executes the replay actions, not the policy's own
argmax actions. Its success metrics must not be presented as a reproduction of
the policy's own-rollout SR/SPL. The existing Z own run above is the available
own-rollout evidence; no new D10 S or own run was launched.

## 5. M identity and reusable D9 S data

The paper's local stratified-table workflow selects Try5 iter28000 through
`scripts/analysis/dump_eval.sh` -> `scripts/eval_d7/d7d_eval.sh:16`.
The PA decision header and its evaluation log identify the same checkpoint
path as `scripts/eval_d9/dump_shard.sh:15` and the D9 Try5 S logs:

`/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth`.

Thus **M does not require a new dump merely to change its identity**. The
available S data are:

`/data/xukai/checkpoint/etp-r1-snapshot-data-logs/logs/d9_reference_layout/try5/S/`

The existing `.complete.json` records **1833 episodes / 13992 frames** and the
D8 Try5 S reference directory. Four logical-rank completion records retain
reference world size 4 and four environments per rank. The original verifier
checks exact step, position, candidate IDs/positions, and replayed actions;
`run_shard.py` additionally checks heading. This audit read those records and
code without rerunning a verifier that writes completion files.

The 64.33 paper-reference control and 65.04 PA/D9-harness result are different
evaluations of iter28000, not evidence of a different selected model. Relevant
local control metrics reside at:
`/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/llm_try5_control_val_unseen/eval_results/stats_ckpt_28000_val_unseen.json`.

## 6. Work deliberately not started

Per the mandatory stop rule, no Z S launcher or sbatch command is supplied,
and there are no D10 attention tables, confidence intervals, figure data, or
map-uniformity conclusions. Existing D9 data remain unchanged.

Proceeding with these checkpoints would require explicitly relaxing the
"map input is the only difference" premise and interpreting the result as a
descriptive comparison of two differently initialized models. A comparison
that retains that premise needs a suitably initialization-matched control.
Neither alternative was automatically implemented.
