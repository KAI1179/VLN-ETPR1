# Paper qualitative cases: setup and current status

## Status

CPU preparation completed. **No GPU evaluation has been launched and no new
episode result table or selected-case figure is claimed.** Awaiting authorized
evaluation GPUs and the two episode IDs used in `attn_case.pdf`. The PDF was not
located; previous D10 exports1392/92 alone do not prove they are its episodes.

## Exact checkpoints and reference harness

| Arm | Checkpoint | Reference |
|---|---|---|
| ours | `/home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/ckpt.iter27000.pth` | SR67.482327 / SPL56.631077, control |
| zeromap | `/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-000-baseline-dagger-28k/ckpt.iter28000.pth` | Archived SR63.295269 / SPL53.197892; current PA reproduction62.969005 /53.167436 |

Ours' final reference is
`data/logs/checkpoints/kd_full27000_control_val_unseen_missing_zero/eval_results/`.
Its val_seen reference is75.964010 /67.954375. These runs have stats only, no
decision logs. Existing PA KD logs are not an exact replacement: they have
six zero-score cache misses and two additional nonmissing episode differences.
Reference67.48 uses the `__missing-all-zero` cache and actually runs all1839
episodes; the new batch retains that convention for both arms. Zeromap gets
all four encoder input tensors zero, including start pose. Thus the new batch
does not promise to reproduce the archived63.30 exactly.

The batch reuses `scripts/eval_d7/d7d_eval.sh` unchanged: one rank per GPU listed
in `GPUS`, four environments per rank, control, seed100, y elevation,150 text tokens,15 high-level steps,
5000 low-level steps. `EVAL.decision_dump=True` activates the already-existing
eval-only dumper; no model/trainer source edit is needed. Jobs run serially.

## Step1: batch evaluation and JSONL export

After GPUs are approved, in tmux (replace the GPU list):

```bash
GPUS=7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/paper_cases/run_batch.sh
```

Single-GPU evaluation uses1rank×4environments. This differs from the historical
4rank layout, so per-episode outcomes and aggregate metrics need not be identical.
Four-GPU lists remain supported. Ports29910–29913. The previous20–40min estimate
was for four GPUs; single-GPU runtime has not been benchmarked. Scripts do not start watchers or
background jobs. Per-job logs:
`data/logs/checkpoints/paper_case_<ours|zeromap>_<split>/eval.log`.

Each job invokes CPU `collect.py` after completing. Outputs:

- `results/ours_val_unseen_episodes.jsonl`
- `results/zeromap_val_unseen_episodes.jsonl`
- `results/ours_val_seen_episodes.jsonl`
- `results/zeromap_val_seen_episodes.jsonl`

Each row includes all requested metrics, per-high-level-step position/heading
and yaw-derived quaternion, low-level Position-measure positions, chosen nodes,
all visited/ghost/STOP candidates and probabilities, effective STOP step and
forced-stop flags, raw decisions and checkpoint/source provenance. Ground
waypoint probabilities are not confused with forced STOP. All1839/778 rows
must correspond to actual completed rollouts; missing-score placeholders,
incomplete records or disagreement with stats cause explicit errors.

New zero cache: `outputs/paper_case_zero_cache/`, key
`paper.case.zero_all_inputs.v1`;1839+778 complete entries link to one validated
all-zero template. Original caches remain unchanged. Ours reuses its existing
complete `outputs/kd27000_missing_zero_cache/`.

## Step2: CPU shortlist

After the result tables exist and the exclusion IDs are confirmed:

```bash
CUDA_VISIBLE_DEVICES='' /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/paper_cases/select_cases.py --exclude ID1 ID2
```

Outputs `reports/paper_cases/candidates.{md,json}`, up to20 candidates.
Mandatory implemented filters: ours success/zeromap failure, original JSON and
schema valid,GT route≥2turns and≥2distinct MP3D regions, ours8–15high-level
steps, and the explicit exclusions. Quality is ranked by pooled mentioned
object+region categoryF1, with component scores and counts also reported; no
arbitrary F1 cutoff is invented.

GT route turns/regions are **not instruction-semantic labels**. The report
retains full instructions and lexical turn cues for manual review of implied
layout before choosing two episodes. Fewer than20 qualifying episodes are
reported honestly without relaxed thresholds.

Quality sources: `reports/map_quality/t0_val_unseen_episodes.jsonl` and
`t1_val_unseen_episodes.jsonl`; complexity source:
`data/logs/analysis/features_val_unseen.json`.

## Step3: materials, after selection

Not started before the new evaluations and instruction review. The selected
episodes will use their actual logged poses/candidates/visited nodes and action
distributions, with per-step panoramas and top-down graph overlays, the two
own trajectories/reference path, and original predicted cognitive-map channels
and headings under the explicitly assumed start registration. Outputs follow
the requested `vis/<episode>/<ours|zeromap>/step_XX_*` layout and `meta.json`.
No old D10 candidate is silently reused as a final-model case.

## Preparation checks

- Existing decision dumper inspected; model and trainer untouched.
- Both argument parsers and shell syntax checked under etpr1-py38.
- Collector synthetic tests cover exact metric agreement, forced STOP and
  incomplete/missing placeholder rejection; selection tests cover pooledF1
  and instruction-cue extraction.
- Original final-eval cache has1839/778 entries; new zero cache validated for
  both splits. No dependencies installed or environment changed.
