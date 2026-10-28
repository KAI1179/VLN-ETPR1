# Paper qualitative cases: setup and current status

## Status

The user completed all four single-GPU7 evaluations. CPU integrity checks passed
for1839+1839+778+778 actual rollouts. The four requested result JSONL tables
exist. CPU screening has completed; see [candidates.md](candidates.md) and the
reviewed selections below. No new case rendering is claimed at this stage.

Exclusions1392/92 remain **provisional** from previous D10 exports: `attn_case.pdf`
was not located and its episode IDs have not been independently confirmed.

## Completed batch results (single rank, four environments)

All SR/SPL/OSR/nDTW values below are percentages; NE is metres.

| Model | Split | Episodes | SR | SPL | NE | OSR | nDTW | Low steps | High steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ours | val_unseen |1839|67.4823|56.6115|3.7925|72.3763|65.5891|95.1175|8.7281|
| zeromap | val_unseen |1839|63.1865|53.3227|4.0551|69.2224|63.8921|97.7107|8.7656|
| ours | val_seen |778|75.9640|67.9593|2.9041|80.7198|73.3863|82.6748|8.4987|
| zeromap | val_seen |778|75.8355|68.4463|2.9441|80.7198|73.4375|77.5501|8.1542|

Every exported episode has decisions/trajectory/STOP records, unique IDs, and
six metrics exactly equal to its corresponding `stats_ep_*` record: zero
mismatches. No missing-cache zero-score placeholders were found. Source logs:
`data/logs/checkpoints/paper_case_<model>_<split>/eval_results/`.

Ours SR reproduces67.48 but SPL56.6115 differs from historical56.6311. Zeromap
is63.1865 rather than archived63.2953. The run uses one rank and executes every
episode (including absent original-map cases); it is not labelled a per-episode
exact reproduction of the historical evaluations.

## CPU shortlist and reviewed selections

Counts:222 ours-success/zero-failure →221 after provisional exclusions →220
valid JSON/schema →111 with≥2GT turns and≥2GT regions →85 with ours8–15high-level
steps. The first20 are listed in [candidates.md](candidates.md).

Selected for material preparation: **456 and1043**, after reading the full
instructions rather than treating GT region/turn counts as text labels.

| Episode | Mentioned F1 pooled/object/region | GT turns/regions | Steps ours/zero | Ours SPL/NE | Zero SPL/NE |
|---|---|---|---|---|---|
|456|1/1/1|4/3|11/8|0.5590/0.901m|0/18.486m|
|1043|1/0/1|3/5|11/13|0.9046/2.913m|0/4.338m|

Episode1043's object-only F1 is zero because its mentioned-object target count
is zero; its pooled score is supported by mentioned regions. Do not present it
as evidence of object grounding.

###456: cabinet → bedroom → bathroom

> Walk down the hall leading to the cabinet. At the cabinet take a right and
> enter the bedroom. In the room take a left and enter the bathroom on the far
> left. Stop on the rug in front of the sink.

Two explicit turns, sequential room transitions, and cabinet/sink landmarks
support the intended route-layout explanation. Ours11steps versus zero8;
zero's NE18.49m makes the trajectory contrast substantial. Both models already
choose different candidates at step0 (ours g3, zero g0; IDs are model-local).

###1043: hallway → den → bathroom → exercise room

> Turn left then walk down the hallway. Then turn right and enter the den.
> Then walk through the bath room and enter the exercise room.

Two explicit turns and an ordered chain of named rooms support the implied
layout example without requiring an object-grounding claim. Ours11steps,
zero13; ours SPL0.9046. Initial choices differ (ours g3, zero g1; compare actual
coordinates/probabilities, not ID strings alone).

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

## Step3: CPU materials prepared; GPU rendering pending

`vis/456/meta.json` and `vis/1043/meta.json` are prepared from the actual new
own-rollout records. All per-step poses, visited/ghost nodes, chosen nodes,
action distributions, metrics, full trajectories, original maps and separate
zeromap all-zero input arrays are preserved. Region/object/heading PNGs are
also prepared. Episode456 has11ours/8zero frames;1043 has11ours/13zero frames.

To preserve the exact batch trajectories, material collection renders the
recorded poses rather than rerunning the policy at a different rank layout.
This is a material-collection implementation choice, not a new episode eval.
No old D10 candidate is silently reused as a final-model case.

User-run rendering command (only GPU/simulator stage):

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/paper-case-mpl /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/paper_cases/materials.py --episodes 456 1043 --render
```

This uses the existing d7_case2 renderer, one simulator/case at a time. It writes
`vis/<episode>/<ours|zeromap>/step_XX_{pano,topdown,fpv}.png`, native0.02m
top-down crops and `cogmap_{regions,objects,heading}_projected.png`, then marks
the case's `meta.json` complete. Panoramas contain12overlapping90°views, not
equirectangular images. Map projections use assumed translation-only start
registration, not an assertion that predicted room positions are physically
correct. Top-down node panels label all visited/candidate/selected/goal nodes,
with a faint full trajectory and a separately labelled high-level prefix.
The heading rays are numbered directions anchored at the start, not invented
intermediate positions. GPU rendering has **not** been launched by the CLI.

## Expanded top20 rendering: both splits

Prepared40cases: the first20 entries of `candidates.json` (val_unseen) and the
first20 entries of `val_seen/candidates.json`. The lists are loaded from the
existing candidate tables, not resampled. Across both arms this is835decision
frames:439val_unseen and396val_seen. CPU manifests, map inputs and semantic
plots are present for all40. Four prior cases are already marked complete;
the remaining36 await the user's GPU rendering command.

```bash
GPU=7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/analysis/paper_cases/render_top20.sh
```

The foreground job runs one simulator/case at a time, skips cases marked
complete, and reuses existing RGB images on partial reruns. Errors still stop
the job; no failed case is silently counted as complete. It does not load any
navigation policy or change training/evaluation jobs.

Outputs retain existing split-specific locations:

- `vis/<episode>/` for val_unseen.
- `vis/val_seen/<episode>/` for val_seen.
- `vis/top20_manifest.json` lists40cases, source ranks/tables, frame counts,
  directories and completion status. Status is refreshed after a successful
  batch; each case's `meta.json` is updated immediately when that case finishes.

The top20 list is numerical screening, not a claim that every instruction has
the same implied-layout richness; retain the manual instruction review when
choosing the final paper panels.

## Preparation checks

- Existing decision dumper inspected; model and trainer untouched.
- Both argument parsers and shell syntax checked under etpr1-py38.
- Collector synthetic tests cover exact metric agreement, forced STOP and
  incomplete/missing placeholder rejection; selection tests cover pooledF1
  and instruction-cue extraction.
- Original final-eval cache has1839/778 entries; new zero cache validated for
  both splits. No dependencies installed or environment changed.
