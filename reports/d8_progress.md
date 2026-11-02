# D8 — Does the KD student carry more "progress" information than the baseline?

Task book for the CLI session. GPU work is run by the user; the CLI prepares
scripts, runs everything CPU-side, and reports raw numbers first.

## 0. What we are doing and why

### Claim under test

The paper's behavioural finding so far: the KD student D (map branch zeroed
at test) gains over the ETP-R1 baseline mostly on long routes, and the
episodes it rescues are mostly ones where the baseline **stops voluntarily
more than 3 m from the goal** (116 of 155 rescued episodes; 49 of 62 in the
long-path tercile T3). We would like to write a sentence of the form

> "the student trained with the cognitive map stops prematurely less often
> on long routes, and its decision-time representation carries more
> information about how far along the route it is."

Neither half is established yet:

- The "stops prematurely less often" half rests on the rescued set only (the
  direction baseline-fail / D-success). The reverse direction and the full
  split were never counted.
- The "representation carries progress" half has no measurement at all.

D8 fills both with one CPU analysis (Part A) and one probe experiment
(Part B). Part B needs one GPU pass per model to dump features; the user
runs it.

### Design

**Part A — termination taxonomy on the full split (CPU only).**
For each of baseline / B / D on val_unseen (HEAD harness, the same three
`eval_results` directories used by `summarize_d7.py`), classify every episode
as `success`, `early_stop` (stopped voluntarily, distance_to_goal > 3 m) or
`timeout` (hit the step limit). Three classes, exhaustive. Report
rates per path-length tercile with 11-scene bootstrap CIs, and the paired
differences D − baseline, D − B, B − baseline. This is the number that
supports or kills the "fewer premature stops" sentence.

**Part B — progress probe.**
At every high-level decision of a val_unseen episode, dump the vector the
policy uses to produce its STOP logit (the fused global representation just
before the action / stop head), together with the agent position, the
stop probability and the step index. Label each frame with its **geodesic
progress**

    progress = 1 − geo(pos → goal) / geo(start → goal)

computed with the habitat PathFinder, discretised into 5 bins
[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]. Geodesic (not step count) so that
backtracking lowers progress instead of raising it.

Frame filter: keep decisions where the agent is within 1 m of the GT
polyline (the same `on_gt` rule as in the d7_case2 manifests), regardless of
whether the episode succeeds. Reason: using only SPL = 1 episodes would give
the three models different frame sets and would drop exactly the frames
before a premature STOP, which are the ones we care about.

Two readouts, cheapest first:

1. **Stop-probability curve (no training).** Mean stop probability per
   progress bin, one curve per model, with CIs. "Premature stop" shows up
   directly as a higher stop probability in bins 3–4 (0.4–0.8) for the
   baseline. This uses the same dump as the probe.
2. **Linear probe.** Multinomial logistic regression from the dumped vector
   to the 5-bin label. Leave-scenes-out cross-validation over the 11
   val_unseen scenes (train on 10, test on 1, rotate). Report accuracy and
   Spearman ρ between predicted and true bin (bins are ordered). Controls:
   (a) labels permuted within scene → chance level ≈ 0.2 and its spread;
   (b) a probe that sees only the step index t → tells whether the
   representation merely encodes "how many steps have I taken".

Reading rule, fixed before running: the probe supports the "carries
progress" sentence only if D's accuracy exceeds baseline's by more than the
spread of the permuted control **and** D also beats the step-index-only
probe. If D ≈ baseline, the sentence is dropped and only Part A's
behavioural result is kept.

What D8 cannot show: that the extra progress information comes from the
map rather than from distillation itself. That separation is the job of the
C / C0 arms (currently paused). B is included in D8 so that D − B is at
least visible.

## 1. Part A — termination taxonomy (CLI, CPU)

Inputs: the three HEAD-harness val_unseen `eval_results` directories
(baseline iter25000, B iter28000 deploy-zero, D iter17000 deploy-zero), the
per-episode files `stats_ep_ckpt_<iter>_val_unseen_r*_w*.json`, and the R2R
val_unseen episodes (for GT path length and scene).

Script: `scripts/distill/termination_taxonomy.py`

- Per episode, per model: `success` if success == 1; else `timeout` if
  `steps_taken` reached the low-level limit **or** `high_level_step` reached
  the 15-step limit (check which field the harness records; do not infer
  from distance); else `early_stop`.
- Terciles T1/T2/T3 by GT geodesic path length, the same cut points as in
  d7 (reuse the code, do not recompute them differently).
- Output `reports/d8/termination_taxonomy.md`:
  - table: model × tercile × {success, early_stop, timeout} rate, n;
  - paired deltas D − baseline, D − B, B − baseline for `early_stop` rate
    per tercile, with 11-scene bootstrap 95 % CI (reuse
    `paired_analysis.py` machinery);
  - the 2×2 transition counts baseline-class × D-class for T3.
- Sanity: the three columns sum to 1 per model and tercile; the T3 count of
  baseline `early_stop` → D `success` must equal 49 (from d7, same inputs).

## 2. Part B — feature dump (CLI prepares, user runs on GPU)

### 2.1 Locate the dump point

In the policy forward used at evaluation (R1Policy for the baseline,
PriorGTPolicy for B and D under `__deploy-zero`), find the tensor from which
the STOP logit is computed — the fused global token after cross-modal
encoding, before the action head. Confirm by reading the code that the same
tensor (same role, same dimensionality) exists in both policies; report the
file:line and the tensor name for each. If the two policies compute the STOP
logit from different tensors, stop and report; do not pick "the closest
thing".

### 2.2 Dump switch

Add an evaluation-only flag (e.g. `EVAL.dump_decision_features <dir>`),
default off, that writes per episode one `.npz` with arrays over high-level
steps:

    feat      [T, 768]  float16   the tensor from 2.1 (detached, cpu)
    pos       [T, 3]              agent position at decision time
    heading   [T]
    stop_prob [T]
    sel_id    [T]                 selected node id (-1 for STOP)
    step      [T]
    episode_id, scene_id          scalars

No model code changes beyond reading the flag and copying the tensor; the
numbers produced by the eval must stay identical (verify: SR/SPL of a
dumped run equals the existing d7 result for the same model).

### 2.3 Launcher

`scripts/submit/d8-dump.sh`: one sbatch job, 4 GPUs, runs the three models
serially (baseline 25000, B 28000 deploy-zero, D 17000 deploy-zero), HEAD
harness, `IL.back_algo teleport`, seed 100, val_unseen, dump dir
`data/logs/d8_dump/<model>/`. Skips a model whose dump dir is complete
(1839 files). Push to GitHub as a patch in the series; the user applies and
submits.

## 3. Part B — labels, curves, probe (CLI, CPU, after the dump is back)

Script: `scripts/distill/progress_probe.py`

- Load dumps; compute `progress` per frame with
  `PathFinder.geodesic_distance` on the episode's scene (one PathFinder per
  scene, cache it). Frames whose geodesic is infinite (agent off the
  navmesh floor) are dropped and counted.
- `on_gt` filter: distance from `pos` to the 3D GT polyline ≤ 1 m (same rule
  as the d7_case2 manifests). Report frame counts per model before/after
  the filter; they should be within ~20 % of each other across models.
- Readout 1: `reports/d8/stop_prob_by_progress.md` + `.png`: mean stop
  probability per bin per model, bootstrap CI over episodes.
- Readout 2: probe as in §0. `sklearn` LogisticRegression (lbfgs, C = 1,
  max_iter 1000), features standardised on the training folds only.
  Controls (a) and (b) as in §0; 20 permutations for (a).
  Output `reports/d8/progress_probe.md`: per model accuracy ± spread over
  folds, Spearman ρ, permuted accuracy mean ± sd, step-only accuracy; plus
  the per-bin confusion matrices.
- Apply the reading rule from §0 verbatim and state which case holds.

## 4. Deliverables and order

1. Part A report (CPU, can be delivered today).
2. Dump patch + launcher pushed; message the user with the apply/submit
   commands.
3. After the dump: Readout 1, then Readout 2, in one report
   `reports/d8/README.md` linking the three sub-reports, raw tables first,
   interpretation last and limited to the reading rule.

Do not add further variants (other layers, other bin counts, other
filters) unless a sanity check fails and names the reason.
