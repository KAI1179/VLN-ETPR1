# D8 v2 — Goal-proximity information in the decision representation

Revision of `d8_progress.md` after `reports/d8_progress_review.md`. The
review's corrections are accepted; this version replaces the original. The
CLI writes all scripts and runs everything CPU-side; GPU jobs are submitted
by the user.

## 0. What changed and why

| Review point | Decision |
|---|---|
| Models / protocol out of date | baseline iter25000, Try5 no-KD iter28000, KD FULL iter27000 — the paper's final triple. B and D are probed and classified under the **deploy-zero** setting (grid and direction_vectors zeroed, start pose kept), not control, because the claim is that the map acts only at training time. The control-setting counts already computed stay in the review as reference and do not enter D8. |
| STOP feature is 1536-d | Dump the full `global_sap_head` input row 0: `fusion_input[:, 0, :]`, `[1536]`, via a read-only forward pre-hook. |
| Own-rollout states differ in difficulty | Main probe on a **shared state set**: teacher-forced replay of the GT expert trajectory (§2.2). Own-rollout curves are kept as behavioural description only. |
| Label is goal proximity, not instruction progress | Name it `geodesic goal proximity`; results are never phrased as "understands which instruction stage it is in". |
| Out-of-range labels | proximity < 0 → bin 0, counted; infinite geodesic → dropped, counted; invalid start distance → report episode and stop. |
| Significance | Paired differences on the same test frames with 11-scene bootstrap CI; majority-class baseline and balanced accuracy reported; permutation is a leakage check only. |
| Premature-stop curve | Bins are conditioned on actual geodesic distance > 3 m. |
| Part A classes | Reuse D7-d's A/B/C/D/E/U; premature stop = C; U reported separately and excluded from paired comparisons. |
| Old constants ("49", iter17000, teleport) | Removed. Consistency check is against the D7-d deploy-zero termination counts. |

The claim D8 can support, if the numbers come out: *under identical
decision states, the KD student's STOP-head input carries more linearly
decodable goal-proximity information than the baseline's (and than the
no-KD Try5 student's).* It cannot show that this information drives STOP,
nor that it comes from the LLM map rather than from distillation; those
remain with the C / C0 arms.

## 1. Part A — termination taxonomy, deploy-zero (CPU)

Inputs: the deploy-zero val_unseen `eval_results` for the three models (if
the deploy-zero evals for Try5 28000 and KD FULL 27000 do not exist yet under
the HEAD harness, list what is missing and prepare the eval launcher; do not
substitute control-setting runs).

- Classify every episode with the D7-d A/B/C/D/E/U rule, unchanged.
- Table: model × {all, T1, T2, T3} × class rate and n. Terciles by the
  original R2R GT path length, cut points 8.2367618 / 10.3762675 m.
- Paired C-rate differences KD − baseline, KD − Try5, Try5 − baseline per
  tercile with 11-scene bootstrap 95 % CI; episodes with U in either model
  of a pair are excluded from that pair.
- Also the reverse loss count: baseline A → KD C, per tercile.
- Output `reports/d8/part_a_termination.md`. Raw tables first.

## 2. Part B — feature dump (CLI prepares; user runs)

### 2.1 Hook

Read-only forward pre-hook on `global_sap_head` in both policies
(`ETP_R1_vilmodel_cmt.py:982–983`, `etp_prior_gt/vilmodel_cmt.py:1025–1027`);
copy `input[0][:, 0, :]` detached to CPU as float16. No change to forward
returns, model structure or action logic. Eval-only flag
`EVAL.dump_decision_features <dir>`, default off.

Equivalence smoke test, mandatory before the full dump: one model, one
scene, hook on vs off → per-episode success/spl/distance_to_goal identical.
Report the diff (must be empty).

Per episode one `.npz`:

    feat      [T, 1536] float16
    pos       [T, 3]
    heading   [T]
    stop_prob [T]
    sel_id    [T]          (-1 = STOP)
    step      [T]
    episode_id, scene_id

### 2.2 Two state sources

**(S) shared states — main experiment.** Teacher-forced replay of the GT
expert path: at every high-level step the executed action is the expert's
next waypoint (the existing DAgger expert / `IL` teacher-forcing path in the
trainer, not a new mechanism). All three models see the same observation
sequence, build the same candidate graph (the waypoint predictor is shared),
and encode it with their own weights. The GT path is used only to drive the
replay and to label frames; it is not a policy input. Report for one
episode that the three dumps have identical `pos`, `step` and candidate
counts per step.

**(O) own rollouts — behavioural description.** Normal deploy-zero
evaluation with the hook on, same harness/seed/rank as the Part A evals.

### 2.3 Launcher

`scripts/submit/d8-dump.sh`: one sbatch job, 4 GPUs, serial over
{baseline, Try5, KD} × {S, O}; dump dirs
`data/logs/d8_dump/<model>/<S|O>/`; skips complete dirs. For the O runs
the eval metrics written alongside must reproduce the Part A inputs
exactly (same ckpt, harness, seed, rank, deploy-zero). Push as a patch in
the series and give the user the apply + sbatch commands.

## 3. Part B — labels, curves, probe (CPU, after the dump)

`scripts/distill/progress_probe.py`

Labels: `proximity = 1 − geo(pos→goal) / geo(start→goal)` with one cached
PathFinder per scene; 5 bins [0,.2) … [.8,1]; the out-of-range rules of §0.
For S frames also record `on_gt` (≤ 1 m to the GT polyline); it should be
true for nearly all S frames and is a sanity check, not a filter. For O
frames apply the `on_gt` filter and report counts before/after.

**Readout 1 — STOP probability vs proximity (O frames).** Mean stop_prob
per bin per model, restricted to frames with geodesic distance to goal
> 3 m, bootstrap CI over episodes. Also the same curve on S frames (where
every model sees the same states) — that one is the cleaner comparison of
STOP tendency.

**Readout 2 — linear probe (S frames, main).**
- Multinomial logistic regression, lbfgs, C = 1, max_iter 1000,
  standardisation fit on training folds only; same capacity and
  regularisation for every model.
- Leave-one-scene-out over the 11 scenes; the test frames of a fold are
  identical for the three models by construction.
- Report per model: accuracy, balanced accuracy, Spearman ρ, per-bin
  confusion matrix, bin distribution of the labels.
- Controls, same folds: majority class; step-index-only probe (input = t
  and t / T); label permutation within scene, 20 draws (leakage check).
- Paired differences on the same test frames, 11-scene bootstrap 95 % CI:
  KD − baseline, KD − Try5, Try5 − baseline, and each model − its
  step-only control.

**Readout 2b — same probe on O frames**, reported as a secondary table
with the explicit caveat that the state distributions differ.

Reading rule, fixed now: the sentence in §0 is supported only if, on S
frames, KD − baseline and KD − step-only both have CIs excluding 0. KD −
Try5 decides whether the wording may mention distillation. Otherwise the
representational sentence is dropped and only Part A's behavioural numbers
are kept.

## 4. Deliverables and order

1. Part A (or the list of missing deploy-zero evals plus their launcher).
2. Hook + smoke-test patch, launcher patch, pushed; commands to the user.
3. After the dumps: Readout 1 → 2 → 2b in `reports/d8/README.md`, raw
   tables first, interpretation last and limited to the reading rule.

No additional layers, bin counts, filters or probe families unless a sanity
check in this book fails and the failure is named.
