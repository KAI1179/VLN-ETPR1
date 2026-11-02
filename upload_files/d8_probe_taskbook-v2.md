# D8-B — Goal-proximity probe on the STOP-head input

Task book for the CLI. CPU only; all features are already on disk
(`data/logs/d8_dump/{base,try5,kd}/{S,O}/`). No GPU job, no model code
change. Framework: baseline (no map) / Try5 (LLM cognitive map, no KD) /
KD FULL (LLM cognitive map + KD), each evaluated in its own normal
inference setting.

## 1. Purpose

Part A showed that the two map-equipped models stop before reaching the
goal less often than the baseline on medium and long routes (KD − baseline
T3: −4.43 pp, CI [−7.19, −1.82]; Try5 − baseline T3: −3.44 pp,
CI [−7.23, −0.36]). D8-B asks whether this behavioural difference is
accompanied by a difference in what the decision representation encodes:

> Under identical decision states, does the STOP-head input of a
> map-equipped model carry more linearly decodable information about how
> close the agent is to the goal than the baseline's? Does KD add to that?

A positive answer links "stops prematurely less often" to "knows better how
far it still has to go". A negative answer leaves Part A as a behavioural
finding only.

## 2. Design rationale

- **Probe the STOP-head input, not an arbitrary layer.** The quantity of
  interest is what the STOP decision is computed from; that is
  `global_sap_head`'s input row 0 (`fusion_input[:, 0, :]`, 1536-d). One
  tensor, same role and dimension in all three policies.
- **Compare on shared states.** Each model's own rollout visits different
  states, and a model that navigates better sees easier states; a probe
  advantage there would be confounded. The main comparison therefore uses
  the S dumps: one expert-driven action sequence, replayed for all three
  models, each encoding the same observations and candidates with its own
  weights. 1833 episodes / 13992 frames, verified identical across models.
- **Label = geodesic goal proximity**, p = 1 − geo(pos→goal)/geo(start→goal),
  5 bins. Geodesic so that backtracking lowers p. The label is proximity to
  the goal, not instruction-stage completion; wording must respect that.
- **Linear probe with identical capacity** for every model, so the
  comparison is about the representation, not the probe.
- **Controls that remove the trivial explanations**: step-index-only probe
  (is it just "how many steps have I taken"?), majority class (class
  imbalance), label permutation (leakage / overfitting check).
- **Paired, scene-clustered statistics on the same test frames**, because
  the 11 val_unseen scenes are the real unit of generalisation and the
  three models are evaluated on identical frames.
- **Own-rollout (O) results are secondary**: they describe each model in
  its own operating regime but cannot separate representation from state
  difficulty.

## 3. Protocol

### 3.1 Data
- S dumps for base / try5 / kd: all 13992 frames, no on-reference filter
  (the 1118 off-reference frames stay in; they are labelled by geodesic
  proximity like any other frame).
- O dumps: frames within 1 m of the reference polyline (11445 / 11463 /
  11579 frames).
- Labels as already implemented in `scripts/distill/progress_probe.py`:
  p < 0 → bin 0 (counted), infinite geodesic → dropped (counted), invalid
  start distance → error.

### 3.2 Probe
- `sklearn.linear_model.LogisticRegression`, multinomial, `lbfgs`, C = 1,
  **`max_iter = 5000`** (raised from 1000; the only change), `StandardScaler`
  fitted on the training folds only.
- Leave-one-scene-out over the 11 scenes. For S the test frames of every
  fold are identical across the three models by construction.
- Record per fold: `n_iter_`, converged (yes/no), accuracy, balanced
  accuracy, Spearman ρ, confusion matrix.
- Controls on the same folds: majority class; step-only probe (inputs t
  and t/15, same regressor and cap); label permutation within scene, 20
  draws, feature probe only.

### 3.3 Statistics
- Out-of-fold predictions pooled; point estimates = pooled accuracy / BA /
  Spearman.
- Paired differences, 2000 scene-cluster bootstrap draws over the 11
  scenes, seed 0, fixed out-of-fold predictions (no refitting inside the
  bootstrap; state this in the report):
  KD − baseline, KD − Try5, Try5 − baseline, and each model − its
  step-only control. Report BA differences as primary, accuracy
  differences alongside.
- Permutation: mean ± sd of BA over the 20 draws, per model; used only to
  check the real-label BA is far outside it.

### 3.4 STOP curves
- Mean STOP probability per proximity bin, frames with actual geodesic
  distance to goal > 3 m, 2000 episode-bootstrap draws for CIs.
- On S (shared states, main) and on O (own states, secondary).
- Bins 0–3 plotted; bin 4 reported with its n but not plotted.

### 3.5 Reading rule (fixed before running)
Three independent comparisons on S, each judged on its own BA difference
with 95 % scene-bootstrap CI; a non-significant comparison means only that
no difference was detected there and says nothing about the other two:

| Comparison | If CI lower bound > 0, write |
|---|---|
| Try5 − baseline | the Try5 representation decodes goal proximity more accurately than the baseline's |
| KD − baseline | the KD representation decodes goal proximity more accurately than the baseline's |
| KD − Try5 | KD adds a further improvement over the map alone |

Each model − its step-only control must also have CI lower bound > 0 for
that model's absolute statement to be made.

Wording: results support "map-equipped models encode goal proximity more
accurately"; they do not separate training-time from test-time use of the
map, and the report does not attempt that separation.

Provisional results: a model whose folds did not all converge at
`max_iter = 5000` is reported with its numbers marked **provisional** and
is excluded from the final reading; the cap is not raised again.

## 4. CLI tasks, in order

1. **Re-run the S probes with `max_iter = 5000`** for base, try5, kd —
   including base, so all three share the cap. Isolated output directory
   `reports/d8/probe_s/`. Save per-fold `n_iter_` and convergence flags.
   If any fold still fails to converge at 5000, report which (model,
   fold), mark that model's numbers provisional, and do not raise the cap
   again.
2. **Step-only and majority-class controls** on the same folds; write them
   into the same JSON.
3. **Paired BA / accuracy differences** per §3.3; table
   `reports/d8/probe_s/paired_differences.md`.
4. **STOP curves on S**, three models side by side, table + one figure
   (`stop_curve_S.png`, bins 0–3, CI bands).
5. **Permutation control**, 20 draws × 3 models. Parallelism capped: at
   most 4 worker processes, each with `OMP_NUM_THREADS=1`,
   `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; check free memory before
   launching and keep total resident memory under half of the machine's
   RAM. Table of permuted BA mean ± sd next to real BA.
6. **O-side**: probes and STOP curves for the three models, same settings,
   `reports/d8/probe_o/`, labelled secondary.
7. **Final report `reports/d8/README.md`**: §3.5 applied verbatim —
   three comparisons judged separately, provisional numbers kept out of
   the reading — raw tables first, interpretation last. Record the commit hash of
   `progress_probe.py` used.

Constraints: no change to features, labels, bins, folds, C, solver or
standardisation; no additional layers, probe families or filters. Any
sanity failure is reported with its cause, not patched silently. No GPU
job is started.
