# D8 — consolidated execution report

Status: **partially complete; CPU probe stopped after convergence warnings**.
This report consolidates existing artifacts only. No additional inference,
training, feature collection or probe fitting was started to write it.

## 1. Executive summary

Termination statistics, all six feature dumps and integrity checks are complete.
The three models have exactly matching shared states: 1,833 episodes and 13,992
decision frames. Own-rollout metrics are unchanged by feature recording.

Baseline's shared-state linear probe completed: balanced accuracy (BA) 62.78%,
versus 41.16% for step-only. Try5's real-label probe hit the prescribed
1,000-iteration limit and did not complete its result. KD and cross-model probe
comparisons are unavailable. Consequently, **D8 has not yet established that
Try5 or KD represents progress more accurately than the baseline**.

Baseline STOP-curve values did survive in the partial JSON, even though the final
multi-model curve document/figure was not generated. This corrects the earlier
overly broad statement that STOP curves were entirely unfinished.

## 2. Accepted protocol and model identities

The user explicitly revised the original `d8_progress_v2.md`: baseline remains
map-free; Try5 and KD use their **own maps**, not deploy-zero. All tables here
refer to that accepted, with-map protocol. Do not mix them with historical
teleport, single-rank, deploy-zero or 65.42% release results.

| Setting | Value |
|---|---|
| Dataset / split | R2R-CE / val_unseen |
| Evaluation | control; tryout=True; elevation_axis=y |
| Ranks / environments | 4 ranks × 4 environments |
| Seed | 100 |
| Instruction / high-level / low-level limits | 150 tokens / 15 decisions / 5,000 simulator steps |
| Try5/KD map architecture / source | try5 / llm_grid |
| Map key | llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree |
| Map ablation / evaluation source | none / p0 |
| Refiner / coordinate_fusion | disabled / False |
| load_pretrained_map_modules | False |
| GPU collection | GPUs 4–7; one inference job at a time |
| CPU environment | /home/xukai/anaconda3/envs/etpr1-py38/bin/python; no package changes |

| Model | Policy / trainer | DAgger checkpoint |
|---|---|---|
| Baseline | R1Policy / SS-ETP-R1; map disabled | /data/xukai/etp-r1-prior/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth |
| Try5, no KD | LLMGridTry5Policy / SS-ETP-LLM | /data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/dagger.iter28000.pth |
| KD FULL | LLMGridTry5Policy / SS-ETP-LLM | /home/xukai/code/ETP-R1-snapshot/ETP-R1/data/logs/checkpoints/kd_dual/ckpt.iter27000.pth |

Launcher initialization paths (not substitutes for the evaluated DAgger weights):

- Baseline: `/home/xukai/code/ETP-R1-snapshot/ETP-R1/pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt`.
- Try5/KD: `/data/xukai/etp-r1-snapshot/checkpoints/llm-grid-try5-r1p5/model_step_460000.pt`.

Evidence: `scripts/eval_d7/d7d_eval.sh`, `scripts/eval_d8/dump.sh`,
`reports/d8/setup.md`. The launcher uses the explicit evaluation configuration,
not the checkpoint's embedded configuration (`EVAL.USE_CKPT_CONFIG=False`).

## 3. Completion checklist

| Deliverable | Status | Artifact / qualification |
|---|---|---|
| Part A termination taxonomy | Complete | reports/d8/part_a_termination.md and .json |
| STOP-head feature logger | Complete | scripts/eval_d8/feature_dump.py; detached 1536-d float16 |
| Hook equivalence smoke | Passed | 4 episodes; five metrics unchanged |
| Six full feature dumps | Complete | data/logs/d8_dump/{base,try5,kd}/{S,O}/ |
| Shared-state all-frame alignment | Passed | 1,833 episodes; 13,992 frames |
| Own-rollout metric equivalence | Passed | 1,839 metric records per model; no changes |
| CPU geodesic labels / accounting | Complete | primary/progress_probe_partial.json |
| Baseline/S probe and controls | Complete in primary run | No convergence warning reported in this run |
| Baseline/S STOP curve numeric values | Complete in partial JSON | Final figure/document not generated |
| Try5/S real-label probe | Incomplete | Iteration-limit warnings; no completed row |
| KD/S and all O probes | Not completed | Not reached in primary run |
| 20-draw permutation control | Incomplete | Full run printed draw 1/20 for baseline before termination; no complete control output |
| Paired model-to-model BA differences | Unavailable | KD−baseline, KD−Try5, Try5−baseline not computed |
| Final multi-model STOP curves / figure | Unavailable | Only baseline/S numeric values preserved |
| Final progress-improvement conclusion | Pending | Do not infer it from Part A or baseline alone |

## 4. Part A: navigation and termination results

The own-rollout feature runs reproduce the existing D7-d evaluations. Percent
metrics are percentages; NE and PL are metres. The main scoring denominator is
1,839; six map-cache-missing episodes are scored zero, but remain **U, unrun**,
not early-stop/timeout examples.

| Model | SR | SPL | OSR | nDTW | NE | PL | Low-level steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline 25000 | 63.02 | 53.31 | 68.52 | 64.21 | 4.13 | 13.83 | 99.84 |
| Try5 28000 | 65.04 | 55.01 | 70.36 | 64.06 | 3.99 | 14.03 | 99.48 |
| KD FULL 27000 | 67.26 | 56.48 | 72.10 | 65.41 | 3.78 | 13.54 | 94.71 |

Termination categories:

- A: success.
- B: failure after having reached goal range.
- C: never reached goal range, termination=stop.
- D: never reached goal range, termination=budget.
- E: never reached goal range, termination=no_candidate.
- U: missing map cache; episode not run.

True termination labels are used, not the low-level step-count proxy.
For simultaneous conditions, the recorded priority is budget → no_candidate → stop.

GT path-length terciles use full-precision cuts 8.236761794539953 and
10.376267476756617 metres, giving 612 / 612 / 615 episodes. Rates include U
in the denominator; paired differences exclude U in either member.

| Model | Group | n | A | B | C | D | E | U |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | All | 1839 | 1159 | 101 | 382 | 197 | 0 | 0 |
| Try5 | All | 1839 | 1196 | 98 | 350 | 189 | 0 | 6 |
| KD | All | 1839 | 1237 | 89 | 335 | 172 | 0 | 6 |
| Baseline | T1 | 612 | 479 | 45 | 49 | 39 | 0 | 0 |
| Try5 | T1 | 612 | 489 | 42 | 50 | 31 | 0 | 0 |
| KD | T1 | 612 | 500 | 38 | 48 | 26 | 0 | 0 |
| Baseline | T2 | 612 | 379 | 46 | 127 | 60 | 0 | 0 |
| Try5 | T2 | 612 | 403 | 39 | 116 | 53 | 0 | 1 |
| KD | T2 | 612 | 419 | 36 | 109 | 47 | 0 | 1 |
| Baseline | T3 | 615 | 301 | 10 | 206 | 98 | 0 | 0 |
| Try5 | T3 | 615 | 304 | 17 | 184 | 105 | 0 | 5 |
| KD | T3 | 615 | 318 | 15 | 178 | 99 | 0 | 5 |

All-episode C rates are 20.77% / 19.03% / 18.22% for baseline / Try5 / KD.

| Group | Comparison | Paired n | C-rate difference (pp) | 95% scene-bootstrap CI (pp) |
|---|---|---:|---:|---|
| All | KD − baseline | 1833 | −2.51 | [−4.03, −0.88] |
| All | KD − Try5 | 1833 | −0.82 | [−2.19, +0.53] |
| All | Try5 − baseline | 1833 | −1.69 | [−3.27, −0.22] |
| T1 | KD − baseline | 612 | −0.16 | [−2.08, +2.07] |
| T1 | KD − Try5 | 612 | −0.33 | [−2.09, +1.24] |
| T1 | Try5 − baseline | 612 | +0.16 | [−1.30, +1.78] |
| T2 | KD − baseline | 611 | −2.95 | [−6.05, −0.88] |
| T2 | KD − Try5 | 611 | −1.15 | [−2.76, +0.41] |
| T2 | Try5 − baseline | 611 | −1.80 | [−4.94, +0.75] |
| T3 | KD − baseline | 610 | −4.43 | [−7.19, −1.82] |
| T3 | KD − Try5 | 610 | −0.98 | [−3.71, +2.04] |
| T3 | Try5 − baseline | 610 | −3.44 | [−7.23, −0.36] |

Bootstrap: 2,000 scene-cluster draws, seed 0.

| Group | Baseline A → KD C |
|---|---:|
| All | 79 |
| T1 | 13 |
| T2 | 27 |
| T3 | 39 |

Evidence: `reports/d8/part_a_termination.md/.json`;
`reports/d7d_results_summary.md`. These are behavioral findings, not the
missing cross-model progress-probe results.

## 5. Feature collection and equivalence

### 5.1 Shared (S) versus own-rollout (O) states

S: baseline generates actions with the existing SPL/geodesic DAgger expert;
Try5/KD replay the saved candidate IDs. Each model computes its own features
on matching observations/candidates. This expert can take shortest-route
shortcuts; it is **not guaranteed to reproduce the literal reference path**.
GT drives replay/labels, not the policy forward inputs.

O: each model follows its own normal evaluation decisions. These state
distributions differ, so O comparisons are secondary/descriptive.

| Model | S episodes | S frames | O episodes with features | O frames |
|---|---:|---:|---:|---:|
| Baseline | 1833 | 13992 | 1839 | 16157 |
| Try5 | 1833 | 13992 | 1833 | 16071 |
| KD | 1833 | 13992 | 1833 | 15983 |

Each directory has a `.complete.json` marker. Checks confirm file coverage,
finite float16 features, shape [T,1536], and consecutive per-episode steps.
The six omitted map episodes are 308, 349, 455, 804, 1318, 1331.

### 5.2 Recorded schema

The read-only pre-hook captures the full STOP-head input
`global_sap_head` input row `[:,0,:]`, not just one 768-dimensional half.

| Field | Content |
|---|---|
| feat | [T,1536], float16, detached CPU copy |
| pos / heading | Agent world position / heading per high-level decision |
| stop_prob | STOP probability |
| sel_id / step | Selected candidate ID (−1=STOP) / decision index |
| candidate_count | Candidate count per step |
| candidate_ids / candidate_positions | JSON-encoded per-step candidate metadata |
| episode_id / scene_id / source | Episode, scene and S/O identifiers |

Evidence: `scripts/eval_d8/feature_dump.py`; eval-only call sites in
`vlnce_baselines/ss_trainer_ETP_R1.py` and `ss_trainer_ETP_PriorGT.py`.

### 5.3 Equivalence evidence

| Check | Coverage | Result |
|---|---|---|
| Hook off vs on smoke | 4 episodes | Five original metrics exactly unchanged |
| Shared replay smoke | 4 episodes / 25 frames / 3 models | Matching states/candidates |
| Full S alignment | 1833 episodes / 13992 frames / 3 models | All frames identical in checked fields |
| Baseline O vs original D7-d | 1839 metric records | changes=[] |
| Try5 O vs original D7-d | 1839 metric records | changes=[] |
| KD O vs original D7-d | 1839 metric records | changes=[] |

Shared fields checked: step, position, candidate count, candidate IDs,
candidate positions, scene ID. Metric fields checked: success,
oracle_success, SPL, steps_taken, distance_to_goal.

Evidence files under `/home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/d8/`:
`hook_equivalence.json`, `shared_validation.json`,
`base_own_equivalence.json`, `try5_own_equivalence.json`,
`kd_own_equivalence.json`.

An initial four-environment, single-scene smoke repeated episodes and was
stopped. The corrected one-environment smoke passed. Full-split collection
then passed its coverage checks; the failed smoke is not counted as a pass.

## 6. CPU labels and data accounting

Label: `p = 1 − geodesic(current,goal) / geodesic(start,goal)`;
five bins [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]. Negative p is
clipped into bin 0; invalid starting distance raises an error; infinite
frame distance is counted and removed. Scene navmeshes are loaded through
CPU PathFinder. The label measures **goal proximity**, not instruction-stage
completion.

S retains off-reference-path frames and reports them. O keeps only frames
within 1 m of the reference polyline. Counts below precede the additional
distance>3m restriction used only for STOP curves.

| Model/source | Raw frames | Infinite removed | Negative clipped | Off-reference frames | Probe frames retained |
|---|---:|---:|---:|---:|---:|
| Baseline/S | 13992 | 0 | 166 | 1118 | 13992 |
| Try5/S | 13992 | 0 | 166 | 1118 | 13992 |
| KD/S | 13992 | 0 | 166 | 1118 | 13992 |
| Baseline/O | 16157 | 0 | 2547 | 4712 | 11445 |
| Try5/O | 16071 | 0 | 2349 | 4608 | 11463 |
| KD/O | 15983 | 0 | 2236 | 4404 | 11579 |

S off-reference frames are 1,118/13,992 (about 8%). Therefore describe S as
shared expert replay, not as strictly on-reference-path observations.

Evidence: `reports/d8/primary/progress_probe_partial.json`, `accounting`;
`scripts/distill/progress_probe.py:88`.

## 7. Available primary probe results: baseline/S only

Multinomial logistic regression: C=1, LBFGS, max_iter=1000; StandardScaler
fitted on the training fold only. Leave-one-scene-out over 11 scenes.
Step-only input is t and t/15, not t divided by the future trajectory length.
Reported CI is a paired bootstrap over scenes with fixed out-of-fold predictions;
it does not include refitting the probe inside each bootstrap sample.

| Readout | Accuracy (%) | Balanced accuracy (%) | Spearman |
|---|---:|---:|---:|
| Baseline/S feature probe | 64.8871 | 62.7832 | 0.8983 |
| Step-only | 46.2336 | 41.1604 | 0.7045 |
| Majority class | 28.7593 | 20.0000 | Undefined (constant predictions) |

Feature BA minus step-only BA: **+21.6228 pp**, 95% scene-bootstrap CI
**[+18.2893,+25.0749] pp**. No convergence warning was observed in this
single-thread baseline/S run. This shows baseline features carry decodable
goal-proximity information; it does not compare map-enhanced models.

### Held-out scene results

| Scene | Frames | Feature accuracy (%) | Feature BA (%) | Step accuracy (%) | Step BA (%) |
|---|---:|---:|---:|---:|---:|
| 2azQ1b91cZZ | 1848 | 64.23 | 62.78 | 44.48 | 40.21 |
| 8194nk5LbLH | 258 | 63.95 | 61.51 | 44.96 | 38.25 |
| EU6Fwq7SyZv | 978 | 59.92 | 52.29 | 36.09 | 27.07 |
| QUCTc6BB5sX | 2184 | 67.54 | 66.90 | 47.57 | 44.86 |
| TbHJrupSAjP | 2129 | 64.30 | 61.44 | 52.51 | 48.21 |
| X7HyMhZNoso | 998 | 66.83 | 64.54 | 43.29 | 38.83 |
| Z6MFQCViBuw | 1044 | 66.09 | 65.24 | 39.94 | 34.64 |
| oLBMNvg9in8 | 1430 | 60.91 | 58.45 | 47.34 | 45.59 |
| pLe4wQe7qrG | 109 | 44.95 | 39.06 | 33.94 | 28.32 |
| x8F5xyUWy9e | 672 | 63.39 | 59.76 | 41.52 | 33.94 |
| zsNo4HB9uLZ | 2342 | 68.06 | 67.73 | 50.34 | 46.34 |

### Confusion matrix

Rows=true bin; columns=predicted bin.

| True bin | 0 | 1 | 2 | 3 | 4 | Total |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3469 | 514 | 39 | 2 | 0 | 4024 |
| 1 | 510 | 1530 | 504 | 37 | 7 | 2588 |
| 2 | 31 | 578 | 1283 | 593 | 99 | 2584 |
| 3 | 3 | 73 | 636 | 1257 | 664 | 2633 |
| 4 | 0 | 1 | 92 | 530 | 1540 | 2163 |

Evidence: `reports/d8/primary/progress_probe_partial.json`, `probes.base/S`.

## 8. Available STOP curve: baseline/S only

Restriction: actual geodesic distance to goal >3m. Values below are
probabilities expressed as percentages; CI uses 2,000 episode-bootstrap draws.

| Proximity bin | Frames | Mean STOP probability (%) | 95% CI (%) |
|---|---:|---:|---|
| 0 | 4024 | 0.001403 | [0.001084,0.001851] |
| 1 | 2576 | 0.211376 | [0.065643,0.405999] |
| 2 | 2315 | 3.021761 | [2.328504,3.735477] |
| 3 | 986 | 11.688300 | [9.803202,13.692624] |
| 4 | 9 | 5.773677 | [0.972805,11.813841] |

The highest bin has only nine eligible frames; its mean is not a stable
basis for a curve-shape claim. No Try5/KD STOP curve was saved by the primary
run; therefore no model-to-model STOP-tendency conclusion is available.

Evidence: `reports/d8/primary/progress_probe_partial.json`, `curves.base/S`.

## 9. Convergence issue and stopped computations

Two CPU runs were used, with isolated output directories:

1. Full run: default 20 permutations, four CPU threads. It printed baseline
   permutation 1/20 but did not finish the control or write a completed
   final model table. Convergence warnings appeared. Stopped own PID 563146
   with SIGTERM; execution session exited 143.
2. Primary-only run: zero permutations, one CPU thread,
   `--output reports/d8/primary`. Baseline/S completed. Multiple warnings
   appeared after `Fitting try5/S` and before completion. Because this run
   had zero permutations, these are **real-label fitting warnings**, not
   shuffled-label warnings. Stopped its own execution session 75781 with
   Ctrl-C, exit 130. Its host PID was not recorded before stopping.

Warning: `ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT`.

This identifies the probe optimizer's prescribed iteration limit as an
unresolved numerical issue. It does not establish a navigation-model bug,
corrupt features or failed GPU evaluation. The underlying cause of slow
convergence has not been diagnosed. Do not invent final Try5/KD numbers or
silently suppress the warning. No GPU process was stopped by these actions.

A proposed, **unexecuted** correction is to keep C, labels, folds, solver,
features and 20 permutation draws fixed, increase the common iteration cap,
and record convergence/iterations for every fold. Raising the cap to 5,000
is a proposal, not a guarantee of convergence. No such code change or rerun
has been made; user approval is pending.

## 10. Artifact inventory and provenance

Repository root: `/home/xukai/code/ETP-R1-snapshot/ETP-R1`.
All relative paths below are relative to that root.

| Artifact | Path |
|---|---|
| Original / revised taskbooks | reports/d8_progress.md; reports/d8_progress_v2.md |
| Design review / accepted actual settings | reports/d8_progress_review.md; reports/d8/setup.md |
| Termination results | reports/d8/part_a_termination.md; reports/d8/part_a_termination.json |
| Feature NPZs | data/logs/d8_dump/{base,try5,kd}/{S,O}/<episode_id>.npz |
| Dump completion markers | Each of those six directories: .complete.json |
| Evaluation metrics / logs | data/logs/checkpoints/d8_<model>_<S_or_O>_control_val_unseen/ |
| Original matching metrics | data/logs/checkpoints/d7d_<model>_control_val_unseen/eval_results/ |
| Validation evidence | reports/d8/*equivalence.json; reports/d8/shared_validation.json |
| Completed primary values / accounting | reports/d8/primary/progress_probe_partial.json |
| Primary stopping record | reports/d8/primary/status.md |
| Short status | reports/d8/README.md |
| CPU labeling / probe script | scripts/distill/progress_probe.py |
| CPU taxonomy script | scripts/distill/termination_taxonomy.py |
| Feature hook | scripts/eval_d8/feature_dump.py |
| Collection / verification scripts | scripts/eval_d8/{dump.sh,run_all.sh,verify.py,validate_features.py} |
| Slurm wrapper | scripts/submit/d8-dump.sh |

Relevant implementation/report commits: da56f02 (D8 logger/probe);
446d70c (collection integrity note); daf497e (convergence stopping record).
HEAD at report preparation: `daf497e378070aad6b6f2aa64e4e2cb98575d43c`.
The working tree includes unrelated changes; this HEAD hash alone is not
a claim that every evaluation ran from a clean tree. The evidence for
unchanged outputs is the explicit hook and full-rollout equivalence checks.

Commands already used (record only; **not executed by this report**):

```bash
GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d8/run_all.sh
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/progress_probe.py
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/progress_probe.py --permutations 0 --threads 1 --output /home/xukai/code/ETP-R1-snapshot/ETP-R1/reports/d8/primary
```

## 11. Remaining work and current conclusion

1. Resolve/report per-fold convergence, without changing probe capacity or
   the comparison data by silent fallback.
2. Complete real-label S probes for Try5/KD and paired BA differences.
3. Complete the 20-draw permutation control, remaining STOP curves, and
   secondary O probes.
4. Generate final tables/figures and assess the pre-agreed reading rule:
   shared-state KD−baseline and KD−step-only CI lower bounds >0; KD−Try5
   determines whether the comparison supports a KD-specific improvement.

The existing data supports reporting fewer stop-before-reaching failures
for map-enhanced runs and decodable proximity in baseline representations.
It **does not yet answer the intended cross-model progress-accuracy question**.
All collected features are preserved; no repeat GPU collection is presently
required to continue the CPU analysis.
