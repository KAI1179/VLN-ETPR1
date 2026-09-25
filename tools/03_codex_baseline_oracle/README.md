# tools/03_codex_baseline_oracle — task book #3: codex baseline, oracle injection, rule true-positive checks

Task book: `docs/reports/03_codex_baseline_oracle/task_book.md`. Same loop as tasks 1 and 2: written here, run
by a human on the server, results returned through git (`collect.sh`) into
`docs/reports/03_codex_baseline_oracle/server/<host>_<date>/`.

`run_task3.sh` runs every step in the task-book order (T3.0 → T3.4 → T3.5 → T3.1 → T3.2 → T3.3 → T3.6 →
T3.7a/b/c); the paid codex steps T3.2 / T3.3 are gated by `RUN_PAID=1`, T3.4 is delivered in-repo
(`docs/reports/03_codex_baseline_oracle/mip_code_map.md`), and steps whose script is not provided yet
(T3.5 `t35_apply_oracle.sh`, T3.7a `t37a_rule_roc.py`, T3.7c `t37c_backtrack.py`) record SKIP.
`py/t3x_run_report.py` summarises a MIP run archive for T3.2 / T3.3.

This README describes the **zero-cost, data-only** steps that need neither a model nor the simulator:
**T3.0** (task-2 fixes and `rules_v0.json`), **T3.6** (O-commit branch-deviation oracle calibrated on GT)
and **T3.7(b)** (synthetic wrong-branch recall), plus `py/oracles.py`, the module the simulator-side
injector (T3.5) imports.

## Run (server)

```bash
cd ~/code/agentic-nav/ctl && git pull
STEPS=T3.0,T3.6,T3.7b bash tools/03_codex_baseline_oracle/run_task3.sh   # zero-cost steps (CPU, a few minutes)
bash tools/03_codex_baseline_oracle/run_task3.sh                         # every step; T3.2 / T3.3 need RUN_PAID=1
bash tools/03_codex_baseline_oracle/collect.sh && git push -u origin claude/trusting-brown-1hqo4c
```

Task 2's outputs must exist in `$OUT` (`$WORKDIR/oracles`); nothing there is modified. Exit codes of a
python step: 0 PASS, 1 FAIL (criterion not met, outputs still written), 2 SKIP (an input is missing; partial
outputs are written and marked provisional), other = ERROR. Each step writes its fragment to
`$OUT3/md/<step>.md`; `py/assemble.py` rebuilds `$OUT3/task3_report.md` after every step.

## Steps

| step | script | inputs | outputs (`$OUT3`) | criterion |
|---|---|---|---|---|
| T3.0 | `py/t30_fixes.py` | `$OUT/oracle_progress_{rand100,val_unseen,train}.json(.gz)`, GT + episodes of rand100 / val_unseen / train | `oracle_progress_<split>.json` (re-typed), `t30_replay_rand100.json`, `clause_budget_train.json`, `rule_calibration_<split>.json`, `rules_v0.json` | rand100 replay 100 % monotone / 100 % final hit / 0 offtrack; R1 (P99), R5 (> 2.0x), R6 (1.0 m / 10) each < 5 % episode FP on `RULE_SPLIT` |
| T3.6 | `py/t36_commit_oracle.py` | `oracle_progress_<split>` ($OUT3, else $OUT), GT + episodes of `COMMIT_SPLIT` and rand100, `MP3D_DIR` | `t36_commit_<split>.json`, `rules_v0.json["commit_oracle"]` | a (θ, d) with episode FP < 2 % on `COMMIT_SPLIT` |
| T3.7b | `py/t37b_synthetic_branch.py` | rand100 oracle / GT / episodes, `$R2R_DISC/R2R_val_unseen.json`, `$CONN/<scan>_connectivity.json`, `rules_v0.json` | `t37b_synthetic_rand100.json` | none (descriptive); SKIP when an input is missing |
| — | `py/assemble.py` | `$OUT3/md/*.md`, `status.txt`, `issues.txt` | `task3_report.md` | — |

### T3.0 — what changed relative to task 2

1. **Clause-start tolerance** `CLAUSE_TOL_MODE=final_only`: intermediate clause starts are strict (0 m);
   only a final zero-length clause gets `CLAUSE_TOL_M` (0.5 m). Implemented as an opt-in knob of
   `tools/02_oracles/py/common.py::ClauseTrack(tol_mode=...)` (default `"all"` = task 2 as run).
2. **Turn keyword rule** `TURN_RULE=v1` adds `left | right` to `turn | veer | bear`; the exclusion (a move
   verb in the clause) is unchanged. Knob: `common.classify(text, turn_rule=...)` (default `"v0"`). T3.0
   re-types the clauses of task 2's oracle files from their text (the geometry is unchanged) and writes the
   re-typed copies to `$OUT3`; the type transition counts are in the report.
3. **Offtrack GT episodes**: listed with `episode_id`, max distance and first offtrack step.
4. **R6**: same clause; the position after a forward step lies within `radius_m` of a position recorded
   ≥ `gap_forward_steps` forward steps earlier; turn actions do not count. Implemented as the state machine
   `oracles.LoopDetector` (one call per action) and cross-checked against task 2's `r6_flags` on GT
   (identical there, because GT `locations` already hold one point per forward step).
5. **R5** long side only: total walked distance > factor × the train median reference arc length of the same
   clause count; factors 2.0 (chosen) and 2.5 both reported.

`rules_v0.json`:

```json
{"version": "v0", "generated": "...",
 "R1": {"level": "P99", "type": {"stop_at": 9.32, "enter": 10.31, "pass": 11.59, "turn": 7.85, "move": 11.07},
        "steps": {"stop_at": 40, ...} | null, "all_levels": {"type": {"P90": {...}, ...}, "steps": {...}}},
 "R5": {"side": "long_only", "long_factor": 2.0, "long_factors": [2.0, 2.5], "median_by_nclauses": {"1": 8.107, ...}},
 "R6": {"radius_m": 1.0, "gap_forward_steps": 10},
 "offtrack_m": 3.0, "clause_tol": {"mode": "final_only", "m": 0.5}, "turn_rule": "v1",
 "commit_oracle": {"theta_deg": ..., "advance_m": ..., "vertex_radius_m": 2.0, "turn_deg": 30.0, "ontrack_m": 1.0, "sustained": true, ...},
 "provenance": {"budget_split": "train", "calibration_split": "val_unseen", "provisional": false, "fp_episode": {...}, ...}}
```

`R1.steps` (forward-step budgets, P99 of GT forward steps per clause type on train) is `null` when the train
GT is unreadable. `provenance.provisional` is true when the calibration split's GT was unavailable and the
numbers come from rand100 (the step then exits 2).

### T3.6 — commit oracle

`oracles.CommitOracle(ref_xz, CommitParams(theta_deg, advance_m, vertex_radius_m=2.0, turn_deg=30.0,
ontrack_m=1.0, sustained=True))`, one `step(xz, yaw)` per agent action, returns True on the action that
triggers. Turning vertices: direction change > `turn_deg` between the incoming and outgoing reference
segments (zero-length segments skipped). The oracle arms when the agent enters the 2 m disc of the next
unconsumed turning vertex k; after it leaves the disc, the heading is compared with the tangent of the GT
segment k→k+1 at every action; while the deviation exceeds θ the distance moved is accumulated (reset when
the heading re-aligns, `sustained=True`; set False for the literal "advanced ≥ d since leaving" reading);
reaching d triggers once. The window closes without a trigger once the agent projects beyond vertex k+1
within `ontrack_m` of the route. Sweep θ ∈ {30, 45, 60}°, d ∈ {0.5, 1.0, 1.5} m; the tightest setting
(ascending (θ, d)) with episode FP < 2 % is written to `rules_v0.json["commit_oracle"]`.

GT heading: the GT files carry no heading but do carry the action sequence, so the yaw is replayed exactly
from `start_rotation` (quaternion [x, y, z, w] about +y, yaw = 2·atan2(y, w)) and the 15° turn actions
(`oracles.gt_steps`); the report prints the angle between this heading and the actual GT motion (median
0.00° on rand100). Forward direction = (−sin yaw, −cos yaw) in habitat (x, z).

### T3.7(b) — synthetic wrong branches

For every turning vertex of every rand100 reference path: GT prefix up to the GT location nearest the
turning viewpoint, then a different connectivity edge (`unobstructed`, `included`, not on the R2R path),
extended greedily (smallest turn) to `BRANCH_DEPTH_MAX` (3) viewpoints, interpolated at `STEP_M` (0.25 m),
yaw = direction of motion. Up to `BRANCH_PER_VERTEX` (3) branches per vertex, chosen from the neighbours
sorted by angle to the GT next segment (first / middle / last, so small- and large-angle branches are both
covered). Replays R1 (`oracles.BudgetRule`), R6 (`oracles.LoopDetector`), offtrack (`ClauseTrack.locate`)
and the commit oracle with the `rules_v0.json` parameters; reports trigger rate after the branch point,
trigger delay in steps (P50 / P90 / max), triggers on the GT prefix, and breakdowns by branch angle and depth.

## `py/oracles.py` (for the simulator-side injector, T3.5)

Depends on numpy only. Exact T3.5 strings: `PROGRESS_FMT`, `OFFTRACK_LINE`, `COMMIT_LINE`, `BRIEFING_LINE`;
helpers `progress_line(clauses, idx, dist_to_goal_m=..., progress_mask_final_zero=True, goal_radius_m=3.0)`
(with the mask on, the transition into a final zero-length clause is not announced within 3 m of the goal;
the previous clause is announced instead), `offtrack(dist_to_route_m, threshold_m=3.0)`, `commit_line(bool)`.
State machines: `CommitOracle`, `LoopDetector` (R6), `BudgetRule` (R1); heading helpers
`yaw_from_quaternion`, `forward_xz`, `yaw_from_forward`, `gt_steps`, `steps_from_polyline`, `turning_vertices`.
The clause index and the distance to the route come from task 2's `common.ClauseTrack.locate(...)` with
`tol_mode="final_only"`.

## Knobs

`WORKDIR` (`/home/xukai/code/agentic-nav`), `PY`, `OUT` (`$WORKDIR/oracles`, task 2), `OUT3` (`$WORKDIR/task3`),
`LOGDIR`, `CE_ORIG`, `FGR2R`, `R2R_DISC`, `CONN`, `RAND100` (as task 2); `CLAUSE_TOL_MODE` (`final_only`),
`CLAUSE_TOL_M` (0.5), `OFFTRACK_M` (3.0), `TURN_RULE` (`v1`), `RULE_SPLIT` / `COMMIT_SPLIT` (`val_unseen`),
`BUDGET_SPLIT` (`train`), `MP3D_DIR` (`/data/xukai/mp3d`), `BRANCH_DEPTH_MAX` (3), `BRANCH_PER_VERTEX` (3), `STEP_M` (0.25).
The task-3 knobs are read by `py/t3common.py` with these defaults, so they need not be exported;
`CLAUSE_TOL_MODE=all TURN_RULE=v0` reproduces task 2's settings. Runner-level knobs (`RUN_PAID`, `CODEX_MODEL`,
`CODEX_VERSION`, `EMBODIEDSCORE_GPU_ID`, proxies, `T33_EPISODES`) are documented in `run_task3.sh`.

## Local check (dev container, no CE val_unseen / train GT)

```bash
export WORKDIR=<scratch> OUT=docs/reports/02_oracles/server/admin123-WZ-SERVER_20260924-0423 OUT3=<scratch>/task3 \
       RAND100=<scratch>/MIP/splits/r2r/rand100 CONN=precompute_img_features/connectivity \
       R2R_DISC=<scratch>/Fine-Grained-R2R/source/R2R-original PY=<venv with numpy>/bin/python
STEPS=T3.0,T3.6,T3.7b bash tools/03_codex_baseline_oracle/run_task3.sh
```

T3.0 and T3.6 then exit 2 (provisional, calibrated on rand100 only); T3.7b runs fully.
Lint: `ruff check tools/03_codex_baseline_oracle` and
`ty check --extra-search-path tools/02_oracles/py --python-version 3.11 tools/03_codex_baseline_oracle/py`.

## Added by the orchestrating session (runner, T3.5, T3.3 export, T3.7a/c)

| step | script | notes |
|---|---|---|
| runner | `run_task3.sh` | task-book order T3.0 → T3.4 → T3.5 → T3.1 → T3.2 → T3.3 → T3.6 → T3.7a/b/c; paid steps need `RUN_PAID=1`; `CODEX_MODEL` (gpt-5.5), `CODEX_VERSION` (pin), `T33_EPISODES` (0-19), `T33_SMOKE_EPISODES` (0-2) |
| T3.4 | in-repo `docs/reports/03_codex_baseline_oracle/mip_code_map.md` | the server step only records the MIP commit |
| T3.5 | `t35_apply_oracle.sh` → `py/t35_make_arm.py` + `oracle_arm/oracle_inject.py` (+ `py/oracles.py` vendored) | builds `exp_workspace/oracleES` on MIP branch `e0-oracle` (bareES untouched), config `std_r2r_es_oracle.yaml` with the `oracle=` seat; dry-runs each condition with `+run.fake=true` and codex `api=fake`; `py/t35_unittest.py` replays rand100 GT through the injector (100/100 locally) |
| T3.3 | `py/t33_export_traj.py`, `py/t3x_run_report.py` | the 20-episode baseline runs on the oracleES arm with `oracle=none` (byte-identical tool outputs, proven in T3.5) so `live_i/poses.jsonl` exists; bareES logs no pose |
| T3.7a | `py/t37a_rule_roc.py <traj dir>` | R1/R5/R6/offtrack/commit on the exported real trajectories: TPR / FPR / delay vs first off-route step; R1 level and R6 (radius, gap) sweeps |
| T3.7c | `py/t37c_backtrack.py` | GT walk with a client-side greedy follower over MIP's env server, then reverse replay (180° turn, reversed sequence with L/R swapped, 180° turn); anchor error / heading error / blocked forwards; `T37C_EPISODES` (20), `T37C_PORT` (9231) |

Oracle injection points (oracleES bridge): `step()` result JSON gains an `ORACLE` field (one line per active hint);
`observe()` appends a top-down trajectory sketch image (spatial) and the same lines after the RGB frame; every primitive
appends `{n, x, z, yaw, dist_goal, clause, dist_route, commit, action, collided, step_count}` to `live_i/poses.jsonl`.
The briefing note is added by `oracleES/prompts.py` from the runner's environment: export `BAREES_ORACLE=<seat value>`
when launching an oracle run (`run_task3.sh` does).

Local check of the orchestrator's parts (this container has no simulator): `t35_make_arm.py` on a scratch MIP clone,
`t35_unittest.py` 100/100, `t37a_rule_roc.py` on synthetic trajectories, `t3x_run_report.py` on a task-1 summary.json.

Extra steps added after the first server run: `STEPS=T3.3s` (3-episode `oracle=all` smoke alone) and `STEPS=T3.3r`
(re-run, with `run.resume=true` into the same run, the baseline episodes that died on a provider-side error such as
"Selected model is at capacity" — MIP scores those 0 instead of excluding them; `py/t33_failed_indices.py` lists them).

Variant runs: `CODEX_MODEL=<models.yaml row> CODEX_EFFORT=<low|medium|high|xhigh> ARCHIVE_TAG=_<tag>` (e.g. `CODEX_MODEL=gpt-5.6 CODEX_EFFORT=low ARCHIVE_TAG=_gpt56low STEPS=T3.3 T33_EPISODES=0-9 T33_SMOKE=0 RUN_PAID=1`); the tag keeps the archives (`runs/t33_baseline<tag>`, `traj<tag>`) apart from the gpt-5.5 baseline. `effort=` is only passed when CODEX_EFFORT is not `default`.

`STEPS=T3.8` (image-reach probe): `py/t38_image_reach.py` runs `codex exec --json` with MIP's exact overrides against a one-tool MCP server (`py/t38_probe_bridge.py`) whose observe() returns a synthetic image with a 3-digit code; PASS when the code is read in >= 80% of `T38_TRIALS` (5). `T38_CODEX_BIN="npx @openai/codex@0.152.0"` is not supported directly (single binary expected); install that version alongside and point T38_CODEX_BIN at it.
