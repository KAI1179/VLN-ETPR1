# tools/02_oracles — task book #2: O-progress oracle and contradiction-rule calibration

No model API, no training. Only T2.0 starts the simulator (one fake episode on one GPU, a few hundred MB).
Written here, run by a human on the server, results returned through git — same loop as `tools/01_mip_task1`.

## Run (server, inside tmux)

```bash
cd ~/code/agentic-nav/ctl && git pull
EMBODIEDSCORE_GPU_ID=<GPU with the most free memory> bash tools/02_oracles/run_task2.sh
bash tools/02_oracles/collect.sh && git push -u origin claude/trusting-brown-1hqo4c
```

CPU only (skip T2.0): `STEPS=T2.1,T2.2,T2.3,T2.4,T2.5,T2.6 bash tools/02_oracles/run_task2.sh`.
Any step can be rerun alone (`STEPS=T2.5 ...`); the report `$OUT/task2_report.md` is rebuilt after every step.

Defaults follow the task book (`WORKDIR=/home/xukai/code/agentic-nav`, `OUT=$WORKDIR/oracles`,
`CE_ORIG=/data/xukai/data/scene_datasets/datasets/R2R_VLNCE_v1-3_preprocessed`, FGR2R and R2R from the
task-1 clone, `CONN=/data/xukai/VLN-GOAT/datasets/R2R/connectivity`); override any of them in the environment.

| step | script | output |
|---|---|---|
| T2.0 | `run_task2.sh` (relink) + `py/t20_compare.py` | split links -> `$CE_ORIG/<split>`; A7-(1) rerun, `t20/{task1,rerun}_summary.json` |
| T2.1 | `py/t21_instr_index.py` | `ce_instr_index_{val_unseen,train,rand100}.json` |
| T2.2 | `py/t22_oracle.py` | `oracle_progress_{rand100,val_unseen,train}.json` |
| T2.3 | `py/t23_replay.py` (uses `common.ClauseTrack` / `current_clause`) | `t23_replay_rand100.json` |
| T2.4 | `py/t24_budget.py` | `clause_budget_train.json` |
| T2.5 | `py/t25_rules.py` | `rule_calibration_val_unseen.json` |
| T2.6 | `py/t26_known_failures.py` | report section only; SKIP when no recorded non-scripted runs with positions exist |

Conventions: `mp3d(x, y, z) = (hab_x, -hab_z, hab_y)`; every distance is horizontal (habitat x-z plane).

One deliberate deviation, reported in T2.3 and in the report's last section: the clause-start test uses an
arc-length tolerance `CLAUSE_TOL_M` (default 0.5 m). With the task book's strict rule (0 m) the GT replay
reaches the final clause in only about two thirds of episodes, because GT stops 0.25–0.5 m short of the
goal, exactly where the zero-length final "stop" clause starts. T2.3 prints the sweep {0, 0.25, 0.5, 1.0};
`CLAUSE_TOL_M=0 bash ... ` reproduces the strict variant for T2.3–T2.5.

`collect.sh` copies the report, the per-step fragments, all JSON outputs (gzipped above 5 MB, left out and
listed with sha256 above 20 MB compressed) and logs to `docs/reports/02_oracles/server/<host>_<date>/`.
