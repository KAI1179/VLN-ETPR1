# D8 primary-only CPU run: stopped

Command: `/home/xukai/anaconda3/envs/etpr1-py38/bin/python scripts/distill/progress_probe.py --permutations 0 --threads 1 --output reports/d8/primary`

The isolated CPU run was stopped through its own execution session (75781, Ctrl-C, exit code 130). No GPU job was started or interrupted. Its host PID was not recorded before termination; the post-stop process listing contains no `progress_probe.py` process with `--output reports/d8/primary`.

Shared-state validation passed before fitting: 1,833 episodes, 13,992 frames, identical recorded states and candidate metadata across baseline, Try5 and KD.

|Completed baseline/S readout|Value|
|---|---:|
|Accuracy|0.6488707833|
|Balanced accuracy|0.6278321967|
|Step-only balanced accuracy|0.4116037281|
|Majority balanced accuracy|0.2000000000|
|Baseline minus step-only BA|0.2162284687|
|95% scene-bootstrap CI|[0.1828925877, 0.2507489572]|

No convergence warning appeared during baseline/S fitting. Multiple `ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT` messages appeared after `Fitting try5/S: 13992 frames, 11 scenes`, before Try5/S completed. This run used zero permutations, so these warnings belong to the real-label Try5/S stage, not a permutation null fit.

Try5/S did not produce a completed result; KD/S and all O readouts were not reached. No model-to-model primary difference or CI is available from this run. Do not treat this partial run as a final comparison. Solver settings, source code and dependencies were not changed.

Preserved artifacts: `shared_validation.json` and `progress_probe_partial.json` in this directory.
