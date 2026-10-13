# D8 execution status

## Completed

All six feature dumps passed coverage, shape and finite-value checks.
Shared-state sets: 1,833 episodes / 13,992 decisions for each model, with
identical episode, step, position, candidate IDs/coordinates and scene.
Own rollouts reproduce all five checked original metrics exactly across
1,839 metric records per model. Try5/KD have six unrun cache-missing episodes,
which are excluded from feature comparisons.

## CPU readout: incomplete

The independent single-thread primary run completed baseline/S without
convergence warnings:

| Readout | Value |
|---|---:|
| Accuracy | 64.8871% |
| Balanced accuracy | 62.7832% |
| BA minus step-only | +21.6228 percentage points |
| 95% paired scene-bootstrap CI | [+18.2893, +25.0749] percentage points |

Try5/S real-label fits emitted `ConvergenceWarning`: LBFGS reached the
specified `max_iter=1000`. Its complete model comparison is unavailable.
The full 20-permutation process also emitted convergence warnings. Both
our CPU processes were stopped; no GPU job was stopped or altered.

No claim about improved progress decoding follows from the baseline-only
readout. No missing comparison, permutation control or STOP curve is treated
as complete. Partial output is retained in `primary/`; the feature dumps do
not need recollection.

## Proposed numerical correction — not executed

Keep the five bins, shared frames, LOSO folds, C=1, solver, scaling and
20 permutation draws unchanged. Increase only the common optimization
iteration cap (e.g. 5,000), record iterations and convergence per fold,
and refuse to label unconverged fits as final. This changes the prescribed
1,000-iteration cap, so awaits the user's decision before rerunning.

The label remains geodesic goal proximity, not instruction-stage progress.
