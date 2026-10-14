# D8-B v2 execution

Authorized task book: `upload_files/d8_probe_taskbook-v2.md`.

CPU-only: existing features are reused; no GPU inference, training or feature
changes. The logistic-regression iteration cap is 5,000 for every model and
control. Labels, five bins, LOSO folds, C=1, LBFGS and train-fold-only scaling
are unchanged. Unconverged folds remain visible and make the affected
interpretation provisional; the cap is not increased again.

The main comparison is shared states; own-rollout results are secondary.
Primary results and curves are saved before the 20 permutation draws.
Folds use at most four worker processes, each with one BLAS thread. Memory
availability and a conservative working-memory budget are checked before
every pool starts. Initial host memory: about 1,031,675 MiB total and
1,011,287 MiB available; estimated fit memory is far below half of RAM.

Implementation:

- `scripts/distill/progress_probe.py`: labels, stage ordering, pooled metrics,
  paired accuracy/BA bootstrap, incremental output, execution provenance.
- `scripts/eval_d8/probe_fit.py`: bounded CPU LOSO fitting, per-fold iterations,
  convergence diagnostics and metrics; no exception suppression/fallback.
- `scripts/eval_d8/probe_report.py`: raw tables, STOP curves (bins 0–3 plotted),
  three independent prespecified comparisons and provisional exclusions.

Checks: syntax compilation and CLI help passed; one small synthetic worker
test and one synthetic report test passed. `ruff` and `ty` were not found
in the available PATH or the specified conda environment; no packages were
installed or changed.

Command:

```bash
/home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/distill/progress_probe.py --workers 4
```

Planned outputs: `reports/d8/probe_s/`, `reports/d8/probe_o/`,
`reports/d8/progress_probe_partial.json`, final `reports/d8/progress_probe.json`
and `reports/d8/README.md`. Source JSONs and per-model OOF NPZs are saved
incrementally. Reports must distinguish finished real-label fits from pending
permutation controls and secondary fits.

This setup record does not claim that the numerical experiment is complete.

## Shared-state primary readout completed

| Model | Accuracy (%) | Balanced accuracy (%) | Feature-fold convergence |
|---|---:|---:|---|
| Baseline | 64.99 | 62.90 | 11/11 |
| Try5 | 65.40 | 63.09 | 10/11; provisional |
| KD | 65.82 | 63.58 | 11/11 |

All step-only folds converge, with identical BA 41.1604% across models.
Try5's held-out `x8F5xyUWy9e` fold reaches 5,000 iterations without
convergence. It is retained, not dropped; its model comparisons are provisional.

| S comparison | BA difference (pp) | 95% scene-bootstrap CI (pp) | Reading status |
|---|---:|---|---|
| KD − baseline | +0.68 | [−0.89,+2.69] | No positive difference detected |
| KD − Try5 | +0.49 | [−0.19,+1.45] | Provisional |
| Try5 − baseline | +0.19 | [−1.26,+2.04] | Provisional |

These primary results do not meet the prespecified positive-improvement rule.
This is not proof that the representations are equivalent. The permutation
controls and O-side analysis are still pending. Primary tables and the S STOP
figure are available under `probe_s/`; no complete-task claim is made.
