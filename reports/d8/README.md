# D8 goal-proximity representation results

Baseline has no map; Try5 and KD use their normal with-map inference inputs.
S shared decision states are primary. O own rollouts are secondary and do not isolate representation from state difficulty.

## S raw results

|Model|Input|Accuracy (%)|Balanced accuracy (%)|Spearman rho|Status|
|---|---|---|---|---|---|
|base|model|64.99|62.90|0.8985|final|
|base|step_only|46.23|41.16|0.7045|final|
|base|majority|28.76|20.00|NA|final|
|try5|model|65.40|63.09|0.8990|provisional|
|try5|step_only|46.23|41.16|0.7045|provisional|
|try5|majority|28.76|20.00|NA|provisional|
|kd|model|65.82|63.58|0.9023|final|
|kd|step_only|46.23|41.16|0.7045|final|
|kd|majority|28.76|20.00|NA|final|

|Comparison|Metric|Difference (pp)|95% scene-bootstrap CI (pp)|Status|
|---|---|---|---|---|
|base-step/S|balanced_accuracy|21.74|[18.37, 25.22]|final|
|base-step/S|accuracy|18.75|[16.05, 21.66]|final|
|try5-step/S|balanced_accuracy|21.93|[18.77, 25.88]|provisional|
|try5-step/S|accuracy|19.17|[16.28, 23.03]|provisional|
|kd-step/S|balanced_accuracy|22.42|[19.44, 26.48]|final|
|kd-step/S|accuracy|19.59|[17.04, 23.38]|final|
|kd-base/S|balanced_accuracy|0.68|[-0.89, 2.69]|final|
|kd-base/S|accuracy|0.84|[-0.79, 2.96]|final|
|kd-try5/S|balanced_accuracy|0.49|[-0.19, 1.45]|provisional|
|kd-try5/S|accuracy|0.42|[-0.26, 1.33]|provisional|
|try5-base/S|balanced_accuracy|0.19|[-1.26, 2.04]|provisional|
|try5-base/S|accuracy|0.41|[-1.15, 2.39]|provisional|

## O raw results (secondary)

|Model|Input|Accuracy (%)|Balanced accuracy (%)|Spearman rho|Status|
|---|---|---|---|---|---|
|base|model|66.02|61.31|0.8838|final|
|base|step_only|39.60|27.99|0.4744|final|
|base|majority|35.88|20.00|NA|final|
|try5|model|66.67|61.66|0.8856|final|
|try5|step_only|40.36|29.16|0.4830|final|
|try5|majority|35.77|20.00|NA|final|
|kd|model|67.23|62.84|0.8948|final|
|kd|step_only|41.82|31.85|0.5109|final|
|kd|majority|35.14|20.00|NA|final|

## Detailed outputs

[S probe](probe_s/progress_probe.md) · [S paired differences](probe_s/paired_differences.md) · [S permutations](probe_s/permutation_control.md) · [S STOP curves](probe_s/stop_prob_by_progress.md)

[O probe](probe_o/progress_probe.md) · [O paired differences](probe_o/paired_differences.md) · [O permutations](probe_o/permutation_control.md) · [O STOP curves](probe_o/stop_prob_by_progress.md)

## Independent prespecified readings

- **try5-base**: Provisional: a feature or step-only probe did not fully converge; excluded from final interpretation.
- **kd-base**: No positive difference detected for this comparison; this says nothing about the other comparisons.
- **kd-try5**: Provisional: a feature or step-only probe did not fully converge; excluded from final interpretation.

The readout concerns geodesic goal proximity, not instruction-stage completion. These experiments do not separate training-time map assistance from test-time map use, nor establish a map-specific causal mechanism.

## Execution provenance

```json
{
  "proximity": "1-geodesic(position,goal)/geodesic(start,goal)",
  "bins": 5,
  "inputs": "base no map; try5/kd WITH MAP; control; 4 rank",
  "step_control": "t,t/15",
  "main": "shared S; all-frame metadata checked; LOSO; balanced accuracy",
  "O_filter": "distance to GT polyline <=1m",
  "curve_filter": "geodesic distance>3m",
  "S_expert": "existing SPL/geodesic expert replay, possibly shortcuts; off-GT S frames retained as diagnostic, not filtered",
  "max_iter": 5000,
  "C": 1,
  "solver": "lbfgs",
  "bootstrap": "2000 scene-cluster draws; fixed out-of-fold predictions; no refitting",
  "workers": 11,
  "worker_blas_threads": 1,
  "permutation_draws": 20,
  "script_commit": "344dd763fc2f5ab28c48cdc92f0abb088cb0148a",
  "script_sha256": "1c948e108e87031aab4b2a8955caa421fec86808f962b474f61e6b5c56e38178",
  "execution_history": [
    {
      "script_commit": "344dd763fc2f5ab28c48cdc92f0abb088cb0148a",
      "script_sha256": "1c948e108e87031aab4b2a8955caa421fec86808f962b474f61e6b5c56e38178",
      "workers": 4,
      "worker_blas_threads": 1
    },
    {
      "script_commit": "4281107e79777ba9f3eb61f433647e9e973be64c",
      "script_sha256": "5abb6cac5db9fad1eea64bc116ea2eb6810fbb8ed09fcbd8726d54d1cff8885b",
      "workers": 11,
      "worker_blas_threads": 1,
      "stage": "resume"
    }
  ],
  "resume_script_commit": "4281107e79777ba9f3eb61f433647e9e973be64c",
  "resume_script_sha256": "5abb6cac5db9fad1eea64bc116ea2eb6810fbb8ed09fcbd8726d54d1cff8885b"
}
```
