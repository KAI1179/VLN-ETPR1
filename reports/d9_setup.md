# D9 unattended attention experiment

## Scope and status

Protocol: `upload_files/d9_attention_taskbook-v2.md`. The user explicitly
authorised unattended execution after the design review. No training is
scheduled. Normal map inputs remain enabled for Try5 and KD FULL.

The workflow must pass its four-episode hook-on/off equivalence checks and
alignment checks against D8 before collecting full dumps. A failed check
terminates the workflow; it is not skipped or repaired by changing decisions.
GPU collection is serial. CPU statistics use single-thread BLAS, with no more
than four workers. Existing jobs are not terminated or reconfigured.

## Read-only architecture audit

- Baseline: `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py`,
  `graph_query_text` constructed at line 864 and called at line 976.
- Try5/KD: `vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py`,
  `graph_query_text` constructed at line 889 and called at line 1019.
- Each is one dedicated graph-to-text attention module, not the last of the
  graph encoder's cross-modal layers. The active head count is recorded from
  the loaded module rather than inferred from the map encoder.
- `BertOutAttention` returns the masked, pre-softmax scores alongside the
  context. A read-only output hook can recover probabilities without changing
  the context or the original forward calculation.
- `GraphMapCrossAttention` in
  `vlnce_baselines/models/etp_prior_gt/map_fusion.py` calls multihead attention
  with `need_weights=False`. An observer must not change that call's execution
  path. Map probabilities can instead be calculated separately from the
  original query/key projections and masks.
- Map fusion precedes graph-text interaction (map-policy lines 1000/1019).
  Its probabilities are upstream attention, not contributions to action scores.
- `map_encoder.py:329` flattens the convolutional output row-major:
  token `10*r+c` corresponds nominally to raster rows `[10*r,10*r+10)` and
  columns `[10*c,10*c+10)`. Metadata is appended at index 100 (line 351).
  All tokens are subsequently mixed by the token transformer. A spatial
  token is therefore not an isolated physical cell.
- `map_utils.py:94` defines the assumed origin as
  `(start_world_x - start_position[0]*meters_per_unit,
  start_world_z - start_position[1]*meters_per_unit)`.
  Raster rows follow world x and columns world z, at 0.5 m/cell. Overlays
  are labelled **assumed registration**; the LLM prediction need not be
  physically correct.

## Statistical clarifications

1. A1–A3 use paired comparisons on their explicitly common valid frames.
   A4 instead compares different high/low STOP-tendency groups by a
   scene-cluster bootstrap.
2. STOP argmax has no candidate position. Candidate-region mass is NA;
   no high-versus-low STOP candidate-region contrast is fabricated.
   The spatial "other" union on such a frame contains only agent and route.
3. Special-token filtering is never performed independently before a paired
   comparison. Undefined statistics are reported as unavailable, not zero.
4. Attention descriptions do not establish causal use, training-time
   internalisation, or necessity of the map branch.

## Resource policy

At initial inspection, all eight GPUs had existing allocations: GPUs 0–3
used about 17–18 GiB each, and GPUs 4–7 about 12 GiB each. To preserve D8's
four-rank episode partition, collection waits for GPUs 4–7 together rather
than changing to one rank. It starts only after the launcher's documented
free-memory/utilisation checks pass. No running GPU process is killed.

The unattended driver log and final status are recorded under `reports/d9/`.
This file documents setup, not completed experimental results.

## Implementation and validation

Changed core files are limited to one default-off eval flag in
`vlnce_baselines/config/default.py` and recorder lifecycle/decision calls in
`ss_trainer_ETP_R1.py` and `ss_trainer_ETP_PriorGT.py`. No model source,
checkpoint, optimizer, training configuration, or environment package was
modified. Hooks live in `scripts/eval_d9/attention_dump.py`.

CPU checks completed before launch:

- Separate map QK calculation matches PyTorch MHA's per-head probabilities
  exactly in the tested masked two-sample case (maximum absolute difference
  0); the original forward outputs remain bit-identical.
- Recorder synthetic episode round-trip passed, including the distinction
  between policy argmax and executed action and float16 attention arrays.
- All D8 reference dumps and four-rank metric files exist. The smoke list
  is episodes 1, 2, 3, and 13; all are available for every model/source.
- Shell syntax, Python syntax, CLI parsing and whitespace checks passed.
- `ruff` and `ty` are unavailable in PATH and the existing conda environment;
  no packages were installed to obtain them.

The GPU equivalence and replay checks are **pending**, not represented as
passed by these CPU checks.

## Driver and outputs

Local unattended driver: `scripts/eval_d9/run_all.sh`.
Slurm entrypoint: `scripts/submit/d9-dump.sh` (not submitted by this run).

```bash
GPUS=4,5,6,7 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d9/run_all.sh
```

The local driver is launched with `nohup`; do not launch another copy while
the existing lock is held. It performs the prescribed finite smoke checks,
then six full collection passes and CPU statistics. It never retries a
failed check or silently overwrites a partial episode dump.

- Driver log: `reports/d9/run.log`
- Driver state and PID: `reports/d9/status.json`
- Full dumps: `data/logs/d9_dump/<base|try5|kd>/<S|O>/`
- Smoke results: `reports/d9/*_smoke_equivalence.json`
- Tables: `reports/d9/stats.md`, `reports/d9/stats.json`
- Final index: `reports/d9/README.md`
- Case assets (no compositing): `reports/figs/d9_case/{302,1813}/`

Waiting time depends on the existing jobs. Do not report an inference ETA
before the GPU smoke/collection throughput is observed.
