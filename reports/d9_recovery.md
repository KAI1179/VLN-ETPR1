# D9 exact-layout recovery

## Confirmed interruption

The two-physical-rank, one-environment run stopped at baseline shared-state
episode 1423, step 5. Its 434 completed episodes (3,336 frames) match D8
exactly in position, heading, candidate IDs/coordinates and executed actions.
No partial data are promoted to completed output.

A targeted one-episode reproduction saved:

`data/logs/d9_diagnostic_1423/base/S/replay_mismatch_step5.json`.

At the failing step:

- Agent position, heading and candidate IDs are exactly equal to D8.
- Heading in both runs is 1.5707963267948966 radians.
- Ghost g13 differs by `[-0.12431524467091393, 0, 0.01306598417306537]` m.
- g13 is the average of two raw waypoint positions. The delta is consistent
  with a 0.25 m distance-bin change for the 6-degree candidate, averaged over
  those two positions. This is not a rounding-tolerance issue.
- The preceding five recorded frames match D8 exactly.

The original D8 layout was four logical ranks with four environments each.
Reducing that to two ranks with one environment changes visual-inference
batch composition, episode assignment and environment histories. Restoring
the original layout is the next direct test of this explanation; a source
inspection alone does not establish which GPU operation changed the bin.

## Recovery design

Run each **original logical rank** as a separate one-GPU process, with its
original complete episode shard and **four environments**. Run logical ranks
0/1 on physical GPUs 4/5, then 2/3 on the same GPUs. There is never more than
one model inference process on either GPU. All six model/source combinations
use this layout. Existing GPU jobs are neither stopped nor reconfigured.

`scripts/eval_d9/run_shard.py` changes only the eval dataset shard returned by
`BaseVLNCETrainer.collect_val_traj`: after the original routine populates
`gt_data`, it returns `list(gt_data)[logical_rank::4]`. It does not alter model
weights, predictions, graph geometry, action selection, seeds or controller.
The original eval seed is 100 for all ranks; training's rank-offset seed code
is not called by eval. Preserving complete shards and environment count also
preserves scene assignment, batch peers and episode reset histories.

Shared actions still come from the frozen D8 baseline dumps. Positions,
headings, candidates and executed actions are checked exactly. No reference
coordinates are copied into the newly constructed graph; no tolerance is
relaxed and no mismatching episode is skipped. Per-rank completion markers
are created only after validation of every expected episode in that shard.

## Files and operation

- `scripts/eval_d9/run_shard.py`: original logical partition and CPU verifier.
- `scripts/eval_d9/dump_shard.sh`: original eval arguments, one GPU / four envs.
- `scripts/eval_d9/run_reference_layout.sh`: two-GPU unattended serial-model
  pipeline, validation and final CPU statistics.
- `scripts/eval_d9/attention_dump.py`: mismatch diagnostics and exact heading
  check, in addition to the existing exact state checks.
- Two tiny eval-only trainer calls provide diagnostic pose/waypoint context.

```bash
GPUS=4,5 bash /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d9/run_reference_layout.sh
```

New dumps: `data/logs/d9_reference_layout/<model>/<S|O>/`.
The failed `data/logs/d9_dump/` and diagnostic outputs are retained unchanged.
Runtime state: `reports/d9/status.json`.
Recovery driver log: `reports/d9/reference_layout.log`.
Final tables/index remain `reports/d9/stats.md` and `reports/d9/README.md`.

The three models' existing four-episode hook equivalence checks all passed.
This recovery adds no repeated smoke loop; its full reference-layout replay
is the strict validation of the actual collection states.

## Status

The recovered full logical-rank collection has now passed **all 15 frames of
episode 1423** with exact equality of position, heading, candidates and
executed actions to D8. Restoring the original batching repairs the previously
failing case without modifying graph coordinates or tolerances. The full
split remains under validation. Initial measured additional memory is about
4–5 GiB per physical GPU, with approximately 6–7 GiB still free.

A CPU audit using the real 25 shared smoke frames per model passed geodesic
labelling, A1–A4 aggregation and strict JSON serialization. For A4, STOP-query
"other" consistently excludes agent ∪ route in **both** groups; no candidate
region is defined for a nonspatial STOP query. A3 selected-query geometry is
unchanged. If a float16 head has zero retained word mass, its word statistics
and the corresponding head spread stay NA; summary counts expose how many
frames are defined rather than filling them with zero.

This is a recovery/protocol record, not a completed result report. The final
tables are written only after all full dumps pass validation.
