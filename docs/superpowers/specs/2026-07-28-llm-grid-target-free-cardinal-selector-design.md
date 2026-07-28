# LLM-Grid Target-Free Cardinal Selector Design

## Status

Accepted on 2026-07-28 under the user's delegated spec-approval authority.
Amended before the P3 run to schema v2 so soft aggregation reports exact
object/region endpoints instead of an unrelated proxy. This document freezes
P2 before the P3 test result is calculated.

## Purpose

The start-centred four-way oracle and semantic cross-fit controls established
that cardinal rotation can recover held-out raster overlap around the true
episode start. They did not produce a deployable correction because every
selected angle still depended on ground-truth target cells.

This experiment asks whether a deterministic correction chosen only from
runtime-available start metadata generalises from R2R `val_seen` to the sealed
R2R `val_unseen` population. It distinguishes three conclusions:

1. a heading-conditioned selector is useful;
2. one fixed global correction is sufficient;
3. neither target-free correction is supported.

The experiment also reports a direct direction-to-heading selector and a
uniform four-hypothesis soft aggregation as frozen descriptive baselines. They
cannot replace the primary candidate after the test split is opened.

## Design Decision

Three approaches were considered:

1. **Heading-calibrated hard selection — selected primary.** Choose one of
   eight signed heading-to-correction mappings on `val_seen`, freeze it, and
   apply it to `val_unseen` using only the known start heading. This is the
   smallest model that directly tests whether the LLM's cardinal error depends
   on the heading supplied in its prompt.
2. **Parameter-free direction-to-heading selection — descriptive baseline.**
   Align the first nonzero predicted route direction with the known start
   heading. This is cheap and target-free, but the first route segment need not
   equal the initial facing direction.
3. **Uniform soft aggregation — descriptive baseline.** Average the four
   start-centred rotated rasters without selecting an angle. This preserves all
   hypotheses but is expected to trade precision for recall and does not prove
   that a navigation policy can consume four separate token banks.

A learned selector and navigation-conditioned four-bank attention are
deferred. They add trainable degrees of freedom, require a separate training
contract, and cannot be validated honestly by cache-only raster IoU. A
panoramic RGB-D selector is also deferred until deployable perception exists;
ground-truth semantic observations are not an acceptable substitute.

## Non-Goals

- Do not generate new LLM predictions, train a selector, run navigation, or
  access the login node.
- Do not modify reusable registration, navigation, map-encoder, cache, dataset,
  or training code.
- Do not use target rasters, target direction vectors, oracle angles, oracle
  margins, cross-fit outputs, teacher actions, goal distances, trajectories,
  trajectory keypoints, future observations, scene identity, episode identity,
  or cache paths as selector features.
- Do not search non-cardinal angles, alternative pivots, translations, scales,
  reflections, feature sets, thresholds, channel folds, bootstrap seeds, or
  decision gates.
- Do not infer navigation benefit from offline raster overlap.

## Frozen Data and Provenance

Use the existing epoch-2 R2R prediction cache:

```text
data/llm_navigation/
  llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2/
```

The development split is exactly:

- R2R `val_seen`;
- 778 episodes from 53 scenes;
- 770 schema-valid and 8 schema-invalid-or-missing predictions;
- prediction manifest SHA-256
  `1d3eb25eb7583a2c6373f430119da25ef71472d2d065697e673e2f70c307d146`.

The sealed test split is exactly:

- R2R `val_unseen`;
- 1,839 episodes from 11 scenes;
- 1,830 schema-valid and 9 schema-invalid-or-missing predictions;
- prediction manifest SHA-256
  `31e7c5f8d186e75222b12f1aa8862b16fa9904c75221a0fd55cfba61f93fa8ce`.

Both splits use:

- cognitive-map namespace `gt.legacy.r1p5.direction5.v1`;
- scale-2 boolean rasters with shape `37×50×50`;
- object channels `[0, 27)` and region channels `[27, 37)`;
- effective cell width `1 m`;
- physical angle order `(0°, 90°, 180°, 270°)`;
- the continuous episode start position as the rotation pivot.

Every expected episode remains in the denominator. An invalid or missing
prediction is an empty raster, selects identity, and contributes zero
improvement. No repair or fallback prediction is permitted.

The known start direction must contain exactly two finite components and have
unit norm within absolute tolerance `1e-4`. Corrupt start metadata aborts the
entire run because it violates the runtime-input contract; it is not a
prediction failure. Predicted direction vectors must be finite and each row
must be unit length within `1e-4` or exact zero padding. A raw prediction that
violates that schema is handled as the already-declared invalid empty
prediction.

## Coordinate and Hypothesis Contract

For each valid prediction, construct four hypotheses with the existing padded
nearest-neighbour warp in `prior.analyze.llm_grid_registration`. Rotate the
complete 37-channel prediction around the scale-2 continuous start pivot.
Rotate the LLM-predicted `direction_vectors` consistently with the grid.
Positive `angle` means the physical grid rotation accepted by
`warp_grid_about_pivot`; the stored display-frame `[right, up]` vector is
therefore transformed numerically by `-angle`, exactly as implemented by
`rotate_direction_vectors`.

The episode `start_position` and `start_direction_vector` are external
world-frame anchors and remain fixed. The generic cognitive-map rotation helper
must not be used because it rotates around map centre and rotates the observed
start metadata as augmentation.

Scoring embeds the target in the original `50×50` frame and retains all
out-of-frame predicted support as false-positive support. No hypothesis may
improve its score by clipping support at the frame boundary.

## Allowed Selector Input

A narrow immutable selector input contains only:

- schema-valid status;
- the known start direction `[right, up]`;
- the predicted grid;
- the predicted five direction vectors.

Only the start direction is used by the primary selector. The predicted
direction vectors are used by the direct baseline. The grid is carried only
for applying the selected transform and for the uniform aggregation baseline.

Instruction text, scene ID, example ID, and paths are not needed by the frozen
candidates and therefore are excluded from the selector API. Target data is a
separate scorer input and cannot be represented by the selector type.

## Primary Heading-Calibrated Selector

Interpret a finite unit start direction `[right, up]` as a clockwise heading
from display-frame up:

```text
heading = atan2(right, up) mod 360°
```

Quantise it to the nearest member of `(0°, 90°, 180°, 270°)` by circular
distance. Exact boundary ties retain the first angle in that declared order.
Call the result `q`.

The complete candidate family contains exactly these eight mappings, evaluated
in this order:

```text
(sign=+1, offset=0°)
(sign=+1, offset=90°)
(sign=+1, offset=180°)
(sign=+1, offset=270°)
(sign=-1, offset=0°)
(sign=-1, offset=90°)
(sign=-1, offset=180°)
(sign=-1, offset=270°)
```

Each mapping emits the physical correction:

```text
angle = (sign * q + offset) mod 360°
```

On the complete `val_seen` denominator, select the mapping with the largest
episode-macro category-aware IoU after correction. An exact aggregate tie
retains the first mapping in the declared order. This single selected mapping
is the locked primary candidate; there is no per-scene or per-category
calibration.

## Frozen Baselines

### Identity

Apply `0°` to every episode. Identity is the primary quality baseline.

### Val-Seen-Calibrated Global Correction

On the complete `val_seen` denominator, choose one angle from the declared
angle order by maximum episode-macro category-aware IoU. Exact ties retain the
first angle. Apply that one angle to every `val_unseen` episode.

This is the strongest simpler calibrated baseline. If it explains the result,
the experiment must not claim that heading-conditioned selection is needed.

### Direct Direction-to-Heading Selector

For each hypothesis, rotate the predicted direction vectors consistently and
calculate cosine similarity between the first nonzero predicted vector and the
fixed known start direction. Select the maximum with declared-angle-order tie
breaking. If every predicted vector is zero or the prediction is invalid,
select identity and record `direct_margin=0`.

### Uniform Random-Angle Expectation

Report the arithmetic mean of the four per-angle episode scores. This is the
exact expectation of uniform random selection; do not sample random angles.

### Uniform Soft Aggregation

Embed all four padded boolean hypotheses in their smallest common canvas and
take their elementwise arithmetic mean. Embed the target in the corresponding
original-frame location. Score soft category-aware IoU:

```text
sum(min(prediction, target)) / sum(max(prediction, target))
```

The same soft formula is applied to the boolean identity raster for its paired
contrast. An empty union scores zero. Do not threshold, tune, OR, or take
consensus after observing either split.

The common canvas has inclusive bounds equal to the componentwise minimum and
maximum of the four padded hypothesis bounds and the target frame bounds
`rows=[0,49], columns=[0,49]`. Cells outside an individual hypothesis are zero.
Soft intersection is `sum(min(prediction, target))`; predicted and target mass
are their respective sums; soft union is predicted mass plus target mass minus
intersection. Soft precision and recall use the same intersection divided by
predicted and target mass, respectively, and are zero when their denominator
is zero. Soft F1 is zero when soft precision plus recall is zero.

### Oracle Ceiling

Report the maximum category-aware IoU among the four angles per episode.
This remains explicitly nondeployable and is used only for oracle regret and
fraction-of-oracle-gain diagnostics.

## Development Lock and Test Sealing

The full command performs these phases in order:

1. Validate both cache manifests and exact population contracts.
2. Load `val_seen`, score the eight mapping functions and four global angles,
   and freeze exactly one primary mapping and one global angle.
3. Serialize `selector_lock.json`, including the chosen definitions,
   development scores, tie outcome, and protocol constants. The manifest later
   records the lock file's SHA-256.
4. Generate every `val_unseen` candidate assignment from selector inputs
   without reading a target raster or target direction vector.
5. Serialize `test_assignments.csv` and calculate its SHA-256.
6. Only then load `val_unseen` targets and score the sealed assignments and
   frozen baselines.

The implementation must enforce phase separation with typed inputs. Tests must
show that changing target rasters or vectors cannot change assignment bytes.
Artifact validation independently regenerates the lock and assignments from
allowed inputs and requires byte equality before validating scores.

An `EpisodeKey(scene_id, example_id)` envelope associates selector inputs,
assignments, and scores and establishes stable row order. The key is never
passed to candidate logic. Every CSV is sorted lexicographically by
`(scene_id, example_id)` and then by the declared candidate or angle order.

If the official output directory already exists, the command fails before
loading development or test data. It never resumes, merges, or overwrites an
official result. Re-running requires the user to move or remove that exact
directory explicitly.

The command is fixed and is run exactly once after tests and a small synthetic
artifact smoke pass:

```text
python -m prior.analyze.d2026_07_28.llm_grid_target_free_cardinal
```

## Endpoints

The primary endpoint is the episode-level category-aware raster IoU contrast:

```text
locked heading selector - identity
```

Also report:

- locked heading selector minus calibrated global correction;
- calibrated global correction minus identity;
- direct selector minus identity;
- uniform soft aggregation minus soft identity;
- object-only and region-only contrasts;
- cell-level precision, recall, and F1, using soft intersection mass for the
  soft aggregator and ordinary counts for boolean candidates;
- oracle regret and fraction of the four-way oracle gain recovered;
- selected-angle counts and heading-bin-by-angle tables;
- direction-vector cosine as a target-scored secondary diagnostic only;
- predicted, target, in-frame, out-of-frame, and union support;
- invalid, empty-prediction, empty-target, and empty-union counts;
- development ties and the direct selector's top-minus-second cosine margin;
- deterministic warp count and maximum materialized array bytes per candidate.

Angle agreement with the oracle is descriptive, not a primary endpoint,
because oracle labels can tie or have negligible margins.

Oracle-gain recovery is calculated from aggregate episode sums:

```text
sum(candidate_iou - identity_iou) /
sum(oracle_iou - identity_iou)
```

When the denominator is nonpositive, report `0.0` and
`oracle_gain_available=false`; otherwise report the ratio and
`oracle_gain_available=true`. Do not average unstable per-episode ratios.

Direction-vector cosine uses the existing `direction_cosine` contract: average
cosine only over paired rows where both vectors are nonzero, clip each cosine
to `[-1,1]`, and return zero when no pair is eligible. Timing is excluded from
the reproducible artifact payload; only deterministic operation and array-byte
counts are recorded.

## Statistical Contract

Use one shared paired scene-cluster bootstrap:

- cluster: the 11 `val_unseen` scenes;
- repetitions: 10,000;
- seed: 42;
- sampling: draw 11 scenes with replacement;
- weighting: preserve episode multiplicity by summing episode outcomes over
  sampled scenes and dividing by the summed episode count;
- interval: `np.percentile` at 2.5 and 97.5 using its NumPy 1.24 default linear
  interpolation.

Calculate paired per-episode differences before resampling. Reuse the same
sampled scene-index matrix for every reported candidate contrast. Also report
leave-one-scene-out minimum and maximum episode-macro means for the primary
selector contrast, its contrast with the global correction, and the global
correction contrast.

Secondary baseline intervals are descriptive. They do not create additional
ways to pass the primary gate.

## Frozen Decision Gate

### Selector GO

The conclusion is **SELECTOR GO** only if every condition passes:

1. locked heading-selector mean IoU gain over identity is at least `+0.01`;
2. its paired 95% bootstrap lower bound is above zero;
3. object-only and region-only point-estimate gains are both above zero;
4. its leave-one-scene-out minimum gain over identity is above zero;
5. the paired 95% bootstrap lower bound for heading selector minus calibrated
   global correction is above zero;
6. the leakage audit and exact `1,839/11/1,830/9` denominator validation pass.

This outcome authorises a separate P4 navigation-ablation design. It does not
authorise an unreviewed navigation run or a core-code change.

### Fixed-Correction GO

The conclusion is **FIXED-CORRECTION GO** when the calibrated global correction
passes conditions 1–4 and 6 above against identity and the heading selector
does not pass all six selector conditions. The honest action is then to test
one pre-rotated map cache in a separately designed navigation ablation, not to
build a dynamic selector.

### Partial Evidence and No Go

After both GO predicates fail, classify the result as **PARTIAL** if either the
heading selector or global correction has mean gain at least `+0.01` and a
paired 95% bootstrap lower bound above zero, but misses another required
robustness, semantic-family, or selector-over-global condition. Classify every
other result as **NO GO**. These predicates are mutually exclusive.

The direct selector and soft aggregator cannot be promoted after seeing
`val_unseen`. A promising descriptive result may authorise only a separately
preregistered follow-up.

Decision precedence is exact: evaluate `SELECTOR GO`, then
`FIXED-CORRECTION GO`, then `PARTIAL`, then `NO GO`. Post-unsealing angle
scores, oracle values, target-vector cosines, direct-selector margins, and
descriptive baselines cannot alter assignments or the decision except where a
quantity is explicitly named in the frozen gate.

## Exact Artifact Transaction

Atomically publish exactly these nine files:

```text
manifest.json
development_mapping_scores.csv
selector_lock.json
test_assignments.csv
test_angle_scores.csv
test_episode_scores.csv
summary.json
bootstrap.json
target_free_control_intervals.png
```

CSV uses Python's standard `csv` writer with comma delimiter, minimal quoting,
UTF-8, and `\n` line endings. Integers use base-10 text, booleans use
`true`/`false`, and every finite float uses `format(value, ".17g")`. Negative
zero is normalised to `0`. JSON uses UTF-8, sorted keys, compact separators,
`allow_nan=False`, and one trailing newline.

The exact CSV headers are:

```text
development_mapping_scores.csv
candidate_kind,mapping_sign,heading_offset_degrees,global_angle_degrees,episode_count,valid_count,all_mean_iou,object_mean_iou,region_mean_iou,selected

test_assignments.csv
scene_id,example_id,schema_valid,heading_bin_degrees,primary_angle_degrees,global_angle_degrees,direct_angle_degrees,direct_margin

test_angle_scores.csv
scene_id,example_id,angle_degrees,all_iou,object_iou,region_iou,direction_cosine,predicted_support,target_support,in_frame_support,out_of_frame_support,union

test_episode_scores.csv
scene_id,example_id,schema_valid,heading_bin_degrees,primary_angle_degrees,global_angle_degrees,direct_angle_degrees,identity_iou,primary_iou,global_iou,direct_iou,random_expected_iou,soft_identity_iou,soft_aggregate_iou,oracle_iou,identity_object_iou,primary_object_iou,identity_region_iou,primary_region_iou,soft_identity_object_iou,soft_aggregate_object_iou,soft_identity_region_iou,soft_aggregate_region_iou,primary_predicted_support,primary_target_support,primary_in_frame_support,primary_out_of_frame_support,primary_union,soft_prediction_mass,soft_target_mass,soft_intersection_mass,soft_union_mass,primary_direction_cosine,direct_direction_cosine
```

Development rows contain the eight heading mappings followed by the four global
angles. A non-applicable integer field uses `-1`; CSV contains no empty cells.
Test angle rows contain four rows per episode in declared angle order.
Assignment and episode files contain one row per episode.

The exact top-level JSON keys are:

```text
selector_lock.json:
schema_version,chosen_mapping,chosen_global_angle,development,protocol

summary.json:
schema_version,population,means,contrasts,semantic_families,cell_metrics,
support,directions,angles,cost,oracle_gain,decision

bootstrap.json:
schema_version,seed,repetitions,scene_ids,intervals,leave_one_scene_out

manifest.json:
schema_version,sources,git_commit,protocol,population,schemas,artifacts
```

`schema_version` is the exact string
`llm-grid-target-free-cardinal-v2`. JSON array order follows the declared
candidate, angle, or sorted scene order. Nested key sets and value types are
represented by typed frozen dataclasses and are asserted exactly by tests and
the validator; unknown or missing keys fail validation. The implementation
plan must enumerate every nested dataclass field before code is written.
Changing that field list after the plan is accepted requires a design
amendment and a new schema version, not an implementation-only choice.

`manifest.json` records protocol constants, source-manifest paths and hashes,
git commit, exact populations, mappings, candidates, angle order, coordinate
contract, selector-input schema, phase-separation audit, bootstrap contract,
decision gate, row counts, and SHA-256 hashes for the other eight files.

The validator requires the exact file set, exact stable CSV headers and row
orders, finite numeric values, exact counts, canonical JSON, decodable PNG
bytes, matching hashes, recomputed selector assignments, recomputed endpoints,
recomputed bootstrap values, and a decision label derived from the frozen
gate. Any mismatch is an explicit failure. Publication uses a temporary sibling
directory and an atomic directory rename so a failed run leaves no partial
official output.

The fixed output directory is:

```text
outputs/llm_grid_analysis/
  target_free_cardinal_selector_r2r_epoch2/
```

## Architecture and Files

All implementation remains experiment-only:

```text
prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py
tests/analyze/test_llm_grid_target_free_cardinal.py
```

The module uses `tap.Tap`, typed frozen dataclasses, explicit enums, NumPy,
Matplotlib, the existing LLM-Grid loaders/parsers, and
`prior.analyze.llm_grid_registration`. It adds no dependency and exposes no new
production API. It must not import private behavior from the July 27 or
cross-fit experiment modules.

No reusable or core file is modified. If the experiment cannot be implemented
without such a change, implementation stops and reports the exact missing
contract instead of widening scope silently.

## Verification

Before the fixed run:

- synthetic unit tests cover heading quantisation, all eight mappings,
  circular and aggregate ties, invalid/zero-vector behavior, start-centred
  warps, direction rotation, common-canvas soft aggregation, support
  accounting, development locking, target-independent assignment bytes,
  bootstrap weighting, gate classification, artifact hashes, atomic
  publication, and tamper rejection;
- the full new test file passes;
- `tests/analyze` passes;
- Ruff passes on the date-scoped analysis and analysis tests;
- ty passes on the new module and test;
- `git diff --check` passes;
- a synthetic artifact directory passes full validation.

After the one fixed run, independently validate the exact artifact directory
and record the result in `docs/daily/2026-07-28.md`. Mark P2 complete and P3
complete only after the committed design and implementation, validated
artifacts, and documented finding agree.
