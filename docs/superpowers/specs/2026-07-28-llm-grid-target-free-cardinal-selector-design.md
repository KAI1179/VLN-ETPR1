# LLM-Grid Target-Free Cardinal Selector Design

## Status

Accepted on 2026-07-28 under the user's delegated spec-approval authority.
This document freezes P2 before the P3 test result is calculated.

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

## Coordinate and Hypothesis Contract

For each valid prediction, construct four hypotheses with the existing padded
nearest-neighbour warp in `prior.analyze.llm_grid_registration`. Rotate the
complete 37-channel prediction around the scale-2 continuous start pivot.
Rotate the LLM-predicted `direction_vectors` consistently with the grid.

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
select identity.

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
- development ties and test per-episode selector margins where defined;
- per-candidate wall time and exact materialized array bytes.

Angle agreement with the oracle is descriptive, not a primary endpoint,
because oracle labels can tie or have negligible margins.

## Statistical Contract

Use one shared paired scene-cluster bootstrap:

- cluster: the 11 `val_unseen` scenes;
- repetitions: 10,000;
- seed: 42;
- sampling: draw 11 scenes with replacement;
- weighting: preserve episode multiplicity by summing episode outcomes over
  sampled scenes and dividing by the summed episode count;
- interval: 2.5th and 97.5th percentiles.

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
passes conditions 1–4 and 6 above against identity, but the heading selector
does not satisfy condition 5. The honest action is then to test one pre-rotated
map cache in a separately designed navigation ablation, not to build a dynamic
selector.

### Partial Evidence and No Go

Classify the result as **PARTIAL** when a candidate has positive evidence but
misses a robustness, semantic-family, or selector-over-global condition.
Classify it as **NO GO** when neither primary nor global correction satisfies
the mean-gain and positive-CI conditions.

The direct selector and soft aggregator cannot be promoted after seeing
`val_unseen`. A promising descriptive result may authorise only a separately
preregistered follow-up.

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
