# LLM-Grid Cross-Fitted Registration Controls Design

## Status

Approved in conversation on 2026-07-28. This document freezes the protocol
before any result is calculated.

## Purpose

The previous four-way oracle raised R2R `val_unseen` episode-macro
category-aware IoU from `0.135701` to `0.186357`, but it selected the best
angle using the same target cells that it scored. This follow-up asks two
narrower questions:

1. Does an angle selected using one disjoint semantic family improve IoU on
   the held-out semantic family?
2. Is that transfer specifically stronger around the true episode start than
   around geometrically plausible control pivots?

A passing result is evidence for shared start-centred cardinal angular
alignment across disjoint semantic channels. It is not evidence for a
deployable angle selector because the selection family still uses
ground-truth raster cells.

## Non-Goals

- Do not train, generate new LLM predictions, run navigation, or access the
  login node.
- Do not modify the reusable registration implementation or the July 27
  diagnosis.
- Do not search translations, reflections, scales, non-cardinal angles,
  channel partitions, shuffle seeds, or decision thresholds.
- Do not claim recovery of general metric geometry, direction metadata, or a
  target-free orientation estimate.
- Do not add mentioned/unmentioned or arbitrary channel-fold analyses to the
  primary experiment. Those remain future exploratory sensitivities only if
  this prespecified object/region analysis is inconclusive.

## Fixed Population and Provenance

Use only existing local artifacts:

- dataset/split: R2R `val_unseen`;
- prediction cache key:
  `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2`;
- ground-truth cognitive-map namespace:
  `gt.legacy.r1p5.direction5.v1`;
- population: exactly 1,839 episodes from exactly 11 scenes;
- expected prediction status: exactly 1,830 schema-valid and 9
  schema-invalid-or-missing rows in aggregate;
- raster shape: `37×50×50`;
- effective cell width: `1 m`;
- angle order: `(0.0, 90.0, 180.0, 270.0)`.

Invalid or missing predictions remain explicit empty predictions in the full
denominator. Angle ties retain the first declared angle, so an empty selector
chooses identity.

The output manifest records the prediction split-manifest path and SHA-256,
ground-truth namespace, git commit, population, scene count, channel folds,
pivot definitions, shuffle algorithm and seed, angle order, bootstrap
contract, gate thresholds, output schema, and oracle-only limitation.

## Semantic Cross-Fit

Use the canonical disjoint ontology folds:

- object channels: indices `[0, 27)`;
- region channels: indices `[27, 37)`.

For episode `i`, pivot condition `p`, selection family `A`, and held-out
family `B`:

1. Warp the complete prediction by every declared angle around pivot `p`.
2. Score only channels in `A` against target channels in `A`.
3. Select `a(i,p,A)` by maximum category-aware IoU with identity-first
   tie-breaking.
4. Score the already selected angle only on held-out channels in `B`.
5. Record:

```text
delta(i,p,A→B)
  = heldout_iou(i,p,A→B)
  - heldout_identity_iou(i,B)
```

The two directional endpoints are:

- `object→region`;
- `region→object`.

The per-episode symmetric endpoint is:

```text
symmetric_delta(i,p)
  = 0.5 * (
      delta(i,p,object→region)
      + delta(i,p,region→object)
    )
```

The episode, not a cell or category, is the aggregation unit. For both the
selector and held-out family, record three distinct flags at identity:

- `predicted_support_empty`: family-specific predicted input support is zero;
- `target_support_empty`: family-specific target support is zero;
- `union_empty`: the identity RasterScore union is zero.

Report all six rates without dropping those episodes. Do not use a single
ambiguous `empty` field.

## Pivot Conditions

Evaluate the same prediction, target, angles, and folds under three pivot
conditions.

### True start

Use the exact continuous scale-2 pivot:

```text
(start_x_m / 1 m, start_z_m / 1 m)
```

### Map center

Use the exact continuous boundary-coordinate center:

```text
(25.0, 25.0)
```

### Within-Scene Shuffled Start

Use one prespecified within-scene one-to-one reassignment with seed `43`.
Each receiver episode receives the true start pivot of a different donor
episode in the same scene.

The assignment must:

- preserve every scene's pivot multiset exactly;
- assign each donor exactly once;
- reject donor and receiver identity equality;
- reject an assigned pivot exactly equal to the receiver's true pivot,
  including equality caused by duplicate starts;
- be independent of prediction and target scores.

Construct the assignment as the following deterministic perfect bipartite
matching. This algorithm uses SHA-256 as a stable seeded ordering function and
does not use a language or numerical-library random-number generator:

1. Represent an episode identity by the UTF-8 byte string
   `scene_id + "\0" + example_id`.
2. Process scenes in ascending UTF-8 byte order.
3. Within a scene, rank receivers by the lexicographic byte order of
   `SHA256(b"43\0receiver\0" + identity_bytes)`, breaking a digest collision
   by ascending `identity_bytes`.
4. For each receiver, rank eligible donors by the lexicographic byte order of
   `SHA256(b"43\0donor\0" + receiver_identity_bytes + b"\0" +
   donor_identity_bytes)`, again breaking a digest collision by ascending
   donor identity.
5. Run the standard receiver-ordered augmenting-path bipartite matcher. Its
   depth-first search tries donors in that receiver-specific order; on an
   occupied donor it recursively tries to reassign the current receiver
   assigned to that donor. A fresh visited-donor set is used for each
   top-level receiver.

Edges connect a receiver only to eligible donors with a different identity and
pivot. Fail explicitly if a scene has no perfect matching. Do not retry
alternative seeds after observing scores. Tests freeze a nontrivial synthetic
assignment fixture, and the manifest records the SHA-256 of the complete
stable-order `pivot_assignments.csv` bytes.

Record receiver identity, donor identity, true pivot, and assigned pivot in
`pivot_assignments.csv`.

One derangement is the primary shuffled control. Averaging multiple shuffled
assignments is outside this run because it multiplies the warp workload and
changes the frozen estimand.

## Geometry and Scoring

Reuse `prior.analyze.llm_grid_registration` unchanged:

- rotate about continuous pivots using inverse nearest-neighbor sampling;
- use one spatial transform for all 37 channels;
- preserve padded support;
- count all support outside the `50×50` target frame as false positives;
- keep target rasters and start metadata fixed.

Warp the complete grid once for each episode/pivot/angle combination. Slice
the warped channels into object and region views while retaining the same
spatial bounds and recalculating each view's support invariants. This prevents
the two selection directions from receiving different spatial transforms.

## Estimands and Inference

Report episode-macro means for both directional deltas and the symmetric delta
under every pivot.

Primary common-layout estimand:

```text
mean(symmetric_delta(i,true_start))
```

Primary start-specific contrasts:

```text
mean(symmetric_delta(i,true_start)
     - symmetric_delta(i,map_center))

mean(symmetric_delta(i,true_start)
     - symmetric_delta(i,shuffled_start))
```

Use one shared paired scene-cluster bootstrap:

- sampling unit: scene;
- clusters: the 11 `val_unseen` scenes;
- repetitions: 10,000;
- seed: 42;
- interval: percentile 2.5th and 97.5th percentiles.

The same bootstrap multiplicity matrix must calculate every endpoint and
contrast. This preserves pairing across folds and pivots. For a per-episode
quantity `x`, first calculate each scene's episode sum `S_s` and episode count
`N_s`. For bootstrap scene multiplicities `m_s`, every replicate is exactly:

```text
sum_s(m_s * S_s) / sum_s(m_s * N_s)
```

Thus the bootstrap retains episode weighting when scene sizes differ; it must
not average scene means equally. For a paired contrast, calculate the
per-episode difference first and apply the same formula.

Also report leave-one-scene-out point-estimate ranges for both directional
endpoints, the symmetric endpoint, and the two pivot contrasts. These ranges
are the minimum and maximum episode-macro means after removing exactly one
scene at a time. They are robustness diagnostics, not additional gates or
confidence intervals.

## Frozen Decision Gate

Full **GO** requires every condition:

1. true-start symmetric held-out mean is at least `+0.01`;
2. its 95% scene-cluster bootstrap lower bound is above zero;
3. both true-start directional point estimates are above zero;
4. the 95% bootstrap lower bound for true-start minus map-center is above
   zero;
5. the 95% bootstrap lower bound for true-start minus shuffled-start is above
   zero.

This is an intersection-union decision: all conditions must pass for the
single composite claim, so no multiplicity correction is applied to the gate.
Directional confidence intervals are still reported but are not separately
required to exclude zero.

Classify outcomes as:

- **GO:** every condition passes; proceed to design a target-free
  four-hypothesis selector or aggregator.
- **Partial evidence:** the symmetric true-start gain passes conditions 1 and
  2, but a directional point estimate or pivot contrast fails.
- **NO GO:** condition 1 or 2 fails.

No outcome authorizes navigation training or use of a ground-truth-selected
angle.

## Architecture and Files

Work directly on `main`, as explicitly requested. Keep changes confined to
new experiment-only code, tests, the design/plan, and today's daily note.

Create:

- `prior/analyze/d2026_07_28/__init__.py`;
- `prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py`;
- `tests/analyze/test_llm_grid_transform_crossfit.py`.

Modify after the verified full run:

- `docs/daily/2026-07-28.md`.

Do not modify:

- `prior/analyze/llm_grid_registration.py`;
- `prior/analyze/d2026_07_27/llm_grid_transform_diagnosis.py`;
- their existing tests or generated artifacts.

The new CLI uses `tap.Tap` and fixes all scientific parameters. It has no
partial-population mode: unit tests use injected synthetic rows, while the
public command runs only the complete 1,839-episode protocol.

## Data Flow

1. Validate fixed CLI arguments and input/output path disjointness.
2. Validate the prediction manifest before loading examples.
3. Load the exact R2R `val_unseen` population and scale-2 targets.
4. Parse each prediction; convert only missing files and
   `LLMGridValidationError` to an invalid empty prediction.
5. Validate identities, shapes, scene count, population, and status counts.
6. Build and validate the within-scene pivot matching before scoring.
7. For each episode, pivot, and angle, warp once and score object/region views.
8. Select on one family and evaluate on the other.
9. Calculate paired summaries, bootstrap intervals, leave-one-scene-out
   ranges, and the frozen gate.
10. Write all artifacts only after every invariant passes.

Unexpected I/O errors, malformed targets, duplicate identities, inconsistent
schema-invalid data, an impossible pivot matching, non-finite metrics, or
scientific-parameter overrides fail explicitly.

## Output Contract

Default ignored directory:

```text
outputs/llm_grid_analysis/
  start_centered_crossfit_controls_r2r_rxr_epoch2/
```

Required artifacts:

```text
manifest.json
angle_scores.csv
crossfit_results.csv
pivot_assignments.csv
summary.json
bootstrap.json
crossfit_control_intervals.png
```

`angle_scores.csv` contains one row per episode, pivot mode, and angle with
object/region intersection, union, support, out-of-frame support, and IoU.

`crossfit_results.csv` contains one row per episode, pivot mode, and selection
direction with selector identity/best scores, selected angle, margin, held-out
identity/selected/delta scores, and the six explicitly defined
predicted-support, target-support, and union-empty flags.

All CSV headers are fixed English names. JSON uses sorted keys and trailing
newlines. Rows use stable identity/pivot/angle/direction order. The plot shows
directional and symmetric means with 95% scene-cluster intervals for all
pivots, plus the two paired start-specific contrasts. It must not present
oracle values as deployable performance.

## Testing and Verification

Develop test-first. Synthetic tests must prove:

- object selection is unaffected by region-only score changes;
- region selection is unaffected by object-only score changes;
- each selected angle is reused unchanged on its held-out family;
- identity-first ties and invalid-empty rows remain zero and retained;
- full-grid warping gives both families identical bounds;
- out-of-frame support remains a false positive after channel slicing;
- a constructed true-start fixture beats center and wrong pivots;
- seeded matching is deterministic, one-to-one, within-scene, and forbids
  equal assigned pivots;
- impossible matching fails explicitly;
- bootstrap multiplicities are shared and deterministic;
- directional, symmetric, and paired contrast calculations are literal;
- the gate yields GO, partial evidence, and NO GO on constructed summaries;
- artifacts are complete, deterministic, and reject malformed populations or
  path overlap.

Before recording results, run:

```text
pytest tests/analyze/test_llm_grid_transform_crossfit.py -v
pytest tests/analyze -q
ruff check prior/analyze/d2026_07_28 tests/analyze
ty check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
git diff --check
```

Run the fixed full CLI once. Validate every row count, identity, pivot
assignment, support partition, finite metric, angle membership, paired
contrast, bootstrap interval, gate input, manifest field, and plot before
writing the Chinese result and decision to `docs/daily/2026-07-28.md`.
