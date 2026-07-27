# LLM-Grid Start-Centered Transform Diagnosis Design

## Purpose

Determine whether the low absolute category-aware Raster IoU of the
instruction-derived LLM-Grid cognitive map is primarily explained by a coherent
episode-level orientation error around the known start position.

The first decision experiment is deliberately limited to the four cardinal
angles. It uses existing cached predictions and ground-truth cognitive maps
only. It does not train a model, generate predictions, access the login node, or
modify navigation behavior.

The result is an oracle upper bound, not a deployable correction. A
ground-truth-selected angle must never be passed to navigation.

## Established Evidence

- Mixed R2R+RxR-EN epoch 2 is the primary checkpoint because it has the best
  observed R2R `val_unseen` Raster IoU, approximately `0.136`.
- Its mentioned object and region category-presence F1 values are high while
  mentioned and unmentioned spatial IoU remain low. The predictor more reliably
  estimates what may be present than where it lies in the unseen house.
- Matched prediction-target IoU is higher than within-scene and global
  reassignment controls, so the prediction contains episode-specific spatial
  signal.
- The prior eight-square-symmetry audit selected identity. That excludes a
  universal transpose or cardinal coordinate-contract error, but it did not
  search for an episode-specific rotation around the exact start position.
- No start-centered rotation, scale, translation, or similarity registration
  experiment has run.

## Scope and Decision Sequence

This design implements only the first gate:

1. identity baseline;
2. per-episode oracle over `{0, 90, 180, 270}` degrees;
3. support, clipping, angle, tail, and scene-cluster-bootstrap diagnostics;
4. a stop/go decision for the four-hypothesis navigation proposal.

A later design may add 5-degree rotation, radial analysis, mismatch controls,
translation, and scale. Those stages are not implemented before inspecting this
four-way result.

If the four-way macro IoU delta is below `0.01`, its 95% scene-cluster
confidence interval includes zero, and few episodes gain more than `0.05`, the
four-map/MoE proposal stops. A fine-angle diagnostic still remains useful to
exclude a non-cardinal orientation error.

## Package Boundary

Reusable typed analysis primitives live in:

```text
prior/analyze/llm_grid_registration.py
```

The fixed experiment CLI and report orchestration live in:

```text
prior/analyze/d2026_07_27/llm_grid_transform_diagnosis.py
```

Tests live in:

```text
tests/analyze/test_llm_grid_registration.py
tests/analyze/test_llm_grid_transform_diagnosis.py
```

No analysis code is added to `vlnce_baselines.models.etp_llm`. Existing strict
LLM-Grid parsing, cache path construction, target loading, and downsampling are
reused.

## Fixed Inputs

Primary prediction cache:

```text
data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2/
```

Primary split:

```text
R2R val_unseen
```

Ground-truth namespace:

```text
data/cognitive_maps/gt.legacy.r1p5.direction5.v1/
```

Reference evaluator output:

```text
outputs/llm_grid_eval/checkpoint-sweep-r1p5/runs/001/
```

The analysis population is exactly 1,839 `val_unseen` episodes. Missing or
schema-invalid predictions remain in the population as empty predictions with
zero IoU. The analysis must not silently restrict itself to the 1,830
schema-valid prediction rasters.

The navigation-used final cache:

```text
data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/
```

is reserved for a later robustness replication after the primary result.

## Coordinate and Rotation Contract

Ground-truth raster artifacts contain a `37×100×100` grid at `0.5 m` per cell.
The LLM-Grid scale-2 evaluator downsamples this to `37×50×50`, so a coarse cell
is exactly `1 m`.

The stored `start_position` is level-local `(x, z)` in meters. For this scale-2
experiment, the continuous coarse-grid pivot is therefore:

```text
s = (x / 1 m, z / 1 m)
```

For a semantic cell center `p = (row + 0.5, col + 0.5)`, the transformed point
is:

```text
p' = s + R(theta) @ (p - s)
```

The implementation:

- keeps ground truth fixed;
- applies one angle to all 37 semantic channels;
- uses nearest-neighbor inverse warping of boolean channels;
- uses the same signed angle to rotate predicted direction vectors;
- leaves start position and start heading unchanged;
- records warped support inside and outside the `50×50` target frame;
- counts all out-of-frame warped support as false positives;
- never rotates about the map center, prediction centroid, or target centroid.

The existing whole-grid right-angle helper is not reused because it rotates the
square canvas and moves the start.

## Padded Warp Representation

`warp_grid_about_pivot` returns a typed `WarpedGrid` with:

- the boolean warped grid;
- integer row and column offsets that locate its first cell in target-frame
  coordinates;
- input support;
- total warped support;
- in-frame warped support;
- out-of-frame warped support.

The spatial output bounds are derived from the transformed continuous corners
of the source raster. Every output cell center inside those bounds is sampled
by inverse rotation. This dynamic padded representation avoids a global magic
margin while preserving all transformed support for scoring.

An identity transform must return the same support and coordinates. Synthetic
right-angle examples around fractional pivots establish the sign and pivot
convention independently of production data.

## Metrics

For each episode and angle, compute across category, row, and column:

- intersection;
- union;
- category-aware Raster IoU;
- predicted support;
- target support;
- in-frame and out-of-frame warped support;
- warped-to-input support ratio.

The chosen angle maximizes category-aware Raster IoU. Ties use the declared
angle order, which places identity first.

For the chosen angle, additionally report:

- paired IoU delta from identity;
- best and second-best IoU and margin;
- Cell precision, recall, and F1;
- best-angle distribution and identity rate;
- proportions with IoU gain above `0.01` and `0.05`;
- median, mean, and distribution of paired IoU deltas;
- retained and out-of-frame support distributions;
- direction-vector cosine before and after applying the same angle;
- object-only, region-only, mentioned, unmentioned, occupancy-collapsed, and
  broad-channel-excluded IoU sensitivities.

The complete episode table retains scene and example identity, schema validity,
all four angle scores, and the selected transform diagnostics.

## Statistical Contract

R2R `val_unseen` has only 11 scenes. Uncertainty therefore uses a paired
scene-cluster bootstrap:

- resample scenes with replacement;
- include every episode belonging to each sampled scene;
- use 10,000 repetitions;
- use seed 42;
- report the 2.5th and 97.5th percentiles of the macro IoU delta.

The bootstrap probability of positive delta is descriptive and is not reported
as a p-value.

## Outputs

The fixed CLI writes:

```text
outputs/llm_grid_analysis/start_centered_registration_r2r_rxr_epoch2/
├── manifest.json
├── episodes.csv
├── summary.json
├── bootstrap.json
├── angle_distribution.csv
├── iou_delta_distribution.png
└── angle_distribution.png
```

`manifest.json` pins:

- prediction cache key and both split-manifest SHA-256 values where applicable;
- ground-truth namespace;
- analysis population and split;
- angle order;
- grid scale, shape, and cell size;
- bootstrap repetitions and seed;
- git commit when available.

The experiment refuses missing manifests, unexpected cache keys, a population
other than 1,839, duplicate identities, or target shape/coordinate violations.
It does not guess similarly named caches.

## Experiment Notes

Record the run and findings in Chinese under `docs/daily/2026-07-27.md`.
Include:

- exact cache and target provenance;
- verification commands;
- baseline and four-way IoU;
- confidence interval and tail changes;
- angle and support distributions;
- whether the four-map/MoE proposal passes the stop/go gate;
- limitations stating that the selected angle is an oracle.

## Verification

Before running the complete cache analysis:

- synthetic tests recover known right-angle transforms around integer and
  fractional pivots;
- identity preserves the grid and support;
- out-of-frame support is retained and counted as false positive;
- one transform is shared by all channels;
- the angle sign matches direction-vector rotation;
- invalid predictions remain zero and in the denominator;
- angle ties choose identity;
- the scene-cluster bootstrap is deterministic.

Run targeted Pytest, Ruff, and `ty` over every changed Python file. Then run the
full four-way CLI and verify that its identity column reproduces the existing
epoch-2 `val_unseen` baseline within floating-point tolerance.

## Out of Scope

- Fine-angle, translation, scale, or joint similarity search.
- Any model training or LLM generation.
- Navigation integration or a GT-selected navigation angle.
- RGB-D segmenter collection, installation, or evaluation.
- Changes to the established LLM-Grid evaluator contract.
- Remote login-node access or job submission.
