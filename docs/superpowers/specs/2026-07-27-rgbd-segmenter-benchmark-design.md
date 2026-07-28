# RGB-D Segmenter Benchmark Design

> Operational details are frozen by
> `2026-07-28-rgbd-segmenter-benchmark-contract.md`, which supersedes ambiguous
> cohort, branch, metric, latency, statistics, provenance, and gate language in
> this design.

## Purpose

Select a deployable RGB-D perception front end to replace the GT semantic
sensor used by observation-bounded oracle evidence.

This design preserves the established representation decision:

- efficient closed-vocabulary RGB-D semantic segmentation is the primary path;
- detector-generated instance masks are the latency fallback;
- raw boxes are only a latency and representation-distortion ablation;
- object semantics and panoramic region recognition are separate heads because
  the final 37-channel cognitive map is multi-label.

No concrete model is selected before a paired local benchmark.

## Current Readiness

The repository already has:

- local Matterport3D scenes;
- a tested depth/semantic projection path;
- a hash-pinned oracle-evidence index for 393 unique R2R `val_unseen` starts;
- fixed twelve-view, `256×256`, 90-degree-HFOV, `0–10 m` sensor geometry;
- the canonical 27 object and 10 region vocabularies.

The repository does not yet have:

- local code or weights for DFormer/DFormerv2, ESANet, YOLOE, or YOLO-World;
- audited licenses, maintained checkpoint identifiers, or preprocessing
  contracts for those candidates;
- candidate-vocabulary-to-27-category mappings;
- a deployable 10-region classifier;
- saved raw RGB-D and per-pixel oracle frames for the proposed 50-start cohort;
- candidate adapters, confidence fusion, latency instrumentation, or a paired
  benchmark runner.

The existing oracle NPZ artifacts contain projected grids, masks, and pose.
They do not contain the raw frames needed to score or reproject segmenter
predictions.

## Staged Benchmark

### Stage 1: Static Candidate Gate

For each primary candidate, record from maintained primary sources:

- exact repository and revision;
- exact pretrained checkpoint and source dataset;
- license for code and weights;
- RGB and depth preprocessing, units, normalization, and calibration;
- supported source vocabulary;
- direct, merged, and unmapped coverage of the 27 object categories;
- twelve-view batching and expected memory behavior.

Reject candidates lacking runnable weights, an acceptable license, explicit
preprocessing, or meaningful target-vocabulary coverage. Unmapped classes are
reported explicitly and are never silently mapped to `other`.

Audit order:

1. ESANet as the efficient baseline;
2. the best maintained DFormer or DFormerv2 checkpoint as the accuracy
   candidate;
3. YOLOE masks only if dense segmentation violates the declared latency
   budget;
4. YOLO-World boxes last as the distortion ablation.

### Stage 2: Shared Raw-Frame Contract

Freeze and hash-pin the deterministically reconstructible 50-start R2R
`val_unseen` cohort before model integration.

Collect once:

- twelve aligned RGB and metric-depth views per start;
- per-pixel oracle object and region labels;
- camera intrinsics and extrinsics;
- episode start pose and target-grid origin;
- existing observed/free masks;
- artifact and cohort manifests.

Validate one observation by reprojecting its saved oracle frames and requiring
exact agreement with the existing oracle projector. Stop on RGB-depth
alignment, extrinsic, taxonomy, or grid-frame mismatch.

### Stage 3: Paired Front-End Benchmark

Run a one-observation smoke, then a small-cohort smoke, before the complete
50-start benchmark. Eliminate OOM, preprocessing, and Pareto-dominated
candidates early.

Every candidate uses the same:

- 50 starts and twelve views;
- raw RGB-D frames and camera calibration;
- depth geometry and projector;
- observed/free masks;
- vocabulary mapping;
- grid resolution;
- region-control protocol.

Evaluate the 27 object channels separately from the 10-region head. A composed
37-channel result is reported only after both deployable heads exist; oracle
regions may isolate an object-head ablation but cannot be presented as a
deployable system.

Primary metrics:

- projected category-cell precision, recall, F1, and IoU;
- category coverage and per-category failures;
- error by distance/depth, boundary, and occlusion;
- confidence calibration and multi-view consistency;
- twelve-view end-to-end P50/P95 latency;
- views per second, peak VRAM, preprocessing time, and projection time.

Statistics are paired by unique observation and clustered by scene. Sibling
instructions are not independent benchmark samples.

### Stage 4: Predictor and Navigation Bridge

Only front ends on the projected-quality/latency Pareto frontier proceed.

First evaluate predicted evidence with the existing predictor as a distribution
shift screen. Then train at least three paired seeds with matched, null,
within-scene, and global evidence conditions. Report observed and unobserved
precision/recall separately.

Navigation begins only if predicted matched evidence:

- improves R2R `val_unseen` unobserved-cell completion across paired seeds;
- degrades under relevant-evidence shuffling;
- exceeds the deterministic prediction-union-evidence baseline.

Try5 navigation then uses a fixed budget and matched no-map, zero, shuffled,
predicted-evidence, and PriorGT controls.

## Branch and Package Policy

This benchmark receives a separate experiment branch and worktree:

```text
exp/rgbd-segmenter-benchmark
```

Candidate-specific adapters remain experiment-local until one is accepted.
Vendor repositories are not copied into production packages. Reusable frame,
projection, and metric contracts move into narrow typed modules only after the
first candidate proves the interface.

Verified notes and results may be committed to `main`; one-off candidate
integration and launchers remain on the experiment branch.

## Out of Scope for the Transform Branch

- Downloading or installing segmenters.
- Raw RGB-D collection.
- Candidate vocabulary implementation.
- Region-head selection.
- Predictor retraining or navigation.
