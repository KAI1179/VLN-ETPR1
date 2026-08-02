# RGB-D Segmenter Benchmark Frozen Contract

## Status and Scope

This document amends and operationalizes
`2026-07-27-rgbd-segmenter-benchmark-design.md`. It freezes the P5 benchmark
before raw-frame collection, candidate integration, or inspection of candidate
scores.

The benchmark selects a deployable object-semantic RGB-D front end for
observation-bounded evidence. It does not select a region head, retrain the map
predictor, run navigation, or modify the navigation architecture.

The later user authorization to keep analysis experiments on `main` supersedes
the earlier experiment-worktree requirement. Only experiment-specific modules,
tests, specifications, notes, dependency locks, configurations, and launchers
may be changed on `main`. Core model, navigation, and evidence code remain
unchanged. Vendor code, weights, generated frames, and benchmark outputs are
never committed.

## Fixed Source Population

The source population is the local R2R `val_unseen` oracle-evidence index:

- 1,839 examples;
- 393 unique start observations;
- 11 scenes;
- evidence key `oracle-t0-v1`;
- index SHA-256
  `0acc9d18aea6d369db4b0a73f8ab3eebbe8fab51257b34614d1a7c407443acb9`;
- manifest SHA-256
  `be2d25890c01234dd1bb8191af6c1d87121330991299287b635205f2ebf0327a`;
- raw R2R split SHA-256
  `6140b46759fe332ee96aa849d4bb64e1c1829b8f65acee127355040a6ee23484`.

Episode reconstruction may read only episode ID, scene ID, start position, and
start rotation. Instruction, goal, reference path, ground-truth trajectory,
connectivity, later observations, semantic support, rendered pixels, and model
outputs cannot influence cohort selection.

## Sealed 50-Observation Cohort

The cohort is proportionally stratified by scene rather than inheriting the
arbitrary first 50 globally sorted observation hashes.

Scenes are ordered lexically. Hamilton largest-remainder apportionment assigns
50 slots in proportion to the 393 observation counts, with lexical tie breaks.
For the fixed population the ordered scene quotas are:

```text
8, 1, 4, 8, 7, 4, 4, 5, 1, 3, 5
```

Within a scene, each canonical selection key is serialized as compact UTF-8
JSON with sorted keys and no insignificant whitespace. It contains only the
observation ID, scene ID, exact start position and normalized rotation, and
sorted example aliases. Selection order is:

```text
SHA-256(
  b"etp-r1:rgbd-segmenter-benchmark:cohort-v1\0" + canonical_selection_key_bytes
)
```

with observation ID as the collision tie break. The pinned oracle-artifact
SHA-256 is excluded from the selection key so semantic artifact contents cannot
influence membership or order. It remains in each published cohort row solely
as provenance. The sealed two-file package is ordered first by lexical scene ID
and then by the selection digest. It is published and independently validated
before rendering. `cohort.jsonl` records all selected canonical rows; the
manifest records the source hashes, algorithm/version, scene populations and
quotas, and a `selection_sha256` over the concatenated published selected-row
bytes. The complete manifest has an external SHA-256 recorded after
publication; it does not contain its own hash. Selection code must never load
oracle arrays or images.

The inferential target is this fixed sealed cohort. Results must not be
described as pristine held-out generalization or as an estimate over all 393
starts.

## Shared Raw-Frame Contract

Each selected start contains exactly twelve collocated RGB, metric-depth, and
oracle-semantic views at yaw angles `0, 30, ..., 330` degrees:

- RGB: `uint8[12, 256, 256, 3]`, RGB channel order, range `[0, 255]`;
- depth: `float32[12, 256, 256]`, metres, finite, unnormalized, range
  `[0, 10]`;
- object and region labels: `int16[12, 256, 256]`, with `-1` for no mapped
  label;
- camera-to-world position: `float64[12, 3]`;
- camera-to-world quaternion `(x, y, z, w)`: `float64[12, 4]`;
- horizontal field of view: 90 degrees;
- sensor position relative to the agent: `(0, 1.25, 0)` metres;
- exact episode start pose and target-grid origin;
- existing target/ego observed and free masks.

For integer pixel indices `u` increasing right and `v` increasing down, axial
depth `d` is converted to camera coordinates by:

```text
x = (u + 0.5 - 128) * d / 128
y = (128 - v - 0.5) * d / 128
z = -d
```

This is equivalently `fx=fy=128`, `cx=cy=127.5` under the integer-index
convention. Sensor orientation is configured exactly as Habitat Euler
orientation `[0, radians(yaw), 0]` relative to the agent start state. The
stored camera-to-world quaternion is authoritative for projection; no adapter
reconstructs or recomposes yaw.

RGB, depth, and semantic sensors must have equal intrinsics, shapes, and
extrinsics for every view. NPZ files are pickle-free, have an exact member
schema, and are written through an atomic no-overwrite publisher.

Collection provenance records:

- source split, oracle index, scene mesh, semantic asset, and sensor-config
  hashes;
- clean collection commit;
- Habitat/Habitat-Sim, Python, NumPy, CUDA, driver, and GPU identity;
- byte hash, dtype, and shape of every stored member;
- complete collection command.

Loaded bytes and their hashes are coupled. After collection, all 50 observations
are rehydrated through the existing oracle projector and must exactly reproduce
all six semantic, observed, and free arrays in the pinned evidence artifacts.
Pose metadata uses exact equality where serialized values permit it and a
documented tight tolerance otherwise. One-observation and one-per-scene
technical smokes precede the full collection, but publication requires all 50
replays to pass in a fresh process.

## Object Evaluation Scope

The primary endpoint uses the 23 deployable object channels. It excludes:

- `void`;
- `structure`;
- `other`;
- `free-space`.

All 27 object channels are reported only as a diagnostic. Region channels are
absent from the primary object benchmark.

Candidate logits are resized to the shared `256×256` pixel grid with bilinear
interpolation and `align_corners=False`. The source argmax label is then mapped
through a candidate-specific table frozen before benchmark scores are visible.
Direct and many-to-one mappings produce one target label. Intentionally ignored
or unmapped source labels abstain. There is no confidence threshold, retry,
repair, or mapping to `other`.

Mapped hard labels are projected with the exact semantic path of
`project_oracle_frames` from the frozen experiment commit: original metric
depth, stored camera pose, stored target origin, 10-metre saturation, pixel
centres, and boolean union across all twelve views. Regions are set to `-1`,
and only `target_semantic_grid[:27]` is consumed. Invalid depth at or below zero
is ignored; non-finite depth is a schema failure; saturated depth contributes
no semantic endpoint. Conflicting target categories from pixels or views set
multiple grid channels. Candidate-generated observed/free masks are discarded
in favour of the shared canonical geometry masks.

Candidate spatial preprocessing may not crop, stretch, or otherwise discard or
distort the shared camera field of view. A candidate that cannot consume
`256×256` directly may use only an aspect-preserving resize plus symmetric
padding. Its exact inverse unpadding and resize back to the original integer
pixel grid must be frozen and pass a synthetic coordinate-grid round trip
before model inference.

Inference or schema failure remains in the denominator as an empty prediction.
For one observation:

- a non-empty target with empty prediction has IoU and F1 equal to zero;
- target-empty and prediction-empty observations are excluded from
  observation-macro IoU/F1 and counted separately;
- target-empty but prediction-nonempty observations have IoU and F1 equal to
  zero.

For observation `o`, flatten the 23 target channels and the `50×50` target grid
into one set of positive `(channel, row, column)` cells. Let `TP_o`, `FP_o`, and
`FN_o` be the resulting category-cell counts. Then:

```text
IoU_o = TP_o / (TP_o + FP_o + FN_o)
F1_o  = 2*TP_o / (2*TP_o + FP_o + FN_o)
```

The primary metric is the arithmetic mean of `IoU_o` over observations with a
non-zero denominator. It is not a macro-average of 23 category IoUs. Mean
per-observation F1 uses the same inclusion rule. Pooled metrics sum `TP_o`,
`FP_o`, and `FN_o` before applying the formulas. Scene-macro metrics first
average observation endpoints within each of the 11 scenes and then average
scenes equally.

All-27 and per-category results are descriptive diagnostics. Distance/depth,
boundary, occlusion, calibration, and multi-view-consistency formulas are
descriptive-deferred and non-gating until separately frozen before their
values are inspected. In particular, no calibration claim is made from an
unmapped source probability.

## Static Coverage Gate

A dense candidate enters paired benchmarking only when its frozen source-to-
target mapping covers:

- at least 14 of the 23 deployable categories by direct or declared many-to-one
  mapping; and
- at least 80% of the cohort's oracle deployable-object target-cell support.

Support is the number of positive `(observation, category, row, column)` target
cells over the 23 channels. A target category is covered iff at least one
frozen source class maps to it. The numerator counts target cells whose
category is covered; the denominator counts all deployable target cells.
Both sufficient counts and their ratio are published.

The support calculation is performed only after the cohort is sealed and before
candidate inference. Direct, merged, ignored, and unmapped source labels are
reported separately. A candidate cannot silently map an unsupported source
class to `other`.

## Region Controls

Three results remain distinct:

1. **Object-only primary:** no region prediction and no region channel in the
   score.
2. **Oracle-region isolation diagnostic:** candidate objects may be composed
   with oracle regions only to isolate object-head effects. It is explicitly
   non-deployable and cannot affect candidate selection.
3. **Deployable composed system:** reported only after a separate frozen
   region-head contract and benchmark. It must include region-head latency.

Oracle region labels never select an object candidate. P5.8 must separately
freeze whether a deployable region head is panorama-, view-, or pixel-level,
its multi-label semantics, confidence fusion, and projection.

## Latency and Resource Contract

The reference device is one local NVIDIA GeForce RTX 3090. The run manifest
records its UUID, driver, CUDA, PyTorch, CPU, and software environment.
The process has exclusive use of that GPU, verified by the absence of unrelated
compute processes before and after timing. No clocks, power limit, or
persistence setting is changed: the system-default power limit, application
clock policy, persistence mode, temperature before/after, and
`PYTORCH_CUDA_ALLOC_CONF` are recorded. The allocator variable is unset for the
official run. A run starting or ending outside `30–80 °C` is invalid.

Primary steady-state timing uses:

- one batch containing all twelve views;
- `torch.inference_mode()` and candidate-supported FP16 autocast;
- the first 20 sealed observations in manifest order as untimed warm-up
  batches;
- two measured passes over all 50 sealed observations in manifest order,
  producing 100 observation-batch measurements;
- explicit device synchronization before and after each measured boundary;
- inputs resident in pinned host memory before timing;
- model preprocessing, host-to-device transfer, inference, logit resize,
  category mapping, device-to-host transfer, and grid projection inside the
  end-to-end boundary.

Disk loading, model construction, and checkpoint loading are reported
separately as cold-start measurements. Candidate-native model-only time,
preprocessing, postprocessing, and projection are also reported separately.
Peak allocated and reserved VRAM are reset and measured for the twelve-view
batch in one isolated fresh process per candidate. The manifest records both
the pre-adapter CUDA baseline and absolute/incremental peak allocated and
reserved bytes. The resource gate uses absolute peak reserved bytes. An OOM is
a retained technical failure and is not retried with a smaller batch.

P50 and P95 use the linearly interpolated order-statistic rule
`h=(n-1)*p`, interpolating between `floor(h)` and `ceil(h)`.

A dense candidate passes deployability only when:

- twelve-view end-to-end P95 is at most 1,000 ms; and
- peak reserved VRAM is at most 16 GiB.

These limits leave the remaining device capacity available to the navigation
pipeline and make the detector-mask fallback an explicit gate rather than a
post-result preference.

## Statistics and Decision Rules

All candidate contrasts are paired by observation. The primary contrast is
accuracy-candidate minus ESANet mean observation IoU.

Preregistered scene-composition robustness uses 10,000 deterministic
scene-cluster bootstrap replicates with seed `20260728`. It is not a sampling
confidence interval for the fixed-cohort estimand. Each replicate draws 11
scenes with replacement. Every draw contributes all sealed observations from
that scene; a duplicated scene contributes a duplicate copy of every one of
its rows. The replicate endpoint is the ordinary observation mean over the
concatenated rows. Report percentile 95% robustness intervals, scene-macro
endpoints, and leave-one-scene-out ranges. Per-category results are descriptive
and do not drive selection, so no multiplicity claim is made.

The draw matrix is generated row-major by:

```python
numpy.random.Generator(numpy.random.PCG64(20260728)).integers(
    0, 11, size=(10000, 11), endpoint=False
)
```

Indices refer to lexically ordered scene IDs. Interval endpoints use the same
linear quantile rule frozen for P50/P95. The manifest records the exact NumPy
version and SHA-256 of the integer draw matrix.

A candidate passes the minimum quality gate only when:

- static coverage passes;
- mean observation IoU is at least `0.15`;
- mean observation F1 is at least `0.25`;
- all 50 observations produce auditable rows, including retained failures.

Selection is deterministic:

- discard candidates failing the quality, latency, resource, license, or
  provenance gate;
- if exactly one dense candidate passes, select it;
- for candidates `A` and `B`, quality `A` is no worse when the paired
  robustness-interval lower bound for `IoU_A - IoU_B` is greater than `-0.01`,
  and materially better when that lower bound is at least `+0.01`;
- latency `A` is no worse when `P95_A <= 1.05 * P95_B`, and materially better
  when `P95_A <= 0.95 * P95_B`;
- `A` dominates `B` only when it is no worse on both quality and latency and
  materially better on at least one;
- discard every candidate dominated by another passing candidate, evaluating
  all ordered pairs from lexically sorted candidate IDs;
- retain multiple candidates on the frontier when none dominates the others.

YOLOE masks are integrated only if no dense candidate passes all minimum gates.
YOLOE must pass the same applicable coverage, quality, latency, resource,
license, provenance, and complete-row gates. If YOLOE also fails, P5 returns
`NO GO`. YOLO-World boxes remain a final representation-distortion diagnostic
and cannot rescue the deployable decision or replace a passing mask or dense
candidate merely by being faster.

The date-scoped YOLOE implementation checklist, frozen mask protocol, package
security work, and execution status are maintained in
[`docs/daily/2026-07-31.md`](../../daily/2026-07-31.md).

Technical smokes may eliminate only schema-invalid, non-runnable, OOM, or
latency-infeasible integrations. Smoke semantic quality is never inspected for
adaptive candidate elimination.

## Candidate and Result Provenance

Before inference, each candidate record freezes:

- official repository URL and exact revision;
- exact checkpoint identifier, source dataset, URL, and downloaded SHA-256;
- code and weight license status;
- isolated dependency lock/environment;
- native RGB/depth preprocessing and units;
- complete source vocabulary and frozen mapping-table hash;
- precision and batch contract.

The license gate requires documented permission for local non-commercial
research execution of both code and weights and for creation of local derived
benchmark metrics. Commercial use and redistribution rights are not required
by this research benchmark, so a candidate may still be called deployable only
within that stated scope. Missing, contradictory, or merely assumed permission
fails the gate until clarified. No vendor source or weight is redistributed.

Benchmark publication is atomic and no-overwrite. Its manifest records every
input and payload hash, row count, candidate/environment commitment, timing
protocol, and complete command. It explicitly binds the clean experiment code
commit, cohort-manifest external hash, raw-frame-manifest external hash,
candidate weight hash, mapping-table hash, and projector source-file hash. A
public fresh-process validator independently recomputes metrics, statistics,
gates, and the final decision from the sealed rows.

## Execution Order

1. P5.0: freeze this contract.
2. P5.1: audit ESANet and DFormer/DFormerv2, exact checkpoints, licenses,
   preprocessing, vocabularies, and mappings without installation.
3. P5.2: implement and publish the cohort sealer and independent validator.
4. P5.3: collect shared raw frames and pass all-observation oracle replay.
5. P5.4: implement typed experiment-only adapter, projection, timing, metric,
   and artifact contracts with a synthetic adapter.
6. P5.5: integrate and benchmark ESANet.
7. P5.6: integrate and benchmark the accepted DFormer-family candidate.
8. P5.7: apply the frozen gates; integrate YOLOE masks only if fallback fires.
9. P5.8: separately freeze and benchmark a deployable region head.
10. P5.9: only after a front-end GO, write a separate predictor/navigation
    bridge plan.

No later stage begins merely because an earlier stage produced artifacts. Every
gate remains literal.
