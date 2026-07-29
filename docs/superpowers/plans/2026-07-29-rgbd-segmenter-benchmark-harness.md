# P5.4 RGB-D Segmenter Benchmark Harness Plan

## Status and Boundary

This plan operationalizes P5.4 of
`docs/superpowers/specs/2026-07-28-rgbd-segmenter-benchmark-contract.md`.
It is frozen before any segmenter repository, checkpoint, or candidate score is
loaded. It changes only July-29 experiment modules, tests, specifications, and
notes on `main`. It does not change the sealed cohort, shared raw-frame
collector/package/projector, model code, navigation code, or evidence code.

P5.4 proves the benchmark machinery with a deterministic synthetic adapter. It
does not select a model and makes no semantic-quality, GPU-latency, or
deployability claim. Its manifest must contain:

```text
run_kind = synthetic-contract
latency_claim = false
decision = NOT_APPLICABLE
```

P5.5 remains blocked until the ESANet checkpoint permission, checkpoint hash,
isolated environment, and preprocessing records pass their frozen gates.

## Fixed Inputs and Findings

The P5.4 reader binds:

- raw-frame root:
  `data/rgbd_segmenter_benchmark/r2r-val-unseen-50-raw-v1`;
- raw producer commit:
  `923d44bc53ced1f4685e62fd8086a01056d513c5`;
- external raw manifest SHA-256:
  `7aa278dd3b511bbf38aa647e3294a7fb9476b35bf7a2eb505a5bf3d2cf6c1c87`;
- external raw index SHA-256:
  `38a68369f72654271648589c0d19508831a3b484473fe382dadeee457db04d17`;
- sealed cohort SHA-256:
  `89ae70f3e489fa702c66110f9bd9e7666ba0e16a9fbc3ac20aaa91a95adef0ce`;
- selection SHA-256:
  `32a7adddf32291f059eb1045e63e077a693ca64d6fdf6e7418b54dd5d3644cb2`;
- projector source SHA-256:
  `daa4bfbb07d524e71341fdc33127aa8bf4e7fc502ff92773ebcf516ae8debe75`.

The machine-readable NYU40 mapping is frozen before inference. It covers 17 of
23 deployable categories. The once-only sealed-cohort support calculation is:

```text
covered category-cells = 5,447
all deployable category-cells = 6,120
coverage = 0.890032679738562
```

Unsupported support is `plant=354`, `shower=14`, `stool=36`,
`fireplace=116`, `gym_equipment=29`, `seating=124`, totaling 673. Both the
14-category and 80% support gates pass. This does not satisfy the independent
license gate.

## Files

Add:

```text
prior/analyze/d2026_07_29/rgbd_segmenter_benchmark_contract.py
prior/analyze/d2026_07_29/rgbd_segmenter_benchmark_package.py
prior/analyze/d2026_07_29/rgbd_segmenter_synthetic.py
prior/analyze/d2026_07_29/rgbd_segmenter_nyu40_mapping.json
tests/analyze/test_rgbd_segmenter_benchmark_contract.py
tests/analyze/test_rgbd_segmenter_benchmark_package.py
tests/analyze/test_rgbd_segmenter_synthetic.py
```

Do not modify the already hash-bound P5.3 collector or package modules. Later
candidate adapters depend only on the new P5.4 public APIs.

## Validated Raw-Observation Reader

Before `iter_validated_raw_observations`, the launcher runs a separate,
path-pinned P5.3 interpreter subprocess with `CUDA_VISIBLE_DEVICES` absent, the
original unmasked physical GPU 0, and the exact P5.3 installed-distribution
environment. That subprocess invokes the existing public
`validate_raw_frame_directory` against the fixed producer commit and emits a
canonical validation attestation binding its interpreter, environment, device,
producer commit, raw root, external hashes, and successful exit. The launcher
requires exit zero, parses exactly one canonical attestation, independently
checks its fields and SHA-256, and passes the expected hash through the trusted
launcher boundary before starting the benchmark process. Empty or multiple
stdout records, timeout/signal/nonzero exit, malformed JSON, unexpected stderr,
or field/hash mismatch fail closed.

The benchmark process may run in the isolated candidate environment with
exactly one visible benchmark GPU. It records a separate candidate-environment
attestation. `iter_validated_raw_observations` verifies the P5.3 attestation,
then opens one new descriptor-anchored package root and, relative to that
descriptor:

1. rejects symlinks, non-regular files, and extra entries;
2. strict-reads the externally pinned manifest and index;
3. parses the exact 50 canonical index rows with the existing public parser;
4. strict-reads each row's hash-pinned NPZ in ordinal order;
5. parses the exact 17-member artifact with the existing public parser;
6. captures and finally recaptures the root/path binding and complete tree,
   rejecting any identity or content change.

Content continuity between subprocess validation and the new reader is
established by the validation-attestation hash and external
manifest/index/package hashes, not by an inode-identity claim across processes.
Do not import or copy P5.3 private helpers. The synthetic technical run uses the
same two-phase protocol, although both phases use the P5.3 environment and its
benchmark phase has no official GPU timing/resource claim.

The adapter receives only `SegmenterInput(rgb, depth_m)`. Observation identity,
scene, start pose, camera pose, semantic labels, masks, target origin, oracle
support, and target grids remain outside the adapter boundary. The runner may
use the remaining raw arrays only after inference for fixed projection and
audit.

## Typed Adapter and Spatial Contract

Freeze immutable typed values:

```python
SegmenterInput
SpatialTransform
PreparedHostBatch
DeviceBatch
CandidateCommitment
MappingEntry
Prediction
ObservationResult
TimingSample
ResourceMeasurement
```

`SegmenterInput` contains CPU pinned `torch.uint8[12,256,256,3]` RGB and
`torch.float32[12,256,256]` finite metric depth. The adapter protocol exposes:

```python
preprocess_host(SegmenterInput) -> PreparedHostBatch
infer(DeviceBatch) -> torch.Tensor
```

The runner, not the adapter, owns host-to-device transfer.
One named public runner function converts `PreparedHostBatch` to `DeviceBatch`
so the protocol has exactly one device-batch producer.
The adapter output is finite floating source logits `[12,C,H,W]`, where `C`
equals the frozen source vocabulary. Candidate-specific model construction and
checkpoint loading occur outside this interface.

Preprocessing may use only aspect-preserving resize plus symmetric padding.
The scale is `min(model_height/raw_height, model_width/raw_width)`. Each resized
dimension is `floor(raw_dimension * scale + 0.5)`, clamped to
`[1, model_dimension]`.
For an odd padding remainder, the extra pixel goes to bottom or right.
`SpatialTransform` records raw, resized, and model sizes plus
top/bottom/left/right padding. `CandidateCommitment` freezes the exact RGB and
depth interpolation/coordinate semantics, normalization, invalid-depth policy,
and padding values. The synthetic candidate uses RGB bilinear interpolation
with `align_corners=False`, depth `nearest-exact`, no normalization, and zero
padding.

Shared postprocessing, outside the adapter:

1. bilinearly resize logits to the padded model-input size;
2. remove exactly the recorded integer padding;
3. bilinearly resize logits to `256x256` with `align_corners=False`;
4. source argmax, with ties resolved by the lowest source index;
5. apply the explicit source-index mapping.

Diagnostic wall, window, and otherstructure labels map to canonical `structure`;
floor, floor mat, and ceiling map to canonical `free-space`. Ignored and
unmapped source labels abstain as `-1`. Mapping to canonical `other` is
forbidden. Mapping is applied after source argmax, never by merging logits.
Primary metrics exclude diagnostic canonical channels; all-27 metrics retain
them.

## Projection and Metrics

For each observation, build twelve public `OracleSensorFrame` values using the
original metric depth and stored camera poses, the mapped hard object labels,
and region labels fixed to `-1`. Call the existing public
`project_oracle_frames`; consume only `target_semantic_grid[:27]`. Discard its
candidate-derived geometry masks and retain the shared canonical masks.

The primary tensor selects all object channels except `void`, `structure`,
`other`, and `free-space`, producing `23x50x50` boolean category-cells.
Per observation:

```text
TP = count(prediction & target)
FP = count(prediction & ~target)
FN = count(~prediction & target)
IoU = TP / (TP + FP + FN)
F1 = 2*TP / (2*TP + FP + FN)
```

Only target-empty and prediction-empty rows are excluded from macro IoU/F1 and
counted separately. Target-nonempty/prediction-empty and
target-empty/prediction-nonempty rows score zero. JSON represents undefined
precision, recall, IoU, or F1 as `null`, never NaN. Recompute observation mean,
pooled, scene-macro, all-27, and per-category descriptive results from integer
counts. Per-category results never become a category macro or gate input.

The closed per-observation failure enum is `PREPROCESS_FAILURE`,
`INFERENCE_FAILURE`, or `OUTPUT_SCHEMA_FAILURE`. Only
`AdapterObservationError` may carry one of these stable codes; raw exception
text is never stored, and the runner catches no other adapter exception.
Failures produce auditable empty predictions and then follow the ordinary
metric eligibility rules, rather than being forced into every denominator.
Catch `torch.cuda.OutOfMemoryError` before typed failures; adapters are forbidden
to translate OOM. Projection, mapping, harness, and all other unexpected
exceptions abort. CUDA OOM is a fatal resource failure: do not retry a smaller
batch and publish only the separate aborted-run record.

## Statistics

Linear quantiles use `h=(n-1)p`. The scene-composition matrix is generated
exactly by:

```python
numpy.random.Generator(numpy.random.PCG64(20260728)).integers(
    0, 11, size=(10000, 11), endpoint=False
)
```

It is serialized C-order as little-endian `int64`. Its SHA-256 under the frozen
contract is
`a89573633a8efd5dac5ffe03f292491a8aba5202ff2951f7ac186f07f2c07f99`.
Scene indices refer to lexical scene order. Duplicate draws duplicate all rows
from that scene. Each candidate preserves its own frozen row eligibility. A
contrast uses the shared scene-resample matrix and is exactly
`contrast(A, B) = eligible_mean(A) - eligible_mean(B)` for each draw.
It is a paired contrast of candidate endpoints, not a mean of per-row deltas.
Use the same ordered direction for leave-one-scene-out. Report linear
2.5%/97.5% percentiles and leave-one-scene-out min/max.

Pure per-candidate gate helpers implement the already frozen quality, latency,
resource, license, provenance, and complete-row rules. Multi-candidate
Pareto/frontier/fallback orchestration is deferred to P5.7. The synthetic run
bypasses gates by declaring `decision=NOT_APPLICABLE`; tests still prove the
literal per-candidate behavior.

## Timing and Resource State Machine

Inputs are loaded and pinned before timing. Disk reads, model/checkpoint
construction, and artifact encoding are outside end-to-end timing.

The exact state sequence is:

```text
setup
20 warmups: sealed ordinals 0..19, untimed
measured pass 1: ordinals 0..49
measured pass 2: ordinals 0..49
done
```

Any missing, duplicate, reordered, or extra transition fails. Each timing row
stores `source_labels_sha256`, `mapped_labels_sha256`, `status`, and
`failure_code`. Label hashes are over the deterministic logical-array bytes:
fixed little-endian `int16`, C order, and fixed `[12,256,256]` shape. The first
measured pass supplies the canonical prediction. The second pass must reproduce
both source and mapped label hashes plus typed status and failure code exactly.

For each of the 100 measurements:

1. synchronize the recorded CUDA stream/device;
2. start a monotonic wall-clock measurement;
3. host preprocessing;
4. runner-owned host-to-device transfer;
5. inference under `torch.inference_mode()` and the frozen precision mode;
6. device-side shared logit restoration, argmax, and mapping;
7. runner-owned device-to-host transfer and synchronization before CPU use;
8. CPU grid projection;
9. synchronize the recorded CUDA stream/device;
10. end the wall-clock measurement.

This wall-clock boundary is the authoritative end-to-end value. Component
diagnostics use host monotonic timestamps for preprocessing and projection, and
runner-owned CUDA events on the recorded stream for H2D, inference, device
postprocessing, and D2H. Failed or unreached components are JSON `null`.
Components are not required to sum to end-to-end time and cannot replace it.
P50/P95 use the 100 end-to-end measurements. Views/s is
`12 * 100 / total_seconds`.

Cold-start records separate model construction, checkpoint byte read, and
checkpoint deserialization/state load. Peak allocated/reserved memory records
the pre-adapter baseline and absolute/incremental peaks in one fresh process.
The official CUDA backend requires:

- exactly one visible CUDA device, mapped to the recorded physical RTX 3090
  UUID;
- no unrelated compute PID before or after timing;
- only the current process PID may appear;
- absent `PYTORCH_CUDA_ALLOC_CONF`;
- start and end temperatures in `[30,80]` Celsius;
- unchanged UUID, driver, clock policy, persistence mode, and power limit.

The synthetic run uses `timing_backend=deterministic-fake`,
`timing_unit=synthetic-tick`, and `timing_comparable=false`. Its manifest
forbids latency P50/P95, views/s, resource-gate, and latency-gate fields. It
records no GPU claim and cannot be compared with the 1,000 ms or 16 GiB gates.
Because its validation phase invokes the P5.3 validator subprocess, the
technical run still requires the original unmasked physical GPU 0 plus the
exact P5.3 environment, assets, and oracles; it is not a CPU-only
isolated-environment run.

## Candidate Package

The fixed successful package contains:

```text
manifest.json
observations.jsonl
timings.jsonl
predictions/<scene>/<ordinal>-<observation-id>.npz  # 50
```

Each prediction NPZ stores exact
`source_labels int16[12,256,256]` and
`mapped_labels int16[12,256,256]`. A typed failed row stores both arrays as
`-1`. Observation rows bind ordinal/identity/status/failure code, prediction
path/hash/member schema, primary/all-27/per-category integer counts, and
derived endpoints. Timing rows contain exactly 100 ordered `(pass, ordinal)`
samples, each binding both output hashes and the typed outcome.

The manifest binds:

- clean experiment commit and complete command;
- external raw manifest/index/package hashes and producer commit;
- contract, package, synthetic-adapter, mapping, projector, and constants
  source hashes;
- candidate/repository/revision/checkpoint/license/preprocessing/precision
  commitments;
- separately hash-bound P5.3-validator and candidate benchmark environment
  attestations, embedding both canonical objects and their SHA-256 values rather
  than referring to mutable external paths, plus GPU/resource evidence when
  applicable;
- mapping category/support coverage;
- timing/statistics protocols and bootstrap matrix hash;
- observation/timing/prediction file aggregates;
- recomputed summaries and per-candidate gates/status; only the synthetic
  package additionally records `decision=NOT_APPLICABLE`.

An aborted OOM package uses a separate destination
`<candidate-id>-aborted-oom-<producer-commit-prefix>` and contains only canonical
`manifest.json` with `run_status=aborted`, `candidate_status=FAIL_RESOURCE`,
`resource_gate=FAIL`, stable code `CUDA_OUT_OF_MEMORY`, failure stage,
device/environment evidence, resource snapshot, available peak
allocated/reserved measurements, and candidate/source commitments. It has no
quality, latency, or P5.7 decision fields and never occupies the successful
destination. A code-corrected full-batch rerun is allowed; a smaller-batch retry
is forbidden.

The validator independently validates the raw package, strict-reads all
candidate files, reapplies mapping, reprojects mapped labels, reloads pinned
targets only after predictions are accepted, and recomputes every count,
endpoint, bootstrap result, gate, and aggregate. It rejects noncanonical JSON,
pickle, changed/range-invalid label arrays, wrong mappings/projections, missing,
duplicate, reordered, extra or symlink entries, source/environment drift, and
wrong per-candidate status or gates. For the synthetic schema it also requires
the literal `decision=NOT_APPLICABLE`; real candidate schemas forbid a
selection-decision field. Validation cannot prove that source labels came from the model;
their provenance is instead bound by the runner, candidate/revision/checkpoint
commitments, and equal per-pass hashes.

Publication uses a process-private sibling, exclusive file creation, file and
directory fsync, complete validation before rename, Linux
`renameat2(RENAME_NOREPLACE)`, parent fsync, staging-only cleanup on failure,
and fresh-process validation. It never overwrites an existing destination.
A later P5.7 decision package consumes immutable candidate-package hashes and
alone computes the multi-candidate frontier/fallback selection decision.

## Synthetic Adapter and Run

The synthetic adapter uses the actual frozen NYU40 vocabulary and mapping. It
produces deterministic finite 40-class low-resolution logits from RGB/depth
only and exercises an aspect-preserving transform with odd padding. It has no
weights, receives no identity/pose/oracle input, and emits identical mapped
labels in both measured passes.

Run it across all 50 sealed observations with the exact warmup/two-pass state
machine and the strongly separated fake-timing schema above. Do not print,
inspect, compare, or adapt to its semantic score. Publish once to:

```text
data/rgbd_segmenter_benchmark/synthetic-contract-v1
```

Fresh-process validate the package, record only contract PASS/failure,
hashes/counts, static coverage, and non-claim flags, then never rerun the
publisher.

## Implementation Tasks

### Task 1: Contract Types, Mapping, and Validated Reader

- Write failing tests for all immutable schemas, input/logit validation,
  mapping kinds/ranges/abstention, raw-package pins/order, and adapter leakage
  boundary.
- Add the exact 40-row machine-readable NYU mapping and hash test.
- Test the path-pinned P5.3 subprocess attestation, separate benchmark
  attestation, hash coupling, incompatible environment/device boundaries, and
  fail-closed subprocess outcomes.
- Implement the validated streaming reader without modifying P5.3 sources.
- Run focused tests, Ruff, ty, diff check; independent review and commit.

### Task 2: Spatial Restore, Projection, and Metrics

- Write discriminating tests for odd padding, poison padding, bilinear
  `align_corners=False`, resize-before-argmax, tie behavior, many-to-one,
  ignored/diagnostic/unmapped labels, and coordinate round trip.
- Test all twelve stored poses, multi-view category union, zero/saturated
  depth, region `-1`, shared-mask authority, and exact projector reuse.
- Test every metric eligibility/count/mean/pooled/scene/per-category edge.
- Implement the minimum pure functions; review, verify, and commit.

### Task 3: Statistics, Gates, and Timing

- Test the exact bootstrap matrix/hash, paired duplicate-scene weighting,
  linear quantiles, candidate-specific eligibility contrasts (including a
  target-empty false-positive row versus an empty prediction),
  leave-one-scene-out, and all per-candidate gates.
- Test the complete timing transition machine, fake clock/sync call order,
  20/50/50 sequence, first-pass output authority, second-pass determinism,
  source/mapped output hashes, typed failure retention, fatal OOM, component
  nullability, and non-claim fields.
- Implement injected CPU and official CUDA backends without running candidate
  inference; review, verify, and commit.

### Task 4: Candidate Package, Validator, and Publisher

- Freeze canonical JSONL/NPZ/manifest schemas and deterministic encoders.
- Test every count/hash/order/path/symlink/extra/mutation/recomputation case.
- Implement descriptor-safe validation, atomic no-overwrite publication,
  aborted OOM record, and fresh-process validator.
- Review security, scientific recomputation, and tests; verify and commit.

### Task 5: Synthetic Adapter and Technical Publication

- Implement the deterministic 40-class synthetic adapter and fixed `tap.Tap`
  CLI.
- Run unit/integration tests, full `tests/analyze`, Ruff, ty, and diff check.
- Obtain independent contract, leakage, artifact, timing, and test reviews.
- At a reviewed clean commit, require absent final/staging output and run the
  synthetic command until one successful publication. Pre-rename failures
  publish nothing and restart from row zero after reviewed fixes.
- Fresh-process validate without reading semantic scores; independently audit
  hashes/counts/contracts. Never rerun after success.

### Task 6: Finding and Cleanup

- Record P5.4 in Chinese in `docs/daily/2026-07-29.md`: reviewed commit,
  mapping/static coverage, package hashes/counts, validation/reviews, synthetic
  non-claim flags, preserved P5.3/cohort/core, and P5.5 license blocker.
- Mark P5.4 complete and P5.5 blocked pending explicit weight permission.
- Commit `docs: record RGB-D benchmark harness contract`.
- Remove only ignored P5.4 subagent scratch. Preserve both sealed packages and
  do not push unless requested.
