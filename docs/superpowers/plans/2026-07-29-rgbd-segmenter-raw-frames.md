# RGB-D Segmenter Shared Raw Frames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` task by task, with specification
> and code-quality review gates after every task.

**Goal:** Publish one immutable shared RGB-D/oracle-frame package for the
sealed 50-start cohort and prove that its stored frames exactly replay all six
pinned oracle grid arrays.

**Architecture:** Two July 29 experiment modules separate pure package
format/validation from Habitat collection. The collector validates only sealed
selection metadata before rendering, renders from private scene-asset
snapshots with one simulator per scene, then opens each hash-pinned oracle NPZ
only after that observation's raw NPZ is written and strictly reloaded.

**Tech Stack:** Python 3.8, NumPy 1.24.3, existing Habitat-Lab/Habitat-Sim,
`tap.Tap`, Python stdlib, pytest, Ruff, and ty.

## Global Constraints

- Date/note target: July 29, 2026 CST; write findings in
  `docs/daily/2026-07-29.md` in Chinese.
- Work on `main`; touch only `prior/analyze/d2026_07_29/`,
  `tests/analyze/`, this plan, and the post-run daily note.
- Do not modify core model, navigation, evidence, mapping, or Habitat code.
- Never mutate sealed cohort
  `data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1/`.
- Final ignored root is separately fixed at
  `data/rgbd_segmenter_benchmark/r2r-val-unseen-50-raw-v1/`.
- Fixed GPU is device `0`; expose no override.
- No output/cohort/evidence/asset/sensor/GPU override, limit, overwrite,
  force, retry, repair, or resume control.
- Failed unpublished full collection deletes only its private staging sibling.
  A later attempt starts all 50 again; never resume across commit, process,
  environment, driver, CUDA, GPU, source, or asset changes.
- Run disposable first-row and first-per-scene smokes before the official
  full collection. Smokes never address the final root. “One-shot” means
  exactly one successful no-overwrite final publication; failed unpublished
  attempts may be recorded and restarted from zero.
- Source selection fields only may be read before rendering. Oracle arrays are
  opened only after that observation's raw NPZ is rendered, written, reloaded,
  and validated.
- Reuse public `OracleSensorFrame` and `project_oracle_frames`; adapters for
  RGB, semantic mapping, level origin, Habitat configuration, and strict
  artifact reading remain experiment-local.
- Fail closed on every mismatch. No dtype coercion, fallback, warning-only
  path, suppressed Ruff/ty error, or compatibility shim.

## Files

- Create
  `prior/analyze/d2026_07_29/rgbd_segmenter_raw_frame_package.py`:
  frozen schemas, canonical bytes, deterministic NPZ encoding, same-buffer
  readers, aggregates, validator, and atomic publisher.
- Create
  `prior/analyze/d2026_07_29/rgbd_segmenter_raw_frames.py`:
  fixed CLI, source preparation, scene snapshots, Habitat adapter, rendering,
  replay, environment capture, smokes, and full collection.
- Create after first-per-scene smoke
  `prior/analyze/d2026_07_29/rgbd_segmenter_asset_roles.json`.
- Create
  `tests/analyze/test_rgbd_segmenter_raw_frame_package.py`.
- Create
  `tests/analyze/test_rgbd_segmenter_raw_frames.py`.
- Modify after official publication only:
  `docs/daily/2026-07-29.md`.

## Frozen Inputs

### Cohort

- ID/directory: `r2r-val-unseen-50-v1` /
  `data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1`
- external manifest SHA-256:
  `d71f04f102d80df3799e5fea76162147ad76c060c82ccbca88813d8b14a0b191`
- `cohort.jsonl` SHA-256:
  `89ae70f3e489fa702c66110f9bd9e7666ba0e16a9fbc3ac20aaa91a95adef0ce`
- selection SHA-256:
  `32a7adddf32291f059eb1045e63e077a693ca64d6fdf6e7418b54dd5d3644cb2`
- sealing commit: `80780e5a8a3736dc666b8dd861c00ce55f4685d8`
- 50 observations, 11 lexical scenes.

Before rendering, accept exactly these cohort-row fields:
`artifact_sha256`, `example_ids`, `observation_id`, `scene_id`,
`start_position`, `start_rotation`. Aliases are validation-only.

### Evidence

- root/split:
  `data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen`
- key/dataset/split: `oracle-t0-v1`, `R2R`, `val_unseen`
- manifest SHA:
  `be2d25890c01234dd1bb8191af6c1d87121330991299287b635205f2ebf0327a`
- index SHA:
  `0acc9d18aea6d369db4b0a73f8ab3eebbe8fab51257b34614d1a7c407443acb9`
- raw split/SHA:
  `data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz`,
  `6140b46759fe332ee96aa849d4bb64e1c1829b8f65acee127355040a6ee23484`.

Before rendering, index metadata may bind scene, observation, aliases, artifact
path, and artifact SHA; evidence NPZ payloads remain unopened. Post-render,
open the pinned NPZ once with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`, hash it, and
parse the same accepted bytes via `io.BytesIO`; do not call a path-reopening
core loader.

### Scene assets and classification

From `data/scene_datasets/mp3d/<scene_id>/`, snapshot exactly:

```text
glb           <scene_id>.glb
house         <scene_id>.house
semantic_ply  <scene_id>_semantic.ply
navmesh       <scene_id>.navmesh
```

Copy into private `scene-assets/<scene_id>/` before simulator construction.
Open source with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`; require regular file. Open
destination with `O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW`, mode `0444`.
Hash while copying, verify source/destination byte counts and destination hash,
`fsync` file and directory, and reject symlinked components/path escape.

Keep snapshots in a process-private sibling outside both smoke and final
package staging. After each simulator closes, rehash every snapshot and
require its pre-render, post-render, and copied-source hashes to agree. On
failure delete only that attempt's package staging and snapshot sibling. On
success delete snapshots after all rows replay but before exact 52-file
package validation; final validation rehashes the current originals against
the recorded commitments.

The first-per-scene smoke runs an all-assets control plus four single-role
omission variants for the first sealed row in every lexical scene. A role is
globally required iff its omission prevents construction, exact sensor
collection, or six-array replay in any scene. The expected result to freeze is:

```json
{
  "algorithm": "single-role-omission-first-row-per-scene-v1",
  "auxiliary": ["navmesh"],
  "required": ["glb", "house", "semantic_ply"],
  "schema_version": 1
}
```

Write sorted, indented JSON with one trailing newline. If live classification
differs, stop and amend/review this plan; do not adapt silently. All four roles
remain copied and hash-bound even when auxiliary.

## Fixed Commands

```bash
python -m prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames \
  --smoke first-row
python -m prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames \
  --smoke first-per-scene
python -m prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames
```

`RawFrameArgs(Tap)` has only
`smoke: Literal["none","first-row","first-per-scene"]="none"`.
Smokes use a unique process-private sibling below
`data/rgbd_segmenter_benchmark/.r2r-val-unseen-50-raw-v1-smoke/`, validate
each NPZ with the exact artifact parser, perform exact replay, print a
canonical smoke report, and delete it in `finally`. Smokes do not create or
validate a partial index/manifest package.

## Sensor, Pose, and Replay Contract

- yaws, ordered: `0,30,...,330` (12 views);
- RGB/depth/semantic are collocated at each yaw;
- `256x256`, HFOV `90.0`, relative position `[0,1.25,0]`;
- orientation `[0,radians(yaw),0]`;
- depth min/max `0/10` metres, `NORMALIZE_DEPTH=False`;
- RGB exactly `uint8[256,256,3]`, RGB order; fail RGBA;
- depth exactly finite metric `float32[256,256]`; accept only Habitat's
  extra final singleton axis before requiring this shape/range;
- semantic IDs are integer `[256,256]`.

Call `get_observations_at(position=list(start_position),
rotation=list(start_rotation), keep_agent_at_new_pose=True)` exactly. Require
the returned agent position to be exactly equal to the sealed float64 start
position. For the returned raw agent quaternion and every raw sensor
quaternion, first require a finite float64 vector with norm within exact
float32 epsilon `2**-23` (`1.1920928955078125e-7`) of one (`rtol=0`), then
normalize exactly once. This replaces the preregistered `5e-8` after the
reviewed live first-row smoke exposed Habitat's float32 quaternion components
promoted to float64. A read-only scan of all 50 sealed observations across 11
scenes measured 1,800 sensor quaternions: maximum raw sensor norm error
`7.99814713e-8` (`0.671 * float32 epsilon`), maximum raw agent norm error
`1.71142710e-8`, maximum normalized agent-vs-sealed error `2.22e-16`, and all
600 normalized RGB/depth/semantic pose triples remained exactly
`np.array_equal`. Compare the normalized agent quaternion to the sealed
normalized quaternion by
`1 - abs(dot(agent, sealed)) <= 1e-12`, making sign equivalence explicit.

Only after the raw norm check, compare normalized RGB/depth/semantic positions
and quaternions with `np.array_equal` per yaw. Store the normalized depth state
as authoritative; never reconstruct pose from yaw.

The local semantic adapter builds `semantic_id -> object` using the integer
suffix after the final underscore in each non-null semantic object ID. For
each observed ID, missing object/category remains `-1`; otherwise obtain
`raw_object = category.index(mapping="mpcat40")`. Map raw `0` to `0`,
`0 < raw_object < len(OBJECT_MAPPING)` through `OBJECT_MAPPING`, and every
other raw object to `MAPPED_OBJECT_NAMES.index("other")`. If region/category
exists, obtain `raw_region = region.category.index()`, require
`0 <= raw_region < len(REGION_MAPPING)`, and map through `REGION_MAPPING`;
missing region/category remains `-1`.

The local level adapter defines each level floor Y as the minimum AABB-min Y
over its regions, falling back to the level AABB-min Y when it has no regions.
Sort levels by that floor; select the lowest when start Y is below all floors,
otherwise the highest floor not above start Y. Store the selected level's
float64 AABB-min X/Z as `target_origin_xz`.

Compute float64 target-local start X/Z as start X/Z minus target origin.
Cast the replayed target-local start and replayed start direction to float32,
then require `np.array_equal` with the pinned float32 evidence metadata. This
matches the existing save/load boundary exactly without admitting a tunable
tolerance. Cohort and stored float64 start pose require exact equality.

Build 12 public `OracleSensorFrame`s and call public `project_oracle_frames`.
Require `np.array_equal` for:

```text
ego_semantic_grid       ego_observed_mask       ego_free_mask
target_semantic_grid    target_observed_mask    target_free_mask
```

Store only the four observed/free masks in the raw artifact; semantic grids
remain in the pinned evidence and replay result.

## Exact 17-Member NPZ Schema

```text
schema_version             int64    []
rgb                        uint8    [12,256,256,3]
depth_m                    float32  [12,256,256]
object_categories          int16    [12,256,256]
region_categories          int16    [12,256,256]
sensor_positions           float64  [12,3]
sensor_rotations_xyzw      float64  [12,4]
sensor_yaw_degrees         int16    [12]
sensor_hfov_degrees        float64  []
sensor_position_relative   float64  [3]
start_position             float64  [3]
start_rotation_xyzw        float64  [4]
target_origin_xz           float64  [2]
ego_observed_mask          bool     [50,50]
ego_free_mask              bool     [50,50]
target_observed_mask       bool     [50,50]
target_free_mask           bool     [50,50]
```

Schema version is scalar `1`. Require `sys.byteorder == "little"` before
collection. Arrays are C-contiguous, finite where floating, and never
implicitly cast. Exact index dtype tokens are: `schema_version=<i8`,
`rgb=|u1`, `depth_m=<f4`, `object_categories=<i2`,
`region_categories=<i2`, `sensor_positions=<f8`,
`sensor_rotations_xyzw=<f8`, `sensor_yaw_degrees=<i2`,
`sensor_hfov_degrees=<f8`, `sensor_position_relative=<f8`,
`start_position=<f8`, `start_rotation_xyzw=<f8`,
`target_origin_xz=<f8`, and each mask `|b1`. Yaw/HFOV/relative position equal
constants; depth/category ranges are strict; free masks are subsets of
observed masks.

Write exact NumPy `.npy` v1.0 bytes into lexical `<member>.npy` ZIP entries,
`ZIP_DEFLATED`, level 6, fixed timestamp 1980-01-01, mode 0600, no pickle.
Freeze `ZipInfo.create_system=3`, empty comment/extra fields, and
`external_attr=(stat.S_IFREG | 0o600) << 16`; ASCII member names need no
platform-dependent encoding flag. Whole-NPZ byte determinism is scoped to the
captured Python/zlib implementation and version. Exact uncompressed NPY and
logical-array hashes remain the cross-environment commitments.
For every member hash/size both C-order logical array bytes and exact
uncompressed `.npy` bytes. The same-buffer parser rejects duplicate/extra/
missing/encrypted/directory/symlink entries, wrong compression/CRC/NPY
version/order/dtype/shape/hash, object dtype, trailing bytes, and size excess.

## Exact Index Schema

`index.jsonl` has 50 compact sorted-key JSON rows with trailing newlines in
sealed order. Every row has exactly:

```json
{
  "artifact": "observations/<scene>/<ordinal:02d>-<observation_id>.npz",
  "cohort_row_sha256": "<sha256 of row bytes without newline>",
  "members": {
    "<each exact member>": {
      "array_byte_length": 0,
      "array_sha256": "<sha256>",
      "dtype": "<dtype>",
      "npy_byte_length": 0,
      "npy_sha256": "<sha256>",
      "shape": []
    }
  },
  "npz": {"byte_length": 0, "sha256": "<sha256>"},
  "observation_id": "<20 lowercase hex>",
  "oracle_artifact_sha256": "<pinned sha256>",
  "ordinal": 0,
  "scene_id": "<scene>"
}
```

Unknown/missing nested keys fail. Ordinals are exactly 0–49; derived relative
paths, IDs, scenes, order, oracle hashes, and cohort-row hashes must reproduce
independently from sealed cohort bytes.

## Exact Manifest Schema

`manifest.json`: UTF-8, `indent=2`, `sort_keys=True`, trailing newline, no
self-hash. Exact top-level keys:

```text
schema_version collection_id collection cohort evidence sensor
scene_assets environment replay files
```

Fixed version/ID: `1`, `r2r-val-unseen-50-raw-v1`.

- `collection`: exact `git_commit`, `collector_source`, `package_source`,
  `asset_roles`, `command`, `gpu_device_id`. Each source/config record has
  exact `path`, `byte_length`, `sha256`; command is the no-argument official
  command; GPU is `0`.
- `cohort`: exact `cohort_id`, `directory`, `manifest_sha256`,
  `cohort_jsonl_sha256`, `selection_sha256`, `sealing_git_commit`,
  `observation_count`, `scene_count`, with frozen values/counts 50/11.
- `evidence`: exact `root`, `evidence_key`, `dataset`, `split`, `manifest`,
  `index`, `raw_split`, `projector_source`, `mapping_source`,
  `mapping_sha256`; each file record has exact
  `path`, `byte_length`, `sha256`. Projector path is
  `vlnce_baselines/models/etp_llm/llm_grid_oracle_cache.py`; mapping source is
  `prior/constants.py`. `mapping_sha256` hashes compact sorted JSON containing
  exact `OBJECT_MAPPING`, `REGION_MAPPING`, `MAPPED_OBJECT_NAMES`, and
  `MAPPED_REGION_NAMES`.
- `sensor`: exact `config`, `config_sha256`. Config exact keys:
  `views`, `yaw_degrees`, `height`, `width`, `hfov_degrees`, `position`,
  `min_depth_m`, `max_depth_m`, `normalize_depth`, `rgb_channel_order`,
  `depth_units`, `orientation_rule`, `camera_pose_authority`,
  `resolved_specs`. `resolved_specs` contains 36 lexical records with exact
  UUID, modality, Habitat sensor type/subtype, resolution, HFOV, relative
  position/orientation, depth bounds, and normalization. Hash the complete
  compact sorted config JSON and validate it against the resolved simulator
  configuration before rendering.
- `scene_assets`: exact `source_root`,
  `role_classification_algorithm`, `required_roles`, `auxiliary_roles`,
  `scenes`. Each lexical scene has exact `bundle_sha256`, `files`; each exact
  role record has `path`, `required`, `byte_length`, `sha256`. Bundle hash is
  compact sorted JSON of `files`.
- `environment`: exact `python_version`, `python_implementation`, `platform`,
  `numpy_version`, `zlib_version`, `habitat_version`, `habitat_sim_version`,
  `cuda_runtime_version`, `nvidia_driver_version`, `gpu_name`, `gpu_uuid`,
  `gpu_device_id`, `installed_distributions`,
  `installed_distributions_sha256`. Distributions are unique PEP-503
  normalized `{name,version}` records sorted by name/version; hash compact
  sorted JSON. Capture before rendering and before publication; require exact
  equality.
- `replay`: exact `observation_count`, `passed_count`, `scene_count`,
  `array_equal_fields`, `metadata_comparison`: values 50/50/11, the six
  lexical fields, and `cast-replayed-values-to-float32-then-array-equal`.
- `files`: exact `index`, `artifacts`, `payload`. Index has
  `path`, `byte_length`, `row_count`, `sha256`; each aggregate has
  `file_count`, `total_byte_length`, `tree_sha256`. Counts are 50 artifacts,
  and 51 payload files including index. Tree hash consumes lexical
  lines `<sha256>  <byte_length>  <relative-path>\n`. Exclude manifest.

Validator accepts exactly 52 final regular non-symlink files: 50 NPZ, index,
and manifest. It rejects every extra entry. Scene snapshots are private
collection inputs, not output payload: delete them after their post-render
hash check and before package validation, while retaining source-relative
paths, byte lengths, hashes, bundle hashes, and required/auxiliary
classification in the manifest. The validator rehashes the current original
assets against those commitments.

## Task 1: Pure Package Contract

**Files:** create package module and package tests.

**Produces:** frozen artifact/index types, `encode_raw_frame_npz`,
`parse_raw_frame_npz_bytes`, `parse_index_bytes`, and `tree_aggregate`.

- [ ] Write failing tests for the exact 17 members, dtype/shape/value
  invariants, deterministic pickle-free NPZ, logical/NPY hashes, same-buffer
  parser mutations, exact index nesting, canonical bytes, and aggregates.
- [ ] Run
  `pytest -q tests/analyze/test_rgbd_segmenter_raw_frame_package.py`;
  confirm import failure.
- [ ] Implement minimal frozen dataclasses and byte-buffer functions. Parse
  JSON with duplicate-key rejection; reject bool where integer is required.
- [ ] Run focused pytest, Ruff, and ty on the new files.
- [ ] Obtain specification and code-quality PASS; fix every
  critical/important finding.
- [ ] Commit: `feat: define shared RGB-D frame package`.

## Task 2: Coupled Sources and Asset Snapshot

**Files:** modify package module/tests; create collector module/tests.

**Produces:** `strict_read_bytes`, `load_collection_inputs`,
`load_pinned_oracle_after_render`, and `snapshot_scene_bundle`.

- [ ] Write failing tests proving every source is parsed from its accepted
  buffer with no reopen; collection input loading never opens oracle arrays;
  all 50 rows/11 scenes reproduce; source/alias/hash drift fails.
- [ ] Write failing asset tests for strict flags, hash-while-copy, exact four
  roles/names, read-only copies, fsync, symlink/path escape, and source
  preservation.
- [ ] Implement direct cohort/evidence metadata parsers and strict existing
  11-member oracle NPZ parser; do not call path-reopening core loaders.
- [ ] Implement immutable private bundle metadata; later simulator APIs accept
  only this bundle, never an original path.
- [ ] Run focused pytest, Ruff, ty; obtain both review PASSes.
- [ ] Commit: `feat: bind RGB-D collection inputs`.

## Task 3: Habitat Adapter, Replay, First-Row Smoke

**Files:** modify collector/tests.

**Produces:** `build_scene_simulator`, `render_raw_frame_artifact`,
`replay_and_require_exact`, and `run_first_row_smoke`.

- [ ] Write failing fake-simulator tests for exactly 36 sensors, yaw/order,
  modalities, RGBA failure, equal states, authoritative pose, mapping,
  target origin/tolerances, and six exact arrays.
- [ ] Add event-order tests requiring:
  metadata → snapshot → simulator → render → NPZ write → raw reload/validate
  → pinned oracle open → replay.
- [ ] Implement local Habitat/mapping/origin adapters and public projector
  calls; close each simulator in `finally`.
- [ ] Implement an isolated all-assets first-row smoke; emit only a canonical
  smoke report, not a partial package manifest.
- [ ] Run focused pytest, Ruff, ty; obtain both review PASSes.
- [ ] Commit: `feat: replay shared RGB-D frames`.
- [ ] On clean HEAD, run the fixed first-row smoke. Require control/replay PASS,
  removed smoke output, and absent final output.

## Task 4: Full Builder and Per-Scene Smoke

**Files:** modify both modules and both tests.

**Produces:** `build_manifest`, `collect_attempt`,
`validate_raw_frame_directory`, and `run_first_per_scene_smoke`.

- [ ] Write failing exact-manifest tests for every nested key/value/hash,
  environment lock, mapping/source/sensor-spec commitments, 50/51 aggregates,
  canonical bytes, and schema mutation.
- [ ] Write failing orchestration tests for sealed order, one simulator per
  scene, 50 post-render replays, close-on-error, no partial index, no resume,
  and abort on commit/environment/source changes.
- [ ] Implement environment capture, canonical index/manifest construction,
  and one-simulator-per-lexical-scene collection.
- [ ] Implement the first sealed row per each of 11 scenes smoke using the
  exact artifact validator and six-array replay, with an all-assets control
  plus four single-role omission variants per scene. Emit only a canonical
  report.
- [ ] Run both focused suites, Ruff, ty; obtain both review PASSes.
- [ ] Commit: `feat: build shared RGB-D frame collection`.
- [ ] On clean HEAD, run fixed first-per-scene smoke. Require 11/11 exact
  all-assets replay, expected global role classification across all scenes,
  snapshot-only simulator paths, cleanup, and absent final output.
- [ ] Add exact asset-role JSON and a same-buffer content/hash test; review and
  commit: `test: freeze RGB-D scene asset roles`.

## Task 5: Validator, Atomic Publisher, Fixed CLI

**Files:** modify both modules and both tests.

- [ ] Write mutation tests for every manifest/index/NPZ/source/cohort/asset/
  environment/order/count/path/symlink/extra-entry case and external expected
  commit.
- [ ] Write crash tests for private sibling staging, `O_EXCL`, file and
  bottom-up directory fsync, complete staging validation, second clean
  Git/HEAD/source/environment/GPU check, Linux
  `renameat2(RENAME_NOREPLACE)`, parent fsync, and staging-only cleanup.
- [ ] Write CLI/preflight tests for exact smoke choices, fixed paths/GPU,
  unknown override rejection, final-output absence before source load, clean
  Git, unchanged HEAD, and exact GPU identity recapture. Collection is not a
  timing gate and does not require exclusive GPU occupancy.
- [ ] Implement validator reconstruction from external cohort/source pins,
  all 17 members, all 50 same-buffer pinned oracle loads, and all 50 six-array
  replays. Validate staging in a fresh process before rename.
- [ ] Run focused tests, full `tests/analyze`, Ruff, ty, and
  `git diff --check`.
- [ ] Obtain independent contract, leakage/order, artifact/validator, and test
  PASSes; resolve every critical/important finding and rerun verification.
- [ ] Commit: `feat: publish shared RGB-D frames`.

## Task 6: Official One-Shot Publication and Finding

- [ ] Require clean committed HEAD, absent final/smoke/staging output, unchanged
  inputs, and the frozen GPU-0 identity. Record HEAD and source/config/
  projector hashes.
- [ ] Run the no-argument official command until exactly one successful
  no-overwrite publication occurs. Any pre-rename failure publishes nothing:
  record it, diagnose/fix/review/commit, discard that exact attempt's staging
  and snapshots, and restart all 50; never resume. After success, never rerun.
- [ ] Fresh-process validate with an exported reviewed commit:

  ```bash
  export P5_RAW_COMMIT="$(git rev-parse HEAD)"
  python -c "from pathlib import Path; import os; from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import validate_raw_frame_directory; validate_raw_frame_directory(Path('data/rgbd_segmenter_benchmark/r2r-val-unseen-50-raw-v1'), expected_git_commit=os.environ['P5_RAW_COMMIT'])"
  sha256sum \
    data/rgbd_segmenter_benchmark/r2r-val-unseen-50-raw-v1/manifest.json \
    data/rgbd_segmenter_benchmark/r2r-val-unseen-50-raw-v1/index.jsonl
  ```

- [ ] Independently audit exact schemas/hashes/counts, external cohort pins,
  event order, private assets, poses/modalities, tolerances, 50/50 six-array
  replay, environment/source/command/GPU bindings, validator, and no-overwrite.
- [ ] In Chinese, record in `docs/daily/2026-07-29.md`: P5.3 status, reviewed
  commit, external raw manifest/index hashes, cohort pins, environment/projector/
  sensor/asset hashes, file/byte totals, role classification, both smokes,
  50/50 replay, tolerances, reviews, and next checklist item. State sealed
  cohort/core were untouched and oracle arrays were post-render only.
- [ ] Run `git diff --check`, commit:
  `docs: record shared RGB-D frame collection`.
- [ ] Remove ignored subagent scratch state; preserve sealed cohort and final
  raw-frame package. Do not push unless requested.
