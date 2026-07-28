# LLM-Grid Target-Free Cardinal Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the frozen cache-only experiment that calibrates a
heading-conditioned cardinal correction on R2R `val_seen`, seals target-free
assignments, and evaluates them once on R2R `val_unseen`.

**Architecture:** Add one date-scoped experiment module and one test module.
The experiment separates runtime-only assignment generation from target
scoring with typed inputs, reuses the existing padded start-centred warp,
publishes one atomically validated nine-file artifact transaction, and never
changes reusable or navigation code.

**Tech Stack:** Python 3.8, `tap.Tap`, frozen dataclasses, enums, NumPy,
Matplotlib, standard-library CSV/JSON/hash/path utilities, Pytest, Ruff, ty.

## Global Constraints

- Work directly on `main`; do not create a branch or worktree.
- Create only
  `prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py` and
  `tests/analyze/test_llm_grid_target_free_cardinal.py` before the final note.
- Do not modify core, reusable registration, navigation, cache, dataset,
  training, or existing experiment code.
- Reuse `prior.analyze.llm_grid_registration` without modifying it.
- Do not import private behavior from the July 27 diagnosis or cross-fit
  experiment.
- Do not add dependencies, compatibility wrappers, dynamic imports, `Any`,
  ignored type errors, lint suppressions, silent fallbacks, or target repairs.
- Use the exact populations, hashes, angles, mappings, schemas, statistics,
  decision predicates, output path, and run command in the accepted design.
- Use schema version `llm-grid-target-free-cardinal-v2`; episode rows and CSV
  contain exact soft identity/aggregate object and region IoUs, never proxy
  values copied from boolean per-angle scores.
- A corrupt runtime start direction aborts the run. A malformed prediction is
  an explicit empty prediction in the denominator.
- Generate and hash all `val_unseen` assignments before loading any
  `val_unseen` target grid or target direction vector.
- Refuse to run when the official output directory already exists.
- Run the fixed full command exactly once, only after every preflight passes.
- Use Chinese only for the final update to `docs/daily/2026-07-28.md`; use
  English in code, tests, specifications, plans, commits, and communication.

---

### Task 1: Typed Selector and Aggregation Core

**Files:**
- Create: `prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py`
- Create: `tests/analyze/test_llm_grid_target_free_cardinal.py`

**Interfaces:**
- Consumes:
  - `WarpedGrid`, `warp_grid_about_pivot`, and `rotate_direction_vectors` from
    `prior.analyze.llm_grid_registration`.
- Produces:
  - `ANGLE_ORDER: tuple[float, ...]`
  - `EpisodeKey`
  - `SelectorInput`
  - `HeadingMapping`
  - `SelectorAssignment`
  - `SoftRasterScore`
  - `quantize_heading(start_direction: NDArray[np.float32]) -> float`
  - `direct_heading_assignment(selector_input: SelectorInput) -> tuple[float, float]`
  - `uniform_soft_score(hypotheses: tuple[WarpedGrid, ...], target: NDArray[np.bool_]) -> SoftRasterScore`

- [ ] **Step 1: Write failing validation and heading tests**

Create the production module with only its module docstring so importing the
test target succeeds without implementing behavior. Add imports and tests
equivalent to:

```python
def test_quantize_heading_uses_clockwise_display_frame_and_declared_ties() -> None:
    assert target_free.quantize_heading(np.asarray([0.0, 1.0], np.float32)) == 0.0
    assert target_free.quantize_heading(np.asarray([1.0, 0.0], np.float32)) == 90.0
    boundary = np.asarray([2**-0.5, 2**-0.5], np.float32)
    assert target_free.quantize_heading(boundary) == 0.0


@pytest.mark.parametrize("bad", [
    [0.0, 0.0],
    [float("nan"), 1.0],
    [0.5, 0.0],
])
def test_selector_input_rejects_invalid_start_direction(bad: list[float]) -> None:
    with pytest.raises(ValueError, match="start_direction"):
        target_free.SelectorInput(
            schema_valid=True,
            start_direction=np.asarray(bad, np.float32),
            predicted_grid=np.zeros((37, 50, 50), dtype=np.bool_),
            predicted_directions=np.zeros((5, 2), dtype=np.float32),
        )
```

- [ ] **Step 2: Run the heading tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_target_free_cardinal.py -q
```

Expected: test failures for missing attributes such as
`quantize_heading` and `SelectorInput`; collection itself succeeds.

- [ ] **Step 3: Implement the narrow domain types**

Implement frozen dataclasses with explicit validation:

```python
ANGLE_ORDER = (0.0, 90.0, 180.0, 270.0)


@dataclass(frozen=True, order=True)
class EpisodeKey:
    scene_id: str
    example_id: str


@dataclass(frozen=True)
class SelectorInput:
    schema_valid: bool
    start_direction: NDArray[np.float32]
    predicted_grid: NDArray[np.bool_]
    predicted_directions: NDArray[np.float32]


@dataclass(frozen=True)
class HeadingMapping:
    sign: int
    offset_degrees: float

    def angle_for(self, start_direction: NDArray[np.float32]) -> float:
        heading_bin = quantize_heading(start_direction)
        return float((self.sign * heading_bin + self.offset_degrees) % 360.0)


@dataclass(frozen=True)
class SelectorAssignment:
    key: EpisodeKey
    schema_valid: bool
    heading_bin_degrees: float
    primary_angle_degrees: float
    global_angle_degrees: float
    direct_angle_degrees: float
    direct_margin: float


@dataclass(frozen=True)
class SoftRasterScore:
    intersection_mass: float
    union_mass: float
    predicted_mass: float
    target_mass: float
    precision: float
    recall: float
    f1: float
    iou: float
```

`SelectorInput` copies arrays into immutable C-contiguous arrays; validates
shapes `(37,50,50)`, `(2,)`, `(5,2)`; requires finite directions; requires
start norm within `1e-4` of one; and requires every predicted direction row to
be zero or unit length within `1e-4`. `HeadingMapping` accepts only sign
`{-1,+1}` and declared offsets.

Implement `quantize_heading` using `atan2(right, up) mod 360`, circular
distance, and declared-order ties. Invalid input raises rather than choosing a
default.

- [ ] **Step 4: Write failing mapping, direct-selector, and soft-score tests**

Cover all eight mappings, invalid identity behavior, zero predicted directions,
direct cosine ties, positive-angle vector sign, common padded bounds,
out-of-frame mass, empty union, and exact soft precision/recall/F1. Include:

```python
def test_direct_selector_empty_prediction_is_identity_with_zero_margin() -> None:
    selector_input = valid_selector_input(
        schema_valid=False,
        predicted_directions=np.zeros((5, 2), np.float32),
    )
    assert target_free.direct_heading_assignment(selector_input) == (0.0, 0.0)


def test_uniform_soft_score_keeps_out_of_frame_false_positive_mass() -> None:
    hypotheses = cardinal_warps(single_corner_cell_grid(), pivot=(1.0, 1.0))
    score = target_free.uniform_soft_score(hypotheses, empty_target())
    assert score.predicted_mass > 0.0
    assert score.intersection_mass == 0.0
    assert score.iou == 0.0
```

- [ ] **Step 5: Run the new tests and confirm RED**

Run the named mapping/direct/soft tests with `pytest -q`. Expected: missing
methods/functions fail for the intended reasons.

- [ ] **Step 6: Implement mappings, direct selection, and soft aggregation**

Use the existing warp/vector helpers. For direct selection, find the first
nonzero predicted vector once, rotate it for each declared physical angle,
score against the fixed start direction, and retain the first maximum. Margin
is top minus second cosine; invalid/all-zero inputs return `(0.0, 0.0)`.

For soft aggregation, construct inclusive common bounds from all hypothesis
bounds and `[0,49]×[0,49]`, embed every boolean grid and the target, average
the four float grids, and calculate the exact soft statistics in the design.
Do not threshold or clip.

- [ ] **Step 7: Verify and commit Task 1**

Run:

```bash
pytest tests/analyze/test_llm_grid_target_free_cardinal.py -q
ruff check prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py tests/analyze/test_llm_grid_target_free_cardinal.py
ty check prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py tests/analyze/test_llm_grid_target_free_cardinal.py
git diff --check
```

Commit:

```bash
git add prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py tests/analyze/test_llm_grid_target_free_cardinal.py
git commit -m "feat: add target-free cardinal selectors"
```

---

### Task 2: Runtime Population, Development Lock, and Test Sealing

**Files:**
- Modify: `prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py`
- Modify: `tests/analyze/test_llm_grid_target_free_cardinal.py`

**Interfaces:**
- Consumes: Task 1 selector types and functions.
- Produces:
  - `PopulationContract`
  - `RuntimeEpisode`
  - `TargetEpisode`
  - `DevelopmentCandidateScore`
  - `SelectorLock`
  - `load_runtime_population(split: str, contract: PopulationContract, cache_dir: Path, cache_model_key: str, cognitive_map_namespace: str, quiet: bool) -> tuple[RuntimeEpisode, ...]`
  - `_load_target_arrays(path: Path) -> tuple[NDArray[np.bool_], NDArray[np.float32]]`
  - `load_target_population(runtime_episodes: Sequence[RuntimeEpisode], cognitive_map_namespace: str, quiet: bool) -> tuple[TargetEpisode, ...]`
  - `develop_selector(development_targets: Sequence[TargetEpisode]) -> SelectorLock`
  - `assign_population(runtime_episodes: Sequence[RuntimeEpisode], selector_lock: SelectorLock) -> tuple[SelectorAssignment, ...]`
  - `assignment_csv_bytes(assignments: Sequence[SelectorAssignment]) -> bytes`

- [ ] **Step 1: Write failing population-contract tests**

Construct synthetic episodes and assert exact population, scene, valid, invalid,
unique-key, split, manifest-hash, and sorted-order checks. Assert corrupt start
metadata aborts while malformed prediction text becomes an empty prediction.

Add a target-access spy:

```python
def test_assignments_do_not_read_target_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = synthetic_runtime_population()
    monkeypatch.setattr(target_free, "_load_target_array", fail_if_called)
    assignments = target_free.assign_population(runtime, frozen_lock())
    assert target_free.assignment_csv_bytes(assignments)
```

The test module defines `fail_if_called` as a local helper that raises
`AssertionError("target access during assignment")`; it also defines
`synthetic_runtime_population()` and `frozen_lock()` from explicit small
dataclass fixtures.

- [ ] **Step 2: Run the population tests and confirm RED**

Run the new population tests. Expected: missing population/lock APIs.

- [ ] **Step 3: Implement split population types and loading**

Implement:

```python
@dataclass(frozen=True)
class PopulationContract:
    split: str
    episodes: int
    scenes: int
    valid: int
    invalid: int
    manifest_sha256: str


@dataclass(frozen=True)
class RuntimeEpisode:
    key: EpisodeKey
    split: str
    selector_input: SelectorInput
    start_pivot: tuple[float, float]


@dataclass(frozen=True)
class TargetEpisode:
    runtime: RuntimeEpisode
    target_grid: NDArray[np.bool_]
    target_directions: NDArray[np.float32]
```

Freeze constants for `val_seen=778/53/770/8` and
`val_unseen=1839/11/1830/9` plus both manifest hashes.

Use `load_llm_grid_examples` only to enumerate the expected R2R split and
obtain its raster path. During runtime loading, open the NPZ and read only
`start_position` and `start_direction_vector`; never access `grid` or
`direction_vectors`. Parse the raw prediction text with `parse_grid_text`.
Missing/malformed prediction becomes the explicit empty selector input.

`load_target_population` is the only function allowed to read target `grid`
and `direction_vectors`; downsample with the authoritative scale-2 helper and
join by exact `EpisodeKey`.

- [ ] **Step 4: Write failing calibration and sealing tests**

Use small synthetic populations where each of the eight mappings and each
global angle can be the unique winner. Test aggregate tie order, full
denominator scoring, lock canonicality, assignment sort order, invalid
identity assignment, `.17g` floats, lowercase booleans, no empty cells,
negative-zero normalisation, and byte equality after target mutation.

- [ ] **Step 5: Implement the development lock**

Implement:

```python
@dataclass(frozen=True)
class DevelopmentCandidateScore:
    candidate_kind: str
    mapping_sign: int
    heading_offset_degrees: int
    global_angle_degrees: int
    episode_count: int
    valid_count: int
    all_mean_iou: float
    object_mean_iou: float
    region_mean_iou: float
    selected: bool


@dataclass(frozen=True)
class SelectorLock:
    chosen_mapping: HeadingMapping
    chosen_global_angle: float
    development_scores: tuple[DevelopmentCandidateScore, ...]
    mapping_tied_candidate_count: int
    global_tied_candidate_count: int
```

`develop_selector` scores exactly eight mappings followed by four global
angles on the complete development denominator, with declared-order aggregate
ties. `assign_population` receives runtime episodes plus the frozen lock and
returns sorted assignments without accepting or importing a target type.

Implement canonical CSV primitives once with standard `csv.writer`,
`\n`, `.17g`, finite-only floats, and explicit sentinel `-1`.

- [ ] **Step 6: Verify and commit Task 2**

Run focused and full new tests, Ruff, ty, and `git diff --check`.

Commit:

```bash
git add prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py tests/analyze/test_llm_grid_target_free_cardinal.py
git commit -m "feat: seal target-free selector assignments"
```

---

### Task 3: Episode Scores, Statistics, and Decision Gate

**Files:**
- Modify: `prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py`
- Modify: `tests/analyze/test_llm_grid_target_free_cardinal.py`

**Interfaces:**
- Consumes: sealed assignments, target episodes, and Task 1 scoring helpers.
- Produces:
  - `AngleScoreRow`
  - `EpisodeScoreRow`
  - `ContrastInterval`
  - `BootstrapIntervals`
  - `DecisionLabel`
  - `DecisionResult`
  - `score_population(targets: Sequence[TargetEpisode], assignments: Sequence[SelectorAssignment]) -> tuple[tuple[AngleScoreRow, ...], tuple[EpisodeScoreRow, ...]]`
  - `paired_scene_bootstrap(rows: Sequence[EpisodeScoreRow], repetitions: int, seed: int) -> BootstrapIntervals`
  - `leave_one_scene_out(rows: Sequence[EpisodeScoreRow], contrast_name: str) -> tuple[float, float]`
  - `classify_decision(selector_identity: ContrastInterval, selector_global: ContrastInterval, global_identity: ContrastInterval, selector_object_mean: float, selector_region_mean: float, global_object_mean: float, global_region_mean: float, audit_passed: bool) -> DecisionResult`

- [ ] **Step 1: Write failing episode-score tests**

Use hand-computable two-channel grids to test identity, primary, global,
direct, random expectation, oracle, object/region, support, soft aggregation,
soft identity/aggregate object and region IoUs, direction cosine, invalid
denominator rows, and oracle-gain availability.
Assert four angle rows and one episode row per episode in canonical order.

- [ ] **Step 2: Run score tests and confirm RED**

Run the new score tests. Expected: missing score rows and functions.

- [ ] **Step 3: Implement typed score rows**

Define `AngleScoreRow` and `EpisodeScoreRow` with fields in exactly the CSV
header order from the design. Validate finite numeric values, declared angles,
nonnegative support/mass, `in_frame + out_of_frame = predicted_support`,
`intersection <= union`, and exact key association.

`score_population` verifies assignments match targets one-to-one, constructs
the four padded warps once per episode, derives every boolean and soft endpoint
without reselecting a candidate, and returns canonical tuples.

- [ ] **Step 4: Write failing bootstrap and gate tests**

Test shared resamples, unequal scene sizes, duplicate sampled scenes, seed 42,
10,000 repetitions, NumPy linear percentiles, paired-difference-first
behavior, LOSO ranges, all selector/fixed/partial/no-go branches, decision
precedence, and exact boundary semantics:

```python
def test_gate_requires_selector_to_beat_global_by_positive_ci() -> None:
    result = target_free.classify_decision(
        selector_identity=passing_identity_contrast(),
        selector_global=contrast(ci_lower=0.0),
        global_identity=passing_identity_contrast(),
        audit_passed=True,
    )
    assert result.label is target_free.DecisionLabel.FIXED_CORRECTION_GO


def test_below_point_zero_one_is_no_go_even_with_positive_ci() -> None:
    result = gate_fixture(mean_gain=0.009, ci_lower=0.005)
    assert result.label is target_free.DecisionLabel.NO_GO
```

- [ ] **Step 5: Implement statistics and mutually exclusive gate**

Define:

```python
class DecisionLabel(str, Enum):
    SELECTOR_GO = "SELECTOR GO"
    FIXED_CORRECTION_GO = "FIXED-CORRECTION GO"
    PARTIAL = "PARTIAL"
    NO_GO = "NO GO"


@dataclass(frozen=True)
class ContrastInterval:
    mean: float
    ci_lower: float
    ci_upper: float
    loso_min: float
    loso_max: float


@dataclass(frozen=True)
class DecisionResult:
    label: DecisionLabel
    selector_conditions: tuple[bool, ...]
    global_conditions: tuple[bool, ...]
```

Generate one scene-index bootstrap matrix with
`np.random.default_rng(42)`, reuse it for every contrast, preserve episode
multiplicity, and call `np.percentile` without changing its NumPy 1.24 linear
interpolation.

Gate in exact precedence:

1. selector all six conditions -> `SELECTOR GO`;
2. global analogous conditions 1–4 and audit condition 6 ->
   `FIXED-CORRECTION GO`;
3. either candidate has mean `>=0.01` and CI lower `>0` -> `PARTIAL`;
4. otherwise -> `NO GO`.

- [ ] **Step 6: Verify and commit Task 3**

Run the complete new test file, Ruff, ty, and diff check.

Commit:

```bash
git add prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py tests/analyze/test_llm_grid_target_free_cardinal.py
git commit -m "feat: score target-free selector controls"
```

---

### Task 4: Exact Artifact Transaction, Validator, and Fixed CLI

**Files:**
- Modify: `prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py`
- Modify: `tests/analyze/test_llm_grid_target_free_cardinal.py`

**Interfaces:**
- Consumes: all prior typed rows, lock, statistics, and decision result.
- Produces:
  - exact JSON artifact dataclasses below;
  - `ArtifactBundle`;
  - `TargetFreeArgs(Tap)`;
  - `build_artifacts(selector_lock: SelectorLock, assignments: Sequence[SelectorAssignment], angle_rows: Sequence[AngleScoreRow], episode_rows: Sequence[EpisodeScoreRow], summary: SummaryArtifact, bootstrap: BootstrapArtifact, manifest_inputs: ManifestInputs) -> ArtifactBundle`;
  - `publish_artifacts(output_dir: Path, artifacts: ArtifactBundle, validation_inputs: ValidationInputs) -> None`;
  - `validate_artifact_directory(output_dir: Path, *, expected_development_episodes: int, expected_development_scenes: int, expected_development_valid: int, expected_development_invalid: int, expected_test_episodes: int, expected_test_scenes: int, expected_test_valid: int, expected_test_invalid: int) -> None`;
  - `main(argv: Optional[Sequence[str]] = None) -> None`.

**Exact nested JSON dataclasses:**

Implement these field sets before any serializer:

```text
PopulationSummary:
  episodes, scenes, valid, invalid

PopulationArtifact:
  development, test

CandidateMeans:
  all_iou, object_iou, region_iou

MeansSummary:
  identity, primary, global_correction, direct, random_expected,
  soft_identity, soft_aggregate, oracle

ContrastSummary:
  mean, ci_lower, ci_upper, loso_min, loso_max

ContrastCollection:
  primary_identity, primary_global, global_identity, direct_identity,
  soft_aggregate_identity

SemanticFamilySummary:
  primary_object_delta, primary_region_delta, global_object_delta,
  global_region_delta

CellMetricSummary:
  precision, recall, f1

CellMetricsCollection:
  identity, primary, global_correction, direct, soft_identity, soft_aggregate

SupportSummary:
  predicted_support, target_support, in_frame_support, out_of_frame_support,
  union, soft_predicted_mass, soft_target_mass, soft_intersection_mass,
  soft_union_mass, invalid_count, empty_prediction_count, empty_target_count,
  empty_union_count

DirectionSummary:
  identity_cosine, primary_cosine, global_cosine, direct_cosine

AngleSummary:
  primary_counts, global_counts, direct_counts, oracle_counts,
  primary_oracle_agreement, direct_oracle_agreement, heading_angle_counts

CostSummary:
  primary_warps, global_warps, direct_warps, soft_warps,
  maximum_materialized_array_bytes

OracleGainSummary:
  available, primary_fraction, global_fraction, direct_fraction, soft_fraction

DecisionSummary:
  label, selector_conditions, global_conditions

ChosenMappingArtifact:
  sign, offset_degrees, development_mean_iou, tied_candidate_count

ChosenGlobalAngleArtifact:
  angle_degrees, development_mean_iou, tied_candidate_count

DevelopmentArtifact:
  population, candidate_scores

LockProtocolArtifact:
  angle_order, mapping_order, heading_tie_order, invalid_angle

SelectorLockArtifact:
  schema_version, chosen_mapping, chosen_global_angle, development, protocol

SummaryArtifact:
  schema_version, population, means, contrasts, semantic_families,
  cell_metrics, support, directions, angles, cost, oracle_gain, decision

BootstrapArtifact:
  schema_version, seed, repetitions, scene_ids, intervals,
  leave_one_scene_out

BootstrapIntervals:
  primary_identity, primary_global, global_identity, direct_identity,
  soft_aggregate_identity

LeaveOneSceneOutSummary:
  primary_identity, primary_global, global_identity

SourceArtifact:
  path, sha256, dataset, split

SourcesArtifact:
  development, test

ProtocolArtifact:
  cache_model_key, cognitive_map_namespace, scale, raster_shape,
  object_channels, region_channels, angle_order, mapping_order,
  heading_tie_order, invalid_angle, bootstrap_seed, bootstrap_repetitions,
  gate_threshold, schema_version, target_free_limitations

SchemaArtifact:
  csv_headers, json_top_level_keys, float_format, row_order,
  json_serialization

CsvHeadersArtifact:
  development_mapping_scores, test_assignments, test_angle_scores,
  test_episode_scores

JsonTopLevelKeysArtifact:
  selector_lock, summary, bootstrap, manifest

ArtifactEntry:
  name, sha256, size_bytes, data_rows

ManifestArtifact:
  schema_version, sources, git_commit, protocol, population, schemas, artifacts

ManifestInputs:
  sources, git_commit, protocol, population, schemas

ValidationInputs:
  development_contract, test_contract, cache_dir, cache_model_key,
  cognitive_map_namespace, quiet

ArtifactBundle:
  manifest_json, development_mapping_scores_csv, selector_lock_json,
  test_assignments_csv, test_angle_scores_csv, test_episode_scores_csv,
  summary_json, bootstrap_json, target_free_control_intervals_png
```

Mappings inside JSON use only `dataclasses.asdict` from these frozen typed
objects; no loose input mappings enter the artifact builder.

- [ ] **Step 1: Write failing serializer and plot tests**

Assert the exact nine filenames, exact headers, canonical JSON keys, schema
version, row counts, hashes, PNG decoding, finite values, and summary/gate
agreement. Add hand-derived tests that distinguish exact soft
identity/aggregate object and region IoUs from the boolean per-angle family
means. Extend `EpisodeScoreRow`, `test_episode_scores.csv`, and aggregate
candidate means with those four v2 fields. Load the PNG with Matplotlib, not a
signature-only check.

- [ ] **Step 2: Write failing tamper and transaction tests**

For each artifact, modify bytes or a semantic value and require validator
failure. Cover assignment re-generation, lock re-development, hash mismatch,
unknown/missing JSON key, row reorder, wrong bootstrap, wrong decision,
malformed PNG, official output already present, temp-build failure, and atomic
publish leaving no partial official directory.

- [ ] **Step 3: Run artifact tests and confirm RED**

Run only artifact/transaction tests. Expected: missing artifact APIs.

- [ ] **Step 4: Implement canonical serializers and artifact builder**

Use:

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8") + b"\n"
```

Use one CSV cell formatter for booleans, integers, and `.17g` finite floats
with negative zero normalized. Build the interval plot from frozen summary
values and close its figure.

- [ ] **Step 5: Implement independent validation and atomic publication**

`validate_artifact_directory` requires exact filenames; parses all bytes;
recomputes hashes, development selection, test assignments, per-angle and
per-episode scores, bootstrap, LOSO, summary, and decision; and rejects any
mismatch. It accepts exact expected development/test contracts as keyword-only
arguments.

`publish_artifacts` fails if the official directory exists, creates one
temporary sibling directory, writes all nine files there, validates the
temporary directory, and atomically renames it to the official path. On
failure it removes only its exact temporary sibling.

- [ ] **Step 6: Implement the fixed Tap CLI and orchestration**

`TargetFreeArgs` exposes typed defaults but `_validate_args` requires every
scientific value and path to equal the frozen protocol. `_run` performs:

1. preflight;
2. development load and lock;
3. runtime-only test load;
4. in-memory assignment CSV serialization and SHA-256 sealing;
5. test target load;
6. scoring/statistics/decision;
7. artifact build;
8. atomic publish and second independent validation.

The CLI must not accept `--limit` and must not provide overwrite/resume flags.

- [ ] **Step 7: Run the complete verification and commit Task 4**

Run:

```bash
pytest tests/analyze/test_llm_grid_target_free_cardinal.py -q
pytest tests/analyze -q
ruff check prior/analyze/d2026_07_28 tests/analyze
ty check prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py tests/analyze/test_llm_grid_target_free_cardinal.py
git diff --check
```

Commit:

```bash
git add prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py tests/analyze/test_llm_grid_target_free_cardinal.py
git commit -m "feat: publish target-free selector audit"
```

---

### Task 5: Fixed Run, Independent Audit, and Daily Finding

**Files:**
- Modify after the run: `docs/daily/2026-07-28.md`
- Generated, ignored:
  `outputs/llm_grid_analysis/target_free_cardinal_selector_r2r_epoch2/`

**Interfaces:**
- Consumes: the committed fixed CLI and exact validator from Task 4.
- Produces: one validated official artifact directory and one Chinese finding.

- [ ] **Step 1: Run a clean pre-experiment verification**

Run:

```bash
pytest tests/analyze/test_llm_grid_target_free_cardinal.py -q
pytest tests/analyze -q
ruff check prior/analyze/d2026_07_28 tests/analyze
ty check prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py tests/analyze/test_llm_grid_target_free_cardinal.py
git diff --check
git status --short --branch
test ! -e outputs/llm_grid_analysis/target_free_cardinal_selector_r2r_epoch2
```

Stop and repair any failure. Do not remove or overwrite an existing official
output directory.

- [ ] **Step 2: Run the fixed command exactly once**

Run exactly:

```bash
python -m prior.analyze.d2026_07_28.llm_grid_target_free_cardinal
```

Do not rerun it to fix notes, plots, formatting, or interpretation. Any
artifact failure after this command is a failed experiment requiring explicit
diagnosis before a new protocol/run.

- [ ] **Step 3: Independently validate and inspect the result**

Run the public validator in a fresh Python process with exact
`778/53/770/8` and `1839/11/1830/9` contracts. Confirm exactly nine output
files. Inspect `selector_lock.json`, `summary.json`, `bootstrap.json`, and the
decoded interval plot. Recompute the selected mapping, global angle, primary
contrasts, all decision conditions, and label without trusting console text.

- [ ] **Step 4: Obtain independent scientific and code review**

Give one reviewer the frozen spec plus artifacts, but not the implementation
author's interpretation. Require it to check target independence, population,
development/test separation, bootstrap, LOSO, gate arithmetic, and conclusion.
Give a second reviewer the final code/test diff and require
Critical/Important/Minor findings.

Repair code or documentation findings with TDD without rerunning the fixed
experiment or modifying artifact bytes. If a valid repair would change an
artifact, stop and report that the run is invalid instead of editing evidence.

- [ ] **Step 5: Record the finding and checklist status in Chinese**

Append to `docs/daily/2026-07-28.md`:

- exact chosen heading mapping and global angle;
- development and test populations/provenance;
- identity, primary, global, direct, soft, and oracle means;
- paired contrasts, CIs, LOSO ranges, object/region results, and support;
- all gate booleans and the literal decision;
- leakage audit and assignment hash;
- limitations and the one authorised next action.

Mark P3 complete. Mark P4 active only for `SELECTOR GO` or
`FIXED-CORRECTION GO`; for `PARTIAL` or `NO GO`, record the separately
preregistered follow-up authorised by the frozen rules instead.

- [ ] **Step 6: Final verification and finding commit**

Run fresh:

```bash
pytest tests/analyze/test_llm_grid_target_free_cardinal.py -q
pytest tests/analyze -q
ruff check prior/analyze/d2026_07_28 tests/analyze
ty check prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py tests/analyze/test_llm_grid_target_free_cardinal.py
python -c "from pathlib import Path; from prior.analyze.d2026_07_28.llm_grid_target_free_cardinal import validate_artifact_directory; validate_artifact_directory(Path('outputs/llm_grid_analysis/target_free_cardinal_selector_r2r_epoch2'), expected_development_episodes=778, expected_development_scenes=53, expected_development_valid=770, expected_development_invalid=8, expected_test_episodes=1839, expected_test_scenes=11, expected_test_valid=1830, expected_test_invalid=9)"
git diff --check
```

Commit only the finding and any review-driven code/test fix:

```bash
git add docs/daily/2026-07-28.md prior/analyze/d2026_07_28/llm_grid_target_free_cardinal.py tests/analyze/test_llm_grid_target_free_cardinal.py
git commit -m "docs: record target-free selector finding"
```

Do not push unless the user asks.
