# LLM-Grid Cross-Fitted Registration Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the frozen cache-only experiment that tests whether
cardinal rotations selected on one semantic family transfer to the held-out
family and are specifically stronger around the true episode start.

**Architecture:** Add one date-scoped experiment module that reuses
`prior.analyze.llm_grid_registration` unchanged. Typed immutable records carry
episode inputs, family scores, cross-fit results, pivot assignments, and
inference outputs; private helpers perform deterministic matching, paired
scene inference, validation, and deterministic artifact writing. One date-scoped test
module exercises every scientific invariant before the full fixed-population
CLI is run.

**Tech Stack:** Python 3.8, `dataclasses`, `enum.Enum`, `hashlib`, `csv`,
`json`, NumPy, Matplotlib, `tap.Tap`, pytest, Ruff, and ty; no new dependency.

## Global Constraints

- Work directly on `main`; do not create a worktree or experiment branch.
- Create only
  `prior/analyze/d2026_07_28/__init__.py`,
  `prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py`, and
  `tests/analyze/test_llm_grid_transform_crossfit.py`; modify
  `docs/daily/2026-07-28.md` only after the verified full run.
- Do not modify `prior/analyze/llm_grid_registration.py`, the July 27
  diagnosis, their tests, or existing generated artifacts.
- Use R2R `val_unseen`, cache key
  `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2`, target namespace
  `gt.legacy.r1p5.direction5.v1`, exactly 1,839 episodes, 11 scenes, 1,830
  schema-valid predictions, 9 schema-invalid-or-missing predictions, shape
  `37×50×50`, scale 2, and 1 m cells.
- Fix object channels to `[0, 27)`, region channels to `[27, 37)`, angle order
  to `(0.0, 90.0, 180.0, 270.0)`, shuffle seed to 43, bootstrap repetitions
  to 10,000, and bootstrap seed to 42.
- Preserve invalid or missing predictions as explicit empty predictions.
  Catch only `FileNotFoundError` and `LLMGridValidationError`; fail on all
  unexpected errors and malformed scientific inputs.
- Warp all 37 channels once per episode/pivot/angle, then score sliced
  `WarpedGrid` instances with recalculated family input support.
- The primary endpoint is the episode-level mean of the two directional
  held-out deltas. Bootstrap scenes with replacement while preserving episode
  weighting and pairing.
- Full GO requires all five frozen gate conditions from the design; no
  post-result threshold, seed, fold, or population changes.
- Write artifacts only after all population, score, inference, and path
  invariants pass.
- Keep public APIs typed and narrow. Do not use `Any`, `getattr`, wildcard
  imports, import fallbacks, suppressions, compatibility wrappers, or silent
  defaults.
- Begin each new Python file with `from __future__ import annotations`; use
  `class Name(str, Enum)` for string enums because the repository is pinned to
  Python 3.8 and does not provide `enum.StrEnum`.
- Use `typing.Tuple[float, ...]` for `Tap` class fields because Tap resolves
  those annotations at runtime under Python 3.8; postponed built-in generic
  annotations remain acceptable for ordinary dataclasses and functions.

---

### Task 1: Typed Cross-Fit Scoring Core

**Files:**

- Create: `prior/analyze/d2026_07_28/__init__.py`
- Create: `prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py`
- Create: `tests/analyze/test_llm_grid_transform_crossfit.py`

**Interfaces:**

- Consumes:
  `WarpedGrid`, `RasterScore`, `SpatialBounds`,
  `warp_grid_about_pivot`, and `score_warped_grid` from
  `prior.analyze.llm_grid_registration`.
- Produces:
  `PivotMode`, `Direction`, `EpisodeCase`, `FamilyAngleScore`,
  `CrossFitResult`, `score_angle_families()`, and `crossfit_scores()`.

- [ ] **Step 1: Write failing scoring tests**

Add imports for the planned interfaces and this literal score helper:

```python
def family_score(
    angle: float, *, object_iou_tenths: int, region_iou_tenths: int
) -> FamilyAngleScore:
    bounds = SpatialBounds(0, 50, 0, 50)

    def raster(intersection: int) -> RasterScore:
        return RasterScore(
            intersection=intersection,
            union=10,
            predicted_support=intersection,
            target_support=10,
            in_frame_support=intersection,
            out_of_frame_support=0,
        )

    return FamilyAngleScore(
        angle_degrees=angle,
        bounds=bounds,
        object_input_support=object_iou_tenths,
        region_input_support=region_iou_tenths,
        object_score=raster(object_iou_tenths),
        region_score=raster(region_iou_tenths),
    )
```

Use canonical `37×50×50` boolean prediction/target fixtures. Write these
tests:

```python
def test_crossfit_selection_uses_only_declared_family() -> None:
    scores = (
        family_score(0.0, object_iou_tenths=2, region_iou_tenths=8),
        family_score(90.0, object_iou_tenths=9, region_iou_tenths=1),
        family_score(180.0, object_iou_tenths=1, region_iou_tenths=1),
        family_score(270.0, object_iou_tenths=1, region_iou_tenths=1),
    )

    object_to_region = crossfit_scores(
        scores, Direction.OBJECT_TO_REGION
    )
    region_to_object = crossfit_scores(
        scores, Direction.REGION_TO_OBJECT
    )

    assert object_to_region.selected_angle_degrees == 90.0
    assert object_to_region.heldout_selected_iou == 0.1
    assert object_to_region.delta_iou == pytest.approx(-0.7)
    assert region_to_object.selected_angle_degrees == 0.0
    assert region_to_object.heldout_selected_iou == 0.2
    assert region_to_object.delta_iou == 0.0
```

```python
def test_crossfit_tie_retains_declared_identity_angle() -> None:
    scores = (
        family_score(0.0, object_iou_tenths=0, region_iou_tenths=0),
        family_score(90.0, object_iou_tenths=0, region_iou_tenths=0),
        family_score(180.0, object_iou_tenths=0, region_iou_tenths=0),
        family_score(270.0, object_iou_tenths=0, region_iou_tenths=0),
    )

    result = crossfit_scores(scores, Direction.OBJECT_TO_REGION)

    assert result.selected_angle_degrees == 0.0
    assert result.selector_margin == 0.0
    assert result.delta_iou == 0.0
```

Construct a non-central pivot fixture and assert that
`score_angle_families()`:

- returns one row for every angle in caller order;
- gives object and region scores the same `SpatialBounds`;
- records object input support from channels `[0, 27)` and region input
  support from `[27, 37)`;
- retains rotated out-of-frame family support as false positives;
- rejects non-boolean grids, noncanonical channel counts, unequal target
  spatial shape, duplicate/missing identity angle, and non-finite pivots.

- [ ] **Step 2: Run the scoring tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py \
  -k "crossfit_selection or crossfit_tie or angle_families" -v
```

Expected: collection fails because the date-scoped module or named interfaces
do not exist.

- [ ] **Step 3: Implement the typed scoring records**

Create an empty package initializer. In the experiment module define:

```python
class PivotMode(str, Enum):
    TRUE_START = "true_start"
    MAP_CENTER = "map_center"
    SHUFFLED_START = "shuffled_start"


class Direction(str, Enum):
    OBJECT_TO_REGION = "object_to_region"
    REGION_TO_OBJECT = "region_to_object"


@dataclass(frozen=True)
class EpisodeCase:
    split: str
    scene_id: str
    example_id: str
    schema_valid: bool
    predicted_grid: NDArray[np.bool_]
    target_grid: NDArray[np.bool_]
    true_start_pivot: tuple[float, float]


@dataclass(frozen=True)
class FamilyAngleScore:
    angle_degrees: float
    bounds: SpatialBounds
    object_input_support: int
    region_input_support: int
    object_score: RasterScore
    region_score: RasterScore


@dataclass(frozen=True)
class CrossFitResult:
    direction: Direction
    selected_angle_degrees: float
    second_angle_degrees: float
    selector_identity_iou: float
    selector_selected_iou: float
    selector_margin: float
    heldout_identity_iou: float
    heldout_selected_iou: float
    delta_iou: float
    selector_predicted_support_empty: bool
    selector_target_support_empty: bool
    selector_union_empty: bool
    heldout_predicted_support_empty: bool
    heldout_target_support_empty: bool
    heldout_union_empty: bool
```

Validate every record in `__post_init__`: strings are nonempty, grids are
canonical boolean arrays, IoUs are finite and within `[0, 1]`, deltas are
finite and within `[-1, 1]`, margins are finite and within `[0, 1]`, supports
are nonnegative, and stored deltas/margins equal their operands.
`crossfit_scores()` validates that identity exists, angles are unique, and
selected/second angles come from the supplied score rows.

- [ ] **Step 4: Implement one-warp family scoring and held-out reuse**

Implement:

```python
def score_angle_families(
    predicted_grid: NDArray[np.bool_],
    target_grid: NDArray[np.bool_],
    pivot: tuple[float, float],
    angles: tuple[float, ...],
) -> tuple[FamilyAngleScore, ...]:
```

For each angle call `warp_grid_about_pivot()` once on all channels. Build two
family views:

```python
object_warp = WarpedGrid(
    grid=warped.grid[:27],
    bounds=warped.bounds,
    input_support=int(np.count_nonzero(predicted_grid[:27])),
)
region_warp = WarpedGrid(
    grid=warped.grid[27:],
    bounds=warped.bounds,
    input_support=int(np.count_nonzero(predicted_grid[27:])),
)
```

Score them against the corresponding target slices. Implement:

```python
def crossfit_scores(
    scores: Sequence[FamilyAngleScore],
    direction: Direction,
) -> CrossFitResult:
```

Select the highest selector-family IoU with stable caller-order ties, identify
the next row by the same stable ordering, and reuse the selected row's
held-out score. Calculate all six empty flags from the identity row:
predicted support `== 0`, target support `== 0`, and union `== 0`.

- [ ] **Step 5: Run scoring tests and confirm GREEN**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py \
  -k "crossfit_selection or crossfit_tie or angle_families" -v
ruff check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
ty check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
```

Expected: selected tests pass; Ruff and ty report no diagnostics.

- [ ] **Step 6: Commit the scoring core**

```bash
git add prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
git commit -m "feat: add cross-fit scoring core"
```

---

### Task 2: Deterministic Pivot Conditions

**Files:**

- Modify: `prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py`
- Modify: `tests/analyze/test_llm_grid_transform_crossfit.py`

**Interfaces:**

- Consumes: `EpisodeCase` from Task 1 and the frozen SHA-256 matching contract.
- Produces: `PivotAssignment`, `build_pivot_assignments()`,
  `pivot_for_mode()`, `EpisodePivotResult`, and `evaluate_episode()`.

- [ ] **Step 1: Write failing matching and pivot tests**

Add a fixed synthetic scene with four receiver identities and four distinct
start pivots. Freeze the complete expected mapping produced by the design's
SHA-256 ordering and augmenting-path algorithm:

```python
def test_pivot_assignment_matches_frozen_sha256_fixture() -> None:
    assignments = build_pivot_assignments(four_episode_cases())

    assert [
        (row.example_id, row.donor_example_id)
        for row in assignments
    ] == [
        ("episode-a", "episode-b"),
        ("episode-b", "episode-c"),
        ("episode-c", "episode-d"),
        ("episode-d", "episode-a"),
    ]
```

The fixture uses scene ID `scene-1`, episode IDs `episode-a` through
`episode-d`, and respective pivots `(1.0, 1.0)` through `(4.0, 4.0)`. The
expected list above was calculated from the frozen digest and DFS rules and
must not be changed in response to production output.

Also test:

- input order does not change assignment;
- assignments stay within scene, use every donor once, and never preserve
  identity or an equal pivot;
- duplicate pivot coordinates remove all corresponding edges;
- a one-episode scene and an impossible duplicate-pivot scene raise
  `ValueError("scene <id> has no valid shuffled-start perfect matching")`;
- `pivot_for_mode()` returns the exact true start, `(25.0, 25.0)`, or assigned
  shuffled pivot;
- a constructed off-centre episode scores better at the true start than at
  the map center and a supplied wrong pivot;
- an invalid `EpisodeCase` with an empty prediction remains in all three
  pivots and both directions with zero delta.

- [ ] **Step 2: Run the pivot tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py \
  -k "pivot or matching or invalid" -v
```

Expected: failures identify missing pivot assignment/evaluation interfaces.

- [ ] **Step 3: Implement deterministic matching**

Define:

```python
@dataclass(frozen=True)
class PivotAssignment:
    scene_id: str
    example_id: str
    donor_example_id: str
    true_pivot: tuple[float, float]
    assigned_pivot: tuple[float, float]


def build_pivot_assignments(
    episodes: Sequence[EpisodeCase],
) -> tuple[PivotAssignment, ...]:
```

Implement the design literally:

- identity bytes are
  `(scene_id + "\0" + example_id).encode("utf-8")`;
- scene order is ascending UTF-8 bytes;
- receiver keys are
  `sha256(b"43\0receiver\0" + identity_bytes).digest()`;
- donor keys are
  `sha256(b"43\0donor\0" + receiver_bytes + b"\0" + donor_bytes).digest()`;
- digest ties use identity bytes;
- a receiver-donor edge exists only if identities and exact pivot tuples
  differ;
- receiver-ordered DFS matching uses a fresh visited-donor set for each
  top-level receiver and recursively reassigns an occupied donor.

Sort returned records by `(scene_id UTF-8 bytes, example_id UTF-8 bytes)` and
validate the one-to-one/multiset invariants after matching.

- [ ] **Step 4: Implement episode evaluation across all pivots**

Define:

```python
@dataclass(frozen=True)
class EpisodePivotResult:
    split: str
    scene_id: str
    example_id: str
    schema_valid: bool
    pivot_mode: PivotMode
    pivot: tuple[float, float]
    donor_example_id: str | None
    angle_scores: tuple[FamilyAngleScore, ...]
    object_to_region: CrossFitResult
    region_to_object: CrossFitResult

    @property
    def symmetric_delta(self) -> float:
        return 0.5 * (
            self.object_to_region.delta_iou
            + self.region_to_object.delta_iou
        )
```

Implement:

```python
def evaluate_episode(
    episode: EpisodeCase,
    assignment: PivotAssignment,
    angles: tuple[float, ...],
) -> tuple[EpisodePivotResult, ...]:
```

Return results in `PivotMode` declaration order. Require the assignment
identity to match the episode, evaluate each pivot once, and derive both
cross-fit directions from the same family-angle score tuple.

- [ ] **Step 5: Run pivot tests and confirm GREEN**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py \
  -k "pivot or matching or invalid" -v
ruff check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
ty check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
```

Expected: selected tests pass; Ruff and ty report no diagnostics.

- [ ] **Step 6: Commit pivot controls**

```bash
git add prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py \
  tests/analyze/test_llm_grid_transform_crossfit.py
git commit -m "feat: add deterministic pivot controls"
```

---

### Task 3: Paired Scene Inference and Frozen Gate

**Files:**

- Modify: `prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py`
- Modify: `tests/analyze/test_llm_grid_transform_crossfit.py`

**Interfaces:**

- Consumes: complete `EpisodePivotResult` triples from Task 2.
- Produces: `GateDecision`, `EndpointEstimate`,
  `scene_bootstrap_multiplicities()`, `_bootstrap_episode_macro()`,
  `summarize_results()`,
  `bootstrap_results()`, `leave_one_scene_out_ranges()`, and
  `classify_gate()`.

- [ ] **Step 1: Write failing inference tests**

Build small typed results for two unequal scenes: scene A with one episode and
scene B with three. Assert:

```python
def test_bootstrap_replicates_preserve_episode_weighting() -> None:
    scene_ids = ("A", "B")
    multiplicities = np.asarray(((1, 1), (2, 0), (0, 2)), dtype=np.int64)
    values = {"A": (1.0,), "B": (0.0, 0.0, 0.0)}

    replicates = _bootstrap_episode_macro(values, scene_ids, multiplicities)

    np.testing.assert_allclose(replicates, (0.25, 1.0, 0.0))
```

Add tests proving:

- `scene_bootstrap_multiplicities()` returns sorted scenes and identical
  seed-42 matrices on repeated calls;
- one shared matrix is used for both directions, all pivots, symmetric
  endpoints, and paired start-minus-control contrasts;
- paired contrasts are formed per episode before scene aggregation;
- leave-one-scene-out values are episode-macro means after removing one
  scene, and only their min/max are reported;
- summary rows contain directional means, symmetric means, angle counts, and
  all six empty rates without dropping invalid rows;
- all values and intervals are finite and the population identities/pivot
  triples are complete.

Parameterize gate fixtures:

```python
@pytest.mark.parametrize(
    ("mean", "lower", "object_delta", "region_delta",
     "center_lower", "shuffle_lower", "expected"),
    (
        (0.011, 0.001, 0.002, 0.020, 0.001, 0.001, GateDecision.GO),
        (0.011, 0.001, -0.001, 0.023, 0.001, 0.001,
         GateDecision.PARTIAL_EVIDENCE),
        (0.009, 0.001, 0.002, 0.016, 0.001, 0.001,
         GateDecision.NO_GO),
        (0.011, 0.000, 0.002, 0.020, 0.001, 0.001,
         GateDecision.NO_GO),
    ),
)
def test_gate_is_literal(
    mean: float,
    lower: float,
    object_delta: float,
    region_delta: float,
    center_lower: float,
    shuffle_lower: float,
    expected: GateDecision,
) -> None:
    symmetric = estimate(mean=mean, lower=lower)
    center = estimate(mean=0.0, lower=center_lower)
    shuffle = estimate(mean=0.0, lower=shuffle_lower)

    assert classify_gate(
        true_start_symmetric=symmetric,
        true_start_object_to_region_mean=object_delta,
        true_start_region_to_object_mean=region_delta,
        start_minus_center=center,
        start_minus_shuffled=shuffle,
    ) is expected
```

Here `estimate()` is a test helper that returns `EndpointEstimate` with the
given mean/lower bound, `ci_upper=lower + 0.1`, and both LOSO bounds equal to
the mean.

- [ ] **Step 2: Run inference tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py \
  -k "bootstrap or leave_one or summary or gate" -v
```

Expected: failures identify missing inference and gate interfaces.

- [ ] **Step 3: Implement typed summaries and shared bootstrap**

Define:

```python
class GateDecision(str, Enum):
    GO = "GO"
    PARTIAL_EVIDENCE = "partial_evidence"
    NO_GO = "NO_GO"


@dataclass(frozen=True)
class EndpointEstimate:
    mean: float
    ci_lower: float
    ci_upper: float
    leave_one_scene_out_min: float
    leave_one_scene_out_max: float
```

Implement `scene_bootstrap_multiplicities()` with
`np.random.default_rng(42)` and 10,000 draws of 11 scenes, matching the July 27
implementation. Implement the internal episode-macro replicate exactly as:

```python
scene_sums = np.asarray([sum(values[scene]) for scene in scene_ids])
scene_counts = np.asarray([len(values[scene]) for scene in scene_ids])
return (
    multiplicities @ scene_sums
    / (multiplicities @ scene_counts)
)
```

Reject empty scenes, missing scene keys, non-finite values, non-integer or
negative multiplicities, zero replicate denominators, duplicate episode
identities, and incomplete pivot triples.

- [ ] **Step 4: Implement summaries, LOSO, and the literal gate**

Use a single multiplicity matrix to calculate:

- object→region, region→object, and symmetric deltas for each pivot;
- true-start symmetric minus map-center symmetric;
- true-start symmetric minus shuffled-start symmetric.

Use `np.percentile(replicates, (2.5, 97.5))`. Calculate every paired contrast
by matching `(scene_id, example_id)` first. Implement the gate in fixed order:

```python
def classify_gate(
    *,
    true_start_symmetric: EndpointEstimate,
    true_start_object_to_region_mean: float,
    true_start_region_to_object_mean: float,
    start_minus_center: EndpointEstimate,
    start_minus_shuffled: EndpointEstimate,
) -> GateDecision:
    symmetric_passes = (
        true_start_symmetric.mean >= 0.01
        and true_start_symmetric.ci_lower > 0.0
    )
    if not symmetric_passes:
        return GateDecision.NO_GO
    if (
        true_start_object_to_region_mean > 0.0
        and true_start_region_to_object_mean > 0.0
        and start_minus_center.ci_lower > 0.0
        and start_minus_shuffled.ci_lower > 0.0
    ):
        return GateDecision.GO
    return GateDecision.PARTIAL_EVIDENCE
```

- [ ] **Step 5: Run inference tests and confirm GREEN**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py \
  -k "bootstrap or leave_one or summary or gate" -v
ruff check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
ty check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
```

Expected: selected tests pass; Ruff and ty report no diagnostics.

- [ ] **Step 6: Commit inference**

```bash
git add prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py \
  tests/analyze/test_llm_grid_transform_crossfit.py
git commit -m "feat: add paired cross-fit inference"
```

---

### Task 4: Fixed Population Loader and CLI Contract

**Files:**

- Modify: `prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py`
- Modify: `tests/analyze/test_llm_grid_transform_crossfit.py`

**Interfaces:**

- Consumes:
  `load_llm_grid_examples`, `LLMGridDataset`, `parse_grid_text`,
  `llm_navigation_prediction_path`, `llm_navigation_split_dir`,
  `validate_llm_navigation_manifest`, `GRID_SCALE`, `GRID_SHAPE`, and
  `CELL_SIZE` from their maintained modules.
- Produces: `CrossFitArgs`, `_validate_args()`, `_prediction_manifest_path()`,
  and `_load_episode_cases()`.

- [ ] **Step 1: Write failing loader and argument tests**

Parameterize mutations of the fixed scientific arguments and assert each is
rejected:

```python
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("cache_model_key", "other"),
        ("cognitive_map_namespace", "other.namespace"),
        ("angles", (0.0, 90.0)),
        ("shuffle_seed", 44),
        ("bootstrap_repetitions", 9999),
        ("bootstrap_seed", 41),
    ),
)
def test_args_reject_scientific_overrides(field: str, value: object) -> None:
    args = fixed_args()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        _validate_args(args)
```

Also test:

- the CLI exposes dashed options through
  `CrossFitArgs(underscores_to_dashes=True).parse_args(argv)`;
- it has no `limit` or partial-population argument;
- equal, descendant, or ancestor cache/output paths fail before writes;
- manifest validation requires dataset R2R, split `val_unseen`, generator
  `llm-grid`, scale 2, and the fixed cache key;
- loader order mismatches, duplicate identities, wrong target/prediction
  shapes, non-finite starts, nonempty invalid predictions, wrong population,
  wrong scene count, or wrong valid/invalid counts fail explicitly;
- a missing prediction and an `LLMGridValidationError` become an empty
  schema-invalid `EpisodeCase`;
- an unexpected `OSError` or runtime parser error propagates unchanged.

- [ ] **Step 2: Run loader tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py \
  -k "args or loader or manifest or population" -v
```

Expected: failures identify missing loader/CLI interfaces.

- [ ] **Step 3: Implement the fixed `Tap` contract**

Define `CrossFitArgs(Tap)` with only:

```python
cache_dir: Path = Path("data/llm_navigation")
cache_model_key: str = _DEFAULT_CACHE_MODEL_KEY
cognitive_map_namespace: str = _DEFAULT_COGNITIVE_MAP_NAMESPACE
output_dir: Path = Path(
    "outputs/llm_grid_analysis/"
    "start_centered_crossfit_controls_r2r_rxr_epoch2"
)
angles: Tuple[float, ...] = _DEFAULT_ANGLES
shuffle_seed: int = 43
bootstrap_repetitions: int = 10_000
bootstrap_seed: int = 42
quiet: bool = False
```

Validate every fixed value and path disjointness before reading a prediction
or creating the output directory.

- [ ] **Step 4: Implement strict population loading**

Mirror the July 27 loader's authoritative APIs without adding a shared cache
or changing that module. Load only R2R `val_unseen`. Convert each target and
prediction to canonical boolean grids. Validate the two start coordinates are
finite and divide each by `CELL_SIZE * GRID_SCALE` for the exact scale-2
pivot. Do not import the July 27 experiment module. Catch exactly:

```python
except (FileNotFoundError, LLMGridValidationError):
    schema_valid = False
```

After loading, validate exact identity uniqueness, 1,839 rows, 11 scenes,
1,830 valid rows, and 9 invalid-or-missing rows. There is no smoke path in the
public runner; synthetic unit tests inject typed cases below the loader.

- [ ] **Step 5: Run loader tests and confirm GREEN**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py \
  -k "args or loader or manifest or population" -v
ruff check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
ty check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
```

Expected: selected tests pass; Ruff and ty report no diagnostics.

- [ ] **Step 6: Commit the fixed loader**

```bash
git add prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py \
  tests/analyze/test_llm_grid_transform_crossfit.py
git commit -m "feat: load fixed cross-fit population"
```

---

### Task 5: Validated Artifact Transaction

**Files:**

- Modify: `prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py`
- Modify: `tests/analyze/test_llm_grid_transform_crossfit.py`

**Interfaces:**

- Consumes: typed cases, assignments, pivot results, summaries, bootstrap
  intervals, and gate decision from Tasks 1–4.
- Produces: `_run_cases()`, `_validate_analysis()`, `_write_artifacts()`,
  `validate_artifact_directory()`, `run_crossfit()`, `main()`, and the seven
  fixed output artifacts.

- [ ] **Step 1: Write failing artifact tests**

Use three small synthetic scenes and an injected population contract to call:

```python
result = _run_cases(
    fixed_args(tmp_path),
    synthetic_cases,
    expected_population=len(synthetic_cases),
    expected_scenes=3,
    expected_valid=len(synthetic_cases),
    expected_invalid=0,
    prediction_manifest_path=manifest_path,
)
```

Assert the exact artifact set:

```python
assert {path.name for path in result.output_dir.iterdir()} == {
    "manifest.json",
    "angle_scores.csv",
    "crossfit_results.csv",
    "pivot_assignments.csv",
    "summary.json",
    "bootstrap.json",
    "crossfit_control_intervals.png",
}
```

Assert:

- `angle_scores.csv` has
  `episodes × 3 pivots × 4 angles` rows and all object/region intersection,
  union, predicted/target/in-frame/out-of-frame/input support, IoU, and bounds
  columns;
- `crossfit_results.csv` has
  `episodes × 3 pivots × 2 directions` rows and all score, selected-angle,
  margin, delta, donor, pivot, schema, and six empty-flag columns;
- `pivot_assignments.csv` has one stable-order row per episode and its exact
  file SHA-256 appears in `manifest.json`;
- JSON keys are sorted with trailing newlines; CSV/JSON output is byte-stable
  across two output directories;
- `summary.json` includes population/status counts, pivot/direction endpoint
  means, symmetric endpoints, angle counts, empty rates, LOSO ranges,
  start-specific contrasts, five gate booleans, and final decision;
- `bootstrap.json` includes the fixed contract and all directional,
  symmetric, and contrast intervals;
- the manifest includes every provenance field and the explicit statement
  that selection uses target raster cells and is not deployable;
- malformed row counts, identities, pivot assignments, support partitions,
  non-finite metrics, angle membership, contrast arithmetic, intervals, gate
  inputs, or absent prediction manifest fail before output creation;
- an existing nonempty output directory fails explicitly rather than mixing
  runs.

Exercise `validate_artifact_directory()` against the synthetic output and
against one deliberately altered CSV byte. The intact directory must pass;
the altered directory must fail with the mismatched file/row invariant.

- [ ] **Step 2: Run artifact tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py \
  -k "artifact or runner or output" -v
```

Expected: failures identify missing runner, validators, and writers.

- [ ] **Step 3: Implement complete pre-write validation**

Implement `_run_cases()` as the only synthetic seam. It must:

1. validate args, manifest, and injected population contract;
2. sort cases by UTF-8 scene/example identity;
3. build all pivot assignments;
4. evaluate every case;
5. build the shared bootstrap matrix, summaries, LOSO ranges, contrasts, and
   gate;
6. build all in-memory CSV rows and JSON payloads;
7. call `_validate_analysis()` over every invariant;
8. reject an existing nonempty output directory;
9. create the directory and write artifacts.

No writer may calculate a scientific value; it serializes already validated
typed results.

- [ ] **Step 4: Implement deterministic writers and plot**

Use `csv.DictWriter` with literal header tuples,
`json.dump(payload, stream, sort_keys=True, indent=2)` plus one newline, and
`hashlib.sha256(path.read_bytes()).hexdigest()`. Write rows in
scene/example/pivot/angle/direction order. Serialize `PivotMode`, `Direction`,
and `GateDecision` explicitly through `.value`; never rely on
`str(EnumMember)`.

After the validated runner exists, implement:

```python
def run_crossfit(args: CrossFitArgs) -> dict[str, object]:
    episodes, prediction_manifest_path = _load_episode_cases(args)
    return _run_cases(
        args,
        episodes,
        expected_population=1_839,
        expected_scenes=11,
        expected_valid=1_830,
        expected_invalid=9,
        prediction_manifest_path=prediction_manifest_path,
    )


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    args = CrossFitArgs(underscores_to_dashes=True).parse_args(argv)
    result = run_crossfit(args)
    if not args.quiet:
        print(f"Wrote cross-fit artifacts to {result['output_dir']}")
    return result
```

Create one Matplotlib figure showing:

- directional and symmetric means with 95% scene-cluster intervals for true
  start, map center, and shuffled start;
- true-start-minus-center and true-start-minus-shuffled symmetric contrasts;
- a zero reference line and wording “ground-truth cross-fit diagnostic”.

Close the figure after saving. Do not label any value as deployable
performance.

- [ ] **Step 5: Run artifact tests and confirm GREEN**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py \
  -k "artifact or runner or output" -v
pytest tests/analyze/test_llm_grid_transform_crossfit.py -v
ruff check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
ty check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
```

Expected: all new tests pass; Ruff and ty report no diagnostics.

- [ ] **Step 6: Commit the artifact transaction**

```bash
git add prior/analyze/d2026_07_28/llm_grid_transform_crossfit.py \
  tests/analyze/test_llm_grid_transform_crossfit.py
git commit -m "feat: write validated cross-fit artifacts"
```

---

### Task 6: Full Verification, Fixed Run, and Chinese Finding

**Files:**

- Modify: `docs/daily/2026-07-28.md`
- Generate ignored artifacts under:
  `outputs/llm_grid_analysis/start_centered_crossfit_controls_r2r_rxr_epoch2/`

**Interfaces:**

- Consumes: the complete fixed CLI and local cached R2R population.
- Produces: validated analysis artifacts and the dated Chinese finding,
  decision, limitations, and next action.

- [ ] **Step 1: Run the complete pre-experiment verification**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py -v
pytest tests/analyze -q
ruff check prior/analyze/d2026_07_28 tests/analyze
ty check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
git diff --check
```

Expected: every command exits 0 with no test, lint, type, or whitespace
failure. Stop and repair any failure before running the experiment.

- [ ] **Step 2: Run the fixed full analysis exactly once**

First confirm the target output directory does not exist or is empty. Then
run:

```bash
python -m prior.analyze.d2026_07_28.llm_grid_transform_crossfit
```

Expected: the command reports the fixed output directory and writes exactly
the seven contracted artifacts. Do not rerun with changed scientific
parameters.

- [ ] **Step 3: Independently validate produced artifacts**

Run the production serialization validator against the output directory:

```bash
python -c 'from pathlib import Path; from prior.analyze.d2026_07_28.llm_grid_transform_crossfit import validate_artifact_directory; validate_artifact_directory(Path("outputs/llm_grid_analysis/start_centered_crossfit_controls_r2r_rxr_epoch2"), expected_population=1839, expected_scenes=11, expected_valid=1830, expected_invalid=9)'
```

The validator must verify 1,839 unique episode identities, 11 scenes, 1,830 valid
and 9 invalid-or-missing rows, 22,068 angle rows, 11,034 cross-fit rows, 1,839
pivot assignments, finite metrics, exact angle/pivot/direction membership,
support partitions, paired arithmetic, manifest/file hashes, all bootstrap
interval inputs, and the literal gate decision.

- [ ] **Step 4: Record the result in today's note**

Append a Chinese section to `docs/daily/2026-07-28.md` containing:

- the frozen question, population, object/region folds, three pivots, and
  scene-cluster inference;
- both true-start directional means and CIs;
- true-start symmetric mean, CI, and LOSO range;
- map-center and shuffled-start symmetric means;
- both paired start-specific contrasts and CIs;
- all five gate conditions and the final GO/partial evidence/NO GO decision;
- the empty-support rates and selected-angle distributions needed to
  interpret ties;
- the limitation that target raster cells still select each direction's
  angle, so the result is not deployable;
- exactly the next action authorized by the observed gate.

Do not alter the July 27 note or reinterpret the earlier all-channel oracle as
held-out evidence.

- [ ] **Step 5: Re-run verification after documentation**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_crossfit.py -v
pytest tests/analyze -q
ruff check prior/analyze/d2026_07_28 tests/analyze
ty check prior/analyze/d2026_07_28 tests/analyze/test_llm_grid_transform_crossfit.py
git diff --check
git status --short --branch
```

Expected: tests, Ruff, ty, and diff check exit 0; Git shows only the intended
Chinese daily-note modification.

- [ ] **Step 6: Commit the verified finding**

```bash
git add docs/daily/2026-07-28.md
git commit -m "docs: record cross-fit control findings"
```

- [ ] **Step 7: Final review checkpoint**

Dispatch one requirements reviewer and one code-quality reviewer. The leader
must independently inspect their findings, the full Git diff since `fbebc92`,
the final artifact summaries, and fresh verification output. Fix and
re-verify every concrete Critical or Important finding before reporting the
experiment result.
