# LLM-Grid Start-Centered Transform Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a reproducible cache-only four-way start-centered rotation oracle over all 1,839 mixed epoch-2 R2R `val_unseen` LLM-Grid predictions and decide whether four rotated cognitive-map hypotheses warrant a navigation experiment.

**Architecture:** A reusable typed module owns padded nearest-neighbor rotation, support-aware raster scoring, direction rotation, and deterministic angle selection. A thin date-specific `tap.Tap` CLI owns fixed cache loading, episode metrics, scene-cluster bootstrap, plots, provenance, and the complete report. Existing LLM-Grid parsing, target loading, cache path construction, mention extraction, and downsampling remain authoritative.

**Tech Stack:** Python 3.8, NumPy, Matplotlib, typed-argument-parser, CSV/JSON and hashlib standard library, Pytest, Ruff, ty.

## Global Constraints

- Use only existing local prediction and ground-truth caches; do not train, generate predictions, access the login node, submit jobs, or download dependencies.
- Primary input is exactly R2R `val_unseen` from cache key `llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2`.
- Ground truth is exactly namespace `gt.legacy.r1p5.direction5.v1`.
- The complete experiment evaluates exactly 1,839 episodes; missing and invalid predictions remain in the denominator as empty predictions with zero IoU. An explicit positive `--limit` is permitted only for a manifest-labeled smoke run.
- Search angles in the declared tie-break order `(0.0, 90.0, 180.0, 270.0)`.
- Apply one angle to all 37 channels around the exact continuous scale-2 start pivot; keep ground truth, start position, and start heading fixed.
- Cell centers are `(row + 0.5, col + 0.5)`; boolean grids use nearest-neighbor inverse warping.
- Preserve all warped support on dynamic padded bounds; count support outside the `50×50` target frame as false positives.
- Rotate predicted direction vectors with the same signed angle as the grid.
- Scene-cluster bootstrap uses 10,000 repetitions and seed 42 over the 11 `val_unseen` scenes.
- The oracle-selected angle is diagnostic only and must never be passed to navigation.
- Use Chinese in `docs/daily/`; use English in code, tests, plans, specs, JSON keys, CSV headers, and plots.
- New CLIs use `tap.Tap`; add no dependencies, compatibility wrappers, wildcard imports, lint suppressions, type suppressions, or silent fallbacks.
- New public data contracts are frozen typed dataclasses; invalid shapes, pivots, angles, manifests, cache keys, populations, and duplicate identities fail explicitly.

---

## File Structure

- Create `prior/analyze/llm_grid_registration.py`: reusable padded spatial warp, support-aware scores, direction rotation, and angle selection.
- Create `tests/analyze/test_llm_grid_registration.py`: synthetic transform, support, scoring, and tie-break tests.
- Create `prior/analyze/d2026_07_27/__init__.py`: date-package marker.
- Create `prior/analyze/d2026_07_27/llm_grid_transform_diagnosis.py`: fixed cache orchestration, episode rows, summaries, bootstrap, plots, and `Tap` CLI.
- Create `tests/analyze/test_llm_grid_transform_diagnosis.py`: invalid-output, pivot conversion, sensitivity metrics, bootstrap, and artifact tests.
- Modify `docs/daily/2026-07-27.md`: Chinese protocol, exact findings, decision, and limitations.
- Generate ignored artifacts under `outputs/llm_grid_analysis/start_centered_registration_r2r_rxr_epoch2/`.

### Task 1: Reusable start-centered rotation and scoring

**Files:**
- Create: `prior/analyze/llm_grid_registration.py`
- Create: `tests/analyze/test_llm_grid_registration.py`

**Interfaces:**
- Consumes: boolean category grids, a continuous `(row, col)` pivot, direction vectors, and the fixed angle order.
- Produces:
  - `SpatialBounds(row_min: int, row_max: int, col_min: int, col_max: int)`
  - `WarpedGrid(grid: NDArray[np.bool_], bounds: SpatialBounds, input_support: int)`
  - `RasterScore(intersection: int, union: int, predicted_support: int, target_support: int, in_frame_support: int, out_of_frame_support: int)`
  - `AngleScore(angle_degrees: float, raster: RasterScore, warped_support_ratio: float)`
  - `warp_grid_about_pivot(grid, pivot, angle_degrees) -> WarpedGrid`
  - `score_warped_grid(warped, target) -> RasterScore`
  - `rotate_direction_vectors(vectors, angle_degrees) -> NDArray[np.float32]`
  - `direction_cosine(predicted, target) -> float`
  - `select_best_angle(grid, target, pivot, angles) -> tuple[AngleScore, tuple[AngleScore, ...]]`

- [ ] **Step 1: Write failing validation and identity tests**

Create direct tests with hand-built boolean arrays. Name the production breaks:
wrong shape acceptance, non-finite pivot acceptance, identity coordinate shift,
and identity support loss.

```python
def test_identity_preserves_coordinates_and_support() -> None:
    grid = np.zeros((2, 4, 5), dtype=np.bool_)
    grid[0, 0, 1] = True
    grid[1, 3, 4] = True

    warped = warp_grid_about_pivot(grid, (1.2, 2.7), 0.0)

    assert warped.bounds == SpatialBounds(0, 4, 0, 5)
    assert np.array_equal(warped.grid, grid)
    assert warped.input_support == 2
    assert warped.total_support == 2
```

Also assert explicit `ValueError` for a non-3D grid, non-boolean grid,
non-finite pivot, and non-finite angle.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_registration.py -v
```

Expected: collection fails because
`prior.analyze.llm_grid_registration` does not exist.

- [ ] **Step 3: Implement frozen typed bounds and identity warp**

Use `@dataclass(frozen=True)` and validate every dataclass invariant in
`__post_init__`. Add typed properties:

```python
@property
def total_support(self) -> int:
    return int(np.count_nonzero(self.grid))


@property
def iou(self) -> float:
    return 0.0 if self.union == 0 else self.intersection / self.union
```

The identity path returns a copied boolean grid with bounds
`SpatialBounds(0, rows, 0, cols)`. It must not return the caller's mutable
array.

- [ ] **Step 4: Add failing fractional-pivot and clipping tests**

Use a one-channel `5×5` grid with a source cell at `(2, 3)`, pivot
`(1.2, 2.2)`, and `90°`. With:

```text
R(theta) = [[cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]]
```

the transformed source center is `(-0.1, 3.5)`, whose nearest output cell is
`(-1, 3)`. Assert that:

- the padded bounds include row `-1`;
- the expected padded cell is true;
- scoring against an empty `5×5` target records one out-of-frame false
  positive;
- scoring against a target cell cannot silently crop the outside support.

Add a two-channel fixture with identical spatial support and assert both
channels use the same bounds and mapping.

- [ ] **Step 5: Implement generic nearest-neighbor inverse warp**

Transform the four continuous source corners `(0,0)`, `(0,cols)`,
`(rows,0)`, `(rows,cols)` to obtain dynamic integer output bounds with
`floor(min)` and `ceil(max)`.

For every output cell center in those bounds:

1. apply `R(-theta)` around the unchanged pivot;
2. accept source coordinates only in `[0, rows) × [0, cols)`;
3. choose the nearest source cell center using `floor(source_coordinate)`;
4. sample all channels with the same spatial index arrays.

Do not use bilinear interpolation, thresholding, per-channel transforms, or a
fixed magic padding size.

- [ ] **Step 6: Add failing score, direction, and tie-break tests**

Assert literal intersection/union/support counts for:

- an in-frame true positive plus an out-of-frame false positive;
- empty prediction against non-empty target;
- empty/empty with IoU zero;
- occupancy duplicated across two categories, which must count twice in
  category-aware support.

Assert:

```python
rotated = rotate_direction_vectors(
    np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    90.0,
)
assert rotated[0] == pytest.approx([0.0, 1.0], abs=1e-6)
assert rotated[1] == pytest.approx([0.0, 0.0], abs=1e-6)
```

Build a symmetric fixture where all four angles tie and assert that
`select_best_angle(...).angle_degrees == 0.0`.

- [ ] **Step 7: Implement support-aware scoring and angle selection**

Embed only the overlap between padded bounds and target bounds to calculate
intersection. Calculate union as:

```text
total warped predicted support + target support - intersection
```

so all outside support remains a false positive. `in_frame_support` and
`out_of_frame_support` partition total warped support exactly.

`direction_cosine` averages cosine only over rows where both vectors have
nonzero norm. It returns zero when no pair is eligible. Angle selection
iterates in caller-provided order and replaces the best only on strict greater
IoU.

- [ ] **Step 8: Run Task 1 verification**

Run:

```bash
pytest tests/analyze/test_llm_grid_registration.py -v
ruff check prior/analyze/llm_grid_registration.py tests/analyze/test_llm_grid_registration.py
ty check prior/analyze/llm_grid_registration.py tests/analyze/test_llm_grid_registration.py
```

Expected: every command exits zero with no diagnostics.

- [ ] **Step 9: Commit Task 1**

```bash
git add prior/analyze/llm_grid_registration.py tests/analyze/test_llm_grid_registration.py
git commit -m "feat(analysis): add start-centered grid rotation"
```

### Task 2: Four-way cache experiment and deterministic report

**Files:**
- Create: `prior/analyze/d2026_07_27/__init__.py`
- Create: `prior/analyze/d2026_07_27/llm_grid_transform_diagnosis.py`
- Create: `tests/analyze/test_llm_grid_transform_diagnosis.py`

**Interfaces:**
- Consumes:
  - Task 1 `AngleScore`, `RasterScore`, `select_best_angle`,
    `rotate_direction_vectors`, `direction_cosine`;
  - `load_llm_grid_examples` and `LLMGridDataset`;
  - `parse_grid_text` and `LLMGridValidationError`;
  - `llm_navigation_prediction_path`;
  - `extract_categories`;
  - canonical category constants.
- Produces:
  - `FourWayDiagnosisArgs(Tap)`
  - `EpisodeTransformResult`
  - `start_pivot_for_scale(start_position_m, scale) -> tuple[float, float]`
  - `evaluate_episode(...) -> EpisodeTransformResult`
  - `bootstrap_scene_delta(rows, repetitions, seed) -> dict[str, object]`
  - `run_diagnosis(args) -> dict[str, object]`
  - `main(argv=None) -> dict[str, object]`

- [ ] **Step 1: Write failing pivot and invalid-output tests**

Assert the scale-2 conversion independently:

```python
assert start_pivot_for_scale((18.487621, 6.2339373), 2) == pytest.approx(
    (18.487621, 6.2339373)
)
```

because `CELL_SIZE == 0.5 m` and the scale-2 cell is `1 m`.

Pass an explicit empty prediction with `schema_valid=False` to
`evaluate_episode`. Assert the episode remains present, every angle IoU and
delta is zero, identity is selected, target support is retained, and the
schema-valid field is false.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_transform_diagnosis.py -v
```

Expected: collection fails because the date-specific module does not exist.

- [ ] **Step 3: Implement typed episode evaluation**

`EpisodeTransformResult` is frozen and contains:

- split, scene ID, and example ID;
- schema validity;
- identity IoU and best IoU;
- delta, best/second angle, and margin;
- one IoU per declared angle;
- Cell precision/recall/F1 for the chosen transform;
- direction cosine before and after;
- input, warped, in-frame, and out-of-frame support;
- object, region, mentioned, unmentioned, occupancy-collapsed, and
  broad-channel-excluded identity/best IoU pairs.

Use one best angle selected on all 37 channels for every sensitivity metric.
Mentioned region IDs are offset by `OBJECT_CATEGORIES`. Occupancy collapses all
channels with `np.any(grid, axis=0, keepdims=True)`. Broad exclusion removes
`void`, `structure`, `other`, and `free-space` object channels only.

- [ ] **Step 4: Add failing sensitivity and angle-sharing tests**

Create a two-channel prediction where one channel prefers `90°` and the other
would prefer `270°` alone. Assert the combined all-channel selection produces
one angle and every sensitivity score evaluates that same angle.

Create a mentioned-object plus unmentioned-region fixture and assert their
support and IoU calculations use complementary canonical channel masks.

- [ ] **Step 5: Implement strict local cache loading**

Define `FourWayDiagnosisArgs` defaults:

```python
cache_dir: Path = Path("data/llm_navigation")
cache_model_key: str = (
    "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2"
)
cognitive_map_namespace: str = "gt.legacy.r1p5.direction5.v1"
output_dir: Path = Path(
    "outputs/llm_grid_analysis/"
    "start_centered_registration_r2r_rxr_epoch2"
)
angles: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
bootstrap_repetitions: int = 10_000
bootstrap_seed: int = 42
limit: Optional[int] = None
quiet: bool = False
```

Validate exact angles, positive bootstrap count, distinct output/input paths,
an absent or positive limit, and required prediction manifests before loading
examples. A limited run records `"smoke": true` in its manifest and summary.

Load exactly `R2R val_unseen`, construct `LLMGridDataset(..., scale=2)`, and
require exactly 1,839 unique `(scene_id, example_id)` identities. Read raw
prediction text through `llm_navigation_prediction_path`. Parse with
`parse_grid_text`; on `LLMGridValidationError`, provide an explicit empty grid
and mark the row invalid. No other exception is swallowed.

- [ ] **Step 6: Write failing bootstrap and artifact tests**

Use two scenes with two episodes each and literal deltas. Assert that:

- all episodes in a sampled scene share the same multiplicity;
- two calls with 10,000 repetitions and seed 42 are identical;
- the JSON reports mean delta and 2.5/97.5 percentiles;
- the output writer creates all required JSON/CSV/PNG artifacts;
- duplicate episode identities and a population other than the injected
  expected count raise explicit errors.

Use a private test runner parameter for synthetic expected counts. The public
CLI always fixes 1,839.

- [ ] **Step 7: Implement summary, bootstrap, provenance, and plots**

Write UTF-8, deterministic, sorted-key JSON with a trailing newline. Write CSV
with explicit stable field order.

`summary.json` includes:

- examples, scenes, valid and invalid counts;
- identity and best episode-macro IoU;
- mean and median paired delta;
- proportions above `0.01` and `0.05`;
- identity rate and angle counts;
- best/second margin distribution;
- Cell and direction metrics;
- every sensitivity identity/best/delta;
- input/warped/in-frame/out-of-frame support summaries;
- stop/go gate inputs without claiming deployability.

`manifest.json` includes exact input paths, cache key, SHA-256 of the prediction
manifest and ground-truth namespace identifier, angles, shape, scale, cell
size, bootstrap contract, population, and current git commit when available.

Render:

- `iou_delta_distribution.png` with fixed zero, `0.01`, and `0.05` reference
  lines;
- `angle_distribution.png` with all four declared angles, including zero-count
  bars.

- [ ] **Step 8: Run Task 2 verification**

Run:

```bash
pytest tests/analyze/test_llm_grid_registration.py tests/analyze/test_llm_grid_transform_diagnosis.py -v
ruff check prior/analyze/llm_grid_registration.py prior/analyze/d2026_07_27 tests/analyze/test_llm_grid_registration.py tests/analyze/test_llm_grid_transform_diagnosis.py
ty check prior/analyze/llm_grid_registration.py prior/analyze/d2026_07_27 tests/analyze/test_llm_grid_registration.py tests/analyze/test_llm_grid_transform_diagnosis.py
```

Expected: every command exits zero with no diagnostics.

- [ ] **Step 9: Commit Task 2**

```bash
git add prior/analyze/d2026_07_27 tests/analyze/test_llm_grid_transform_diagnosis.py
git commit -m "feat(analysis): add four-way LLM-Grid diagnosis"
```

### Task 3: Run the complete experiment and record the decision

**Files:**
- Modify: `docs/daily/2026-07-27.md`
- Generate, do not commit: `outputs/llm_grid_analysis/start_centered_registration_r2r_rxr_epoch2/*`
- Modify Task 1 or Task 2 files only if a newly reproduced bug first receives a failing regression test.

**Interfaces:**
- Consumes: the Task 2 CLI and fixed local caches.
- Produces: complete ignored analysis artifacts and a Chinese experiment record with a stop/go decision.

- [ ] **Step 1: Run a limited smoke**

Run:

```bash
python -m prior.analyze.d2026_07_27.llm_grid_transform_diagnosis \
  --limit 8 \
  --bootstrap-repetitions 100 \
  --output-dir outputs/llm_grid_analysis/start_centered_registration_smoke
```

The CLI must mark the run as a smoke and must not apply the 1,839 population
assertion when an explicit positive `--limit` is supplied. Verify eight unique
episode rows, four angle columns, finite metrics, and all required artifacts.
Delete only this generated ignored smoke directory after inspection.

- [ ] **Step 2: Run the full four-way diagnosis**

Run:

```bash
python -m prior.analyze.d2026_07_27.llm_grid_transform_diagnosis
```

Require:

- 1,839 episode rows;
- 11 scenes;
- 1,830 valid and 9 invalid predictions;
- identity episode-macro Raster IoU matching the existing epoch-2
  `val_unseen` baseline within `1e-9`;
- all bootstrap and support fields finite;
- every selected angle belongs to the declared angle order.

- [ ] **Step 3: Inspect numerical and visual outputs**

Read `summary.json`, `bootstrap.json`, `manifest.json`, and the first/last rows
of `episodes.csv`. Confirm:

- out-of-frame support is never negative;
- in-frame plus out-of-frame equals total warped support per row;
- best IoU is never below identity IoU;
- invalid rows remain present with zero delta;
- angle counts sum to 1,839;
- bootstrap confidence limits are ordered.

Open both PNGs and reject clipped labels, unreadable legends, misleading axes,
or blank panels.

- [ ] **Step 4: Record the Chinese experiment note**

Append a new section to `docs/daily/2026-07-27.md` containing:

- hypothesis and fixed protocol;
- exact cache/target/commit provenance;
- verification commands and counts;
- identity and four-way macro IoU;
- mean/median delta and 95% scene-cluster CI;
- gain-tail proportions;
- angle, margin, direction, sensitivity, and support findings;
- the stop/go gate outcome;
- the oracle-only limitation and the next conditional experiment.

Do not call an oracle angle deployable and do not propose navigation training
unless the recorded gate passes.

- [ ] **Step 5: Run final branch verification**

Run:

```bash
pytest tests/analyze -q
ruff check prior/analyze/llm_grid_registration.py prior/analyze/d2026_07_27 tests/analyze
ty check prior/analyze/llm_grid_registration.py prior/analyze/d2026_07_27 tests/analyze/test_llm_grid_registration.py tests/analyze/test_llm_grid_transform_diagnosis.py
git diff --check
git status --short
```

Expected: tests pass, Ruff and ty have no diagnostics, the diff has no
whitespace errors, and only the intended daily-note change remains uncommitted.

- [ ] **Step 6: Commit Task 3**

```bash
git add docs/daily/2026-07-27.md
git commit -m "analysis: diagnose LLM-Grid orientation error"
```
