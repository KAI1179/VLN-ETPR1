# LLM-Grid Sanity Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible qualitative and quantitative sanity checks for the cached R2R+RxR-EN LLM-Grid checkpoints at epochs 1, 2, 5, and 10 without training or prediction.

**Architecture:** Reusable evaluator-record calculations and plots live directly in `prior.analyze`. Two thin CLIs in `prior.analyze.d2026_07_25` bind those functions to today's fixed experiment paths. Existing evaluator CSVs provide complete quantitative populations; existing raster caches provide only the approved qualitative comparisons.

**Tech Stack:** Python 3.8, typed-argument-parser, NumPy, Matplotlib, CSV/JSON standard library, Pytest, Ruff, ty.

## Global Constraints

- Do not train a model, run checkpoint inference, or generate a navigation cache.
- Analyze exactly epochs 1, 2, 5, and 10 of `r2r-rxr-legacy-r1p5-direction5-s2-no-dataset-tag`.
- Quantitative populations are exactly 778 `val_seen` and 1,839 `val_unseen` episodes at every epoch.
- Missing or invalid predictions remain in the denominator as empty predictions with zero IoU.
- The predict-all baseline predicts all 27 object and 10 region category presences per episode; it does not populate spatial cells.
- Mention partitions come from evaluator-exported instruction-derived counts, never model-generated `mentioned` fields.
- The missing R2R-only epoch cache and metrics are reported as postponed; checkpoints must not be evaluated.
- Leave the first 2026-07-24 sampling-analysis checkbox unchecked for the user.
- Use Chinese in `docs/daily/`; use English in code, tests, and other documentation.
- New CLIs use `tap.Tap`; add no dependencies and no compatibility wrappers.

---

## File Structure

- Modify `prior/analyze/batch_vis.py`: expose exact-path rendering as reusable common code.
- Create `prior/analyze/llm_grid_eval_records.py`: typed CSV loading, population validation, F1 calculations, histogram tables, and summaries.
- Create `prior/analyze/llm_grid_eval_plots.py`: reusable quantitative figures.
- Create `prior/analyze/d2026_07_25/__init__.py`: valid date-package marker.
- Create `prior/analyze/d2026_07_25/plot_llm_grid_sanity.py`: today's quantitative CLI.
- Create `prior/analyze/d2026_07_25/visualize_llm_grid.py`: today's qualitative CLI and comparison sheets.
- Modify `tests/test_grid_map_visualization.py`: exact-path renderer coverage.
- Create `tests/analyze/test_llm_grid_eval_records.py`: calculation and contract tests.
- Create `tests/analyze/test_llm_grid_eval_plots.py`: figure tests.
- Create `tests/analyze/test_llm_grid_visualization.py`: fixed-sample and sheet tests.
- Modify `docs/daily/2026-07-24.md`: links and completed-child checkboxes.
- Modify `docs/daily/2026-07-25.md`: findings, decisions, limitations, and artifact links.
- Create generated figures under `docs/images/llm_grid_r2r_rxr_sanity/`.

### Task 1: Reusable evaluator-record calculations

**Files:**
- Create: `prior/analyze/llm_grid_eval_records.py`
- Create: `tests/analyze/test_llm_grid_eval_records.py`

**Interfaces:**
- Consumes: the existing `episodes.csv` columns documented in the design.
- Produces:
  - `CSVValue = Union[str, int, float]`
  - `EpisodeRecord`
  - `CategoryF1`
  - `load_episode_records(epoch: int, path: Path) -> tuple[EpisodeRecord, ...]`
  - `validate_epoch_populations(records_by_epoch, expected_counts) -> None`
  - `predict_all_category_f1(record: EpisodeRecord) -> CategoryF1`
  - `actual_category_f1(record: EpisodeRecord) -> CategoryF1`
  - `iou_histogram_rows(records_by_epoch, bin_count=20) -> list[dict[str, CSVValue]]`
  - `iou_survival_rows(records_by_epoch, thresholds) -> list[dict[str, CSVValue]]`
  - `category_f1_rows(records_by_epoch) -> list[dict[str, CSVValue]]`
  - `category_count_histogram_rows(records_by_epoch) -> list[dict[str, CSVValue]]`
  - `f1_summary(rows) -> dict[str, dict[str, float]]`
  - `write_csv_rows(path, rows) -> None`

- [ ] **Step 1: Write failing formula and CSV-loading tests**

Create synthetic rows containing every required metric column. Assert the exact
episode-wise formulas:

```python
def test_predict_all_and_actual_category_f1() -> None:
    record = _record(
        object_target=3,
        object_predicted=4,
        object_true_positive=2,
        region_target=2,
        region_predicted=2,
        region_true_positive=1,
    )

    baseline = predict_all_category_f1(record)
    actual = actual_category_f1(record)

    assert baseline.object == pytest.approx(6 / 30)
    assert baseline.region == pytest.approx(4 / 12)
    assert baseline.combined == pytest.approx(10 / 42)
    assert actual.object == pytest.approx(4 / 7)
    assert actual.region == pytest.approx(2 / 4)
    assert actual.combined == pytest.approx(6 / 11)
```

Also assert that `load_episode_records` rejects a missing required column and
parses `schema_valid=0` with `category_aware_raster_iou=0` without dropping the
episode.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_eval_records.py -v
```

Expected: collection fails because
`prior.analyze.llm_grid_eval_records` does not exist.

- [ ] **Step 3: Implement typed records and category F1**

Define immutable value objects and strict conversion:

```python
@dataclass(frozen=True)
class CategoryF1:
    object: float
    region: float
    combined: float


@dataclass(frozen=True)
class EpisodeRecord:
    epoch: int
    split: str
    scene_id: str
    example_id: str
    iou: float
    schema_valid: bool
    object_target: int
    object_predicted: int
    object_true_positive: int
    region_target: int
    region_predicted: int
    region_true_positive: int
    mentioned_object_predicted: int
    unmentioned_object_predicted: int
    mentioned_region_predicted: int
    unmentioned_region_predicted: int


def _f1(true_positive: int, predicted: int, target: int) -> float:
    denominator = predicted + target
    return 0.0 if denominator == 0 else 2.0 * true_positive / denominator


def predict_all_category_f1(record: EpisodeRecord) -> CategoryF1:
    target = record.object_target + record.region_target
    return CategoryF1(
        object=_f1(record.object_target, 27, record.object_target),
        region=_f1(record.region_target, 10, record.region_target),
        combined=_f1(target, 37, target),
    )
```

`actual_category_f1` applies `_f1` to object, region, and summed counts. Reject
negative counts, unsupported splits, non-binary schema validity, non-finite
IoU, and nonzero IoU on invalid rows with explicit `ValueError` messages.

- [ ] **Step 4: Add population-contract tests**

Build all four required synthetic epochs containing the same identities.
Verify acceptance, then change one example ID and verify:

```python
with pytest.raises(ValueError, match="episode identities differ"):
    validate_epoch_populations(
        mismatched,
        expected_counts={"val_seen": 2, "val_unseen": 1},
    )
```

Also test wrong split counts, duplicate `(split, scene_id, example_id)` keys,
and an unsupported epoch.

- [ ] **Step 5: Implement population validation**

Use epoch 1 as the identity reference after requiring the exact epoch set
`{1, 2, 5, 10}`. Count each split independently and compare complete identity
sets, including invalid rows.

- [ ] **Step 6: Add histogram and summary tests**

For IoUs `[0.0, 0.25, 0.75, 1.0]`, assert each epoch/split histogram conserves
four observations and survival counts at thresholds `[0.0, 0.5, 1.0]` are
`[4, 2, 1]`. Assert object and region category-count histograms conserve the
same population for total, mentioned, and unmentioned partitions.

For F1 rows, assert separate model rows for all four epochs and one
`predict_all` row per episode identity, rather than four duplicated baseline
rows.

- [ ] **Step 7: Implement table builders and CSV writing**

Use shared IoU bin edges from `numpy.linspace(0.0, 1.0, bin_count + 1)`.
Represent the final bin as right-inclusive so IoU 1.0 is retained. Use integer
category bins `0..27` for objects and `0..10` for regions. `write_csv_rows`
sorts field names deterministically, refuses an empty row sequence, creates
the parent directory, and writes UTF-8 with `newline=""`.

- [ ] **Step 8: Run tests and quality checks**

Run:

```bash
pytest tests/analyze/test_llm_grid_eval_records.py -v
ruff check prior/analyze/llm_grid_eval_records.py tests/analyze/test_llm_grid_eval_records.py
ty check prior/analyze/llm_grid_eval_records.py tests/analyze/test_llm_grid_eval_records.py
```

Expected: all pass with no diagnostics.

- [ ] **Step 9: Commit**

```bash
git add prior/analyze/llm_grid_eval_records.py tests/analyze/test_llm_grid_eval_records.py
git commit -m "feat: add LLM-Grid analysis records"
```

### Task 2: Reusable quantitative plots

**Files:**
- Create: `prior/analyze/llm_grid_eval_plots.py`
- Create: `tests/analyze/test_llm_grid_eval_plots.py`

**Interfaces:**
- Consumes: `EpisodeRecord`, `category_f1_rows`, and table rows from Task 1.
- Produces:
  - `plot_iou_histograms(records_by_epoch, output_path, bin_count=20) -> None`
  - `plot_iou_survival(records_by_epoch, output_path, thresholds) -> None`
  - `plot_category_f1(records_by_epoch, output_path) -> None`
  - `plot_category_counts(records_by_epoch, output_path, kind) -> None`

- [ ] **Step 1: Write failing PNG and validation tests**

Create a minimal complete `{1, 2, 5, 10}` record mapping with both splits.
Call every plotter and assert the PNG signature:

```python
assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
```

Assert `plot_category_counts(..., kind="furniture")` raises
`ValueError("kind must be 'object' or 'region'")`.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_eval_plots.py -v
```

Expected: collection fails because `prior.analyze.llm_grid_eval_plots` does not
exist.

- [ ] **Step 3: Implement IoU figures**

Use fixed axes:

- Histogram: two rows (`val_seen`, `val_unseen`) by four epoch columns, x range
  `[0, 1]`, shared bin edges, y label `Episodes`.
- Survival: one split per panel, four epoch curves, x label `IoU threshold`,
  left y axis `Episodes`, and a secondary y axis showing the equivalent
  population fraction.

Every plotter rejects an empty record mapping, creates the output parent,
saves at 180 DPI with `bbox_inches="tight"`, and closes the figure.

- [ ] **Step 4: Implement episode-wise F1 figure**

Create a 3-by-2 grid: object/region/combined rows and seen/unseen columns.
For each panel, draw distributions for epochs 1, 2, 5, 10 and one
`predict all` baseline. Label the y axis `Episode-wise F1` and fix it to
`[-0.05, 1.05]`.

- [ ] **Step 5: Implement category-count figures**

For `kind="object"`, use total/mentioned/unmentioned rows, four epoch columns,
and integer bins from `-0.5` to `27.5`. For `kind="region"`, use the same layout
with bins from `-0.5` to `10.5`. Overlay seen and unseen step histograms, use
the y label `Episodes`, and keep identical limits across an entire figure.

- [ ] **Step 6: Run tests and quality checks**

Run:

```bash
pytest tests/analyze/test_llm_grid_eval_plots.py -v
ruff check prior/analyze/llm_grid_eval_plots.py tests/analyze/test_llm_grid_eval_plots.py
ty check prior/analyze/llm_grid_eval_plots.py tests/analyze/test_llm_grid_eval_plots.py
```

Expected: all pass with no diagnostics.

- [ ] **Step 7: Commit**

```bash
git add prior/analyze/llm_grid_eval_plots.py tests/analyze/test_llm_grid_eval_plots.py
git commit -m "feat: plot LLM-Grid sanity metrics"
```

### Task 3: Exact qualitative sampling and comparison sheets

**Files:**
- Modify: `prior/analyze/batch_vis.py`
- Modify: `tests/test_grid_map_visualization.py`
- Create: `prior/analyze/d2026_07_25/__init__.py`
- Create: `prior/analyze/d2026_07_25/visualize_llm_grid.py`
- Create: `tests/analyze/test_llm_grid_visualization.py`

**Interfaces:**
- Consumes: historical R2R-only PNGs, four mixed-model raster roots, and the
  existing ground-truth visualization cache.
- Produces:
  - `render_prediction_paths(prediction_paths, ground_truth_root, output_root) -> tuple[Path, ...]`
  - `QualitativeExample`
  - `select_common_examples(historical_root, raster_roots, expected_count=None) -> tuple[QualitativeExample, ...]`
  - `write_comparison_sheet(columns, output_path) -> None`
  - `VisualizationArgs(Tap)` and `run_visualization(args) -> dict[str, object]`

- [ ] **Step 1: Write a failing exact-path renderer test**

Create three fake prediction paths, monkeypatch the map loaders and comparison
renderer as in the existing tests, and call:

```python
rendered = render_prediction_paths(
    [prediction_paths[2], prediction_paths[0]],
    ground_truth_root,
    output_root,
)
assert rendered == (
    output_root / "scene" / "episode_2.png",
    output_root / "scene" / "episode_0.png",
)
```

Also assert a missing explicitly requested prediction raises
`FileNotFoundError` rather than being silently skipped.

- [ ] **Step 2: Run the renderer test and confirm RED**

Run:

```bash
pytest tests/test_grid_map_visualization.py -v
```

Expected: import fails because `render_prediction_paths` does not exist.

- [ ] **Step 3: Refactor the renderer without changing random-sampling behavior**

Extract the current loop into `render_prediction_paths`. Preserve caller order,
validate every prediction, raster, and boxes path, and return output paths.
Make `render_comparisons` call `_sample_prediction_paths` followed by the new
function and continue returning an integer count.

- [ ] **Step 4: Write failing fixed-selection and sheet tests**

Build a historical tree with two `R2R_val_unseen_*.png` files and four raster
roots. Assert only examples present in all four roots are returned and that
the order is `(scene_id, example_id)`. Assert duplicate historical identities
and fewer than the requested common examples fail explicitly.

Create five 8-by-8 RGB PNGs, call `write_comparison_sheet`, and assert a valid
PNG is written.

- [ ] **Step 5: Run the daily-visualization tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_visualization.py -v
```

Expected: collection fails because
`prior.analyze.d2026_07_25.visualize_llm_grid` does not exist.

- [ ] **Step 6: Implement the fixed qualitative CLI**

Define:

```python
EPOCH_CACHE_KEYS = {
    1: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-1",
    2: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2",
    5: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-5",
    10: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree",
}


class VisualizationArgs(Tap):
    navigation_root: Path = Path("data/llm_navigation")
    ground_truth_root: Path = Path(
        "data/cognitive_maps/gt.legacy.r1p5.direction5.blurred.v1"
    )
    historical_root: Path = Path(
        "docs/images/llm-grid-s2.legacy.r1p5.direction5.comparison"
    )
    output_root: Path = Path(
        "outputs/llm_grid_analysis/r2r_rxr_sanity/visualizations"
    )
    docs_image_root: Path = Path(
        "docs/images/llm_grid_r2r_rxr_sanity/comparisons"
    )
```

`select_common_examples` requires exactly 17 common historical `val_unseen`
identities for the default run. `run_visualization` renders each epoch into
`output_root/epoch-N`, writes one five-column sheet per example, and writes
`qualitative_examples.json` containing all source and output paths. Historical
columns are titled `Historical R2R-only (checkpoint unknown)`.

- [ ] **Step 7: Run tests and quality checks**

Run:

```bash
pytest tests/test_grid_map_visualization.py tests/analyze/test_llm_grid_visualization.py -v
ruff check prior/analyze/batch_vis.py prior/analyze/d2026_07_25 tests/test_grid_map_visualization.py tests/analyze/test_llm_grid_visualization.py
ty check prior/analyze/batch_vis.py prior/analyze/d2026_07_25 tests/analyze/test_llm_grid_visualization.py
```

Expected: all pass with no diagnostics.

- [ ] **Step 8: Commit**

```bash
git add prior/analyze/batch_vis.py prior/analyze/d2026_07_25 tests/test_grid_map_visualization.py tests/analyze/test_llm_grid_visualization.py
git commit -m "feat: compare fixed LLM-Grid samples"
```

### Task 4: Today's quantitative orchestration CLI

**Files:**
- Create: `prior/analyze/d2026_07_25/plot_llm_grid_sanity.py`
- Modify: `tests/analyze/test_llm_grid_eval_records.py`
- Modify: `tests/analyze/test_llm_grid_eval_plots.py`

**Interfaces:**
- Consumes: Task 1 record/table APIs and Task 2 plot APIs.
- Produces:
  - `SanityPlotArgs(Tap)`
  - `epoch_episode_paths(sweep_root: Path) -> dict[int, Path]`
  - `run_analysis(args: SanityPlotArgs) -> dict[str, object]`
  - quantitative CSV, JSON, and PNG artifacts.

- [ ] **Step 1: Write failing fixed-path and orchestration tests**

Assert:

```python
assert epoch_episode_paths(Path("sweep")) == {
    1: Path("sweep/runs/000/episodes.csv"),
    2: Path("sweep/runs/001/episodes.csv"),
    5: Path("sweep/runs/004/episodes.csv"),
    10: Path("sweep/runs/009/episodes.csv"),
}
```

Use synthetic complete epoch CSVs with one episode per split and pass
`expected_counts={"val_seen": 1, "val_unseen": 1}` through the internal
testable runner. Assert the output contains:

- `iou_histograms.csv`
- `iou_survival.csv`
- `category_f1.csv`
- `category_count_histograms.csv`
- `summary.json`
- five quantitative PNGs.

Assert `summary.json` records the R2R-only metric comparison as
`"status": "postponed"` and names the absent cache and evaluator outputs.

- [ ] **Step 2: Run the orchestration tests and confirm RED**

Run:

```bash
pytest tests/analyze/test_llm_grid_eval_records.py tests/analyze/test_llm_grid_eval_plots.py -v
```

Expected: import fails because
`prior.analyze.d2026_07_25.plot_llm_grid_sanity` does not exist.

- [ ] **Step 3: Implement the Tap arguments and fixed epoch mapping**

Define:

```python
class SanityPlotArgs(Tap):
    sweep_root: Path = Path(
        "outputs/llm_grid_eval/checkpoint-sweep-r1p5"
    )
    output_root: Path = Path(
        "outputs/llm_grid_analysis/r2r_rxr_sanity"
    )
    docs_image_root: Path = Path(
        "docs/images/llm_grid_r2r_rxr_sanity"
    )
```

The public CLI always validates the fixed counts 778 and 1,839. A private
`_run_analysis(args, expected_counts)` accepts synthetic counts only for tests.

- [ ] **Step 4: Implement deterministic artifact generation**

Load and validate records once, then:

1. Write the four CSV tables.
2. Render `iou_histograms.png`, `iou_survival.png`, `category_f1.png`,
   `object_category_counts.png`, and `region_category_counts.png`.
3. Copy those five figures into `docs_image_root`.
4. Write `summary.json` with input paths, counts, invalid totals by epoch/split,
   episode-macro actual and predict-all means, figure paths, and the postponed
   R2R-only comparison.

Use `json.dumps(summary, indent=2, sort_keys=True) + "\n"` and refuse to
overwrite an input path.

- [ ] **Step 5: Run tests and quality checks**

Run:

```bash
pytest tests/analyze/test_llm_grid_eval_records.py tests/analyze/test_llm_grid_eval_plots.py -v
ruff check prior/analyze/llm_grid_eval_records.py prior/analyze/llm_grid_eval_plots.py prior/analyze/d2026_07_25/plot_llm_grid_sanity.py tests/analyze
ty check prior/analyze/llm_grid_eval_records.py prior/analyze/llm_grid_eval_plots.py prior/analyze/d2026_07_25/plot_llm_grid_sanity.py tests/analyze
```

Expected: all pass with no diagnostics.

- [ ] **Step 6: Commit**

```bash
git add prior/analyze/d2026_07_25/plot_llm_grid_sanity.py tests/analyze
git commit -m "feat: orchestrate LLM-Grid sanity report"
```

### Task 5: Generate artifacts, verify conclusions, and update notes

**Files:**
- Create: `docs/images/llm_grid_r2r_rxr_sanity/*.png`
- Create: `docs/images/llm_grid_r2r_rxr_sanity/comparisons/*.png`
- Modify: `docs/daily/2026-07-24.md`
- Modify: `docs/daily/2026-07-25.md`

**Interfaces:**
- Consumes: completed Tasks 1–4 and existing local caches.
- Produces: inspected figures, exact findings, linked notes, and checklist
  status.

- [ ] **Step 1: Run the cached quantitative analysis**

Run:

```bash
python -m prior.analyze.d2026_07_25.plot_llm_grid_sanity
```

Expected: no GPU use; summary reports 778 seen and 1,839 unseen episodes for
every epoch and writes all five figures.

- [ ] **Step 2: Run the cached qualitative visualization**

Run:

```bash
python -m prior.analyze.d2026_07_25.visualize_llm_grid
```

Expected: exactly 17 manifests and comparison sheets; no training or
prediction process starts.

- [ ] **Step 3: Inspect numerical outputs**

Read:

```bash
python -m json.tool outputs/llm_grid_analysis/r2r_rxr_sanity/summary.json
wc -l outputs/llm_grid_analysis/r2r_rxr_sanity/*.csv
```

Confirm histogram totals equal split populations, baseline rows are
episode-macro, epoch 5 invalid counts match the evaluator, and the R2R-only
comparison remains postponed.

- [ ] **Step 4: Inspect figures visually**

Open with the workspace image viewer:

- `docs/images/llm_grid_r2r_rxr_sanity/iou_histograms.png`
- `docs/images/llm_grid_r2r_rxr_sanity/iou_survival.png`
- `docs/images/llm_grid_r2r_rxr_sanity/category_f1.png`
- `docs/images/llm_grid_r2r_rxr_sanity/object_category_counts.png`
- `docs/images/llm_grid_r2r_rxr_sanity/region_category_counts.png`
- At least four comparison sheets spanning different scenes.

Reject clipped labels, unreadable legends, misleading axes, or blank panels;
fix the responsible plotter under its targeted tests before continuing.

- [ ] **Step 5: Write the Chinese findings**

Add a `LLM-Grid 合理性检查` section to `docs/daily/2026-07-25.md`. Include exact
split-wise baseline and model F1 means, IoU distribution observations,
category-count shifts, qualitative observations, epoch 5 invalid counts, and
the missing-data reason for postponing the R2R-only epoch comparison. Embed or
link every selected figure and the comparison-sheet directory.

- [ ] **Step 6: Update the 2026-07-24 checklist**

Add a link to the new 2026-07-25 section. Leave the first sampling-analysis
checkbox unchecked. Check only:

- tolerant IoU distribution;
- predict-all-category F1;
- predicted category-count statistics.

Leave the R2R-only epoch comparison and its parent sanity-check checkbox
unchecked.

- [ ] **Step 7: Run final verification**

Run:

```bash
pytest tests/test_grid_map_visualization.py tests/analyze -v
ruff check prior/analyze/batch_vis.py prior/analyze/llm_grid_eval_records.py prior/analyze/llm_grid_eval_plots.py prior/analyze/d2026_07_25 tests/test_grid_map_visualization.py tests/analyze
ty check prior/analyze/batch_vis.py prior/analyze/llm_grid_eval_records.py prior/analyze/llm_grid_eval_plots.py prior/analyze/d2026_07_25 tests/analyze
git diff --check
git status --short
```

Expected: tests and static checks pass; only intended source, tests, figures,
and note changes remain.

- [ ] **Step 8: Commit results**

```bash
git add prior/analyze tests docs/daily/2026-07-24.md docs/daily/2026-07-25.md docs/images/llm_grid_r2r_rxr_sanity
git commit -m "analysis: verify mixed LLM-Grid predictions"
```
