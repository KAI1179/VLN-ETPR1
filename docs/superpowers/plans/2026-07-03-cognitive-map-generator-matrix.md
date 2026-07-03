# Cognitive Map Generator Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate namespaced cognitive-map caches for `(legacy, bbox) x (1.5m, 2.5m)` without overwriting existing caches.

**Architecture:** Keep `prior.__main__` as the VLN-CE cache generator entrypoint. Add explicit `map_source`, `radius_m`, and cache namespace handling there, with `bbox` preserving the existing relevant-box raster path and `legacy` producing a Try5-style path-neighborhood raster from a full selected-level semantic grid.

**Tech Stack:** Python stdlib `argparse`, existing `prior.bbox`, `prior.grid_map`, pytest, ruff, ty.

---

### Task 1: Pin Generator CLI and Cache Namespace Behavior

**Files:**
- Modify: `tests/test_cache_generator_clis.py`
- Modify: `prior/__main__.py`

- [x] **Step 1: Write failing parser and namespace tests**

Add tests for `parse_args`, derived namespace labels, and `generate_cognitive_maps(..., namespace=...)` writing under `data/cognitive_maps/<namespace>/boxes|raster`.

- [x] **Step 2: Run the focused tests and confirm failure**

Run: `pytest tests/test_cache_generator_clis.py -q`
Expected: FAIL because `parse_args`, namespace routing, and radius args are not implemented.

- [x] **Step 3: Add minimal parser and namespace helpers**

Implement `parse_args`, `map_cache_namespace`, and namespace-aware output paths in `prior.__main__`.

- [x] **Step 4: Run the focused tests**

Run: `pytest tests/test_cache_generator_clis.py -q`
Expected: parser and namespace tests pass.

### Task 2: Add Explicit Radius for BBox Maps

**Files:**
- Modify: `tests/test_cache_generator_clis.py`
- Modify: `prior/__main__.py`

- [x] **Step 1: Write failing radius plumbing test**

Add a fake `SceneSemanticBoxes.relevant_to` assertion that `max_distance=2.5` is passed when requested.

- [x] **Step 2: Run the focused test and confirm failure**

Run: `pytest tests/test_cache_generator_clis.py::test_vlnce_bbox_generator_passes_radius_and_namespace -q`
Expected: FAIL because current generator does not pass `max_distance`.

- [x] **Step 3: Pass `radius_m` into `relevant_to`**

Update bbox generation to call `relevant_to(..., max_distance=radius_m)`.

- [x] **Step 4: Run the focused tests**

Run: `pytest tests/test_cache_generator_clis.py -q`
Expected: all generator tests pass.

### Task 3: Add Legacy Raster Generation

**Files:**
- Modify: `tests/test_cache_generator_clis.py`
- Modify: `prior/__main__.py`

- [x] **Step 1: Write failing legacy mode tests**

Add tests that `map_source="legacy"` writes a raster from `_legacy_cognitive_map` and still writes a compatibility boxes file.

- [x] **Step 2: Run the focused legacy test and confirm failure**

Run: `pytest tests/test_cache_generator_clis.py::test_vlnce_legacy_generator_writes_legacy_raster_and_compat_boxes -q`
Expected: FAIL because legacy mode does not exist.

- [x] **Step 3: Implement minimal legacy mode**

Add `_legacy_cognitive_map` that builds a selected-level full semantic grid, copies square path neighborhoods by radius, scales unmentioned categories, stores trajectory keypoints and start direction, and returns `CognitiveGridMap`.

- [x] **Step 4: Run focused and quality checks**

Run:
`pytest tests/test_cache_generator_clis.py tests/test_box_construct.py tests/test_grid_map_cache.py -q`
`ruff check prior tests/test_cache_generator_clis.py`
`ty check prior tests/test_cache_generator_clis.py`

Expected: all commands exit 0.

### Task 4: Commit

**Files:**
- Commit: `docs/superpowers/plans/2026-07-03-cognitive-map-generator-matrix.md`
- Commit: `prior/__main__.py`
- Commit: `tests/test_cache_generator_clis.py`

- [x] **Step 1: Review diff**

Run: `git diff -- prior/__main__.py tests/test_cache_generator_clis.py docs/superpowers/plans/2026-07-03-cognitive-map-generator-matrix.md`

- [ ] **Step 2: Commit**

Run:
`git add prior/__main__.py tests/test_cache_generator_clis.py docs/superpowers/plans/2026-07-03-cognitive-map-generator-matrix.md`
`git commit -m "feat(prior): namespace cognitive map generators"`
