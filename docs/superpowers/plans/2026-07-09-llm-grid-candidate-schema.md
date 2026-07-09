# LLM-Grid Candidate Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the current LLM-Grid-Probe runtime contract into the LLM-Grid candidate schema by predicting sparse grid anchors plus fixed Try5-style `direction_vectors`.

**Architecture:** Keep the implementation under `vlnce_baselines/models/etp_llm` and migrate the current grid trainer/evaluator contract in place. The parser, serializer, prompt, tests, and submit script all move to the candidate JSON keys with no old-key aliases. Navigation-policy integration is a later plan.

**Tech Stack:** Python 3.8, PyTorch, Transformers causal LM, PEFT LoRA, NumPy, Tap, pytest, ruff, ty.

---

## File Structure

- Modify: `prior/llm_grid_samples.py`
  - Serialize candidate JSON with `predicted_regions`, `predicted_objects`, `regions`, `objects`, and `direction_vectors`.
  - Read `direction_vectors` from raster cache samples.
- Modify: `vlnce_baselines/models/etp_llm/prompts/llm_grid_probe_system.md`
  - Replace Probe wording with candidate wording.
  - Use Allowed object/region category headings.
  - Explain row/col and x/z axes.
- Modify: `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py`
  - Load `direction_vectors` from raster targets.
  - Parse and validate candidate JSON.
  - Add direction-vector metrics.
  - Update labels/messages from Probe to candidate where they describe the generated contract.
- Modify: `scripts/submit/llm-grid-probe-train.sh`
  - Keep memory-safe settings.
  - Update output directory to an LLM-Grid candidate path.
- Modify tests:
  - `tests/test_llm_grid_samples.py`
  - `tests/etp_llm/test_train_llm_grid_probe.py`

No compatibility branch should accept `region_candidates`, `object_candidates`, or `motion_vectors`.

---

### Task 1: Update Target Serializer Contract

**Files:**
- Modify: `tests/test_llm_grid_samples.py`
- Modify: `tests/etp_llm/test_train_llm_grid_probe.py`
- Modify: `prior/llm_grid_samples.py`

- [ ] **Step 1: Write failing serializer tests**

In `tests/test_llm_grid_samples.py`, update the keyed JSON expectations so the top-level keys are candidate keys and `direction_vectors` is required:

```python
def test_serialize_grid_target_orders_nonzero_binary_cells() -> None:
    grid = np.zeros((37, 4, 4), dtype=np.float32)
    grid[1, 0, 1] = 1.0
    grid[28, 3, 2] = 1.0
    direction_vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )

    text = serialize_grid_target(grid, scale=1, direction_vectors=direction_vectors)

    assert json.loads(text) == {
        "predicted_regions": ["living/social space"],
        "predicted_objects": ["chair"],
        "regions": {
            "living/social space": {"cells": [[3, 2]], "mentioned": False}
        },
        "objects": {"chair": {"cells": [[0, 1]], "mentioned": False}},
        "direction_vectors": [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    }
```

In `tests/etp_llm/test_train_llm_grid_probe.py`, update `EMPTY_GRID_TEXT`:

```python
EMPTY_GRID_TEXT = (
    '{"predicted_regions":[],"predicted_objects":[],"regions":{},'
    '"objects":{},"direction_vectors":[[0.0,0.0],[0.0,0.0],'
    '[0.0,0.0],[0.0,0.0],[0.0,0.0]]}'
)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_llm_grid_samples.py::test_serialize_grid_target_orders_nonzero_binary_cells tests/etp_llm/test_train_llm_grid_probe.py::test_serialize_grid_target_uses_keyed_binary_cells -q
```

Expected: fail because `serialize_grid_target` does not accept `direction_vectors` and still emits `region_candidates` / `object_candidates`.

- [ ] **Step 3: Implement serializer migration**

In `prior/llm_grid_samples.py`, change `serialize_grid_target` signature and output:

```python
def _direction_vector_records(
    direction_vectors: NDArray[np.float32],
) -> List[List[float]]:
    if direction_vectors.shape != (5, 2):
        raise ValueError(
            f"direction_vectors must have shape (5, 2), got {direction_vectors.shape}"
        )
    return [
        [float(round(float(vector[0]), 3)), float(round(float(vector[1]), 3))]
        for vector in direction_vectors
    ]


def serialize_grid_target(
    grid: NDArray[np.float32],
    direction_vectors: NDArray[np.float32],
    scale: int = 1,
    mentioned_objects: Optional[set[int]] = None,
    mentioned_regions: Optional[set[int]] = None,
) -> str:
    """Serialize sparse grid anchors and direction vectors as LLM-Grid JSON."""
    sampled = downsample_grid(grid, scale)
    mentioned_objects = mentioned_objects or set()
    mentioned_regions = mentioned_regions or set()
    objects: Dict[str, Dict[str, Any]] = {}
    regions: Dict[str, Dict[str, Any]] = {}

    for category in range(sampled.shape[0]):
        row_cols = np.argwhere(sampled[category] > 0)
        if len(row_cols) == 0:
            continue
        cells = [_cell_record(int(row), int(col)) for row, col in row_cols]
        if category < OBJECT_CATEGORIES:
            name = MAPPED_OBJECT_NAMES[category]
            objects[name] = {
                "cells": cells,
                "mentioned": category in mentioned_objects,
            }
        else:
            region_id = category - OBJECT_CATEGORIES
            name = MAPPED_REGION_NAMES[region_id]
            regions[name] = {
                "cells": cells,
                "mentioned": region_id in mentioned_regions,
            }

    return json.dumps(
        {
            "predicted_regions": list(regions),
            "predicted_objects": list(objects),
            "regions": regions,
            "objects": objects,
            "direction_vectors": _direction_vector_records(direction_vectors),
        },
        separators=(",", ":"),
    )
```

Update `analyze_grid_sample` to load `direction_vectors`:

```python
with np.load(npz_path, allow_pickle=True) as data:
    grid = data["grid"]
    direction_vectors = data["direction_vectors"]
target_text = serialize_grid_target(grid, direction_vectors=direction_vectors, scale=scale)
```

- [ ] **Step 4: Run serializer tests**

Run:

```bash
pytest tests/test_llm_grid_samples.py tests/etp_llm/test_train_llm_grid_probe.py::test_serialize_grid_target_uses_keyed_binary_cells -q
```

Expected: pass after all call sites in those tests pass `direction_vectors`.

- [ ] **Step 5: Commit serializer contract**

Run:

```bash
git add prior/llm_grid_samples.py tests/test_llm_grid_samples.py tests/etp_llm/test_train_llm_grid_probe.py
git commit -m "feat(llm-grid): serialize candidate targets"
```

---

### Task 2: Update Prompt Contract

**Files:**
- Modify: `vlnce_baselines/models/etp_llm/prompts/llm_grid_probe_system.md`
- Modify: `tests/etp_llm/test_train_llm_grid_probe.py`

- [ ] **Step 1: Write prompt expectation test**

In `tests/etp_llm/test_train_llm_grid_probe.py`, add:

```python
def test_load_system_prompt_uses_candidate_schema_terms():
    prompt = train_llm_grid_probe.load_system_prompt()

    assert "predicted_regions" in prompt
    assert "predicted_objects" in prompt
    assert "direction_vectors" in prompt
    assert "Allowed object categories" in prompt
    assert "Allowed region categories" in prompt
    assert "region_candidates" not in prompt
    assert "object_candidates" not in prompt
    assert "motion_vectors" not in prompt
```

- [ ] **Step 2: Run prompt test to verify failure**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py::test_load_system_prompt_uses_candidate_schema_terms -q
```

Expected: fail because the current prompt still uses Probe keys and old category headings.

- [ ] **Step 3: Replace prompt text**

Replace `vlnce_baselines/models/etp_llm/prompts/llm_grid_probe_system.md` with:

```markdown
Generate sparse cognitive-map anchors and direction vectors for VLN.

Each cell is [row,col] in a 50x50 grid with integers 0-49.
Grid columns follow the x axis; grid rows follow the z axis.
Return only compact valid JSON with keys: predicted_regions, predicted_objects, regions, objects, direction_vectors.
Use only categories from Allowed object categories and Allowed region categories.
Select categories that are explicitly mentioned or can be inferred from the instruction and route context.
Use canonical category names; predicted_regions/predicted_objects must match the keys of regions/objects.
direction_vectors contains exactly five [dx,dz] vectors in the x-z frame, ordered by route progress, with [0.0,0.0] padding when needed.
No markdown, prose, comments, or extra keys.

Allowed object categories:
- void
- chair
- door
- table
- cushion
- sofa
- bed
- plant
- sink
- toilet
- tv_monitor
- shower
- bathtub
- counter
- appliances
- structure
- other
- free-space
- picture
- cabinet
- chest_of_drawers
- stool
- towel
- fireplace
- gym_equipment
- seating
- clothes

Allowed region categories:
- outdoor/semi-outdoor
- living/social space
- recreation/fitness
- utility/service
- work/study
- circulation
- private room
- bathroom/sanitary
- dining/food
- other/miscellaneous
```

- [ ] **Step 4: Run prompt test**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py::test_load_system_prompt_uses_candidate_schema_terms -q
```

Expected: pass.

- [ ] **Step 5: Commit prompt contract**

Run:

```bash
git add vlnce_baselines/models/etp_llm/prompts/llm_grid_probe_system.md tests/etp_llm/test_train_llm_grid_probe.py
git commit -m "feat(llm-grid): update candidate prompt"
```

---

### Task 3: Parse Candidate JSON and Direction Vectors

**Files:**
- Modify: `tests/etp_llm/test_train_llm_grid_probe.py`
- Modify: `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py`

- [ ] **Step 1: Write parser tests**

Update valid parse tests to use candidate keys:

```python
def test_parse_grid_probe_text_accepts_candidate_records_and_direction_vectors():
    result = train_llm_grid_probe.parse_grid_probe_text(
        (
            '{"predicted_regions":["living/social space"],'
            '"predicted_objects":["chair"],'
            '"regions":{"living/social space":{"cells":[[1,2]],'
            '"mentioned":false}},'
            '"objects":{"chair":{"cells":[[0,0],[0,0]],'
            '"mentioned":true}},'
            '"direction_vectors":[[1.0,0.0],[0.0,1.0],[0.0,0.0],'
            '[0.0,0.0],[0.0,0.0]]}'
        ),
        shape=(37, 50, 50),
    )

    assert result.grid.shape == (37, 50, 50)
    assert result.grid[1, 0, 0] == pytest.approx(1.0)
    assert result.grid[28, 1, 2] == pytest.approx(1.0)
    assert result.direction_vectors.tolist() == [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
```

Add rejection tests:

```python
def test_parse_grid_probe_text_rejects_old_candidate_keys():
    with pytest.raises(
        train_llm_grid_probe.LLMGridProbeValidationError,
        match="predicted_regions",
    ):
        train_llm_grid_probe.parse_grid_probe_text(
            '{"region_candidates":[],"object_candidates":[],"regions":{},'
            '"objects":{},"direction_vectors":[[0,0],[0,0],[0,0],[0,0],[0,0]]}',
            shape=(37, 50, 50),
        )


def test_parse_grid_probe_text_rejects_bad_direction_vector_shape():
    with pytest.raises(
        train_llm_grid_probe.LLMGridProbeValidationError,
        match="direction_vectors",
    ):
        train_llm_grid_probe.parse_grid_probe_text(
            '{"predicted_regions":[],"predicted_objects":[],"regions":{},'
            '"objects":{},"direction_vectors":[[1.0,0.0]]}',
            shape=(37, 50, 50),
        )
```

- [ ] **Step 2: Run parser tests to verify failure**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py::test_parse_grid_probe_text_accepts_candidate_records_and_direction_vectors tests/etp_llm/test_train_llm_grid_probe.py::test_parse_grid_probe_text_rejects_old_candidate_keys tests/etp_llm/test_train_llm_grid_probe.py::test_parse_grid_probe_text_rejects_bad_direction_vector_shape -q
```

Expected: fail because parser expects `region_candidates` / `object_candidates` and does not return direction vectors.

- [ ] **Step 3: Implement parser migration**

In `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py`, extend the parse result:

```python
@dataclass(frozen=True)
class LLMGridProbeParseResult:
    grid: NDArray[np.float32]
    direction_vectors: NDArray[np.float32]
    record_count: int
    duplicate_record_count: int
```

Update the parser top-level key validation:

```python
expected_keys = {
    "predicted_regions",
    "predicted_objects",
    "regions",
    "objects",
    "direction_vectors",
}
if set(payload) != expected_keys:
    raise LLMGridProbeValidationError(
        "top-level JSON object must contain only predicted_regions, "
        "predicted_objects, regions, objects, direction_vectors"
    )
```

Replace candidate list parsing:

```python
predicted_regions = _parse_candidate_list(
    payload["predicted_regions"],
    "predicted_regions",
    MAPPED_REGION_NAMES,
)
predicted_objects = _parse_candidate_list(
    payload["predicted_objects"],
    "predicted_objects",
    MAPPED_OBJECT_NAMES,
)
if list(region_section) != predicted_regions:
    raise LLMGridProbeValidationError("predicted_regions must match regions keys")
if list(object_section) != predicted_objects:
    raise LLMGridProbeValidationError("predicted_objects must match objects keys")
```

Add direction-vector parsing:

```python
def _parse_direction_vectors(value: Any) -> NDArray[np.float32]:
    if not isinstance(value, list) or len(value) != 5:
        raise LLMGridProbeValidationError(
            "direction_vectors must contain exactly five vectors"
        )
    rows: List[List[float]] = []
    for index, vector in enumerate(value):
        if not isinstance(vector, list) or len(vector) != 2:
            raise LLMGridProbeValidationError(
                f"direction_vectors[{index}] must be [dx,dz]"
            )
        dx, dz = vector
        if not isinstance(dx, (int, float)) or not isinstance(dz, (int, float)):
            raise LLMGridProbeValidationError(
                f"direction_vectors[{index}] must contain numbers"
            )
        rows.append([float(dx), float(dz)])
    return np.asarray(rows, dtype=np.float32)
```

Return:

```python
return LLMGridProbeParseResult(
    grid=grid,
    direction_vectors=_parse_direction_vectors(payload["direction_vectors"]),
    record_count=record_count,
    duplicate_record_count=duplicate_record_count,
)
```

- [ ] **Step 4: Run parser tests**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py::test_parse_grid_probe_text_accepts_candidate_records_and_direction_vectors tests/etp_llm/test_train_llm_grid_probe.py::test_parse_grid_probe_text_rejects_old_candidate_keys tests/etp_llm/test_train_llm_grid_probe.py::test_parse_grid_probe_text_rejects_bad_direction_vector_shape -q
```

Expected: pass.

- [ ] **Step 5: Commit parser contract**

Run:

```bash
git add vlnce_baselines/models/etp_llm/train_llm_grid_probe.py tests/etp_llm/test_train_llm_grid_probe.py
git commit -m "feat(llm-grid): parse candidate direction vectors"
```

---

### Task 4: Load Direction Vectors for Training and Evaluation

**Files:**
- Modify: `tests/etp_llm/test_train_llm_grid_probe.py`
- Modify: `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py`

- [ ] **Step 1: Write dataset loading test**

Update the raster fixture in `test_evaluate_model_writes_metrics_and_prediction_artifact` and nearby dataset tests to save direction vectors:

```python
direction_vectors = np.asarray(
    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    dtype=np.float32,
)
np.savez_compressed(
    raster_path,
    grid=full_grid,
    direction_vectors=direction_vectors,
    start_position=np.asarray([1.2, 3.4], dtype=np.float32),
    start_direction_vector=np.asarray([0.0, 1.0], dtype=np.float32),
)
```

Add a dataset assertion:

```python
def test_llm_grid_probe_dataset_serializes_direction_vectors(tmp_path, monkeypatch):
    raster_path = tmp_path / "raster" / "scene-a" / "grid.npz"
    raster_path.parent.mkdir(parents=True)
    grid = np.zeros((37, 100, 100), dtype=np.float32)
    direction_vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    np.savez_compressed(
        raster_path,
        grid=grid,
        direction_vectors=direction_vectors,
        start_position=np.asarray([1.2, 3.4], dtype=np.float32),
        start_direction_vector=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    _save_box_payload(tmp_path / "boxes" / "scene-a" / "grid.npz")
    example = train_llm_grid_probe.LLMGridProbeExample(
        example_id="R2R_train_42",
        dataset_tag="R2R",
        split="train",
        scene_id="scene-a",
        episode_id=42,
        instruction="Go to the chair.",
        raster_path=raster_path,
    )

    item = train_llm_grid_probe.LLMGridProbeDataset([example])[0]

    assert json.loads(item["target_text"])["direction_vectors"] == [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
```

- [ ] **Step 2: Run dataset test to verify failure**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py::test_llm_grid_probe_dataset_serializes_direction_vectors -q
```

Expected: fail because `_load_raster_target` does not return `direction_vectors`.

- [ ] **Step 3: Implement raster load migration**

Change `_load_raster_target` to return four values:

```python
def _load_raster_target(
    path: Path,
) -> Tuple[
    NDArray[np.float32],
    Tuple[float, float],
    Tuple[float, float],
    NDArray[np.float32],
]:
    with np.load(path, allow_pickle=True) as data:
        grid = data["grid"].astype(np.float32)
        start_position = tuple(float(value) for value in data["start_position"])
        start_direction = tuple(
            float(value) for value in data["start_direction_vector"]
        )
        direction_vectors = data["direction_vectors"].astype(np.float32)
    if direction_vectors.shape != (5, 2):
        raise ValueError(
            f"{path} direction_vectors must have shape (5, 2), "
            f"got {direction_vectors.shape}"
        )
    return grid, start_position, start_direction, direction_vectors
```

Update `LLMGridProbeDataset.__getitem__`:

```python
full_grid, start_position, start_direction, direction_vectors = _load_raster_target(
    example.raster_path
)
...
"target_text": serialize_grid_target(
    full_grid,
    direction_vectors=direction_vectors,
    scale=self.scale,
    mentioned_objects=mentioned_objects,
    mentioned_regions=mentioned_regions,
),
```

- [ ] **Step 4: Run dataset and training fixture tests**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py::test_llm_grid_probe_dataset_serializes_direction_vectors tests/etp_llm/test_train_llm_grid_probe.py::test_train_model_validates_parameters_and_writes_outputs -q
```

Expected: pass after test fixtures use the candidate target text.

- [ ] **Step 5: Commit dataset migration**

Run:

```bash
git add vlnce_baselines/models/etp_llm/train_llm_grid_probe.py tests/etp_llm/test_train_llm_grid_probe.py
git commit -m "feat(llm-grid): train with direction vectors"
```

---

### Task 5: Add Direction-Vector Metrics

**Files:**
- Modify: `tests/etp_llm/test_train_llm_grid_probe.py`
- Modify: `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py`

- [ ] **Step 1: Write metric tests**

Add:

```python
def test_evaluate_grid_probe_prediction_reports_direction_vector_metrics():
    target_grid = np.zeros((37, 50, 50), dtype=np.float32)
    target_direction_vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    generated_text = (
        '{"predicted_regions":[],"predicted_objects":[],"regions":{},'
        '"objects":{},"direction_vectors":[[1.0,0.0],[0.0,1.0],'
        '[0.0,0.0],[0.0,0.0],[0.0,0.0]]}'
    )

    metrics = train_llm_grid_probe.evaluate_grid_probe_prediction(
        generated_text,
        target_grid,
        target_direction_vectors,
    )

    assert metrics["direction_vector_valid_rate"] == pytest.approx(1.0)
    assert metrics["direction_vector_l2"] == pytest.approx(0.0)
    assert metrics["direction_vector_cosine"] == pytest.approx(1.0)
    assert metrics["direction_vector_padding_accuracy"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run metric test to verify failure**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py::test_evaluate_grid_probe_prediction_reports_direction_vector_metrics -q
```

Expected: fail because `evaluate_grid_probe_prediction` does not accept target direction vectors.

- [ ] **Step 3: Implement metrics**

Add:

```python
def _direction_vector_metrics(
    predicted: NDArray[np.float32],
    target: NDArray[np.float32],
) -> Dict[str, float]:
    if predicted.shape != (5, 2) or target.shape != (5, 2):
        raise ValueError("direction vectors must have shape (5, 2)")
    l2 = np.linalg.norm(predicted - target, axis=1)
    target_non_padding = np.linalg.norm(target, axis=1) > 0
    predicted_non_padding = np.linalg.norm(predicted, axis=1) > 0
    cosine_values: List[float] = []
    for pred, truth, use_row in zip(predicted, target, target_non_padding):
        if not use_row:
            continue
        denom = float(np.linalg.norm(pred) * np.linalg.norm(truth))
        cosine_values.append(float(np.dot(pred, truth) / denom) if denom else 0.0)
    padding_accuracy = float(
        np.mean(predicted_non_padding == target_non_padding)
    )
    return {
        "direction_vector_valid_rate": 1.0,
        "direction_vector_l2": float(np.mean(l2)),
        "direction_vector_cosine": (
            float(np.mean(cosine_values)) if cosine_values else 1.0
        ),
        "direction_vector_padding_accuracy": padding_accuracy,
    }
```

Change `evaluate_grid_probe_prediction` signature:

```python
def evaluate_grid_probe_prediction(
    generated_text: str,
    target_grid: NDArray[np.float32],
    target_direction_vectors: NDArray[np.float32],
) -> Dict[str, float]:
```

On parse success:

```python
metrics.update(
    _direction_vector_metrics(parsed.direction_vectors, target_direction_vectors)
)
```

On parse failure include zeros:

```python
"direction_vector_valid_rate": 0.0,
"direction_vector_l2": 0.0,
"direction_vector_cosine": 0.0,
"direction_vector_padding_accuracy": 0.0,
```

- [ ] **Step 4: Update evaluator call sites**

Add `target_direction_vectors` to `LLMGridProbeItem` and pass it into evaluation:

```python
metrics = evaluate_grid_probe_prediction(
    generated_text,
    item["target_grid"],
    item["target_direction_vectors"],
)
```

- [ ] **Step 5: Run metrics and eval tests**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py::test_evaluate_grid_probe_prediction_reports_direction_vector_metrics tests/etp_llm/test_train_llm_grid_probe.py::test_evaluate_model_writes_metrics_and_prediction_artifact -q
```

Expected: pass.

- [ ] **Step 6: Commit metrics**

Run:

```bash
git add vlnce_baselines/models/etp_llm/train_llm_grid_probe.py tests/etp_llm/test_train_llm_grid_probe.py
git commit -m "feat(llm-grid): score direction vectors"
```

---

### Task 6: Update Submit Script and Docs References

**Files:**
- Modify: `scripts/submit/llm-grid-probe-train.sh`
- Modify: `prior/README.md`
- Modify: `CONTEXT.md` if runtime naming text is stale after implementation.

- [ ] **Step 1: Update submit output path**

Change:

```bash
--output-dir outputs/llm_grid_probe/r2r-legacy-r1p5-direction5-scale2
```

to:

```bash
--output-dir outputs/llm_grid/r2r-legacy-r1p5-direction5-scale2
```

Keep:

```bash
--batch-size 1
--gradient-accumulation-steps 2
--gradient-checkpointing
--max-new-tokens 4096
```

- [ ] **Step 2: Update docs examples**

In `prior/README.md`, replace the LLM-Grid target example with:

```json
{"predicted_regions":["living/social space"],"predicted_objects":["chair"],"regions":{"living/social space":{"cells":[[1,2],[1,3]],"mentioned":false}},"objects":{"chair":{"cells":[[4,5]],"mentioned":true}},"direction_vectors":[[1.0,0.0],[0.0,1.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]}
```

State that `direction_vectors` is exactly five `[dx,dz]` vectors and that old
`region_candidates` / `object_candidates` keys are not valid for the candidate
schema.

- [ ] **Step 3: Run shell and docs-facing tests**

Run:

```bash
bash -n scripts/submit/llm-grid-probe-train.sh
pytest tests/test_llm_grid_samples.py tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: pass.

- [ ] **Step 4: Commit script/docs**

Run:

```bash
git add scripts/submit/llm-grid-probe-train.sh prior/README.md CONTEXT.md
git commit -m "docs(llm-grid): document candidate target"
```

---

### Task 7: Full Verification

**Files:**
- No source edits unless verification finds a real issue.

- [ ] **Step 1: Run lint**

Run:

```bash
ruff check prior/llm_grid_samples.py vlnce_baselines/models/etp_llm/train_llm_grid_probe.py tests/test_llm_grid_samples.py tests/etp_llm/test_train_llm_grid_probe.py
```

Expected: `All checks passed!`

- [ ] **Step 2: Run type checking**

Run:

```bash
ty check prior/llm_grid_samples.py vlnce_baselines/models/etp_llm/train_llm_grid_probe.py tests/test_llm_grid_samples.py tests/etp_llm/test_train_llm_grid_probe.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Run focused tests**

Run:

```bash
pytest tests/test_llm_grid_samples.py tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run full tests**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Confirm no stray changes**

Run:

```bash
git status --short
```

Expected: empty output.
