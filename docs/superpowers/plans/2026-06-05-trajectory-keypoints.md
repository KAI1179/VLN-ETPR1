# Trajectory Keypoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the bbox-based cognitive-map pipeline around dense ground-truth trajectories for relevance and compact trajectory keypoints for all `(5, 2)` model metadata.

**Architecture:** `VLNCEEpisodeEntry` and ETP-R1 annotations expose a required `ground_truth_trajectory`. Bbox relevance uses the full dense trajectory, then derives selected-level `trajectory_keypoints` using the old abrupt-turn rule plus final point and zero padding. Internal tensors, caches, LLM-Boxes APIs, and prompts use `trajectory_keypoints`; `reference_path` remains only where it truly means a non-keypoint external path, and should disappear from bbox/model-metadata code.

**Tech Stack:** Python, pydantic, NumPy npz caches, PyTorch tensors, pytest.

---

## File Structure

- Modify `prior/vlnce.py`: load VLN-CE ground-truth files and expose `ground_truth_trajectory`.
- Create `prior/trajectory.py`: pure trajectory/keypoint helpers.
- Modify `prior/bbox/_relevance.py` and `prior/bbox/_types.py`: accept dense trajectory, store dense selected-level trajectory and compact keypoints.
- Modify `prior/grid_map/__init__.py`: replace `CognitiveGridMap.reference_path` with `trajectory_keypoints`.
- Modify `vlnce_baselines/models/etp_prior_gt/map_utils.py` and `map_encoder.py`: rename tensors and validate keypoint metadata.
- Modify PriorGT/Imagined/LLM/pretrain callers: rename `reference_paths` tensor/key/API to `trajectory_keypoints`.
- Modify `prior/bbox/__main__.py`, `prior/__main__.py`, `prior/etp_r1/__main__.py`: generation uses dense trajectory, warns/skips bad entries.
- Modify `vlnce_baselines/models/etp_llm/*`: generated text entity becomes `keypoints`; schema/API becomes `trajectory_keypoints`.
- Modify tests under `tests/`, `tests/etp_imagined/`, `tests/etp_llm/`, `tests/etp_prior_gt/`: assert new contracts.
- Modify `prior/README.md`, `vlnce_baselines/models/etp_prior_gt/README.md`, `vlnce_baselines/models/etp_imagined/README.md`, `vlnce_baselines/models/etp_llm/README.md`: document new names and cache break.

### Task 1: VLN-CE Dense Trajectory Loading

**Files:**
- Modify: `prior/vlnce.py`
- Test: `tests/test_vlnce.py`

- [ ] **Step 1: Write failing VLN-CE loader tests**

Replace `test_vlnce_episode_entry_iter_from_uses_episode_reference_path_without_gt` with:

```python
def test_vlnce_episode_entry_iter_from_uses_ground_truth_trajectory(tmp_path, monkeypatch):
    from prior import vlnce

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    payload = {
        "episodes": [
            {
                "scene_id": "mp3d/TestScene/TestScene.glb",
                "episode_id": 7,
                "instruction": {
                    "instruction_text": "Walk to the table.",
                    "instruction_tokens": [1, 2, 3],
                },
                "start_position": [0.0, 0.0, 0.0],
                "start_rotation": [0.0, 0.0, 0.0, 1.0],
                "reference_path": [[99.0, 0.0, 99.0]],
            }
        ]
    }
    gt_payload = {"7": {"locations": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]}}
    with gzip.open(split_dir / "train.json.gz", "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    with gzip.open(split_dir / "train_gt.json.gz", "wt", encoding="utf-8") as f:
        json.dump(gt_payload, f)

    monkeypatch.setattr(vlnce, "R2R_DIR", tmp_path)

    entries = list(vlnce.VLNCEEpisodeEntry.iter_from("R2R", splits=["train"]))

    assert len(entries) == 1
    assert entries[0].scene_id == "TestScene"
    assert entries[0].episode_id == 7
    assert entries[0].ground_truth_trajectory == [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert not hasattr(entries[0], "reference_path")
    assert not hasattr(entries[0], "positions")
```

Add:

```python
def test_vlnce_episode_entry_iter_from_fails_when_gt_missing(tmp_path, monkeypatch):
    from prior import vlnce

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    with gzip.open(split_dir / "train.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"episodes": []}, f)
    monkeypatch.setattr(vlnce, "R2R_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing VLN-CE ground-truth file"):
        list(vlnce.VLNCEEpisodeEntry.iter_from("R2R", splits=["train"]))
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_vlnce.py -q`

Expected: tests fail because `ground_truth_trajectory` and GT file loading do not exist.

- [ ] **Step 3: Implement VLN-CE loader contract**

In `prior/vlnce.py`:

- Replace `reference_path: list[list[float]]` with `ground_truth_trajectory: list[list[float]]`.
- `_file_for_split()` returns `(data_path, gt_path)`.
- `iter_from()` opens both files, raises `FileNotFoundError(f"Missing VLN-CE ground-truth file: {gt_path}")` if GT file missing.
- For each episode, require `gt_data[str(episode_id)]["locations"]`; missing key raises `ValueError(f"Missing ground-truth trajectory for {dataset} {split} episode {episode_id}")`.
- Keep `unique_id` unchanged.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_vlnce.py -q`

Expected: pass.

### Task 2: Trajectory Keypoint Helper

**Files:**
- Create: `prior/trajectory.py`
- Test: `tests/test_trajectory.py`

- [ ] **Step 1: Write failing trajectory helper tests**

Create `tests/test_trajectory.py`:

```python
import pytest

from prior.trajectory import (
    TRAJECTORY_KEYPOINT_COUNT,
    select_trajectory_keypoints,
)


def test_select_trajectory_keypoints_keeps_start_abrupt_turn_and_final():
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
    ]

    assert select_trajectory_keypoints(points) == [
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]


def test_select_trajectory_keypoints_straight_path_keeps_start_and_final():
    assert select_trajectory_keypoints([(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)]) == [
        (0.0, 0.0),
        (3.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]


def test_select_trajectory_keypoints_rejects_too_short_path():
    with pytest.raises(ValueError, match="at least 2 selected-level points"):
        select_trajectory_keypoints([(0.0, 0.0)])


def test_keypoint_count_is_five():
    assert TRAJECTORY_KEYPOINT_COUNT == 5
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_trajectory.py -q`

Expected: import failure for `prior.trajectory`.

- [ ] **Step 3: Implement helper**

Create `prior/trajectory.py` with:

```python
from __future__ import annotations

from math import hypot
from typing import Sequence, Tuple

Point2D = Tuple[float, float]

TRAJECTORY_KEYPOINT_COUNT = 5
TRAJECTORY_KEYPOINT_SIMILARITY_THRESHOLD = 0.8


def _segment_direction(start: Point2D, end: Point2D) -> Point2D | None:
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    norm = hypot(dx, dz)
    if norm == 0.0:
        return None
    return dx / norm, dz / norm


def select_trajectory_keypoints(points: Sequence[Point2D]) -> list[Point2D]:
    if len(points) < 2:
        raise ValueError(
            "ground_truth_trajectory must contain at least 2 selected-level points "
            "for trajectory_keypoints"
        )

    keypoints: list[Point2D] = [(float(points[0][0]), float(points[0][1]))]
    previous_direction: Point2D | None = None

    for index in range(len(points) - 1):
        direction = _segment_direction(points[index], points[index + 1])
        if direction is None:
            continue
        if previous_direction is not None:
            similarity = (
                direction[0] * previous_direction[0]
                + direction[1] * previous_direction[1]
            )
            if similarity < TRAJECTORY_KEYPOINT_SIMILARITY_THRESHOLD:
                turn_point = (float(points[index][0]), float(points[index][1]))
                if turn_point != keypoints[-1]:
                    keypoints.append(turn_point)
        previous_direction = direction
        if len(keypoints) >= TRAJECTORY_KEYPOINT_COUNT - 1:
            break

    final_point = (float(points[-1][0]), float(points[-1][1]))
    if final_point != keypoints[-1] and len(keypoints) < TRAJECTORY_KEYPOINT_COUNT:
        keypoints.append(final_point)

    return keypoints + [(0.0, 0.0)] * (TRAJECTORY_KEYPOINT_COUNT - len(keypoints))
```

- [ ] **Step 4: Verify**

Run: `pytest tests/test_trajectory.py -q`

Expected: pass.

### Task 3: Bbox Relevance Uses Dense Trajectory

**Files:**
- Modify: `prior/bbox/_relevance.py`
- Modify: `prior/bbox/_types.py`
- Test: `tests/test_box_construct.py`

- [ ] **Step 1: Write failing bbox contract tests**

Update `test_scene_semantic_boxes_relevant_to_returns_typed_relevant_level` expected fields:

```python
assert relevant.ground_truth_trajectory == [(0.0, 0.0)]
assert relevant.trajectory_keypoints == [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
assert not hasattr(relevant, "reference_path")
```

Add:

```python
def test_relevant_semantic_boxes_uses_dense_trajectory_for_relevance_and_keypoints_for_metadata():
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
    )
    level.objects[3] = [box.OBB2D(center=(2.0, 0.0), half_extents=(0.4, 0.4))]

    relevant = box.SceneSemanticBoxes([level]).relevant_to(
        "walk to the table",
        ground_truth_trajectory=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
        ],
        start_direction_vector=(0.0, 1.0),
        max_distance=0.5,
        category_extractor=lambda instruction: ({3}, set()),
    )

    assert [obb.center for obb in relevant.level.objects[3]] == [(2.0, 0.0)]
    assert relevant.ground_truth_trajectory == [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
    ]
    assert relevant.trajectory_keypoints[:3] == [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0)]
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_box_construct.py -q`

Expected: failures for keyword/name changes.

- [ ] **Step 3: Implement bbox contract**

In `RelevantSemanticBoxes`, replace:

- `reference_path: List[Point2D]`

with:

- `ground_truth_trajectory: List[Point2D]`
- `trajectory_keypoints: List[Point2D]`

In `SceneSemanticBoxes.relevant_to`, rename parameter to `ground_truth_trajectory`.

In `_relevance.py`:

- Rename `_first_encountered_level(..., reference_path)` to use `ground_truth_trajectory`.
- Build `level_points` from every dense selected-level point.
- Call `select_trajectory_keypoints(level_points)`.
- Return both fields.

In `rotate_by_right_angle`, rotate both `ground_truth_trajectory` and `trajectory_keypoints`.

- [ ] **Step 4: Verify**

Run:

```bash
pytest tests/test_box_construct.py tests/test_trajectory.py -q
```

Expected: pass.

### Task 4: CognitiveGridMap Cache Break

**Files:**
- Modify: `prior/grid_map/__init__.py`
- Modify: `vlnce_baselines/models/etp_prior_gt/map_utils.py`
- Test: `tests/test_grid_map_cache.py`
- Test: `tests/test_vis_rotate.py`

- [ ] **Step 1: Write failing cache tests**

In `tests/test_grid_map_cache.py`, rename fixtures/assertions:

```python
grid_map.trajectory_keypoints = [(1.0, 2.0), (2.0, 3.0)]
assert loaded.trajectory_keypoints == grid_map.trajectory_keypoints
assert not hasattr(loaded, "reference_path")
```

Add:

```python
def test_cognitive_grid_map_load_rejects_old_reference_path_cache(tmp_path: Path):
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        grid=np.zeros_like(CognitiveGridMap().grid),
        range_y=np.asarray([None, None], dtype=object),
        reference_path=np.asarray([(1.0, 2.0)], dtype=np.float32),
        start_direction_vector=np.asarray((0.0, 1.0), dtype=np.float32),
    )

    with pytest.raises(KeyError, match="trajectory_keypoints"):
        CognitiveGridMap.load(path)
```

Rename tensor assertions:

```python
assert torch.equal(tensors["trajectory_keypoints"][0], torch.tensor([2.0, 4.0]))
assert "reference_paths" not in tensors
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest tests/test_grid_map_cache.py tests/test_vis_rotate.py -q
```

Expected: failures for missing `trajectory_keypoints`.

- [ ] **Step 3: Implement cache/tensor rename**

In `CognitiveGridMap`:

- Rename attribute `reference_path` to `trajectory_keypoints`.
- Save npz key `trajectory_keypoints`.
- Load requires `trajectory_keypoints`; do not read `reference_path`.
- Visualization plots `trajectory_keypoints`.

In `map_utils.py`:

- Rename `_reference_path_to_grid_tensor` to `_trajectory_keypoints_to_grid_tensor`.
- Rename returned key `reference_paths` to `trajectory_keypoints`.
- Rename rotation code to rotate `trajectory_keypoints`.
- `start_position` comes from first `trajectory_keypoints` point.

- [ ] **Step 4: Verify**

Run:

```bash
pytest tests/test_grid_map_cache.py tests/test_vis_rotate.py -q
```

Expected: pass.

### Task 5: Generation Entry Skips Bad Trajectories

**Files:**
- Modify: `prior/__main__.py`
- Modify: `prior/bbox/__main__.py`
- Modify: `prior/etp_r1/__main__.py`
- Test: `tests/test_bbox_cli.py`

- [ ] **Step 1: Write failing generation behavior tests**

Add a focused test around `prior.bbox.__main__._relevant_episode_boxes` using a monkeypatched entry with `ground_truth_trajectory=[[0.0, 0.0, 0.0]]`; assert `ValueError` message contains `at least 2 selected-level points`.

For full generation loops, assert stdout contains:

```text
WARNING: skipping
```

when helper raises that error.

- [ ] **Step 2: Implement generation behavior**

In generation scripts:

- Use `episode.ground_truth_trajectory` or `entry.positions()` as `ground_truth_trajectory`.
- Wrap `scene_boxes.relevant_to(...)` and `to_cognitive_map()` per entry.
- On `ValueError` from keypoint selection, print `WARNING: skipping {id}: {error}` and continue.
- At end, print summary counts `generated=N skipped=M`.
- Keep training/cache loaders strict: no missing-cache fallback.

- [ ] **Step 3: Verify**

Run:

```bash
pytest tests/test_bbox_cli.py -q
```

Expected: pass.

### Task 6: Model Tensor Rename Everywhere

**Files:**
- Modify: `vlnce_baselines/models/etp_prior_gt/map_encoder.py`
- Modify: `vlnce_baselines/models/etp_prior_gt/policy.py`
- Modify: `vlnce_baselines/ss_trainer_ETP_PriorGT.py`
- Modify: `vlnce_baselines/GRPO_trainer_ETP_PriorGT.py`
- Modify: `pretrain_src/pretrain_src/data/dataset.py`
- Modify: `pretrain_src/pretrain_src/data/tasks.py`
- Modify: `pretrain_src/pretrain_src/model/pretrain_cmt.py`
- Modify: `vlnce_baselines/models/etp_imagined/*`
- Test: `tests/etp_prior_gt/test_map_encoder_tokens.py`
- Test: `tests/etp_imagined/test_*`
- Test: `tests/test_cli_parsers.py`

- [ ] **Step 1: Global search**

Run:

```bash
rg -n "reference_paths|reference_path_loss|pred_reference_paths|target_reference_paths|reference path" vlnce_baselines pretrain_src tests
```

Expected: list of names to replace where they refer to `(B,5,2)` metadata.

- [ ] **Step 2: Rename map encoder API**

In `map_encoder.py`, rename:

- `reference_paths` -> `trajectory_keypoints`
- error string to `trajectory_keypoints must have shape (B, 5, 2)`
- docstring `(B, REFERENCE_PATH_LENGTH, 2)` -> `(B, TRAJECTORY_KEYPOINT_COUNT, 2)` if constants are renamed.

Keep shape `(B, 5, 2)`.

- [ ] **Step 3: Rename tensor keys and model loss names**

Every batch/cache/model dict key that means `(B,5,2)` becomes `trajectory_keypoints`.

Rename CLI/config only if semantically tied to target:

- `reference_path_loss_weight` -> `trajectory_keypoint_loss_weight`

No alias CLI flag unless explicit user request.

- [ ] **Step 4: Verify no metadata misuse remains**

Run:

```bash
rg -n "reference_paths|reference_path_loss|pred_reference_paths|target_reference_paths" vlnce_baselines pretrain_src tests
```

Expected: no hits, except LLM text/history docs explicitly about old term removal.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/etp_prior_gt/test_map_encoder_tokens.py tests/etp_imagined tests/test_cli_parsers.py -q
```

Expected: pass.

### Task 7: LLM-Boxes Keypoints API/Text

**Files:**
- Modify: `vlnce_baselines/models/etp_llm/boxes_schema.py`
- Modify: `vlnce_baselines/models/etp_llm/prompts/llm_boxes_system.md`
- Modify: `vlnce_baselines/models/etp_llm/train_llm_boxes.py`
- Modify: `vlnce_baselines/models/etp_llm/generate_navigation_cache.py`
- Modify: `vlnce_baselines/models/etp_llm/navigation.py`
- Test: `tests/etp_llm/test_boxes_schema.py`
- Test: `tests/etp_llm/test_train_llm_boxes.py`
- Test: `tests/etp_llm/test_generate_navigation_cache.py`

- [ ] **Step 1: Write failing LLM schema tests**

Update tests to require:

- schema property `trajectory_keypoints`
- generated compact text entity starts with `keypoints`
- parser rejects old `path` entity with validation error
- `spec_to_relevant_semantic_boxes` accepts `trajectory_keypoints`

- [ ] **Step 2: Implement LLM rename**

In LLM code:

- Rename `LLMBoxesSpec.reference_path` -> `trajectory_keypoints`.
- Parser accepts `keypoints` entity, not `path`.
- Prompt says: `Emit one keypoints entity first when trajectory keypoints are predicted.`
- Artifacts use `trajectory_keypoints` field.

- [ ] **Step 3: Verify**

Run:

```bash
pytest tests/etp_llm -q
```

Expected: pass.

### Task 8: Docs And Final Verification

**Files:**
- Modify: `prior/README.md`
- Modify: `vlnce_baselines/models/etp_prior_gt/README.md`
- Modify: `vlnce_baselines/models/etp_imagined/README.md`
- Modify: `vlnce_baselines/models/etp_llm/README.md`
- Already modified: `CONTEXT.md`
- Already added: `docs/adr/0008-use-trajectory-keypoints-for-map-metadata.md`

- [ ] **Step 1: Update docs**

Replace old wording:

- `reference_path` in NPZ -> `trajectory_keypoints`
- `reference_paths` tensor -> `trajectory_keypoints`
- LLM `path` entity -> `keypoints`
- Note: cache format changed; regenerate bbox-based cognitive-map caches.

- [ ] **Step 2: Run global stale-name search**

Run:

```bash
rg -n "reference_path|reference_paths|path entity|direction_vectors" prior vlnce_baselines pretrain_src tests docs CONTEXT.md
```

Expected: remaining hits only where `reference_path` truly means non-keypoint external path, or in historical ADR/commit notes. Remove all bbox/model-metadata misuse.

- [ ] **Step 3: Run relevant tests**

Run:

```bash
pytest tests/test_vlnce.py tests/test_trajectory.py tests/test_box_construct.py tests/test_grid_map_cache.py tests/test_bbox_cli.py tests/etp_prior_gt tests/etp_imagined tests/etp_llm tests/test_cli_parsers.py -q
```

Expected: pass.

- [ ] **Step 4: Run formatting/lint if repo has configured commands**

Inspect `pyproject.toml`, `setup.cfg`, `ruff.toml`, and Makefile. Run configured formatter/linter/type checker. If none exists, state that no configured lint/type command exists.

---

## Self-Review

Spec coverage:

- Dense ground-truth trajectory restored for bbox relevance: Tasks 1, 3, 5.
- First 5 abrupt-turn keypoints plus final and zero padding: Task 2.
- Rename away from `reference_path` where not exact: Tasks 3, 4, 6, 7, 8.
- LLM predicts trajectory keypoints via `keypoints`: Task 7.
- No compatibility slop: Tasks 4, 6, 7, 8.
- Warning/skip for bad generation entries, strict training cache loading: Task 5.
- ETP-R1 annotation path uses best available positions: Task 5.

No placeholders remain. Type names are consistent: `ground_truth_trajectory`, `trajectory_keypoints`, `keypoints`.
