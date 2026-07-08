# LLM-Grid-Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a predictor-only LLM-Grid-Probe trainer/evaluator that fine-tunes a causal LM to emit compact JSON grids from VLN-CE instructions and Try5-style cached raster maps.

**Architecture:** Keep v1 beside LLM-Boxes under `vlnce_baselines.models.etp_llm`, reuse the existing LLM-Boxes chat/LoRA/EOS mechanics, and add only grid-specific target parsing, metrics, prompt text, and dataset loading. The probe reads VLN-CE raster caches from `data/cognitive_maps/gt.legacy.r1p5.direction5.v1`, trains on `scale=2` compact JSON, and reports predictor metrics directly against the downsampled grid without generating navigation caches.

**Tech Stack:** Python, `tap.Tap`, NumPy, PyTorch `DataLoader`, Hugging Face causal LM APIs through existing LLM-Boxes helpers, pytest, ruff, ty, Bash submit scripts.

---

## File Structure

- Modify `prior/llm_grid_samples.py`: expose the existing grid downsampler as a public helper used by training and sampling.
- Create `vlnce_baselines/models/etp_llm/prompts/llm_grid_probe_system.md`: system prompt with compact JSON schema and all 37 category ids.
- Create `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py`: dataset loading, JSON parsing, grid metrics, collators, train/eval CLI.
- Create `tests/etp_llm/test_train_llm_grid_probe.py`: unit tests for target serialization, parsing, metrics, dataset loading, EOS-safe collate, parser defaults, and train/eval boundaries.
- Create `scripts/submit/llm-grid-probe-train.sh`: Slurm launcher for the first R2R VLN-CE grid probe run.

Do not add VLN policy integration, direction-vector prediction, ETP-R1 pretraining cache support, grammar decoding, or shared utility extraction in this pass.

---

### Task 1: Expose Downsampled Grid Targets

**Files:**
- Modify: `prior/llm_grid_samples.py`
- Create: `tests/etp_llm/test_train_llm_grid_probe.py`

- [ ] **Step 1: Write failing downsample and serialization tests**

Create `tests/etp_llm/test_train_llm_grid_probe.py` with this initial content:

```python
import json

import numpy as np
import pytest

from prior.llm_grid_samples import downsample_grid, serialize_grid_target


def test_downsample_grid_scale_2_max_pools_cells():
    grid = np.zeros((37, 4, 4), dtype=np.float32)
    grid[1, 0, 1] = 0.25
    grid[1, 1, 0] = 1.0
    grid[28, 3, 3] = 0.5

    sampled = downsample_grid(grid, scale=2)

    assert sampled.shape == (37, 2, 2)
    assert sampled[1, 0, 0] == pytest.approx(1.0)
    assert sampled[28, 1, 1] == pytest.approx(0.5)
    assert int(np.count_nonzero(sampled)) == 2


def test_serialize_grid_target_uses_compact_json_and_omits_unit_values():
    grid = np.zeros((37, 4, 4), dtype=np.float32)
    grid[1, 0, 0] = 1.0
    grid[28, 2, 2] = 0.6

    text = serialize_grid_target(grid, scale=2)

    assert text == '{"grid":[[1,0,0],[28,1,1,0.6]]}'
    assert json.loads(text) == {"grid": [[1, 0, 0], [28, 1, 1, 0.6]]}
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: FAIL with `ImportError` because `prior.llm_grid_samples.downsample_grid` does not exist.

- [ ] **Step 3: Rename the private downsampler**

In `prior/llm_grid_samples.py`, rename `_downsample_grid` to `downsample_grid`, then update the two internal call sites in `serialize_grid_target` and `analyze_grid_sample`:

```python
def downsample_grid(
    grid: NDArray[np.float32],
    scale: int,
) -> NDArray[np.float32]:
    if scale not in (1, 2):
        raise ValueError(f"scale must be 1 or 2, got {scale}")
    if scale == 1:
        return grid

    channels, rows, cols = grid.shape
    if rows % scale != 0 or cols % scale != 0:
        raise ValueError(
            f"grid shape {(channels, rows, cols)} is not divisible by scale {scale}"
        )
    return grid.reshape(
        channels,
        rows // scale,
        scale,
        cols // scale,
        scale,
    ).max(axis=(2, 4))
```

The two call sites should read:

```python
sampled = downsample_grid(grid, scale)
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: PASS for the two tests.

- [ ] **Step 5: Commit the public downsampler**

Run:

```bash
git add prior/llm_grid_samples.py tests/etp_llm/test_train_llm_grid_probe.py
git commit -m "feat(llm-grid): expose grid downsampling"
```

Expected: commit succeeds with only the downsampler and initial tests.

---

### Task 2: Add Grid JSON Parser, Prompt, and Metrics

**Files:**
- Create: `vlnce_baselines/models/etp_llm/prompts/llm_grid_probe_system.md`
- Create: `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py`
- Modify: `tests/etp_llm/test_train_llm_grid_probe.py`

- [ ] **Step 1: Add parser and metric tests**

Append these tests to `tests/etp_llm/test_train_llm_grid_probe.py`:

```python
from vlnce_baselines.models.etp_llm import train_llm_grid_probe


def test_parse_grid_probe_text_accepts_compact_records_and_max_merges_duplicates():
    result = train_llm_grid_probe.parse_grid_probe_text(
        '{"grid":[[1,0,0],[1,0,0,0.4],[28,1,2,0.6]]}',
        shape=(37, 50, 50),
    )

    assert result.grid.shape == (37, 50, 50)
    assert result.grid[1, 0, 0] == pytest.approx(1.0)
    assert result.grid[28, 1, 2] == pytest.approx(0.6)
    assert result.record_count == 3
    assert result.duplicate_record_count == 1


def test_parse_grid_probe_text_rejects_invalid_json_and_bad_records():
    with pytest.raises(train_llm_grid_probe.LLMGridProbeValidationError):
        train_llm_grid_probe.parse_grid_probe_text("not json")
    with pytest.raises(train_llm_grid_probe.LLMGridProbeValidationError):
        train_llm_grid_probe.parse_grid_probe_text('{"grid":[[37,0,0]]}')
    with pytest.raises(train_llm_grid_probe.LLMGridProbeValidationError):
        train_llm_grid_probe.parse_grid_probe_text('{"grid":[[1,0,0,1.2]]}')


def test_compute_grid_probe_metrics_counts_invalid_predictions_explicitly():
    target = np.zeros((37, 50, 50), dtype=np.float32)
    target[1, 0, 0] = 1.0
    target[28, 1, 2] = 0.6

    valid = train_llm_grid_probe.evaluate_grid_probe_prediction(
        '{"grid":[[1,0,0],[28,9,9]]}',
        target,
    )
    invalid = train_llm_grid_probe.evaluate_grid_probe_prediction(
        "not json",
        target,
    )

    assert valid["json_valid"] == 1.0
    assert valid["schema_valid"] == 1.0
    assert valid["cell_precision"] == pytest.approx(0.5)
    assert valid["cell_recall"] == pytest.approx(0.5)
    assert valid["cell_f1"] == pytest.approx(0.5)
    assert valid["category_aware_raster_iou"] == pytest.approx(1 / 3)
    assert valid["category_aware_raster_recall"] == pytest.approx(0.5)
    assert valid["duplicate_record_count"] == 0.0
    assert valid["duplicate_record_rate"] == 0.0
    assert invalid["json_valid"] == 0.0
    assert invalid["schema_valid"] == 0.0
    assert invalid["cell_precision"] == 0.0
    assert invalid["cell_recall"] == 0.0
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: FAIL because `train_llm_grid_probe` does not exist.

- [ ] **Step 3: Add the system prompt**

Create `vlnce_baselines/models/etp_llm/prompts/llm_grid_probe_system.md`:

```markdown
You predict a Try5-style cognitive-map raster grid from a navigation instruction.

Return only compact JSON with this exact shape:
{"grid":[[category,row,col],[category,row,col,value]]}

Each record marks one nonzero grid cell.
- category is an integer from 0 to 36.
- row and col are integers from 0 to 49 for scale 2 grids.
- value is optional; omit it when the value is 1.0.
- if present, value must be a number from 0.0 to 1.0.
- do not include markdown, code fences, comments, prose, or extra keys.

Category ids:
0 void
1 chair
2 door
3 table
4 cushion
5 sofa
6 bed
7 plant
8 sink
9 toilet
10 tv_monitor
11 shower
12 bathtub
13 counter
14 appliances
15 structure
16 other
17 free-space
18 picture
19 cabinet
20 chest_of_drawers
21 stool
22 towel
23 fireplace
24 gym_equipment
25 seating
26 clothes
27 outdoor/semi-outdoor
28 living/social space
29 recreation/fitness
30 utility/service
31 work/study
32 circulation
33 private room
34 bathroom/sanitary
35 dining/food
36 other/miscellaneous
```

- [ ] **Step 4: Add parser and metric implementation**

Create `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py` with this initial content:

```python
"""Dataset, training, and evaluation CLI for the LLM-Grid-Probe milestone."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

GRID_CHANNELS = 37
GRID_SCALE = 2
GRID_SHAPE = (GRID_CHANNELS, 50, 50)
DEFAULT_GRID_NAMESPACE = "gt.legacy.r1p5.direction5.v1"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "llm_grid_probe_system.md"


class LLMGridProbeValidationError(ValueError):
    """Raised when generated LLM-Grid-Probe JSON fails validation."""


@dataclass(frozen=True)
class ParsedGridProbe:
    grid: NDArray[np.float32]
    record_count: int
    duplicate_record_count: int


def parse_grid_probe_text(
    text: str,
    shape: Tuple[int, int, int] = GRID_SHAPE,
) -> ParsedGridProbe:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMGridProbeValidationError(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise LLMGridProbeValidationError("top-level JSON value must be an object")
    if set(payload) != {"grid"}:
        raise LLMGridProbeValidationError("top-level JSON object must contain only grid")
    records = payload["grid"]
    if not isinstance(records, list):
        raise LLMGridProbeValidationError("grid must be a list")

    channels, rows, cols = shape
    grid = np.zeros(shape, dtype=np.float32)
    seen: set[tuple[int, int, int]] = set()
    duplicates = 0
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, list) or len(raw_record) not in (3, 4):
            raise LLMGridProbeValidationError(
                f"grid[{index}] must be [category,row,col] or [category,row,col,value]"
            )
        category, row, col = raw_record[:3]
        if not all(isinstance(value, int) for value in (category, row, col)):
            raise LLMGridProbeValidationError(f"grid[{index}] category,row,col must be ints")
        if not (0 <= category < channels and 0 <= row < rows and 0 <= col < cols):
            raise LLMGridProbeValidationError(f"grid[{index}] index out of bounds")
        value = 1.0
        if len(raw_record) == 4:
            raw_value = raw_record[3]
            if not isinstance(raw_value, (int, float)):
                raise LLMGridProbeValidationError(f"grid[{index}] value must be numeric")
            value = float(raw_value)
        if not (0.0 <= value <= 1.0):
            raise LLMGridProbeValidationError(f"grid[{index}] value out of range")
        key = (category, row, col)
        if key in seen:
            duplicates += 1
        seen.add(key)
        grid[category, row, col] = max(float(grid[category, row, col]), value)
    return ParsedGridProbe(grid=grid, record_count=len(records), duplicate_record_count=duplicates)


def _binary_grid(grid: NDArray[np.float32]) -> NDArray[np.bool_]:
    return np.asarray(grid > 0, dtype=np.bool_)


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _grid_metrics(
    pred_grid: NDArray[np.float32],
    target_grid: NDArray[np.float32],
) -> Dict[str, float]:
    pred = _binary_grid(pred_grid)
    target = _binary_grid(target_grid)
    intersection = int(np.logical_and(pred, target).sum())
    pred_count = int(pred.sum())
    target_count = int(target.sum())
    union = int(np.logical_or(pred, target).sum())
    precision = _safe_div(intersection, pred_count)
    recall = _safe_div(intersection, target_count)
    f1 = _safe_div(2 * intersection, pred_count + target_count)
    return {
        "cell_precision": precision,
        "cell_recall": recall,
        "cell_f1": f1,
        "category_aware_raster_iou": _safe_div(intersection, union),
        "category_aware_raster_recall": recall,
        "category_aware_raster_support": float(target_count),
        "predicted_cell_count": float(pred_count),
        "target_cell_count": float(target_count),
    }


def evaluate_grid_probe_prediction(
    generated_text: str,
    target_grid: NDArray[np.float32],
) -> Dict[str, float]:
    try:
        parsed = parse_grid_probe_text(generated_text, shape=target_grid.shape)
    except LLMGridProbeValidationError:
        return {
            "json_valid": 0.0,
            "schema_valid": 0.0,
            "record_count": 0.0,
            "duplicate_record_count": 0.0,
            "duplicate_record_rate": 0.0,
            "cell_precision": 0.0,
            "cell_recall": 0.0,
            "cell_f1": 0.0,
            "category_aware_raster_iou": 0.0,
            "category_aware_raster_recall": 0.0,
            "category_aware_raster_support": float(np.count_nonzero(target_grid > 0)),
            "predicted_cell_count": 0.0,
            "target_cell_count": float(np.count_nonzero(target_grid > 0)),
        }
    metrics = _grid_metrics(parsed.grid, target_grid)
    metrics.update(
        {
            "json_valid": 1.0,
            "schema_valid": 1.0,
            "record_count": float(parsed.record_count),
            "duplicate_record_count": float(parsed.duplicate_record_count),
            "duplicate_record_rate": _safe_div(
                parsed.duplicate_record_count,
                parsed.record_count,
            ),
        }
    )
    return metrics
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: PASS for downsampling, parsing, and metrics tests.

- [ ] **Step 6: Commit parser, metrics, and prompt**

Run:

```bash
git add vlnce_baselines/models/etp_llm/prompts/llm_grid_probe_system.md vlnce_baselines/models/etp_llm/train_llm_grid_probe.py tests/etp_llm/test_train_llm_grid_probe.py
git commit -m "feat(llm-grid): add probe JSON metrics"
```

Expected: commit succeeds with prompt, parser, metrics, and tests.

---

### Task 3: Add VLN-CE Raster Dataset and Collators

**Files:**
- Modify: `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py`
- Modify: `tests/etp_llm/test_train_llm_grid_probe.py`

- [ ] **Step 1: Add dataset and collate tests**

Append these tests to `tests/etp_llm/test_train_llm_grid_probe.py`:

```python
import torch
from pathlib import Path


class _Episode:
    dataset = "R2R"
    split = "train"
    scene_id = "scene-a"
    episode_id = 42
    unique_id = "R2R_train_42"
    instruction = "Go to the chair."
    start_position = [1.2, 0.0, 3.4]
    start_direction_vector = (0.0, 1.0)


class _EpisodeSource:
    calls = []

    @staticmethod
    def iter_from(dataset, splits):
        _EpisodeSource.calls.append((dataset, tuple(splits)))
        yield _Episode()


class _ChatTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ""
        for message in messages:
            text += f"<{message['role']}>{message['content']}</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, texts, max_length, padding, truncation, return_tensors):
        encoded = [self.encode(text, add_special_tokens=False)[:max_length] for text in texts]
        width = max(len(row) for row in encoded)
        input_ids = []
        attention_mask = []
        for row in encoded:
            padded = row + [self.pad_token_id] * (width - len(row))
            input_ids.append(padded)
            attention_mask.append([1] * len(row) + [0] * (width - len(row)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 97 + 3 for char in text]


def test_load_llm_grid_probe_examples_loads_raster_paths(monkeypatch, tmp_path):
    _EpisodeSource.calls = []
    raster_path = tmp_path / "scene-a" / "R2R_train_42.npz"
    raster_path.parent.mkdir()
    np.savez_compressed(raster_path, grid=np.zeros((37, 100, 100), dtype=np.float32))

    monkeypatch.setattr(train_llm_grid_probe, "VLNCEEpisodeEntry", _EpisodeSource)
    monkeypatch.setattr(
        train_llm_grid_probe,
        "cognitive_map_cache_path",
        lambda scene_id, cache_id, namespace: raster_path,
    )

    examples = train_llm_grid_probe.load_llm_grid_probe_examples(
        "R2R",
        ["train"],
        limit=1,
        cognitive_map_namespace="gt.legacy.r1p5.direction5.v1",
    )

    assert _EpisodeSource.calls == [("R2R", ("train",))]
    assert len(examples) == 1
    assert examples[0].example_id == "R2R_train_42"
    assert examples[0].raster_path == raster_path


def test_llm_grid_probe_dataset_builds_boxes_style_input_and_scale_2_target(tmp_path):
    raster_path = tmp_path / "grid.npz"
    grid = np.zeros((37, 100, 100), dtype=np.float32)
    grid[1, 0, 0] = 1.0
    grid[28, 2, 2] = 0.6
    np.savez_compressed(raster_path, grid=grid)
    example = train_llm_grid_probe.LLMGridProbeExample(
        example_id="R2R_train_42",
        dataset_tag="R2R",
        split="train",
        scene_id="scene-a",
        episode_id=42,
        instruction="Go to the chair.",
        start_position=[1.2, 0.0, 3.4],
        start_direction=(0.0, 1.0),
        raster_path=raster_path,
    )

    item = train_llm_grid_probe.LLMGridProbeDataset([example])[0]

    assert item["input_text"] == (
        "dataset R2R | start x = 1.2 | start z = 3.4 | "
        "direction x = 0 | direction z = 1 | instruction Go to the chair."
    )
    assert item["target_text"] == '{"grid":[[1,0,0],[28,1,1,0.6]]}'
    assert item["target_grid"].shape == (37, 50, 50)


def test_collate_llm_grid_probe_masks_prompt_and_padding_tokens(tmp_path):
    raster_path = tmp_path / "grid.npz"
    grid = np.zeros((37, 100, 100), dtype=np.float32)
    grid[1, 0, 0] = 1.0
    np.savez_compressed(raster_path, grid=grid)
    example = train_llm_grid_probe.LLMGridProbeExample(
        example_id="R2R_train_42",
        dataset_tag="R2R",
        split="train",
        scene_id="scene-a",
        episode_id=42,
        instruction="Go to the chair.",
        start_position=[1.2, 0.0, 3.4],
        start_direction=(0.0, 1.0),
        raster_path=raster_path,
    )
    item = train_llm_grid_probe.LLMGridProbeDataset([example])[0]

    batch = train_llm_grid_probe.collate_llm_grid_probe_batch(
        [item],
        tokenizer=_ChatTokenizer(),
        system_prompt="system",
        max_input_length=512,
        max_new_tokens=64,
    )

    labels = batch["labels"][0]
    prompt_length = batch["prompt_lengths"][0]
    assert torch.all(labels[:prompt_length] == -100)
    assert torch.any(labels[prompt_length:] != -100)
    assert batch["example_ids"] == ["R2R_train_42"]
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: FAIL because dataset and collate functions are not defined.

- [ ] **Step 3: Add dataset imports and types**

Extend imports in `train_llm_grid_probe.py`:

```python
import warnings
from typing import Iterable, Iterator, Literal, List, TypedDict

import torch
from torch.utils.data import Dataset

from prior.llm_grid_samples import downsample_grid, serialize_grid_target
from prior.vlnce import VLNCEEpisodeEntry
from vlnce_baselines.models.etp_prior_gt.map_utils import cognitive_map_cache_path

from .boxes_schema import build_llm_boxes_input
from .train_llm_boxes import (
    _causal_lm_labels,
    _render_chat_completion,
    _render_chat_prompt,
    _token_count,
    _validate_supervised_labels,
)
```

Add these definitions after constants:

```python
TRAIN_SPLITS = ("train",)
EVAL_SPLITS = ("val_seen", "val_unseen")


class LLMGridProbeItem(TypedDict):
    input_text: str
    target_text: str
    target_grid: NDArray[np.float32]
    example_id: str
    instruction: str
    start_position: Sequence[float]
    start_direction: Sequence[float]
    scene_id: str
```

- [ ] **Step 4: Add example loading and dataset**

Add this code before `parse_grid_probe_text`:

```python
@dataclass(frozen=True)
class LLMGridProbeExample:
    example_id: str
    dataset_tag: str
    split: str
    scene_id: str
    episode_id: int
    instruction: str
    start_position: Sequence[float]
    start_direction: Sequence[float]
    raster_path: Path


def load_llm_grid_probe_examples(
    dataset: Literal["R2R", "RxR"],
    splits: Iterable[str],
    limit: Optional[int] = None,
    quiet: bool = False,
    skip_missing_cache: bool = False,
    cognitive_map_namespace: str = DEFAULT_GRID_NAMESPACE,
) -> List[LLMGridProbeExample]:
    if limit == 0:
        return []
    examples: List[LLMGridProbeExample] = []
    skipped_missing_cache: List[Tuple[str, str]] = []
    for episode in VLNCEEpisodeEntry.iter_from(dataset, splits=splits):
        raster_path = cognitive_map_cache_path(
            episode.scene_id,
            episode.unique_id,
            namespace=cognitive_map_namespace,
        )
        if not raster_path.exists():
            if not skip_missing_cache:
                raise FileNotFoundError(raster_path)
            warnings.warn(
                f"skipping {episode.unique_id}: missing cached raster: {raster_path}",
                RuntimeWarning,
                stacklevel=2,
            )
            skipped_missing_cache.append((episode.unique_id, str(raster_path)))
            continue
        examples.append(
            LLMGridProbeExample(
                example_id=episode.unique_id,
                dataset_tag=episode.dataset,
                split=episode.split,
                scene_id=episode.scene_id,
                episode_id=episode.episode_id,
                instruction=episode.instruction,
                start_position=episode.start_position,
                start_direction=episode.start_direction_vector,
                raster_path=raster_path,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    if skipped_missing_cache and not quiet:
        print(f"skipped_missing_cache={len(skipped_missing_cache)}")
        for example_id, path in skipped_missing_cache:
            print(f"  {example_id}: {path}")
    return examples


def _load_grid(path: Path) -> NDArray[np.float32]:
    with np.load(path, allow_pickle=True) as data:
        return np.asarray(data["grid"], dtype=np.float32)


class LLMGridProbeDataset(Dataset):
    def __init__(self, examples: Sequence[LLMGridProbeExample], scale: int = GRID_SCALE) -> None:
        self.examples = list(examples)
        self.scale = scale

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LLMGridProbeItem:
        example = self.examples[index]
        full_grid = _load_grid(example.raster_path)
        target_grid = downsample_grid(full_grid, self.scale)
        return {
            "input_text": build_llm_boxes_input(
                example.dataset_tag,
                example.instruction,
                example.start_position,
                example.start_direction,
            ),
            "target_text": serialize_grid_target(full_grid, scale=self.scale),
            "target_grid": target_grid,
            "example_id": example.example_id,
            "instruction": example.instruction,
            "start_position": example.start_position,
            "start_direction": example.start_direction,
            "scene_id": example.scene_id,
        }

    def __iter__(self) -> Iterator[LLMGridProbeItem]:
        for index in range(len(self)):
            yield self[index]
```

- [ ] **Step 5: Add collators**

Add this code before metric helpers:

```python
def collate_llm_grid_probe_batch(
    batch: Sequence[LLMGridProbeItem],
    tokenizer: Any,
    system_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt_texts = [
        _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        for item in batch
    ]
    full_texts = [
        _render_chat_completion(
            tokenizer,
            system_prompt,
            item["input_text"],
            item["target_text"],
        )
        for item in batch
    ]
    prompt_lengths = [_token_count(tokenizer, text) for text in prompt_texts]
    encoded = tokenizer(
        full_texts,
        max_length=max_input_length + max_new_tokens,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded["labels"] = _causal_lm_labels(
        encoded["input_ids"],
        encoded["attention_mask"],
        prompt_lengths,
    )
    _validate_supervised_labels(
        encoded["labels"],
        [item["example_id"] for item in batch],
        prompt_lengths,
    )
    encoded["prompt_lengths"] = prompt_lengths
    encoded["example_ids"] = [item["example_id"] for item in batch]
    encoded["items"] = list(batch)
    return encoded


def collate_llm_grid_probe_prompt_batch(
    batch: Sequence[LLMGridProbeItem],
    tokenizer: Any,
    system_prompt: str,
    max_input_length: int,
) -> Dict[str, Any]:
    prompt_texts = [
        _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        for item in batch
    ]
    encoded = tokenizer(
        prompt_texts,
        max_length=max_input_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded["prompt_lengths"] = [
        int(encoded["attention_mask"][row].sum().item())
        for row in range(len(batch))
    ]
    encoded["example_ids"] = [item["example_id"] for item in batch]
    encoded["items"] = list(batch)
    return encoded
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: PASS for downsampling, parser, metrics, dataset, and collate tests.

- [ ] **Step 7: Commit dataset and collators**

Run:

```bash
git add vlnce_baselines/models/etp_llm/train_llm_grid_probe.py tests/etp_llm/test_train_llm_grid_probe.py
git commit -m "feat(llm-grid): load probe raster targets"
```

Expected: commit succeeds with dataset and collator implementation.

---

### Task 4: Add Train/Eval CLI

**Files:**
- Modify: `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py`
- Modify: `tests/etp_llm/test_train_llm_grid_probe.py`

- [ ] **Step 1: Add CLI and boundary tests**

Append these tests to `tests/etp_llm/test_train_llm_grid_probe.py`:

```python
def test_llm_grid_probe_args_defaults_to_grid_probe_namespace_and_scale():
    args = train_llm_grid_probe.LLMGridProbeArgs().parse_args(["train"])

    assert args.mode == "train"
    assert args.cognitive_map_namespace == "gt.legacy.r1p5.direction5.v1"
    assert args.scale == 2
    assert args.max_new_tokens == 6144
    assert args.lora_r == 32
    assert args.lora_alpha == 64
    assert args.lora_dropout == 0.05


def test_train_model_rejects_full_finetuning():
    args = train_llm_grid_probe.LLMGridProbeArgs().parse_args(
        ["train", "--finetune-method", "full"]
    )

    with pytest.raises(NotImplementedError):
        train_llm_grid_probe.train_model(args)


def test_evaluate_model_writes_metrics_and_prediction_artifact(monkeypatch, tmp_path):
    raster_path = tmp_path / "grid.npz"
    full_grid = np.zeros((37, 100, 100), dtype=np.float32)
    full_grid[1, 0, 0] = 1.0
    np.savez_compressed(raster_path, grid=full_grid)
    example = train_llm_grid_probe.LLMGridProbeExample(
        example_id="R2R_val_seen_42",
        dataset_tag="R2R",
        split="val_seen",
        scene_id="scene-a",
        episode_id=42,
        instruction="Go to the chair.",
        start_position=[1.2, 0.0, 3.4],
        start_direction=(0.0, 1.0),
        raster_path=raster_path,
    )

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def generate(self, **kwargs):
            input_ids = kwargs["input_ids"]
            suffix = torch.tensor([[91, 97, 93]], dtype=torch.long)
            return torch.cat([input_ids, suffix], dim=1)

    class FakeTokenizer(_ChatTokenizer):
        def batch_decode(self, rows, skip_special_tokens=True):
            return ['{"grid":[[1,0,0]]}' for _row in rows]

    monkeypatch.setattr(
        train_llm_grid_probe,
        "load_llm_grid_probe_examples",
        lambda *args, **kwargs: [example],
    )
    monkeypatch.setattr(
        train_llm_grid_probe,
        "_load_causal_lm_model_and_tokenizer",
        lambda *args, **kwargs: (FakeModel(), FakeTokenizer()),
    )

    args = train_llm_grid_probe.LLMGridProbeArgs().parse_args(
        [
            "eval",
            "--output-dir",
            str(tmp_path / "run"),
            "--limit",
            "1",
            "--device",
            "cpu",
            "--device-map",
            "none",
        ]
    )
    metrics = train_llm_grid_probe.evaluate_model(args)

    assert metrics["json_valid"] == pytest.approx(1.0)
    assert metrics["cell_recall"] == pytest.approx(1.0)
    assert metrics["target_truncation_rate"] == pytest.approx(0.0)
    assert metrics["generated_token_count"] > 0.0
    metrics_path = tmp_path / "run" / "metrics.json"
    artifact_path = tmp_path / "run" / "artifacts" / "R2R_val_seen_42.json"
    assert metrics_path.exists()
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text())
    assert artifact["generated_text"] == '{"grid":[[1,0,0]]}'
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: FAIL because `LLMGridProbeArgs`, `train_model`, and `evaluate_model` are not defined.

- [ ] **Step 3: Add training/eval imports**

Extend imports in `train_llm_grid_probe.py`:

```python
from tap import Tap
from torch.utils.data import DataLoader

from model_paths import LLAMA_3_1_8B_INSTRUCT_MODEL

from .train_llm_boxes import (
    _cast_trainable_parameters_to_float32,
    _default_device,
    _generation_kwargs,
    _load_causal_lm_model_and_tokenizer,
    _model_batch,
    _model_uses_device_map,
    _normalize_device_map,
    _progress,
    _validate_trainable_parameters_finite,
    decode_generated_completion,
)
```

Add default model constant:

```python
DEFAULT_MODEL_NAME_OR_PATH = LLAMA_3_1_8B_INSTRUCT_MODEL
```

- [ ] **Step 4: Add run helpers and argument parser**

Add this code near the bottom of `train_llm_grid_probe.py`:

```python
def load_system_prompt(path: Path = DEFAULT_SYSTEM_PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8").strip()


def _write_run_system_prompt(output_dir: str | Path, system_prompt: str) -> None:
    artifact_dir = Path(output_dir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "system_prompt.md").write_text(system_prompt + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _aggregate_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(sum(row.get(key, 0.0) for row in rows) / len(rows))
        for key in keys
    }


class LLMGridProbeArgs(Tap):
    mode: Literal["train", "eval"]
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH
    checkpoint_path: Optional[str] = None
    output_dir: str = "outputs/llm_grid_probe"
    dataset: Literal["R2R", "RxR"] = "R2R"
    cognitive_map_namespace: str = DEFAULT_GRID_NAMESPACE
    scale: int = GRID_SCALE
    max_input_length: int = 1024
    max_new_tokens: int = 6144
    finetune_method: Literal["lora", "full"] = "lora"
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    limit: Optional[int] = None
    device: str = ""
    device_map: Literal["auto", "balanced", "balanced_low_0", "sequential", "none"] = "auto"
    quiet: bool = False

    def configure(self) -> None:
        self.add_argument("mode")

    def process_args(self) -> None:
        if not self.device:
            self.device = _default_device()
        if self.scale != GRID_SCALE:
            raise ValueError("LLM-Grid-Probe v1 only supports --scale 2")
```

- [ ] **Step 5: Add train and eval functions**

Add this code below `LLMGridProbeArgs`:

```python
def _apply_grid_probe_lora(model: Any, args: LLMGridProbeArgs) -> Any:
    if args.finetune_method != "lora":
        raise ValueError(f"Unsupported finetune method: {args.finetune_method}")
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(args.lora_target_modules),
    )
    return get_peft_model(model, config)


def _text_diagnostics(
    tokenizer: Any,
    target_text: str,
    generated_text: str,
    max_new_tokens: int,
) -> Dict[str, float]:
    target_tokens = _token_count(tokenizer, target_text)
    generated_tokens = _token_count(tokenizer, generated_text)
    return {
        "target_token_count": float(target_tokens),
        "target_truncation_rate": 1.0 if target_tokens > max_new_tokens else 0.0,
        "generated_token_count": float(generated_tokens),
        "generated_char_count": float(len(generated_text)),
    }


def train_model(args: LLMGridProbeArgs) -> Dict[str, float]:
    if args.finetune_method == "full":
        raise NotImplementedError("full fine-tuning is not implemented for LLM-Grid-Probe")
    examples = load_llm_grid_probe_examples(
        args.dataset,
        TRAIN_SPLITS,
        limit=args.limit,
        quiet=args.quiet,
        skip_missing_cache=True,
        cognitive_map_namespace=args.cognitive_map_namespace,
    )
    if not examples:
        raise ValueError("No LLM-Grid-Probe training examples were loaded")

    system_prompt = load_system_prompt()
    _write_run_system_prompt(args.output_dir, system_prompt)
    model, tokenizer = _load_causal_lm_model_and_tokenizer(
        args.model_name_or_path,
        device_map=_normalize_device_map(args.device_map),
    )
    model = _apply_grid_probe_lora(model, args)
    _cast_trainable_parameters_to_float32(model)
    device = torch.device(args.device)
    if not _model_uses_device_map(model):
        model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    dataset = LLMGridProbeDataset(examples, scale=args.scale)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_llm_grid_probe_batch(
            batch,
            tokenizer,
            system_prompt,
            args.max_input_length,
            args.max_new_tokens,
        ),
    )

    losses: List[float] = []
    model.train()
    for epoch_index in range(args.epochs):
        for batch in _progress(loader, desc="train LLM-Grid-Probe", quiet=args.quiet):
            optimizer.zero_grad(set_to_none=True)
            model_batch = _model_batch(batch, device)
            outputs = model(**model_batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            _validate_trainable_parameters_finite(
                model,
                f"llm-grid-probe training epoch={epoch_index + 1}",
            )
            losses.append(float(loss.detach().cpu()))
        epoch_dir = Path(args.output_dir) / "checkpoints" / f"epoch-{epoch_index + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

    checkpoint_dir = Path(args.output_dir) / "checkpoints" / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    metrics = {"train_loss": float(sum(losses) / len(losses))}
    _write_json(Path(args.output_dir) / "metrics.json", metrics)
    return metrics


def evaluate_model(args: LLMGridProbeArgs) -> Dict[str, float]:
    examples = load_llm_grid_probe_examples(
        args.dataset,
        EVAL_SPLITS,
        limit=args.limit,
        quiet=args.quiet,
        skip_missing_cache=True,
        cognitive_map_namespace=args.cognitive_map_namespace,
    )
    if not examples:
        raise ValueError("No LLM-Grid-Probe eval examples were loaded")

    system_prompt = load_system_prompt()
    _write_run_system_prompt(args.output_dir, system_prompt)
    model_path = args.checkpoint_path or args.model_name_or_path
    model, tokenizer = _load_causal_lm_model_and_tokenizer(
        model_path,
        device_map=_normalize_device_map(args.device_map),
    )
    device = torch.device(args.device)
    if not _model_uses_device_map(model):
        model.to(device)
    model.eval()

    dataset = LLMGridProbeDataset(examples, scale=args.scale)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_llm_grid_probe_prompt_batch(
            batch,
            tokenizer,
            system_prompt,
            args.max_input_length,
        ),
    )
    rows: List[Dict[str, float]] = []
    artifact_dir = Path(args.output_dir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in _progress(loader, desc="eval LLM-Grid-Probe", quiet=args.quiet):
            model_batch = _model_batch(batch, device, include_labels=False)
            generated = model.generate(
                **model_batch,
                **_generation_kwargs(tokenizer, args.max_new_tokens),
            )
            completions = [
                decode_generated_completion(tokenizer, sequence, prompt_length)
                for sequence, prompt_length in zip(generated, batch["prompt_lengths"])
            ]
            for item, generated_text in zip(batch["items"], completions):
                metrics = evaluate_grid_probe_prediction(
                    generated_text,
                    item["target_grid"],
                )
                metrics.update(
                    _text_diagnostics(
                        tokenizer,
                        item["target_text"],
                        generated_text,
                        args.max_new_tokens,
                    )
                )
                rows.append(metrics)
                _write_json(
                    artifact_dir / f"{item['example_id']}.json",
                    {
                        "example_id": item["example_id"],
                        "input_text": item["input_text"],
                        "target_text": item["target_text"],
                        "generated_text": generated_text,
                        "metrics": metrics,
                    },
                )
    metrics = _aggregate_metrics(rows)
    metrics["example_count"] = float(len(rows))
    _write_json(Path(args.output_dir) / "metrics.json", metrics)
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    args = LLMGridProbeArgs(underscores_to_dashes=True).parse_args(argv)
    if args.mode == "train":
        return train_model(args)
    if args.mode == "eval":
        return evaluate_model(args)
    raise ValueError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: PASS.

- [ ] **Step 7: Run quality checks for touched Python**

Run:

```bash
ruff check prior/llm_grid_samples.py vlnce_baselines/models/etp_llm/train_llm_grid_probe.py tests/etp_llm/test_train_llm_grid_probe.py
ty check prior/llm_grid_samples.py vlnce_baselines/models/etp_llm/train_llm_grid_probe.py tests/etp_llm/test_train_llm_grid_probe.py
```

Expected: both commands exit 0. If `ty` reports an issue from importing private LLM-Boxes helpers, fix the annotation locally instead of suppressing the diagnostic.

- [ ] **Step 8: Commit CLI**

Run:

```bash
git add vlnce_baselines/models/etp_llm/train_llm_grid_probe.py tests/etp_llm/test_train_llm_grid_probe.py
git commit -m "feat(llm-grid): add probe train eval CLI"
```

Expected: commit succeeds with the CLI implementation and tests.

---

### Task 5: Add Submit Script and Final Verification

**Files:**
- Create: `scripts/submit/llm-grid-probe-train.sh`
- Verify: `prior/llm_grid_samples.py`
- Verify: `vlnce_baselines/models/etp_llm/train_llm_grid_probe.py`
- Verify: `tests/etp_llm/test_train_llm_grid_probe.py`

- [ ] **Step 1: Add the submit script**

Create `scripts/submit/llm-grid-probe-train.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=llm-grid-probe
#SBATCH --gpus=6
#SBATCH -p vip_gpu_scze096
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate etpr1-uv

python -m vlnce_baselines.models.etp_llm.train_llm_grid_probe train \
  --batch-size 2 \
  --max-new-tokens 6144 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --cognitive-map-namespace gt.legacy.r1p5.direction5.v1 \
  --output-dir outputs/llm_grid_probe/r2r-legacy-r1p5-direction5-scale2
```

- [ ] **Step 2: Make the script executable**

Run:

```bash
chmod +x scripts/submit/llm-grid-probe-train.sh
```

Expected: command exits 0.

- [ ] **Step 3: Validate shell syntax**

Run:

```bash
bash -n scripts/submit/llm-grid-probe-train.sh
```

Expected: command exits 0 and prints nothing.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/etp_llm/test_train_llm_grid_probe.py -q
```

Expected: PASS.

- [ ] **Step 5: Run LLM package tests**

Run:

```bash
pytest tests/etp_llm -q
```

Expected: PASS. If unrelated tests fail, capture the failing test names and inspect whether the new module changed shared LLM-Boxes behavior.

- [ ] **Step 6: Run lint and type checks**

Run:

```bash
ruff check prior/llm_grid_samples.py vlnce_baselines/models/etp_llm tests/etp_llm scripts/submit/llm-grid-probe-train.sh
ty check prior/llm_grid_samples.py vlnce_baselines/models/etp_llm tests/etp_llm
```

Expected: both commands exit 0.

- [ ] **Step 7: Commit submit script and verification fixes**

Run:

```bash
git add scripts/submit/llm-grid-probe-train.sh prior/llm_grid_samples.py vlnce_baselines/models/etp_llm tests/etp_llm
git commit -m "chore(llm-grid): add probe training submit script"
```

Expected: commit succeeds. If no Python files changed after Task 4, the commit includes only `scripts/submit/llm-grid-probe-train.sh`.

---

## Final Checks

- [ ] Run the full unit suite:

```bash
pytest -q
```

Expected: PASS.

- [ ] Review the final diff:

```bash
git diff --stat origin/main..HEAD
git log --oneline origin/main..HEAD
```

Expected: commits are scoped to LLM-Grid-Probe, prompt, tests, downsample helper, and submit script.

- [ ] Confirm the v1 boundaries:

```bash
rg "LLM-Grid-Probe|gt.legacy.r1p5.direction5.v1|6144|lora_r|lora-r" \
  docs/superpowers/specs/2026-07-08-llm-grid-probe-design.md \
  vlnce_baselines/models/etp_llm/train_llm_grid_probe.py \
  scripts/submit/llm-grid-probe-train.sh
```

Expected: the implementation defaults match the spec: predictor-only, namespace `gt.legacy.r1p5.direction5.v1`, `max_new_tokens=6144`, LoRA rank 32.
