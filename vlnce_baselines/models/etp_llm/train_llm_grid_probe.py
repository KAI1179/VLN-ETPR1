"""Dataset, training, and evaluation CLI for the LLM-Grid-Probe milestone."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from prior.llm_grid_samples import downsample_grid, serialize_grid_target
from prior.vlnce import VLNCEEpisodeEntry
from vlnce_baselines.models.etp_prior_gt.map_utils import cognitive_map_cache_path

from .boxes_schema import build_llm_boxes_input
from .train_llm_boxes import (
    _causal_lm_labels,
    _encoded_width,
    _render_chat_completion,
    _render_chat_prompt,
    _token_count,
    _validate_supervised_labels,
)

GRID_CHANNELS = 37
GRID_SCALE = 2
GRID_SHAPE = (GRID_CHANNELS, 50, 50)
DEFAULT_GRID_NAMESPACE = "gt.legacy.r1p5.direction5.v1"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "llm_grid_probe_system.md"
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


class LLMGridProbeValidationError(ValueError):
    """Raised when generated LLM-Grid-Probe JSON fails validation."""


class _LLMGridProbeJSONError(LLMGridProbeValidationError):
    """Raised when generated text is not valid JSON."""


@dataclass(frozen=True)
class ParsedGridProbe:
    grid: NDArray[np.float32]
    record_count: int
    duplicate_record_count: int


@dataclass(frozen=True)
class LLMGridProbeExample:
    example_id: str
    dataset_tag: str
    split: str
    scene_id: str
    episode_id: int
    instruction: str
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


def _load_raster_target(
    path: Path,
) -> Tuple[
    NDArray[np.float32],
    Tuple[float, float],
    Tuple[float, float],
]:
    with np.load(path, allow_pickle=True) as data:
        start_position = data["start_position"]
        start_direction = data["start_direction_vector"]
        return (
            np.asarray(data["grid"], dtype=np.float32),
            (float(start_position[0]), float(start_position[1])),
            (float(start_direction[0]), float(start_direction[1])),
        )


class LLMGridProbeDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[LLMGridProbeExample],
        scale: int = GRID_SCALE,
    ) -> None:
        self.examples = list(examples)
        self.scale = scale

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LLMGridProbeItem:
        example = self.examples[index]
        full_grid, start_position, start_direction = _load_raster_target(
            example.raster_path
        )
        target_grid = downsample_grid(full_grid, self.scale)
        return {
            "input_text": build_llm_boxes_input(
                example.dataset_tag,
                example.instruction,
                start_position,
                start_direction,
            ),
            "target_text": serialize_grid_target(full_grid, scale=self.scale),
            "target_grid": target_grid,
            "example_id": example.example_id,
            "instruction": example.instruction,
            "start_position": start_position,
            "start_direction": start_direction,
            "scene_id": example.scene_id,
        }

    def __iter__(self) -> Iterator[LLMGridProbeItem]:
        for index in range(len(self)):
            yield self[index]


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
    prompt_width = _encoded_width(encoded["input_ids"])
    encoded["prompt_lengths"] = [prompt_width] * len(batch)
    encoded["example_ids"] = [item["example_id"] for item in batch]
    encoded["items"] = list(batch)
    return encoded


def parse_grid_probe_text(
    text: str,
    shape: Tuple[int, int, int] = GRID_SHAPE,
) -> ParsedGridProbe:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise _LLMGridProbeJSONError(f"invalid JSON: {error}") from error
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
        if not all(type(item) is int for item in (category, row, col)):
            raise LLMGridProbeValidationError(f"grid[{index}] category,row,col must be ints")
        if not (0 <= category < channels and 0 <= row < rows and 0 <= col < cols):
            raise LLMGridProbeValidationError(f"grid[{index}] index out of bounds")
        value = 1.0
        if len(raw_record) == 4:
            raw_value = raw_record[3]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise LLMGridProbeValidationError(f"grid[{index}] value must be numeric")
            value = float(raw_value)
        if not (0.0 <= value <= 1.0):
            raise LLMGridProbeValidationError(f"grid[{index}] value out of range")
        key = (category, row, col)
        if key in seen:
            duplicates += 1
        seen.add(key)
        grid[category, row, col] = max(float(grid[category, row, col]), value)
    return ParsedGridProbe(
        grid=grid,
        record_count=len(records),
        duplicate_record_count=duplicates,
    )


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
        shape = cast(Tuple[int, int, int], target_grid.shape)
        parsed = parse_grid_probe_text(generated_text, shape=shape)
    except LLMGridProbeValidationError as error:
        return {
            "json_valid": float(not isinstance(error, _LLMGridProbeJSONError)),
            "schema_valid": 0.0,
            "record_count": 0.0,
            "duplicate_record_count": 0.0,
            "duplicate_record_rate": 0.0,
            "cell_precision": 0.0,
            "cell_recall": 0.0,
            "cell_f1": 0.0,
            "category_aware_raster_iou": 0.0,
            "category_aware_raster_recall": 0.0,
            "category_aware_raster_support": float(
                np.count_nonzero(target_grid > 0)
            ),
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
