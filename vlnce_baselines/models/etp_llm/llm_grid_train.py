"""Dataset, training, and evaluation CLI for the LLM-Grid milestone."""

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
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from tap import Tap
from torch.utils.data import DataLoader, Dataset
import torch
from typing_extensions import NotRequired

from model_paths import LLAMA_3_1_8B_INSTRUCT_MODEL
from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES, OBJECT_CATEGORIES
from prior.llm_grid_samples import downsample_grid, serialize_grid_target
from prior.vlnce import VLNCEEpisodeEntry
from vlnce_baselines.models.etp_prior_gt.map_utils import cognitive_map_cache_path

from .boxes_schema import build_llm_map_input
from .sft import (
    ExampleLoadResult,
    LengthFilterResult,
    LengthGroupedBatchSampler,
    SourceLoadStats,
    TrainingIndex,
    enable_gradient_checkpointing as _enable_gradient_checkpointing,
    fixed_corpus_metrics,
    rendered_token_counts,
)
from .llm_boxes_train import (
    _causal_lm_labels,
    _cast_trainable_parameters_to_float32,
    _default_device,
    _encoded_width,
    _generation_kwargs,
    _load_causal_lm_model_and_tokenizer,
    _model_batch,
    _model_uses_device_map,
    _non_finite_step_message,
    _normalize_device_map,
    _progress,
    _render_chat_completion,
    _render_chat_prompt,
    _token_count,
    _validate_trainable_parameters_finite,
    _validate_supervised_labels,
    decode_generated_completion,
)

GRID_CHANNELS = 37
GRID_SCALE = 2
GRID_SHAPE = (GRID_CHANNELS, 50, 50)
GRID_SIZE_BY_SCALE = {1: 100, 2: 50}
DEFAULT_MODEL_NAME_OR_PATH = LLAMA_3_1_8B_INSTRUCT_MODEL
DEFAULT_GRID_NAMESPACE = "gt.legacy.r1p5.direction5.v1"
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "llm_grid_system.md"
VECTOR_NORM_TOLERANCE = 1e-3
GRID_TARGET_KEYS = (
    "predicted_regions",
    "predicted_objects",
    "regions",
    "objects",
    "direction_vectors",
)
TRAIN_SPLITS = ("train",)
EVAL_SPLITS = ("val_seen", "val_unseen")
OBJECT_CATEGORY_TO_ID = {
    category: index for index, category in enumerate(MAPPED_OBJECT_NAMES)
}
REGION_CATEGORY_TO_ID = {
    category: index for index, category in enumerate(MAPPED_REGION_NAMES)
}


class LLMGridItem(TypedDict):
    input_text: str
    target_text: str
    target_grid: NDArray[np.float32]
    target_direction_vectors: NDArray[np.float32]
    example_id: str
    instruction: str
    start_position: Sequence[float]
    start_direction: Sequence[float]
    scene_id: str
    dataset: Literal["R2R", "RxR"]
    training_weight: NotRequired[float]
    is_padding: NotRequired[bool]


class LLMGridValidationError(ValueError):
    """Raised when generated LLM-Grid JSON fails validation."""


class _LLMGridJSONError(LLMGridValidationError):
    """Raised when generated text is not valid JSON."""


@dataclass(frozen=True)
class ParsedGrid:
    grid: NDArray[np.float32]
    direction_vectors: NDArray[np.float32]
    record_count: int
    duplicate_record_count: int


@dataclass(frozen=True)
class LLMGridExample:
    example_id: str
    dataset: Literal["R2R", "RxR"]
    split: str
    scene_id: str
    episode_id: int
    instruction: str
    raster_path: Path


def load_llm_grid_examples(
    splits: Iterable[str],
    limit_per_dataset: Optional[int] = None,
    quiet: bool = False,
    skip_missing_cache: bool = False,
    cognitive_map_namespace: str = DEFAULT_GRID_NAMESPACE,
) -> ExampleLoadResult[LLMGridExample]:
    examples: List[LLMGridExample] = []
    skipped_missing_cache: List[Tuple[str, str]] = []
    discovered = {"R2R": 0, "RxR": 0}
    loaded = {"R2R": 0, "RxR": 0}
    missing: Dict[str, List[str]] = {"R2R": [], "RxR": []}
    for episode in VLNCEEpisodeEntry.iter_r2r_rxr(
        splits=splits,
        limit_per_dataset=limit_per_dataset,
    ):
        episode_dataset = _episode_dataset(episode.dataset)
        discovered[episode_dataset] += 1
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
            missing[episode_dataset].append(episode.unique_id)
            continue
        examples.append(
            LLMGridExample(
                example_id=episode.unique_id,
                dataset=episode_dataset,
                split=episode.split,
                scene_id=episode.scene_id,
                episode_id=episode.episode_id,
                instruction=episode.instruction,
                raster_path=raster_path,
            )
        )
        loaded[episode_dataset] += 1
    if skipped_missing_cache and not quiet:
        print(f"skipped_missing_cache={len(skipped_missing_cache)}")
        for example_id, path in skipped_missing_cache:
            print(f"  {example_id}: {path}")
    return ExampleLoadResult(
        examples=tuple(examples),
        by_dataset={
            dataset: SourceLoadStats(
                discovered=discovered[dataset],
                loaded=loaded[dataset],
                missing_cache_example_ids=tuple(missing[dataset]),
            )
            for dataset in ("R2R", "RxR")
        },
    )


def _episode_dataset(dataset: str) -> Literal["R2R", "RxR"]:
    if dataset == "R2R":
        return "R2R"
    if dataset == "RxR":
        return "RxR"
    raise ValueError(f"unsupported LLM-Grid dataset: {dataset}")


def _load_raster_target(
    path: Path,
) -> Tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    Tuple[float, float],
    Tuple[float, float],
]:
    with np.load(path, allow_pickle=True) as data:
        direction_vectors = np.asarray(data["direction_vectors"], dtype=np.float32)
        if direction_vectors.shape != (5, 2):
            raise ValueError(
                "direction_vectors must have shape "
                f"(5, 2), got {direction_vectors.shape}"
            )
        start_position = data["start_position"]
        start_direction = data["start_direction_vector"]
        return (
            np.asarray(data["grid"], dtype=np.float32),
            direction_vectors,
            (float(start_position[0]), float(start_position[1])),
            (float(start_direction[0]), float(start_direction[1])),
        )


def _boxes_path_for_raster(raster_path: Path) -> Path:
    parts = list(raster_path.parts)
    try:
        raster_index = parts.index("raster")
    except ValueError as error:
        raise ValueError(
            f"Raster path does not contain a raster directory: {raster_path}"
        ) from error
    parts[raster_index] = "boxes"
    return Path(*parts)


def _load_grid_mentions(raster_path: Path) -> Tuple[set[int], set[int]]:
    boxes_path = _boxes_path_for_raster(raster_path)
    with np.load(boxes_path, allow_pickle=True) as data:
        payload = json.loads(str(data["payload"]))
    objects = {
        category_id
        for category_id, boxes in enumerate(payload["level"]["objects"])
        if any(bool(box.get("mentioned", False)) for box in boxes)
    }
    regions = {
        category_id
        for category_id, boxes in enumerate(payload["level"]["regions"])
        if any(bool(box.get("mentioned", False)) for box in boxes)
    }
    return objects, regions


class LLMGridDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[LLMGridExample],
        scale: int = GRID_SCALE,
    ) -> None:
        self.examples = list(examples)
        self.scale = scale

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> LLMGridItem:
        example = self.examples[index]
        (
            full_grid,
            direction_vectors,
            start_position,
            start_direction,
        ) = _load_raster_target(
            example.raster_path,
        )
        mentioned_objects, mentioned_regions = _load_grid_mentions(example.raster_path)
        target_grid = downsample_grid(full_grid, self.scale)
        return {
            "input_text": build_llm_map_input(
                example.instruction,
                start_position,
                start_direction,
            ),
            "target_text": serialize_grid_target(
                full_grid,
                direction_vectors=direction_vectors,
                scale=self.scale,
                mentioned_objects=mentioned_objects,
                mentioned_regions=mentioned_regions,
            ),
            "target_grid": target_grid,
            "target_direction_vectors": direction_vectors,
            "example_id": example.example_id,
            "instruction": example.instruction,
            "start_position": start_position,
            "start_direction": start_direction,
            "scene_id": example.scene_id,
            "dataset": example.dataset,
        }

    def __iter__(self) -> Iterator[LLMGridItem]:
        for index in range(len(self)):
            yield self[index]


class LLMGridItemsDataset(Dataset):
    def __init__(self, items: Sequence[LLMGridItem]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: Union[int, TrainingIndex]) -> LLMGridItem:
        if isinstance(index, TrainingIndex):
            item = self.items[index.index].copy()
            item["training_weight"] = index.loss_scale
            item["is_padding"] = index.is_padding
            return item
        return self.items[index]


def _completion_token_count(
    item: LLMGridItem,
    tokenizer: Any,
    system_prompt: str,
    prompt_length: int,
) -> int:
    prompt = _render_chat_prompt(
        tokenizer,
        system_prompt,
        item["input_text"],
    )
    completion = _render_chat_completion(
        tokenizer,
        system_prompt,
        item["input_text"],
        item["target_text"],
    )
    counts = rendered_token_counts(tokenizer, prompt, completion)
    if counts.prompt_tokens != prompt_length:
        raise ValueError("prompt token count changed between renderings")
    return counts.completion_tokens


def filter_grid_training_items(
    items: Sequence[LLMGridItem],
    tokenizer: Any,
    system_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
) -> LengthFilterResult[LLMGridItem]:
    filtered: List[LLMGridItem] = []
    dropped_prompt: List[str] = []
    dropped_completion: List[str] = []
    for item in items:
        prompt = _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        completion = _render_chat_completion(
            tokenizer,
            system_prompt,
            item["input_text"],
            item["target_text"],
        )
        counts = rendered_token_counts(tokenizer, prompt, completion)
        prompt_over_budget = counts.prompt_tokens > max_input_length
        completion_over_budget = counts.completion_tokens > max_new_tokens
        if prompt_over_budget:
            dropped_prompt.append(item["example_id"])
        if completion_over_budget:
            dropped_completion.append(item["example_id"])
        if not prompt_over_budget and not completion_over_budget:
            filtered.append(item)
    return LengthFilterResult(
        kept=tuple(filtered),
        dropped_prompt_example_ids=tuple(dropped_prompt),
        dropped_completion_example_ids=tuple(dropped_completion),
    )


def _training_sequence_lengths(
    items: Sequence[LLMGridItem],
    tokenizer: Any,
    system_prompt: str,
) -> List[int]:
    return [
        rendered_token_counts(
            tokenizer,
            _render_chat_prompt(tokenizer, system_prompt, item["input_text"]),
            _render_chat_completion(
                tokenizer, system_prompt, item["input_text"], item["target_text"]
            ),
        ).sequence_tokens
        for item in items
    ]


def collate_llm_grid_batch(
    batch: Sequence[LLMGridItem],
    tokenizer: Any,
    system_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    training_weights = torch.tensor(
        [item["training_weight"] for item in batch],
        dtype=torch.float32,
    )
    is_padding = torch.tensor(
        [item["is_padding"] for item in batch],
        dtype=torch.bool,
    )
    prompt_texts = [
        _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        for item in batch
    ]
    prompt_lengths = [_token_count(tokenizer, text) for text in prompt_texts]
    oversized_prompts = [
        f"{item['example_id']} ({length} tokens)"
        for item, length in zip(batch, prompt_lengths)
        if length > max_input_length
    ]
    if oversized_prompts:
        raise ValueError(
            f"Prompt for {', '.join(oversized_prompts)} exceeds "
            f"max_input_length={max_input_length}"
        )
    oversized_targets = [
        f"{item['example_id']} ({completion_tokens} tokens)"
        for item, prompt_length in zip(batch, prompt_lengths)
        for completion_tokens in [
            _completion_token_count(
                item,
                tokenizer,
                system_prompt,
                prompt_length,
            )
        ]
        if completion_tokens > max_new_tokens
    ]
    if oversized_targets:
        raise ValueError(
            f"Target for {', '.join(oversized_targets)} exceeds "
            f"max_new_tokens={max_new_tokens}"
        )
    full_texts = [
        _render_chat_completion(
            tokenizer,
            system_prompt,
            item["input_text"],
            item["target_text"],
        )
        for item in batch
    ]
    encoded = tokenizer(
        full_texts,
        add_special_tokens=False,
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
    encoded["training_weights"] = training_weights
    encoded["is_padding"] = is_padding
    return encoded


def collate_llm_grid_prompt_batch(
    batch: Sequence[LLMGridItem],
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
        add_special_tokens=False,
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


def parse_grid_text(
    text: str,
    shape: Tuple[int, int, int] = GRID_SHAPE,
) -> ParsedGrid:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise _LLMGridJSONError(f"invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise LLMGridValidationError("top-level JSON value must be an object")
    if tuple(payload) != GRID_TARGET_KEYS:
        raise LLMGridValidationError(
            "top-level JSON object keys must be ordered as predicted_regions, "
            "predicted_objects, regions, objects, direction_vectors"
        )
    predicted_objects = _parse_candidate_list(
        payload["predicted_objects"],
        "predicted_objects",
        OBJECT_CATEGORY_TO_ID,
    )
    predicted_regions = _parse_candidate_list(
        payload["predicted_regions"],
        "predicted_regions",
        REGION_CATEGORY_TO_ID,
    )
    direction_vectors = _parse_direction_vectors(payload["direction_vectors"])
    object_section = payload["objects"]
    region_section = payload["regions"]
    if not isinstance(object_section, dict):
        raise LLMGridValidationError("objects must be an object")
    if not isinstance(region_section, dict):
        raise LLMGridValidationError("regions must be an object")
    if list(object_section) != predicted_objects:
        raise LLMGridValidationError("predicted_objects must match objects keys")
    if list(region_section) != predicted_regions:
        raise LLMGridValidationError("predicted_regions must match regions keys")

    channels, rows, cols = shape
    grid = np.zeros(shape, dtype=np.float32)
    seen: set[tuple[int, int, int]] = set()
    duplicates = 0
    record_count = 0
    for name in predicted_objects:
        category = OBJECT_CATEGORY_TO_ID[name]
        count, duplicate_count = _read_entity_cells(
            object_section[name],
            f"objects.{name}",
            category,
            rows,
            cols,
            grid,
            seen,
        )
        record_count += count
        duplicates += duplicate_count
    for name in predicted_regions:
        category = OBJECT_CATEGORIES + REGION_CATEGORY_TO_ID[name]
        if category >= channels:
            raise LLMGridValidationError(f"regions.{name} index out of bounds")
        count, duplicate_count = _read_entity_cells(
            region_section[name],
            f"regions.{name}",
            category,
            rows,
            cols,
            grid,
            seen,
        )
        record_count += count
        duplicates += duplicate_count
    return ParsedGrid(
        grid=grid,
        direction_vectors=direction_vectors,
        record_count=record_count,
        duplicate_record_count=duplicates,
    )


def _parse_direction_vectors(value: Any) -> NDArray[np.float32]:
    if not isinstance(value, list) or len(value) != 5:
        raise LLMGridValidationError("direction_vectors must be five [dx,dz] vectors")
    rows: List[List[float]] = []
    for index, raw_vector in enumerate(value):
        if not isinstance(raw_vector, list) or len(raw_vector) != 2:
            raise LLMGridValidationError(f"direction_vectors[{index}] must be [dx,dz]")
        row: List[float] = []
        for component_index, component in enumerate(raw_vector):
            if type(component) is int or type(component) is float:
                try:
                    with np.errstate(over="ignore", invalid="ignore"):
                        numeric_component = np.float32(component)
                except (OverflowError, ValueError) as error:
                    raise LLMGridValidationError(
                        f"direction_vectors[{index}][{component_index}] must be finite"
                    ) from error
            else:
                raise LLMGridValidationError(
                    f"direction_vectors[{index}][{component_index}] must be numeric"
                )
            if not np.isfinite(numeric_component):
                raise LLMGridValidationError(
                    f"direction_vectors[{index}][{component_index}] must be finite"
                )
            row.append(float(numeric_component))
        norm = float(np.linalg.norm(row))
        if norm > VECTOR_NORM_TOLERANCE and abs(norm - 1.0) > VECTOR_NORM_TOLERANCE:
            raise LLMGridValidationError(
                f"direction_vectors[{index}] must be unit length or zero padding"
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _parse_candidate_list(
    raw_candidates: Any,
    name: str,
    category_to_id: Dict[str, int],
) -> List[str]:
    if not isinstance(raw_candidates, list):
        raise LLMGridValidationError(f"{name} must be a list")
    candidates: List[str] = []
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, str):
            raise LLMGridValidationError(f"{name}[{index}] must be a string")
        if item not in category_to_id:
            raise LLMGridValidationError(f"{name}[{index}] is unknown: {item}")
        if item in candidates:
            raise LLMGridValidationError(f"{name}[{index}] is duplicated: {item}")
        candidates.append(item)
    return candidates


def _read_entity_cells(
    raw_entity: Any,
    name: str,
    category: int,
    rows: int,
    cols: int,
    grid: NDArray[np.float32],
    seen: set[tuple[int, int, int]],
) -> Tuple[int, int]:
    if not isinstance(raw_entity, dict):
        raise LLMGridValidationError(f"{name} must be an object")
    if set(raw_entity) != {"cells", "mentioned"}:
        raise LLMGridValidationError(f"{name} must contain only cells and mentioned")
    if type(raw_entity["mentioned"]) is not bool:
        raise LLMGridValidationError(f"{name}.mentioned must be boolean")
    cells = raw_entity["cells"]
    if not isinstance(cells, list):
        raise LLMGridValidationError(f"{name}.cells must be a list")
    duplicates = 0
    for index, raw_cell in enumerate(cells):
        if not isinstance(raw_cell, list) or len(raw_cell) != 2:
            raise LLMGridValidationError(f"{name}.cells[{index}] must be [row,col]")
        raw_row, raw_col = raw_cell
        if type(raw_row) is not int or type(raw_col) is not int:
            raise LLMGridValidationError(f"{name}.cells[{index}] row,col must be ints")
        row = raw_row
        col = raw_col
        if not (0 <= row < rows and 0 <= col < cols):
            raise LLMGridValidationError(f"{name}.cells[{index}] index out of bounds")
        key = (category, row, col)
        if key in seen:
            duplicates += 1
        seen.add(key)
        grid[category, row, col] = 1.0
    return len(cells), duplicates


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


def _zero_direction_vector_metrics() -> Dict[str, float]:
    return {
        "direction_vector_valid_rate": 0.0,
        "direction_vector_l2": 0.0,
        "direction_vector_l2_support": 0.0,
        "direction_vector_cosine": 0.0,
        "direction_vector_cosine_support": 0.0,
        "direction_vector_padding_accuracy": 0.0,
    }


def _direction_vector_metrics(
    pred_vectors: NDArray[np.float32],
    target_vectors: NDArray[np.float32],
) -> Dict[str, float]:
    pred_norms = np.linalg.norm(pred_vectors, axis=1)
    target_norms = np.linalg.norm(target_vectors, axis=1)
    pred_nonzero = pred_norms > 0.0
    target_nonzero = target_norms > 0.0
    non_padding_count = int(np.count_nonzero(target_nonzero))
    if non_padding_count == 0:
        cosine = 0.0
    else:
        dot_products = np.sum(
            pred_vectors[target_nonzero] * target_vectors[target_nonzero],
            axis=1,
        )
        denominators = pred_norms[target_nonzero] * target_norms[target_nonzero]
        row_cosines = np.divide(
            dot_products,
            denominators,
            out=np.zeros_like(dot_products, dtype=np.float32),
            where=denominators > 0.0,
        )
        cosine = float(np.mean(row_cosines))
    return {
        "direction_vector_valid_rate": 1.0,
        "direction_vector_l2": float(
            np.mean(np.linalg.norm(pred_vectors - target_vectors, axis=1))
        ),
        "direction_vector_l2_support": 1.0,
        "direction_vector_cosine": cosine,
        "direction_vector_cosine_support": float(non_padding_count > 0),
        "direction_vector_padding_accuracy": float(
            np.mean(pred_nonzero == target_nonzero)
        ),
    }


def evaluate_grid_prediction(
    generated_text: str,
    target_grid: NDArray[np.float32],
    target_direction_vectors: NDArray[np.float32],
) -> Dict[str, float]:
    try:
        shape = cast(Tuple[int, int, int], target_grid.shape)
        parsed = parse_grid_text(generated_text, shape=shape)
    except LLMGridValidationError as error:
        return {
            "json_valid": float(not isinstance(error, _LLMGridJSONError)),
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
            **_zero_direction_vector_metrics(),
        }
    metrics = _grid_metrics(parsed.grid, target_grid)
    metrics.update(
        _direction_vector_metrics(
            parsed.direction_vectors,
            target_direction_vectors,
        )
    )
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


class LLMGridArgs(Tap):
    mode: Literal["train", "eval"]
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH
    checkpoint_path: Optional[str] = None
    output_dir: str = "outputs/llm_grid"
    cognitive_map_namespace: str = DEFAULT_GRID_NAMESPACE
    scale: int = GRID_SCALE
    max_input_length: int = 1152
    max_new_tokens: int = 4096
    finetune_method: Literal["lora", "full"] = "lora"
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    epochs: int = 10
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
    limit_per_dataset: Optional[int] = None
    seed: int = 42
    device: str = ""
    device_map: Literal[
        "auto",
        "balanced",
        "balanced_low_0",
        "sequential",
        "none",
    ] = "auto"
    quiet: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("underscores_to_dashes", True)
        super().__init__(*args, **kwargs)

    def configure(self) -> None:
        self.add_argument("mode")

    def process_args(self) -> None:
        if not self.device:
            self.device = _default_device()
        if self.scale not in GRID_SIZE_BY_SCALE:
            raise ValueError("LLM-Grid v1 only supports --scale 1 or 2")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("--gradient-accumulation-steps must be >= 1")


def _grid_size_for_scale(scale: int) -> int:
    try:
        return GRID_SIZE_BY_SCALE[scale]
    except KeyError as error:
        raise ValueError(f"scale must be 1 or 2, got {scale}") from error


def load_system_prompt(
    path: Path = DEFAULT_SYSTEM_PROMPT_PATH,
    scale: int = GRID_SCALE,
) -> str:
    grid_size = _grid_size_for_scale(scale)
    return (
        path.read_text(encoding="utf-8")
        .format(
            grid_size=grid_size,
            max_grid_index=grid_size - 1,
        )
        .strip()
    )


def _write_run_system_prompt(output_dir: str, system_prompt: str) -> None:
    artifact_dir = Path(output_dir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "system_prompt.md").write_text(
        system_prompt + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _aggregate_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    weighted_keys = {
        "direction_vector_l2": "direction_vector_l2_support",
        "direction_vector_cosine": "direction_vector_cosine_support",
    }
    metrics: Dict[str, float] = {}
    for key in keys:
        support_key = weighted_keys.get(key)
        if support_key is None:
            metrics[key] = float(sum(row.get(key, 0.0) for row in rows) / len(rows))
            continue
        support = sum(row.get(support_key, 0.0) for row in rows)
        metrics[key] = (
            float(
                sum(row.get(key, 0.0) * row.get(support_key, 0.0) for row in rows)
                / support
            )
            if support
            else 0.0
        )
    return metrics


def _text_diagnostics(
    tokenizer: Any,
    system_prompt: str,
    item: LLMGridItem,
    generated_text: str,
    max_new_tokens: int,
) -> Dict[str, float]:
    prompt = _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
    completion = _render_chat_completion(
        tokenizer,
        system_prompt,
        item["input_text"],
        item["target_text"],
    )
    counts = rendered_token_counts(tokenizer, prompt, completion)
    generated_tokens = _token_count(tokenizer, generated_text)
    return {
        "target_token_count": float(counts.completion_tokens),
        "target_completion_token_count": float(counts.completion_tokens),
        "target_over_budget_rate": (
            1.0 if counts.completion_tokens > max_new_tokens else 0.0
        ),
        "generated_token_count": float(generated_tokens),
        "generated_char_count": float(len(generated_text)),
    }


def train_model(args: LLMGridArgs) -> Dict[str, float]:
    if args.finetune_method == "full":
        raise NotImplementedError("full fine-tuning is not implemented for LLM-Grid")
    load_result = load_llm_grid_examples(
        TRAIN_SPLITS,
        limit_per_dataset=args.limit_per_dataset,
        quiet=args.quiet,
        skip_missing_cache=True,
        cognitive_map_namespace=args.cognitive_map_namespace,
    )
    if not load_result.examples:
        raise ValueError("No LLM-Grid training examples were loaded")

    system_prompt = load_system_prompt(scale=args.scale)
    _write_run_system_prompt(args.output_dir, system_prompt)
    model, tokenizer = _load_causal_lm_model_and_tokenizer(
        args.model_name_or_path,
        device_map=_normalize_device_map(args.device_map),
    )
    model = _apply_grid_lora(model, args)
    _cast_trainable_parameters_to_float32(model)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    device = torch.device(args.device)
    if not _model_uses_device_map(model):
        model.to(device)

    all_items = list(LLMGridDataset(load_result.examples, scale=args.scale))
    filtered = filter_grid_training_items(
        all_items,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )
    train_items = list(filtered.kept)
    if not train_items:
        raise ValueError("No LLM-Grid training examples fit the length budgets")
    dropped_over_budget = set(filtered.dropped_prompt_example_ids) | set(
        filtered.dropped_completion_example_ids
    )
    if dropped_over_budget and not args.quiet:
        print(f"skipped_over_budget={len(dropped_over_budget)}")
    dataset = LLMGridItemsDataset(train_items)
    batch_sampler = LengthGroupedBatchSampler(
        _training_sequence_lengths(train_items, tokenizer, system_prompt),
        batch_size=args.per_device_batch_size,
        rank=0,
        world_size=1,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=lambda batch: collate_llm_grid_batch(
            batch,
            tokenizer,
            system_prompt,
            args.max_input_length,
            args.max_new_tokens,
        ),
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
    )
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer_steps = 0
    optimizer.zero_grad()
    for epoch_index in range(args.epochs):
        batch_sampler.set_epoch(epoch_index)
        batch_count = len(loader)
        for batch_index, batch in enumerate(
            _progress(
                loader,
                desc="train LLM-Grid",
                quiet=args.quiet,
            ),
            start=1,
        ):
            outputs = model(**_model_batch(batch, device))
            loss = outputs.loss
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(
                    _non_finite_step_message(
                        "loss",
                        epoch_index + 1,
                        steps + 1,
                        batch,
                    )
                )
            (loss / args.gradient_accumulation_steps).backward()
            should_step_optimizer = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == batch_count
            )
            if should_step_optimizer:
                grad_norm = None
                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        args.max_grad_norm,
                        error_if_nonfinite=False,
                    )
                    if not torch.isfinite(grad_norm.detach()):
                        raise FloatingPointError(
                            _non_finite_step_message(
                                "gradient norm",
                                epoch_index + 1,
                                steps + 1,
                                batch,
                                value=float(grad_norm.detach().cpu()),
                            )
                        )
                optimizer.step()
                optimizer_steps += 1
                optimizer.zero_grad()
                _validate_trainable_parameters_finite(
                    model,
                    context=_non_finite_step_message(
                        "trainable parameter",
                        epoch_index + 1,
                        steps + 1,
                        batch,
                        value=(
                            float(grad_norm.detach().cpu())
                            if grad_norm is not None
                            else None
                        ),
                    ),
                )
            total_loss += float(loss.detach().cpu())
            steps += 1
        epoch_dir = Path(args.output_dir) / "checkpoints" / f"epoch-{epoch_index + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

    checkpoint_dir = Path(args.output_dir) / "checkpoints" / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    metrics = {
        "train_loss": total_loss / steps if steps else 0.0,
        "steps": float(steps),
        "optimizer_steps": float(optimizer_steps),
        "training_example_count": float(len(train_items)),
        "skipped_over_budget_count": float(len(dropped_over_budget)),
        **fixed_corpus_metrics(load_result.by_dataset, all_items, filtered),
    }
    _write_json(Path(args.output_dir) / "metrics.json", metrics)
    return metrics


def _apply_grid_lora(model: Any, args: LLMGridArgs) -> Any:
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


def evaluate_model(args: LLMGridArgs) -> Dict[str, float]:
    load_result = load_llm_grid_examples(
        EVAL_SPLITS,
        limit_per_dataset=args.limit_per_dataset,
        quiet=args.quiet,
        skip_missing_cache=True,
        cognitive_map_namespace=args.cognitive_map_namespace,
    )
    if not load_result.examples:
        raise ValueError("No LLM-Grid eval examples were loaded")

    system_prompt = load_system_prompt(scale=args.scale)
    _write_run_system_prompt(args.output_dir, system_prompt)
    model_path = args.checkpoint_path or args.model_name_or_path
    model, tokenizer = _load_causal_lm_model_and_tokenizer(
        model_path,
        device_map=_normalize_device_map(args.device_map),
    )
    tokenizer.padding_side = "left"
    device = torch.device(args.device)
    if not _model_uses_device_map(model):
        model.to(device)
    model.eval()

    eval_dataset = LLMGridDataset(load_result.examples, scale=args.scale)
    eval_provenance = tuple(
        {
            "example_id": example.example_id,
            "dataset": example.dataset,
        }
        for example in load_result.examples
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_llm_grid_prompt_batch(
            batch,
            tokenizer,
            system_prompt,
            args.max_input_length,
        ),
    )
    rows: List[Dict[str, float]] = []
    rows_by_dataset: Dict[str, List[Dict[str, float]]] = {
        "R2R": [],
        "RxR": [],
    }
    artifact_dir = Path(args.output_dir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in _progress(
            loader,
            desc="eval LLM-Grid",
            quiet=args.quiet,
        ):
            model_batch = _model_batch(batch, device, include_labels=False)
            generated = model.generate(
                **model_batch,
                **_generation_kwargs(tokenizer, args.max_new_tokens),
            )
            completions = [
                decode_generated_completion(tokenizer, sequence, prompt_length)
                for sequence, prompt_length in zip(
                    generated,
                    batch["prompt_lengths"],
                )
            ]
            for item, generated_text in zip(batch["items"], completions):
                metrics = evaluate_grid_prediction(
                    generated_text,
                    item["target_grid"],
                    item["target_direction_vectors"],
                )
                metrics.update(
                    _text_diagnostics(
                        tokenizer,
                        system_prompt,
                        item,
                        generated_text,
                        args.max_new_tokens,
                    )
                )
                rows.append(metrics)
                rows_by_dataset[item["dataset"]].append(metrics)
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
    for prefix, dataset_rows in (
        ("combined", rows),
        ("r2r", rows_by_dataset["R2R"]),
        ("rxr", rows_by_dataset["RxR"]),
    ):
        group_metrics = {name: 0.0 for name in metrics if name != "example_count"}
        group_metrics.update(_aggregate_metrics(dataset_rows))
        group_metrics["example_count"] = float(len(dataset_rows))
        metrics.update(
            {f"{prefix}/{name}": value for name, value in group_metrics.items()}
        )
    metrics.update(
        fixed_corpus_metrics(
            load_result.by_dataset,
            eval_provenance,
            LengthFilterResult(
                kept=eval_provenance,
                dropped_prompt_example_ids=(),
                dropped_completion_example_ids=(),
            ),
        )
    )
    _write_json(Path(args.output_dir) / "metrics.json", metrics)
    return metrics


def main(argv: Optional[List[str]] = None) -> Dict[str, float]:
    args = LLMGridArgs().parse_args(argv)
    if args.mode == "train":
        return train_model(args)
    return evaluate_model(args)


if __name__ == "__main__":
    main()
