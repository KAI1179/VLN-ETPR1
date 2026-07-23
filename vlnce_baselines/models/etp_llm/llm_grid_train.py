"""Dataset and training CLI for the LLM-Grid milestone."""

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
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed as _set_seed
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
    TrainingManifest,
    TrainingIndex,
    clear_cuda_cache_for_long_sequences,
    configure_peft_fsdp,
    distributed_batch_metrics,
    enable_gradient_checkpointing as _enable_gradient_checkpointing,
    fixed_corpus_metrics,
    load_or_create_training_manifest,
    make_sft_accelerator,
    reduce_training_totals,
    rendered_token_counts,
    run_backward_preflight,
    save_peft_checkpoint,
    scale_training_loss,
    validate_fixed_corpus,
    validate_distributed_device_map,
)
from .llm_boxes_train import (
    _causal_lm_labels,
    _cast_trainable_parameters_to_float32,
    _default_device,
    _encoded_width,
    _model_batch,
    _non_finite_step_message,
    _normalize_device_map,
    _progress,
    _render_chat_completion,
    _render_chat_prompt,
    _token_count,
    _validate_trainable_parameters_finite,
    _validate_supervised_labels,
)
from .llm_grid_evidence import GridEvidenceIndex

GRID_CHANNELS = 37
GRID_SCALE = 2
GRID_SHAPE = (GRID_CHANNELS, 50, 50)
GRID_SIZE_BY_SCALE = {1: 100, 2: 50}
DEFAULT_MODEL_NAME_OR_PATH = LLAMA_3_1_8B_INSTRUCT_MODEL
DEFAULT_GRID_NAMESPACE = "gt.legacy.r1p5.direction5.v1"  # Will be scaled ("blurred") by `GRID_SCALE`, so using non-blurred cache is acceptable (identical to using blurred cache)
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
    split: str
    training_weight: NotRequired[float]
    is_padding: NotRequired[bool]


class LLMGridTrainingItem(TypedDict):
    input_text: str
    target_text: str
    example_id: str
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
    datasets: Iterable[Literal["R2R", "RxR"]] = ("R2R", "RxR"),
) -> ExampleLoadResult[LLMGridExample]:
    examples: List[LLMGridExample] = []
    skipped_missing_cache: List[Tuple[str, str]] = []
    discovered = {"R2R": 0, "RxR": 0}
    loaded = {"R2R": 0, "RxR": 0}
    missing: Dict[str, List[str]] = {"R2R": [], "RxR": []}
    episodes = _progress(
        VLNCEEpisodeEntry.iter_datasets(
            datasets=datasets,
            splits=splits,
            limit_per_dataset=limit_per_dataset,
        ),
        desc="load LLM-Grid examples",
        quiet=quiet,
        total=(limit_per_dataset * 2 if limit_per_dataset is not None else None),
    )
    for episode in episodes:
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
        evidence_indexes: Optional[Mapping[Tuple[str, str], GridEvidenceIndex]] = None,
    ) -> None:
        self.examples = list(examples)
        self.scale = scale
        self.evidence_indexes = evidence_indexes
        self._evidence_prompt_cache: Dict[Tuple[str, str, str], str] = {}

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
        input_text = build_llm_map_input(
            example.instruction,
            start_position,
            start_direction,
        )
        if self.evidence_indexes is not None:
            try:
                evidence_index = self.evidence_indexes[(example.dataset, example.split)]
            except KeyError as error:
                raise KeyError(
                    f"missing evidence index for {example.dataset}/{example.split}"
                ) from error
            episode = evidence_index.episode(example.example_id)
            if episode.scene_id != example.scene_id:
                raise ValueError(
                    f"evidence scene mismatch for {example.example_id}: "
                    f"{episode.scene_id} != {example.scene_id}"
                )
            cache_key = (example.dataset, example.split, episode.observation_id)
            evidence_prompt = self._evidence_prompt_cache.get(cache_key)
            if evidence_prompt is None:
                evidence = evidence_index.load_evidence(example.example_id)
                if not np.allclose(
                    evidence.start_position,
                    start_position,
                    atol=1e-4,
                ):
                    raise ValueError(
                        f"evidence start position mismatch for {example.example_id}"
                    )
                if not np.allclose(
                    evidence.start_direction,
                    start_direction,
                    atol=1e-4,
                ):
                    raise ValueError(
                        f"evidence start direction mismatch for {example.example_id}"
                    )
                evidence_prompt = evidence.prompt_block()
                self._evidence_prompt_cache[cache_key] = evidence_prompt
            input_text = f"{input_text}\n{evidence_prompt}"
        return {
            "input_text": input_text,
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
            "split": example.split,
        }

    def __iter__(self) -> Iterator[LLMGridItem]:
        for index in range(len(self)):
            yield self[index]


class LLMGridItemsDataset(Dataset):
    def __init__(self, items: Sequence[LLMGridTrainingItem]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: Union[int, TrainingIndex]) -> LLMGridTrainingItem:
        if isinstance(index, TrainingIndex):
            item = self.items[index.index].copy()
            item["training_weight"] = index.loss_scale
            item["is_padding"] = index.is_padding
            return item
        return self.items[index]


def filter_grid_training_items(
    items: Iterable[LLMGridItem],
    tokenizer: Any,
    system_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    max_sequence_length: int,
) -> LengthFilterResult[LLMGridItem]:
    filtered: List[LLMGridItem] = []
    dropped_prompt: List[str] = []
    dropped_completion: List[str] = []
    dropped_sequence: List[str] = []
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
        sequence_over_budget = counts.sequence_tokens > max_sequence_length
        if prompt_over_budget:
            dropped_prompt.append(item["example_id"])
        if completion_over_budget:
            dropped_completion.append(item["example_id"])
        if sequence_over_budget:
            dropped_sequence.append(item["example_id"])
        if not (prompt_over_budget or completion_over_budget or sequence_over_budget):
            filtered.append(item)
    return LengthFilterResult(
        kept=tuple(filtered),
        dropped_prompt_example_ids=tuple(dropped_prompt),
        dropped_completion_example_ids=tuple(dropped_completion),
        dropped_sequence_example_ids=tuple(dropped_sequence),
    )


def collate_llm_grid_batch(
    batch: Sequence[LLMGridTrainingItem],
    tokenizer: Any,
    system_prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    max_sequence_length: int,
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
    full_texts = [
        _render_chat_completion(
            tokenizer,
            system_prompt,
            item["input_text"],
            item["target_text"],
        )
        for item in batch
    ]
    token_counts = [
        rendered_token_counts(tokenizer, prompt, completion)
        for prompt, completion in zip(prompt_texts, full_texts)
    ]
    oversized_targets = [
        f"{item['example_id']} ({counts.completion_tokens} tokens)"
        for item, counts in zip(batch, token_counts)
        if counts.completion_tokens > max_new_tokens
    ]
    if oversized_targets:
        raise ValueError(
            f"Target for {', '.join(oversized_targets)} exceeds "
            f"max_new_tokens={max_new_tokens}"
        )
    oversized_sequences = [
        f"{item['example_id']} ({counts.sequence_tokens} tokens)"
        for item, counts in zip(batch, token_counts)
        if counts.sequence_tokens > max_sequence_length
    ]
    if oversized_sequences:
        raise ValueError(
            f"Sequence for {', '.join(oversized_sequences)} exceeds "
            f"max_sequence_length={max_sequence_length}"
        )
    encoded = tokenizer(
        full_texts,
        add_special_tokens=False,
        max_length=max_sequence_length,
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
    oversized = [
        item["example_id"]
        for item, prompt in zip(batch, prompt_texts)
        if _token_count(tokenizer, prompt) > max_input_length
    ]
    if oversized:
        raise ValueError(
            f"Prompt for {', '.join(oversized)} exceeds "
            f"max_input_length={max_input_length}"
        )
    encoded = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        max_length=max_input_length,
        padding=True,
        truncation=False,
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
        raise LLMGridValidationError(
            "direction_vectors must be five [right,up] vectors"
        )
    rows: List[List[float]] = []
    for index, raw_vector in enumerate(value):
        if not isinstance(raw_vector, list) or len(raw_vector) != 2:
            raise LLMGridValidationError(
                f"direction_vectors[{index}] must be [right,up]"
            )
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


class LLMGridArgs(Tap):
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH
    output_dir: str = "outputs/llm_grid"
    cognitive_map_namespace: str = DEFAULT_GRID_NAMESPACE
    evidence_root: str = ""
    evidence_key: str = ""
    scale: int = GRID_SCALE
    max_input_length: int = 1152
    max_new_tokens: int = 3072
    max_sequence_length: int = 4096
    cuda_cache_clear_min_sequence_length: int = 3072
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

    def process_args(self) -> None:
        if not self.device:
            self.device = _default_device()
        if self.scale not in GRID_SIZE_BY_SCALE:
            raise ValueError("LLM-Grid v1 only supports --scale 1 or 2")
        _validate_llm_grid_training_args(self)


def _validate_llm_grid_training_args(args: LLMGridArgs) -> None:
    if bool(args.evidence_root) != bool(args.evidence_key):
        raise ValueError("--evidence-root and --evidence-key must be provided together")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be >= 1")
    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            "LLM-Grid training requires "
            "--gradient-accumulation-steps 1 because its pre-partitioned "
            "loader cannot flush partial accumulation windows"
        )
    if not args.gradient_checkpointing:
        raise ValueError("--gradient-checkpointing is required for LLM-Grid training")
    if args.max_sequence_length < 1:
        raise ValueError("--max-sequence-length must be >= 1")
    if not 1 <= args.cuda_cache_clear_min_sequence_length <= args.max_sequence_length:
        raise ValueError(
            "--cuda-cache-clear-min-sequence-length must be in "
            "[1, --max-sequence-length]"
        )


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
        path
        .read_text(encoding="utf-8")
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


def _training_oom_message(
    accelerator: Accelerator,
    batch: Mapping[str, Any],
    *,
    stage: str,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    sequence_tokens: int,
    error: torch.cuda.OutOfMemoryError,
) -> str:
    memory = "cuda_memory=unavailable"
    if accelerator.device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(accelerator.device)
        allocated_bytes = torch.cuda.memory_allocated(accelerator.device)
        reserved_bytes = torch.cuda.memory_reserved(accelerator.device)
        gib = 1024**3
        memory = (
            f"free_gib={free_bytes / gib:.2f},total_gib={total_bytes / gib:.2f},"
            f"allocated_gib={allocated_bytes / gib:.2f},"
            f"reserved_gib={reserved_bytes / gib:.2f}"
        )
    return (
        f"LLM-Grid CUDA OOM during {stage}: epoch={epoch}, "
        f"step_in_epoch={step_in_epoch}, global_step={global_step}, "
        f"rank={accelerator.process_index}, example_ids={batch['example_ids']}, "
        f"sequence_tokens={sequence_tokens}, {memory}; {error}"
    )


def train_model(args: LLMGridArgs) -> Dict[str, float]:
    _validate_llm_grid_training_args(args)
    if args.finetune_method == "full":
        raise NotImplementedError("full fine-tuning is not implemented for LLM-Grid")

    accelerator = make_sft_accelerator(args.gradient_accumulation_steps)
    validate_distributed_device_map(accelerator, args.device_map)
    _set_seed(args.seed, device_specific=False)
    batch_metrics = distributed_batch_metrics(
        accelerator,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
    )
    system_prompt = load_system_prompt(scale=args.scale)
    if accelerator.is_main_process:
        _write_run_system_prompt(args.output_dir, system_prompt)
    tokenizer = _load_grid_training_tokenizer(args.model_name_or_path)
    manifest = load_or_create_training_manifest(
        accelerator,
        Path(args.output_dir) / "artifacts" / "training_manifest.jsonl",
        lambda: _build_grid_training_manifest(args, tokenizer, system_prompt),
    )
    train_items = [_grid_training_item_from_manifest(item) for item in manifest.items]
    sequence_lengths = [int(item["sequence_tokens"]) for item in manifest.items]
    skipped_over_budget_count = int(manifest.metadata["skipped_over_budget_count"])
    corpus_metrics = {
        str(name): float(value)
        for name, value in dict(manifest.metadata["fixed_corpus_metrics"]).items()
    }

    model = _load_grid_training_model(
        args.model_name_or_path,
        device_map=(
            None
            if accelerator.num_processes > 1
            else _normalize_device_map(args.device_map)
        ),
    )
    model = _apply_grid_lora(model, args)
    _cast_trainable_parameters_to_float32(model)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    configure_peft_fsdp(accelerator, model)
    dataset = LLMGridItemsDataset(train_items)
    batch_sampler = LengthGroupedBatchSampler(
        sequence_lengths,
        batch_size=args.per_device_batch_size,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
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
            args.max_sequence_length,
        ),
    )
    model = accelerator.prepare(model)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
    )
    optimizer = accelerator.prepare(optimizer)
    model.train()
    longest_index = max(
        range(len(sequence_lengths)),
        key=lambda index: sequence_lengths[index],
    )
    longest_item = dataset[TrainingIndex(longest_index)]
    if accelerator.is_main_process and not args.quiet:
        print(
            "preflight_longest_sequence="
            f"{longest_item['example_id']}:{sequence_lengths[longest_index]}"
        )
    preflight_batch = collate_llm_grid_batch(
        [longest_item],
        tokenizer,
        system_prompt,
        args.max_input_length,
        args.max_new_tokens,
        args.max_sequence_length,
    )
    run_backward_preflight(
        accelerator,
        model,
        optimizer,
        _model_batch(preflight_batch, accelerator.device),
        example_id=longest_item["example_id"],
        sequence_tokens=sequence_lengths[longest_index],
    )
    local_loss_sum = 0.0
    local_example_count = 0
    local_batch_count = 0
    optimizer_steps = 0
    for epoch_index in range(args.epochs):
        batch_sampler.set_epoch(epoch_index)
        progress_loader = _progress(
            loader,
            desc=f"train LLM-Grid epoch {epoch_index + 1}/{args.epochs}",
            quiet=args.quiet or not accelerator.is_main_process,
        )
        for step_in_epoch, batch in enumerate(progress_loader, start=1):
            sequence_tokens = int(batch["input_ids"].shape[-1])
            clear_cuda_cache_for_long_sequences(
                accelerator,
                local_sequence_tokens=sequence_tokens,
                minimum_sequence_tokens=(args.cuda_cache_clear_min_sequence_length),
            )
            oom_stage = "forward"
            try:
                with accelerator.accumulate(model):
                    loss = model(**_model_batch(batch, accelerator.device)).loss
                    if not torch.isfinite(loss.detach()):
                        raise FloatingPointError(
                            _non_finite_step_message(
                                "loss",
                                epoch_index + 1,
                                local_batch_count + 1,
                                batch,
                            )
                        )
                    weighted_loss = scale_training_loss(
                        loss,
                        batch["training_weights"],
                        batch["is_padding"],
                    )
                    oom_stage = "backward"
                    accelerator.backward(weighted_loss)
                    del weighted_loss
                    grad_norm = None
                    if accelerator.sync_gradients and args.max_grad_norm > 0:
                        oom_stage = "gradient clipping"
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )
                        if not torch.isfinite(grad_norm.detach()):
                            raise FloatingPointError(
                                _non_finite_step_message(
                                    "gradient norm",
                                    epoch_index + 1,
                                    local_batch_count + 1,
                                    batch,
                                    value=float(grad_norm.detach().cpu()),
                                )
                            )
                    oom_stage = "optimizer step"
                    optimizer.step()
                    optimizer.zero_grad()
                    if accelerator.sync_gradients:
                        optimizer_steps += 1
                        oom_stage = "parameter validation"
                        _validate_trainable_parameters_finite(
                            model,
                            context=_non_finite_step_message(
                                "trainable parameter",
                                epoch_index + 1,
                                local_batch_count + 1,
                                batch,
                                value=(
                                    float(grad_norm.detach().cpu())
                                    if grad_norm is not None
                                    else None
                                ),
                            ),
                        )
            except torch.cuda.OutOfMemoryError as error:
                raise torch.cuda.OutOfMemoryError(
                    _training_oom_message(
                        accelerator,
                        batch,
                        stage=oom_stage,
                        epoch=epoch_index + 1,
                        step_in_epoch=step_in_epoch,
                        global_step=local_batch_count + 1,
                        sequence_tokens=sequence_tokens,
                        error=error,
                    )
                ) from error
            real_example_count = int((~batch["is_padding"].bool()).sum().item())
            local_loss_sum += float(loss.detach().cpu()) * real_example_count
            del loss
            local_example_count += real_example_count
            local_batch_count += 1
        save_peft_checkpoint(
            accelerator,
            model,
            tokenizer,
            Path(args.output_dir) / "checkpoints" / f"epoch-{epoch_index + 1}",
        )

    save_peft_checkpoint(
        accelerator,
        model,
        tokenizer,
        Path(args.output_dir) / "checkpoints" / "final",
    )
    training_totals = reduce_training_totals(
        accelerator,
        local_loss_sum,
        local_example_count,
        local_batch_count,
    )
    metrics = {
        "train_loss": training_totals["loss"],
        "examples": training_totals["example_count"],
        "steps": training_totals["batch_count"],
        "optimizer_steps": float(optimizer_steps),
        "batches_per_epoch": training_totals["batch_count"] / args.epochs,
        "optimizer_steps_per_epoch": float(optimizer_steps) / args.epochs,
        "world_size": float(batch_metrics.world_size),
        "per_device_batch_size": float(batch_metrics.per_device_batch_size),
        "gradient_accumulation_steps": float(batch_metrics.gradient_accumulation_steps),
        "global_batch_size": float(batch_metrics.global_batch_size),
        "seed": float(args.seed),
        "skipped_over_budget_count": float(skipped_over_budget_count),
        **corpus_metrics,
    }
    if accelerator.is_main_process:
        _write_json(Path(args.output_dir) / "metrics.json", metrics)
    return metrics


def _build_grid_training_manifest(
    args: LLMGridArgs,
    tokenizer: Any,
    system_prompt: str,
) -> TrainingManifest:
    load_result = load_llm_grid_examples(
        TRAIN_SPLITS,
        limit_per_dataset=args.limit_per_dataset,
        quiet=args.quiet,
        skip_missing_cache=True,
        cognitive_map_namespace=args.cognitive_map_namespace,
    )
    validate_fixed_corpus(load_result.by_dataset)
    evidence_indexes = _load_grid_evidence_indexes(
        args.evidence_root,
        args.evidence_key,
        TRAIN_SPLITS,
    )
    all_items = list(
        _progress(
            LLMGridDataset(
                load_result.examples,
                scale=args.scale,
                evidence_indexes=evidence_indexes,
            ),
            desc="serialize LLM-Grid targets",
            quiet=args.quiet,
            total=len(load_result.examples),
        )
    )
    filtered = filter_grid_training_items(
        _progress(
            all_items,
            desc="filter LLM-Grid token lengths",
            quiet=args.quiet,
            total=len(all_items),
        ),
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        max_sequence_length=args.max_sequence_length,
    )
    train_items = list(filtered.kept)
    validate_fixed_corpus(load_result.by_dataset, retained_items=train_items)
    dropped_over_budget = (
        set(filtered.dropped_prompt_example_ids)
        | set(filtered.dropped_completion_example_ids)
        | set(filtered.dropped_sequence_example_ids)
    )
    if dropped_over_budget and not args.quiet:
        print(f"skipped_over_budget={len(dropped_over_budget)}")

    manifest_items = []
    for item in _progress(
        train_items,
        desc="assemble LLM-Grid manifest",
        quiet=args.quiet,
        total=len(train_items),
    ):
        prompt = _render_chat_prompt(tokenizer, system_prompt, item["input_text"])
        completion = _render_chat_completion(
            tokenizer,
            system_prompt,
            item["input_text"],
            item["target_text"],
        )
        counts = rendered_token_counts(tokenizer, prompt, completion)
        manifest_items.append({
            "input_text": item["input_text"],
            "target_text": item["target_text"],
            "example_id": item["example_id"],
            "dataset": item["dataset"],
            "prompt_tokens": counts.prompt_tokens,
            "completion_tokens": counts.completion_tokens,
            "sequence_tokens": counts.sequence_tokens,
        })
    return TrainingManifest(
        metadata={
            "seed": args.seed,
            "scale": args.scale,
            "cognitive_map_namespace": args.cognitive_map_namespace,
            "evidence": (
                None
                if evidence_indexes is None
                else {
                    "root": args.evidence_root,
                    "key": args.evidence_key,
                    "manifests": {
                        f"{dataset}/{split}": index.manifest_sha256
                        for (dataset, split), index in sorted(evidence_indexes.items())
                    },
                }
            ),
            "skipped_over_budget_count": len(dropped_over_budget),
            "token_budgets": {
                "max_input_length": args.max_input_length,
                "max_new_tokens": args.max_new_tokens,
                "max_sequence_length": args.max_sequence_length,
            },
            "dropped_sequence_example_ids": filtered.dropped_sequence_example_ids,
            "fixed_corpus_metrics": fixed_corpus_metrics(
                load_result.by_dataset,
                all_items,
                filtered,
            ),
        },
        items=tuple(manifest_items),
    )


def _load_grid_evidence_indexes(
    evidence_root: str,
    evidence_key: str,
    splits: Iterable[str],
    *,
    datasets: Iterable[Literal["R2R", "RxR"]] = ("R2R", "RxR"),
) -> Optional[Dict[Tuple[str, str], GridEvidenceIndex]]:
    if not evidence_root and not evidence_key:
        return None
    if not evidence_root or not evidence_key:
        raise ValueError("evidence_root and evidence_key must be provided together")
    return {
        (dataset, split): GridEvidenceIndex.load(
            evidence_root,
            evidence_key,
            dataset,
            split,
        )
        for dataset in datasets
        for split in splits
    }


def _grid_training_item_from_manifest(
    item: Mapping[str, Any],
) -> LLMGridTrainingItem:
    dataset = _episode_dataset(str(item["dataset"]))
    return {
        "input_text": str(item["input_text"]),
        "target_text": str(item["target_text"]),
        "example_id": str(item["example_id"]),
        "dataset": dataset,
    }


def _load_grid_training_tokenizer(model_name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_grid_training_model(
    model_name_or_path: str,
    device_map: Optional[str],
) -> Any:
    from transformers import AutoModelForCausalLM

    model_kwargs: Dict[str, Any] = {"torch_dtype": "auto"}
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)


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


def main(argv: Optional[List[str]] = None) -> Dict[str, float]:
    args = LLMGridArgs().parse_args(argv)
    return train_model(args)


if __name__ == "__main__":
    main()
