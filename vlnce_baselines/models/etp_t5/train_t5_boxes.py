"""Dataset, training, and evaluation CLI for the T5-Boxes milestone."""

from __future__ import annotations
from torch.utils.data import Dataset

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import prior.bbox as bbox
from prior.bbox import SceneSemanticBoxes
from prior.vlnce import DEFAULT_SPLITS, VLNCEEpisodeEntry

from .boxes_metrics import evaluate_t5_boxes_prediction
from .boxes_schema import (
    T5BoxesSpec,
    build_t5_boxes_input,
    parse_t5_boxes_json,
    relevant_semantic_boxes_to_spec,
    spec_to_json,
    spec_to_relevant_semantic_boxes,
    write_prediction_artifact,
)

DEFAULT_MODEL_NAME_OR_PATH = "data/models/t5-large"


@dataclass
class T5BoxesExample:
    example_id: str
    dataset_tag: str
    split: str
    episode_id: int
    instruction: str
    start_position: Sequence[float]
    start_direction: Sequence[float]
    reference_path: Sequence[Sequence[float]]
    target_relevant: bbox.RelevantSemanticBoxes
    target_spec: T5BoxesSpec = field(init=False)

    def __post_init__(self) -> None:
        self.target_spec = relevant_semantic_boxes_to_spec(self.target_relevant)


def load_t5_boxes_examples(
    dataset: Literal["R2R", "RxR"],
    splits: Iterable[str],
    limit: Optional[int] = None,
) -> List[T5BoxesExample]:
    """Load VLN-CE episodes and attach target relevant semantic boxes."""
    if limit == 0:
        return []

    examples: List[T5BoxesExample] = []
    scene_cache: Dict[str, SceneSemanticBoxes] = {}
    for episode in VLNCEEpisodeEntry.iter_from(dataset, splits=splits):
        scene_boxes = scene_cache.get(episode.scene_id)
        if scene_boxes is None:
            scene_boxes = SceneSemanticBoxes.from_scene_id(episode.scene_id)
            scene_cache[episode.scene_id] = scene_boxes

        target_relevant = scene_boxes.relevant_to(
            episode.instruction,
            episode.reference_path,
            episode.start_direction_vector,
        )
        examples.append(
            T5BoxesExample(
                example_id=episode.unique_id,
                dataset_tag=episode.dataset,
                split=episode.split,
                episode_id=episode.episode_id,
                instruction=episode.instruction,
                start_position=episode.start_position,
                start_direction=episode.start_direction_vector,
                reference_path=episode.reference_path,
                target_relevant=target_relevant,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


class T5BoxesDataset(Dataset):
    def __init__(self, examples: Sequence[T5BoxesExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        return {
            "input_text": build_t5_boxes_input(
                example.dataset_tag,
                example.instruction,
                example.start_position,
                example.start_direction,
            ),
            "target_text": spec_to_json(example.target_spec),
            "example_id": example.example_id,
            "target_spec": example.target_spec,
            "target_relevant": example.target_relevant,
            "instruction": example.instruction,
            "level_idx": example.target_relevant.level_idx,
            "reference_path": example.reference_path,
            "start_direction": example.start_direction,
        }


def collate_t5_boxes_batch(
    batch: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_input_length: int,
    max_output_length: int,
) -> Dict[str, Any]:
    input_texts = [item["input_text"] for item in batch]
    target_texts = [_target_text(item) for item in batch]

    encoded = tokenizer(
        input_texts,
        max_length=max_input_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    try:
        target_encoded = tokenizer(
            None,
            max_length=max_output_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            text_target=target_texts,
        )
        encoded["labels"] = target_encoded["input_ids"]
    except TypeError as exc:
        if not _is_unsupported_text_target_error(exc):
            raise
        target_encoded = tokenizer(
            target_texts,
            max_length=max_output_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded["labels"] = target_encoded["input_ids"]

    encoded["labels"] = _mask_pad_tokens(encoded["labels"], tokenizer.pad_token_id)
    encoded["example_ids"] = [item["example_id"] for item in batch]
    encoded["items"] = list(batch)
    return encoded


def train_model(args: argparse.Namespace) -> Dict[str, float]:
    """Fine-tune a seq2seq model end-to-end on T5-Boxes examples."""
    examples = load_t5_boxes_examples(args.dataset, args.splits, limit=args.limit)
    if not examples:
        raise ValueError("No T5-Boxes training examples were loaded")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    device = torch.device(args.device)
    model.to(device)

    dataset = T5BoxesDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_t5_boxes_batch(
            batch, tokenizer, args.max_input_length, args.max_output_length
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    total_loss = 0.0
    steps = 0
    for _epoch in range(args.epochs):
        for batch in loader:
            model_inputs = _model_batch(batch, device)
            outputs = model(**model_inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach().cpu())
            steps += 1

    save_t5_boxes_checkpoint(model, tokenizer, args.output_dir)
    return {"train_loss": total_loss / steps if steps else 0.0, "steps": float(steps)}


def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: Iterable[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Generate, validate, artifact, and score T5-Boxes predictions."""
    import torch

    if hasattr(model, "to"):
        model.to(args.device)
    if hasattr(model, "eval"):
        model.eval()

    loader = _iter_collated_batches(
        dataset,
        args.batch_size,
        lambda batch: collate_t5_boxes_batch(
            batch,
            tokenizer,
            args.max_input_length,
            args.max_output_length,
        ),
    )

    metric_sums: Dict[str, float] = {}
    example_count = 0
    json_parse_count = 0
    schema_valid_count = 0
    entity_valid_rate_sum = 0.0
    entity_valid_support_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            model_inputs = _model_batch(batch, args.device, include_labels=False)
            generated = model.generate(
                **model_inputs,
                max_length=args.max_output_length,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for item, generated_text in zip(batch["items"], decoded):
                example_count += 1
                json_parsed, parsed_json = _parse_json_value(generated_text)
                if json_parsed:
                    json_parse_count += 1
                entity_valid_rate, entity_valid_support = _entity_valid_stats(
                    parsed_json if json_parsed else None
                )
                entity_valid_rate_sum += entity_valid_rate
                entity_valid_support_sum += entity_valid_support

                try:
                    pred_spec = parse_t5_boxes_json(generated_text)
                except Exception as exc:
                    write_prediction_artifact(
                        args.output_dir,
                        item["example_id"],
                        invalid_text=generated_text,
                        error=exc,
                    )
                    continue

                schema_valid_count += 1
                write_prediction_artifact(
                    args.output_dir,
                    item["example_id"],
                    valid_spec=pred_spec,
                )
                pred_relevant = spec_to_relevant_semantic_boxes(
                    pred_spec,
                    instruction=item["instruction"],
                    level_idx=item["level_idx"],
                    reference_path=item["reference_path"],
                    start_direction_vector=item["start_direction"],
                    range_y=item["target_relevant"].level.range_y,
                )
                metrics = evaluate_t5_boxes_prediction(
                    pred_spec,
                    item["target_spec"],
                    pred_relevant,
                    item["target_relevant"],
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metric_key = _aggregate_metric_key(key)
                        metric_sums[metric_key] = metric_sums.get(
                            metric_key, 0.0
                        ) + float(value)

    metrics = (
        {key: value / example_count for key, value in metric_sums.items()}
        if example_count
        else {}
    )
    metrics["examples"] = float(example_count)
    metrics["json_parse_rate"] = (
        float(json_parse_count) / float(example_count) if example_count else 0.0
    )
    metrics["schema_valid_rate"] = (
        float(schema_valid_count) / float(example_count) if example_count else 0.0
    )
    metrics["entity_valid_rate"] = (
        entity_valid_rate_sum / float(example_count) if example_count else 0.0
    )
    metrics["entity_valid_support_mean"] = (
        entity_valid_support_sum / float(example_count) if example_count else 0.0
    )
    return metrics


def save_t5_boxes_checkpoint(
    model: Any, tokenizer: Any, output_dir: str | Path
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _add_common_args(subparsers.add_parser("train", help="Fine-tune T5-Boxes"))
    _add_common_args(subparsers.add_parser("eval", help="Evaluate T5-Boxes"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    args = parse_args(argv)
    if args.mode == "train":
        return train_model(args)

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    examples = load_t5_boxes_examples(args.dataset, args.splits, limit=args.limit)
    metrics = evaluate_model(model, tokenizer, T5BoxesDataset(examples), args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name-or-path",
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="Pretrained or checkpoint path for the seq2seq model.",
    )
    parser.add_argument("--output-dir", default=Path("./data/logs/t5/"))
    parser.add_argument("--dataset", default="R2R", choices=["R2R", "RxR"])
    parser.add_argument("--splits", type=_split_csv, default=list(DEFAULT_SPLITS))
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-output-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=_default_device())


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _mask_pad_tokens(labels: Any, pad_token_id: Optional[int]) -> Any:
    if pad_token_id is None:
        return labels
    try:
        return labels.masked_fill(labels == pad_token_id, -100)
    except AttributeError:
        return [
            [-100 if token == pad_token_id else token for token in row]
            for row in labels
        ]


def _is_unsupported_text_target_error(exc: TypeError) -> bool:
    message = str(exc)
    if "text_target" not in message:
        return False
    return (
        "unexpected" in message
        or "unsupported" in message
        or "got an unexpected keyword argument" in message
    )


def _model_batch(
    batch: Dict[str, Any],
    device: str,
    include_labels: bool = True,
) -> Dict[str, Any]:
    keys = ["input_ids", "attention_mask"]
    if include_labels:
        keys.append("labels")

    model_inputs = {key: batch[key] for key in keys if key in batch}
    for key, value in list(model_inputs.items()):
        if hasattr(value, "to"):
            model_inputs[key] = value.to(device)
    return model_inputs


def _target_text(item: Dict[str, Any]) -> str:
    if "target_text" in item:
        return item["target_text"]
    return spec_to_json(item["target_spec"])


def _iter_collated_batches(
    dataset: Iterable[Dict[str, Any]], batch_size: int, collate_fn
):
    batch: List[Dict[str, Any]] = []
    for item in dataset:
        batch.append(item)
        if len(batch) >= batch_size:
            yield collate_fn(batch)
            batch = []
    if batch:
        yield collate_fn(batch)


def _parse_json_value(text: str) -> Tuple[bool, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False, None
    return True, payload


def _entity_valid_stats(payload: Any) -> Tuple[float, int]:
    if not isinstance(payload, dict):
        return 0.0, 0

    object_items = payload.get("objects", [])
    region_items = payload.get("regions", [])
    if not isinstance(object_items, list):
        object_items = []
    if not isinstance(region_items, list):
        region_items = []

    total_entities = len(object_items) + len(region_items)
    if total_entities == 0:
        try:
            parse_t5_boxes_json(json.dumps(payload, ensure_ascii=True))
        except Exception:
            return 0.0, 0
        return 1.0, 0

    valid_entities = 0
    for item in object_items:
        if _entity_is_valid({"objects": [item], "regions": []}):
            valid_entities += 1
    for item in region_items:
        if _entity_is_valid({"objects": [], "regions": [item]}):
            valid_entities += 1
    return float(valid_entities) / float(total_entities), total_entities


def _entity_is_valid(payload: Dict[str, Any]) -> bool:
    try:
        parse_t5_boxes_json(json.dumps(payload, ensure_ascii=True))
    except Exception:
        return False
    return True


def _aggregate_metric_key(key: str) -> str:  # What does it do???
    if key == "category_aware_raster_support":
        return "category_aware_raster_support_mean"
    return key


def _default_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
