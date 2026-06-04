"""Generate precomputed LLM-Navigation cognitive-map caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from prior.bbox import SceneSemanticBoxes
from prior.etp_r1 import (
    ANNOTATION_FILES as PRETRAIN_ANNOTATION_FILES,
    AnnotationEntry as PretrainAnnotationEntry,
)

from .boxes_schema import (
    LLMBoxesSpec,
    build_llm_boxes_input,
    parse_llm_boxes_text,
    parse_llm_boxes_text_salvage,
    spec_to_llm_boxes_text,
    spec_to_relevant_semantic_boxes,
)
from .navigation import (
    DEFAULT_LLM_NAVIGATION_MODEL_KEY,
    llm_navigation_cognitive_map_path,
    llm_navigation_prediction_path,
    llm_navigation_split_dir,
)
from .train_llm_boxes import (
    DEFAULT_MODEL_NAME_OR_PATH,
    LLMBoxesDataset,
    LLMBoxesItem,
    collate_llm_boxes_prompt_batch,
    decode_generated_completion,
    load_llm_boxes_examples,
    load_system_prompt,
    _batch_count,
    _default_device,
    _iter_collated_batches,
    _load_causal_lm_model_and_tokenizer,
    _model_batch,
    _model_uses_device_map,
    _normalize_device_map,
    _progress,
)

VLNCE_DATASETS = ("R2R", "RxR")
VLNCE_SPLITS = ("train", "val_seen", "val_unseen")
PRETRAIN_DATASET_KEY = "pretrain"
PRETRAIN_SPLIT = "mixed"


def generate_navigation_cache(
    model: Any,
    tokenizer: Any,
    dataset: Iterable[LLMBoxesItem],
    args: argparse.Namespace,
    *,
    dataset_key: str,
    split: str,
) -> Dict[str, float]:
    """Generate prediction text and cognitive-map caches for one namespace."""
    import torch

    if hasattr(model, "to") and not _model_uses_device_map(model):
        model.to(args.device)
    if hasattr(model, "eval"):
        model.eval()

    system_prompt = getattr(args, "system_prompt", None) or load_system_prompt()
    split_dir = llm_navigation_split_dir(
        dataset_key,
        split,
        cache_dir=args.cache_dir,
        model_key=args.cache_model_key,
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    _write_navigation_cache_manifest(split_dir, args, system_prompt, dataset_key, split)

    loader = _iter_collated_batches(
        dataset,
        args.batch_size,
        lambda batch: collate_llm_boxes_prompt_batch(
            batch,
            tokenizer,
            system_prompt,
            args.max_input_length,
        ),
    )
    progress_loader = _progress(
        loader,
        desc=f"generate LLM-Navigation cache {dataset_key}/{split}",
        quiet=bool(getattr(args, "quiet", False)),
        total=_batch_count(dataset, args.batch_size),
    )

    examples = 0
    strict_valid = 0
    salvaged = 0
    missing_path = 0
    fallback_empty = 0
    failure_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in progress_loader:
            model_inputs = _model_batch(batch, args.device, include_labels=False)
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            decoded = [
                decode_generated_completion(tokenizer, sequence, prompt_length)
                for sequence, prompt_length in zip(generated, batch["prompt_lengths"])
            ]

            for item, generated_text in zip(batch["items"], decoded):
                examples += 1
                cache_id = item["example_id"]
                scene_id = item["scene_id"]
                prediction_path = llm_navigation_prediction_path(
                    scene_id,
                    cache_id,
                    dataset_key,
                    split,
                    cache_dir=args.cache_dir,
                    model_key=args.cache_model_key,
                )
                prediction_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    strict_spec = parse_llm_boxes_text(generated_text)
                except Exception as exc:
                    strict_spec = None
                    failure_records.append(
                        _navigation_cache_failure_record(
                            item,
                            stage="strict_parse",
                            error=exc,
                            prediction_path=prediction_path,
                        )
                    )
                    _warn_navigation_cache_failure(item, "strict_parse", exc)
                else:
                    strict_valid += 1

                salvage = parse_llm_boxes_text_salvage(generated_text)
                spec = strict_spec if strict_spec is not None else salvage.spec
                if salvage.dropped_entity_count:
                    salvaged += 1
                    error_text = "; ".join(salvage.errors)
                    failure_records.append(
                        _navigation_cache_failure_record(
                            item,
                            stage="salvage_parse",
                            error=error_text,
                            prediction_path=prediction_path,
                            dropped_entities=salvage.dropped_entities,
                        )
                    )
                    _warn_navigation_cache_failure(item, "salvage_parse", error_text)

                if not spec.reference_path:
                    missing_path += 1
                    spec = LLMBoxesSpec(
                        objects=spec.objects,
                        regions=spec.regions,
                        reference_path=(tuple(item["start_position"]),),
                    )
                    failure_records.append(
                        _navigation_cache_failure_record(
                            item,
                            stage="missing_reference_path",
                            error="using start-position fallback path",
                            prediction_path=prediction_path,
                        )
                    )
                    _warn_navigation_cache_failure(
                        item,
                        "missing_reference_path",
                        "using start-position fallback path",
                    )

                if (
                    strict_spec is None
                    and not spec.objects
                    and not spec.regions
                    and len(spec.reference_path) <= 1
                ):
                    fallback_empty += 1

                prediction_path.write_text(
                    f"{spec_to_llm_boxes_text(spec)}\n",
                    encoding="utf-8",
                )
                relevant = spec_to_relevant_semantic_boxes(
                    spec,
                    instruction=item["instruction"],
                    level_idx=item["level_idx"],
                    reference_path=(item["start_position"],),
                    start_direction_vector=item["start_direction"],
                    range_y=item["target_relevant"].level.range_y,
                )
                cognitive_map_path = llm_navigation_cognitive_map_path(
                    scene_id,
                    cache_id,
                    dataset_key,
                    split,
                    cache_dir=args.cache_dir,
                    model_key=args.cache_model_key,
                )
                cognitive_map_path.parent.mkdir(parents=True, exist_ok=True)
                relevant.to_cognitive_map().save(cognitive_map_path)

    _write_jsonl(split_dir / "failures.jsonl", failure_records)
    metrics = {
        "examples": float(examples),
        "strict_valid": float(strict_valid),
        "salvaged": float(salvaged),
        "missing_reference_path": float(missing_path),
        "fallback_empty": float(fallback_empty),
        "strict_parse_failure_rate": (
            float(examples - strict_valid) / float(examples) if examples else 0.0
        ),
        "salvage_rate": float(salvaged) / float(examples) if examples else 0.0,
        "missing_reference_path_rate": (
            float(missing_path) / float(examples) if examples else 0.0
        ),
        "fallback_empty_rate": (
            float(fallback_empty) / float(examples) if examples else 0.0
        ),
    }
    (split_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def load_pretrain_cache_items(
    annotation_files: Sequence[str] = PRETRAIN_ANNOTATION_FILES,
    limit: Optional[int] = None,
    quiet: bool = False,
) -> List[LLMBoxesItem]:
    """Load ETP-R1 pretraining annotations as LLM-Navigation cache items."""
    if limit == 0:
        return []

    items: List[LLMBoxesItem] = []
    for annotation_file in annotation_files:
        entries = _progress(
            PretrainAnnotationEntry.iter_from(annotation_file),
            desc=f"load pretrain cache items {annotation_file}",
            quiet=quiet,
            total=limit,
        )
        dataset_tag = _pretrain_dataset_tag(annotation_file)
        for entry in entries:
            positions = entry.positions()
            target_relevant = SceneSemanticBoxes.from_scene_id(entry.scan).relevant_to(
                entry.instruction,
                positions,
                entry.start_direction_vector,
            )
            start_position = _level_local_start_position(
                target_relevant.reference_path,
                positions[0],
            )
            target_spec = LLMBoxesSpec(objects=(), regions=())
            items.append(
                {
                    "input_text": build_llm_boxes_input(
                        dataset_tag,
                        entry.instruction,
                        start_position,
                        entry.start_direction_vector,
                    ),
                    "target_text": spec_to_llm_boxes_text(target_spec),
                    "target_spec": target_spec,
                    "target_relevant": target_relevant,
                    "example_id": entry.instr_id,
                    "scene_id": entry.scan,
                    "instruction": entry.instruction,
                    "level_idx": target_relevant.level_idx,
                    "reference_path": target_relevant.reference_path,
                    "start_direction": entry.start_direction_vector,
                    "start_position": start_position,
                }
            )
            if limit is not None and len(items) >= limit:
                return items
    return items


def generate_all_navigation_caches(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    """Generate all VLN-CE and pretraining LLM-Navigation caches."""
    metrics: Dict[str, Dict[str, float]] = {}
    for dataset_key in VLNCE_DATASETS:
        for split in VLNCE_SPLITS:
            examples = load_llm_boxes_examples(
                dataset_key,
                (split,),
                limit=args.limit,
                quiet=args.quiet,
            )
            metrics[f"{dataset_key.lower()}/{split}"] = generate_navigation_cache(
                model,
                tokenizer,
                LLMBoxesDataset(examples),
                args,
                dataset_key=dataset_key,
                split=split,
            )

    pretrain_items = load_pretrain_cache_items(limit=args.limit, quiet=args.quiet)
    metrics[f"{PRETRAIN_DATASET_KEY}/{PRETRAIN_SPLIT}"] = generate_navigation_cache(
        model,
        tokenizer,
        pretrain_items,
        args,
        dataset_key=PRETRAIN_DATASET_KEY,
        split=PRETRAIN_SPLIT,
    )
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name-or-path",
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="Pretrained or checkpoint path for the causal language model.",
    )
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=["auto", "balanced", "balanced_low_0", "sequential", "none"],
        help=(
            "Optional Transformers/Accelerate model-parallel device map. "
            "Use none for ordinary single-device loading."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Cache root. Empty uses data/llm_navigation.",
    )
    parser.add_argument(
        "--cache-model-key",
        default=DEFAULT_LLM_NAVIGATION_MODEL_KEY,
        help="Model namespace under the cache root.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    args = parse_args(argv)
    model, tokenizer = _load_causal_lm_model_and_tokenizer(
        args.model_name_or_path,
        device_map=_normalize_device_map(args.device_map),
    )
    return generate_all_navigation_caches(model, tokenizer, args)


def _pretrain_dataset_tag(annotation_file: str) -> str:
    name = Path(annotation_file).name.lower()
    if name.startswith("rxr_"):
        return "RxR"
    if "gemini" in name:
        return "Gemini"
    if "prevalent" in name:
        return "Prevalent"
    return "R2R"


def _level_local_start_position(
    reference_path: Sequence[Sequence[float]],
    start_position: Sequence[float],
) -> Tuple[float, float]:
    if reference_path:
        return float(reference_path[0][0]), float(reference_path[0][1])
    if len(start_position) >= 3:
        return float(start_position[0]), float(start_position[2])
    return float(start_position[0]), float(start_position[1])


def _write_navigation_cache_manifest(
    split_dir: Path,
    args: argparse.Namespace,
    system_prompt: str,
    dataset_key: str,
    split: str,
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_key,
        "split": split,
        "model_name_or_path": args.model_name_or_path,
        "cache_model_key": args.cache_model_key,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _navigation_cache_failure_record(
    item: LLMBoxesItem,
    stage: str,
    error: BaseException | str,
    prediction_path: Path,
    dropped_entities: Sequence[str] = (),
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "example_id": item["example_id"],
        "scene_id": item["scene_id"],
        "stage": stage,
        "error": str(error),
        "prediction_path": str(prediction_path),
    }
    if dropped_entities:
        record["dropped_entities"] = list(dropped_entities)
    return record


def _warn_navigation_cache_failure(
    item: LLMBoxesItem,
    stage: str,
    error: BaseException | str,
) -> None:
    warnings.warn(
        f"LLM-Navigation cache warning for {item['example_id']} "
        f"at {stage}: {error}",
        RuntimeWarning,
        stacklevel=2,
    )


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            file.write("\n")


if __name__ == "__main__":
    main()
