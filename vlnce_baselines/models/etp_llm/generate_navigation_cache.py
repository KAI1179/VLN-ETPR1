"""Generate precomputed LLM-Navigation cognitive-map caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from prior.bbox import SceneSemanticBoxes
from prior.trajectory import InsufficientTrajectoryPointsError
from prior.vlnce import VLNCEEpisodeEntry
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
    LLMBoxesItem,
    collate_llm_boxes_prompt_batch,
    decode_generated_completion,
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

VLNCE_DATASETS: Tuple[Literal["R2R", "RxR"], ...] = ("R2R", "RxR")
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

    examples = 0
    cached = 0

    def pending_items() -> Iterable[LLMBoxesItem]:
        nonlocal cached, examples
        for item in dataset:
            examples += 1
            if not bool(getattr(args, "overwrite", False)) and _cache_complete(
                item["scene_id"],
                item["example_id"],
                dataset_key,
                split,
                args,
            ):
                cached += 1
                continue
            yield item

    loader = _iter_collated_batches(
        pending_items(),
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

    attempted = 0
    strict_valid = 0
    salvaged = 0
    missing_keypoints = 0
    generated = 0
    skipped = 0
    failure_records: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for batch in progress_loader:
            attempted += len(batch["items"])
            model_inputs = _model_batch(batch, args.device, include_labels=False)
            generated_sequences = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            decoded = [
                decode_generated_completion(tokenizer, sequence, prompt_length)
                for sequence, prompt_length in zip(
                    generated_sequences, batch["prompt_lengths"]
                )
            ]

            for item, generated_text in zip(batch["items"], decoded):
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

                if not spec.trajectory_keypoints:
                    missing_keypoints += 1
                    skipped += 1
                    failure_records.append(
                        _navigation_cache_failure_record(
                            item,
                            stage="missing_trajectory_keypoints",
                            error="refusing to generate cache without trajectory keypoints",
                            prediction_path=prediction_path,
                        )
                    )
                    _warn_navigation_cache_failure(
                        item,
                        "missing_trajectory_keypoints",
                        "refusing to generate cache without trajectory keypoints",
                    )
                    prediction_path.write_text(
                        f"{generated_text.strip()}\n",
                        encoding="utf-8",
                    )
                    continue

                prediction_path.write_text(
                    f"{spec_to_llm_boxes_text(spec)}\n",
                    encoding="utf-8",
                )
                relevant = spec_to_relevant_semantic_boxes(
                    spec,
                    instruction=item["instruction"],
                    level_idx=item["level_idx"],
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
                generated += 1

    _write_jsonl(split_dir / "failures.jsonl", failure_records)
    metrics = {
        "examples": float(examples),
        "cached": float(cached),
        "attempted": float(attempted),
        "strict_valid": float(strict_valid),
        "salvaged": float(salvaged),
        "missing_trajectory_keypoints": float(missing_keypoints),
        "generated": float(generated),
        "skipped": float(skipped),
        "strict_parse_failure_rate": (
            float(attempted - strict_valid) / float(attempted) if attempted else 0.0
        ),
        "salvage_rate": float(salvaged) / float(attempted) if attempted else 0.0,
        "missing_trajectory_keypoints_rate": (
            float(missing_keypoints) / float(attempted) if attempted else 0.0
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
    args: Optional[argparse.Namespace] = None,
) -> List[LLMBoxesItem]:
    """Load ETP-R1 pretraining annotations as LLM-Navigation cache items."""
    if limit == 0:
        return []

    items: List[LLMBoxesItem] = []
    for annotation_file in annotation_files:
        start_count = len(items)
        total = 0
        cached = 0
        raw_entries = PretrainAnnotationEntry.iter_from(annotation_file)
        entries = _progress(
            raw_entries,
            desc=f"load pretrain cache items {annotation_file}",
            quiet=quiet,
            total=limit,
        )
        dataset_tag = _pretrain_dataset_tag(annotation_file)
        try:
            for entry in entries:
                if args is not None and not _belongs_to_worker(entry.instr_id, args):
                    continue
                total += 1
                if args is not None and not bool(getattr(args, "overwrite", False)):
                    if _cache_complete(
                        entry.scan,
                        entry.instr_id,
                        PRETRAIN_DATASET_KEY,
                        PRETRAIN_SPLIT,
                        args,
                    ):
                        cached += 1
                        continue
                positions = entry.positions()
                try:
                    target_relevant = SceneSemanticBoxes.from_scene_id(
                        entry.scan
                    ).relevant_to(
                        entry.instruction,
                        positions,
                        entry.start_direction_vector,
                    )
                except InsufficientTrajectoryPointsError as error:
                    warnings.warn(
                        f"skipping {entry.instr_id}: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                start_position = _level_local_start_position(
                    target_relevant.trajectory_keypoints,
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
                        "trajectory_keypoints": target_relevant.trajectory_keypoints,
                        "start_direction": entry.start_direction_vector,
                        "start_position": start_position,
                    }
                )
                if limit is not None and len(items) >= limit:
                    _print_resume_summary(
                        f"pretrain/{annotation_file}",
                        total=total,
                        cached=cached,
                        pending=len(items) - start_count,
                    )
                    return items
        except FileNotFoundError as error:
            warnings.warn(
                f"skipping missing pretrain annotation {annotation_file}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            if total:
                _print_resume_summary(
                    f"pretrain/{annotation_file}",
                    total=total,
                    cached=cached,
                    pending=len(items) - start_count,
                )
    return items


def load_vlnce_cache_items(
    dataset_key: Literal["R2R", "RxR"],
    split: str,
    limit: Optional[int] = None,
    quiet: bool = False,
    args: Optional[argparse.Namespace] = None,
) -> List[LLMBoxesItem]:
    """Load uncached VLN-CE episodes as LLM-Navigation cache items."""
    if limit == 0:
        return []

    items: List[LLMBoxesItem] = []
    total = 0
    cached = 0
    episodes: Iterable[VLNCEEpisodeEntry] = _progress(
        VLNCEEpisodeEntry.iter_from(dataset_key, splits=(split,)),
        desc=f"load VLN-CE cache items {dataset_key}/{split}",
        quiet=quiet,
        total=limit,
    )
    for episode in episodes:
        if args is not None and not _belongs_to_worker(episode.unique_id, args):
            continue
        total += 1
        if args is not None and not bool(getattr(args, "overwrite", False)):
            if _cache_complete(
                episode.scene_id,
                episode.unique_id,
                dataset_key,
                split,
                args,
            ):
                cached += 1
                continue
        try:
            target_relevant = SceneSemanticBoxes.from_scene_id(
                episode.scene_id
            ).relevant_to(
                episode.instruction,
                episode.ground_truth_trajectory,
                episode.start_direction_vector,
            )
        except InsufficientTrajectoryPointsError as error:
            warnings.warn(
                f"skipping {episode.unique_id}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        start_position = _level_local_start_position(
            target_relevant.trajectory_keypoints,
            episode.start_position,
        )
        target_spec = LLMBoxesSpec(objects=(), regions=())
        items.append(
            {
                "input_text": build_llm_boxes_input(
                    episode.dataset,
                    episode.instruction,
                    start_position,
                    episode.start_direction_vector,
                ),
                "target_text": spec_to_llm_boxes_text(target_spec),
                "target_spec": target_spec,
                "target_relevant": target_relevant,
                "example_id": episode.unique_id,
                "scene_id": episode.scene_id,
                "instruction": episode.instruction,
                "level_idx": target_relevant.level_idx,
                "trajectory_keypoints": target_relevant.trajectory_keypoints,
                "start_direction": episode.start_direction_vector,
                "start_position": start_position,
            }
        )
        if limit is not None and len(items) >= limit:
            break
    _print_resume_summary(
        f"{dataset_key}/{split}",
        total=total,
        cached=cached,
        pending=len(items),
    )
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
            examples = load_vlnce_cache_items(
                dataset_key,
                split,
                limit=args.limit,
                quiet=args.quiet,
                args=args,
            )
            metrics[f"{dataset_key.lower()}/{split}"] = generate_navigation_cache(
                model,
                tokenizer,
                examples,
                args,
                dataset_key=dataset_key,
                split=split,
            )

    pretrain_items = load_pretrain_cache_items(
        limit=args.limit,
        quiet=args.quiet,
        args=args,
    )
    metrics[f"{PRETRAIN_DATASET_KEY}/{PRETRAIN_SPLIT}"] = generate_navigation_cache(
        model,
        tokenizer,
        pretrain_items,
        args,
        dataset_key=PRETRAIN_DATASET_KEY,
        split=PRETRAIN_SPLIT,
    )
    return metrics


def _skipped_cache_count(metrics: Dict[str, Dict[str, float]]) -> int:
    return int(
        sum(split_metrics.get("skipped", 0.0) for split_metrics in metrics.values())
    )


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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate prediction text and cognitive maps even when both exist.",
    )
    parser.add_argument(
        "--parallel-workers",
        default="auto",
        help=(
            "Number of cache-generation worker processes. Use auto to launch one "
            "worker per visible CUDA device; use 1 for single-process generation."
        ),
    )
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    args = parse_args(argv)
    worker_count = _resolve_parallel_worker_count(args.parallel_workers)
    if worker_count > 1 and args.worker_count == 1:
        return _run_parallel_workers(args, worker_count)

    model, tokenizer = _load_causal_lm_model_and_tokenizer(
        args.model_name_or_path,
        device_map=_normalize_device_map(args.device_map),
    )
    metrics = generate_all_navigation_caches(model, tokenizer, args)
    skipped = _skipped_cache_count(metrics)
    print(f"skipped={skipped}")
    if skipped:
        raise SystemExit(1)
    return metrics


def _resolve_parallel_worker_count(value: str) -> int:
    if value == "auto":
        return max(1, len(_visible_cuda_devices()))
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError("--parallel-workers must be auto or a positive integer") from error
    if count < 1:
        raise ValueError("--parallel-workers must be at least 1")
    return count


def _visible_cuda_devices() -> List[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        return [
            device.strip()
            for device in cuda_visible_devices.split(",")
            if device.strip() and device.strip() != "-1"
        ]

    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    return [str(index) for index in range(torch.cuda.device_count())]


def _run_parallel_workers(
    args: argparse.Namespace,
    worker_count: int,
) -> Dict[str, Dict[str, float]]:
    devices = _visible_cuda_devices()
    processes: List[subprocess.Popen] = []
    print(f"parallel_workers={worker_count} visible_cuda_devices={devices or ['cpu']}")
    for worker_index in range(worker_count):
        command = _worker_command(args, worker_count, worker_index)
        env = os.environ.copy()
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices[worker_index % len(devices)]
        processes.append(subprocess.Popen(command, env=env))

    failed = []
    for worker_index, process in enumerate(processes):
        return_code = process.wait()
        if return_code:
            failed.append((worker_index, return_code))

    if failed:
        raise SystemExit(f"LLM-Navigation cache workers failed: {failed}")
    return {}


def _worker_command(
    args: argparse.Namespace,
    worker_count: int,
    worker_index: int,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "vlnce_baselines.models.etp_llm.generate_navigation_cache",
        "--model-name-or-path",
        str(args.model_name_or_path),
        "--max-input-length",
        str(args.max_input_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--device-map",
        (
            "none"
            if _normalize_device_map(args.device_map) is None
            else str(args.device_map)
        ),
        "--cache-dir",
        str(args.cache_dir),
        "--cache-model-key",
        str(args.cache_model_key),
        "--parallel-workers",
        "1",
        "--worker-count",
        str(worker_count),
        "--worker-index",
        str(worker_index),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.quiet:
        command.append("--quiet")
    if args.overwrite:
        command.append("--overwrite")
    return command


def _belongs_to_worker(cache_id: str, args: argparse.Namespace) -> bool:
    worker_count = int(getattr(args, "worker_count", 1))
    worker_index = int(getattr(args, "worker_index", 0))
    if worker_count <= 1:
        return True
    digest = hashlib.sha1(str(cache_id).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % worker_count
    return bucket == worker_index


def _pretrain_dataset_tag(annotation_file: str) -> str:
    name = Path(annotation_file).name.lower()
    if name.startswith("rxr_"):
        return "RxR"
    if "gemini" in name:
        return "Gemini"
    if "prevalent" in name:
        return "Prevalent"
    return "R2R"


def _cache_complete(
    scene_id: str,
    cache_id: str,
    dataset_key: str,
    split: str,
    args: argparse.Namespace,
) -> bool:
    cognitive_map_path = llm_navigation_cognitive_map_path(
        scene_id,
        cache_id,
        dataset_key,
        split,
        cache_dir=args.cache_dir,
        model_key=args.cache_model_key,
    )
    return cognitive_map_path.exists()


def _print_resume_summary(
    label: str,
    *,
    total: int,
    cached: int,
    pending: int,
) -> None:
    print(f"cache_resume {label}: total={total} cached={cached} pending={pending}")


def _level_local_start_position(
    trajectory_keypoints: Sequence[Sequence[float]],
    start_position: Sequence[float],
) -> Tuple[float, float]:
    if trajectory_keypoints:
        return float(trajectory_keypoints[0][0]), float(trajectory_keypoints[0][1])
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
        f"LLM-Navigation cache warning for {item['example_id']} at {stage}: {error}",
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
