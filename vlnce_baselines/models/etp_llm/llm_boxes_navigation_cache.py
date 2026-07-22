"""Generate precomputed LLM-Navigation cognitive-map caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import warnings
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
    build_llm_map_input,
    parse_llm_boxes_text,
    spec_to_llm_boxes_text,
    spec_to_relevant_semantic_boxes,
)
from .generation import generate_with_oom_splitting
from .navigation import (
    DEFAULT_LLM_NAVIGATION_MODEL_KEY,
    ensure_llm_navigation_manifest,
    llm_navigation_cache_complete,
    llm_navigation_cognitive_map_boxes_path,
    llm_navigation_cognitive_map_raster_path,
    llm_navigation_prediction_path,
    llm_navigation_split_dir,
    llm_navigation_status_path,
)
from .llm_boxes_train import (
    DEFAULT_MODEL_NAME_OR_PATH,
    LLMBoxesItem,
    collate_llm_boxes_prompt_batch,
    decode_generated_completion,
    load_system_prompt,
    _batch_count,
    _default_device,
    _generation_kwargs,
    _iter_collated_batches,
    _load_causal_lm_model_and_tokenizer,
    _model_batch,
    _model_uses_device_map,
    _normalize_device_map,
    _progress,
)

BOXES_VLNCE_DATASETS: Tuple[Literal["R2R"], ...] = ("R2R",)
VLNCE_SPLITS = ("train", "val_seen", "val_unseen")
PRETRAIN_DATASET_KEY = "pretrain"
PRETRAIN_SPLIT = "mixed"


def llm_boxes_navigation_cache(
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
    tokenizer.padding_side = "left"

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
            if _cache_complete(
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
    generated = 0
    skipped = 0
    oom_split_retries = 0

    with torch.inference_mode():
        for batch in progress_loader:
            attempted += len(batch["items"])
            model_inputs = _model_batch(batch, args.device, include_labels=False)
            # input_ids/attention_mask: (B, T_prompt).
            # generated_sequences: (B, T_prompt + T_generated).
            generated_sequences, split_retries = generate_with_oom_splitting(
                model,
                model_inputs,
                _generation_kwargs(tokenizer, args.max_new_tokens),
            )
            oom_split_retries += split_retries
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
                cognitive_map_boxes_path = llm_navigation_cognitive_map_boxes_path(
                    scene_id,
                    cache_id,
                    dataset_key,
                    split,
                    cache_dir=args.cache_dir,
                    model_key=args.cache_model_key,
                )
                cognitive_map_raster_path = llm_navigation_cognitive_map_raster_path(
                    scene_id,
                    cache_id,
                    dataset_key,
                    split,
                    cache_dir=args.cache_dir,
                    model_key=args.cache_model_key,
                )
                _write_prediction_text(prediction_path, generated_text)
                failures: List[Dict[str, Any]] = []

                try:
                    spec = parse_llm_boxes_text(generated_text)
                except Exception as exc:
                    skipped += 1
                    failures.append(_failure_detail("parse_failed", exc))
                    _warn_navigation_cache_failure(item, "parse_failed", exc)
                    _write_navigation_cache_status(
                        item,
                        status="parse_failed",
                        dataset_key=dataset_key,
                        split=split,
                        args=args,
                        prediction_path=prediction_path,
                        cognitive_map_boxes_path=cognitive_map_boxes_path,
                        cognitive_map_raster_path=cognitive_map_raster_path,
                        strict_valid=False,
                        error=str(exc),
                        failures=failures,
                    )
                    continue
                strict_valid += 1

                # Spec geometry remains level-local meters:
                # keypoints=(5, 2), object boxes=(center[2], half_extents[2], rot),
                # region boxes=(min[2], max[2]).
                try:
                    relevant = spec_to_relevant_semantic_boxes(
                        spec,
                        instruction=item["instruction"],
                        level_idx=item["level_idx"],
                        start_direction_vector=item["start_direction"],
                        range_y=item["target_relevant"].level.range_y,
                    )
                    cognitive_map_boxes_path.parent.mkdir(parents=True, exist_ok=True)
                    relevant.save(cognitive_map_boxes_path)

                    cognitive_map = relevant.to_cognitive_map()
                    cognitive_map_raster_path.parent.mkdir(parents=True, exist_ok=True)
                    cognitive_map.save(cognitive_map_raster_path)
                except Exception as exc:
                    skipped += 1
                    failures.append(_failure_detail("conversion_failed", exc))
                    _warn_navigation_cache_failure(item, "conversion_failed", exc)
                    _write_navigation_cache_status(
                        item,
                        status="conversion_failed",
                        dataset_key=dataset_key,
                        split=split,
                        args=args,
                        prediction_path=prediction_path,
                        cognitive_map_boxes_path=cognitive_map_boxes_path,
                        cognitive_map_raster_path=cognitive_map_raster_path,
                        strict_valid=True,
                        error=str(exc),
                        failures=failures,
                    )
                    continue
                _write_navigation_cache_status(
                    item,
                    status="complete",
                    dataset_key=dataset_key,
                    split=split,
                    args=args,
                    prediction_path=prediction_path,
                    cognitive_map_boxes_path=cognitive_map_boxes_path,
                    cognitive_map_raster_path=cognitive_map_raster_path,
                    strict_valid=True,
                    failures=failures,
                )
                generated += 1

    metrics = {
        "examples": float(examples),
        "cached": float(cached),
        "attempted": float(attempted),
        "strict_valid": float(strict_valid),
        "generated": float(generated),
        "skipped": float(skipped),
        "oom_split_retries": float(oom_split_retries),
        "strict_parse_failure_rate": (
            float(attempted - strict_valid) / float(attempted) if attempted else 0.0
        ),
    }
    _write_split_metrics(split_dir, metrics, args)
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
        raw_entries = PretrainAnnotationEntry.iter_from(
            annotation_file,
            english_only=True,
        )
        entries = _progress(
            raw_entries,
            desc=f"load pretrain cache items {annotation_file}",
            quiet=quiet,
            total=limit,
        )
        try:
            for entry in entries:
                if args is not None and not _belongs_to_worker(entry.instr_id, args):
                    continue
                total += 1
                if args is not None:
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
                    if args is not None:
                        _write_navigation_cache_status_record(
                            scene_id=entry.scan,
                            cache_id=entry.instr_id,
                            dataset_key=PRETRAIN_DATASET_KEY,
                            split=PRETRAIN_SPLIT,
                            args=args,
                            status="skipped_input",
                            error=str(error),
                        )
                    continue
                start_position = _level_local_start_position(
                    target_relevant.trajectory_keypoints,
                    positions[0],
                )
                target_spec = LLMBoxesSpec(objects=(), regions=())
                items.append(
                    {
                        "input_text": build_llm_map_input(
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
    require_navigation_cache: bool = True,
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
        if args is not None:
            if _cache_complete(
                episode.scene_id,
                episode.unique_id,
                dataset_key,
                split,
                args,
                require_navigation_cache=require_navigation_cache,
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
            if args is not None:
                _write_navigation_cache_status_record(
                    scene_id=episode.scene_id,
                    cache_id=episode.unique_id,
                    dataset_key=dataset_key,
                    split=split,
                    args=args,
                    status="skipped_input",
                    error=str(error),
                )
            continue
        start_position = _level_local_start_position(
            target_relevant.trajectory_keypoints,
            episode.start_position,
        )
        target_spec = LLMBoxesSpec(objects=(), regions=())
        items.append(
            {
                "input_text": build_llm_map_input(
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
    pretrain_items = load_pretrain_cache_items(
        limit=args.limit,
        quiet=args.quiet,
        args=args,
    )
    metrics[f"{PRETRAIN_DATASET_KEY}/{PRETRAIN_SPLIT}"] = llm_boxes_navigation_cache(
        model,
        tokenizer,
        pretrain_items,
        args,
        dataset_key=PRETRAIN_DATASET_KEY,
        split=PRETRAIN_SPLIT,
    )
    for dataset_key in BOXES_VLNCE_DATASETS:
        for split in VLNCE_SPLITS:
            examples = load_vlnce_cache_items(
                dataset_key,
                split,
                limit=args.limit,
                quiet=args.quiet,
                args=args,
            )
            metrics[f"{dataset_key.lower()}/{split}"] = llm_boxes_navigation_cache(
                model,
                tokenizer,
                examples,
                args,
                dataset_key=dataset_key,
                split=split,
            )

    return metrics


def _skipped_cache_count(metrics: Dict[str, Dict[str, float]]) -> int:
    return int(
        sum(split_metrics.get("skipped", 0.0) for split_metrics in metrics.values())
    )


def _cache_split_keys() -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = []
    for dataset_key in BOXES_VLNCE_DATASETS:
        for split in VLNCE_SPLITS:
            keys.append((dataset_key, split))
    keys.append((PRETRAIN_DATASET_KEY, PRETRAIN_SPLIT))
    return keys


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name-or-path",
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="Pretrained or checkpoint path for the causal language model.",
    )
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
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
        "--parallel-workers",
        default="auto",
        help=(
            "Number of cache-generation worker processes. Use auto to launch one "
            "worker per visible CUDA device; use 1 for single-process generation."
        ),
    )
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-shard-seed", default="", help=argparse.SUPPRESS)
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
        raise ValueError(
            "--parallel-workers must be auto or a positive integer"
        ) from error
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
    worker_shard_seed = _new_worker_shard_seed()
    print(
        f"parallel_workers={worker_count} visible_cuda_devices={devices or ['cpu']} "
        f"worker_shard_seed={worker_shard_seed}"
    )
    for worker_index in range(worker_count):
        command = _worker_command(
            args,
            worker_count,
            worker_index,
            worker_shard_seed,
        )
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
    metrics: Dict[str, Dict[str, float]] = {}
    for dataset_key, split in _cache_split_keys():
        split_dir = llm_navigation_split_dir(
            dataset_key,
            split,
            cache_dir=args.cache_dir,
            model_key=args.cache_model_key,
        )
        metrics[f"{dataset_key.lower()}/{split}"] = _aggregate_worker_metrics(
            split_dir,
            worker_count,
        )
    return metrics


def _worker_command(
    args: argparse.Namespace,
    worker_count: int,
    worker_index: int,
    worker_shard_seed: str,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "vlnce_baselines.models.etp_llm.llm_boxes_navigation_cache",
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
        "--worker-shard-seed",
        worker_shard_seed,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.quiet:
        command.append("--quiet")
    return command


def _belongs_to_worker(cache_id: str, args: argparse.Namespace) -> bool:
    worker_count = int(getattr(args, "worker_count", 1))
    worker_index = int(getattr(args, "worker_index", 0))
    if worker_count <= 1:
        return True
    shard_seed = str(getattr(args, "worker_shard_seed", ""))
    digest = hashlib.sha1(f"{shard_seed}\0{cache_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % worker_count
    return bucket == worker_index


def _new_worker_shard_seed() -> str:
    return secrets.token_hex(16)


def _cache_complete(
    scene_id: str,
    cache_id: str,
    dataset_key: str,
    split: str,
    args: argparse.Namespace,
    *,
    require_navigation_cache: bool = True,
) -> bool:
    if not require_navigation_cache:
        return llm_navigation_prediction_path(
            scene_id,
            cache_id,
            dataset_key,
            split,
            cache_dir=args.cache_dir,
            model_key=args.cache_model_key,
        ).is_file()
    return llm_navigation_cache_complete(
        scene_id,
        cache_id,
        dataset_key,
        split,
        cache_dir=args.cache_dir,
        model_key=args.cache_model_key,
    )


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
    ensure_llm_navigation_manifest(split_dir, {
        "dataset": dataset_key,
        "split": split,
        "model_name_or_path": args.model_name_or_path,
        "cache_model_key": args.cache_model_key,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
    })


def _write_split_metrics(
    split_dir: Path,
    metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    worker_count = int(getattr(args, "worker_count", 1))
    worker_index = int(getattr(args, "worker_index", 0))
    if worker_count > 1:
        metrics_path = split_dir / "worker_metrics" / f"worker_{worker_index}.json"
    else:
        metrics_path = split_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _aggregate_worker_metrics(
    split_dir: Path,
    worker_count: Optional[int] = None,
) -> Dict[str, float]:
    paths = (
        [
            split_dir / "worker_metrics" / f"worker_{index}.json"
            for index in range(worker_count)
        ]
        if worker_count is not None
        else sorted((split_dir / "worker_metrics").glob("worker_*.json"))
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing worker metrics: " + ", ".join(str(path) for path in missing)
        )
    worker_metrics = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    metrics = _sum_metrics(worker_metrics)
    (split_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _sum_metrics(metrics_by_worker: Sequence[Dict[str, float]]) -> Dict[str, float]:
    metrics = {
        "examples": 0.0,
        "cached": 0.0,
        "attempted": 0.0,
        "strict_valid": 0.0,
        "generated": 0.0,
        "skipped": 0.0,
    }
    for worker_metrics in metrics_by_worker:
        for key in metrics:
            metrics[key] += float(worker_metrics.get(key, 0.0))

    attempted = metrics["attempted"]
    metrics["strict_parse_failure_rate"] = (
        (attempted - metrics["strict_valid"]) / attempted if attempted else 0.0
    )
    if any("oom_split_retries" in item for item in metrics_by_worker):
        metrics["oom_split_retries"] = sum(
            float(item.get("oom_split_retries", 0.0)) for item in metrics_by_worker
        )
    return metrics


def _write_prediction_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "" if text.endswith("\n") else "\n"
    path.write_text(f"{text}{suffix}", encoding="utf-8")


def _write_navigation_cache_status(
    item: LLMBoxesItem,
    *,
    status: str,
    dataset_key: str,
    split: str,
    args: argparse.Namespace,
    prediction_path: Optional[Path] = None,
    cognitive_map_boxes_path: Optional[Path] = None,
    cognitive_map_raster_path: Optional[Path] = None,
    strict_valid: Optional[bool] = None,
    error: Optional[str] = None,
    failures: Sequence[Dict[str, Any]] = (),
) -> None:
    _write_navigation_cache_status_record(
        scene_id=item["scene_id"],
        cache_id=item["example_id"],
        dataset_key=dataset_key,
        split=split,
        args=args,
        status=status,
        prediction_path=prediction_path,
        cognitive_map_boxes_path=cognitive_map_boxes_path,
        cognitive_map_raster_path=cognitive_map_raster_path,
        strict_valid=strict_valid,
        error=error,
        failures=failures,
    )


def _write_navigation_cache_status_record(
    *,
    scene_id: str,
    cache_id: str,
    dataset_key: str,
    split: str,
    args: argparse.Namespace,
    status: str,
    prediction_path: Optional[Path] = None,
    cognitive_map_boxes_path: Optional[Path] = None,
    cognitive_map_raster_path: Optional[Path] = None,
    strict_valid: Optional[bool] = None,
    error: Optional[str] = None,
    failures: Sequence[Dict[str, Any]] = (),
) -> None:
    record: Dict[str, Any] = {
        "example_id": cache_id,
        "scene_id": scene_id,
        "dataset": dataset_key,
        "split": split,
        "status": status,
    }
    if prediction_path is not None:
        record["prediction_path"] = str(prediction_path)
    if cognitive_map_boxes_path is not None:
        record["cognitive_map_boxes_path"] = str(cognitive_map_boxes_path)
    if cognitive_map_raster_path is not None:
        record["cognitive_map_raster_path"] = str(cognitive_map_raster_path)
    if strict_valid is not None:
        record["strict_valid"] = strict_valid
    if error is not None:
        record["error"] = error
    if failures:
        record["failures"] = list(failures)

    status_path = llm_navigation_status_path(
        scene_id,
        cache_id,
        dataset_key,
        split,
        cache_dir=args.cache_dir,
        model_key=args.cache_model_key,
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(record, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _failure_detail(
    stage: str,
    error: BaseException | str,
    dropped_entities: Sequence[str] = (),
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"stage": stage, "error": str(error)}
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


if __name__ == "__main__":
    main()
