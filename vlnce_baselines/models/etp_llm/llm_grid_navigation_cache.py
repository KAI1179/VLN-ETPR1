"""Generate precomputed LLM-Grid navigation raster caches."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from tap import Tap

from .llm_boxes_navigation_cache import (
    PRETRAIN_DATASET_KEY,
    PRETRAIN_SPLIT,
    VLNCE_SPLITS,
    _aggregate_worker_metrics,
    _batch_count,
    _default_device,
    _failure_detail,
    _generation_kwargs,
    _iter_collated_batches,
    _load_causal_lm_model_and_tokenizer,
    _model_batch,
    _model_uses_device_map,
    _normalize_device_map,
    _progress,
    _resolve_parallel_worker_count,
    _skipped_cache_count,
    _visible_cuda_devices,
    _warn_navigation_cache_failure,
    _write_navigation_cache_status_record,
    _write_prediction_text,
    _write_split_metrics,
    load_pretrain_cache_items,
    load_vlnce_cache_items,
)
from .navigation import (
    DEFAULT_LLM_NAVIGATION_MODEL_KEY,
    llm_navigation_cache_complete,
    llm_navigation_cognitive_map_raster_path,
    llm_navigation_prediction_path,
    llm_navigation_split_dir,
)
from .llm_boxes_train import decode_generated_completion
from .llm_grid_train import (
    DEFAULT_MODEL_NAME_OR_PATH,
    GRID_CHANNELS,
    GRID_SCALE,
    GRID_SHAPE,
    collate_llm_grid_prompt_batch,
    load_system_prompt,
    parse_grid_text,
)

GRID_VLNCE_DATASETS: Tuple[Literal["R2R"], ...] = ("R2R",)


class LLMGridNavigationCacheArgs(Tap):
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH
    """Pretrained or checkpoint path for the causal language model."""
    max_input_length: int = 1024
    max_new_tokens: int = 2048
    batch_size: int = 1
    limit: Optional[int] = None
    device: str = ""
    device_map: str = "auto"
    quiet: bool = False
    cache_dir: str = ""
    cache_model_key: str = DEFAULT_LLM_NAVIGATION_MODEL_KEY
    parallel_workers: str = "auto"
    scale: int = GRID_SCALE
    worker_count: int = 1
    worker_index: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("underscores_to_dashes", True)
        super().__init__(*args, **kwargs)

    def process_args(self) -> None:
        if not self.device:
            self.device = _default_device()
        if self.scale != GRID_SCALE:
            raise ValueError("LLM-Grid navigation cache generation supports scale=2")


def llm_grid_navigation_cache(
    model: Any,
    tokenizer: Any,
    dataset: Iterable[Any],
    args: Any,
    *,
    dataset_key: str,
    split: str,
) -> Dict[str, float]:
    """Generate LLM-Grid prediction text and direction5 raster caches."""
    import torch

    if hasattr(model, "to") and not _model_uses_device_map(model):
        model.to(args.device)
    if hasattr(model, "eval"):
        model.eval()

    system_prompt = load_system_prompt(scale=args.scale)
    split_dir = llm_navigation_split_dir(
        dataset_key,
        split,
        cache_dir=args.cache_dir,
        model_key=args.cache_model_key,
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    _write_grid_navigation_cache_manifest(
        split_dir,
        args,
        system_prompt,
        dataset_key,
        split,
    )

    examples = 0
    cached = 0

    def pending_items() -> Iterable[Any]:
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
        lambda batch: collate_llm_grid_prompt_batch(
            batch,
            tokenizer,
            system_prompt,
            args.max_input_length,
        ),
    )
    progress_loader = _progress(
        loader,
        desc=f"generate LLM-Grid navigation cache {dataset_key}/{split}",
        quiet=args.quiet,
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
            generated_sequences, split_retries = _generate_with_oom_splitting(
                model,
                _model_batch(batch, args.device, include_labels=False),
                _generation_kwargs(tokenizer, args.max_new_tokens),
            )
            oom_split_retries += split_retries
            decoded = [
                decode_generated_completion(tokenizer, sequence, prompt_length)
                for sequence, prompt_length in zip(
                    generated_sequences,
                    batch["prompt_lengths"],
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
                raster_path = llm_navigation_cognitive_map_raster_path(
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
                    parsed = parse_grid_text(generated_text, shape=GRID_SHAPE)
                    strict_valid += 1
                    _write_direction5_raster(
                        raster_path,
                        grid=_scale2_grid_to_full(parsed.grid),
                        direction_vectors=parsed.direction_vectors,
                        start_direction=item["start_direction"],
                        start_position=item["start_position"],
                    )
                except Exception as exc:
                    skipped += 1
                    failures.append(_failure_detail("grid_parse_or_write", exc))
                    _warn_navigation_cache_failure(
                        item,
                        "grid_parse_or_write",
                        exc,
                    )
                    _write_navigation_cache_status_record(
                        scene_id=scene_id,
                        cache_id=cache_id,
                        dataset_key=dataset_key,
                        split=split,
                        args=args,
                        status="grid_parse_or_write_failed",
                        prediction_path=prediction_path,
                        cognitive_map_raster_path=raster_path,
                        strict_valid=False,
                        error=str(exc),
                        failures=failures,
                    )
                    continue

                _write_navigation_cache_status_record(
                    scene_id=scene_id,
                    cache_id=cache_id,
                    dataset_key=dataset_key,
                    split=split,
                    args=args,
                    status="complete",
                    prediction_path=prediction_path,
                    cognitive_map_raster_path=raster_path,
                    strict_valid=True,
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


def generate_all_grid_navigation_caches(
    model: Any,
    tokenizer: Any,
    args: Any,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    pretrain_items = load_pretrain_cache_items(
        limit=args.limit,
        quiet=args.quiet,
        args=args,
    )
    metrics[f"{PRETRAIN_DATASET_KEY}/{PRETRAIN_SPLIT}"] = (
        llm_grid_navigation_cache(
            model,
            tokenizer,
            pretrain_items,
            args,
            dataset_key=PRETRAIN_DATASET_KEY,
            split=PRETRAIN_SPLIT,
        )
    )
    for dataset_key in GRID_VLNCE_DATASETS:
        for split in VLNCE_SPLITS:
            items = load_vlnce_cache_items(
                dataset_key,
                split,
                limit=args.limit,
                quiet=args.quiet,
                args=args,
            )
            metrics[f"{dataset_key.lower()}/{split}"] = llm_grid_navigation_cache(
                model,
                tokenizer,
                items,
                args,
                dataset_key=dataset_key,
                split=split,
            )
    return metrics


def _generate_with_oom_splitting(
    model: Any,
    model_inputs: Dict[str, Any],
    generation_kwargs: Dict[str, Any],
) -> Tuple[List[Any], int]:
    import torch

    batch_size = len(model_inputs["input_ids"])
    try:
        return list(model.generate(**model_inputs, **generation_kwargs)), 0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if batch_size <= 1:
            raise
        midpoint = batch_size // 2
        warnings.warn(
            f"CUDA OOM during generation; splitting batch of {batch_size} into "
            f"{midpoint} and {batch_size - midpoint}",
            RuntimeWarning,
            stacklevel=2,
        )
        left, left_retries = _generate_with_oom_splitting(
            model,
            _slice_model_inputs(model_inputs, 0, midpoint),
            generation_kwargs,
        )
        right, right_retries = _generate_with_oom_splitting(
            model,
            _slice_model_inputs(model_inputs, midpoint, batch_size),
            generation_kwargs,
        )
        return left + right, 1 + left_retries + right_retries


def _slice_model_inputs(
    model_inputs: Dict[str, Any],
    start: int,
    end: int,
) -> Dict[str, Any]:
    return {key: value[start:end] for key, value in model_inputs.items()}


def parse_args(argv: Optional[Sequence[str]] = None) -> LLMGridNavigationCacheArgs:
    return LLMGridNavigationCacheArgs().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    args = parse_args(argv)
    worker_count = _resolve_parallel_worker_count(args.parallel_workers)
    if worker_count > 1 and args.worker_count == 1:
        return _run_parallel_workers(args, worker_count)

    model, tokenizer = _load_causal_lm_model_and_tokenizer(
        args.model_name_or_path,
        device_map=_normalize_device_map(args.device_map),
    )
    tokenizer.padding_side = "left"
    metrics = generate_all_grid_navigation_caches(model, tokenizer, args)
    skipped = _skipped_cache_count(metrics)
    print(f"skipped={skipped}")
    if skipped:
        raise SystemExit(1)
    return metrics


def _scale2_grid_to_full(grid: NDArray[np.float32]) -> NDArray[np.float32]:
    if tuple(grid.shape) != GRID_SHAPE:
        raise ValueError(f"scale-2 parsed grid must have shape {GRID_SHAPE}")
    upsampled = np.repeat(np.repeat(grid, 2, axis=1), 2, axis=2)
    expected_shape = (GRID_CHANNELS, 100, 100)
    if tuple(upsampled.shape) != expected_shape:
        raise ValueError(f"upsampled grid must have shape {expected_shape}")
    return np.asarray(upsampled, dtype=np.float32)


def _write_direction5_raster(
    path: Path,
    *,
    grid: NDArray[np.float32],
    direction_vectors: NDArray[np.float32],
    start_direction: Sequence[float],
    start_position: Sequence[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        grid=np.asarray(grid, dtype=np.float32),
        range_y=np.asarray([None, None], dtype=object),
        direction_vectors=np.asarray(direction_vectors, dtype=np.float32),
        start_direction_vector=np.asarray(start_direction, dtype=np.float32),
        start_position=np.asarray(start_position, dtype=np.float32),
    )


def _cache_complete(
    scene_id: str,
    cache_id: str,
    dataset_key: str,
    split: str,
    args: Any,
) -> bool:
    return llm_navigation_cache_complete(
        scene_id,
        cache_id,
        dataset_key,
        split,
        cache_dir=args.cache_dir,
        model_key=args.cache_model_key,
    )


def _write_grid_navigation_cache_manifest(
    split_dir: Path,
    args: Any,
    system_prompt: str,
    dataset_key: str,
    split: str,
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_key,
        "split": split,
        "generator": "llm-grid",
        "scale": args.scale,
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


def _run_parallel_workers(
    args: LLMGridNavigationCacheArgs,
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
        raise SystemExit(f"LLM-Grid navigation cache workers failed: {failed}")
    metrics: Dict[str, Dict[str, float]] = {}
    for dataset_key, split in _grid_cache_split_keys():
        split_dir = llm_navigation_split_dir(
            dataset_key,
            split,
            cache_dir=args.cache_dir,
            model_key=args.cache_model_key,
        )
        metrics[f"{dataset_key.lower()}/{split}"] = _aggregate_worker_metrics(
            split_dir
        )
    return metrics


def _grid_cache_split_keys() -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = [
        (dataset_key, split)
        for dataset_key in GRID_VLNCE_DATASETS
        for split in VLNCE_SPLITS
    ]
    keys.append((PRETRAIN_DATASET_KEY, PRETRAIN_SPLIT))
    return keys


def _worker_command(
    args: LLMGridNavigationCacheArgs,
    worker_count: int,
    worker_index: int,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "vlnce_baselines.models.etp_llm.llm_grid_navigation_cache",
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
        "--scale",
        str(args.scale),
        "--worker-count",
        str(worker_count),
        "--worker-index",
        str(worker_index),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.quiet:
        command.append("--quiet")
    return command


if __name__ == "__main__":
    main()
