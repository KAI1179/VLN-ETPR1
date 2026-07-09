"""Analyze LLM-Grid target text samples from cognitive-map raster caches."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from tap import Tap

from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES, OBJECT_CATEGORIES

TokenCounter = Callable[[str], int]


class SampleArgs(Tap):
    cache_root: Path = Path("data/cognitive_maps")
    """Root cognitive-map cache directory."""
    namespace: str
    """Cache namespace, such as gt.bbox.r1p5.path5.v1."""
    count: int
    """Number of raster samples to write."""
    output_root: Path = Path("outputs/llm_grid_samples")
    """Directory where a timestamped run directory is created."""
    scale: int = 1
    """Downsample factor: 1 or 2."""
    seed: int = 0
    """Random seed for deterministic sampling."""
    tokenizer_path: Optional[Path] = None
    """Optional local Hugging Face tokenizer path for token counts."""
    max_new_tokens: int = 4096
    """Candidate LLM-Grid generation budget used for over-budget reporting."""


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


def _format_value(value: float) -> str:
    return np.format_float_positional(
        np.float32(value),
        trim="-",
        fractional=False,
    )


def _cell_record(row: int, col: int, value: float) -> List[Union[int, float]]:
    cell: List[Union[int, float]] = [int(row), int(col)]
    if not np.isclose(value, 1.0):
        cell.append(float(_format_value(value)))
    return cell


def serialize_grid_target(
    grid: NDArray[np.float32],
    scale: int = 1,
    mentioned_objects: Optional[set[int]] = None,
    mentioned_regions: Optional[set[int]] = None,
) -> str:
    """Serialize nonzero category cells as keyed compact JSON LLM-Grid text."""
    sampled = downsample_grid(grid, scale)
    mentioned_objects = mentioned_objects or set()
    mentioned_regions = mentioned_regions or set()
    objects: Dict[str, Dict[str, Any]] = {}
    regions: Dict[str, Dict[str, Any]] = {}

    for category in range(sampled.shape[0]):
        row_cols = np.argwhere(sampled[category] > 0)
        if len(row_cols) == 0:
            continue
        cells = [
            _cell_record(int(row), int(col), float(sampled[category, row, col]))
            for row, col in row_cols
        ]
        if category < OBJECT_CATEGORIES:
            name = MAPPED_OBJECT_NAMES[category]
            objects[name] = {
                "cells": cells,
                "mentioned": category in mentioned_objects,
            }
        else:
            region_id = category - OBJECT_CATEGORIES
            name = MAPPED_REGION_NAMES[region_id]
            regions[name] = {
                "cells": cells,
                "mentioned": region_id in mentioned_regions,
            }

    return json.dumps(
        {
            "region_candidates": list(regions),
            "object_candidates": list(objects),
            "regions": regions,
            "objects": objects,
        },
        separators=(",", ":"),
    )


def _grid_stats(
    grid: NDArray[np.float32],
    text: str,
    token_count: int,
) -> Dict[str, int]:
    positive = grid > 0
    category_positive = positive.reshape(positive.shape[0], -1).any(axis=1)
    return {
        "text_length": len(text),
        "token_count": token_count,
        "positive_cell_count": int(positive.sum()),
        "category_count": int(category_positive.sum()),
    }


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def _load_token_counter(tokenizer_path: Optional[Path]) -> TokenCounter:
    if tokenizer_path is None:
        return whitespace_token_count

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def analyze_grid_sample(
    npz_path: Path,
    scale: int = 1,
    token_counter: TokenCounter = whitespace_token_count,
) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        grid = data["grid"]
    sampled = downsample_grid(grid, scale)
    target_text = serialize_grid_target(grid, scale=scale)
    return {
        "sample_id": npz_path.stem,
        "path": str(npz_path),
        "scale": scale,
        "target_text": target_text,
        "stats": _grid_stats(sampled, target_text, token_counter(target_text)),
    }


def _raster_paths(cache_root: Path, namespace: str) -> List[Path]:
    raster_root = cache_root / namespace / "raster"
    if not raster_root.exists():
        raise FileNotFoundError(f"Missing raster namespace: {raster_root}")
    return sorted(raster_root.glob("*/*.npz"))


def _sample_paths(paths: Sequence[Path], count: int, seed: int) -> List[Path]:
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count > len(paths):
        raise ValueError(f"count {count} exceeds available raster files {len(paths)}")
    sampled = list(paths)
    random.Random(seed).shuffle(sampled)
    return sorted(sampled[:count])


def _summarize(
    samples: Iterable[Dict[str, Any]],
    max_new_tokens: int,
) -> Dict[str, Any]:
    rows = list(samples)
    stat_names = ("text_length", "token_count", "positive_cell_count", "category_count")
    stats: Dict[str, Dict[str, float]] = {}
    for name in stat_names:
        values = [row["stats"][name] for row in rows]
        stats[name] = {
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": mean(values) if values else 0.0,
        }
    over_budget = [
        row for row in rows
        if row["stats"]["token_count"] > max_new_tokens
    ]
    return {
        "sample_count": len(rows),
        "max_new_tokens": max_new_tokens,
        "target_over_budget_count": len(over_budget),
        "target_over_budget_rate": len(over_budget) / len(rows) if rows else 0.0,
        "stats": stats,
    }


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_sample_run(
    cache_root: Path,
    namespace: str,
    count: int,
    output_root: Path,
    scale: int = 1,
    seed: int = 0,
    tokenizer_path: Optional[Path] = None,
    max_new_tokens: int = 4096,
) -> Path:
    token_counter = _load_token_counter(tokenizer_path)
    paths = _sample_paths(_raster_paths(cache_root, namespace), count, seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = output_root / f"{timestamp}-{namespace}-n{count}-s{scale}"
    output_dir.mkdir(parents=True, exist_ok=False)

    samples = [
        analyze_grid_sample(path, scale=scale, token_counter=token_counter)
        for path in paths
    ]
    manifest = {
        "cache_root": str(cache_root),
        "namespace": namespace,
        "count": count,
        "available_count": len(_raster_paths(cache_root, namespace)),
        "scale": scale,
        "seed": seed,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "token_count_mode": "hf" if tokenizer_path is not None else "whitespace",
        "max_new_tokens": max_new_tokens,
    }

    _write_json(output_dir / "manifest.json", manifest)
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, sort_keys=True) + "\n")
    _write_json(output_dir / "summary.json", _summarize(samples, max_new_tokens))
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = SampleArgs(underscores_to_dashes=True).parse_args(argv)
    output_dir = write_sample_run(
        args.cache_root,
        args.namespace,
        args.count,
        args.output_root,
        scale=args.scale,
        seed=args.seed,
        tokenizer_path=args.tokenizer_path,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
