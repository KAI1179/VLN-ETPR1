#!/usr/bin/env python3
"""Derive ablated GT raster namespaces for training-free map ablations.

Each episode keeps its metadata (direction vectors, start); only ``grid`` is
rewritten.  The derived namespace is ``<namespace>.abl_<mode>[_s<seed>]`` and
is passed as MODEL.MAP_ENCODER.cache_namespace (eval_gt_namespace.sh).

Modes
  occupancy_only    union of all channels -> 1.0 in the single 'other' object
                    channel, everything else 0: route-corridor geometry, no
                    semantics.
  cell_shuffle      one random permutation of the 100x100 cell positions,
                    applied to every channel: per-category counts and per-cell
                    co-occurrence kept, all geometry destroyed.
  regions_only      object channels zeroed.
  objects_only      region channels zeroed.
  category_scramble per-episode random permutation of the object channels and
                    of the region channels (both non-identity): geometry and
                    counts kept, category identity destroyed.
  unmentioned_only  channels of categories the instruction mentions zeroed
                    (complement of make_mentioned_only_gt_cache.py; needs the
                    spacy extractor, run in etpr1-uv).

    python scripts/distill/make_ablated_gt_cache.py --mode occupancy_only
"""

from __future__ import annotations

import sys
from pathlib import Path

import gzip
import json
import os
from typing import Literal, Optional, Sequence

import numpy as np
from tap import Tap

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prior import R2R_DIR  # noqa: E402
from prior.constants import MAPPED_OBJECT_OTHER_INDEX, OBJECT_CATEGORIES  # noqa: E402
from vlnce_baselines.models.etp_prior_gt.map_utils import (  # noqa: E402
    VLNCE_COGNITIVE_MAP_DIR,
    cognitive_map_cache_path,
)

Mode = Literal[
    "occupancy_only",
    "cell_shuffle",
    "regions_only",
    "objects_only",
    "category_scramble",
    "unmentioned_only",
]
RANDOM_MODES = {"cell_shuffle", "category_scramble"}
NUM_CHANNELS = 37


class Args(Tap):
    mode: Mode
    namespace: str = "gt.legacy.r1p5.direction5.v1"
    split: str = "val_unseen"
    seed: int = 0
    overwrite: bool = False


def derived_namespace(namespace: str, mode: str, seed: int) -> str:
    suffix = f"_s{seed}" if mode in RANDOM_MODES else ""
    return f"{namespace}.abl_{mode}{suffix}"


def _non_identity_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(n)
    while n > 1 and np.array_equal(order, np.arange(n)):
        order = rng.permutation(n)
    return order


def ablate(grid: np.ndarray, mode: str, rng: np.random.Generator, mentioned: Optional[np.ndarray]) -> np.ndarray:
    channels, rows, cols = grid.shape
    if channels != NUM_CHANNELS:
        raise ValueError(f"expected {NUM_CHANNELS} channels, got {grid.shape}")
    if mode == "occupancy_only":
        out = np.zeros_like(grid)
        out[MAPPED_OBJECT_OTHER_INDEX] = (grid > 0).any(axis=0).astype(grid.dtype)
        return out
    if mode == "cell_shuffle":
        order = _non_identity_permutation(rows * cols, rng)
        return grid.reshape(channels, -1)[:, order].reshape(channels, rows, cols)
    if mode == "regions_only":
        out = grid.copy()
        out[:OBJECT_CATEGORIES] = 0
        return out
    if mode == "objects_only":
        out = grid.copy()
        out[OBJECT_CATEGORIES:] = 0
        return out
    if mode == "category_scramble":
        objects = _non_identity_permutation(OBJECT_CATEGORIES, rng)
        regions = _non_identity_permutation(channels - OBJECT_CATEGORIES, rng) + OBJECT_CATEGORIES
        return grid[np.concatenate([objects, regions])]
    if mode == "unmentioned_only":
        assert mentioned is not None
        return grid * (~mentioned)[:, None, None].astype(grid.dtype)
    raise ValueError(f"unknown mode {mode}")


def main(argv: Optional[Sequence[str]] = None) -> str:
    args = Args(underscores_to_dashes=True).parse_args(argv)
    target_ns = derived_namespace(args.namespace, args.mode, args.seed)
    target_root = VLNCE_COGNITIVE_MAP_DIR / target_ns
    summary_path = target_root / f"summary_{args.split}.json"
    if summary_path.is_file() and not args.overwrite:
        print(f"exists: {summary_path}")
        print(target_ns)
        return target_ns

    extract_categories = None
    if args.mode == "unmentioned_only":
        from prior.grid_map._cognitive import extract_categories  # spacy

    with gzip.open(R2R_DIR / args.split / f"{args.split}.json.gz", "rt", encoding="utf-8") as file:
        episodes = json.load(file)["episodes"]
    source_boxes = VLNCE_COGNITIVE_MAP_DIR / args.namespace / "boxes"
    target_boxes = target_root / "boxes"
    if source_boxes.exists() and not target_boxes.exists():
        target_root.mkdir(parents=True, exist_ok=True)
        target_boxes.symlink_to(source_boxes.resolve(), target_is_directory=True)

    written = 0
    missing = 0
    nonzero_before = []
    nonzero_after = []
    for episode in episodes:
        cache_id = f"R2R_{args.split}_{episode['episode_id']}"
        source = cognitive_map_cache_path(episode["scene_id"], cache_id, namespace=args.namespace)
        if not source.is_file():
            missing += 1
            continue
        mentioned = None
        if extract_categories is not None:
            objects, regions = extract_categories(episode["instruction"]["instruction_text"])
            mentioned = np.zeros(NUM_CHANNELS, dtype=bool)
            mentioned[sorted(objects)] = True
            mentioned[[OBJECT_CATEGORIES + r for r in sorted(regions)]] = True
        with np.load(source, allow_pickle=True) as data:
            arrays = {name: data[name] for name in data.files}
        rng = np.random.default_rng([args.seed, int(episode["episode_id"])])
        grid = arrays["grid"]
        new_grid = ablate(grid, args.mode, rng, mentioned)
        nonzero_before.append(float((grid > 0).sum()))
        nonzero_after.append(float((new_grid > 0).sum()))
        arrays["grid"] = np.ascontiguousarray(new_grid, dtype=grid.dtype)
        out = cognitive_map_cache_path(episode["scene_id"], cache_id, namespace=target_ns)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(out)
        written += 1

    summary = {
        "namespace": target_ns,
        "source_namespace": args.namespace,
        "mode": args.mode,
        "seed": args.seed,
        "split": args.split,
        "episodes_written": written,
        "episodes_missing_source": missing,
        "mean_nonzero_cells_before": float(np.mean(nonzero_before)) if nonzero_before else 0.0,
        "mean_nonzero_cells_after": float(np.mean(nonzero_after)) if nonzero_after else 0.0,
    }
    target_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(target_ns)
    return target_ns


if __name__ == "__main__":
    main()
