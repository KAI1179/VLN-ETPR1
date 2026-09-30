#!/usr/bin/env python3
"""Derive an LLM-Navigation cache whose raster comes from another source.

Every episode keeps its own metadata (direction vectors, start direction,
start position) and its own status/prediction files; only the 37x100x100
``grid`` array is replaced. The derived cache lives under a new model key next
to the source key, so evaluation needs no code change: pass the printed key as
``MODEL.MAP_ENCODER.llm_cache_model_key``.

Modes
  cross_scene    grid of a random episode from a different scene. Removes every
                 episode- and scene-specific raster content while keeping the
                 raster's marginal distribution (a realistic LLM map, not zeros).
  within_scene   grid of a random episode from the same scene but a different
                 R2R trajectory_id (paraphrases of the same path are excluded,
                 since their LLM maps are near copies).
  block_permute  the episode's own grid with its 10x10 blocks of 10x10 cells
                 permuted. The map encoder has no positional encoding, so the
                 policy is analytically invariant to this; an eval that moves
                 by more than float noise means a position path exists.

Donors are drawn only from episodes whose source cache is complete, with a
fixed seed; the assignment is written to donor_assignment.csv.

Example:
    python scripts/distill/make_raster_donor_cache.py --mode cross_scene
"""

from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
from tap import Tap

from prior import R2R_DIR
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_cache_complete,
    llm_navigation_cognitive_map_raster_path,
    llm_navigation_split_dir,
)

Mode = Literal["cross_scene", "within_scene", "block_permute"]
TOKEN_GRID = 10  # spatial_tokenizer: kernel 10, stride 10 on a 100x100 raster


class Args(Tap):
    mode: Mode
    """cross_scene | within_scene | block_permute"""
    source_model_key: str = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
    """LLM-Navigation cache key whose metadata is kept."""
    split: str = "val_unseen"
    seed: int = 0
    cache_dir: Optional[str] = None
    """LLM-Navigation cache root; default data/llm_navigation."""
    overwrite: bool = False


def derived_model_key(source_model_key: str, mode: str, seed: int) -> str:
    return f"{source_model_key}__raster-{mode}-s{seed}"


class _Episode:
    __slots__ = ("episode_id", "scene_id", "trajectory_id", "cache_id")

    def __init__(self, episode_id: str, scene_id: str, trajectory_id: int, split: str):
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.trajectory_id = trajectory_id
        self.cache_id = f"R2R_{split}_{episode_id}"


def _load_episodes(split: str) -> List[_Episode]:
    path = R2R_DIR / split / f"{split}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as file:
        raw = json.load(file)["episodes"]
    return [
        _Episode(str(ep["episode_id"]), ep["scene_id"], int(ep["trajectory_id"]), split)
        for ep in raw
    ]


def _scene_of(episode: _Episode) -> str:
    return Path(episode.scene_id).stem


def assign_donors(
    episodes: Sequence[_Episode], mode: Mode, seed: int
) -> Dict[str, _Episode]:
    """Map target episode_id -> donor episode (sampled with replacement)."""
    rng = np.random.default_rng(seed)
    ordered = sorted(episodes, key=lambda ep: int(ep.episode_id))
    by_scene: Dict[str, List[_Episode]] = defaultdict(list)
    for ep in ordered:
        by_scene[_scene_of(ep)].append(ep)
    donors: Dict[str, _Episode] = {}
    for target in ordered:
        if mode == "block_permute":
            donors[target.episode_id] = target
            continue
        if mode == "cross_scene":
            pool = [ep for ep in ordered if _scene_of(ep) != _scene_of(target)]
        else:
            pool = [
                ep
                for ep in by_scene[_scene_of(target)]
                if ep.trajectory_id != target.trajectory_id
            ]
        if not pool:
            raise ValueError(
                f"no {mode} donor for episode {target.episode_id} "
                f"(scene {_scene_of(target)})"
            )
        donors[target.episode_id] = pool[int(rng.integers(len(pool)))]
    return donors


def block_permute(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    channels, rows, cols = grid.shape
    if rows % TOKEN_GRID or cols % TOKEN_GRID:
        raise ValueError(
            f"grid {grid.shape} is not divisible into {TOKEN_GRID}x{TOKEN_GRID} blocks"
        )
    br, bc = rows // TOKEN_GRID, cols // TOKEN_GRID
    blocks = (
        grid
        .reshape(channels, TOKEN_GRID, br, TOKEN_GRID, bc)
        .transpose(0, 1, 3, 2, 4)
        .reshape(channels, TOKEN_GRID * TOKEN_GRID, br, bc)
    )
    order = rng.permutation(TOKEN_GRID * TOKEN_GRID)
    while np.array_equal(order, np.arange(order.size)):
        order = rng.permutation(TOKEN_GRID * TOKEN_GRID)
    return (
        blocks[:, order]
        .reshape(channels, TOKEN_GRID, TOKEN_GRID, br, bc)
        .transpose(0, 1, 3, 2, 4)
        .reshape(channels, rows, cols)
    )


def _link(source: Path, target: Path) -> None:
    if source.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main(argv: Optional[Sequence[str]] = None) -> str:
    args = Args(underscores_to_dashes=True).parse_args(argv)
    key = derived_model_key(args.source_model_key, args.mode, args.seed)
    source_dir = llm_navigation_split_dir(
        "R2R", args.split, args.cache_dir, args.source_model_key
    )
    target_dir = llm_navigation_split_dir("R2R", args.split, args.cache_dir, key)
    if not (source_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            f"missing source manifest: {source_dir / 'manifest.json'}"
        )
    if (target_dir / "summary.json").is_file() and not args.overwrite:
        print(f"exists: {target_dir}")
        print(key)
        return key

    episodes = [
        ep
        for ep in _load_episodes(args.split)
        if llm_navigation_cache_complete(
            ep.scene_id,
            ep.cache_id,
            "R2R",
            args.split,
            cache_dir=args.cache_dir,
            model_key=args.source_model_key,
        )
    ]
    donors = assign_donors(episodes, args.mode, args.seed)

    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        cache_model_key=key,
        derived_from_model_key=args.source_model_key,
        raster_donor_mode=args.mode,
        raster_donor_seed=args.seed,
    )
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for name in ("status", "predictions"):
        _link(source_dir / name, target_dir / name)
    _link(
        source_dir / "cognitive_maps" / "boxes", target_dir / "cognitive_maps" / "boxes"
    )

    rows = []
    identical = 0
    for target in episodes:
        donor = donors[target.episode_id]
        own_path = llm_navigation_cognitive_map_raster_path(
            target.scene_id,
            target.cache_id,
            "R2R",
            args.split,
            cache_dir=args.cache_dir,
            model_key=args.source_model_key,
        )
        donor_path = llm_navigation_cognitive_map_raster_path(
            donor.scene_id,
            donor.cache_id,
            "R2R",
            args.split,
            cache_dir=args.cache_dir,
            model_key=args.source_model_key,
        )
        out_path = llm_navigation_cognitive_map_raster_path(
            target.scene_id,
            target.cache_id,
            "R2R",
            args.split,
            cache_dir=args.cache_dir,
            model_key=key,
        )
        with np.load(own_path, allow_pickle=True) as own_file:
            arrays = {name: own_file[name] for name in own_file.files}
        own_grid = arrays["grid"]
        if args.mode == "block_permute":
            episode_rng = np.random.default_rng([args.seed, int(target.episode_id)])
            new_grid = block_permute(own_grid, episode_rng)
        else:
            with np.load(donor_path, allow_pickle=True) as donor_file:
                new_grid = donor_file["grid"]
        if new_grid.shape != own_grid.shape or new_grid.dtype != own_grid.dtype:
            raise ValueError(
                f"grid mismatch {target.episode_id}<-{donor.episode_id}: "
                f"{new_grid.shape}/{new_grid.dtype} vs {own_grid.shape}/{own_grid.dtype}"
            )
        same_grid = bool(np.array_equal(new_grid, own_grid))
        identical += same_grid
        arrays["grid"] = new_grid
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(out_path)
        rows.append({
            "episode_id": target.episode_id,
            "donor_episode_id": donor.episode_id,
            "scene": _scene_of(target),
            "donor_scene": _scene_of(donor),
            "same_scene": int(_scene_of(target) == _scene_of(donor)),
            "same_trajectory": int(target.trajectory_id == donor.trajectory_id),
            "grid_identical": int(same_grid),
        })

    with (target_dir / "donor_assignment.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n = len(rows)
    summary = {
        "model_key": key,
        "source_model_key": args.source_model_key,
        "mode": args.mode,
        "seed": args.seed,
        "split": args.split,
        "episodes": n,
        "same_scene_rate": sum(r["same_scene"] for r in rows) / n,
        "same_trajectory_rate": sum(r["same_trajectory"] for r in rows) / n,
        "grid_identical_rate": identical / n,
        "unique_donors": len({r["donor_episode_id"] for r in rows}),
    }
    (target_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(key)
    return key


if __name__ == "__main__":
    main()
