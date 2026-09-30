#!/usr/bin/env python3
"""Preflight for coordinate fusion: do graph positions land on the raster?

Maps each episode's reference path (world coordinates) into the raster with
the same transform the trainer uses (map_origin_xz + world_to_map_cells) and
reports how often the path falls on a cell that the map labels with any
region.  A GT map must score high under the configured unit and clearly
lower under the wrong unit or with x/z swapped; otherwise the frame is wrong
and coordinate fusion would bias attention towards the wrong tokens.  For an
LLM map the same number measures how well its regions sit under the true path.

Examples:
    python scripts/distill/check_map_frame.py --source gt \
        --namespace gt.online121c369.r1p5.direction5.v1 --meters-per-unit 0.5
    python scripts/distill/check_map_frame.py --source llm --meters-per-unit 1.0
"""

from __future__ import annotations

import gzip
import json
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from tap import Tap

from prior import R2R_DIR
from vlnce_baselines.models.etp_llm.navigation import llm_cached_cognitive_map_to_tensors
from vlnce_baselines.models.etp_prior_gt.map_utils import (
    SIZE,
    cached_cognitive_map_to_tensors,
    map_origin_xz,
    world_to_map_cells,
)

REGION_CHANNELS = slice(27, 37)


class Args(Tap):
    source: Literal["gt", "llm"]
    meters_per_unit: float
    """MAP_ENCODER.start_position_meters_per_unit to check (1.0 metres, 0.5 cells)."""
    namespace: str = "gt.online121c369.r1p5.direction5.v1"
    """GT cache namespace (source gt)."""
    llm_model_key: str = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
    split: str = "val_unseen"
    limit: int = 300


def _load_map(args: Args, episode: dict) -> Optional[dict]:
    cache_id = f"R2R_{args.split}_{episode['episode_id']}"
    try:
        if args.source == "gt":
            return cached_cognitive_map_to_tensors(
                episode["scene_id"], cache_id, namespace=args.namespace,
                metadata_schema="direction5",
            )
        return llm_cached_cognitive_map_to_tensors(
            episode["scene_id"], cache_id, "R2R", args.split,
            metadata_schema="direction5", model_key=args.llm_model_key,
        )
    except FileNotFoundError:
        return None


def _region_hit_rate(grid: torch.Tensor, cells: torch.Tensor) -> float:
    index = torch.floor(cells).long()
    inside = ((index >= 0) & (index < SIZE)).all(dim=-1)
    if not bool(inside.any()):
        return 0.0
    regions = grid[REGION_CHANNELS]
    hits = regions[:, index[inside, 0], index[inside, 1]].amax(dim=0) > 0
    return float(hits.sum()) / float(len(cells))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = Args(underscores_to_dashes=True).parse_args(argv)
    path = R2R_DIR / args.split / f"{args.split}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as file:
        episodes = json.load(file)["episodes"]
    variants = {
        f"configured unit {args.meters_per_unit}": (args.meters_per_unit, False),
        f"unit x2 ({args.meters_per_unit * 2})": (args.meters_per_unit * 2, False),
        f"unit /2 ({args.meters_per_unit / 2})": (args.meters_per_unit / 2, False),
        "configured unit, x/z swapped": (args.meters_per_unit, True),
    }
    rates: dict[str, list[float]] = {name: [] for name in variants}
    start_outside = 0
    used = 0
    for episode in episodes:
        if used >= args.limit:
            break
        cognitive_map = _load_map(args, episode)
        if cognitive_map is None:
            continue
        used += 1
        grid = cognitive_map["grid"].float()
        reference = np.asarray(episode["reference_path"], dtype=np.float64)
        for name, (meters_per_unit, swap) in variants.items():
            origin = map_origin_xz(
                episode["start_position"], cognitive_map["start_position"], meters_per_unit
            )
            cells = world_to_map_cells(reference, origin)
            if swap:
                cells = cells.flip(-1)
            rates[name].append(_region_hit_rate(grid, cells))
        origin = map_origin_xz(
            episode["start_position"], cognitive_map["start_position"], args.meters_per_unit
        )
        start = world_to_map_cells([episode["start_position"]], origin)[0]
        start_outside += int(not bool(((start >= 0) & (start < SIZE)).all()))

    print(f"source={args.source} episodes={used} start_outside_raster={start_outside}")
    for name, values in rates.items():
        print(f"  {name:32s} path-on-region rate {np.mean(values) * 100:6.2f}%")


if __name__ == "__main__":
    main()
