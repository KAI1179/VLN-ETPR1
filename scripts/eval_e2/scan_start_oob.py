#!/usr/bin/env python3
"""Scan cached start positions with the exact coordinate-fusion transform."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(ROOT))

from prior import R2R_DIR  # noqa: E402
from vlnce_baselines.models.etp_llm.navigation import (  # noqa: E402
    llm_navigation_cognitive_map_raster_path,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import (  # noqa: E402
    SIZE,
    cognitive_map_cache_path,
    map_origin_xz,
    world_to_map_cells,
)

KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
NAMESPACE = "gt.legacy.r1p5.direction5.v1"
OOB_FILE = ROOT / "scripts/eval_e2/oob_val_unseen.txt"


def scan(source, split):
    with gzip.open(R2R_DIR / split / f"{split}.json.gz", "rt") as handle:
        episodes = json.load(handle)["episodes"]
    outside = []
    coords = []
    missing = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        cache_id = f"R2R_{split}_{episode_id}"
        path = (
            llm_navigation_cognitive_map_raster_path(
                episode["scene_id"], cache_id, "R2R", split, model_key=KEY
            )
            if source == "llm"
            else cognitive_map_cache_path(
                episode["scene_id"], cache_id, namespace=NAMESPACE
            )
        )
        if not path.is_file():
            missing.append(episode_id)
            continue
        with np.load(path, allow_pickle=True) as archive:
            start_position = torch.as_tensor(archive["start_position"].astype(np.float64))
        origin = map_origin_xz(episode["start_position"], start_position, 1.0)
        cell = world_to_map_cells([episode["start_position"]], origin)[0].numpy()
        coords.append(cell)
        if not np.all((cell >= 0) & (cell < SIZE)):
            outside.append(episode_id)
    array = np.stack(coords)
    print(
        f"{source} {split}: total={len(episodes)} present={len(coords)} "
        f"missing={len(missing)} outside={len(outside)} "
        f"rate_present={len(outside) / len(coords):.4%}"
    )
    for axis, label in enumerate(("row", "col")):
        print(
            f"  {label} min/p01/median/p99/max "
            f"{np.percentile(array[:, axis], [0, 1, 50, 99, 100]).round(2).tolist()}"
        )
    if source == "llm" and split == "val_unseen":
        OOB_FILE.write_text("\n".join(outside) + "\n")
        print(f"  wrote {OOB_FILE}")


if __name__ == "__main__":
    for split_name in ("train", "val_unseen"):
        for source_name in ("llm", "gt"):
            scan(source_name, split_name)
