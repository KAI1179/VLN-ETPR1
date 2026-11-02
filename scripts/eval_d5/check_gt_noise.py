#!/usr/bin/env python3
"""CPU validation of a proposed GT-noise namespace on train_90."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from tap import Tap

REPO = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
GT = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster")
TRAIN = REPO / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train_90.json.gz"
REGION = slice(27, 37)


class Args(Tap):
    candidate_namespace: str
    """Absolute path to a new, already-built GT-noise namespace."""


def read(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["grid"][REGION] > 0


def main() -> None:
    args = Args().parse_args()
    candidate = Path(args.candidate_namespace)
    assert candidate.is_absolute() and candidate != GT.parent
    with gzip.open(TRAIN, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    ious = []
    precision = []
    recall = []
    matched = 0
    for episode in episodes:
        scene = Path(episode["scene_id"]).stem
        name = f"R2R_train_{episode['episode_id']}.npz"
        own = candidate / "raster" / scene / name
        truth = GT / scene / name
        if not own.is_file() or not truth.is_file():
            continue
        pred = read(own)
        gt = read(truth)
        matched += 1
        p_present = pred.reshape(10, -1).any(axis=1)
        g_present = gt.reshape(10, -1).any(axis=1)
        tp = int((p_present & g_present).sum())
        if p_present.any():
            precision.append(tp / int(p_present.sum()))
        if g_present.any():
            recall.append(tp / int(g_present.sum()))
        union = np.logical_or(pred, gt).sum(axis=(1, 2))
        intersection = np.logical_and(pred, gt).sum(axis=(1, 2))
        ious.extend((intersection[union > 0] / union[union > 0]).tolist())
    assert matched == len(episodes), (matched, len(episodes))
    print(json.dumps({
        "matched": matched,
        "region_iou_mean": float(np.mean(ious)),
        "region_presence_precision": float(np.mean(precision)),
        "region_presence_recall": float(np.mean(recall)),
        "target": {"region_iou_mean": 0.3339, "presence_precision": 0.9192,
                   "presence_recall": 0.9128},
    }, indent=2))


if __name__ == "__main__":
    main()
