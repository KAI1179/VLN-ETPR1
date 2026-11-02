#!/usr/bin/env python3
"""Compare cached LLM and legacy-GT rasters on common R2R episodes (CPU)."""

from __future__ import annotations

import json
import gzip
from pathlib import Path

import numpy as np

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
LLM = ROOT / "data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r"
GT = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster")
OUT = ROOT / "reports/d3_region_quality_metrics.json"
GROUPS = {"region": list(range(27, 37)), "object": list(range(27)), "environment": [0, 15, 17]}


def raster(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["grid"] > 0


def score(split: str) -> dict:
    base = LLM / split / "cognitive_maps/raster"
    files = sorted(base.glob("*/*.npz"))
    if split == "train":
        with gzip.open(ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train_90.json.gz", "rt") as stream:
            allowed = {str(item["episode_id"]) for item in json.load(stream)["episodes"]}
        files = [path for path in files if path.stem.rsplit("_", 1)[-1] in allowed]
    rows = {group: {"iou": [], "precision": [], "recall": [], "coarse_iou": [], "centroid_m": [], "both_empty": 0}
            for group in GROUPS}
    per_region = {index: {"iou": [], "recall": [], "both_empty": 0, "gt_present": 0}
                  for index in GROUPS["region"]}
    matched = 0
    for source in files:
        target = GT / source.parent.name / source.name
        if not target.is_file():
            continue
        predicted = raster(source)
        truth = raster(target)
        matched += 1
        p_present = predicted.reshape(37, -1).any(-1)
        t_present = truth.reshape(37, -1).any(-1)
        p_coarse = predicted.reshape(37, 10, 10, 10, 10).any(axis=(2, 4))
        t_coarse = truth.reshape(37, 10, 10, 10, 10).any(axis=(2, 4))
        for group, channels in GROUPS.items():
            row = rows[group]
            tp = int(np.logical_and(p_present[channels], t_present[channels]).sum())
            pp = int(p_present[channels].sum())
            gp = int(t_present[channels].sum())
            if pp:
                row["precision"].append(tp / pp)
            if gp:
                row["recall"].append(tp / gp)
            for index in channels:
                p, t = predicted[index], truth[index]
                union = np.logical_or(p, t).sum()
                if union == 0:
                    row["both_empty"] += 1
                    if index in per_region:
                        per_region[index]["both_empty"] += 1
                    continue
                iou = float(np.logical_and(p, t).sum() / union)
                row["iou"].append(iou)
                coarse_union = np.logical_or(p_coarse[index], t_coarse[index]).sum()
                row["coarse_iou"].append(float(np.logical_and(p_coarse[index], t_coarse[index]).sum() / coarse_union))
                if index in per_region:
                    per_region[index]["iou"].append(iou)
                    if t_present[index]:
                        per_region[index]["gt_present"] += 1
                        per_region[index]["recall"].append(float(p_present[index]))
                if p_present[index] and t_present[index]:
                    pc = np.argwhere(p).mean(0)
                    tc = np.argwhere(t).mean(0)
                    row["centroid_m"].append(float(np.linalg.norm(pc - tc) * 0.5))
    def aggregate(row: dict) -> dict:
        result = {key: {"mean": float(np.mean(values)) if values else None,
                      "median": float(np.median(values)) if values else None,
                      "p75": float(np.percentile(values, 75)) if values else None,
                      "n": len(values)}
                  for key, values in row.items() if isinstance(values, list)}
        result.update({key: value for key, value in row.items() if not isinstance(value, list)})
        return result
    return {"llm_files": len(files), "matched": matched,
            "groups": {name: aggregate(row) for name, row in rows.items()},
            "regions": {str(index): aggregate(row) for index, row in per_region.items()}}


def main() -> None:
    result = {split: score(split) for split in ("val_unseen", "train")}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({split: {"matched": result[split]["matched"],
                              "region": result[split]["groups"]["region"]}
                      for split in result}, indent=2))


if __name__ == "__main__":
    main()
