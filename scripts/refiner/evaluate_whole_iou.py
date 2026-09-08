#!/usr/bin/env python3
"""Evaluate frozen-refiner whole-map IoU against cognitive-map targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from tap import Tap
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from prior.constants import MAPPED_OBJECT_NAMES
from vlnce_baselines.models.refiner.dataset import (
    OBJECT_CHANNELS,
    RefinerDataset,
    trajectory_files,
)
from vlnce_baselines.models.refiner.model import CognitiveMapRefiner

BROAD_OBJECT_NAMES = frozenset(("void", "structure", "other", "free-space"))


class Arguments(Tap):
    data_root: Path = Path("data/refiner")
    checkpoint: Path = Path("data/refiner/checkpoints_aug/best.pt")
    output: Path = Path("reports/refiner/whole_map_iou.json")
    device: str = "cpu"
    batch_size: int = 8
    num_workers: int = 8


class _WholeMapCounts:
    def __init__(self, object_indices: Sequence[int]) -> None:
        self.object_indices = list(object_indices)
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = {
            source: {
                resolution: {
                    name: 0
                    for metric in (
                        "object_semantic",
                        "object_occupancy",
                        "region_semantic",
                        "combined_semantic",
                    )
                    for name in (f"{metric}_intersection", f"{metric}_union")
                }
                for resolution in ("100x100", "10x10")
            }
            for source in ("p0", "refined")
        }

    def update(
        self,
        p0: torch.Tensor,
        refined: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        target = target >= 0.5
        for source, prediction in (("p0", p0 >= 0.5), ("refined", refined >= 0.5)):
            self._update_resolution(source, "100x100", prediction, target)
            self._update_resolution(
                source,
                "10x10",
                F.max_pool2d(prediction.float(), 10).bool(),
                F.max_pool2d(target.float(), 10).bool(),
            )

    def _update_resolution(
        self,
        source: str,
        resolution: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        object_prediction = prediction[:, self.object_indices]
        object_target = target[:, self.object_indices]
        region_prediction = prediction[:, OBJECT_CHANNELS:]
        region_target = target[:, OBJECT_CHANNELS:]
        combined_prediction = torch.cat((object_prediction, region_prediction), dim=1)
        combined_target = torch.cat((object_target, region_target), dim=1)
        pairs = {
            "object_semantic": (object_prediction, object_target),
            "object_occupancy": (
                object_prediction.any(dim=1),
                object_target.any(dim=1),
            ),
            "region_semantic": (region_prediction, region_target),
            "combined_semantic": (combined_prediction, combined_target),
        }
        counts = self.counts[source][resolution]
        for metric, (metric_prediction, metric_target) in pairs.items():
            counts[f"{metric}_intersection"] += int(
                (metric_prediction & metric_target).sum()
            )
            counts[f"{metric}_union"] += int(
                (metric_prediction | metric_target).sum()
            )

    def result(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        result: Dict[str, Dict[str, Dict[str, float]]] = {}
        for source, resolutions in self.counts.items():
            result[source] = {}
            for resolution, counts in resolutions.items():
                result[source][resolution] = {}
                for metric in (
                    "object_semantic",
                    "object_occupancy",
                    "region_semantic",
                    "combined_semantic",
                ):
                    intersection = counts[f"{metric}_intersection"]
                    union = counts[f"{metric}_union"]
                    result[source][resolution][f"{metric}_iou"] = (
                        intersection / union if union else 0.0
                    )
        return result


def _final_step_dataset(split_dir: Path) -> Subset:
    dataset = RefinerDataset(
        trajectory_files(split_dir, teacher_only=True),
        all_steps=True,
    )
    indices: List[int] = []
    for index, (path, _) in enumerate(dataset.samples):
        if index + 1 == len(dataset.samples) or dataset.samples[index + 1][0] != path:
            indices.append(index)
    return Subset(dataset, indices)


@torch.no_grad()
def _evaluate_split(
    model: CognitiveMapRefiner,
    dataset: Subset,
    args: Arguments,
    object_indices: Sequence[int],
) -> Dict[str, object]:
    counts = _WholeMapCounts(object_indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    for batch in loader:
        inputs = batch["x"].to(args.device)
        p0 = batch["p0"].to(args.device)
        observed = batch["obs"].to(args.device)
        target = batch["y"].to(args.device)
        output = torch.sigmoid(model(inputs))
        refined = torch.where(observed[:, None], output, p0)
        counts.update(p0, refined, target)
    return {"episodes": len(dataset), "metrics": counts.result()}


def main() -> None:
    args = Arguments(underscores_to_dashes=True).parse_args()
    object_indices = [
        index
        for index, name in enumerate(MAPPED_OBJECT_NAMES)
        if name not in BROAD_OBJECT_NAMES
    ]
    model = CognitiveMapRefiner().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()
    split_results = {
        split: _evaluate_split(
            model,
            _final_step_dataset(args.data_root / split),
            args,
            object_indices,
        )
        for split in ("val_seen", "val_unseen")
    }
    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluation_step": "final",
        "threshold": 0.5,
        "excluded_object_categories": sorted(BROAD_OBJECT_NAMES),
        **split_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
