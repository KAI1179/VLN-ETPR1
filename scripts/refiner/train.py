#!/usr/bin/env python3
"""Train the visual-evidence cognitive-map refiner."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tap import Tap
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from vlnce_baselines.models.refiner.dataset import RefinerDataset, trajectory_files
from vlnce_baselines.models.refiner.model import CognitiveMapRefiner

OBJECT_CHANNELS = 27


class Arguments(Tap):
    data_root: Path = Path("data/refiner")
    output_dir: Path = Path("data/refiner/checkpoints")
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"


def _loader(
    dataset: RefinerDataset,
    args: Arguments,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _object_pos_weight(
    dataset: RefinerDataset,
    args: Arguments,
) -> torch.Tensor:
    positive = torch.zeros(OBJECT_CHANNELS, dtype=torch.float64)
    negative = torch.zeros(OBJECT_CHANNELS, dtype=torch.float64)
    for batch in _loader(dataset, args, shuffle=False):
        target = batch["y"][:, :OBJECT_CHANNELS]
        observed = batch["obs"][:, None]
        positive += ((target > 0) & observed).sum(dim=(0, 2, 3))
        negative += ((target == 0) & observed).sum(dim=(0, 2, 3))
    return (negative / positive.clamp_min(1)).clamp(1, 20).float()


class _MetricAccumulator:
    def __init__(self) -> None:
        self.counts: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        for source in ("p0", "refined"):
            self.counts[source] = {}
            for resolution in ("100x100", "10x10"):
                self.counts[source][resolution] = {}
                for group in ("seen", "unseen", "route", "route_seen"):
                    self.counts[source][resolution][group] = {
                        "intersection": torch.zeros(37, dtype=torch.float64),
                        "union": torch.zeros(37, dtype=torch.float64),
                    }

    def update(
        self,
        p0: torch.Tensor,
        refined: torch.Tensor,
        target: torch.Tensor,
        observed: torch.Tensor,
        route: torch.Tensor,
    ) -> None:
        predictions = {"p0": p0 >= 0.5, "refined": refined >= 0.5}
        target_mask = target >= 0.5
        native_groups = {
            "seen": observed,
            "unseen": ~observed,
            "route": route,
            "route_seen": route & observed,
        }
        self._update_resolution(
            predictions,
            target_mask,
            native_groups,
            "100x100",
        )

        pooled_predictions = {
            name: F.max_pool2d(mask.float(), 10).bool()
            for name, mask in predictions.items()
        }
        pooled_target = F.max_pool2d(target_mask.float(), 10).bool()
        pooled_seen = F.max_pool2d(observed[:, None].float(), 10).squeeze(1).bool()
        pooled_route = F.max_pool2d(route[:, None].float(), 10).squeeze(1).bool()
        pooled_groups = {
            "seen": pooled_seen,
            "unseen": ~pooled_seen,
            "route": pooled_route,
            "route_seen": pooled_route & pooled_seen,
        }
        self._update_resolution(
            pooled_predictions,
            pooled_target,
            pooled_groups,
            "10x10",
        )

    def _update_resolution(
        self,
        predictions: Dict[str, torch.Tensor],
        target: torch.Tensor,
        groups: Dict[str, torch.Tensor],
        resolution: str,
    ) -> None:
        for source, prediction in predictions.items():
            for group, cell_mask in groups.items():
                mask = cell_mask[:, None]
                intersection = (prediction & target & mask).sum(dim=(0, 2, 3))
                union = ((prediction | target) & mask).sum(dim=(0, 2, 3))
                counts = self.counts[source][resolution][group]
                counts["intersection"] += intersection.cpu()
                counts["union"] += union.cpu()

    def result(self) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        for source, resolutions in self.counts.items():
            results[source] = {}
            for resolution, groups in resolutions.items():
                results[source][resolution] = {}
                for group, counts in groups.items():
                    intersection = counts["intersection"]
                    union = counts["union"]
                    iou = intersection / union.clamp_min(1)
                    object_valid = union[:OBJECT_CHANNELS] > 0
                    region_valid = union[OBJECT_CHANNELS:] > 0
                    results[source][resolution][group] = {
                        "object_miou": float(
                            iou[:OBJECT_CHANNELS][object_valid].mean()
                        ),
                        "region_miou": float(
                            iou[OBJECT_CHANNELS:][region_valid].mean()
                        ),
                    }
        return results


def _compose_refined(
    output: torch.Tensor,
    p0: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    composed = output.clone()
    observed = observed[:, None]
    composed[:, :OBJECT_CHANNELS] = (
        observed * output[:, :OBJECT_CHANNELS]
        + (~observed) * p0[:, :OBJECT_CHANNELS]
    )
    return composed


@torch.no_grad()
def _validate(
    model: CognitiveMapRefiner,
    dataset: RefinerDataset,
    args: Arguments,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    model.eval()
    metrics = _MetricAccumulator()
    for batch in _loader(dataset, args, shuffle=False):
        inputs = batch["x"].to(args.device, non_blocking=True)
        output = torch.sigmoid(model(inputs))
        p0 = batch["p0"].to(args.device, non_blocking=True)
        observed = batch["obs"].to(args.device, non_blocking=True)
        refined = _compose_refined(output, p0, observed)
        metrics.update(
            p0,
            refined,
            batch["y"].to(args.device, non_blocking=True),
            observed,
            batch["route"].to(args.device, non_blocking=True),
        )
    return metrics.result()


def _train_epoch(
    model: CognitiveMapRefiner,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
    object_pos_weight: torch.Tensor,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        inputs = batch["x"].to(device, non_blocking=True)
        target = batch["y"].to(device, non_blocking=True)
        observed = batch["obs"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(inputs)
        bce = F.binary_cross_entropy_with_logits(
            logits.float(), target, reduction="none"
        )
        weights = torch.ones_like(bce)
        weights[:, :OBJECT_CHANNELS] = observed[:, None]
        positive_weights = torch.ones_like(bce)
        positive_weights[:, :OBJECT_CHANNELS] = torch.where(
            target[:, :OBJECT_CHANNELS] > 0,
            object_pos_weight[None, :, None, None],
            1.0,
        )
        loss = (bce * weights * positive_weights).sum() / weights.sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach())
        total_batches += 1
    return total_loss / total_batches


def main() -> None:
    args = Arguments(underscores_to_dashes=True).parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = trajectory_files(args.data_root / "train")
    train_dataset = RefinerDataset(
        train_paths,
        seed=args.seed,
    )
    val_seen_dataset = RefinerDataset(
        trajectory_files(args.data_root / "val_seen", teacher_only=True),
        all_steps=True,
    )
    val_unseen_dataset = RefinerDataset(
        trajectory_files(args.data_root / "val_unseen", teacher_only=True),
        all_steps=True,
    )
    object_pos_weight = _object_pos_weight(
        RefinerDataset(train_paths, all_steps=True), args
    ).to(args.device)
    print("object_pos_weight", object_pos_weight.cpu().tolist(), flush=True)

    model = CognitiveMapRefiner().to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    history: List[Dict] = []
    best_score = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        train_loss = _train_epoch(
            model,
            _loader(train_dataset, args, shuffle=True),
            optimizer,
            scaler,
            object_pos_weight,
            args.device,
        )
        scheduler.step()
        val_seen = _validate(model, val_seen_dataset, args)
        val_unseen = _validate(model, val_unseen_dataset, args)
        score = val_unseen["refined"]["100x100"]["route_seen"]["object_miou"]
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "lr": scheduler.get_last_lr()[0],
            "val_seen": val_seen,
            "val_unseen": val_unseen,
        }
        history.append(epoch_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        torch.save(model.state_dict(), args.output_dir / "last.pt")
        metrics = {
            "object_pos_weight": object_pos_weight.cpu().tolist(),
            "best_epoch": best_epoch,
            "best_val_unseen_route_seen_object_miou": best_score,
            "epochs": history,
        }
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        print(
            f"epoch={epoch + 1} loss={train_loss:.6f} "
            f"val_unseen_route_seen_object_miou={score:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
