#!/usr/bin/env python3
"""Train the visual-evidence cognitive-map refiner."""

from __future__ import annotations

import json
import random
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import torch
from tap import Tap
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from prior.constants import MAPPED_OBJECT_COLORS, MAPPED_REGION_COLORS
from vlnce_baselines.models.refiner.dataset import RefinerDataset, trajectory_files
from vlnce_baselines.models.refiner.model import CognitiveMapRefiner

OBJECT_CHANNELS = 27


class Arguments(Tap):
    data_root: Path = Path("data/refiner")
    output_dir: Path = Path("data/refiner/checkpoints")
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    num_workers: int = 8
    seed: int = 0
    device: str = "cuda"
    augment: bool = False


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

        self._update_pooled_resolution(predictions, target_mask, native_groups)

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

    def _update_pooled_resolution(
        self,
        predictions: Dict[str, torch.Tensor],
        target: torch.Tensor,
        groups: Dict[str, torch.Tensor],
    ) -> None:
        for source, prediction in predictions.items():
            for group, cell_mask in groups.items():
                mask = cell_mask[:, None]
                pooled_prediction = F.max_pool2d(
                    (prediction & mask).float(), 10
                ).bool()
                pooled_target = F.max_pool2d(
                    (target & mask).float(), 10
                ).bool()
                counts = self.counts[source]["10x10"][group]
                counts["intersection"] += (
                    pooled_prediction & pooled_target
                ).sum(dim=(0, 2, 3)).cpu()
                counts["union"] += (
                    pooled_prediction | pooled_target
                ).sum(dim=(0, 2, 3)).cpu()

    def result(self) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        for resolution in ("100x100", "10x10"):
            for statistic in ("intersection", "union"):
                p0 = self.counts["p0"][resolution]["unseen"][statistic]
                refined = self.counts["refined"][resolution]["unseen"][statistic]
                if not torch.equal(
                    p0[:OBJECT_CHANNELS], refined[:OBJECT_CHANNELS]
                ):
                    raise AssertionError(
                        f"BUG: unseen object {statistic} differs at {resolution}"
                    )
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
    composed[:, :OBJECT_CHANNELS] = torch.where(
        observed,
        output[:, :OBJECT_CHANNELS],
        p0[:, :OBJECT_CHANNELS],
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
        unseen = (~observed)[:, None].expand(-1, OBJECT_CHANNELS, -1, -1)
        if not torch.equal(
            refined[:, :OBJECT_CHANNELS][unseen],
            p0[:, :OBJECT_CHANNELS][unseen],
        ):
            raise AssertionError("BUG: refined unseen object cells differ from P0")
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


def _argmax_labels(
    grid: torch.Tensor,
    start: int,
    end: int,
    offset: int,
) -> np.ndarray:
    values, labels = grid[start:end].max(dim=0)
    rendered = torch.zeros_like(labels)
    present = values >= 0.5
    rendered[present] = labels[present] + offset
    return rendered.cpu().numpy()


@torch.no_grad()
def _save_val_unseen_examples(
    model: CognitiveMapRefiner,
    paths: List[Path],
    args: Arguments,
) -> None:
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    selected: List[Path] = []
    scenes = set()
    for path in paths:
        if path.parent.name not in scenes:
            selected.append(path)
            scenes.add(path.parent.name)
        if len(selected) == 3:
            break

    report_dir = Path("reports/refiner")
    report_dir.mkdir(parents=True, exist_ok=True)
    colormap = colors.ListedColormap(
        ["#000000"] + list(MAPPED_OBJECT_COLORS) + list(MAPPED_REGION_COLORS)
    )
    norm = colors.BoundaryNorm(np.arange(-0.5, 38.5), colormap.N)
    model.eval()
    for path in selected:
        dataset = RefinerDataset([path], all_steps=True)
        sample = dataset[len(dataset) - 1]
        p0 = sample["p0"].to(args.device)
        output = torch.sigmoid(model(sample["x"][None].to(args.device)))[0]
        refined = _compose_refined(
            output[None],
            p0[None],
            sample["obs"][None].to(args.device),
        )[0]
        target = sample["y"].to(args.device)
        grids = (p0, refined, target)
        figure, axes = plt.subplots(2, 3, figsize=(12, 8))
        for column, (title, grid) in enumerate(
            zip(("P0", "refined", "GT"), grids)
        ):
            axes[0, column].imshow(
                _argmax_labels(grid, 0, OBJECT_CHANNELS, 1),
                cmap=colormap,
                norm=norm,
                origin="lower",
            )
            axes[0, column].set_title(f"{title} objects")
            axes[1, column].imshow(
                _argmax_labels(grid, OBJECT_CHANNELS, 37, 28),
                cmap=colormap,
                norm=norm,
                origin="lower",
            )
            axes[1, column].set_title(f"{title} regions")
            axes[0, column].axis("off")
            axes[1, column].axis("off")
        with np.load(path, allow_pickle=True) as trajectory:
            episode_id = str(trajectory["episode_id"].item())
        output_path = report_dir / f"val_unseen_{episode_id}_last.png"
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        print(f"wrote {output_path}", flush=True)


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
        augment=args.augment,
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
    pos_weight_paths = random.Random(args.seed).sample(train_paths, 2000)
    pos_weight_dataset = RefinerDataset(pos_weight_paths, all_steps=True)
    print(
        f"train_trajectories={len(train_paths)} "
        f"train_samples_per_epoch={len(train_dataset)} "
        f"pos_weight_trajectories={len(pos_weight_paths)} "
        f"pos_weight_samples={len(pos_weight_dataset)} "
        f"val_seen_samples={len(val_seen_dataset)} "
        f"val_unseen_samples={len(val_unseen_dataset)}",
        flush=True,
    )
    object_pos_weight = _object_pos_weight(pos_weight_dataset, args).to(
        args.device
    )
    print("object_pos_weight", object_pos_weight.cpu().tolist(), flush=True)

    model = CognitiveMapRefiner().to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    history: List[Dict] = []
    best_score = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        epoch_start = perf_counter()
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
        epoch_seconds = perf_counter() - epoch_start
        score = val_unseen["refined"]["100x100"]["route_seen"]["object_miou"]
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "seconds": epoch_seconds,
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
            "augment": args.augment,
            "object_pos_weight": object_pos_weight.cpu().tolist(),
            "pos_weight_trajectories": len(pos_weight_paths),
            "pos_weight_samples": len(pos_weight_dataset),
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
            f"val_unseen_route_seen_object_miou={score:.6f} "
            f"seconds={epoch_seconds:.2f}",
            flush=True,
        )

    model.load_state_dict(
        torch.load(args.output_dir / "best.pt", map_location=args.device)
    )
    _save_val_unseen_examples(
        model,
        trajectory_files(args.data_root / "val_unseen", teacher_only=True),
        args,
    )


if __name__ == "__main__":
    main()
