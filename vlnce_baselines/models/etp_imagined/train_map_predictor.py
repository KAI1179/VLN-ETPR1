"""Offline trainer for instruction-to-cognitive-map prediction.

This script trains only InstructionCognitiveMapPredictor from paired VLN
instruction tokens and precomputed cognitive maps. It deliberately avoids
Habitat rollout and navigation loss.
"""

import argparse
import gzip
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.etp_imagined.instruction_map_predictor import (
    InstructionCognitiveMapPredictor,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import NUM_MAP_CATEGORIES, SIZE
from vlnce_baselines.models.etp_prior_gt.vlnbert_init import get_vlnbert_models


PAD_ID = 1
DATASET_FLAGS = {"r2r": "R2R", "rxr": "RxR"}
TASK_TYPE_IDS = {"r2r": 1, "rxr": 2}


@dataclass(frozen=True)
class PredictorExample:
    episode_id: str
    scene_id: str
    token_ids: List[int]
    map_path: Path
    dataset: str


def _read_json(path: Path) -> Dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def _scene_key(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(scene_id))[0]


def _infer_dataset(path: Path, explicit_dataset: Optional[str]) -> str:
    if explicit_dataset:
        dataset = explicit_dataset.lower()
    else:
        lower_path = str(path).lower()
        dataset = "rxr" if "rxr" in lower_path else "r2r"
    if dataset not in DATASET_FLAGS:
        raise ValueError(f"dataset must be one of {sorted(DATASET_FLAGS)}, got {dataset}")
    return dataset


def _map_path(cognitive_map_dir: Path, dataset: str, scene_id: str, episode_id: str) -> Path:
    return cognitive_map_dir / _scene_key(scene_id) / f"{DATASET_FLAGS[dataset]}_{episode_id}.npz"


def load_predictor_examples(
    dataset_paths: Sequence[Path],
    cognitive_map_dir: Path,
    dataset: Optional[str] = None,
    limit: Optional[int] = None,
    require_maps: bool = True,
) -> List[PredictorExample]:
    examples: List[PredictorExample] = []
    for dataset_path in dataset_paths:
        dataset_name = _infer_dataset(dataset_path, dataset)
        data = _read_json(dataset_path)
        for episode in data.get("episodes", []):
            instruction = episode.get("instruction", {})
            token_ids = instruction.get("instruction_tokens")
            if not token_ids:
                continue
            episode_id = str(episode["episode_id"])
            scene_id = episode["scene_id"]
            map_path = _map_path(cognitive_map_dir, dataset_name, scene_id, episode_id)
            if require_maps and not map_path.exists():
                continue
            examples.append(
                PredictorExample(
                    episode_id=episode_id,
                    scene_id=_scene_key(scene_id),
                    token_ids=[int(token_id) for token_id in token_ids],
                    map_path=map_path,
                    dataset=dataset_name,
                )
            )
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


class CognitiveMapPredictorDataset(Dataset):
    def __init__(self, examples: Sequence[PredictorExample]):
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("No examples with matching cognitive maps were found.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        with np.load(example.map_path) as data:
            grid = data["grid"].astype("float32")
        if grid.shape != (NUM_MAP_CATEGORIES, SIZE, SIZE):
            raise ValueError(f"Bad grid shape for {example.map_path}: {grid.shape}")
        return {
            "episode_id": example.episode_id,
            "scene_id": example.scene_id,
            "dataset": example.dataset,
            "token_ids": example.token_ids,
            "grid": torch.from_numpy(grid),
        }


def collate_predictor_batch(batch: Sequence[Dict], max_text_len: int, pad_id: int = PAD_ID) -> Dict:
    batch_size = len(batch)
    txt_ids = torch.full((batch_size, max_text_len), pad_id, dtype=torch.long)
    txt_task_encoding = torch.zeros((batch_size, max_text_len), dtype=torch.long)
    grids = torch.stack([item["grid"] for item in batch], dim=0)
    for i, item in enumerate(batch):
        tokens = item["token_ids"][:max_text_len]
        token_len = len(tokens)
        txt_ids[i, :token_len] = torch.tensor(tokens, dtype=torch.long)
        txt_task_encoding[i, :token_len] = TASK_TYPE_IDS[item["dataset"]]
    return {
        "txt_ids": txt_ids,
        "txt_task_encoding": txt_task_encoding,
        "txt_masks": txt_ids != pad_id,
        "grids": grids,
        "episode_ids": [item["episode_id"] for item in batch],
    }


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = dict(batch)
    for key in ("txt_ids", "txt_task_encoding", "txt_masks", "grids"):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def compute_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    mae = torch.mean(torch.abs(probs - target)).item()
    pred_mask = probs > 0.5
    target_mask = target > 0.5
    intersection = torch.logical_and(pred_mask, target_mask).sum().float()
    union = torch.logical_or(pred_mask, target_mask).sum().float()
    iou = (intersection / union.clamp_min(1.0)).item()
    return {"mae": mae, "iou@0.5": iou}


def _iterate_batches(
    dataloader: DataLoader,
    vln_bert: torch.nn.Module,
    predictor: InstructionCognitiveMapPredictor,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    predictor.train(is_train)
    total_loss = 0.0
    total_mae = 0.0
    total_iou = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = _move_batch(batch, device)
        with torch.no_grad():
            txt_embeds = vln_bert.forward_txt(
                batch["txt_ids"],
                batch["txt_task_encoding"],
                batch["txt_masks"],
            )
        logits = predictor(txt_embeds, batch["txt_masks"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["grids"], reduction="mean")
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = compute_metrics(logits.detach(), batch["grids"])
        total_loss += loss.item()
        total_mae += metrics["mae"]
        total_iou += metrics["iou@0.5"]
        total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "mae": 0.0, "iou@0.5": 0.0}
    return {
        "loss": total_loss / total_batches,
        "mae": total_mae / total_batches,
        "iou@0.5": total_iou / total_batches,
    }


def save_checkpoint(
    output_path: Path,
    predictor: InstructionCognitiveMapPredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_predictor = predictor.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "map_predictor": map_predictor,
            "state_dict": {f"map_predictor.{key}": value for key, value in map_predictor.items()},
            "optimizer": optimizer.state_dict(),
        },
        output_path,
    )


def _build_dataloader(
    dataset_paths: Sequence[Path],
    cognitive_map_dir: Path,
    dataset_name: Optional[str],
    max_text_len: int,
    batch_size: int,
    num_workers: int,
    limit: Optional[int],
    shuffle: bool,
) -> DataLoader:
    examples = load_predictor_examples(
        dataset_paths,
        cognitive_map_dir=cognitive_map_dir,
        dataset=dataset_name,
        limit=limit,
    )
    dataset = CognitiveMapPredictorDataset(examples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_predictor_batch(batch, max_text_len=max_text_len),
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", default="run_r2r/iter_train.yaml")
    parser.add_argument(
        "--train-dataset",
        nargs="+",
        type=Path,
        default=[Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train_90.json.gz")],
    )
    parser.add_argument("--val-dataset", nargs="+", type=Path, default=[Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz")])
    parser.add_argument("--dataset", choices=sorted(DATASET_FLAGS), default=None)
    parser.add_argument("--cognitive-map-dir", type=Path, default=Path("data/cognitive_maps"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/logs/checkpoints/release_r2r_imagined_predictor/store/predictor.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-text-len", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = get_config(args.exp_config, args.opts)
    max_text_len = args.max_text_len or config.IL.max_text_len
    device = torch.device(args.device)

    pretrained_path = getattr(config.MODEL, "pretrained_path", None)
    if pretrained_path and not Path(pretrained_path).exists():
        raise FileNotFoundError(
            f"MODEL.pretrained_path does not exist: {pretrained_path}. "
            "Pass a valid VLN checkpoint with --opts MODEL.pretrained_path <path>."
        )

    vln_bert = get_vlnbert_models(config.MODEL, dropout_rate=0.0).to(device)
    vln_bert.eval()
    for param in vln_bert.parameters():
        param.requires_grad_(False)

    predictor = InstructionCognitiveMapPredictor(
        hidden_size=vln_bert.config.hidden_size,
        num_heads=vln_bert.config.num_attention_heads,
        num_layers=2,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=0.01)

    train_loader = _build_dataloader(
        dataset_paths=args.train_dataset,
        cognitive_map_dir=args.cognitive_map_dir,
        dataset_name=args.dataset,
        max_text_len=max_text_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
        shuffle=True,
    )
    val_loader = None
    if args.val_dataset:
        val_loader = _build_dataloader(
            dataset_paths=args.val_dataset,
            cognitive_map_dir=args.cognitive_map_dir,
            dataset_name=args.dataset,
            max_text_len=max_text_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.val_limit,
            shuffle=False,
        )

    print(f"Train examples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val examples: {len(val_loader.dataset)}")

    best_metric = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _iterate_batches(train_loader, vln_bert, predictor, device, optimizer)
        metrics = {"train_" + key: value for key, value in train_metrics.items()}
        if val_loader is not None:
            val_metrics = _iterate_batches(val_loader, vln_bert, predictor, device)
            metrics.update({"val_" + key: value for key, value in val_metrics.items()})
            selection_metric = val_metrics["loss"]
        else:
            selection_metric = train_metrics["loss"]
        if epoch % args.log_every == 0:
            metric_text = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            print(f"epoch={epoch} {metric_text}")
        save_checkpoint(args.output, predictor, optimizer, epoch, metrics, args)
        if selection_metric < best_metric:
            best_metric = selection_metric
            save_checkpoint(args.output.with_suffix(".best.pt"), predictor, optimizer, epoch, metrics, args)


if __name__ == "__main__":
    main()
