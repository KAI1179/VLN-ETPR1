"""Offline trainer for instruction-to-cognitive-map prediction.

This script trains only InstructionCognitiveMapPredictor from paired VLN
instructions and on-the-fly cognitive-map targets. It deliberately avoids
Habitat rollout and navigation loss.
"""

import argparse
import gzip
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tap import Tap

from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.etp_imagined.instruction_map_predictor import (
    InstructionCognitiveMapPredictor,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import (
    DIRECTION_VECTOR_CNT,
    NUM_MAP_CATEGORIES,
    SIZE,
    build_cognitive_map,
    cognitive_map_to_tensors,
)
from vlnce_baselines.models.etp_prior_gt.vlnbert_init import get_vlnbert_models
from prior.directions import start_rotation_to_direction_vector


PAD_ID = 1
DATASET_FLAGS = {"r2r": "R2R", "rxr": "RxR"}
TASK_TYPE_IDS = {"r2r": 1, "rxr": 2}


@dataclass(frozen=True)
class PredictorExample:
    episode_id: str
    scene_id: str
    instruction_text: str
    token_ids: List[int]
    reference_path: List[List[float]]
    start_rotation: List[float]
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


def load_predictor_examples(
    dataset_paths: Sequence[Path],
    dataset: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[PredictorExample]:
    examples: List[PredictorExample] = []
    for dataset_path in dataset_paths:
        dataset_name = _infer_dataset(dataset_path, dataset)
        data = _read_json(dataset_path)
        for episode in data.get("episodes", []):
            instruction = episode.get("instruction", {})
            token_ids = instruction.get("instruction_tokens")
            instruction_text = instruction.get("instruction_text")
            reference_path = episode.get("reference_path")
            start_rotation = episode.get("start_rotation")
            if not token_ids or not instruction_text or not reference_path or start_rotation is None:
                continue
            episode_id = str(episode["episode_id"])
            scene_id = episode["scene_id"]
            examples.append(
                PredictorExample(
                    episode_id=episode_id,
                    scene_id=scene_id,
                    instruction_text=instruction_text,
                    token_ids=[int(token_id) for token_id in token_ids],
                    reference_path=reference_path,
                    start_rotation=start_rotation,
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
        cognitive_map = build_cognitive_map(
            example.scene_id,
            example.instruction_text,
            example.reference_path,
            start_rotation_to_direction_vector(example.start_rotation),
        )
        tensors = cognitive_map_to_tensors(cognitive_map)
        grid = tensors["grid"].float()
        direction_vectors = tensors["direction_vectors"].float()
        start_direction_vector = tensors["start_direction_vector"].float()
        start_position = tensors["start_position"].float()
        if grid.shape != (NUM_MAP_CATEGORIES, SIZE, SIZE):
            raise ValueError(f"Bad grid shape for episode {example.episode_id}: {tuple(grid.shape)}")
        if direction_vectors.shape != (DIRECTION_VECTOR_CNT, 2):
            raise ValueError(
                f"Bad direction_vectors shape for episode {example.episode_id}: "
                f"{tuple(direction_vectors.shape)}"
            )
        if start_direction_vector.shape != (2,):
            raise ValueError(
                f"Bad start_direction_vector shape for episode {example.episode_id}: "
                f"{tuple(start_direction_vector.shape)}"
            )
        if start_position.shape != (2,):
            raise ValueError(
                f"Bad start_position shape for episode {example.episode_id}: "
                f"{tuple(start_position.shape)}"
            )
        return {
            "episode_id": example.episode_id,
            "scene_id": _scene_key(example.scene_id),
            "dataset": example.dataset,
            "token_ids": example.token_ids,
            "grid": grid,
            "direction_vectors": direction_vectors,
            "start_direction_vector": start_direction_vector,
            "start_position": start_position,
        }


class CognitiveMapPredictionModel(torch.nn.Module):
    def __init__(
        self,
        vln_bert: torch.nn.Module,
        predictor: InstructionCognitiveMapPredictor,
    ) -> None:
        super().__init__()
        self.vln_bert = vln_bert
        self.predictor = predictor

    def forward(
        self,
        txt_ids: torch.Tensor,
        txt_task_encoding: torch.Tensor,
        txt_masks: torch.Tensor,
        start_direction_vectors: torch.Tensor,
        start_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            txt_embeds = self.vln_bert.forward_txt(txt_ids, txt_task_encoding, txt_masks)
        return self.predictor(
            txt_embeds,
            txt_masks,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
        )


def collate_predictor_batch(batch: Sequence[Dict], max_text_len: int, pad_id: int = PAD_ID) -> Dict:
    batch_size = len(batch)
    txt_ids = torch.full((batch_size, max_text_len), pad_id, dtype=torch.long)
    txt_task_encoding = torch.zeros((batch_size, max_text_len), dtype=torch.long)
    grids = torch.stack([item["grid"] for item in batch], dim=0)
    direction_vectors = torch.stack([item["direction_vectors"] for item in batch], dim=0)
    start_direction_vectors = torch.stack([item["start_direction_vector"] for item in batch], dim=0)
    start_positions = torch.stack([item["start_position"] for item in batch], dim=0)
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
        "direction_vectors": direction_vectors,
        "start_direction_vectors": start_direction_vectors,
        "start_positions": start_positions,
        "episode_ids": [item["episode_id"] for item in batch],
    }


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    moved = dict(batch)
    for key in (
        "txt_ids",
        "txt_task_encoding",
        "txt_masks",
        "grids",
        "direction_vectors",
        "start_direction_vectors",
        "start_positions",
    ):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def _parse_thresholds(thresholds: str) -> Tuple[float, ...]:
    parsed = tuple(float(item) for item in thresholds.split(",") if item)
    if not parsed:
        raise ValueError("at least one threshold is required")
    return parsed


def _topk_recall(probs: torch.Tensor, target_mask: torch.Tensor, fraction: float) -> float:
    flat_probs = probs.flatten()
    flat_target = target_mask.flatten()
    positives = int(flat_target.sum().item())
    if positives == 0:
        return 0.0
    k = max(1, int(flat_probs.numel() * fraction))
    topk_idx = torch.topk(flat_probs, k=min(k, flat_probs.numel()), largest=True).indices
    return (flat_target[topk_idx].sum().float() / flat_target.sum().float().clamp_min(1.0)).item()


def compute_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    thresholds: Sequence[float],
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    target_mask = target > 0.5
    metrics: Dict[str, float] = {
        "mae": torch.mean(torch.abs(probs - target)).item(),
        "target_pos": target_mask.float().mean().item(),
        "prob_mean": probs.mean().item(),
        "prob_max": probs.max().item(),
        "prob_pos": probs[target_mask].mean().item() if target_mask.any() else 0.0,
        "prob_neg": probs[target_mask.logical_not()].mean().item()
        if target_mask.logical_not().any()
        else 0.0,
        "top1pct_recall": _topk_recall(probs, target_mask, 0.01),
        "top5pct_recall": _topk_recall(probs, target_mask, 0.05),
    }
    for threshold in thresholds:
        pred_mask = probs > threshold
        intersection = torch.logical_and(pred_mask, target_mask).sum().float()
        union = torch.logical_or(pred_mask, target_mask).sum().float()
        pred_pos = pred_mask.float().mean()
        precision = intersection / pred_mask.sum().float().clamp_min(1.0)
        recall = intersection / target_mask.sum().float().clamp_min(1.0)
        metrics[f"pred_pos@{threshold:g}"] = pred_pos.item()
        metrics[f"iou@{threshold:g}"] = (intersection / union.clamp_min(1.0)).item()
        metrics[f"precision@{threshold:g}"] = precision.item()
        metrics[f"recall@{threshold:g}"] = recall.item()
    return metrics


def compute_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pred_direction_vectors: torch.Tensor,
    target_direction_vectors: torch.Tensor,
    loss_type: str,
    max_pos_weight: float,
    focal_gamma: float,
    direction_loss_weight: float,
) -> torch.Tensor:
    target_mask = target > 0.5
    pos = target_mask.sum(dim=(0, 2, 3)).float()
    neg = target_mask.logical_not().sum(dim=(0, 2, 3)).float()
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(max=max_pos_weight).to(logits.device)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight.view(1, -1, 1, 1),
        reduction="none",
    )
    if loss_type == "bce":
        map_loss = bce.mean()
    elif loss_type == "focal":
        probs = torch.sigmoid(logits)
        pt = torch.where(target_mask, probs, 1.0 - probs)
        map_loss = ((1.0 - pt).pow(focal_gamma) * bce).mean()
    else:
        raise ValueError(f"loss_type must be bce or focal, got {loss_type}")
    direction_loss = F.mse_loss(pred_direction_vectors, target_direction_vectors)
    return map_loss + direction_loss_weight * direction_loss


def compute_direction_metrics(
    pred_direction_vectors: torch.Tensor,
    target_direction_vectors: torch.Tensor,
) -> Dict[str, float]:
    target_norm = target_direction_vectors.norm(dim=-1)
    valid = target_norm > 0.5
    mae = torch.mean(torch.abs(pred_direction_vectors - target_direction_vectors)).item()
    if valid.any():
        pred_unit = F.normalize(pred_direction_vectors[valid], dim=-1)
        target_unit = F.normalize(target_direction_vectors[valid], dim=-1)
        cosine = (pred_unit * target_unit).sum(dim=-1).mean().item()
    else:
        cosine = 0.0
    return {
        "direction_mae": mae,
        "direction_cos": cosine,
    }


def initialize_output_prior(predictor: InstructionCognitiveMapPredictor, positive_prob: float) -> None:
    if not 0.0 < positive_prob < 1.0:
        raise ValueError(f"--init-positive-prob must be in (0, 1), got {positive_prob}")
    final_head = predictor.output_head[-1]
    if not isinstance(final_head, torch.nn.Conv2d) or final_head.bias is None:
        return
    prior_logit = float(np.log(positive_prob / (1.0 - positive_prob)))
    torch.nn.init.constant_(final_head.bias, prior_logit)


def _best_iou(metrics: Dict[str, float]) -> float:
    ious = [value for key, value in metrics.items() if key.startswith("iou@")]
    return max(ious) if ious else 0.0


def _iterate_batches(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: Optional[int] = None,
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
    loss_type: str = "bce",
    max_pos_weight: float = 100.0,
    focal_gamma: float = 2.0,
    direction_loss_weight: float = 0.1,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: Dict[str, float] = {}
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = _move_batch(batch, device)
        logits, pred_direction_vectors = model(
            batch["txt_ids"],
            batch["txt_task_encoding"],
            batch["txt_masks"],
            batch["start_direction_vectors"],
            batch["start_positions"],
        )
        loss = compute_loss(
            logits,
            batch["grids"],
            pred_direction_vectors,
            batch["direction_vectors"],
            loss_type=loss_type,
            max_pos_weight=max_pos_weight,
            focal_gamma=focal_gamma,
            direction_loss_weight=direction_loss_weight,
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = compute_metrics(logits.detach(), batch["grids"], thresholds=thresholds)
        metrics.update(
            compute_direction_metrics(
                pred_direction_vectors.detach(),
                batch["direction_vectors"],
            )
        )
        metrics["loss"] = loss.item()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "mae": 0.0}
    return {key: value / total_batches for key, value in totals.items()}


def _args_to_dict(args: object) -> Dict[str, object]:
    if hasattr(args, "as_dict"):
        return args.as_dict()
    return vars(args)


def save_checkpoint(
    output_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: object,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unwrapped_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    predictor = unwrapped_model.predictor if hasattr(unwrapped_model, "predictor") else unwrapped_model
    map_predictor = predictor.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "args": _args_to_dict(args),
            "map_predictor": map_predictor,
            "state_dict": {f"map_predictor.{key}": value for key, value in map_predictor.items()},
            "optimizer": optimizer.state_dict(),
        },
        output_path,
    )


def _build_dataloader(
    dataset_paths: Sequence[Path],
    dataset_name: Optional[str],
    max_text_len: int,
    batch_size: int,
    num_workers: int,
    limit: Optional[int],
    shuffle: bool,
) -> DataLoader:
    examples = load_predictor_examples(
        dataset_paths,
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


class TrainMapPredictorArgs(Tap):
    exp_config: str = "run_r2r/iter_train.yaml"
    train_dataset: List[Path] = [
        Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train_90.json.gz")
    ]
    val_dataset: List[Path] = [
        Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz")
    ]
    dataset: Optional[str] = None
    output: Path = Path("data/logs/checkpoints/release_r2r_imagined_predictor/store/predictor.pt")
    batch_size: int = 8
    epochs: int = 5
    lr: float = 1e-4
    loss: str = "bce"
    max_pos_weight: float = 20.0
    focal_gamma: float = 2.0
    direction_loss_weight: float = 0.1
    init_positive_prob: float = 0.002
    thresholds: str = "0.001,0.002,0.005,0.01,0.02,0.05"
    max_text_len: Optional[int] = None
    num_workers: int = 2
    seed: int = 0
    limit: Optional[int] = None
    val_limit: Optional[int] = None
    log_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    opts: Optional[List[str]] = None

    def configure(self) -> None:
        self.add_argument("--exp-config", default=TrainMapPredictorArgs.exp_config)
        self.add_argument(
            "--train-dataset",
            nargs="+",
            type=Path,
            default=TrainMapPredictorArgs.train_dataset,
        )
        self.add_argument(
            "--val-dataset",
            nargs="+",
            type=Path,
            default=TrainMapPredictorArgs.val_dataset,
        )
        self.add_argument("--dataset", choices=sorted(DATASET_FLAGS), default=None)
        self.add_argument("--output", type=Path, default=TrainMapPredictorArgs.output)
        self.add_argument("--batch-size", type=int, default=TrainMapPredictorArgs.batch_size)
        self.add_argument("--loss", choices=["bce", "focal"], default=TrainMapPredictorArgs.loss)
        self.add_argument("--max-pos-weight", type=float, default=TrainMapPredictorArgs.max_pos_weight)
        self.add_argument("--focal-gamma", type=float, default=TrainMapPredictorArgs.focal_gamma)
        self.add_argument(
            "--direction-loss-weight",
            type=float,
            default=TrainMapPredictorArgs.direction_loss_weight,
        )
        self.add_argument(
            "--init-positive-prob",
            type=float,
            default=TrainMapPredictorArgs.init_positive_prob,
        )
        self.add_argument("--max-text-len", type=int, default=None)
        self.add_argument("--num-workers", type=int, default=TrainMapPredictorArgs.num_workers)
        self.add_argument("--val-limit", type=int, default=None)
        self.add_argument("--log-every", type=int, default=TrainMapPredictorArgs.log_every)
        self.add_argument("--opts", nargs=argparse.REMAINDER, default=None)


def parse_args(argv: Optional[Iterable[str]] = None) -> TrainMapPredictorArgs:
    return TrainMapPredictorArgs().parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = get_config(args.exp_config, args.opts)
    max_text_len = args.max_text_len or config.IL.max_text_len
    device = torch.device(args.device)
    thresholds = _parse_thresholds(args.thresholds)

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
    initialize_output_prior(predictor, args.init_positive_prob)
    model = CognitiveMapPredictionModel(vln_bert=vln_bert, predictor=predictor).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} CUDA devices for text encoder + predictor DataParallel")
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=0.01)

    train_loader = _build_dataloader(
        dataset_paths=args.train_dataset,
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
        train_metrics = _iterate_batches(
            train_loader,
            model,
            device,
            optimizer,
            thresholds=thresholds,
            loss_type=args.loss,
            max_pos_weight=args.max_pos_weight,
            focal_gamma=args.focal_gamma,
            direction_loss_weight=args.direction_loss_weight,
        )
        metrics = {"train_" + key: value for key, value in train_metrics.items()}
        if val_loader is not None:
            val_metrics = _iterate_batches(
                val_loader,
                model,
                device,
                thresholds=thresholds,
                loss_type=args.loss,
                max_pos_weight=args.max_pos_weight,
                focal_gamma=args.focal_gamma,
                direction_loss_weight=args.direction_loss_weight,
            )
            metrics.update({"val_" + key: value for key, value in val_metrics.items()})
            selection_metric = -_best_iou(val_metrics)
        else:
            selection_metric = -_best_iou(train_metrics)
        if epoch % args.log_every == 0:
            metric_text = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            print(f"epoch={epoch} {metric_text}")
        save_checkpoint(args.output, model, optimizer, epoch, metrics, args)
        if selection_metric < best_metric:
            best_metric = selection_metric
            save_checkpoint(args.output.with_suffix(".best.pt"), model, optimizer, epoch, metrics, args)


if __name__ == "__main__":
    main()
