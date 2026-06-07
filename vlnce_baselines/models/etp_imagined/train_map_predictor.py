"""Offline trainer for instruction-to-cognitive-map prediction.

This script trains only InstructionCognitiveMapPredictor from paired VLN
instructions and cached cognitive-map targets. It deliberately avoids Habitat
rollout and navigation loss.
"""

from __future__ import annotations

import os
import random
from collections import OrderedDict
from collections.abc import Sized
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

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
    TRAJECTORY_KEYPOINT_COUNT,
    NUM_MAP_CATEGORIES,
    SIZE,
    cached_cognitive_map_to_tensors,
    load_cached_cognitive_map,
)
from vlnce_baselines.models.etp_prior_gt.vlnbert_init import get_vlnbert_models
from prior._coords import grid_to_meters
from prior.grid_map import CognitiveGridMap
from prior.vlnce import VLNCEEpisodeEntry


PAD_ID = 1
TASK_TYPE_IDS = {"r2r": 1, "rxr": 2}


@dataclass(frozen=True)
class PredictorExample:
    episode_id: str
    cache_id: str
    scene_id: str
    instruction_text: str
    token_ids: List[int]
    dataset: str


def _scene_key(scene_id: str) -> str:
    return os.path.splitext(os.path.basename(scene_id))[0]


def load_predictor_examples(
    dataset: str,
    splits: Sequence[str],
    limit: Optional[int] = None,
) -> List[PredictorExample]:
    examples: List[PredictorExample] = []
    for entry in VLNCEEpisodeEntry.iter_from(
        _dataset_name_for_vlnce(dataset),
        splits=splits,
    ):
        examples.append(
            PredictorExample(
                episode_id=str(entry.episode_id),
                cache_id=entry.unique_id,
                scene_id=entry.scene_id,
                instruction_text=entry.instruction,
                token_ids=entry.instruction_tokens,
                dataset=entry.dataset.lower(),
            )
        )
        if limit is not None and len(examples) >= limit:
            return examples
    return examples


def _dataset_name_for_vlnce(dataset: str) -> Literal["R2R", "RxR"]:
    dataset = dataset.lower()
    if dataset == "r2r":
        return "R2R"
    if dataset == "rxr":
        return "RxR"
    raise ValueError(f"dataset must be r2r or rxr, got {dataset}")


class CognitiveMapPredictorDataset(Dataset):
    def __init__(self, examples: Sequence[PredictorExample]):
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("No examples with matching cognitive maps were found.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        example = self.examples[index]
        tensors = cached_cognitive_map_to_tensors(
            example.scene_id,
            example.cache_id,
            random_rotation_augmentation=True,
        )
        grid = tensors["grid"].float()
        trajectory_keypoints = tensors["trajectory_keypoints"].float()
        start_direction_vector = tensors["start_direction_vector"].float()
        start_position = tensors["start_position"].float()
        if grid.shape != (NUM_MAP_CATEGORIES, SIZE, SIZE):
            raise ValueError(
                f"Bad grid shape for episode {example.episode_id}: {tuple(grid.shape)}"
            )
        if trajectory_keypoints.shape != (TRAJECTORY_KEYPOINT_COUNT, 2):
            raise ValueError(
                f"Bad trajectory_keypoints shape for episode {example.episode_id}: "
                f"{tuple(trajectory_keypoints.shape)}"
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
            "trajectory_keypoints": trajectory_keypoints,
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
        self.vln_bert: Any = vln_bert
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
            txt_embeds = self.vln_bert.forward_txt(
                txt_ids, txt_task_encoding, txt_masks
            )
        return self.predictor(
            txt_embeds,
            txt_masks,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
        )


def collate_predictor_batch(
    batch: Sequence[Dict], max_text_len: int, pad_id: int = PAD_ID
) -> Dict:
    batch_size = len(batch)
    txt_ids = torch.full((batch_size, max_text_len), pad_id, dtype=torch.long)
    txt_task_encoding = torch.zeros((batch_size, max_text_len), dtype=torch.long)
    grids = torch.stack([item["grid"] for item in batch], dim=0)
    trajectory_keypoints = torch.stack(
        [item["trajectory_keypoints"] for item in batch], dim=0
    )
    start_direction_vectors = torch.stack(
        [item["start_direction_vector"] for item in batch], dim=0
    )
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
        "trajectory_keypoints": trajectory_keypoints,
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
        "trajectory_keypoints",
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


def _topk_recall(
    probs: torch.Tensor, target_mask: torch.Tensor, fraction: float
) -> float:
    flat_probs = probs.flatten()
    flat_target = target_mask.flatten()
    positives = int(flat_target.sum().item())
    if positives == 0:
        return 0.0
    k = max(1, int(flat_probs.numel() * fraction))
    topk_idx = torch.topk(
        flat_probs, k=min(k, flat_probs.numel()), largest=True
    ).indices
    return (
        flat_target[topk_idx].sum().float() / flat_target.sum().float().clamp_min(1.0)
    ).item()


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
    pred_trajectory_keypoints: torch.Tensor,
    target_trajectory_keypoints: torch.Tensor,
    loss_type: str,
    max_pos_weight: float,
    focal_gamma: float,
    trajectory_keypoint_loss_weight: float,
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
    trajectory_keypoint_loss = F.smooth_l1_loss(
        pred_trajectory_keypoints,
        target_trajectory_keypoints,
        beta=5.0,
    )
    return map_loss + trajectory_keypoint_loss_weight * trajectory_keypoint_loss


def compute_trajectory_keypoint_metrics(
    pred_trajectory_keypoints: torch.Tensor,
    target_trajectory_keypoints: torch.Tensor,
) -> Dict[str, float]:
    target_norm = target_trajectory_keypoints.norm(dim=-1)
    valid = target_norm > 0.5
    mae = torch.mean(
        torch.abs(pred_trajectory_keypoints - target_trajectory_keypoints)
    ).item()
    if valid.any():
        pred_unit = F.normalize(pred_trajectory_keypoints[valid], dim=-1)
        target_unit = F.normalize(target_trajectory_keypoints[valid], dim=-1)
        cosine = (pred_unit * target_unit).sum(dim=-1).mean().item()
    else:
        cosine = 0.0
    return {
        "trajectory_keypoint_mae": mae,
        "trajectory_keypoint_cos": cosine,
    }


def initialize_output_prior(
    predictor: InstructionCognitiveMapPredictor, positive_prob: float
) -> None:
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
    trajectory_keypoint_loss_weight: float = 0.1,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: Dict[str, float] = {}
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = _move_batch(batch, device)
        logits, pred_trajectory_keypoints = model(
            batch["txt_ids"],
            batch["txt_task_encoding"],
            batch["txt_masks"],
            batch["start_direction_vectors"],
            batch["start_positions"],
        )
        loss = compute_loss(
            logits,
            batch["grids"],
            pred_trajectory_keypoints,
            batch["trajectory_keypoints"],
            loss_type=loss_type,
            max_pos_weight=max_pos_weight,
            focal_gamma=focal_gamma,
            trajectory_keypoint_loss_weight=trajectory_keypoint_loss_weight,
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = compute_metrics(
            logits.detach(), batch["grids"], thresholds=thresholds
        )
        metrics.update(
            compute_trajectory_keypoint_metrics(
                pred_trajectory_keypoints.detach(),
                batch["trajectory_keypoints"],
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
    as_dict = getattr(args, "as_dict", None)
    if callable(as_dict):
        values = as_dict()
        if not isinstance(values, dict):
            raise TypeError("args.as_dict() must return a dict")
        return values
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
    unwrapped_model = (
        model.module if isinstance(model, torch.nn.DataParallel) else model
    )
    predictor = (
        unwrapped_model.predictor
        if hasattr(unwrapped_model, "predictor")
        else unwrapped_model
    )
    if not isinstance(predictor, torch.nn.Module):
        raise TypeError("model predictor must be a torch.nn.Module")
    map_predictor = predictor.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "args": _args_to_dict(args),
            "map_predictor": map_predictor,
            "state_dict": {
                f"map_predictor.{key}": value for key, value in map_predictor.items()
            },
            "optimizer": optimizer.state_dict(),
        },
        output_path,
    )


def _load_predictor_state_dict(
    checkpoint_path: Path,
) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "map_predictor" in checkpoint:
        return OrderedDict(checkpoint["map_predictor"])

    state_dict = checkpoint.get("state_dict", checkpoint)
    predictor_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        if normalized_key.startswith("map_predictor."):
            normalized_key = normalized_key[len("map_predictor.") :]
        elif normalized_key.startswith("predictor."):
            normalized_key = normalized_key[len("predictor.") :]
        else:
            continue
        predictor_state_dict[normalized_key] = value

    if not predictor_state_dict:
        raise ValueError(
            f"No map predictor weights found in checkpoint: {checkpoint_path}"
        )
    return predictor_state_dict


def _build_dataloader(
    dataset_name: str,
    splits: Sequence[str],
    max_text_len: int,
    batch_size: int,
    num_workers: int,
    limit: Optional[int],
    shuffle: bool,
) -> DataLoader:
    examples = load_predictor_examples(
        dataset_name,
        splits=splits,
        limit=limit,
    )
    dataset = CognitiveMapPredictorDataset(examples)
    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(collate_predictor_batch, max_text_len=max_text_len),
        **loader_kwargs,
    )


class TrainMapPredictorArgs(Tap):
    mode: Literal["train", "visualize"] = "train"
    exp_config: str = "run_r2r/iter_train.yaml"
    dataset: Literal["r2r", "rxr"] = "r2r"
    train_splits: List[str] = ["train"]
    val_splits: List[str] = ["val_unseen"]
    visualize_splits: List[str] = ["val_unseen"]
    output: Path = Path(
        "data/logs/checkpoints/release_r2r_imagined_predictor/store/predictor.pt"
    )
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    loss: Literal["bce", "focal"] = "bce"
    max_pos_weight: float = 20.0
    focal_gamma: float = 2.0
    trajectory_keypoint_loss_weight: float = 0.001
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
    predictor_checkpoint: Optional[Path] = None
    episode_id: Optional[str] = None
    episode_index: int = 0
    visualize_output_dir: Path = Path("data/visualizations/map_predictor")
    skip_ground_truth: bool = False


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainMapPredictorArgs:
    return TrainMapPredictorArgs(underscores_to_dashes=True).parse_args(argv)


def _dataset_len(loader: DataLoader) -> int:
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("DataLoader dataset must define __len__")
    return len(dataset)


def _select_visualize_example(args: TrainMapPredictorArgs) -> PredictorExample:
    examples = load_predictor_examples(
        args.dataset,
        splits=args.visualize_splits,
    )
    if args.episode_id is not None:
        for example in examples:
            if example.episode_id == args.episode_id:
                return example
        raise ValueError(
            f"episode_id={args.episode_id!r} was not found in "
            f"{args.dataset}:{args.visualize_splits}"
        )

    if not 0 <= args.episode_index < len(examples):
        raise IndexError(
            f"episode_index={args.episode_index} is out of range for "
            f"{len(examples)} examples in {args.dataset}:{args.visualize_splits}"
        )
    return examples[args.episode_index]


def _prediction_to_cognitive_grid_map(
    grid: torch.Tensor,
    trajectory_keypoints: torch.Tensor,
    start_direction_vector: torch.Tensor,
    start_position: torch.Tensor,
) -> CognitiveGridMap:
    cognitive_map = CognitiveGridMap()
    cognitive_map.grid = grid.detach().cpu().numpy().astype(np.float32)
    cognitive_map.trajectory_keypoints = []
    for row, col in trajectory_keypoints.detach().cpu().tolist():
        x, z = grid_to_meters(float(row), float(col))
        cognitive_map.trajectory_keypoints.append((float(x), float(z)))
    cognitive_map.start_direction_vector = (
        float(start_direction_vector[0].detach().cpu()),
        float(start_direction_vector[1].detach().cpu()),
    )
    if not cognitive_map.trajectory_keypoints:
        start_x, start_z = grid_to_meters(
            float(start_position[0].detach().cpu()),
            float(start_position[1].detach().cpu()),
        )
        cognitive_map.trajectory_keypoints.append((float(start_x), float(start_z)))
    return cognitive_map


def visualize_prediction(args: TrainMapPredictorArgs) -> None:
    if args.predictor_checkpoint is None:
        raise ValueError("--predictor-checkpoint is required for --mode visualize")
    if not args.predictor_checkpoint.exists():
        raise FileNotFoundError(args.predictor_checkpoint)

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
    predictor.load_state_dict(_load_predictor_state_dict(args.predictor_checkpoint))
    predictor.eval()

    model = CognitiveMapPredictionModel(vln_bert=vln_bert, predictor=predictor).to(
        device
    )
    model.eval()

    example = _select_visualize_example(args)
    item = CognitiveMapPredictorDataset([example])[0]
    batch = collate_predictor_batch([item], max_text_len=max_text_len)
    batch = _move_batch(batch, device)

    with torch.no_grad():
        map_logits, trajectory_keypoints = model(
            batch["txt_ids"],
            batch["txt_task_encoding"],
            batch["txt_masks"],
            batch["start_direction_vectors"],
            batch["start_positions"],
        )
        predicted_grid = torch.sigmoid(map_logits[0])

    output_dir = args.visualize_output_dir / f"episode_{example.episode_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_map = _prediction_to_cognitive_grid_map(
        predicted_grid,
        trajectory_keypoints[0],
        batch["start_direction_vectors"][0],
        batch["start_positions"][0],
    )
    predicted_path = output_dir / "predicted.png"
    predicted_map.visualize(
        predicted_path,
        title=f"Predicted Cognitive Map: episode {example.episode_id}",
    )

    if not args.skip_ground_truth:
        ground_truth_map = load_cached_cognitive_map(
            example.scene_id,
            example.cache_id,
        )
        ground_truth_path = output_dir / "ground_truth.png"
        ground_truth_map.visualize(
            ground_truth_path,
            title=f"Ground Truth Cognitive Map: episode {example.episode_id}",
        )

    print(f"episode_id={example.episode_id}")
    print(f"instruction={example.instruction_text}")
    print(f"saved_predicted={predicted_path}")
    if not args.skip_ground_truth:
        print(f"saved_ground_truth={ground_truth_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.mode == "visualize":
        visualize_prediction(args)
        return

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
    model = CognitiveMapPredictionModel(vln_bert=vln_bert, predictor=predictor).to(
        device
    )
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(
            f"Using {torch.cuda.device_count()} CUDA devices for text encoder + predictor DataParallel"
        )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, weight_decay=0.01)

    train_loader = _build_dataloader(
        dataset_name=args.dataset,
        splits=args.train_splits,
        max_text_len=max_text_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
        shuffle=True,
    )
    val_loader = None
    if args.val_splits:
        val_loader = _build_dataloader(
            dataset_name=args.dataset,
            splits=args.val_splits,
            max_text_len=max_text_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.val_limit,
            shuffle=False,
        )

    print(f"Train examples: {_dataset_len(train_loader)}")
    if val_loader is not None:
        print(f"Val examples: {_dataset_len(val_loader)}")

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
            trajectory_keypoint_loss_weight=args.trajectory_keypoint_loss_weight,
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
                trajectory_keypoint_loss_weight=args.trajectory_keypoint_loss_weight,
            )
            metrics.update({"val_" + key: value for key, value in val_metrics.items()})
            selection_metric = -_best_iou(val_metrics)
        else:
            selection_metric = -_best_iou(train_metrics)
        if epoch % args.log_every == 0:
            metric_text = " ".join(
                f"{key}={value:.5f}" for key, value in metrics.items()
            )
            print(f"epoch={epoch} {metric_text}")
        save_checkpoint(args.output, model, optimizer, epoch, metrics, args)
        if selection_metric < best_metric:
            best_metric = selection_metric
            save_checkpoint(
                args.output.with_suffix(".best.pt"),
                model,
                optimizer,
                epoch,
                metrics,
                args,
            )


if __name__ == "__main__":
    main()
