"""Decode updated map tokens into DETR-style cognitive-map boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .map_box_targets import CognitiveMapBoxTarget
from .map_utils import MAP_TOKEN_COUNT, NUM_MAP_CATEGORIES


@dataclass(frozen=True)
class CognitiveMapBoxDecoderOutput:
    pred_logits: torch.Tensor
    pred_boxes: torch.Tensor


class CognitiveMapDecoder(nn.Module):
    """Decode updated map tokens into a set of class-labeled boxes."""

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 100,
        num_decoder_layers: int = 2,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {num_queries}")
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=_default_attention_heads(hidden_size),
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_size)
        self.class_head = nn.Linear(hidden_size, NUM_MAP_CATEGORIES + 1)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
        )

    def _validate_updated_map_tokens(self, updated_map_tokens: torch.Tensor) -> None:
        if updated_map_tokens.dim() != 3:
            raise ValueError(
                "updated_map_tokens must have shape "
                f"(B, {MAP_TOKEN_COUNT}, {self.hidden_size}), "
                f"got {tuple(updated_map_tokens.shape)}"
            )
        if updated_map_tokens.shape[1:] != (MAP_TOKEN_COUNT, self.hidden_size):
            raise ValueError(
                "updated_map_tokens must have shape "
                f"(B, {MAP_TOKEN_COUNT}, {self.hidden_size}), "
                f"got {tuple(updated_map_tokens.shape)}"
            )

    def forward(self, updated_map_tokens: torch.Tensor) -> CognitiveMapBoxDecoderOutput:
        """Return class logits and normalized cx, cz, w, h box predictions."""
        self._validate_updated_map_tokens(updated_map_tokens)
        batch_size = updated_map_tokens.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(tgt=queries, memory=updated_map_tokens)
        return CognitiveMapBoxDecoderOutput(
            pred_logits=self.class_head(decoded),
            pred_boxes=self.box_head(decoded).sigmoid(),
        )


class HungarianBoxMatcher(nn.Module):
    """Match predicted cognitive-map boxes to target boxes."""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        if cost_class == 0.0 and cost_bbox == 0.0 and cost_giou == 0.0:
            raise ValueError("at least one matching cost must be non-zero")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: CognitiveMapBoxDecoderOutput,
        targets: Sequence[CognitiveMapBoxTarget | Mapping[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        probabilities = outputs.pred_logits.softmax(-1)
        indices = []
        for batch_idx, target in enumerate(targets):
            labels = _target_labels(target).to(outputs.pred_logits.device)
            boxes = _target_boxes(target).to(outputs.pred_boxes.device)
            if labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=outputs.pred_boxes.device)
                indices.append((empty, empty))
                continue
            cost_class = -probabilities[batch_idx][:, labels]
            cost_bbox = torch.cdist(outputs.pred_boxes[batch_idx], boxes, p=1)
            cost_giou = -generalized_box_iou(
                box_cxczwh_to_xzxy(outputs.pred_boxes[batch_idx]),
                box_cxczwh_to_xzxy(boxes),
            )
            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            pred_idx, target_idx = linear_sum_assignment(cost.cpu())
            indices.append(
                (
                    torch.as_tensor(
                        pred_idx,
                        dtype=torch.int64,
                        device=outputs.pred_boxes.device,
                    ),
                    torch.as_tensor(
                        target_idx,
                        dtype=torch.int64,
                        device=outputs.pred_boxes.device,
                    ),
                )
            )
        return indices


class CognitiveMapSetCriterion(nn.Module):
    """Compute DETR-style set prediction loss for cognitive-map boxes."""

    def __init__(
        self,
        num_classes: int = NUM_MAP_CATEGORIES,
        matcher: HungarianBoxMatcher | None = None,
        eos_coef: float = 0.1,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianBoxMatcher()
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        outputs: CognitiveMapBoxDecoderOutput,
        targets: Sequence[CognitiveMapBoxTarget | Mapping[str, torch.Tensor]],
    ) -> torch.Tensor:
        if len(targets) != outputs.pred_logits.shape[0]:
            raise ValueError(
                "targets length must match batch size, "
                f"got {len(targets)} and {outputs.pred_logits.shape[0]}"
            )
        targets = [
            _target_on_device(target, outputs.pred_logits.device)
            for target in targets
        ]
        indices = self.matcher(outputs, targets)
        target_classes = torch.full(
            outputs.pred_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=outputs.pred_logits.device,
        )
        batch_idx, src_idx = _source_permutation_indices(indices)
        if src_idx.numel():
            target_classes[batch_idx, src_idx] = torch.cat(
                [
                    _target_labels(target)[target_idx]
                    for target, (_, target_idx) in zip(targets, indices)
                    if target_idx.numel()
                ]
            )
        loss_ce = F.cross_entropy(
            outputs.pred_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )

        if src_idx.numel() == 0:
            return loss_ce

        src_boxes = outputs.pred_boxes[batch_idx, src_idx]
        target_boxes = torch.cat(
            [
                _target_boxes(target)[target_idx]
                for target, (_, target_idx) in zip(targets, indices)
                if target_idx.numel()
            ],
            dim=0,
        )
        num_boxes = max(1, int(target_boxes.shape[0]))
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes
        loss_giou = (
            1.0
            - torch.diag(
                generalized_box_iou(
                    box_cxczwh_to_xzxy(src_boxes),
                    box_cxczwh_to_xzxy(target_boxes),
                )
            )
        ).sum() / num_boxes
        return loss_ce + self.bbox_loss_coef * loss_bbox + self.giou_loss_coef * loss_giou


def box_cxczwh_to_xzxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cz, width, depth = boxes.unbind(-1)
    return torch.stack(
        (
            cx - 0.5 * width,
            cz - 0.5 * depth,
            cx + 0.5 * width,
            cz + 0.5 * depth,
        ),
        dim=-1,
    )


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = _box_iou(boxes1, boxes2)
    left_top = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    width_depth = (right_bottom - left_top).clamp(min=0)
    area = width_depth[:, :, 0] * width_depth[:, :, 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def _box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    width_depth = (right_bottom - left_top).clamp(min=0)
    intersection = width_depth[:, :, 0] * width_depth[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp(min=1e-6), union


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    width_depth = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return width_depth[:, 0] * width_depth[:, 1]


def _default_attention_heads(hidden_size: int) -> int:
    for heads in (8, 4, 2):
        if hidden_size % heads == 0:
            return heads
    return 1


def _source_permutation_indices(
    indices: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    non_empty = [(batch_idx, src) for batch_idx, (src, _) in enumerate(indices) if src.numel()]
    if not non_empty:
        device = indices[0][0].device if indices else torch.device("cpu")
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty
    batch_indices = torch.cat(
        [torch.full_like(src, batch_idx) for batch_idx, src in non_empty]
    )
    source_indices = torch.cat([src for _, src in non_empty])
    return batch_indices, source_indices


def _target_on_device(
    target: CognitiveMapBoxTarget | Mapping[str, torch.Tensor],
    device: torch.device,
) -> CognitiveMapBoxTarget | Mapping[str, torch.Tensor]:
    if isinstance(target, CognitiveMapBoxTarget):
        return target.to(device)
    return {
        "labels": target["labels"].to(device),
        "boxes": target["boxes"].to(device),
    }


def _target_labels(
    target: CognitiveMapBoxTarget | Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if isinstance(target, CognitiveMapBoxTarget):
        return target.labels
    return target["labels"]


def _target_boxes(
    target: CognitiveMapBoxTarget | Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if isinstance(target, CognitiveMapBoxTarget):
        return target.boxes
    return target["boxes"]
