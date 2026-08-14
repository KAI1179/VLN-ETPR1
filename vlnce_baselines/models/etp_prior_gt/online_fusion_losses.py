"""Losses and target masks shared by OnlineFusion pretraining and DAgger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .map_fusion import OnlineMapFusionOutput


ROUTE_ON = 0
ROUTE_OFF = 1
ROUTE_FINISHED = 2
IGNORE_INDEX = -100


@dataclass(frozen=True)
class OnlineFusionLossWeights:
    grid: float = 0.3
    state: float = 0.05
    visual: float = 0.2
    progress: float = 0.1
    ghost: float = 0.1

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} loss weight must be non-negative")


@dataclass
class OnlineFusionTargets:
    dense_grid: Optional[torch.Tensor] = None
    grid_valid_mask: Optional[torch.Tensor] = None
    visual: Optional[torch.Tensor] = None
    visual_valid_mask: Optional[torch.Tensor] = None
    phase: Optional[torch.Tensor] = None
    route_state: Optional[torch.Tensor] = None
    remaining: Optional[torch.Tensor] = None
    remaining_valid_mask: Optional[torch.Tensor] = None
    recovery: Optional[torch.Tensor] = None
    recovery_valid_mask: Optional[torch.Tensor] = None
    expert_action: Optional[torch.Tensor] = None


def sparse_grid_valid_mask(
    target: torch.Tensor,
    *,
    support_radius: int = 2,
) -> torch.Tensor:
    """Mark positives and nearby annotated support without trusting all sparse zeros."""

    if target.dim() not in (3, 4):
        raise ValueError(
            "target must have shape (C,H,W) or (B,C,H,W), "
            f"got {tuple(target.shape)}"
        )
    if support_radius < 0:
        raise ValueError("support_radius must be non-negative")
    batched = target.unsqueeze(0) if target.dim() == 3 else target
    positives = batched > 0
    support = positives.any(dim=1, keepdim=True).to(dtype=batched.dtype)
    if support_radius:
        kernel = support_radius * 2 + 1
        support = F.max_pool2d(support, kernel, stride=1, padding=support_radius)
    valid = positives | support.to(dtype=torch.bool).expand_as(positives)
    return valid.squeeze(0) if target.dim() == 3 else valid


def masked_focal_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gamma: float = 2.0,
) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != valid_mask.shape:
        raise ValueError(
            "grid logits, target and valid mask must have identical shapes; "
            f"got {tuple(logits.shape)}, {tuple(target.shape)}, "
            f"{tuple(valid_mask.shape)}"
        )
    valid = valid_mask.to(dtype=torch.bool)
    if not valid.any():
        return logits.sum() * 0.0
    target = target.to(dtype=logits.dtype).clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probabilities = torch.sigmoid(logits)
    probability_true = probabilities * target + (1.0 - probabilities) * (1.0 - target)
    focal = ((1.0 - probability_true) ** gamma) * bce
    focal = focal[valid].mean()

    valid_float = valid.to(dtype=logits.dtype)
    probabilities = probabilities * valid_float
    target = target * valid_float
    reduce_dims = tuple(range(2, logits.dim()))
    intersection = (probabilities * target).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    class_valid = target.sum(dim=reduce_dims) > 0
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice = dice[class_valid].mean() if class_valid.any() else logits.sum() * 0.0
    return focal + dice


def _masked_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("binary logits, targets and mask must have identical shapes")
    mask = mask.to(dtype=torch.bool)
    if not mask.any():
        return logits.sum() * 0.0
    targets = targets.to(dtype=logits.dtype)
    positive_count = (targets[mask] > 0.5).sum().to(dtype=logits.dtype)
    negative_count = mask.sum().to(dtype=logits.dtype) - positive_count
    positive_weight = (negative_count / positive_count.clamp_min(1.0)).clamp(1.0, 20.0)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=positive_weight,
    )
    return loss[mask].mean()


def _state_loss(output: OnlineMapFusionOutput) -> torch.Tensor:
    spatial = output.updated_map_tokens[:, :-1]
    residual_penalty = output.map_state_residual.square().mean()
    gate_penalty = output.map_write_gate.mean()
    token_spread = spatial.float().std(dim=1, unbiased=False).mean()
    diversity_penalty = F.relu(token_spread.new_tensor(0.05) - token_spread)
    return residual_penalty + 0.1 * gate_penalty + diversity_penalty


def _progress_loss(
    output: OnlineMapFusionOutput,
    targets: OnlineFusionTargets,
) -> torch.Tensor:
    terms = []
    if targets.phase is not None:
        valid = targets.phase != IGNORE_INDEX
        if valid.any():
            terms.append(F.cross_entropy(output.phase_logits[valid], targets.phase[valid]))
    if targets.route_state is not None:
        valid = targets.route_state != IGNORE_INDEX
        if valid.any():
            terms.append(
                F.cross_entropy(
                    output.route_state_logits[valid],
                    targets.route_state[valid],
                )
            )
    if targets.remaining is not None:
        mask = targets.remaining_valid_mask
        if mask is None:
            mask = torch.ones_like(targets.remaining, dtype=torch.bool)
        if mask.any():
            terms.append(
                F.smooth_l1_loss(
                    output.remaining[mask],
                    targets.remaining.to(dtype=output.remaining.dtype)[mask],
                )
            )
    if targets.recovery is not None:
        mask = targets.recovery_valid_mask
        if mask is None:
            raise ValueError("recovery_valid_mask is required with recovery targets")
        terms.append(_masked_binary_loss(output.recovery_logits, targets.recovery, mask))
    return sum(terms, output.phase_logits.sum() * 0.0)


def _ghost_rank_loss(
    output: OnlineMapFusionOutput,
    expert_action: Optional[torch.Tensor],
) -> torch.Tensor:
    if expert_action is None:
        return output.action_logit_residuals.sum() * 0.0
    if expert_action.shape != (output.action_logit_residuals.shape[0],):
        raise ValueError("expert_action must have shape (B,)")
    targets = expert_action.to(dtype=torch.long)
    in_range = (targets >= 0) & (targets < output.ghost_masks.shape[1])
    safe_targets = targets.clamp(0, output.ghost_masks.shape[1] - 1)
    valid = in_range & output.ghost_masks.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    if not valid.any():
        return output.action_logit_residuals.sum() * 0.0
    logits = output.ghost_logit_residuals.masked_fill(~output.ghost_masks, -1e4)
    return F.cross_entropy(logits[valid], targets[valid])


def compute_online_fusion_losses(
    output: OnlineMapFusionOutput,
    targets: OnlineFusionTargets,
    weights: OnlineFusionLossWeights,
) -> Dict[str, torch.Tensor]:
    """Return weighted named losses; absent targets disable only their term."""

    losses: Dict[str, torch.Tensor] = {}
    if weights.grid:
        if output.dense_grid_logits is None:
            raise ValueError("dense_grid_logits are required when grid loss is enabled")
        if targets.dense_grid is None or targets.grid_valid_mask is None:
            raise ValueError("dense grid target and valid mask are required")
        losses["grid"] = weights.grid * masked_focal_bce_dice_loss(
            output.dense_grid_logits,
            targets.dense_grid,
            targets.grid_valid_mask,
        )
    if weights.state:
        losses["state"] = weights.state * _state_loss(output)
    if weights.visual:
        if targets.visual is None or targets.visual_valid_mask is None:
            raise ValueError(
                "visual targets are required when visual evidence loss is enabled"
            )
        visual_mask = targets.visual_valid_mask
        if visual_mask.dim() == 2:
            visual_mask = visual_mask.unsqueeze(-1).expand_as(output.visual_logits)
        losses["visual"] = weights.visual * _masked_binary_loss(
            output.visual_logits,
            targets.visual,
            visual_mask,
        )
    if weights.progress:
        losses["progress"] = weights.progress * _progress_loss(output, targets)
    if weights.ghost:
        losses["ghost"] = weights.ghost * _ghost_rank_loss(
            output,
            targets.expert_action,
        )
    return losses


def total_online_fusion_loss(losses: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not losses:
        raise ValueError("no OnlineFusion loss is enabled")
    return sum(losses.values())
