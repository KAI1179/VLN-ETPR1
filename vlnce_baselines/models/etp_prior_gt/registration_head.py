"""Optional route registration on fused topological-node features."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class RouteRegistrationHead(nn.Module):
    """Predict route order and progress; supervision never enters the features."""

    def __init__(self, hidden_size: int, n_max: int = 16, feedback: bool = False):
        super().__init__()
        if n_max < 1:
            raise ValueError("n_max must be positive")
        self.n_max = n_max
        self.features = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.route = nn.Linear(hidden_size, n_max)
        self.progress = nn.Linear(hidden_size, 1)
        self.feedback_projection = (
            nn.Linear(n_max + 1, hidden_size) if feedback else None
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        route_counts: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if node_features.ndim != 3 or node_mask.shape != node_features.shape[:2]:
            raise ValueError("Expected node features (B,G,H) and mask (B,G)")
        valid = node_mask.bool().clone()
        valid[:, 0] = False  # STOP is an action, not a spatial graph node.
        hidden = self.features(node_features)
        logits = self.route(hidden)
        if route_counts is None:
            route_counts = torch.full(
                (len(node_features),),
                self.n_max,
                device=logits.device,
                dtype=torch.long,
            )
        if route_counts.shape != (len(node_features),) or bool(
            (route_counts < 1).any()
        ):
            raise ValueError("Each episode needs a positive route-node count")
        classes = torch.arange(self.n_max, device=logits.device)
        class_mask = classes[None, :] < route_counts[:, None].clamp(max=self.n_max)
        logits = logits.masked_fill(
            ~class_mask[:, None, :], torch.finfo(logits.dtype).min
        )
        progress = self.progress(hidden).squeeze(-1).sigmoid()
        outputs = {"route_logits": logits, "progress": progress, "node_mask": valid}
        if self.feedback_projection is not None:
            estimates = torch.cat([logits.softmax(-1), progress.unsqueeze(-1)], dim=-1)
            projected = self.feedback_projection(estimates)
            weights = valid.unsqueeze(-1).to(projected.dtype)
            token = (projected * weights).sum(1, keepdim=True)
            token = token / weights.sum(1, keepdim=True).clamp_min(1)
            outputs["feedback_token"] = token
            outputs["feedback_mask"] = valid.any(dim=1, keepdim=True)
        return outputs


def registration_loss(
    predictions: dict[str, torch.Tensor],
    route_targets: torch.Tensor,
    progress_targets: torch.Tensor,
) -> torch.Tensor:
    """Mean CE plus mean progress MSE; -1, STOP and padding are ignored.

    The all-ignored result remains connected to every head parameter for DDP.
    Overflow episodes are masked by the rollout labeller, never relabelled.
    """
    logits = predictions["route_logits"]
    progress = predictions["progress"]
    if (
        route_targets.shape != progress.shape
        or progress_targets.shape != progress.shape
    ):
        raise ValueError("Registration targets must match (B,G) predictions")
    valid = predictions["node_mask"] & (route_targets >= 0)
    if bool((route_targets[valid] >= logits.shape[-1]).any()):
        raise ValueError("Route target exceeds registration capacity")
    if not bool(valid.any()):
        return logits[..., 0].sum() * 0.0 + progress.sum() * 0.0
    return F.cross_entropy(logits[valid], route_targets[valid]) + F.mse_loss(
        progress[valid], progress_targets[valid]
    )


def append_registration_feedback(
    map_tokens: torch.Tensor,
    map_mask: Optional[torch.Tensor],
    predictions: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the predicted estimate, never an oracle route label, as one token."""
    token = predictions["feedback_token"]
    if token.shape != (map_tokens.shape[0], 1, map_tokens.shape[2]):
        raise ValueError("Feedback token must match map-token width")
    if map_mask is None:
        map_mask = torch.ones(
            map_tokens.shape[:2], dtype=torch.bool, device=map_tokens.device
        )
    return torch.cat([map_tokens, token], dim=1), torch.cat(
        [map_mask, predictions["feedback_mask"]], dim=1
    )
