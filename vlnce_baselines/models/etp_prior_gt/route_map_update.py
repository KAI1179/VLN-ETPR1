"""Route-gated visual updates for a Try5 cognitive map."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


MAP_CELL_SIZE_M = 0.5
NUM_AZIMUTHS = 12
NUM_DISTANCE_BINS = 5
NUM_MAP_CATEGORIES = 37


@dataclass
class RouteMapState:
    prior: torch.Tensor
    current: torch.Tensor
    evidence: torch.Tensor
    coverage: torch.Tensor


class SpatialVisualHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, NUM_MAP_CATEGORIES * NUM_DISTANCE_BINS + NUM_DISTANCE_BINS),
        )

    def forward(self, pano_features: torch.Tensor):
        outputs = self.net(pano_features)
        semantic_logits = outputs[..., : NUM_MAP_CATEGORIES * NUM_DISTANCE_BINS]
        coverage_logits = outputs[..., NUM_MAP_CATEGORIES * NUM_DISTANCE_BINS :]
        return (
            semantic_logits.view(
                -1, NUM_AZIMUTHS, NUM_MAP_CATEGORIES, NUM_DISTANCE_BINS
            ),
            coverage_logits.view(-1, NUM_AZIMUTHS, NUM_DISTANCE_BINS),
        )


class RouteUpdateGate(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.map_context = nn.Linear(NUM_MAP_CATEGORIES, hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        pano_features: torch.Tensor,
        text_features: torch.Tensor,
        map_state: torch.Tensor,
    ) -> torch.Tensor:
        visual_context = pano_features.mean(dim=1)
        map_context = self.map_context(map_state.mean(dim=(-1, -2)))
        return self.net(
            torch.cat((visual_context, text_features, map_context), dim=-1)
        ).squeeze(-1)


class PosteriorRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(NUM_MAP_CATEGORIES * 2 + 1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.delta = nn.Conv2d(64, NUM_MAP_CATEGORIES, kernel_size=1)
        self.beta = nn.Conv2d(64, 1, kernel_size=1)

    def forward(
        self,
        prior: torch.Tensor,
        evidence: torch.Tensor,
        coverage: torch.Tensor,
    ):
        features = self.net(torch.cat((prior, evidence, coverage), dim=1))
        return self.delta(features), torch.sigmoid(self.beta(features))


class RouteMapUpdater(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.visual_head = SpatialVisualHead(hidden_size)
        self.route_gate = RouteUpdateGate(hidden_size)
        self.refiner = PosteriorRefiner()

    def initial_state(self, prior: torch.Tensor) -> RouteMapState:
        return RouteMapState(
            prior=prior,
            current=prior,
            evidence=torch.zeros_like(prior),
            coverage=torch.zeros_like(prior[:, :1]),
        )

    @staticmethod
    def _heading_from_quaternion(rotation: torch.Tensor) -> torch.Tensor:
        x, y, z, w = rotation.unbind(dim=-1)
        return torch.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))

    def _project_evidence(
        self,
        semantic_logits: torch.Tensor,
        coverage_logits: torch.Tensor,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        world_starts: torch.Tensor,
        map_starts: torch.Tensor,
        grid_size: int,
    ):
        batch_size = semantic_logits.shape[0]
        device = semantic_logits.device
        semantic = torch.sigmoid(semantic_logits)
        coverage = torch.sigmoid(coverage_logits)
        evidence_grid = semantic.new_zeros(
            batch_size, NUM_MAP_CATEGORIES, grid_size, grid_size
        )
        coverage_grid = semantic.new_zeros(batch_size, 1, grid_size, grid_size)
        heading = self._heading_from_quaternion(rotations)
        current_row = map_starts[:, 0] + (positions[:, 0] - world_starts[:, 0]) / MAP_CELL_SIZE_M
        current_col = map_starts[:, 1] + (positions[:, 2] - world_starts[:, 2]) / MAP_CELL_SIZE_M
        for azimuth in range(NUM_AZIMUTHS):
            angle = heading + azimuth * (2 * torch.pi / NUM_AZIMUTHS)
            for distance_bin in range(NUM_DISTANCE_BINS):
                distance = float(distance_bin * 2 + 1)
                rows = torch.round(current_row - distance * torch.cos(angle) / MAP_CELL_SIZE_M).long()
                cols = torch.round(current_col + distance * torch.sin(angle) / MAP_CELL_SIZE_M).long()
                valid = (rows >= 0) & (rows < grid_size) & (cols >= 0) & (cols < grid_size)
                for batch_index in torch.where(valid)[0]:
                    row = rows[batch_index]
                    col = cols[batch_index]
                    confidence = coverage[batch_index, azimuth, distance_bin]
                    evidence_grid[batch_index, :, row, col] = semantic[
                        batch_index, azimuth, :, distance_bin
                    ]
                    coverage_grid[batch_index, 0, row, col] = confidence
        return evidence_grid, coverage_grid

    def update(
        self,
        state: RouteMapState,
        pano_features: torch.Tensor,
        text_features: torch.Tensor,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        world_starts: torch.Tensor,
        map_starts: torch.Tensor,
    ):
        semantic_logits, coverage_logits = self.visual_head(pano_features)
        gate_logits = self.route_gate(pano_features, text_features, state.current)
        gate = torch.sigmoid(gate_logits).view(-1, 1, 1, 1)
        visual_evidence, visual_coverage = self._project_evidence(
            semantic_logits,
            coverage_logits,
            positions,
            rotations,
            world_starts,
            map_starts,
            state.current.shape[-1],
        )
        evidence = torch.where(
            visual_coverage > 0,
            visual_evidence,
            state.evidence,
        )
        coverage = torch.maximum(state.coverage, visual_coverage)
        delta, beta = self.refiner(state.prior, evidence, coverage)
        unseen = torch.clamp(state.prior + beta * delta, 0, 1)
        observed = (1 - coverage) * state.current + coverage * evidence
        candidate = coverage * observed + (1 - coverage) * unseen
        current = gate * candidate + (1 - gate) * state.current
        return (
            RouteMapState(state.prior, current, evidence, coverage),
            {
                "semantic_logits": semantic_logits,
                "coverage_logits": coverage_logits,
                "route_gate_logits": gate_logits,
            },
        )


def focal_bce(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy(prediction, target, reduction="none")
    pt = torch.where(target > 0.5, prediction, 1 - prediction)
    return ((1 - pt).pow(2) * bce).mean()


def _masked_bce(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    loss = F.binary_cross_entropy(prediction, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def map_losses(
    current: torch.Tensor,
    prior: torch.Tensor,
    target: torch.Tensor,
    coverage: torch.Tensor,
):
    observed = coverage.expand_as(current) > 0.5
    unseen = ~observed
    seen_loss = F.binary_cross_entropy(current, target, reduction="none")
    seen_pt = torch.where(target > 0.5, current, 1 - current)
    seen = ((1 - seen_pt).pow(2) * seen_loss * observed).sum() / observed.sum().clamp_min(1)
    intersection = (current * target * observed).sum()
    dice = 1 - (2 * intersection + 1) / ((current * observed).sum() + (target * observed).sum() + 1)
    fix_mask = unseen & (target > 0.5) & (prior <= 0.5)
    keep_mask = unseen & (target > 0.5) & (prior > 0.5)
    fix = _masked_bce(current, target, fix_mask)
    keep = _masked_bce(current, target, keep_mask)
    return {"seen": seen + dice, "fix": fix, "keep": keep}
