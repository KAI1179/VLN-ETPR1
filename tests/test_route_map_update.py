import torch
import torch.nn as nn

from vlnce_baselines.models.etp_prior_gt.route_map_update import (
    NUM_AZIMUTHS,
    NUM_DISTANCE_BINS,
    NUM_MAP_CATEGORIES,
    RouteMapUpdater,
)


def test_project_evidence_converts_meter_start_to_grid_cells():
    updater = RouteMapUpdater(hidden_size=4)
    semantic_logits = torch.zeros(
        1, NUM_AZIMUTHS, NUM_MAP_CATEGORIES, NUM_DISTANCE_BINS
    )
    semantic_logits[0, 0, 0, 0] = 10
    coverage_logits = torch.zeros(1, NUM_AZIMUTHS, NUM_DISTANCE_BINS)

    evidence, _ = updater._project_evidence(
        semantic_logits,
        coverage_logits,
        positions=torch.tensor([[100.0, 0.0, 200.0]]),
        rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        world_starts=torch.tensor([[100.0, 0.0, 200.0]]),
        map_starts=torch.tensor([[5.0, 10.0]]),
        grid_size=40,
    )

    assert evidence[0, 0, 8, 20] > 0.99


class _VisualHead(nn.Module):
    def forward(self, pano_features):
        batch_size = pano_features.shape[0]
        return (
            pano_features.new_zeros(
                batch_size,
                NUM_AZIMUTHS,
                NUM_MAP_CATEGORIES,
                NUM_DISTANCE_BINS,
            ),
            pano_features.new_full(
                (batch_size, NUM_AZIMUTHS, NUM_DISTANCE_BINS), 10
            ),
        )


class _ClosedGate(nn.Module):
    def forward(self, pano_features, text_features, map_state):
        return pano_features.new_full((pano_features.shape[0],), -1000)


class _Refiner(nn.Module):
    def forward(self, prior, evidence, coverage):
        return torch.zeros_like(prior), torch.zeros_like(coverage)


def test_closed_gate_does_not_accumulate_evidence_or_coverage():
    updater = RouteMapUpdater(hidden_size=4)
    updater.visual_head = _VisualHead()
    updater.route_gate = _ClosedGate()
    updater.refiner = _Refiner()
    state = updater.initial_state(torch.full((1, NUM_MAP_CATEGORIES, 24, 24), 0.5))

    updated, _ = updater.update(
        state,
        pano_features=torch.zeros(1, NUM_AZIMUTHS, 4),
        text_features=torch.zeros(1, 4),
        positions=torch.zeros(1, 3),
        rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        world_starts=torch.zeros(1, 3),
        map_starts=torch.tensor([[5.0, 5.0]]),
    )

    assert torch.equal(updated.current, state.current)
    assert torch.equal(updated.evidence, state.evidence)
    assert torch.equal(updated.coverage, state.coverage)
