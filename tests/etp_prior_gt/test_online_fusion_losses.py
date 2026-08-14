import unittest

import torch

from vlnce_baselines.models.etp_prior_gt.map_fusion import OnlineMapGraphFusion
from vlnce_baselines.models.etp_prior_gt.online_fusion_losses import (
    OnlineFusionLossWeights,
    OnlineFusionTargets,
    compute_online_fusion_losses,
    sparse_grid_valid_mask,
    total_online_fusion_loss,
)


class OnlineFusionLossTest(unittest.TestCase):
    def test_all_losses_are_finite_and_backpropagate(self):
        module = OnlineMapGraphFusion(16, 4, dropout=0.0)
        graph = torch.randn(2, 5, 16)
        graph_masks = torch.tensor(
            [[True, True, True, True, False], [True, True, True, True, True]]
        )
        visited = torch.tensor(
            [[False, True, False, False, False], [False, True, False, False, False]]
        )
        output = module(
            graph,
            graph_masks,
            torch.randn(2, 101, 16),
            torch.ones(2, 101, dtype=torch.bool),
            gmap_visited_masks=visited,
            gmap_step_ids=torch.tensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]),
            txt_embeds=torch.randn(2, 4, 16),
            txt_masks=torch.ones(2, 4, dtype=torch.bool),
            new_evidence_mask=visited,
            decode_dense_grid=True,
        )
        grid_target = torch.zeros(2, 37, 100, 100)
        grid_target[:, 3, 20:24, 30:34] = 1
        visual_target = torch.zeros(2, 5, 37)
        visual_target[:, 1, 3] = 1
        visual_mask = torch.zeros(2, 5, dtype=torch.bool)
        visual_mask[:, 1] = True
        recovery = torch.zeros(2, 5)
        recovery[:, 2] = 1
        recovery_mask = output.ghost_masks.clone()
        losses = compute_online_fusion_losses(
            output,
            OnlineFusionTargets(
                dense_grid=grid_target,
                grid_valid_mask=sparse_grid_valid_mask(grid_target),
                visual=visual_target,
                visual_valid_mask=visual_mask,
                phase=torch.tensor([0, 2]),
                route_state=torch.tensor([0, 1]),
                remaining=torch.tensor([0.8, 0.0]),
                remaining_valid_mask=torch.tensor([True, False]),
                recovery=recovery,
                recovery_valid_mask=recovery_mask,
                expert_action=torch.tensor([2, 3]),
            ),
            OnlineFusionLossWeights(),
        )
        self.assertEqual(set(losses), {"grid", "state", "visual", "progress", "ghost"})
        total = total_online_fusion_loss(losses)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(module.dense_grid_decoder.weight.grad)
        self.assertIsNotNone(module.route_state_head.weight.grad)
        self.assertIsNotNone(module.visual_evidence_head[-1].weight.grad)

    def test_sparse_mask_does_not_trust_every_zero(self):
        target = torch.zeros(37, 100, 100)
        target[2, 50, 50] = 1
        valid = sparse_grid_valid_mask(target, support_radius=1)
        self.assertTrue(valid[2, 50, 50])
        self.assertTrue(valid[:, 49:52, 49:52].all())
        self.assertFalse(valid[:, 0, 0].any())


if __name__ == "__main__":
    unittest.main()
