import unittest

import torch

from vlnce_baselines.models.etp_prior_gt.map_fusion import (
    OnlineMapFusionOutput,
    OnlineMapGraphFusion,
)
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
        visual_graph = torch.randn(2, 5, 16)
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
            visual_node_embeds=visual_graph,
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
        self.assertEqual(
            set(losses),
            {"grid", "state", "visual", "progress", "recovery", "ghost"},
        )
        total = total_online_fusion_loss(losses)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(module.dense_grid_decoder.weight.grad)
        self.assertIsNotNone(module.route_state_head.weight.grad)
        self.assertIsNotNone(module.visual_evidence_head[-1].weight.grad)

    def test_recovery_and_progress_losses_have_independent_output_gradients(self):
        phase_logits = torch.zeros(1, 5, requires_grad=True)
        route_state_logits = torch.zeros(1, 3, requires_grad=True)
        remaining = torch.tensor([0.5], requires_grad=True)
        recovery_logits = torch.zeros(1, 3, requires_grad=True)
        output = OnlineMapFusionOutput(
            updated_gmap_embeds=torch.zeros(1, 3, 4),
            updated_map_tokens=torch.zeros(1, 2, 4),
            visual_logits=torch.zeros(1, 3, 37),
            dense_grid_logits=None,
            phase_logits=phase_logits,
            route_state_logits=route_state_logits,
            remaining=remaining,
            recovery_logits=recovery_logits,
            action_logit_residuals=torch.zeros(1, 3),
            ghost_logit_residuals=torch.zeros(1, 3),
            stop_logit_residuals=torch.zeros(1, 3),
            map_write_gate=torch.zeros(1, 1, 4),
            map_write_residual=torch.zeros(1, 1, 4),
            map_state_residual=torch.zeros(1, 1, 4),
            ghost_masks=torch.tensor([[False, True, True]]),
        )
        targets = OnlineFusionTargets(
            phase=torch.tensor([1]),
            route_state=torch.tensor([2]),
            remaining=torch.tensor([0.0]),
            remaining_valid_mask=torch.tensor([True]),
            recovery=torch.tensor([[0.0, 1.0, 0.0]]),
            recovery_valid_mask=torch.tensor([[False, True, True]]),
        )
        progress_weights = OnlineFusionLossWeights(
            grid=0.0,
            state=0.0,
            visual=0.0,
            progress=1.0,
            recovery=0.0,
            ghost=0.0,
        )
        recovery_weights = OnlineFusionLossWeights(
            grid=0.0,
            state=0.0,
            visual=0.0,
            progress=0.0,
            recovery=1.0,
            ghost=0.0,
        )

        progress_losses = compute_online_fusion_losses(
            output, targets, progress_weights
        )
        recovery_losses = compute_online_fusion_losses(
            output, targets, recovery_weights
        )
        self.assertEqual(set(progress_losses), {"progress"})
        self.assertEqual(set(recovery_losses), {"recovery"})

        heads = (phase_logits, route_state_logits, remaining, recovery_logits)
        progress_gradients = torch.autograd.grad(
            progress_losses["progress"], heads, allow_unused=True
        )
        recovery_gradients = torch.autograd.grad(
            recovery_losses["recovery"], heads, allow_unused=True
        )
        for gradient in progress_gradients[:3]:
            self.assertIsNotNone(gradient)
            self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assertIsNone(progress_gradients[3])
        for gradient in recovery_gradients[:3]:
            self.assertIsNone(gradient)
        self.assertIsNotNone(recovery_gradients[3])
        self.assertGreater(recovery_gradients[3].abs().sum().item(), 0.0)

    def test_sparse_mask_does_not_trust_every_zero(self):
        target = torch.zeros(37, 100, 100)
        target[2, 50, 50] = 1
        valid = sparse_grid_valid_mask(target, support_radius=1)
        self.assertTrue(valid[2, 50, 50])
        self.assertTrue(valid[:, 49:52, 49:52].all())
        self.assertFalse(valid[:, 0, 0].any())


if __name__ == "__main__":
    unittest.main()
