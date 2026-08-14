import unittest

import torch

from vlnce_baselines.models.etp_prior_gt.map_fusion import (
    OnlineMapFusionOutput,
    OnlineMapGraphFusion,
    build_map_token_fusion,
)


HIDDEN_SIZE = 8
SPATIAL_TOKEN_COUNT = 4


def _fusion() -> OnlineMapGraphFusion:
    return OnlineMapGraphFusion(
        hidden_size=HIDDEN_SIZE,
        num_heads=2,
        dropout=0.0,
        spatial_token_count=SPATIAL_TOKEN_COUNT,
        dense_grid_size=20,
    )


def _inputs():
    torch.manual_seed(7)
    return {
        "gmap_embeds": torch.randn(1, 4, HIDDEN_SIZE),
        "visual_node_embeds": torch.randn(1, 4, HIDDEN_SIZE),
        "gmap_masks": torch.ones(1, 4, dtype=torch.bool),
        "gmap_visited_masks": torch.tensor([[False, True, False, False]]),
        "new_evidence_mask": torch.tensor([[False, True, False, False]]),
        "map_tokens": torch.randn(1, SPATIAL_TOKEN_COUNT + 1, HIDDEN_SIZE),
        "map_token_masks": torch.ones(1, SPATIAL_TOKEN_COUNT + 1, dtype=torch.bool),
    }


def _enable_residual_paths(fusion: OnlineMapGraphFusion) -> None:
    torch.manual_seed(11)
    torch.nn.init.normal_(fusion.map_residual_projection.weight, std=0.1)
    torch.nn.init.normal_(fusion.map_self_residual_projection.weight, std=0.1)
    torch.nn.init.normal_(fusion.graph_residual_projection.weight, std=0.1)


class OnlineMapGraphFusionTest(unittest.TestCase):
    def test_factory_preserves_existing_architectures(self):
        fusion = build_map_token_fusion(
            "online_fusion", hidden_size=16, num_heads=4, dropout=0.0
        )

        self.assertIsInstance(fusion, OnlineMapGraphFusion)
        self.assertEqual(
            build_map_token_fusion(
                "current", hidden_size=16, num_heads=4
            ).__class__.__name__,
            "BidirectionalMapTokenFusion",
        )
        self.assertEqual(
            build_map_token_fusion(
                "try5", hidden_size=16, num_heads=4
            ).__class__.__name__,
            "GraphMapCrossAttention",
        )

    def test_zero_initialized_online_fusion_is_an_exact_identity(self):
        fusion = _fusion()
        inputs = _inputs()

        updated_graph, updated_map = fusion(**inputs)

        self.assertTrue(torch.equal(updated_graph, inputs["gmap_embeds"]))
        self.assertTrue(torch.equal(updated_map, inputs["map_tokens"]))

    def test_visual_node_embeds_are_required_and_exclusive_visual_head_input(self):
        fusion = _fusion()
        inputs = _inputs()

        without_visual = dict(inputs)
        without_visual.pop("visual_node_embeds")
        with self.assertRaisesRegex(ValueError, "visual_node_embeds is required"):
            fusion(**without_visual)

        output = fusion(**inputs)
        expected_logits = fusion.visual_evidence_head(inputs["visual_node_embeds"])
        torch.testing.assert_close(
            output.visual_logits, expected_logits, rtol=0.0, atol=0.0
        )

        changed_navigation = dict(inputs)
        changed_navigation["gmap_embeds"] = inputs["gmap_embeds"] + 1000.0
        changed_output = fusion(**changed_navigation)
        torch.testing.assert_close(
            changed_output.visual_logits,
            output.visual_logits,
            rtol=0.0,
            atol=0.0,
        )

    def test_graph_residual_updates_only_ghost_nodes(self):
        fusion = _fusion()
        inputs = _inputs()
        with torch.no_grad():
            fusion.graph_residual_projection.weight.zero_()
            fusion.graph_residual_projection.bias.fill_(1.0)

        output = fusion(**inputs)

        torch.testing.assert_close(
            output.updated_gmap_embeds[:, :2],
            inputs["gmap_embeds"][:, :2],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            output.updated_gmap_embeds[:, 2:],
            inputs["gmap_embeds"][:, 2:] + 1.0,
            rtol=0.0,
            atol=0.0,
        )

    def test_rich_output_shapes_masks_and_optional_dense_decoder(self):
        fusion = _fusion()
        inputs = _inputs()
        inputs["gmap_step_ids"] = torch.tensor([[0, 2, 0, 0]])
        inputs["txt_embeds"] = torch.randn(1, 3, HIDDEN_SIZE)
        inputs["txt_masks"] = torch.tensor([[True, True, False]])

        output = fusion(**inputs)

        self.assertIsInstance(output, OnlineMapFusionOutput)
        self.assertEqual(output.visual_logits.shape, (1, 4, 37))
        self.assertIsNone(output.dense_grid_logits)
        self.assertEqual(output.phase_logits.shape, (1, 5))
        self.assertEqual(output.route_state_logits.shape, (1, 3))
        self.assertEqual(output.remaining.shape, (1,))
        self.assertEqual(output.recovery_logits.shape, (1, 4))
        self.assertEqual(output.action_logit_residuals.shape, (1, 4))
        self.assertEqual(
            output.map_write_gate.shape,
            (1, SPATIAL_TOKEN_COUNT, HIDDEN_SIZE),
        )
        self.assertEqual(
            output.map_write_residual.shape,
            (1, SPATIAL_TOKEN_COUNT, HIDDEN_SIZE),
        )
        self.assertEqual(
            output.map_state_residual.shape,
            (1, SPATIAL_TOKEN_COUNT, HIDDEN_SIZE),
        )
        self.assertTrue(
            torch.equal(
                output.ghost_masks,
                torch.tensor([[False, False, True, True]]),
            )
        )
        self.assertEqual(output.action_logit_residuals.count_nonzero(), 0)
        self.assertIs(output.map_state, output.updated_map_tokens)
        self.assertIs(output[0], output.updated_gmap_embeds)
        self.assertIs(output[1], output.updated_map_tokens)

        dense_output = fusion(**inputs, decode_dense_grid=True)
        self.assertEqual(dense_output.dense_grid_logits.shape, (1, 37, 20, 20))

    def test_only_visited_entries_can_influence_map_update(self):
        fusion = _fusion()
        _enable_residual_paths(fusion)
        inputs = _inputs()

        _, reference_map = fusion(**inputs)
        changed_nonvisited = dict(inputs)
        changed_nonvisited["gmap_embeds"] = inputs["gmap_embeds"].clone()
        changed_nonvisited["gmap_embeds"][:, 0] += 1000.0  # STOP
        changed_nonvisited["gmap_embeds"][:, 2:] -= 1000.0  # ghosts
        _, nonvisited_map = fusion(**changed_nonvisited)

        changed_visited = dict(inputs)
        changed_visited["gmap_embeds"] = inputs["gmap_embeds"].clone()
        changed_visited["gmap_embeds"][:, 1] += 10.0
        _, visited_map = fusion(**changed_visited)

        torch.testing.assert_close(reference_map, nonvisited_map)
        self.assertFalse(torch.allclose(reference_map, visited_map))

    def test_no_new_evidence_preserves_recurrent_map_exactly(self):
        fusion = _fusion()
        _enable_residual_paths(fusion)
        inputs = _inputs()
        previous_map = torch.randn_like(inputs["map_tokens"])
        previous_map[:, -1] = inputs["map_tokens"][:, -1]
        inputs["previous_map_state"] = previous_map
        inputs["new_evidence_mask"] = torch.zeros_like(inputs["new_evidence_mask"])

        _, updated_map = fusion(**inputs)

        torch.testing.assert_close(updated_map, previous_map, rtol=0.0, atol=0.0)

    def test_empty_visited_set_and_empty_text_mask_are_numerically_safe(self):
        fusion = _fusion()
        _enable_residual_paths(fusion)
        inputs = _inputs()
        inputs["gmap_visited_masks"] = torch.zeros_like(inputs["gmap_visited_masks"])
        inputs["new_evidence_mask"] = torch.zeros_like(inputs["new_evidence_mask"])
        inputs["txt_embeds"] = torch.randn(1, 2, HIDDEN_SIZE)
        inputs["txt_masks"] = torch.zeros(1, 2, dtype=torch.bool)

        output = fusion(**inputs)

        self.assertTrue(torch.isfinite(output.updated_gmap_embeds).all())
        self.assertTrue(torch.isfinite(output.updated_map_tokens).all())
        torch.testing.assert_close(
            output.updated_map_tokens,
            inputs["map_tokens"],
            rtol=0.0,
            atol=0.0,
        )

    def test_updated_map_can_be_carried_into_the_next_step(self):
        fusion = _fusion()
        _enable_residual_paths(fusion)
        inputs = _inputs()
        _, first_map = fusion(**inputs)

        second_inputs = dict(inputs)
        second_inputs["previous_map_state"] = first_map
        second_inputs["gmap_embeds"] = inputs["gmap_embeds"].clone()
        second_inputs["gmap_embeds"][:, 1] *= -2.0
        _, second_map = fusion(**second_inputs)

        self.assertFalse(torch.allclose(first_map[:, :-1], second_map[:, :-1]))
        torch.testing.assert_close(second_map[:, -1], inputs["map_tokens"][:, -1])

    def test_rejects_missing_or_invalid_evidence_masks(self):
        fusion = _fusion()
        inputs = _inputs()

        without_visited = dict(inputs)
        without_visited.pop("gmap_visited_masks")
        with self.assertRaisesRegex(ValueError, "gmap_visited_masks is required"):
            fusion(**without_visited)

        ghost_as_new = dict(inputs)
        ghost_as_new["new_evidence_mask"] = torch.tensor([[False, False, True, False]])
        with self.assertRaisesRegex(ValueError, "subset of valid visited"):
            fusion(**ghost_as_new)

        stop_as_visited = dict(inputs)
        stop_as_visited["gmap_visited_masks"] = torch.tensor([
            [True, True, False, False]
        ])
        stop_as_visited["new_evidence_mask"] = torch.tensor([
            [False, True, False, False]
        ])
        with self.assertRaisesRegex(ValueError, "STOP"):
            fusion(**stop_as_visited)

        changed_metadata = dict(inputs)
        changed_metadata["previous_map_state"] = inputs["map_tokens"].clone()
        changed_metadata["previous_map_state"][:, -1] += 1.0
        with self.assertRaisesRegex(ValueError, "immutable metadata"):
            fusion(**changed_metadata)

    def test_backpropagates_through_map_and_graph_paths(self):
        fusion = _fusion()
        _enable_residual_paths(fusion)
        inputs = _inputs()
        inputs["gmap_embeds"].requires_grad_()
        inputs["visual_node_embeds"].requires_grad_()
        inputs["map_tokens"].requires_grad_()

        output = fusion(**inputs, decode_dense_grid=True)
        loss = (
            output.updated_gmap_embeds.square().mean()
            + output.updated_map_tokens.square().mean()
            + output.visual_logits.square().mean()
            + output.dense_grid_logits.square().mean()
            + output.phase_logits.square().mean()
            + output.recovery_logits.square().mean()
        )
        loss.backward()

        self.assertIsNotNone(inputs["gmap_embeds"].grad)
        self.assertIsNotNone(inputs["visual_node_embeds"].grad)
        self.assertIsNotNone(inputs["map_tokens"].grad)
        self.assertIsNotNone(fusion.map_from_graph_attention.in_proj_weight.grad)
        self.assertIsNotNone(fusion.graph_from_map_attention.in_proj_weight.grad)
        self.assertIsNotNone(fusion.write_gate.weight.grad)
        self.assertIsNotNone(fusion.visual_evidence_head[-1].weight.grad)
        self.assertIsNotNone(fusion.dense_grid_decoder.weight.grad)


if __name__ == "__main__":
    unittest.main()
