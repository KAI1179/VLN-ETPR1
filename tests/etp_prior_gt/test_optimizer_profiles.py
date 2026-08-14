import unittest

import torch
from torch import nn

from vlnce_baselines.models.optimizer_profiles import (
    build_lr_parameter_groups,
    configure_dagger_optimizer_profile,
    configure_pretrain_optimizer_profile,
    optimizer_profile_summary,
)


class _OnlineFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_evidence_head = nn.Linear(2, 2)
        self.visual_category_projection = nn.Linear(2, 2)
        self.map_updater = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)
        self.scale = nn.Parameter(torch.ones(2))


class _PretrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Module()
        self.bert.lang_encoder = nn.Linear(2, 2)
        self.bert.img_embeddings = nn.Linear(2, 2)
        self.bert.global_encoder = nn.Module()
        self.bert.global_encoder.encoder = nn.Linear(2, 2)
        self.bert.global_encoder.graph_map_attention = _OnlineFusion()
        self.map_encoder = nn.Linear(2, 2)
        self.graph_query_text = nn.Linear(2, 2)
        self.global_sap_head = nn.Linear(2, 2)
        self.mlm_head = nn.Linear(2, 2)


class _RuntimePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Module()
        self.net.map_encoder = nn.Linear(2, 2)
        self.net.vln_bert = nn.Module()
        self.net.vln_bert.lang_encoder = nn.Linear(2, 2)
        self.net.vln_bert.img_embeddings = nn.Linear(2, 2)
        self.net.vln_bert.global_encoder = nn.Linear(2, 2)
        self.net.vln_bert.graph_map_attention = _OnlineFusion()
        self.net.vln_bert.graph_query_text = nn.Linear(2, 2)
        self.net.vln_bert.graph_attentioned_txt_embeds_transform = nn.Linear(2, 2)
        self.net.vln_bert.global_sap_head = nn.Linear(2, 2)


class OptimizerProfileTest(unittest.TestCase):
    def test_pretrain_online_fusion_profile_has_two_lr_tiers(self):
        model = _PretrainModel()
        scales = configure_pretrain_optimizer_profile(model, "online_fusion")

        self.assertEqual(scales["map_encoder.weight"], 1.0)
        self.assertEqual(
            scales[
                "bert.global_encoder.graph_map_attention.visual_evidence_head.weight"
            ],
            1.0,
        )
        self.assertEqual(
            scales["bert.global_encoder.graph_map_attention.map_updater.weight"], 1.0
        )
        self.assertEqual(scales["bert.global_encoder.encoder.weight"], 0.1)
        self.assertEqual(scales["global_sap_head.weight"], 0.1)
        self.assertFalse(model.bert.lang_encoder.weight.requires_grad)
        self.assertFalse(model.bert.img_embeddings.weight.requires_grad)

    def test_dagger_online_fusion_profile_has_two_lr_tiers(self):
        policy = _RuntimePolicy()
        scales = configure_dagger_optimizer_profile(policy, "online_fusion")

        self.assertEqual(
            scales["net.vln_bert.graph_map_attention.map_updater.weight"], 1.0
        )
        self.assertEqual(
            scales[
                "net.vln_bert.graph_map_attention.visual_category_projection.weight"
            ],
            1.0,
        )
        self.assertNotIn(
            "net.vln_bert.graph_map_attention.visual_evidence_head.weight",
            scales,
        )
        self.assertFalse(
            policy.net.vln_bert.graph_map_attention.visual_evidence_head.weight.requires_grad
        )
        self.assertEqual(scales["net.map_encoder.weight"], 0.1)
        self.assertEqual(scales["net.vln_bert.global_encoder.weight"], 0.1)
        self.assertFalse(policy.net.vln_bert.lang_encoder.weight.requires_grad)
        self.assertFalse(policy.net.vln_bert.img_embeddings.weight.requires_grad)

    def test_dagger_full_profile_also_freezes_visual_evidence_head(self):
        policy = _RuntimePolicy()
        scales = configure_dagger_optimizer_profile(policy, "full")

        self.assertNotIn(
            "net.vln_bert.graph_map_attention.visual_evidence_head.weight",
            scales,
        )
        self.assertFalse(
            policy.net.vln_bert.graph_map_attention.visual_evidence_head.weight.requires_grad
        )
        self.assertTrue(
            policy.net.vln_bert.graph_map_attention.map_updater.weight.requires_grad
        )

    def test_lr_groups_preserve_scale_and_cover_trainable_parameters(self):
        model = _PretrainModel()
        scales = configure_pretrain_optimizer_profile(model, "online_fusion")
        groups = build_lr_parameter_groups(
            model.named_parameters(),
            scales,
            learning_rate=1e-4,
            weight_decay=0.01,
        )

        self.assertEqual({group["lr_scale"] for group in groups}, {0.1, 1.0})
        self.assertEqual({group["lr"] for group in groups}, {1e-5, 1e-4})
        grouped_ids = [id(param) for group in groups for param in group["params"]]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(
            set(grouped_ids),
            {id(param) for param in model.parameters() if param.requires_grad},
        )
        summary = optimizer_profile_summary(scales, model.named_parameters())
        self.assertIn("parameters=", summary)
        self.assertIn("map_encoder.weight (4)", summary)

    def test_layer_norm_and_all_one_dimensional_parameters_have_no_decay(self):
        model = _PretrainModel()
        scales = configure_pretrain_optimizer_profile(model, "online_fusion")
        groups = build_lr_parameter_groups(
            model.named_parameters(),
            scales,
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        decay_by_name = {
            name: group["weight_decay"]
            for group in groups
            for name in group["profile_names"]
        }

        prefix = "bert.global_encoder.graph_map_attention."
        self.assertEqual(decay_by_name[prefix + "map_updater.weight"], 0.01)
        self.assertEqual(decay_by_name[prefix + "map_updater.bias"], 0.0)
        self.assertEqual(decay_by_name[prefix + "norm.weight"], 0.0)
        self.assertEqual(decay_by_name[prefix + "norm.bias"], 0.0)
        self.assertEqual(decay_by_name[prefix + "scale"], 0.0)

    def test_unknown_optimizer_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown optimizer profile"):
            configure_pretrain_optimizer_profile(_PretrainModel(), "everything")


if __name__ == "__main__":
    unittest.main()
