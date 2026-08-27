from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from pretrain_src.pretrain_src.model.pretrain_cmt import (
    GlocalTextPathCMTPreTraining,
)
from pretrain_src.pretrain_src.data.tasks import mlm_collate, sap_collate
from vlnce_baselines.models.etp_prior_gt.route_map_update import RouteMapState


def test_pretrain_initializes_missing_convolution_and_attention_weights():
    model = GlocalTextPathCMTPreTraining.__new__(GlocalTextPathCMTPreTraining)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(initializer_range=0.02)
    convolution = nn.Conv2d(4, 4, 3)
    convolution.weight.data.fill_(float("nan"))
    convolution.bias.data.fill_(float("nan"))
    model._init_weights(convolution)
    assert all(
        torch.isfinite(parameter).all() for parameter in convolution.parameters()
    )

    attention = nn.MultiheadAttention(8, 2)
    attention.in_proj_weight.data.fill_(float("nan"))
    attention.in_proj_bias.data.fill_(float("nan"))
    model._init_weights(attention)
    assert torch.isfinite(attention.in_proj_weight).all()
    assert torch.isfinite(attention.in_proj_bias).all()


def test_pretrain_preserves_explicit_module_initialization():
    model = GlocalTextPathCMTPreTraining.__new__(GlocalTextPathCMTPreTraining)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(initializer_range=0.02)
    module = nn.Linear(4, 4)
    module.weight.data.fill_(3.0)
    module.bias.data.fill_(2.0)
    module._preserve_manual_init = True

    model._init_weights(module)

    assert torch.equal(module.weight, torch.full_like(module.weight, 3.0))
    assert torch.equal(module.bias, torch.full_like(module.bias, 2.0))


class _TextEmbeddings(nn.Module):
    def forward(self, txt_ids, task_encoding, token_types):
        return torch.zeros(txt_ids.shape[0], txt_ids.shape[1], 4)


class _ImageEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_linear = nn.Identity()
        self.img_layer_norm = nn.Identity()
        self.dep_linear = nn.Identity()
        self.dep_layer_norm = nn.Identity()


class _LanguageEncoder(nn.Module):
    def forward(self, embeddings, mask):
        return embeddings


class _Bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_embeddings = _ImageEmbeddings()
        self.embeddings = _TextEmbeddings()
        self.lang_encoder = _LanguageEncoder()


class _TrackingUpdater(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def initial_state(self, prior):
        return RouteMapState(
            prior=prior,
            current=prior,
            evidence=torch.zeros_like(prior),
            coverage=torch.zeros_like(prior[:, :1]),
        )

    def update(
        self,
        state,
        pano_features,
        text_features,
        positions,
        rotations,
        world_starts,
        map_starts,
    ):
        self.calls.append((positions.clone(), rotations.clone()))
        is_negative = len(self.calls) == 2
        next_state = RouteMapState(
            prior=state.prior,
            current=state.current + float(is_negative),
            evidence=state.evidence + 2.0 * float(is_negative),
            coverage=state.coverage + 3.0 * float(is_negative),
        )
        return next_state, {
            "semantic_logits": torch.zeros(1, 12, 37, 5),
            "coverage_logits": torch.zeros(1, 12, 5),
            "route_gate_logits": torch.zeros(1),
        }


def _model():
    model = GlocalTextPathCMTPreTraining.__new__(GlocalTextPathCMTPreTraining)
    nn.Module.__init__(model)
    model.bert = _Bert()
    model.route_map_updater = _TrackingUpdater()
    model.config = SimpleNamespace(
        pose_gated_visual_loss_weight=0.0,
        pose_gated_route_loss_weight=1.0,
        pose_gated_seen_loss_weight=0.0,
        pose_gated_fix_loss_weight=0.0,
        pose_gated_keep_loss_weight=0.0,
    )
    return model


def _batch(negative_valid):
    return {
        "traj_spatial_view_fts": torch.zeros(1, 12, 4),
        "traj_spatial_dep_fts": torch.zeros(1, 12, 4),
        "txt_ids": torch.ones(1, 2, dtype=torch.long),
        "txt_task_encoding": torch.ones(1, 2, dtype=torch.long),
        "txt_lens": torch.tensor([2]),
        "traj_step_lens": [1],
        "cognitive_maps": torch.zeros(1, 37, 2, 2),
        "traj_positions": torch.tensor([[1.0, 2.0, 3.0]]),
        "traj_rotations": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "start_positions": torch.zeros(1, 2),
        "target_cognitive_maps": torch.zeros(1, 37, 2, 2),
        "spatial_semantic_targets": torch.zeros(1, 12, 37, 5),
        "spatial_coverage_targets": torch.zeros(1, 12, 5),
        "route_negative_valid": torch.tensor([negative_valid]),
        "route_negative_spatial_view_fts": torch.zeros(1, 12, 4),
        "route_negative_spatial_dep_fts": torch.zeros(1, 12, 4),
        "route_negative_position": torch.tensor([[9.0, 8.0, 7.0]]),
        "route_negative_rotation": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    }


def _collate_input(task):
    item = {
        "txt_ids": torch.ones(2, dtype=torch.long),
        "txt_task_encoding": torch.ones(2, dtype=torch.long),
        "gmap_task_embeddings": torch.ones(1, dtype=torch.long),
        "traj_view_img_fts": [torch.zeros(1, 4)],
        "traj_view_dep_fts": [torch.zeros(1, 4)],
        "traj_loc_fts": [torch.zeros(1, 4)],
        "traj_nav_types": [torch.zeros(1, dtype=torch.long)],
        "gmap_step_ids": torch.zeros(1, dtype=torch.long),
        "gmap_visited_masks": torch.zeros(1, dtype=torch.bool),
        "gmap_pos_fts": torch.zeros(1, 4),
        "gmap_pair_dists": torch.zeros(1, 1),
        "cognitive_maps": torch.zeros(37, 2, 2),
        "trajectory_keypoints": torch.zeros(5, 2),
        "map_trajectory_metadata": torch.zeros(5, 2),
        "start_direction_vectors": torch.zeros(2),
        "start_positions": torch.zeros(2),
        "traj_spatial_view_fts": [torch.zeros(12, 4)],
        "traj_spatial_dep_fts": [torch.zeros(12, 4)],
        "spatial_semantic_targets": [torch.zeros(12, 37, 5)],
        "spatial_coverage_targets": [torch.zeros(12, 5)],
        "traj_positions": [torch.zeros(3)],
        "traj_rotations": [torch.tensor((0.0, 0.0, 0.0, 1.0))],
        "target_cognitive_maps": torch.zeros(37, 2, 2),
        "route_negative_valid": True,
        "route_negative_spatial_view_fts": torch.zeros(12, 4),
        "route_negative_spatial_dep_fts": torch.zeros(12, 4),
        "route_negative_position": torch.tensor((9.0, 8.0, 7.0)),
        "route_negative_rotation": torch.tensor((0.0, 0.0, 0.0, 1.0)),
    }
    if task == "mlm":
        item["txt_labels"] = torch.ones(2, dtype=torch.long)
    else:
        item["local_act_labels"] = 0
        item["global_act_labels"] = 0
    return item


@pytest.mark.parametrize(("collate", "task"), ((mlm_collate, "mlm"), (sap_collate, "sap")))
def test_pose_gated_collates_stack_route_negative_pose(collate, task):
    batch = collate([_collate_input(task)])

    assert torch.equal(
        batch["route_negative_position"], torch.tensor([[9.0, 8.0, 7.0]])
    )
    assert torch.equal(
        batch["route_negative_rotation"],
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )


def test_route_negative_runs_full_update_and_penalizes_state_change():
    baseline_model = _model()
    _, baseline_loss = baseline_model._pose_gated_map_update(
        _batch(negative_valid=False), compute_loss=True
    )

    model = _model()
    _, loss = model._pose_gated_map_update(
        _batch(negative_valid=True), compute_loss=True
    )

    assert len(model.route_map_updater.calls) == 2
    negative_position, negative_rotation = model.route_map_updater.calls[-1]
    assert torch.equal(negative_position, torch.tensor([[9.0, 8.0, 7.0]]))
    assert torch.equal(
        negative_rotation, torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    )
    assert (loss - baseline_loss).item() == pytest.approx(6.0)
