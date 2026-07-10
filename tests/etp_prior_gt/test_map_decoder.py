import importlib
import sys
from pathlib import Path

import pytest
import torch

from prior import bbox
from vlnce_baselines.models.etp_prior_gt.map_box_targets import (
    relevant_semantic_boxes_to_decoder_target,
)
from vlnce_baselines.models.etp_prior_gt.map_decoder import (
    CognitiveMapDecoder,
    CognitiveMapSetCriterion,
    HungarianBoxMatcher,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import NUM_MAP_CATEGORIES

ROOT = Path(__file__).resolve().parents[2]


def test_cognitive_map_decoder_predicts_box_set_from_updated_tokens():
    decoder = CognitiveMapDecoder(hidden_size=32, num_queries=11)
    updated_map_tokens = torch.randn(2, 101, 32)

    output = decoder(updated_map_tokens)

    assert output.pred_logits.shape == (2, 11, NUM_MAP_CATEGORIES + 1)
    assert output.pred_boxes.shape == (2, 11, 4)
    assert torch.isfinite(output.pred_logits).all()
    assert torch.all((0.0 <= output.pred_boxes) & (output.pred_boxes <= 1.0))


def test_cognitive_map_decoder_rejects_missing_metadata_token():
    decoder = CognitiveMapDecoder(hidden_size=32)
    updated_map_tokens = torch.randn(2, 100, 32)

    with pytest.raises(ValueError, match="updated_map_tokens"):
        decoder(updated_map_tokens)


def test_relevant_semantic_boxes_to_decoder_target_uses_object_envelope_and_region_box():
    level = bbox.LevelSemanticBoxes(
        objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
        range_y=[None, None],
    )
    level.objects[3].append(
        bbox.OBB2D(
            center=(10.0, 20.0),
            half_extents=(1.0, 2.0),
            rotation=0.0,
        )
    )
    level.regions[7].append(bbox.AABB2D(min=(4.0, 6.0), max=(14.0, 16.0)))
    relevant = bbox.RelevantSemanticBoxes(
        level_idx=0,
        level=level,
        instruction="go to the table in the bathroom",
        ground_truth_trajectory=[(0.0, 0.0), (1.0, 1.0)],
        trajectory_keypoints=[
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        start_direction_vector=(0.0, 1.0),
    )

    target = relevant_semantic_boxes_to_decoder_target(relevant)

    assert torch.equal(target.labels, torch.tensor([3, bbox.OBJECT_CATEGORIES + 7]))
    torch.testing.assert_close(
        target.boxes,
        torch.tensor(
            [
                [0.20, 0.40, 0.04, 0.08],
                [0.18, 0.22, 0.20, 0.20],
            ]
        ),
    )


def test_relevant_semantic_boxes_to_decoder_target_converts_rotated_obb_envelope():
    level = bbox.LevelSemanticBoxes(
        objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
        range_y=[None, None],
    )
    level.objects[1].append(
        bbox.OBB2D(
            center=(10.0, 20.0),
            half_extents=(1.0, 2.0),
            rotation=torch.pi / 2.0,
        )
    )
    relevant = bbox.RelevantSemanticBoxes(
        level_idx=0,
        level=level,
        instruction="go to the chair",
        ground_truth_trajectory=[(0.0, 0.0), (1.0, 1.0)],
        trajectory_keypoints=[
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        start_direction_vector=(0.0, 1.0),
    )

    target = relevant_semantic_boxes_to_decoder_target(relevant)

    torch.testing.assert_close(
        target.boxes,
        torch.tensor([[0.20, 0.40, 0.08, 0.04]]),
        atol=1e-6,
        rtol=0.0,
    )


def test_cognitive_map_set_criterion_matches_and_penalizes_boxes():
    matcher = HungarianBoxMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0)
    criterion = CognitiveMapSetCriterion(
        num_classes=NUM_MAP_CATEGORIES,
        matcher=matcher,
        eos_coef=0.1,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
    )
    output = CognitiveMapDecoder(hidden_size=8, num_queries=2)(torch.randn(1, 101, 8))
    targets = [
        {
            "labels": torch.tensor([1], dtype=torch.int64),
            "boxes": torch.tensor([[0.50, 0.50, 0.20, 0.20]]),
        }
    ]

    loss = criterion(output, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss > 0


def test_pretraining_updated_map_loss_requires_box_targets():
    pretrain_root = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_root) not in sys.path:
        sys.path.insert(0, str(pretrain_root))
    pretrain_cmt = importlib.import_module("model.pretrain_cmt")
    model = pretrain_cmt.GlocalTextPathCMTPreTraining.__new__(
        pretrain_cmt.GlocalTextPathCMTPreTraining
    )
    torch.nn.Module.__init__(model)
    model.map_decoder = CognitiveMapDecoder(hidden_size=8, num_queries=2)
    model.map_box_criterion = CognitiveMapSetCriterion()
    model.map_loss_weight = 0.1

    with pytest.raises(
        ValueError,
        match="cognitive_map_box_targets are required",
    ):
        model._compute_updated_cognitive_map_loss(
            torch.randn(1, 101, 8),
            None,
            True,
        )


def test_pretraining_updated_map_loss_uses_box_criterion():
    pretrain_root = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_root) not in sys.path:
        sys.path.insert(0, str(pretrain_root))
    pretrain_cmt = importlib.import_module("model.pretrain_cmt")
    model = pretrain_cmt.GlocalTextPathCMTPreTraining.__new__(
        pretrain_cmt.GlocalTextPathCMTPreTraining
    )
    torch.nn.Module.__init__(model)

    class StubDecoder(torch.nn.Module):
        def forward(self, updated_map_tokens):
            assert updated_map_tokens.shape == (1, 101, 8)
            return "decoded"

    class StubCriterion(torch.nn.Module):
        def forward(self, output, targets):
            assert output == "decoded"
            assert targets == ["target"]
            return torch.tensor(3.0)

    model.map_decoder = StubDecoder()
    model.map_box_criterion = StubCriterion()
    model.map_loss_weight = 0.1

    loss = model._compute_updated_cognitive_map_loss(
        torch.randn(1, 101, 8),
        ["target"],
        True,
    )

    torch.testing.assert_close(loss, torch.tensor(0.3))
