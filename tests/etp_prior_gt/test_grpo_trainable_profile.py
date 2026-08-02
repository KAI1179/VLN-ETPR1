from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vlnce_baselines.config.default import get_config
from vlnce_baselines.GRPO_trainer_ETP_PriorGT import RLTrainer
from vlnce_baselines.models.etp_prior_gt.map_fusion import GraphMapCrossAttention


class _TinyVlnBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.global_encoder = nn.Linear(2, 2)
        self.graph_query_text = nn.Linear(2, 2)
        self.graph_attentioned_txt_embeds_transform = nn.Linear(2, 2)
        self.global_sap_head = nn.Linear(2, 2)
        self.graph_map_attention = nn.Linear(2, 2)
        self.frozen_module = nn.Linear(2, 2)


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vln_bert = _TinyVlnBert()
        self.map_encoder = nn.Linear(2, 2)


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = _TinyNet()


def _trainer(profile: str, *, map_enabled: bool = True, architecture: str = "try5"):
    trainer = RLTrainer.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        GRPO=SimpleNamespace(trainable_profile=profile),
        MODEL=SimpleNamespace(
            MAP_ENCODER=SimpleNamespace(
                enabled=map_enabled,
                architecture=architecture,
            )
        ),
    )
    trainer.policy = _TinyPolicy()
    trainer.trainable_parts = trainer._select_trainable_parts()
    trainer.setup_training_parts()
    return trainer


def _trainable_names(trainer: RLTrainer) -> set[str]:
    policy = trainer.policy
    if policy is None:
        raise AssertionError("test trainer has no policy")
    return {
        name for name, param in policy.named_parameters() if param.requires_grad
    }


def test_default_profile_is_nav4_and_dead_train_all_is_absent() -> None:
    config = get_config()

    assert config.GRPO.trainable_profile == "nav4"
    assert config.GRPO.require_complete_checkpoint is False
    assert "train_all" not in config.GRPO


def test_nav4_selects_only_existing_navigation_modules() -> None:
    trainer = _trainer("nav4")

    assert all(
        name.startswith(
            (
                "net.vln_bert.global_encoder.",
                "net.vln_bert.graph_query_text.",
                "net.vln_bert.graph_attentioned_txt_embeds_transform.",
                "net.vln_bert.global_sap_head.",
            )
        )
        for name in _trainable_names(trainer)
    )
    assert len(_trainable_names(trainer)) == 8


def test_nav4_fusion_adds_only_graph_map_attention() -> None:
    trainer = _trainer("nav4-fusion")
    trainable_names = _trainable_names(trainer)

    assert {
        name
        for name in trainable_names
        if name.startswith("net.vln_bert.graph_map_attention.")
    } == {
        "net.vln_bert.graph_map_attention.bias",
        "net.vln_bert.graph_map_attention.weight",
    }
    assert len(trainable_names) == 10
    assert not any(name.startswith("net.map_encoder.") for name in trainable_names)
    assert not any(name.startswith("net.vln_bert.frozen_module.") for name in trainable_names)


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 'nav4' or 'nav4-fusion'"):
        _trainer("everything")


@pytest.mark.parametrize(
    ("map_enabled", "architecture"),
    [(False, "try5"), (True, "current")],
)
def test_nav4_fusion_requires_enabled_try5_map(
    map_enabled: bool,
    architecture: str,
) -> None:
    with pytest.raises(ValueError, match="requires an enabled Try5 cognitive map"):
        _trainer(
            "nav4-fusion",
            map_enabled=map_enabled,
            architecture=architecture,
        )


def test_optimizer_membership_must_match_trainable_parameters() -> None:
    trainer = _trainer("nav4-fusion")
    trainable = [param for param in trainer.policy.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable)

    trainer._validate_optimizer_membership(trainer.policy, optimizer)

    missing_optimizer = torch.optim.AdamW(trainable[:-1])
    with pytest.raises(RuntimeError, match="do not match"):
        trainer._validate_optimizer_membership(trainer.policy, missing_optimizer)

    duplicate_optimizer = torch.optim.AdamW([{"params": [*trainable, trainable[0]]}])
    with pytest.raises(RuntimeError, match="duplicate"):
        trainer._validate_optimizer_membership(trainer.policy, duplicate_optimizer)


def test_complete_checkpoint_gate_rejects_missing_or_unexpected_keys() -> None:
    trainer = _trainer("nav4")
    trainer.config.GRPO.require_complete_checkpoint = True

    with pytest.raises(RuntimeError, match="missing=.*map_encoder"):
        trainer._validate_checkpoint_compatibility(
            SimpleNamespace(
                missing_keys=["net.map_encoder.weight"],
                unexpected_keys=[],
            )
        )

    trainer._validate_checkpoint_compatibility(
        SimpleNamespace(missing_keys=[], unexpected_keys=[])
    )


def test_try5_fusion_backward_leaves_map_encoder_frozen() -> None:
    fusion = GraphMapCrossAttention(hidden_size=8, num_heads=2, dropout=0.0)
    map_encoder = nn.Linear(8, 8)
    map_encoder.requires_grad_(False)
    map_tokens = map_encoder(torch.randn(2, 5, 8))

    updated_graph, _ = fusion(
        torch.randn(2, 4, 8),
        torch.ones(2, 4, dtype=torch.bool),
        map_tokens,
        torch.ones(2, 5, dtype=torch.bool),
    )
    updated_graph.square().mean().backward()

    assert any(
        param.grad is not None and torch.count_nonzero(param.grad)
        for param in fusion.parameters()
    )
    assert all(param.grad is None for param in map_encoder.parameters())
