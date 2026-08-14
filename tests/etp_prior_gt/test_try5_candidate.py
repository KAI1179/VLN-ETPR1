from pathlib import Path

from gym.spaces import Dict, Discrete
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def test_priorgt_try5_policy_selects_try5_fusion(monkeypatch):
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp_prior_gt import policy as policy_module

    captured = {}

    class FakeETP(torch.nn.Module):
        def __init__(self, observation_space, model_config, num_actions, dropout_rate):
            super().__init__()
            captured["architecture"] = model_config.MAP_ENCODER.architecture
            captured["source"] = model_config.MAP_ENCODER.source

    monkeypatch.setattr(policy_module, "ETP_PriorGT", FakeETP)

    config = get_config()
    config.defrost()
    config.TORCH_GPU_ID = 0
    config.MODEL.MAP_ENCODER.enabled = True
    config.MODEL.MAP_ENCODER.architecture = "try5"
    config.freeze()

    policy_module.PriorGTTry5Policy.from_config(
        config,
        observation_space=Dict({}),
        action_space=Discrete(4),
    )

    assert captured["architecture"] == "try5"
    assert captured["source"] == "prior_gt"


def test_llm_candidate_policies_select_complete_candidate(monkeypatch):
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp_prior_gt import policy as policy_module
    from vlnce_baselines.models.etp_llm.policy import (
        LLMBoxesCurrentPolicy,
        LLMGridOnlineFusionPolicy,
        LLMGridTry5Policy,
    )

    captured = []

    class FakeETP(torch.nn.Module):
        def __init__(self, observation_space, model_config, num_actions, dropout_rate):
            super().__init__()
            captured.append(
                (
                    model_config.MAP_ENCODER.architecture,
                    model_config.MAP_ENCODER.source,
                )
            )

    monkeypatch.setattr(policy_module, "ETP_PriorGT", FakeETP)
    for policy in (
        LLMBoxesCurrentPolicy,
        LLMGridTry5Policy,
        LLMGridOnlineFusionPolicy,
    ):
        config = get_config()
        config.defrost()
        config.TORCH_GPU_ID = 0
        config.MODEL.MAP_ENCODER.enabled = True
        config.MODEL.MAP_ENCODER.architecture = policy.navigation_architecture
        config.MODEL.MAP_ENCODER.source = policy.cognitive_map_source
        config.freeze()
        policy.from_config(config, Dict({}), Discrete(4))

    assert captured == [
        ("current", "llm_boxes"),
        ("try5", "llm_grid"),
        ("online_fusion", "llm_grid"),
    ]


def test_policy_rejects_mismatched_candidate_config():
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp_prior_gt.policy import PriorGTTry5Policy

    config = get_config()
    config.defrost()
    config.TORCH_GPU_ID = 0
    config.MODEL.MAP_ENCODER.enabled = True
    config.freeze()

    with pytest.raises(ValueError, match="PriorGTTry5Policy requires architecture=try5"):
        PriorGTTry5Policy.from_config(config, Dict({}), Discrete(4))


def test_main_server_exposes_priorgt_try5_candidate_modes():
    script = (ROOT / "run_r2r" / "main_server.bash").read_text()

    assert "GT_MAP_FUSION" not in script
    assert "PRIORGT_TRY5_BLURRED_MODEL_ARGS" in script
    assert "MODEL.policy_name PriorGTTry5Policy" in script
    assert "gt.legacy.r1p5.direction5.blurred.v1" in script
    assert "MODEL.MAP_ENCODER.architecture try5" in script
    assert "MODEL.MAP_ENCODER.source prior_gt" in script
    assert "prior_gt_try5_blurred_dagger)" in script
    assert "prior_gt_try5_blurred_grpo)" in script
    assert "prior_gt_try5_blurred_eval_dagger)" in script
    assert "prior_gt_try5_blurred_eval_grpo)" in script
