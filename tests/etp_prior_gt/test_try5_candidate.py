from pathlib import Path

from gym.spaces import Dict, Discrete
import torch


ROOT = Path(__file__).resolve().parents[2]


def test_priorgt_try5_policy_forces_try5_fusion(monkeypatch):
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp_prior_gt import policy as policy_module

    captured = {}

    class FakeETP(torch.nn.Module):
        def __init__(self, observation_space, model_config, num_actions, dropout_rate):
            super().__init__()
            captured["fusion"] = model_config.MAP_ENCODER.fusion
            captured["metadata_schema"] = model_config.MAP_ENCODER.metadata_schema

    monkeypatch.setattr(policy_module, "ETP_PriorGT", FakeETP)

    config = get_config()
    config.defrost()
    config.TORCH_GPU_ID = 0
    config.MODEL.MAP_ENCODER.enabled = True
    config.MODEL.MAP_ENCODER.fusion = "bidirectional"
    config.freeze()

    policy_module.PriorGTTry5Policy.from_config(
        config,
        observation_space=Dict({}),
        action_space=Discrete(4),
    )

    assert captured["fusion"] == "try5"
    assert captured["metadata_schema"] == "direction5"


def test_main_server_exposes_priorgt_try5_candidate_modes():
    script = (ROOT / "run_r2r" / "main_server.bash").read_text()

    assert "GT_MAP_FUSION" not in script
    assert "PRIORGT_TRY5_MODEL_ARGS" in script
    assert "MODEL.policy_name PriorGTTry5Policy" in script
    assert "MODEL.MAP_ENCODER.cache_namespace gt.legacy.r1p5.direction5.v1" in script
    assert "MODEL.MAP_ENCODER.metadata_schema direction5" in script
    assert "priorgt_try5_dagger)" in script
    assert "priorgt_try5_grpo)" in script
    assert "priorgt_try5_eval_ss)" in script
    assert "priorgt_try5_eval_grpo)" in script
