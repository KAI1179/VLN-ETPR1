import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _install_fake_clip(monkeypatch):
    fake_clip = types.ModuleType("clip")

    class FakeClipModel(torch.nn.Module):
        def eval(self):
            return self

        def encode_text(self, tokens):
            n = tokens.size(0)
            return torch.arange(n * 512, dtype=torch.float32).view(n, 512) + 1.0

    fake_clip.load = lambda *args, **kwargs: (FakeClipModel(), None)
    fake_clip.tokenize = lambda prompts: torch.zeros(len(prompts), 77, dtype=torch.long)
    monkeypatch.setitem(sys.modules, "clip", fake_clip)


def _load_imagined_policy(monkeypatch):
    _install_fake_clip(monkeypatch)
    fake_baseline_registry = types.ModuleType("habitat_baselines.common.baseline_registry")

    class FakeRegistry:
        def register_policy(self, cls):
            return cls

    fake_baseline_registry.baseline_registry = FakeRegistry()
    monkeypatch.setitem(
        sys.modules,
        "habitat_baselines.common.baseline_registry",
        fake_baseline_registry,
    )

    fake_gym = types.ModuleType("gym")
    fake_gym.Space = object
    monkeypatch.setitem(sys.modules, "gym", fake_gym)
    fake_habitat = types.ModuleType("habitat")
    fake_habitat.Config = object
    monkeypatch.setitem(sys.modules, "habitat", fake_habitat)

    fake_ppo_policy = types.ModuleType("habitat_baselines.rl.ppo.policy")
    fake_ppo_policy.Net = torch.nn.Module
    monkeypatch.setitem(sys.modules, "habitat_baselines.rl.ppo.policy", fake_ppo_policy)

    fake_policy_module = types.ModuleType("vlnce_baselines.models.policy")

    class FakeILPolicy:
        def __init__(self, net, dim_actions):
            self.net = net
            self.dim_actions = dim_actions

    fake_policy_module.ILPolicy = FakeILPolicy
    monkeypatch.setitem(sys.modules, "vlnce_baselines.models.policy", fake_policy_module)

    fake_vlnbert_init = types.ModuleType("vlnce_baselines.models.etp_prior_gt.vlnbert_init")
    fake_vlnbert_init.get_vlnbert_models = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "vlnce_baselines.models.etp_prior_gt.vlnbert_init",
        fake_vlnbert_init,
    )
    fake_priorgt_policy = types.ModuleType("vlnce_baselines.models.etp_prior_gt.policy")

    class FakeETPPriorGT(torch.nn.Module):
        def forward(self, mode=None, **kwargs):
            if mode == "map_encoding":
                return self.map_encoder(
                    kwargs["cognitive_crops"],
                    kwargs["direction_vectors"],
                    kwargs["start_direction_vectors"],
                    kwargs["start_positions"],
                )
            raise NotImplementedError(mode)

    fake_priorgt_policy.ETP_PriorGT = FakeETPPriorGT
    monkeypatch.setitem(
        sys.modules,
        "vlnce_baselines.models.etp_prior_gt.policy",
        fake_priorgt_policy,
    )

    for name in [
        "vlnce_baselines",
        "vlnce_baselines.models",
        "vlnce_baselines.models.etp_prior_gt",
        "vlnce_baselines.models.etp_imagined",
    ]:
        module = sys.modules.get(name, types.ModuleType(name))
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    for mod_name, rel_path in [
        (
            "vlnce_baselines.models.etp_prior_gt.map_utils",
            "vlnce_baselines/models/etp_prior_gt/map_utils.py",
        ),
        (
            "vlnce_baselines.models.etp_prior_gt.map_encoder",
            "vlnce_baselines/models/etp_prior_gt/map_encoder.py",
        ),
        (
            "vlnce_baselines.models.etp_imagined.instruction_map_predictor",
            "vlnce_baselines/models/etp_imagined/instruction_map_predictor.py",
        ),
        (
            "vlnce_baselines.models.etp_imagined.policy",
            "vlnce_baselines/models/etp_imagined/policy.py",
        ),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, mod_name, module)
        spec.loader.exec_module(module)

    return sys.modules["vlnce_baselines.models.etp_imagined.policy"]


def test_etp_imagined_encodes_map_tokens_from_text(monkeypatch):
    policy_module = _load_imagined_policy(monkeypatch)
    net = policy_module.ETP_Imagined.__new__(policy_module.ETP_Imagined)
    torch.nn.Module.__init__(net)
    net.map_encoder = policy_module.EmbeddingGridMapEncoder(hidden_size=32)
    net.map_predictor = policy_module.InstructionCognitiveMapPredictor(
        hidden_size=32,
        num_heads=4,
        num_layers=1,
    )
    txt_embeds = torch.randn(2, 5, 32)
    txt_masks = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )

    logits, map_tokens, map_token_masks = net.forward(
        mode="imagined_map_encoding",
        txt_embeds=txt_embeds,
        txt_masks=txt_masks,
    )

    assert logits.shape == (2, policy_module.NUM_MAP_CATEGORIES, policy_module.SIZE, policy_module.SIZE)
    assert map_tokens.shape == (2, 101, 32)
    assert map_token_masks.shape == (2, 101)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(map_tokens).all()


def test_etp_imagined_splits_prediction_from_gt_map_encoding(monkeypatch):
    policy_module = _load_imagined_policy(monkeypatch)
    net = policy_module.ETP_Imagined.__new__(policy_module.ETP_Imagined)
    torch.nn.Module.__init__(net)
    net.map_encoder_enabled = True
    net.map_encoder = policy_module.EmbeddingGridMapEncoder(hidden_size=32)
    net.map_predictor = policy_module.InstructionCognitiveMapPredictor(
        hidden_size=32,
        num_heads=4,
        num_layers=1,
    )
    txt_embeds = torch.randn(2, 5, 32)
    txt_masks = torch.ones(2, 5, dtype=torch.bool)

    map_logits = net.forward(
        mode="predict_cognitive_map",
        txt_embeds=txt_embeds,
        txt_masks=txt_masks,
    )
    pred_grid = torch.sigmoid(map_logits)
    map_tokens, map_token_masks = net.forward(
        mode="map_encoding",
        cognitive_crops=pred_grid,
        direction_vectors=torch.zeros(2, policy_module.DIRECTION_VECTOR_CNT, 2),
        start_direction_vectors=torch.zeros(2, 2),
        start_positions=torch.zeros(2, 2),
    )

    assert map_logits.shape == (2, policy_module.NUM_MAP_CATEGORIES, policy_module.SIZE, policy_module.SIZE)
    assert map_tokens.shape == (2, 101, 32)
    assert map_token_masks.shape == (2, 101)
