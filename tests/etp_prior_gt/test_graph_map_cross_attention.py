import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_vilmodel_cmt(monkeypatch):
    fake_transformers = types.ModuleType("transformers")

    class FakeBertPreTrainedModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    fake_transformers.BertPreTrainedModel = FakeBertPreTrainedModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_ops = types.ModuleType("vlnce_baselines.common.ops")
    fake_ops.create_transformer_encoder = lambda *args, **kwargs: None
    fake_ops.extend_neg_masks = lambda masks: masks
    fake_ops.gen_seq_masks = lambda lengths: torch.ones(
        len(lengths), int(torch.max(lengths).item()), dtype=torch.bool
    )
    fake_ops.pad_tensors_wgrad = lambda tensors: torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True
    )

    for name in ["vlnce_baselines", "vlnce_baselines.common"]:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "vlnce_baselines.common.ops", fake_ops)

    spec = importlib.util.spec_from_file_location(
        "vilmodel_cmt_under_test",
        ROOT / "vlnce_baselines/models/etp_prior_gt/vilmodel_cmt.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _load_grpo_replay_helper(monkeypatch):
    for module_name in [
        "gym",
        "habitat",
        "habitat_baselines.common.baseline_registry",
        "habitat_baselines.common.environments",
        "habitat_baselines.common.obs_transformers",
        "habitat_baselines.common.tensorboard_utils",
        "habitat_baselines.utils.common",
        "vlnce_baselines.common.base_il_trainer",
        "vlnce_baselines.common.env_utils",
        "vlnce_baselines.common.utils",
        "vlnce_baselines.models.graph_utils",
        "vlnce_baselines.common.ops",
        "habitat_extensions.measures",
        "fastdtw",
        "tensorflow",
        "cv2",
        "vlnce_baselines.models.etp_prior_gt.map_utils",
        "vlnce_baselines.utils",
    ]:
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    sys.modules["gym"].Space = object
    sys.modules["habitat"].Config = object
    sys.modules["habitat"].logger = object()

    class FakeRegistry:
        def register_trainer(self, name):
            return lambda cls: cls

    sys.modules["habitat_baselines.common.baseline_registry"].baseline_registry = FakeRegistry()
    sys.modules["habitat_baselines.common.environments"].get_env_class = lambda *args, **kwargs: None
    obs_transformers = sys.modules["habitat_baselines.common.obs_transformers"]
    obs_transformers.apply_obs_transforms_batch = lambda *args, **kwargs: None
    obs_transformers.apply_obs_transforms_obs_space = lambda *args, **kwargs: None
    obs_transformers.get_active_obs_transforms = lambda *args, **kwargs: []
    sys.modules["habitat_baselines.common.tensorboard_utils"].TensorboardWriter = object
    sys.modules["habitat_baselines.utils.common"].batch_obs = lambda *args, **kwargs: None
    sys.modules["vlnce_baselines.common.base_il_trainer"].BaseVLNCETrainer = object
    env_utils = sys.modules["vlnce_baselines.common.env_utils"]
    env_utils.construct_envs = lambda *args, **kwargs: None
    env_utils.is_slurm_batch_job = lambda: False
    sys.modules["vlnce_baselines.common.utils"].extract_instruction_tokens = lambda *args, **kwargs: None
    graph_utils = sys.modules["vlnce_baselines.models.graph_utils"]
    graph_utils.GraphMap = object
    graph_utils.MAX_DIST = 30.0
    ops = sys.modules["vlnce_baselines.common.ops"]
    ops.pad_tensors_wgrad = lambda tensors: torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    ops.gen_seq_masks = lambda lengths: torch.ones(len(lengths), int(torch.max(lengths).item()), dtype=torch.bool)
    sys.modules["habitat_extensions.measures"].NDTW = object
    sys.modules["fastdtw"].fastdtw = lambda *args, **kwargs: None
    map_utils = sys.modules["vlnce_baselines.models.etp_prior_gt.map_utils"]
    map_utils.build_cognitive_map_for_episode = lambda *args, **kwargs: None
    map_utils.cognitive_map_to_tensors = lambda *args, **kwargs: {}
    sys.modules["vlnce_baselines.utils"].get_camera_orientations12 = lambda *args, **kwargs: None

    package = types.ModuleType("vlnce_baselines")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "vlnce_baselines", package)
    for name in [
        "vlnce_baselines.common",
        "vlnce_baselines.models",
        "vlnce_baselines.models.etp_prior_gt",
    ]:
        module = sys.modules.get(name, types.ModuleType(name))
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "vlnce_baselines.GRPO_trainer_ETP_PriorGT",
        ROOT / "vlnce_baselines/GRPO_trainer_ETP_PriorGT.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.select_replay_map_inputs


def test_graph_map_cross_attention_is_identity_at_initialization(monkeypatch):
    vilmodel_cmt = _load_vilmodel_cmt(monkeypatch)
    cross_attention = vilmodel_cmt.GraphMapCrossAttention(
        hidden_size=768, num_heads=12, dropout=0.0
    )
    gmap_embeds = torch.randn(2, 5, 768)
    map_tokens = torch.randn(2, 101, 768)
    map_token_masks = torch.ones(2, 101, dtype=torch.bool)

    output = cross_attention(gmap_embeds, map_tokens, map_token_masks)

    assert output.shape == (2, 5, 768)
    assert torch.allclose(output, gmap_embeds, atol=1e-6)


def test_graph_map_cross_attention_handles_masked_map_tokens(monkeypatch):
    vilmodel_cmt = _load_vilmodel_cmt(monkeypatch)
    cross_attention = vilmodel_cmt.GraphMapCrossAttention(
        hidden_size=768, num_heads=12, dropout=0.0
    )
    gmap_embeds = torch.randn(2, 5, 768)
    map_tokens = torch.randn(2, 101, 768)
    map_token_masks = torch.ones(2, 101, dtype=torch.bool)
    map_token_masks[0, 12:] = False
    map_token_masks[1, 77:] = False

    output = cross_attention(gmap_embeds, map_tokens, map_token_masks)

    assert output.shape == (2, 5, 768)
    assert torch.isfinite(output).all()


def test_graph_map_cross_attention_handles_fully_masked_rows_at_initialization(monkeypatch):
    vilmodel_cmt = _load_vilmodel_cmt(monkeypatch)
    cross_attention = vilmodel_cmt.GraphMapCrossAttention(
        hidden_size=768, num_heads=12, dropout=0.0
    )
    gmap_embeds = torch.randn(2, 5, 768)
    map_tokens = torch.randn(2, 101, 768)
    map_token_masks = torch.ones(2, 101, dtype=torch.bool)
    map_token_masks[0] = False

    output = cross_attention(gmap_embeds, map_tokens, map_token_masks)

    assert output.shape == (2, 5, 768)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, gmap_embeds, atol=1e-6)


def test_graph_map_cross_attention_without_map_tokens_is_exact_no_op(monkeypatch):
    vilmodel_cmt = _load_vilmodel_cmt(monkeypatch)
    cross_attention = vilmodel_cmt.GraphMapCrossAttention(
        hidden_size=768, num_heads=12, dropout=0.0
    )
    gmap_embeds = torch.randn(2, 5, 768)
    original = gmap_embeds.clone()

    output = cross_attention(gmap_embeds, None, None)

    assert torch.equal(output, original)


def test_grpo_replay_map_tokens_and_masks_use_same_active_indices(monkeypatch):
    select_replay_map_inputs = _load_grpo_replay_helper(monkeypatch)
    step_map_tokens = torch.arange(4 * 3 * 2).view(4, 3, 2)
    step_map_token_masks = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, True, False],
        ]
    )
    active_indices = [2, 0]

    map_tokens, map_token_masks = select_replay_map_inputs(
        step_map_tokens, step_map_token_masks, active_indices
    )

    assert torch.equal(map_tokens, step_map_tokens[active_indices])
    assert torch.equal(map_token_masks, step_map_token_masks[active_indices])


def test_grpo_replay_keeps_already_active_sized_map_inputs(monkeypatch):
    select_replay_map_inputs = _load_grpo_replay_helper(monkeypatch)
    step_map_tokens = torch.arange(2 * 3 * 2).view(2, 3, 2)
    step_map_token_masks = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
        ]
    )
    active_indices = [2, 0]

    map_tokens, map_token_masks = select_replay_map_inputs(
        step_map_tokens, step_map_token_masks, active_indices
    )

    assert map_tokens is step_map_tokens
    assert map_token_masks is step_map_token_masks
