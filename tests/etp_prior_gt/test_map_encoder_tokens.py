import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _install_fake_clip(monkeypatch):
    fake_clip = types.ModuleType("clip")

    class FakeClipModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def eval(self):
            return self

        def encode_text(self, tokens):
            num_prompts = tokens.size(0)
            features = torch.arange(num_prompts * 512, dtype=torch.float32).view(
                num_prompts, 512
            )
            return features + 1.0

    def load(name, device="cpu"):
        return FakeClipModel(), None

    def tokenize(prompts):
        return torch.zeros(len(prompts), 77, dtype=torch.long)

    fake_clip.load = load
    fake_clip.tokenize = tokenize
    monkeypatch.setitem(sys.modules, "clip", fake_clip)


def _load_priorgt_modules(monkeypatch):
    _install_fake_clip(monkeypatch)
    for name in [
        "vlnce_baselines",
        "vlnce_baselines.models",
        "vlnce_baselines.models.etp_prior_gt",
    ]:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
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
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, mod_name, module)
        spec.loader.exec_module(module)

    return (
        sys.modules["vlnce_baselines.models.etp_prior_gt.map_utils"],
        sys.modules["vlnce_baselines.models.etp_prior_gt.map_encoder"],
    )


def test_map_encoder_returns_101_tokens_and_mask(monkeypatch):
    map_utils, map_encoder = _load_priorgt_modules(monkeypatch)
    encoder = map_encoder.EmbeddingGridMapEncoder(hidden_size=768)
    batch_size = 2
    grid = torch.randn(
        batch_size, map_utils.NUM_MAP_CATEGORIES, map_utils.SIZE, map_utils.SIZE
    )
    trajectory_keypoints = torch.randn(
        batch_size, map_utils.TRAJECTORY_KEYPOINT_COUNT, 2
    )
    start_directions = torch.randn(batch_size, 2)
    starts = torch.randn(batch_size, 2)

    result = encoder(grid, trajectory_keypoints, start_directions, starts)

    assert isinstance(result, tuple)
    assert len(result) == 2
    map_tokens, map_token_masks = result
    assert map_tokens.shape == (batch_size, 101, 768)
    assert map_token_masks.shape == (batch_size, 101)
    assert map_token_masks.dtype == torch.bool
    assert map_token_masks.all().item()


def test_map_encoder_emits_finite_tokens_for_empty_map_metadata(monkeypatch):
    map_utils, map_encoder = _load_priorgt_modules(monkeypatch)
    encoder = map_encoder.EmbeddingGridMapEncoder(hidden_size=768)
    batch_size = 2
    grid = torch.zeros(
        batch_size, map_utils.NUM_MAP_CATEGORIES, map_utils.SIZE, map_utils.SIZE
    )
    trajectory_keypoints = torch.zeros(
        batch_size, map_utils.TRAJECTORY_KEYPOINT_COUNT, 2
    )
    start_directions = torch.zeros(batch_size, 2)
    starts = torch.zeros(batch_size, 2)

    map_tokens, map_token_masks = encoder(
        grid, trajectory_keypoints, start_directions, starts
    )

    assert map_tokens.shape == (batch_size, 101, 768)
    assert torch.isfinite(map_tokens).all()
    assert map_token_masks.shape == (batch_size, 101)
    assert map_token_masks.dtype == torch.bool
    assert map_token_masks.all().item()


def test_map_encoder_rejects_bad_grid_shape(monkeypatch):
    map_utils, map_encoder = _load_priorgt_modules(monkeypatch)
    encoder = map_encoder.EmbeddingGridMapEncoder(hidden_size=768)
    bad_grid = torch.randn(
        2, map_utils.NUM_MAP_CATEGORIES - 1, map_utils.SIZE, map_utils.SIZE
    )
    trajectory_keypoints = torch.randn(2, map_utils.TRAJECTORY_KEYPOINT_COUNT, 2)
    start_directions = torch.randn(2, 2)
    starts = torch.randn(2, 2)

    try:
        encoder(bad_grid, trajectory_keypoints, start_directions, starts)
    except ValueError as exc:
        assert "cognitive_crop" in str(exc)
    else:
        raise AssertionError("Expected ValueError for bad cognitive_crop shape")


def test_map_encoder_rejects_bad_start_direction_shape(monkeypatch):
    map_utils, map_encoder = _load_priorgt_modules(monkeypatch)
    encoder = map_encoder.EmbeddingGridMapEncoder(hidden_size=768)
    grid = torch.randn(2, map_utils.NUM_MAP_CATEGORIES, map_utils.SIZE, map_utils.SIZE)
    trajectory_keypoints = torch.randn(2, map_utils.TRAJECTORY_KEYPOINT_COUNT, 2)
    bad_start_directions = torch.randn(2, 3)
    starts = torch.randn(2, 2)

    try:
        encoder(grid, trajectory_keypoints, bad_start_directions, starts)
    except ValueError as exc:
        assert "start_direction_vectors" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for bad start_direction_vectors shape"
        )
