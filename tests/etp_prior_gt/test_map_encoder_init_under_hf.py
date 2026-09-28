"""The spatial tokenizer must be initialised even when torch.nn.init is disabled.

transformers' from_pretrained() wraps model construction in no_init_weights(),
which turns every torch.nn.init function into a no-op, and BERT's _init_weights
never touches nn.Conv2d.  Both pretraining checkpoints (387500, 460000) shipped
a spatial_tokenizer.weight of exact zeros and an arbitrary bias for this reason.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
transformers = pytest.importorskip("transformers")


def _install_fake_clip(monkeypatch):
    fake_clip = types.ModuleType("clip")

    class FakeClipModel(torch.nn.Module):
        def eval(self):
            return self

        def encode_text(self, tokens):
            num_prompts = tokens.size(0)
            features = torch.arange(num_prompts * 512, dtype=torch.float32).view(
                num_prompts, 512
            )
            return features + 1.0

    fake_clip.load = lambda name, device="cpu": (FakeClipModel(), None)
    fake_clip.tokenize = lambda prompts: torch.zeros(len(prompts), 77, dtype=torch.long)
    monkeypatch.setitem(sys.modules, "clip", fake_clip)


def _load_map_encoder(monkeypatch):
    _install_fake_clip(monkeypatch)
    for name in [
        "vlnce_baselines",
        "vlnce_baselines.models",
        "vlnce_baselines.models.etp_prior_gt",
    ]:
        if sys.modules.get(name) is None:
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
    return sys.modules["vlnce_baselines.models.etp_prior_gt.map_encoder"]


def test_spatial_tokenizer_survives_transformers_no_init_weights(monkeypatch):
    map_encoder = _load_map_encoder(monkeypatch)
    with transformers.modeling_utils.no_init_weights():
        encoder = map_encoder.EmbeddingGridMapEncoder(hidden_size=768)

    weight = encoder.spatial_tokenizer.weight.detach()
    bias = encoder.spatial_tokenizer.bias.detach()
    bound = 1.0 / (512 * 10 * 10) ** 0.5
    assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
    assert weight.abs().max() <= bound and bias.abs().max() <= bound
    # U(-bound, bound) over 39.3M entries: norm ~ bound * sqrt(n / 3) ~ 16.
    assert 12.0 < weight.norm().item() < 20.0
    encoder.assert_initialised()


def test_assert_initialised_rejects_zero_weight(monkeypatch):
    map_encoder = _load_map_encoder(monkeypatch)
    encoder = map_encoder.EmbeddingGridMapEncoder(hidden_size=768)
    with torch.no_grad():
        encoder.spatial_tokenizer.weight.zero_()

    with pytest.raises(RuntimeError, match="spatial_tokenizer.weight: all zeros"):
        encoder.assert_initialised()
