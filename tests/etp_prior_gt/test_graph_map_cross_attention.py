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
