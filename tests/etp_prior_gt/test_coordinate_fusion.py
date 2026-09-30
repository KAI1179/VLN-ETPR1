"""Coordinate fusion: node-to-map-token distance bias and map token positions.

Without it the try5 map path is blind to layout: the map encoder has no
position code, so its 100 spatial tokens form an unordered set, and graph
nodes attend to them by content only.  These tests pin down that the flag-off
path is unchanged, that the flag adds no parameters, and that with the flag
on the policy can tell where on the map each node is.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load(monkeypatch, *modules):
    fake_clip = types.ModuleType("clip")

    class FakeClipModel(torch.nn.Module):
        def eval(self):
            return self

        def encode_text(self, tokens):
            return torch.randn(tokens.size(0), 512, generator=torch.Generator().manual_seed(0))

    fake_clip.load = lambda name, device="cpu": (FakeClipModel(), None)
    fake_clip.tokenize = lambda prompts: torch.zeros(len(prompts), 77, dtype=torch.long)
    monkeypatch.setitem(sys.modules, "clip", fake_clip)
    for name in [
        "vlnce_baselines",
        "vlnce_baselines.models",
        "vlnce_baselines.models.etp_prior_gt",
    ]:
        if sys.modules.get(name) is None:
            package = types.ModuleType(name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, name, package)
    loaded = []
    for stem in modules:
        mod_name = f"vlnce_baselines.models.etp_prior_gt.{stem}"
        spec = importlib.util.spec_from_file_location(
            mod_name, ROOT / f"vlnce_baselines/models/etp_prior_gt/{stem}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, mod_name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def _fusions(map_fusion, hidden=16, heads=4):
    torch.manual_seed(0)
    off = map_fusion.GraphMapCrossAttention(hidden, heads, dropout=0.0)
    on = map_fusion.GraphMapCrossAttention(hidden, heads, dropout=0.0, coordinate_fusion=True)
    on.load_state_dict(off.state_dict())
    for module in (off, on):
        torch.nn.init.normal_(module.residual_projection.weight, std=0.1)
    on.residual_projection.load_state_dict(off.residual_projection.state_dict())
    return off.eval(), on.eval()


def test_flag_adds_no_parameters_and_no_state(monkeypatch):
    (map_fusion,) = _load(monkeypatch, "map_fusion")
    off, on = _fusions(map_fusion)
    assert off.state_dict().keys() == on.state_dict().keys()


def test_nodes_without_position_reproduce_try5(monkeypatch):
    (map_fusion,) = _load(monkeypatch, "map_fusion")
    off, on = _fusions(map_fusion)
    gmap = torch.randn(2, 5, 16)
    tokens = torch.randn(2, 101, 16)
    masks = torch.ones(2, 101, dtype=torch.bool)
    no_position = torch.full((2, 5, 2), float("nan"))
    expected, _ = off(gmap, None, tokens, masks)
    actual, _ = on(gmap, None, tokens, masks, node_coords=no_position)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_zero_residual_projection_is_identity(monkeypatch):
    (map_fusion,) = _load(monkeypatch, "map_fusion")
    fusion = map_fusion.GraphMapCrossAttention(16, 4, dropout=0.0, coordinate_fusion=True)
    gmap = torch.randn(1, 4, 16)
    out, _ = fusion(
        gmap, None, torch.randn(1, 101, 16), torch.ones(1, 101, dtype=torch.bool),
        node_coords=torch.rand(1, 4, 2) * 100,
    )
    assert torch.equal(out, gmap)


def test_missing_coordinates_fail_loudly(monkeypatch):
    (map_fusion,) = _load(monkeypatch, "map_fusion")
    _, on = _fusions(map_fusion)
    with pytest.raises(ValueError, match="gmap_map_coords"):
        on(torch.randn(1, 2, 16), None, torch.randn(1, 101, 16), None)


def test_bias_is_applied_per_batch_item_and_head(monkeypatch):
    (map_fusion,) = _load(monkeypatch, "map_fusion")
    _, on = _fusions(map_fusion)
    gmap = torch.randn(2, 3, 16)
    tokens = torch.randn(2, 101, 16)
    coords = torch.tensor(
        [[[5.0, 5.0], [55.0, 95.0], [float("nan")] * 2],
         [[95.0, 5.0], [12.0, 40.0], [60.0, 60.0]]]
    )
    out, _ = on(gmap, None, tokens, torch.ones(2, 101, dtype=torch.bool), node_coords=coords)

    attention = on.attention
    heads, head_dim = attention.num_heads, 16 // attention.num_heads
    w_q, w_k, w_v = attention.in_proj_weight.chunk(3)
    b_q, b_k, b_v = attention.in_proj_bias.chunk(3)
    q = (gmap @ w_q.T + b_q).view(2, 3, heads, head_dim).transpose(1, 2)
    k = (tokens @ w_k.T + b_k).view(2, 101, heads, head_dim).transpose(1, 2)
    v = (tokens @ w_v.T + b_v).view(2, 101, heads, head_dim).transpose(1, 2)
    bias = on.coordinate_bias(coords, 101).view(2, heads, 3, 101)
    weights = torch.softmax(q @ k.transpose(-1, -2) / head_dim**0.5 + bias, dim=-1)
    context = (weights @ v).transpose(1, 2).reshape(2, 3, 16)
    context = attention.out_proj(context)
    expected = gmap + on.residual_projection(context)
    assert torch.allclose(out, expected, atol=1e-5)


def test_local_heads_prefer_the_token_under_the_node(monkeypatch):
    (map_fusion,) = _load(monkeypatch, "map_fusion")
    fusion = map_fusion.GraphMapCrossAttention(24, 12, coordinate_fusion=True)
    coef = fusion.locality_coef
    assert (coef == 0).sum() == 4 and (coef > 0).sum() == 8
    # Node at raster cell (37.2, 81.9) lies in token row 3, col 8 -> index 38.
    bias = fusion.coordinate_bias(torch.tensor([[[37.2, 81.9]]]), 101).view(12, 101)
    for head in range(12):
        if coef[head] > 0:
            assert int(bias[head, :100].argmax()) == 38
            assert bias[head, 100] == 0  # metadata token is never biased
        else:
            assert torch.all(bias[head] == 0)


def test_factory_rejects_coordinates_for_bidirectional_fusion(monkeypatch):
    (map_fusion,) = _load(monkeypatch, "map_fusion")
    with pytest.raises(ValueError, match="try5"):
        map_fusion.build_map_token_fusion("current", 16, 4, coordinate_fusion=True)


def _block_permute(grid, order):
    blocks = grid.reshape(37, 10, 10, 10, 10).transpose(0, 1, 3, 2, 4).reshape(37, 100, 10, 10)
    return blocks[:, order].reshape(37, 10, 10, 10, 10).transpose(0, 1, 3, 2, 4).reshape(37, 100, 100)


def test_position_code_makes_the_encoder_see_layout(monkeypatch):
    _, map_encoder = _load(monkeypatch, "map_utils", "map_encoder")
    torch.manual_seed(0)
    plain = map_encoder.EmbeddingGridMapEncoder(hidden_size=64).eval()
    coded = map_encoder.EmbeddingGridMapEncoder(hidden_size=64, spatial_position_encoding=True).eval()
    assert plain.state_dict().keys() == coded.state_dict().keys()
    coded.load_state_dict(plain.state_dict())

    rng = np.random.default_rng(0)
    grid = (rng.random((37, 100, 100)) > 0.97).astype(np.float32)
    order = rng.permutation(100)
    inverse = np.argsort(order)
    permuted = _block_permute(grid, order)
    metadata = (torch.randn(1, 5, 2), torch.randn(1, 2), torch.randn(1, 2))

    def tokens(encoder, raster):
        with torch.no_grad():
            out, _ = encoder(torch.from_numpy(np.ascontiguousarray(raster))[None], *metadata)
        return out[0]

    # Without the code the spatial tokens are an unordered set: permuting the
    # blocks only permutes the output tokens.
    base, moved = tokens(plain, grid), tokens(plain, permuted)
    assert torch.allclose(moved[:100][inverse], base[:100], atol=1e-4)
    assert torch.allclose(moved[100], base[100], atol=1e-4)
    # With it, the same blocks at other places encode differently.
    base, moved = tokens(coded, grid), tokens(coded, permuted)
    assert not torch.allclose(moved[:100][inverse], base[:100], atol=1e-2)


def test_start_maps_to_its_cached_cell(monkeypatch):
    map_utils, = _load(monkeypatch, "map_utils")
    start_world = [3.2, 0.1, -7.9]
    for meters_per_unit, start_position in ((1.0, [12.25, 30.5]), (0.5, [24.5, 61.0])):
        origin = map_utils.map_origin_xz(start_world, torch.tensor(start_position), meters_per_unit)
        cell = map_utils.world_to_map_cells([start_world], origin)[0]
        assert torch.allclose(cell, torch.tensor([24.5, 61.0]))
    # One metre along world x is two rows, along world z two columns.
    moved = map_utils.world_to_map_cells([[4.2, 0.1, -7.4]], origin)[0]
    assert torch.allclose(moved, torch.tensor([26.5, 62.0]))
    nan_row = map_utils.world_to_map_cells(np.full((1, 3), np.nan), origin)
    assert torch.isnan(nan_row).all()
