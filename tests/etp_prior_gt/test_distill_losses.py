"""Distillation loss terms: masked KL, STOP factorisation, counterfactual
map-effect matching, the reachability gate and the donor permutation.

Pure tensor math, no policy or simulator involved.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "distill_losses",
    ROOT / "vlnce_baselines" / "models" / "etp_prior_gt" / "distill_losses.py",
)
dl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dl)


def _case(seed=0, b=3, g=6):
    gen = torch.Generator().manual_seed(seed)
    t = torch.randn(b, g, generator=gen)
    s = torch.randn(b, g, generator=gen)
    valid = torch.ones(b, g, dtype=torch.bool)
    valid[0, 3:] = False  # a short graph
    valid[1, 2] = False  # a visited node
    # The policy masks invalid logits to -inf; mimic that on both sides.
    t = t.masked_fill(~valid, float("-inf"))
    s = s.masked_fill(~valid, float("-inf"))
    return t, s, valid


def _plain_kl(t, s, valid, temp=1.0):
    t_logp = torch.log_softmax(t / temp, dim=1)
    s_logp = torch.log_softmax(s / temp, dim=1)
    kl = torch.where(valid, t_logp.exp() * (t_logp - s_logp), torch.zeros_like(t))
    return kl.sum(dim=1) * temp * temp


def test_masked_log_softmax_normalises_over_valid_only():
    t, _, valid = _case()
    logp = dl.masked_log_softmax(t, valid)
    probs = torch.where(valid, logp.exp(), torch.zeros_like(logp))
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.isinf(logp[~valid]).all()


def test_masked_log_softmax_empty_row_is_finite_free_of_nan():
    t, _, valid = _case()
    valid[2] = False
    logp = dl.masked_log_softmax(t, valid)
    assert not torch.isnan(logp).any()


@pytest.mark.parametrize("temp", [1.0, 2.0])
def test_action_kl_matches_historical_formula(temp):
    t, s, valid = _case()
    assert torch.allclose(dl.action_kl(t, s, valid, temp), _plain_kl(t, s, valid, temp), atol=1e-6)


def test_action_kl_zero_for_identical_and_zero_on_empty_rows():
    t, _, valid = _case()
    assert torch.allclose(dl.action_kl(t, t, valid), torch.zeros(3), atol=1e-6)
    valid[1] = False
    assert dl.action_kl(t, t + 1.0, valid)[1] == 0.0


def test_factorized_kl_chain_rule_identity():
    t, s, valid = _case(seed=1)
    stop_kl, move_kl, p_move = dl.factorized_kl(t, s, valid)
    assert torch.allclose(stop_kl + p_move * move_kl, dl.action_kl(t, s, valid), atol=1e-5)
    assert (stop_kl >= -1e-7).all() and (move_kl >= -1e-7).all()


def test_factorized_kl_stop_only_row_has_zero_move_term():
    t, s, valid = _case()
    valid[0, 1:] = False
    t = t.masked_fill(~valid, float("-inf"))
    s = s.masked_fill(~valid, float("-inf"))
    stop_kl, move_kl, _ = dl.factorized_kl(t, s, valid)
    assert move_kl[0] == 0.0
    assert torch.isfinite(stop_kl).all()


def test_factorized_kl_ignored_row_is_zero_and_masked_stop_is_rejected():
    t, s, valid = _case()
    valid[2] = False
    stop_kl, move_kl, _ = dl.factorized_kl(t, s, valid)
    assert stop_kl[2] == 0.0 and move_kl[2] == 0.0
    valid[1, 0] = False
    with pytest.raises(ValueError, match="STOP is masked out in rows \\[1\\]"):
        dl.factorized_kl(t, s, valid)


def test_effect_match_zero_when_student_effect_equals_teacher_effect_up_to_constant():
    t_full, t_cf, valid = _case(seed=2)
    delta = torch.where(valid, t_full - t_cf, torch.zeros_like(t_full))
    s_cf = torch.randn(3, 6).masked_fill(~valid, float("-inf"))
    s_full = s_cf + delta + 3.0  # same effect, shifted by a row constant
    loss = dl.effect_match(t_full, t_cf, s_full, s_cf, valid)
    assert torch.allclose(loss, torch.zeros(3), atol=1e-5)


def test_effect_match_gradient_reaches_only_student_full():
    t_full, t_cf, valid = _case(seed=3)
    s_full = torch.randn(3, 6).masked_fill(~valid, float("-inf")).requires_grad_(True)
    s_cf = torch.randn(3, 6).masked_fill(~valid, float("-inf")).requires_grad_(True)
    t_full = t_full.clone().requires_grad_(True)
    loss = dl.effect_match(t_full, t_cf, s_full, s_cf, valid).sum()
    loss.backward()
    assert s_full.grad is not None and torch.isfinite(s_full.grad).all()
    assert (s_full.grad[~valid] == 0).all()
    assert s_cf.grad is None or (s_cf.grad == 0).all()
    assert t_full.grad is None
    assert loss.item() > 0


def test_effect_match_is_invariant_to_shared_non_map_shift():
    # A change that moves both branches equally (vision, language) cancels.
    t_full, t_cf, valid = _case(seed=4)
    s_full = torch.randn(3, 6).masked_fill(~valid, float("-inf"))
    s_cf = torch.randn(3, 6).masked_fill(~valid, float("-inf"))
    shift = torch.randn(3, 6)
    a = dl.effect_match(t_full, t_cf, s_full, s_cf, valid)
    b = dl.effect_match(t_full, t_cf, s_full + shift, s_cf + shift, valid)
    assert torch.allclose(a, b, atol=1e-5)


def test_js_divergence_range_and_symmetry():
    a, b, valid = _case(seed=5)
    js = dl.js_divergence(a, b, valid)
    assert (js >= 0).all() and (js <= math.log(2) + 1e-6).all()
    assert torch.allclose(js, dl.js_divergence(b, a, valid), atol=1e-6)
    assert torch.allclose(dl.js_divergence(a, a, valid), torch.zeros(3), atol=1e-6)
    # Disjoint supports (as disjoint as -inf masking allows) approach ln 2.
    peaked_a = torch.tensor([[50.0, 0.0, 0.0]])
    peaked_b = torch.tensor([[0.0, 0.0, 50.0]])
    ones = torch.ones(1, 3, dtype=torch.bool)
    assert abs(dl.js_divergence(peaked_a, peaked_b, ones).item() - math.log(2)) < 1e-4


def test_gate_weights_floor_and_scale():
    d = torch.tensor([0.0, 0.1, 0.2, 10.0])
    w = dl.gate_weights(d, tau=0.1, min_weight=0.2)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.2 + 0.8 * math.exp(-1))
    assert w[3] == pytest.approx(0.2, abs=1e-6)
    assert (w >= 0.2).all() and (w <= 1.0).all()
    # tau <= 0: batch median as the scale.
    w_med = dl.gate_weights(d, tau=0.0, min_weight=0.0)
    median = d.median().item()
    assert w_med[1] == pytest.approx(math.exp(-0.1 / median))
    # All-zero divergence: no gating.
    assert torch.equal(dl.gate_weights(torch.zeros(3), tau=0.0, min_weight=0.2), torch.ones(3))
    with pytest.raises(ValueError):
        dl.gate_weights(d, tau=1.0, min_weight=1.5)


def test_gate_weights_are_detached():
    d = torch.tensor([0.1, 0.3], requires_grad=True)
    w = dl.gate_weights(d, tau=1.0, min_weight=0.0)
    assert not w.requires_grad


def test_donor_permutation_is_a_derangement_preferring_other_scenes():
    scenes = ["a", "a", "b", "b"]
    perm = dl.donor_permutation(scenes)
    assert sorted(perm) == [0, 1, 2, 3]
    assert all(p != i for i, p in enumerate(perm))
    # Shift 1 pairs a->a and b->b once each; shift 2 pairs every a with a b.
    assert all(scenes[i] != scenes[p] for i, p in enumerate(perm))
    assert dl.donor_permutation(["a"]) is None
    assert dl.donor_permutation([]) is None
    same = dl.donor_permutation(["a", "a"])
    assert same == [1, 0]
