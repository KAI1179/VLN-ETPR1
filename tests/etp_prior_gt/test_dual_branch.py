"""Dual-branch distillation: the identities the trainer wiring relies on.

The zero branch mimics the map branch through action_kl with the map side
detached; this checks that (1) the loss is zero when both branches agree,
(2) gradient reaches only the zero branch, (3) the config keys exist with
their documented defaults.  Pure tensor math plus an ast read of the config.
"""

import ast
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "distill_losses",
    ROOT / "vlnce_baselines" / "models" / "etp_prior_gt" / "distill_losses.py",
)
dl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dl)


def _pair():
    gen = torch.Generator().manual_seed(1)
    map_logits = torch.randn(3, 6, generator=gen, requires_grad=True)
    zero_logits = torch.randn(3, 6, generator=gen, requires_grad=True)
    valid = torch.ones(3, 6, dtype=torch.bool)
    valid[0, 4:] = False
    return map_logits, zero_logits, valid


def test_self_kl_is_zero_when_branches_agree():
    map_logits, _, valid = _pair()
    kl = dl.action_kl(map_logits.detach(), map_logits, valid)
    assert torch.allclose(kl, torch.zeros(3), atol=1e-6)


def test_self_kl_gradient_reaches_only_the_zero_branch():
    map_logits, zero_logits, valid = _pair()
    kl = dl.action_kl(map_logits.detach(), zero_logits, valid).sum()
    kl.backward()
    assert map_logits.grad is None
    assert zero_logits.grad is not None and torch.isfinite(zero_logits.grad).all()
    assert (zero_logits.grad[~valid] == 0).all()


def test_dual_config_defaults():
    tree = ast.parse((ROOT / "vlnce_baselines" / "config" / "default.py").read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = ast.unparse(node.targets[0])
            if target.startswith("_C.IL.dual_"):
                found[target[len("_C.IL."):]] = ast.literal_eval(node.value)
    assert found == {
        "dual_enabled": False,
        "dual_llm_weight": 1.0,
        "dual_ce_on_map_branch": True,
        "dual_gt_kl_on_map_branch": True,
        "dual_gt_kl_on_zero_branch": True,
        "dual_rollout_branch": "zero",
    }
