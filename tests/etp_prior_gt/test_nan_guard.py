import pytest
import torch
import torch.nn as nn

from vlnce_baselines.models.etp_prior_gt.nan_guard import (
    NonFiniteOutputGuard,
    assert_finite_parameters,
)


class Poison(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x.masked_fill(torch.ones_like(x, dtype=torch.bool), self.value)


def _model(poison_value: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 4), Poison(poison_value), nn.Linear(4, 4))


def test_assert_finite_parameters_names_the_bad_tensor():
    model = _model(0.0)
    with torch.no_grad():
        model[2].bias[1] = float("nan")

    with pytest.raises(RuntimeError, match=r"2\.bias"):
        assert_finite_parameters(model, "toy")


def test_guard_names_first_module_emitting_nan():
    model = _model(float("nan"))
    NonFiniteOutputGuard(model, "toy", max_calls=3)

    with pytest.raises(RuntimeError, match=r"came from 1 \(Poison\)"):
        model(torch.randn(2, 4))


def test_guard_treats_positive_infinity_as_non_finite():
    model = _model(float("inf"))
    NonFiniteOutputGuard(model, "toy", max_calls=3)

    with pytest.raises(RuntimeError, match=r"came from 1 \(Poison\)"):
        model(torch.randn(2, 4))


def test_guard_allows_negative_infinity_masks():
    # -inf must be the last output: a Linear fed -inf would legitimately NaN.
    model = nn.Sequential(nn.Linear(4, 4), Poison(float("-inf")))
    guard = NonFiniteOutputGuard(model, "toy", max_calls=3)

    model(torch.randn(2, 4))

    assert guard.active


def test_guard_removes_hooks_after_max_calls():
    model = _model(0.0)
    guard = NonFiniteOutputGuard(model, "toy", max_calls=2)

    model(torch.randn(2, 4))
    assert guard.active
    model(torch.randn(2, 4))
    assert not guard.active

    model[1].value = float("nan")
    model(torch.randn(2, 4))  # hooks are gone: no raise


def test_guard_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        NonFiniteOutputGuard(_model(0.0), "toy", max_calls=0)


class MixedOutput(nn.Module):
    """Mimics waypoint mode: a dict mixing float, int and nested tensors."""

    def __init__(self, poison_value: float):
        super().__init__()
        self.value = poison_value

    def forward(self, x):
        return {
            "logits": x,
            "ids": torch.zeros(2, dtype=torch.long),
            "nested": (
                x * 2.0,
                [x.masked_fill(torch.ones_like(x, dtype=torch.bool), self.value)],
            ),
        }


def test_guard_walks_nested_dict_outputs_with_non_float_entries():
    model = nn.Sequential(nn.Linear(4, 4), MixedOutput(0.0))
    guard = NonFiniteOutputGuard(model, "toy", max_calls=2)
    model(torch.randn(2, 4))
    assert guard.active

    model[1].value = float("nan")
    with pytest.raises(RuntimeError, match=r"came from 1 \(MixedOutput\)"):
        model(torch.randn(2, 4))
