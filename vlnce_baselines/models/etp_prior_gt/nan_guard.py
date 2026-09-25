"""Fail fast on non-finite weights and activations.

``assert_finite_parameters`` runs once after weights are loaded and names the
offending tensors.  ``NonFiniteOutputGuard`` hooks every submodule of a root
module and, after each root forward, raises naming the first submodule (in
execution order) whose output contained NaN or +inf, so a NaN that surfaces as
"probs do not satisfy Simplex" is traced to the layer that produced it.  -inf
is allowed because attention and action masks legitimately use it.  The hooks
remove themselves after ``max_calls`` root forwards, so a run that proves
finite pays nothing afterwards.
"""

from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn


def assert_finite_parameters(module: nn.Module, name: str) -> None:
    named = list(module.named_parameters())
    if not named:
        return
    flags = torch.stack([torch.isfinite(param).all() for _, param in named])
    bad = [pname for (pname, _), ok in zip(named, flags.tolist()) if not ok]
    if bad:
        raise RuntimeError(
            f"[nan-guard] {name}: {len(bad)} non-finite parameter tensors, "
            f"first: {bad[:5]}"
        )


def _nan_or_posinf(value: Any) -> Optional[torch.Tensor]:
    """0-dim bool tensor: any float tensor inside ``value`` holds NaN or +inf."""
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point() or value.numel() == 0:
            return None
        return torch.isnan(value).any() | torch.isposinf(value).any()
    if isinstance(value, (tuple, list)):
        children = value
    elif isinstance(value, dict):
        children = list(value.values())
    else:
        return None
    parts = [flag for flag in map(_nan_or_posinf, children) if flag is not None]
    if not parts:
        return None
    return torch.stack(parts).any()


class NonFiniteOutputGuard:
    """Name the first submodule whose forward output is NaN or +inf."""

    def __init__(self, root: nn.Module, name: str, max_calls: int) -> None:
        if max_calls <= 0:
            raise ValueError(f"max_calls must be positive, got {max_calls}")
        self._name = name
        self._remaining = max_calls
        self._pending: List[Tuple[str, str, torch.Tensor]] = []
        self._handles: List[Any] = []
        for module_name, module in root.named_modules():
            label = module_name or "<root>"
            self._handles.append(module.register_forward_hook(self._recorder(label)))
        # Registered after the root's own recorder, so it runs last.
        self._handles.append(root.register_forward_hook(self._check_root))

    @property
    def active(self) -> bool:
        return bool(self._handles)

    def _recorder(self, label: str) -> Callable[[nn.Module, Any, Any], None]:
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            flag = _nan_or_posinf(output)
            if flag is not None:
                self._pending.append((label, type(module).__name__, flag))

        return hook

    def _check_root(self, module: nn.Module, inputs: Any, output: Any) -> None:
        pending, self._pending = self._pending, []
        if pending:
            # One device sync per root forward instead of one per submodule.
            flags = torch.stack([flag for _, _, flag in pending]).tolist()
            for (label, class_name, _), bad in zip(pending, flags):
                if bad:
                    self.remove()
                    raise RuntimeError(
                        f"[nan-guard] {self._name}: first non-finite (NaN/+inf) "
                        f"output came from {label} ({class_name}); every module "
                        "that finished earlier in this forward was finite"
                    )
        self._remaining -= 1
        if self._remaining <= 0:
            self.remove()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._pending = []
