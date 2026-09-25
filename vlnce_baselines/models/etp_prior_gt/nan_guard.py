"""Fail fast on non-finite weights and activations.

``assert_finite_parameters`` runs once after weights are loaded and names the
offending tensors.  ``NonFiniteOutputGuard`` hooks every submodule of a root
module and, after each root forward, raises naming the first submodule (in
execution order) whose output contained NaN or +inf, so a NaN that surfaces as
"probs do not satisfy Simplex" is traced to the layer that produced it.  -inf
is allowed because attention and action masks legitimately use it.  The hooks
remove themselves after ``max_calls`` root forwards, so a run that proves
finite pays nothing afterwards.

Outputs may mix devices (the waypoint predictor returns CPU and GPU tensors),
so flags are only ever stacked per device: one sync per device per forward.
"""

from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn


def _float_tensors(value: Any) -> List[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.numel() > 0:
            return [value]
        return []
    if isinstance(value, (tuple, list)):
        children = list(value)
    elif isinstance(value, dict):
        children = list(value.values())
    else:
        return []
    return [tensor for child in children for tensor in _float_tensors(child)]


def _resolve(flags: List[torch.Tensor]) -> List[bool]:
    """Read 0-dim bool flags with one device sync per device."""
    by_device: Dict[torch.device, List[int]] = {}
    for index, flag in enumerate(flags):
        by_device.setdefault(flag.device, []).append(index)
    values = [False] * len(flags)
    for indices in by_device.values():
        stacked = torch.stack([flags[index] for index in indices]).tolist()
        for index, value in zip(indices, stacked):
            values[index] = bool(value)
    return values


def assert_finite_parameters(module: nn.Module, name: str) -> None:
    named = list(module.named_parameters())
    if not named:
        return
    finite = _resolve([torch.isfinite(param).all() for _, param in named])
    bad = [pname for (pname, _), ok in zip(named, finite) if not ok]
    if bad:
        raise RuntimeError(
            f"[nan-guard] {name}: {len(bad)} non-finite parameter tensors, "
            f"first: {bad[:5]}"
        )


class NonFiniteOutputGuard:
    """Name the first submodule whose forward output is NaN or +inf."""

    def __init__(self, root: nn.Module, name: str, max_calls: int) -> None:
        if max_calls <= 0:
            raise ValueError(f"max_calls must be positive, got {max_calls}")
        self._name = name
        self._remaining = max_calls
        # (module label, class name, one flag per float tensor in the output)
        self._pending: List[Tuple[str, str, List[torch.Tensor]]] = []
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
            flags = [
                torch.isnan(tensor).any() | torch.isposinf(tensor).any()
                for tensor in _float_tensors(output)
            ]
            if flags:
                self._pending.append((label, type(module).__name__, flags))

        return hook

    def _check_root(self, module: nn.Module, inputs: Any, output: Any) -> None:
        pending, self._pending = self._pending, []
        if pending:
            owners = [
                entry_index
                for entry_index, (_, _, flags) in enumerate(pending)
                for _ in flags
            ]
            values = _resolve([flag for _, _, flags in pending for flag in flags])
            for entry_index, bad in zip(owners, values):
                if bad:
                    label, class_name, _ = pending[entry_index]
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
