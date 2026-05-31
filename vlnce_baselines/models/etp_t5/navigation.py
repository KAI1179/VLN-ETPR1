"""Shared T5-Navigation scaffold errors and guards."""

from typing import NoReturn


class T5ReferencePathNotImplementedError(NotImplementedError):
    """Raised when T5-Navigation reaches unimplemented reference-path logic."""


def raise_t5_reference_path_not_implemented(context: str) -> NoReturn:
    raise T5ReferencePathNotImplementedError(
        "T5-Navigation reference_path is not implemented yet. "
        f"Cannot build map metadata for {context}."
    )
