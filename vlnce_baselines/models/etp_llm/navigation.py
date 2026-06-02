"""Shared LLM-Navigation scaffold errors and guards."""

from typing import NoReturn


class LLMReferencePathNotImplementedError(NotImplementedError):
    """Raised when LLM-Navigation reaches unimplemented reference-path logic."""


def raise_llm_reference_path_not_implemented(context: str) -> NoReturn:
    raise LLMReferencePathNotImplementedError(
        "LLM-Navigation reference_path is not implemented yet. "
        f"Cannot build map metadata for {context}."
    )
