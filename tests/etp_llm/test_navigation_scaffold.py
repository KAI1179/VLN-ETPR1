import pytest


def test_llm_navigation_policy_and_trainers_register():
    import vlnce_baselines  # noqa: F401
    from habitat_baselines.common.baseline_registry import baseline_registry

    assert baseline_registry.get_policy("LLMPolicy") is not None
    assert baseline_registry.get_trainer("SS-ETP-LLM") is not None
    assert baseline_registry.get_trainer("GRPO-ETP-LLM") is not None


def test_llm_navigation_reference_path_fails_loudly():
    from vlnce_baselines.models.etp_llm.navigation import (
        LLMReferencePathNotImplementedError,
        raise_llm_reference_path_not_implemented,
    )

    with pytest.raises(
        LLMReferencePathNotImplementedError,
        match="LLM-Navigation reference_path is not implemented",
    ):
        raise_llm_reference_path_not_implemented("unit-test")
