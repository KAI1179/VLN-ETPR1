import pytest


def test_t5_navigation_policy_and_trainers_register():
    import vlnce_baselines  # noqa: F401
    from habitat_baselines.common.baseline_registry import baseline_registry

    assert baseline_registry.get_policy("T5Policy") is not None
    assert baseline_registry.get_trainer("SS-ETP-T5") is not None
    assert baseline_registry.get_trainer("GRPO-ETP-T5") is not None


def test_t5_navigation_reference_path_fails_loudly():
    from vlnce_baselines.models.etp_t5.navigation import (
        T5ReferencePathNotImplementedError,
        raise_t5_reference_path_not_implemented,
    )

    with pytest.raises(
        T5ReferencePathNotImplementedError,
        match="T5-Navigation reference_path is not implemented",
    ):
        raise_t5_reference_path_not_implemented("unit-test")
