import pytest


def test_llm_navigation_policy_and_trainers_register():
    import vlnce_baselines  # noqa: F401
    from habitat_baselines.common.baseline_registry import baseline_registry

    assert baseline_registry.get_policy("LLMPolicy") is not None
    assert baseline_registry.get_trainer("SS-ETP-LLM") is not None
    assert baseline_registry.get_trainer("GRPO-ETP-LLM") is not None


def test_llm_navigation_cache_paths_are_namespaced_by_model_dataset_split_scene(tmp_path):
    from vlnce_baselines.models.etp_llm.navigation import (
        llm_navigation_cognitive_map_path,
        llm_navigation_prediction_path,
    )

    prediction_path = llm_navigation_prediction_path(
        "scene/with spaces",
        "R2R_train_42",
        "R2R",
        "train",
        cache_dir=tmp_path,
        model_key="Llama 3.1/8B",
    )
    map_path = llm_navigation_cognitive_map_path(
        "scene/with spaces",
        "R2R_train_42",
        "R2R",
        "train",
        cache_dir=tmp_path,
        model_key="Llama 3.1/8B",
    )

    assert prediction_path == (
        tmp_path
        / "Llama_3.1_8B"
        / "r2r"
        / "train"
        / "predictions"
        / "scene_with_spaces"
        / "R2R_train_42.txt"
    )
    assert map_path == (
        tmp_path
        / "Llama_3.1_8B"
        / "r2r"
        / "train"
        / "cognitive_maps"
        / "scene_with_spaces"
        / "R2R_train_42.npz"
    )


def test_llm_navigation_cache_loader_fails_fast_when_map_missing(tmp_path):
    from vlnce_baselines.models.etp_llm.navigation import (
        llm_cached_cognitive_map_to_tensors,
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Missing LLM-Navigation cognitive map cache: .*"
            "generate_navigation_cache"
        ),
    ):
        llm_cached_cognitive_map_to_tensors(
            "scene-a",
            "R2R_train_42",
            "R2R",
            "train",
            cache_dir=tmp_path,
            model_key="test-model",
        )
