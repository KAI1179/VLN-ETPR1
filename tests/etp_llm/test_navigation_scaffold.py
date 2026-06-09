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


def test_available_llm_navigation_episode_ids_skips_missing(
    tmp_path, monkeypatch, capsys
):
    from vlnce_baselines.models.etp_llm import navigation

    entries = [
        type(
            "Entry",
            (),
            {
                "scene_id": "scene-a",
                "episode_id": 1,
                "unique_id": "R2R_train_1",
            },
        )(),
        type(
            "Entry",
            (),
            {
                "scene_id": "scene-a",
                "episode_id": 2,
                "unique_id": "R2R_train_2",
            },
        )(),
    ]
    monkeypatch.setattr(
        navigation.VLNCEEpisodeEntry,
        "iter_from",
        staticmethod(lambda dataset, splits: iter(entries)),
    )
    map_path = navigation.llm_navigation_cognitive_map_path(
        "scene-a",
        "R2R_train_2",
        "R2R",
        "train",
        cache_dir=tmp_path,
        model_key="test-model",
    )
    map_path.parent.mkdir(parents=True)
    map_path.touch()

    allowed = navigation.available_llm_navigation_episode_ids(
        "R2R",
        "train",
        cache_dir=tmp_path,
        model_key="test-model",
    )

    assert allowed == ["2"]
    assert (
        "finetuning_llm_navigation_maps: available=1 skipped_missing=1"
        in capsys.readouterr().out
    )
