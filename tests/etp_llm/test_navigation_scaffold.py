import pytest
from types import SimpleNamespace


class _StubClipModel:
    def eval(self):
        return self

    def parameters(self):
        return []


def _stub_clip_load(monkeypatch):
    import clip

    monkeypatch.setattr(
        clip,
        "load",
        lambda *args, **kwargs: (_StubClipModel(), None),
    )


def test_llm_navigation_policy_and_trainers_register(monkeypatch):
    _stub_clip_load(monkeypatch)

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
        / "with_spaces"
        / "R2R_train_42.txt"
    )
    assert map_path == (
        tmp_path
        / "Llama_3.1_8B"
        / "r2r"
        / "train"
        / "cognitive_maps"
        / "with_spaces"
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


def test_llm_navigation_cache_loader_normalizes_habitat_scene_paths(
    tmp_path, monkeypatch
):
    _stub_clip_load(monkeypatch)

    from vlnce_baselines.models.etp_llm import navigation

    cache_id = "R2R_val_unseen_120"
    scene_path = "data/scene_datasets/mp3d/X7HyMhZNoso/X7HyMhZNoso.glb"
    map_path = (
        tmp_path
        / "llm5"
        / "r2r"
        / "val_unseen"
        / "cognitive_maps"
        / "X7HyMhZNoso"
        / f"{cache_id}.npz"
    )
    map_path.parent.mkdir(parents=True)
    map_path.touch()

    captured = {}

    def fake_cached_cognitive_map_to_tensors(
        scene_id,
        cache_id,
        cache_dir,
        random_rotation_augmentation,
    ):
        captured["scene_id"] = scene_id
        captured["cache_id"] = cache_id
        captured["cache_dir"] = cache_dir
        captured["random_rotation_augmentation"] = random_rotation_augmentation
        return {"grid": "ok"}

    monkeypatch.setattr(
        navigation,
        "cached_cognitive_map_to_tensors",
        fake_cached_cognitive_map_to_tensors,
    )

    tensors = navigation.llm_cached_cognitive_map_to_tensors(
        scene_path,
        cache_id,
        "r2r",
        "val_unseen",
        cache_dir=tmp_path,
        model_key="llm5",
    )

    assert tensors == {"grid": "ok"}
    assert captured == {
        "scene_id": scene_path,
        "cache_id": cache_id,
        "cache_dir": map_path.parent.parent,
        "random_rotation_augmentation": False,
    }


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


def test_llm_navigation_cache_report_counts_missing(tmp_path, monkeypatch):
    _stub_clip_load(monkeypatch)

    from vlnce_baselines.models.etp_llm import navigation

    entries = [
        type(
            "Entry",
            (),
            {
                "scene_id": "scene-a",
                "episode_id": 1,
                "unique_id": "R2R_val_unseen_1",
            },
        )(),
        type(
            "Entry",
            (),
            {
                "scene_id": "scene-a",
                "episode_id": 2,
                "unique_id": "R2R_val_unseen_2",
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
        "R2R_val_unseen_2",
        "R2R",
        "val_unseen",
        cache_dir=tmp_path,
        model_key="test-model",
    )
    map_path.parent.mkdir(parents=True)
    map_path.touch()

    report = navigation.llm_navigation_cache_report(
        "R2R",
        "val_unseen",
        ["1", "2"],
        cache_dir=tmp_path,
        model_key="test-model",
    )

    assert report.available_episode_ids == ["2"]
    assert report.missing_episode_ids == ["1"]
    assert report.missing_count == 1
    assert report.missing_rate == 0.5


def test_llm_trainer_marks_missing_eval_cache_as_generation_failure(monkeypatch):
    _stub_clip_load(monkeypatch)

    from vlnce_baselines import ss_trainer_ETP_LLM

    trainer = ss_trainer_ETP_LLM.RLTrainer.__new__(
        ss_trainer_ETP_LLM.RLTrainer
    )
    trainer.config = SimpleNamespace(
        MODEL=SimpleNamespace(
            task_type="R2R",
            MAP_ENCODER=SimpleNamespace(
                llm_cache_dir="/tmp/cache",
                llm_cache_model_key="test-model",
            ),
        ),
        TASK_CONFIG=SimpleNamespace(
            DATASET=SimpleNamespace(SPLIT="val_unseen")
        ),
    )

    monkeypatch.setattr(
        ss_trainer_ETP_LLM,
        "llm_navigation_cache_report",
        lambda *args, **kwargs: SimpleNamespace(
            available_episode_ids=["2"],
            missing_episode_ids=["1"],
            available_count=1,
            missing_count=1,
            missing_rate=0.5,
        ),
    )

    allowed = trainer._prepare_eval_episodes_allowed(["1", "2"])
    failure_stats = trainer._eval_prefilled_episode_stats()
    aggregated = trainer._augment_eval_aggregated_states(
        {"success": 0.25, "llm_cache_missing": 0.5},
        total=2,
    )

    assert allowed == ["2"]
    assert failure_stats["1"]["success"] == 0.0
    assert failure_stats["1"]["spl"] == 0.0
    assert failure_stats["1"]["llm_cache_missing"] == 1.0
    assert failure_stats["1"]["llm_generation_failure"] == 1.0
    assert aggregated["llm_cache_missing_count"] == 1
    assert aggregated["llm_cache_missing_rate"] == 0.5
