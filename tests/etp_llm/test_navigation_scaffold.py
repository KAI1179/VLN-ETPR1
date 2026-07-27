import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


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


def test_navigation_manifest_preserves_compatible_cache(tmp_path: Path) -> None:
    from vlnce_baselines.models.etp_llm.navigation import (
        ensure_llm_navigation_manifest,
    )

    expected = {"dataset": "R2R", "split": "val_seen", "max_new_tokens": 64}
    ensure_llm_navigation_manifest(tmp_path, expected)
    manifest_path = tmp_path / "manifest.json"
    original = manifest_path.read_text(encoding="utf-8")

    ensure_llm_navigation_manifest(tmp_path, expected)

    assert manifest_path.read_text(encoding="utf-8") == original


def test_navigation_manifest_rejects_incompatible_resume(tmp_path: Path) -> None:
    from vlnce_baselines.models.etp_llm.navigation import (
        ensure_llm_navigation_manifest,
    )

    ensure_llm_navigation_manifest(
        tmp_path,
        {"dataset": "R2R", "split": "val_seen", "max_new_tokens": 64},
    )

    with pytest.raises(ValueError, match="manifest mismatch.*max_new_tokens"):
        ensure_llm_navigation_manifest(
            tmp_path,
            {"dataset": "R2R", "split": "val_seen", "max_new_tokens": 128},
        )


def test_llm_navigation_policy_and_trainers_register(monkeypatch):
    _stub_clip_load(monkeypatch)

    import vlnce_baselines  # noqa: F401
    from habitat_baselines.common.baseline_registry import baseline_registry

    assert baseline_registry.get_policy("LLMBoxesCurrentPolicy") is not None
    assert baseline_registry.get_policy("LLMGridTry5Policy") is not None
    assert baseline_registry.get_trainer("SS-ETP-LLM") is not None
    assert baseline_registry.get_trainer("GRPO-ETP-LLM") is not None


def test_llm_navigation_cache_paths_are_namespaced_by_model_dataset_split_scene(
    tmp_path,
):
    from vlnce_baselines.models.etp_llm.navigation import (
        llm_navigation_cognitive_map_boxes_path,
        llm_navigation_cognitive_map_raster_path,
        llm_navigation_prediction_path,
        llm_navigation_status_path,
    )

    prediction_path = llm_navigation_prediction_path(
        "scene/with spaces",
        "R2R_train_42",
        "R2R",
        "train",
        cache_dir=tmp_path,
        model_key="Llama 3.1/8B",
    )
    boxes_path = llm_navigation_cognitive_map_boxes_path(
        "scene/with spaces",
        "R2R_train_42",
        "R2R",
        "train",
        cache_dir=tmp_path,
        model_key="Llama 3.1/8B",
    )
    raster_path = llm_navigation_cognitive_map_raster_path(
        "scene/with spaces",
        "R2R_train_42",
        "R2R",
        "train",
        cache_dir=tmp_path,
        model_key="Llama 3.1/8B",
    )
    status_path = llm_navigation_status_path(
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
    assert boxes_path == (
        tmp_path
        / "Llama_3.1_8B"
        / "r2r"
        / "train"
        / "cognitive_maps"
        / "boxes"
        / "with_spaces"
        / "R2R_train_42.npz"
    )
    assert raster_path == (
        tmp_path
        / "Llama_3.1_8B"
        / "r2r"
        / "train"
        / "cognitive_maps"
        / "raster"
        / "with_spaces"
        / "R2R_train_42.npz"
    )
    assert status_path == (
        tmp_path
        / "Llama_3.1_8B"
        / "r2r"
        / "train"
        / "status"
        / "with_spaces"
        / "R2R_train_42.json"
    )


def test_llm_navigation_cache_loader_fails_fast_when_map_missing(tmp_path):
    from vlnce_baselines.models.etp_llm.navigation import (
        llm_cached_cognitive_map_to_tensors,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing LLM-Navigation raster cognitive map cache",
    ):
        llm_cached_cognitive_map_to_tensors(
            "scene-a",
            "R2R_train_42",
            "R2R",
            "train",
            metadata_schema="path5",
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
    raster_path = (
        tmp_path
        / "llm5"
        / "r2r"
        / "val_unseen"
        / "cognitive_maps"
        / "raster"
        / "X7HyMhZNoso"
        / f"{cache_id}.npz"
    )
    raster_path.parent.mkdir(parents=True)
    raster_path.touch()

    captured = {}

    def fake_cognitive_map_file_to_tensors(
        cache_path,
        random_rotation_augmentation,
        metadata_schema,
    ):
        captured["cache_path"] = cache_path
        captured["random_rotation_augmentation"] = random_rotation_augmentation
        captured["metadata_schema"] = metadata_schema
        return {"grid": "ok"}

    monkeypatch.setattr(
        navigation,
        "cognitive_map_file_to_tensors",
        fake_cognitive_map_file_to_tensors,
    )

    tensors = navigation.llm_cached_cognitive_map_to_tensors(
        scene_path,
        cache_id,
        "r2r",
        "val_unseen",
        metadata_schema="direction5",
        cache_dir=tmp_path,
        model_key="llm5",
    )

    assert tensors == {"grid": "ok"}
    assert captured == {
        "cache_path": raster_path,
        "random_rotation_augmentation": False,
        "metadata_schema": "direction5",
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
    raster_path = navigation.llm_navigation_cognitive_map_raster_path(
        "scene-a",
        "R2R_train_2",
        "R2R",
        "train",
        cache_dir=tmp_path,
        model_key="test-model",
    )
    status_path = navigation.llm_navigation_status_path(
        "scene-a",
        "R2R_train_2",
        "R2R",
        "train",
        cache_dir=tmp_path,
        model_key="test-model",
    )
    raster_path.parent.mkdir(parents=True)
    status_path.parent.mkdir(parents=True)
    raster_path.touch()
    status_path.write_text(json.dumps({"status": "complete"}))

    allowed = navigation.available_llm_navigation_episode_ids(
        "R2R",
        "train",
        require_boxes=False,
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
    raster_path = navigation.llm_navigation_cognitive_map_raster_path(
        "scene-a",
        "R2R_val_unseen_2",
        "R2R",
        "val_unseen",
        cache_dir=tmp_path,
        model_key="test-model",
    )
    status_path = navigation.llm_navigation_status_path(
        "scene-a",
        "R2R_val_unseen_2",
        "R2R",
        "val_unseen",
        cache_dir=tmp_path,
        model_key="test-model",
    )
    raster_path.parent.mkdir(parents=True)
    status_path.parent.mkdir(parents=True)
    raster_path.touch()
    status_path.write_text(json.dumps({"status": "complete"}))

    report = navigation.llm_navigation_cache_report(
        "R2R",
        "val_unseen",
        ["1", "2"],
        require_boxes=False,
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

    trainer = ss_trainer_ETP_LLM.RLTrainer.__new__(ss_trainer_ETP_LLM.RLTrainer)
    trainer.config = SimpleNamespace(
        MODEL=SimpleNamespace(
            task_type="R2R",
            MAP_ENCODER=SimpleNamespace(
                architecture="try5",
                source="llm_grid",
                llm_cache_dir="/tmp/cache",
                llm_cache_model_key="test-model",
            ),
        ),
        TASK_CONFIG=SimpleNamespace(DATASET=SimpleNamespace(SPLIT="val_unseen")),
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


def test_eval_aggregation_supports_rank_local_failure_metrics(monkeypatch):
    _stub_clip_load(monkeypatch)

    from vlnce_baselines import ss_trainer_ETP_PriorGT

    rank_stats = [
        {
            "0": {
                "ndtw": np.float64(0.5),
                "steps_taken": np.float32(100.0),
                "success": np.float32(1.0),
            },
            "1": {
                "ndtw": np.float64(1.0),
                "steps_taken": np.float32(100.0),
                "success": np.float32(0.0),
            },
        },
        {
            "2": {
                "llm_cache_missing": 1.0,
                "llm_generation_failure": 1.0,
                "ndtw": np.float64(0.0),
                "steps_taken": np.float32(0.0),
                "success": np.float32(0.0),
            }
        },
    ]
    metric_keys = [
        ["ndtw", "steps_taken", "success"],
        [
            "llm_cache_missing",
            "llm_generation_failure",
            "ndtw",
            "steps_taken",
            "success",
        ],
    ]
    expected_local_vectors = [
        [2.0, 0.0, 0.0, 1.5, 200.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ]
    expected_aggregated = {
        "llm_cache_missing": 1 / 3,
        "llm_generation_failure": 1 / 3,
        "ndtw": 0.5,
        "steps_taken": 200 / 3,
        "success": 1 / 3,
    }

    for rank in range(2):
        trainer = ss_trainer_ETP_PriorGT.RLTrainer.__new__(
            ss_trainer_ETP_PriorGT.RLTrainer
        )
        trainer.world_size = 2
        trainer.device = torch.device("cpu")
        trainer.stat_eps = rank_stats[rank]

        def fake_all_gather_object(output, local_keys):
            assert local_keys == metric_keys[rank]
            output[:] = metric_keys

        def fake_all_reduce(tensor, op):
            assert op == torch.distributed.ReduceOp.SUM
            assert tensor.dtype == torch.float64
            assert tensor.tolist() == expected_local_vectors[rank]
            tensor += torch.tensor(
                expected_local_vectors[1 - rank],
                dtype=torch.float64,
            )

        monkeypatch.setattr(
            ss_trainer_ETP_PriorGT.distr,
            "all_gather_object",
            fake_all_gather_object,
        )
        monkeypatch.setattr(
            ss_trainer_ETP_PriorGT.distr,
            "all_reduce",
            fake_all_reduce,
        )

        _, aggregated, total = trainer._aggregate_eval_episode_stats()

        assert total == 3
        assert aggregated == pytest.approx(expected_aggregated)
