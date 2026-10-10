"""CPU regressions for case-sensitive RxR cognitive-map cache identifiers."""

from types import SimpleNamespace

import pytest

from vlnce_baselines.models.etp_llm import navigation
from vlnce_baselines.ss_trainer_ETP_LLM import RLTrainer as LLMTrainer
from vlnce_baselines.ss_trainer_ETP_PriorGT import RLTrainer as PriorGTTrainer


@pytest.mark.parametrize("trainer", [LLMTrainer, PriorGTTrainer])
@pytest.mark.parametrize("dataset,prefix", [("r2r", "R2R"), ("rxr", "RxR")])
def test_episode_cache_id_preserves_canonical_dataset_case(trainer, dataset, prefix):
    instance = SimpleNamespace(config=SimpleNamespace(
        MODEL=SimpleNamespace(task_type=dataset),
        TASK_CONFIG=SimpleNamespace(DATASET=SimpleNamespace(SPLIT="train")),
    ))
    assert trainer._cognitive_map_cache_id(
        instance, SimpleNamespace(episode_id="42")
    ) == f"{prefix}_train_42"


@pytest.mark.parametrize("dataset,canonical", [
    ("rxr", "RxR"), ("RxR", "RxR"), ("RXR", "RxR"), ("r2r", "R2R"),
])
def test_cache_report_accepts_rxr_and_keeps_r2r_behavior(monkeypatch, dataset, canonical):
    calls = []

    def entries(name, splits):
        calls.append((name, splits))
        return [SimpleNamespace(episode_id="42", scene_id="scene", unique_id=f"{canonical}_train_42")]

    monkeypatch.setattr(navigation.VLNCEEpisodeEntry, "iter_from", entries)
    monkeypatch.setattr(navigation, "llm_navigation_cache_complete", lambda *a, **k: True)
    report = navigation.llm_navigation_cache_report(dataset, "train", require_boxes=False)
    assert calls == [(canonical, ("train",))]
    assert report.available_episode_ids == ["42"]
    assert report.missing_episode_ids == []
