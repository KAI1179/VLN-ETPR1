import gzip
import json

import numpy as np
import pytest

from prior.constants import OBJECT_CATEGORIES
from vlnce_baselines.models.etp_llm import llm_grid_oracle_cache
from vlnce_baselines.models.etp_llm.llm_grid_evidence import GridEvidenceIndex
from vlnce_baselines.models.etp_llm.llm_grid_oracle_cache import (
    LLMGridOracleCacheArgs,
    OracleSensorFrame,
    StartEpisode,
    _provenance,
    collect_oracle_cache,
    group_start_episodes,
    iter_start_episodes,
    project_oracle_frames,
)


def _frame(object_category: int) -> OracleSensorFrame:
    return OracleSensorFrame(
        depth_m=np.asarray([[2.0]], dtype=np.float32),
        object_categories=np.asarray([[object_category]], dtype=np.int16),
        region_categories=np.asarray([[6]], dtype=np.int16),
        sensor_position=(0.0, 1.25, 0.0),
        sensor_rotation=(0.0, 0.0, 0.0, 1.0),
    )


def test_project_oracle_frames_uses_ego_and_target_coordinate_frames() -> None:
    evidence = project_oracle_frames(
        [_frame(1)],
        start_position=(0.0, 0.0, 0.0),
        start_rotation=(0.0, 0.0, 0.0, 1.0),
        target_origin_xz=(-25.0, -25.0),
    )

    assert evidence.ego_semantic_grid[1, 23, 25] == 1
    assert evidence.ego_semantic_grid[OBJECT_CATEGORIES + 6, 23, 25] == 1
    assert evidence.target_semantic_grid[1, 25, 23] == 1
    assert evidence.ego_observed_mask[25, 25]
    assert evidence.ego_observed_mask[23, 25]
    assert evidence.ego_free_mask[25, 25]
    assert not evidence.ego_free_mask[23, 25]
    assert evidence.start_position == (25.0, 25.0)
    assert evidence.start_direction == pytest.approx((1.0, 0.0))


def test_project_oracle_frames_keeps_free_space_endpoint_in_free_mask() -> None:
    evidence = project_oracle_frames(
        [_frame(17)],
        start_position=(0.0, 0.0, 0.0),
        start_rotation=(0.0, 0.0, 0.0, 1.0),
        target_origin_xz=(-25.0, -25.0),
    )

    assert evidence.ego_semantic_grid[17, 23, 25] == 1
    assert evidence.ego_free_mask[23, 25]
    assert np.all(evidence.ego_free_mask <= evidence.ego_observed_mask)


def test_group_start_episodes_deduplicates_shared_start_observation() -> None:
    first = StartEpisode(
        dataset="R2R",
        split="train",
        episode_id="1",
        scene_id="scene",
        start_position=(1.0, 2.0, 3.0),
        start_rotation=(0.0, 0.0, 0.0, 1.0),
    )
    sibling = StartEpisode(
        dataset="R2R",
        split="train",
        episode_id="2",
        scene_id="scene",
        start_position=first.start_position,
        start_rotation=first.start_rotation,
    )

    groups = group_start_episodes([sibling, first])

    assert len(groups) == 1
    assert groups[0].observation_id == first.observation_id
    assert {episode.episode_id for episode in groups[0].episodes} == {"1", "2"}


def test_iter_start_episodes_ignores_forbidden_episode_fields(tmp_path) -> None:
    path = tmp_path / "split.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as destination:
        json.dump(
            {
                "episodes": [
                    {
                        "episode_id": 7,
                        "scene_id": "mp3d/scene/scene.glb",
                        "start_position": [1.0, 2.0, 3.0],
                        "start_rotation": [0.0, 0.0, 0.0, 1.0],
                        "instruction": {"instruction_text": "forbidden"},
                        "goals": [{"position": [9.0, 9.0, 9.0]}],
                        "reference_path": [[8.0, 8.0, 8.0]],
                    }
                ]
            },
            destination,
        )

    episodes = list(iter_start_episodes("R2R", "train", path=path))

    assert episodes == [
        StartEpisode(
            dataset="R2R",
            split="train",
            episode_id="7",
            scene_id="scene",
            start_position=(1.0, 2.0, 3.0),
            start_rotation=(0.0, 0.0, 0.0, 1.0),
        )
    ]
    assert set(vars(episodes[0])) == {
        "dataset",
        "split",
        "episode_id",
        "scene_id",
        "start_position",
        "start_rotation",
    }


def test_provenance_records_fixed_sensor_and_leakage_contract() -> None:
    manifest = _provenance()

    assert manifest["sensor"]["yaw_degrees"] == list(range(0, 360, 30))
    assert manifest["sensor"]["normalize_depth"] is False
    assert manifest["projection"]["cell_size_m"] == 1.0
    assert manifest["leakage_contract"]["allowed_episode_fields"] == [
        "episode_id",
        "scene_id",
        "start_position",
        "start_rotation",
    ]
    assert "instruction" in manifest["leakage_contract"]["forbidden_inputs"]


def test_collect_oracle_cache_writes_strict_split_index(tmp_path, monkeypatch) -> None:
    episode = StartEpisode(
        dataset="R2R",
        split="train",
        episode_id="7",
        scene_id="scene",
        start_position=(0.0, 0.0, 0.0),
        start_rotation=(0.0, 0.0, 0.0, 1.0),
    )
    evidence = project_oracle_frames(
        [_frame(1)],
        start_position=episode.start_position,
        start_rotation=episode.start_rotation,
        target_origin_xz=(-25.0, -25.0),
    )
    monkeypatch.setattr(
        llm_grid_oracle_cache,
        "iter_start_episodes",
        lambda dataset, split: iter((episode,)),
    )
    monkeypatch.setattr(
        llm_grid_oracle_cache,
        "_collect_groups",
        lambda groups, gpu_device_id: iter(((tuple(groups)[0], evidence),)),
    )
    args = LLMGridOracleCacheArgs().parse_args([
        "--output-dir",
        str(tmp_path),
        "--evidence-key",
        "test-oracle",
        "--datasets",
        "R2R",
        "--splits",
        "train",
    ])

    indexes = collect_oracle_cache(args)

    assert len(indexes) == 1
    index = GridEvidenceIndex.load(tmp_path, "test-oracle", "R2R", "train")
    assert index.episodes == (
        llm_grid_oracle_cache.EvidenceEpisode(
            example_id="R2R_train_7",
            scene_id="scene",
            observation_id=episode.observation_id,
        ),
    )
    assert index.load_evidence("R2R_train_7").start_position == (25.0, 25.0)
