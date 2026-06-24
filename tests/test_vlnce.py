from __future__ import annotations

import gzip
import json

import pytest


def _episode(episode_id=7):
    return {
        "scene_id": "mp3d/TestScene/TestScene.glb",
        "episode_id": episode_id,
        "instruction": {
            "instruction_text": "Walk to the table.",
            "instruction_tokens": [1, 2, 3],
        },
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "reference_path": [[99.0, 0.0, 99.0]],
    }


def _write_gzip_json(path, payload):
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump(payload, file)


def test_vlnce_episode_entry_iter_from_uses_r2r_ground_truth_trajectory(
    tmp_path, monkeypatch
):
    from prior import vlnce

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    trajectory = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ]
    _write_gzip_json(split_dir / "train.json.gz", {"episodes": [_episode()]})
    _write_gzip_json(
        split_dir / "train_gt.json.gz",
        {"7": {"locations": trajectory}},
    )
    monkeypatch.setattr(vlnce, "R2R_DIR", tmp_path)

    entries = list(vlnce.VLNCEEpisodeEntry.iter_from("R2R", splits=["train"]))

    assert len(entries) == 1
    assert entries[0].scene_id == "TestScene"
    assert entries[0].episode_id == 7
    assert entries[0].ground_truth_trajectory == trajectory
    assert not hasattr(entries[0], "reference_path")
    assert not hasattr(entries[0], "positions")


def test_vlnce_episode_entry_iter_from_normalizes_full_habitat_scene_path(
    tmp_path, monkeypatch
):
    from prior import vlnce

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    episode = _episode()
    episode["scene_id"] = "data/scene_datasets/mp3d/X7HyMhZNoso/X7HyMhZNoso.glb"
    _write_gzip_json(split_dir / "train.json.gz", {"episodes": [episode]})
    _write_gzip_json(
        split_dir / "train_gt.json.gz",
        {"7": {"locations": [(0.0, 0.0, 0.0)]}},
    )
    monkeypatch.setattr(vlnce, "R2R_DIR", tmp_path)

    entries = list(vlnce.VLNCEEpisodeEntry.iter_from("R2R", splits=["train"]))

    assert entries[0].scene_id == "X7HyMhZNoso"


def test_vlnce_episode_entry_iter_from_uses_rxr_guide_ground_truth_file(
    tmp_path, monkeypatch
):
    from prior import vlnce

    split_dir = tmp_path / "val_seen"
    split_dir.mkdir()
    trajectory = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    _write_gzip_json(
        split_dir / "val_seen_guide.json.gz",
        {"episodes": [_episode(12)]},
    )
    _write_gzip_json(
        split_dir / "val_seen_guide_gt.json.gz",
        {"12": {"locations": trajectory}},
    )
    monkeypatch.setattr(vlnce, "RxR_DIR", tmp_path)

    entries = list(vlnce.VLNCEEpisodeEntry.iter_from("RxR", splits=["val_seen"]))

    assert len(entries) == 1
    assert entries[0].ground_truth_trajectory == trajectory


def test_vlnce_episode_entry_iter_from_skips_non_english_instructions(
    tmp_path, monkeypatch
):
    from prior import vlnce

    split_dir = tmp_path / "val_seen"
    split_dir.mkdir()
    english_episode = _episode(12)
    english_episode["instruction"]["language"] = "en-US"
    non_english_episode = _episode(13)
    non_english_episode["instruction"]["language"] = "hi-IN"
    _write_gzip_json(
        split_dir / "val_seen_guide.json.gz",
        {"episodes": [english_episode, non_english_episode]},
    )
    _write_gzip_json(
        split_dir / "val_seen_guide_gt.json.gz",
        {
            "12": {"locations": [(0.0, 0.0, 0.0)]},
            "13": {"locations": [(1.0, 0.0, 0.0)]},
        },
    )
    monkeypatch.setattr(vlnce, "RxR_DIR", tmp_path)

    entries = list(vlnce.VLNCEEpisodeEntry.iter_from("RxR", splits=["val_seen"]))

    assert [entry.episode_id for entry in entries] == [12]


def test_vlnce_episode_entry_iter_from_fails_when_gt_file_missing(
    tmp_path, monkeypatch
):
    from prior import vlnce

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    _write_gzip_json(split_dir / "train.json.gz", {"episodes": [_episode()]})
    monkeypatch.setattr(vlnce, "R2R_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing VLN-CE ground-truth file"):
        list(vlnce.VLNCEEpisodeEntry.iter_from("R2R", splits=["train"]))


def test_vlnce_episode_entry_iter_from_fails_when_episode_trajectory_missing(
    tmp_path, monkeypatch
):
    from prior import vlnce

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    _write_gzip_json(split_dir / "train.json.gz", {"episodes": [_episode()]})
    _write_gzip_json(split_dir / "train_gt.json.gz", {})
    monkeypatch.setattr(vlnce, "R2R_DIR", tmp_path)

    with pytest.raises(
        ValueError,
        match="Missing ground-truth trajectory for R2R train episode 7",
    ):
        list(vlnce.VLNCEEpisodeEntry.iter_from("R2R", splits=["train"]))


def test_vlnce_episode_entry_keeps_dataset_split_and_episode_explicit():
    from prior import vlnce

    entry = vlnce.VLNCEEpisodeEntry(
        dataset="R2R",
        split="val_unseen",
        scene_id="17DRP5sb8fy",
        episode_id=184,
        instruction="go",
        start_position=[0.0, 0.0, 0.0],
        start_rotation=[0.0, 0.0, 0.0, 1.0],
        instruction_tokens=[],
        ground_truth_trajectory=[],
    )

    assert entry.dataset == "R2R"
    assert entry.split == "val_unseen"
    assert entry.episode_id == 184
    assert entry.unique_id == "R2R_val_unseen_184"
