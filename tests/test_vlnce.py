from __future__ import annotations

import gzip
import json


def test_vlnce_episode_entry_iter_from_uses_episode_reference_path_without_gt(
    tmp_path, monkeypatch
):
    from prior import vlnce

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    payload = {
        "episodes": [
            {
                "scene_id": "mp3d/TestScene/TestScene.glb",
                "episode_id": 7,
                "instruction": {
                    "instruction_text": "Walk to the table.",
                    "instruction_tokens": [1, 2, 3],
                },
                "start_position": [0.0, 0.0, 0.0],
                "start_rotation": [0.0, 0.0, 0.0, 1.0],
                "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            }
        ]
    }
    with gzip.open(split_dir / "train.json.gz", "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    monkeypatch.setattr(vlnce, "R2R_DIR", tmp_path)

    entries = list(vlnce.VLNCEEpisodeEntry.iter_from("R2R", splits=["train"]))

    assert len(entries) == 1
    assert entries[0].scene_id == "TestScene"
    assert entries[0].episode_id == 7
    assert entries[0].reference_path == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert not hasattr(entries[0], "positions")
    assert not hasattr(entries[0], "sample_id")
    assert not hasattr(entries[0], "source")


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
        reference_path=[],
    )

    assert entry.dataset == "R2R"
    assert entry.split == "val_unseen"
    assert entry.episode_id == 184
    assert entry.unique_id == "R2R_val_unseen_184"
