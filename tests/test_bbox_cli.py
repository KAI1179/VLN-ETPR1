from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from prior import bbox
from prior.bbox import __main__ as bbox_main


@dataclass
class FakeEpisode:
    dataset: str = "R2R"
    split: str = "train"
    scene_id: str = "17DRP5sb8fy"
    episode_id: int = 123
    instruction: str = "Walk to the table."
    reference_path: Optional[list[list[float]]] = None

    def __post_init__(self):
        if self.reference_path is None:
            self.reference_path = [[0.0, 5.0, 0.0]]

    @property
    def source(self) -> str:
        return self.split


def test_find_episode_requires_split_for_duplicate_episode_ids(monkeypatch):
    monkeypatch.setattr(
        bbox_main.VLNCEEpisodeEntry,
        "iter_from",
        lambda dataset, splits=("train", "val_unseen"): (
            episode
            for episode in [
                FakeEpisode(episode_id=184, split="train", scene_id="train-scene"),
                FakeEpisode(episode_id=184, split="val_unseen", scene_id="val-scene"),
            ]
            if episode.split in splits
        ),
    )

    with pytest.raises(ValueError, match="found in multiple splits"):
        bbox_main._find_episode("r2r", 184)

    episode = bbox_main._find_episode("r2r", 184, split="val_unseen")

    assert episode.scene_id == "val-scene"


def test_relevant_episode_mode_prints_only_relevant_boxes(monkeypatch, capsys):
    lower_level = bbox.LevelSemanticBoxes(
        objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
        range_y=[None, 3.0],
    )
    upper_level = bbox.LevelSemanticBoxes(
        objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
        range_y=[3.0, None],
    )
    all_boxes = [lower_level, upper_level]
    relevant_boxes = [
        bbox.LevelSemanticBoxes(
            objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
            regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
            range_y=[None, 3.0],
        ),
        bbox.LevelSemanticBoxes(
            objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
            regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
            range_y=[3.0, None],
        ),
    ]
    relevant_boxes[0].objects[3] = [
        bbox.OBB2D(
            id="wrong-level-table",
            center=(0.0, 0.0),
            half_extents=(1.0, 1.0),
            mentioned=True,
        )
    ]
    relevant_boxes[1].objects[3] = [
        bbox.OBB2D(
            id="table-1",
            center=(0.0, 0.0),
            half_extents=(1.0, 1.0),
            mentioned=True,
        )
    ]

    monkeypatch.setattr(
        bbox_main.VLNCEEpisodeEntry,
        "iter_from",
        lambda dataset: iter([FakeEpisode()]),
    )
    monkeypatch.setattr(
        bbox_main,
        "SceneSemanticBoxes",
        type(
            "FakeSceneSemanticBoxesFactory",
            (),
            {
                "from_scene_id": staticmethod(
                    lambda scene_id: bbox.SceneSemanticBoxes(all_boxes)
                )
            },
        ),
    )
    monkeypatch.setattr(
        bbox.SceneSemanticBoxes,
        "relevant_to",
        lambda self, instruction, reference_path: bbox.SceneSemanticBoxes(
            relevant_boxes
        ),
    )

    bbox_main.main(["--dataset", "r2r", "--episode-id", "123"])

    output = capsys.readouterr().out
    assert "Relevant bounding boxes for R2R episode 123" in output
    assert "Scene 17DRP5sb8fy" in output
    assert "Level 1" in output
    assert "Object table" in output
    assert "table-1 mentioned=True" in output
    assert "wrong-level-table" not in output


def test_episode_mode_exports_first_encountered_level_as_single_json_file(
    monkeypatch, tmp_path, capsys
):
    lower_level = bbox.LevelSemanticBoxes(
        objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
        range_y=[None, 3.0],
        offset_x=1.0,
        offset_z=2.0,
    )
    upper_level = bbox.LevelSemanticBoxes(
        objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
        range_y=[3.0, None],
    )
    relevant_levels = [lower_level.model_copy(deep=True), upper_level]
    relevant_levels[1].objects[3].append(
        bbox.OBB2D(
            id="upper-level-table",
            center=(0.0, 0.0),
            half_extents=(1.0, 1.0),
            mentioned=True,
        )
    )

    monkeypatch.setattr(
        bbox_main.VLNCEEpisodeEntry,
        "iter_from",
        lambda dataset: iter([FakeEpisode()]),
    )
    monkeypatch.setattr(
        bbox_main,
        "SceneSemanticBoxes",
        type(
            "FakeSceneSemanticBoxesFactory",
            (),
            {
                "from_scene_id": staticmethod(
                    lambda scene_id: bbox.SceneSemanticBoxes([lower_level, upper_level])
                )
            },
        ),
    )
    monkeypatch.setattr(
        bbox.SceneSemanticBoxes,
        "relevant_to",
        lambda self, instruction, reference_path: bbox.SceneSemanticBoxes(
            relevant_levels
        ),
    )

    output_path = tmp_path / "episode-boxes.json"
    bbox_main.main(
        ["--dataset", "r2r", "--episode-id", "123", "--output", str(output_path)]
    )

    loaded = bbox.LevelSemanticBoxes.load(output_path)
    assert loaded == relevant_levels[1]
    output = capsys.readouterr().out
    assert "Level 1" in output
    assert "Wrote level JSON file" in output
