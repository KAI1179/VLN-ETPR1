from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from prior import bbox
from prior.bbox import __main__ as bbox_main
from prior.trajectory import InsufficientTrajectoryPointsError, WorldTrajectory3D


@dataclass
class FakeEpisode:
    dataset: str = "R2R"
    split: str = "train"
    scene_id: str = "17DRP5sb8fy"
    episode_id: int = 123
    instruction: str = "Walk to the table."
    ground_truth_trajectory: Optional[WorldTrajectory3D] = None
    start_direction_vector: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        if self.ground_truth_trajectory is None:
            self.ground_truth_trajectory = [
                (0.0, 5.0, 0.0),
                (1.0, 5.0, 0.0),
            ]


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
            center=(9.0, 9.0),
            half_extents=(1.0, 1.0),
            mentioned=True,
        )
    ]
    relevant_boxes[1].objects[3] = [
        bbox.OBB2D(
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
        lambda self, instruction, ground_truth_trajectory, start_direction_vector: (
            bbox.RelevantSemanticBoxes(
                level_idx=1,
                level=relevant_boxes[1],
                instruction=instruction,
                ground_truth_trajectory=[(p[0], p[2]) for p in ground_truth_trajectory],
                trajectory_keypoints=[
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
                start_direction_vector=start_direction_vector,
            )
        ),
    )

    bbox_main.main(["--dataset", "r2r", "--episode-id", "123"])

    output = capsys.readouterr().out
    assert "Relevant bounding boxes for R2R episode 123" in output
    assert "Scene 17DRP5sb8fy" in output
    assert "Level 1" in output
    assert "Object table" in output
    assert "mentioned=True center=(0.0, 0.0)" in output
    assert "center=(9.0, 9.0)" not in output


def test_episode_mode_exports_relevant_boxes_as_single_json_file(
    monkeypatch, tmp_path, capsys
):
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
    relevant_levels = [lower_level.model_copy(deep=True), upper_level]
    relevant_levels[1].objects[3].append(
        bbox.OBB2D(
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
        lambda self, instruction, ground_truth_trajectory, start_direction_vector: (
            bbox.RelevantSemanticBoxes(
                level_idx=1,
                level=relevant_levels[1],
                instruction=instruction,
                ground_truth_trajectory=[(p[0], p[2]) for p in ground_truth_trajectory],
                trajectory_keypoints=[
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
                start_direction_vector=start_direction_vector,
            )
        ),
    )

    output_path = tmp_path / "episode-boxes.json"
    bbox_main.main(
        ["--dataset", "r2r", "--episode-id", "123", "--output", str(output_path)]
    )

    loaded = bbox.RelevantSemanticBoxes.load_json(output_path)
    assert loaded == bbox.RelevantSemanticBoxes(
        level_idx=1,
        level=relevant_levels[1],
        instruction="Walk to the table.",
        ground_truth_trajectory=[(0.0, 0.0), (1.0, 0.0)],
        trajectory_keypoints=[
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        start_direction_vector=(0.0, 1.0),
    )
    output = capsys.readouterr().out
    assert "Level 1" in output
    assert "Wrote relevant boxes JSON file" in output


def test_relevant_episode_boxes_rejects_too_short_selected_level_trajectory(
    monkeypatch,
):
    level = bbox.LevelSemanticBoxes(
        objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
        range_y=[None, None],
    )
    episode = FakeEpisode(ground_truth_trajectory=[(0.0, 0.0, 0.0)])
    monkeypatch.setattr(
        bbox_main,
        "_find_episode",
        lambda dataset, episode_id, split=None: episode,
    )
    monkeypatch.setattr(
        bbox_main.SceneSemanticBoxes,
        "from_scene_id",
        staticmethod(lambda scene_id: bbox.SceneSemanticBoxes([level])),
    )

    args = bbox_main.parse_args(["--dataset", "r2r", "--episode-id", "123"])

    with pytest.raises(ValueError, match="at least 2 selected-level points"):
        bbox_main._relevant_episode_boxes(args)


def test_episode_export_warns_and_skips_too_short_trajectory(
    monkeypatch, tmp_path, caplog, capsys
):
    def reject(args):
        raise InsufficientTrajectoryPointsError(
            "ground_truth_trajectory must contain at least 2 "
            "selected-level points for trajectory_keypoints"
        )

    monkeypatch.setattr(bbox_main, "_relevant_episode_boxes", reject)
    output_path = tmp_path / "episode-boxes.json"

    with pytest.raises(SystemExit) as exc_info:
        bbox_main.main(
            ["--dataset", "r2r", "--episode-id", "123", "--output", str(output_path)]
        )

    assert exc_info.value.code == 1
    assert not output_path.exists()
    assert "WARNING" in caplog.text
    assert "123" in caplog.text
    assert "generated=0 skipped=1" in capsys.readouterr().out
