from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List

import pytest

from prior import __main__ as prior_main
from prior.etp_r1 import __main__ as etp_r1_main
from prior.trajectory import InsufficientTrajectoryPointsError, WorldTrajectory3D


@dataclass
class FakeVLNCEEntry:
    episode_id: int
    ground_truth_trajectory: WorldTrajectory3D
    dataset: str = "R2R"
    scene_id: str = "scene"
    instruction: str = "Walk forward."
    start_direction_vector: tuple = (0.0, 1.0)

    @property
    def unique_id(self) -> str:
        return f"R2R_train_{self.episode_id}"


class FakeCognitiveMap:
    saved_paths: ClassVar[List[Path]] = []

    def save(self, path: Path) -> None:
        self.saved_paths.append(path)


class FakeRelevantBoxes:
    level_idx = 0
    saved_paths: ClassVar[List[Path]] = []

    def save(self, path: Path) -> None:
        self.saved_paths.append(path)

    def to_cognitive_map(self) -> FakeCognitiveMap:
        return FakeCognitiveMap()


def test_vlnce_cache_generator_warns_skips_bad_entry_and_continues(
    monkeypatch, tmp_path, caplog, capsys
):
    entries = [
        FakeVLNCEEntry(1, [(0.0, 0.0, 0.0)]),
        FakeVLNCEEntry(
            2,
            [(1.0, 0.0, 2.0), (3.0, 0.0, 4.0)],
        ),
    ]
    received_trajectories = []

    class FakeSceneBoxes:
        @staticmethod
        def from_scene_id(scene_id):
            return FakeSceneBoxes()

        def relevant_to(
            self,
            instruction,
            ground_truth_trajectory,
            start_direction_vector,
        ):
            received_trajectories.append(ground_truth_trajectory)
            if len(ground_truth_trajectory) < 2:
                raise InsufficientTrajectoryPointsError(
                    "ground_truth_trajectory must contain at least 2 "
                    "selected-level points for trajectory_keypoints"
                )
            return FakeRelevantBoxes()

    FakeCognitiveMap.saved_paths.clear()
    FakeRelevantBoxes.saved_paths.clear()
    monkeypatch.setattr(prior_main, "SceneSemanticBoxes", FakeSceneBoxes)

    generated, skipped = prior_main.generate_cognitive_maps(entries, tmp_path)

    assert received_trajectories == [
        [(0.0, 0.0, 0.0)],
        [(1.0, 0.0, 2.0), (3.0, 0.0, 4.0)],
    ]
    assert (generated, skipped) == (1, 1)
    assert FakeRelevantBoxes.saved_paths == [
        tmp_path / "boxes" / "scene" / "R2R_train_2.npz"
    ]
    assert FakeCognitiveMap.saved_paths == [
        tmp_path / "raster" / "scene" / "R2R_train_2.npz"
    ]
    assert "WARNING" in caplog.text
    assert "R2R_train_1" in caplog.text
    output = capsys.readouterr().out
    assert "generated=1 skipped=1" in output
    assert "skipped_invalid_trajectory=1" in output
    assert (
        "R2R_train_1: ground_truth_trajectory must contain at least 2 "
        "selected-level points for trajectory_keypoints"
    ) in output


def test_vlnce_cache_generator_main_allows_invalid_trajectory_skips(
    monkeypatch, tmp_path, capsys
):
    entries = [
        FakeVLNCEEntry(1, [(0.0, 0.0, 0.0)]),
        FakeVLNCEEntry(
            2,
            [(1.0, 0.0, 2.0), (3.0, 0.0, 4.0)],
        ),
    ]

    class FakeSceneBoxes:
        @staticmethod
        def from_scene_id(scene_id):
            return FakeSceneBoxes()

        def relevant_to(
            self,
            instruction,
            ground_truth_trajectory,
            start_direction_vector,
        ):
            if len(ground_truth_trajectory) < 2:
                raise InsufficientTrajectoryPointsError(
                    "ground_truth_trajectory must contain at least 2 "
                    "selected-level points for trajectory_keypoints"
                )
            return FakeRelevantBoxes()

    def iter_from(dataset, splits):
        if dataset == "R2R":
            return iter(entries)
        return iter(())

    FakeCognitiveMap.saved_paths.clear()
    FakeRelevantBoxes.saved_paths.clear()
    monkeypatch.setattr(prior_main, "SceneSemanticBoxes", FakeSceneBoxes)
    monkeypatch.setattr(prior_main.VLNCEEpisodeEntry, "iter_from", iter_from)
    monkeypatch.setattr(prior_main, "OUTPUT_DIR", tmp_path)

    prior_main.main()

    output = capsys.readouterr().out
    assert "generated=1 skipped=1" in output
    assert "skipped_invalid_trajectory=1" in output
    assert FakeRelevantBoxes.saved_paths == [
        tmp_path / "boxes" / "scene" / "R2R_train_2.npz"
    ]
    assert FakeCognitiveMap.saved_paths == [
        tmp_path / "raster" / "scene" / "R2R_train_2.npz"
    ]


def test_etp_r1_cache_generator_uses_positions_warns_and_continues(
    monkeypatch, tmp_path, caplog, capsys
):
    class FakeAnnotationEntry:
        def __init__(self, instr_id, positions):
            self.instr_id = instr_id
            self.scan = "scene"
            self.instruction = "Walk forward."
            self.start_direction_vector = (0.0, 1.0)
            self._positions = positions

        def positions(self):
            return self._positions

    entries = [
        FakeAnnotationEntry("bad", [[0.0, 0.0, 0.0]]),
        FakeAnnotationEntry(
            "good",
            [[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]],
        ),
    ]
    received_trajectories = []

    class FakeSceneBoxes:
        @staticmethod
        def from_scene_id(scene_id):
            return FakeSceneBoxes()

        def relevant_to(
            self,
            instruction,
            ground_truth_trajectory,
            start_direction_vector,
        ):
            received_trajectories.append(ground_truth_trajectory)
            if len(ground_truth_trajectory) < 2:
                raise InsufficientTrajectoryPointsError(
                    "ground_truth_trajectory must contain at least 2 "
                    "selected-level points for trajectory_keypoints"
                )
            return FakeRelevantBoxes()

    FakeCognitiveMap.saved_paths.clear()
    FakeRelevantBoxes.saved_paths.clear()
    monkeypatch.setattr(etp_r1_main, "ANNOTATION_FILES", ["annotations.jsonl"])
    monkeypatch.setattr(
        etp_r1_main.AnnotationEntry,
        "iter_from",
        staticmethod(lambda annotation_file: iter(entries)),
    )
    monkeypatch.setattr(etp_r1_main, "SceneSemanticBoxes", FakeSceneBoxes)
    monkeypatch.setattr(etp_r1_main, "OUTPUT_DIR", tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        etp_r1_main.main([])

    assert received_trajectories == [
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]],
    ]
    assert exc_info.value.code == 1
    assert FakeRelevantBoxes.saved_paths == [
        tmp_path / "boxes" / "scene" / "good.npz"
    ]
    assert FakeCognitiveMap.saved_paths == [
        tmp_path / "raster" / "scene" / "good.npz"
    ]
    assert "WARNING" in caplog.text
    assert "bad" in caplog.text
    assert "generated=1 skipped=1" in capsys.readouterr().out
