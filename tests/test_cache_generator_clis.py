from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List

import pytest
import numpy as np

from prior import __main__ as prior_main
from prior import bbox as box
from prior import cognitive_map_generation
from prior.grid_map import CognitiveGridMap
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


def test_vlnce_generator_parser_accepts_source_radius_and_namespace():
    args = prior_main.parse_args(
        [
            "--map-source",
            "legacy",
            "--radius-m",
            "2.5",
            "--metadata-schema",
            "direction5",
            "--namespace",
            "gt.legacy.r2p5.direction5.v1",
            "--output-dir",
            "data/cognitive_maps",
        ]
    )

    assert args.map_source == "legacy"
    assert args.radius_m == 2.5
    assert args.metadata_schema == "direction5"
    assert args.namespace == "gt.legacy.r2p5.direction5.v1"
    assert args.output_dir == Path("data/cognitive_maps")


@pytest.mark.parametrize(
    ("map_source", "radius_m", "expected"),
    [
        ("legacy", 1.5, "gt.legacy.r1p5.path5.v1"),
        ("legacy", 2.5, "gt.legacy.r2p5.path5.v1"),
        ("bbox", 1.5, "gt.bbox.r1p5.path5.v1"),
        ("bbox", 2.5, "gt.bbox.r2p5.path5.v1"),
    ],
)
def test_map_cache_namespace_uses_source_and_radius_label(
    map_source, radius_m, expected
):
    assert (
        cognitive_map_generation.map_cache_namespace(map_source, radius_m) == expected
    )


def test_map_cache_namespace_includes_direction5_schema():
    assert (
        cognitive_map_generation.map_cache_namespace("legacy", 1.5, "direction5")
        == "gt.legacy.r1p5.direction5.v1"
    )


def test_save_cognitive_map_writes_direction5_metadata(tmp_path):
    cognitive_map = CognitiveGridMap()
    cognitive_map.grid[3, 1, 2] = 1.0
    cognitive_map.range_y = [None, 3.0]
    cognitive_map.start_direction_vector = (0.0, 1.0)
    cognitive_map.trajectory_keypoints = [
        (4.0, 5.0),
        (5.0, 5.0),
        (5.0, 6.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    path = tmp_path / "map.npz"

    cognitive_map_generation.save_cognitive_map(
        cognitive_map,
        path,
        "direction5",
        [
            (10.0, 0.0, 20.0),
            (11.0, 0.0, 20.0),
            (12.0, 0.0, 20.0),
            (12.0, 0.0, 21.0),
        ],
    )

    data = np.load(path, allow_pickle=True)
    assert set(data.files) == {
        "grid",
        "range_y",
        "direction_vectors",
        "start_direction_vector",
        "start_position",
    }
    np.testing.assert_array_equal(data["grid"], cognitive_map.grid)
    assert data["direction_vectors"].shape == (5, 2)
    assert data["direction_vectors"][0].tolist() == [0.0, -1.0]
    assert data["direction_vectors"][1].tolist() == [-1.0, 0.0]
    assert data["direction_vectors"][2:].tolist() == [[0.0, 0.0]] * 3
    assert data["start_direction_vector"].tolist() == [0.0, 1.0]
    assert data["start_position"].tolist() == [4.0, 5.0]


def test_vlnce_generator_rejects_direction5_for_bbox(tmp_path):
    with pytest.raises(ValueError, match="direction5 metadata is only supported"):
        prior_main.generate_cognitive_maps(
            [],
            tmp_path,
            map_source="bbox",
            metadata_schema="direction5",
        )


def test_vlnce_bbox_generator_passes_radius_and_namespace(monkeypatch, tmp_path):
    entries = [
        FakeVLNCEEntry(
            2,
            [(1.0, 0.0, 2.0), (3.0, 0.0, 4.0)],
        ),
    ]
    captured = {}

    class FakeSceneBoxes:
        @staticmethod
        def from_scene_id(scene_id):
            captured["scene_id"] = scene_id
            return FakeSceneBoxes()

        def relevant_to(
            self,
            instruction,
            ground_truth_trajectory,
            start_direction_vector,
            max_distance,
        ):
            captured["instruction"] = instruction
            captured["ground_truth_trajectory"] = ground_truth_trajectory
            captured["start_direction_vector"] = start_direction_vector
            captured["max_distance"] = max_distance
            return FakeRelevantBoxes()

    FakeCognitiveMap.saved_paths.clear()
    FakeRelevantBoxes.saved_paths.clear()
    monkeypatch.setattr(prior_main, "SceneSemanticBoxes", FakeSceneBoxes)

    generated, skipped = prior_main.generate_cognitive_maps(
        entries,
        tmp_path,
        map_source="bbox",
        radius_m=2.5,
        namespace="gt.bbox.r2p5.path5.v1",
    )

    assert (generated, skipped) == (1, 0)
    assert captured == {
        "scene_id": "scene",
        "instruction": "Walk forward.",
        "ground_truth_trajectory": [(1.0, 0.0, 2.0), (3.0, 0.0, 4.0)],
        "start_direction_vector": (0.0, 1.0),
        "max_distance": 2.5,
    }
    assert FakeRelevantBoxes.saved_paths == [
        tmp_path / "gt.bbox.r2p5.path5.v1" / "boxes" / "scene" / "R2R_train_2.npz"
    ]
    assert FakeCognitiveMap.saved_paths == [
        tmp_path / "gt.bbox.r2p5.path5.v1" / "raster" / "scene" / "R2R_train_2.npz"
    ]


def test_vlnce_legacy_generator_writes_legacy_raster_and_compat_boxes(
    monkeypatch, tmp_path
):
    entries = [
        FakeVLNCEEntry(
            2,
            [(1.0, 0.0, 2.0), (3.0, 0.0, 4.0)],
        ),
    ]
    legacy_map = FakeCognitiveMap()
    captured = {}

    class FakeSceneBoxes:
        @staticmethod
        def from_scene_id(scene_id):
            captured["scene_id"] = scene_id
            return FakeSceneBoxes()

        def relevant_to(
            self,
            instruction,
            ground_truth_trajectory,
            start_direction_vector,
            max_distance,
        ):
            captured["max_distance"] = max_distance
            return FakeRelevantBoxes()

    def fake_legacy_cognitive_map(
        scene_boxes,
        instruction,
        ground_truth_trajectory,
        start_direction_vector,
        radius_m,
    ):
        captured["legacy_instruction"] = instruction
        captured["legacy_trajectory"] = ground_truth_trajectory
        captured["legacy_start_direction_vector"] = start_direction_vector
        captured["legacy_radius_m"] = radius_m
        return legacy_map

    FakeCognitiveMap.saved_paths.clear()
    FakeRelevantBoxes.saved_paths.clear()
    monkeypatch.setattr(prior_main, "SceneSemanticBoxes", FakeSceneBoxes)
    monkeypatch.setattr(
        cognitive_map_generation,
        "legacy_cognitive_map",
        fake_legacy_cognitive_map,
    )

    generated, skipped = prior_main.generate_cognitive_maps(
        entries,
        tmp_path,
        map_source="legacy",
        radius_m=1.5,
        namespace="gt.legacy.r1p5.path5.v1",
    )

    assert (generated, skipped) == (1, 0)
    assert captured == {
        "scene_id": "scene",
        "max_distance": 1.5,
        "legacy_instruction": "Walk forward.",
        "legacy_trajectory": [(1.0, 0.0, 2.0), (3.0, 0.0, 4.0)],
        "legacy_start_direction_vector": (0.0, 1.0),
        "legacy_radius_m": 1.5,
    }
    assert FakeRelevantBoxes.saved_paths == [
        tmp_path / "gt.legacy.r1p5.path5.v1" / "boxes" / "scene" / "R2R_train_2.npz"
    ]
    assert FakeCognitiveMap.saved_paths == [
        tmp_path / "gt.legacy.r1p5.path5.v1" / "raster" / "scene" / "R2R_train_2.npz"
    ]


def test_legacy_cognitive_map_lives_in_shared_generation_module():
    assert not hasattr(prior_main, "_legacy_cognitive_map")
    assert hasattr(cognitive_map_generation, "legacy_cognitive_map")


def test_legacy_cognitive_map_copies_square_path_neighborhood():
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
    )
    level.objects[3].append(
        box.OBB2D(
            center=(0.25, 0.25),
            half_extents=(0.25, 0.25),
        )
    )
    level.objects[1].append(
        box.OBB2D(
            center=(1.25, 0.25),
            half_extents=(0.25, 0.25),
        )
    )
    scene_boxes = box.SceneSemanticBoxes([level])

    cognitive_map = cognitive_map_generation.legacy_cognitive_map(
        scene_boxes,
        "walk to the table",
        [(0.25, 0.0, 0.25), (0.75, 0.0, 0.25)],
        (0.0, 1.0),
        radius_m=0.5,
    )

    assert cognitive_map.grid[3, 0, 0] == 1.0
    assert cognitive_map.grid[1, 2, 0] == pytest.approx(0.6)
    assert cognitive_map.grid[3, 4, 0] == 0.0
    assert cognitive_map.trajectory_keypoints == [
        (0.25, 0.25),
        (0.75, 0.25),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    assert cognitive_map.start_direction_vector == (0.0, 1.0)


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
            max_distance,
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
        tmp_path / "gt.bbox.r1p5.path5.v1" / "boxes" / "scene" / "R2R_train_2.npz"
    ]
    assert FakeCognitiveMap.saved_paths == [
        tmp_path / "gt.bbox.r1p5.path5.v1" / "raster" / "scene" / "R2R_train_2.npz"
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
            max_distance,
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

    prior_main.main([])

    output = capsys.readouterr().out
    assert "generated=1 skipped=1" in output
    assert "skipped_invalid_trajectory=1" in output
    assert FakeRelevantBoxes.saved_paths == [
        tmp_path / "gt.bbox.r1p5.path5.v1" / "boxes" / "scene" / "R2R_train_2.npz"
    ]
    assert FakeCognitiveMap.saved_paths == [
        tmp_path / "gt.bbox.r1p5.path5.v1" / "raster" / "scene" / "R2R_train_2.npz"
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
            max_distance,
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
        tmp_path / "gt.bbox.r1p5.path5.v1" / "boxes" / "scene" / "good.npz"
    ]
    assert FakeCognitiveMap.saved_paths == [
        tmp_path / "gt.bbox.r1p5.path5.v1" / "raster" / "scene" / "good.npz"
    ]
    assert "WARNING" in caplog.text
    assert "bad" in caplog.text
    assert "generated=1 skipped=1" in capsys.readouterr().out


def test_etp_r1_generator_parser_accepts_source_radius_namespace_and_samples():
    args = etp_r1_main.parse_args(
        [
            "--map-source",
            "legacy",
            "--radius-m",
            "2.5",
            "--metadata-schema",
            "direction5",
            "--namespace",
            "gt.legacy.r2p5.direction5.v1",
            "--output-dir",
            "data/cognitive_maps_etp_r1",
            "42_0",
        ]
    )

    assert args.map_source == "legacy"
    assert args.radius_m == 2.5
    assert args.metadata_schema == "direction5"
    assert args.namespace == "gt.legacy.r2p5.direction5.v1"
    assert args.output_dir == Path("data/cognitive_maps_etp_r1")
    assert args.sampled_instr_ids == ["42_0"]
