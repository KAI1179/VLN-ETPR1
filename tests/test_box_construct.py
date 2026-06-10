from __future__ import annotations

import math
from dataclasses import dataclass, fields

import pytest
from magnum import Matrix4, Vector3
from pydantic import BaseModel

from prior._coords import meters_to_grid
from prior import bbox as box
from prior.bbox import _construct as box_construct


def test_semantic_box_models_do_not_expose_source_ids():
    assert "id" not in {field.name for field in fields(box.OBB2D)}
    assert "id" not in {field.name for field in fields(box.AABB2D)}


@dataclass
class FakeCategory:
    category_index: int

    def index(self) -> int:
        return self.category_index


@dataclass
class FakeAABB:
    center: Vector3
    sizes: Vector3


@dataclass
class FakeOBB:
    center: Vector3
    half_extents: Vector3
    local_to_world: Matrix4


@dataclass
class FakeObject:
    id: str
    category: FakeCategory
    aabb: FakeAABB
    obb: FakeOBB


@dataclass
class FakeRegion:
    id: str
    category: FakeCategory
    aabb: FakeAABB
    objects: list[FakeObject]


@dataclass
class FakeLevel:
    aabb: FakeAABB
    regions: list[FakeRegion]


@dataclass
class FakeScene:
    levels: list[FakeLevel]


def test_scene_semantic_boxes_from_scene_groups_2d_boxes_by_mapped_category(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(box_construct, "Mp3dObjectCategory", FakeCategory)
    monkeypatch.setattr(box_construct, "Mp3dRegionCategory", FakeCategory)

    obj = FakeObject(
        id="0_0_0",
        category=FakeCategory(5),  # table -> mapped category 3
        aabb=FakeAABB(center=Vector3(4.0, 1.0, 6.0), sizes=Vector3(2.0, 2.0, 4.0)),
        obb=FakeOBB(
            center=Vector3(4.0, 1.0, 6.0),
            half_extents=Vector3(1.0, 0.5, 2.0),
            local_to_world=Matrix4(),
        ),
    )
    region = FakeRegion(
        id="0_0",
        category=FakeCategory(5),  # bathroom -> mapped category 7
        aabb=FakeAABB(center=Vector3(5.0, 0.0, 7.0), sizes=Vector3(6.0, 2.0, 8.0)),
        objects=[obj],
    )
    scene = FakeScene(levels=[FakeLevel(aabb=region.aabb, regions=[region])])

    scene_boxes = box.SceneSemanticBoxes.from_scene(scene)
    levels = scene_boxes.levels

    assert len(levels) == 1
    assert levels[0].range_y == [None, None]
    assert not hasattr(scene_boxes, "level_origins")
    assert levels[0].objects[3] == [
        box.OBB2D(
            center=(2.0, 3.0),
            half_extents=(1.0, 2.0),
            rotation=0.0,
        )
    ]
    assert levels[0].regions[7] == [box.AABB2D(min=(0.0, 0.0), max=(6.0, 8.0))]


def test_scene_semantic_boxes_from_scene_sorts_levels_by_floor_y(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(box_construct, "Mp3dObjectCategory", FakeCategory)
    monkeypatch.setattr(box_construct, "Mp3dRegionCategory", FakeCategory)

    upper_region = FakeRegion(
        id="upper",
        category=FakeCategory(6),
        aabb=FakeAABB(center=Vector3(10.0, 4.0, 10.0), sizes=Vector3(2.0, 2.0, 2.0)),
        objects=[],
    )
    lower_region = FakeRegion(
        id="lower",
        category=FakeCategory(6),
        aabb=FakeAABB(center=Vector3(0.0, 0.0, 0.0), sizes=Vector3(2.0, 2.0, 2.0)),
        objects=[],
    )
    scene = FakeScene(
        levels=[
            FakeLevel(aabb=upper_region.aabb, regions=[upper_region]),
            FakeLevel(aabb=lower_region.aabb, regions=[lower_region]),
        ]
    )

    levels = box.SceneSemanticBoxes.from_scene(scene).levels

    assert levels[0].regions[6][0].min == (0.0, 0.0)
    assert levels[0].range_y == [None, 3.0]
    assert levels[1].regions[6][0].min == (0.0, 0.0)
    assert levels[1].range_y == [3.0, None]


def test_scene_semantic_boxes_relevant_to_returns_typed_relevant_level():
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
    )
    level.objects[3] = [
        box.OBB2D(
            center=(0.0, 0.0),
            half_extents=(1.0, 1.0),
        ),
        box.OBB2D(
            center=(10.0, 0.0),
            half_extents=(1.0, 1.0),
        ),
    ]
    level.objects[1] = [
        box.OBB2D(
            center=(0.0, 2.0),
            half_extents=(1.0, 1.0),
        )
    ]
    level.regions[7] = [box.AABB2D(min=(-2.0, -2.0), max=(2.0, 2.0))]

    relevant = box.SceneSemanticBoxes([level]).relevant_to(
        "walk to the table",
        ground_truth_trajectory=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
        ],
        start_direction_vector=(0.0, 1.0),
        max_distance=1.5,
        category_extractor=lambda instruction: ({3}, set()),
    )

    assert isinstance(relevant, box.RelevantSemanticBoxes)
    assert relevant.level_idx == 0
    assert relevant.instruction == "walk to the table"
    assert relevant.ground_truth_trajectory == [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ]
    assert relevant.trajectory_keypoints == [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    assert relevant.start_direction_vector == (0.0, 1.0)
    assert [obb.center for obb in relevant.level.objects[3]] == [(0.0, 0.0)]
    assert relevant.level.objects[3][0].mentioned is True
    assert [obb.center for obb in relevant.level.objects[1]] == [(0.0, 2.0)]
    assert relevant.level.objects[1][0].mentioned is False
    assert [region.min for region in relevant.level.regions[7]] == [(-2.0, -2.0)]
    assert relevant.level.regions[7][0].mentioned is False


def test_scene_semantic_boxes_skips_one_point_level_for_keypoints():
    first_level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[0.0, 1.0],
    )
    first_level.objects[3] = [
        box.OBB2D(center=(0.0, 0.0), half_extents=(1.0, 1.0))
    ]
    second_level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[1.0, 2.0],
    )
    second_level.objects[3] = [
        box.OBB2D(center=(2.0, 0.0), half_extents=(1.0, 1.0))
    ]

    relevant = box.SceneSemanticBoxes([first_level, second_level]).relevant_to(
        "walk to the table",
        ground_truth_trajectory=[
            (0.0, 0.5, 0.0),
            (2.0, 1.5, 0.0),
            (3.0, 1.5, 0.0),
        ],
        start_direction_vector=(0.0, 1.0),
        max_distance=1.5,
        category_extractor=lambda instruction: ({3}, set()),
    )

    assert relevant.level_idx == 1
    assert relevant.ground_truth_trajectory == [(2.0, 0.0), (3.0, 0.0)]
    assert relevant.trajectory_keypoints == [
        (2.0, 0.0),
        (3.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    assert [obb.center for obb in relevant.level.objects[3]] == [(2.0, 0.0)]


def test_scene_semantic_boxes_from_scene_id_uses_level_wise_disk_cache(
    tmp_path, monkeypatch
):
    scene_id = "scene"
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
    )
    level.objects[3].append(
        box.OBB2D(
            center=(3.0, 4.0),
            half_extents=(1.0, 1.0),
        )
    )
    cache_dir = tmp_path / scene_id
    cache_dir.mkdir()
    level.save(cache_dir / "0.npz")
    (cache_dir / "origins.json").write_text("[[1.0, 2.0]]", encoding="utf-8")

    def fail_load(*args, **kwargs):
        raise AssertionError("scene should not load when cache exists")

    box._scene_semantic_boxes_from_scene_id.cache_clear()
    monkeypatch.setattr(box_construct, "SEMANTIC_BOX_DIR", tmp_path)
    monkeypatch.setattr(box_construct.SemanticScene, "load_mp3d_house", fail_load)

    scene_boxes = box.SceneSemanticBoxes.from_scene_id(scene_id)

    assert not hasattr(scene_boxes, "level_origins")
    assert scene_boxes.levels[0].objects[3][0].center == (3.0, 4.0)
    relevant = scene_boxes.relevant_to(
        "",
        ground_truth_trajectory=[(4.0, 0.0, 6.0), (5.0, 0.0, 6.0)],
        start_direction_vector=(0.0, 1.0),
        category_extractor=lambda instruction: (set(), set()),
    )
    assert relevant.ground_truth_trajectory == [(3.0, 4.0), (4.0, 4.0)]
    assert relevant.trajectory_keypoints == [
        (3.0, 4.0),
        (4.0, 4.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]


def test_relevant_semantic_boxes_saves_and_loads_json(tmp_path):
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[-1.0, 2.0],
    )
    level.objects[3].append(
        box.OBB2D(
            center=(3.0, 4.0),
            half_extents=(1.0, 2.0),
            mentioned=True,
        )
    )
    level.regions[7].append(
        box.AABB2D(
            min=(2.0, 3.0),
            max=(8.0, 11.0),
            mentioned=False,
        )
    )
    relevant = box.RelevantSemanticBoxes(
        level_idx=1,
        level=level,
        instruction="walk to the table",
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

    path = tmp_path / "relevant.json"
    relevant.save_json(path)
    loaded = box.RelevantSemanticBoxes.load_json(path)

    assert isinstance(relevant, BaseModel)
    assert '"rotation"' in path.read_text(encoding="utf-8")
    assert loaded == relevant


def test_relevant_semantic_boxes_requires_exactly_five_trajectory_keypoints():
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
    )

    with pytest.raises(
        ValueError,
        match="trajectory_keypoints must contain exactly 5 points, got 2",
    ):
        box.RelevantSemanticBoxes(
            level_idx=0,
            level=level,
            instruction="",
            ground_truth_trajectory=[(0.0, 0.0), (1.0, 0.0)],
            trajectory_keypoints=[(0.0, 0.0), (1.0, 0.0)],
            start_direction_vector=(0.0, 1.0),
        )


def test_relevant_semantic_boxes_rotates_right_angle_with_positive_grid_frame():
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
    )
    level.objects[3].append(
        box.OBB2D(
            center=(2.0, 3.0),
            half_extents=(0.5, 1.0),
            rotation=0.25,
            mentioned=True,
        )
    )
    level.regions[7].append(
        box.AABB2D(
            min=(1.0, 2.0),
            max=(4.0, 5.0),
            mentioned=True,
        )
    )
    relevant = box.RelevantSemanticBoxes(
        level_idx=0,
        level=level,
        instruction="walk to the table",
        ground_truth_trajectory=[(1.0, 1.0), (2.0, 3.0)],
        trajectory_keypoints=[
            (1.0, 1.0),
            (2.0, 3.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        start_direction_vector=(0.0, 1.0),
    )

    rotated = relevant.rotate_by_right_angle(1)

    assert rotated.level.objects[3][0] == box.OBB2D(
        center=(46.5, 2.0),
        half_extents=(0.5, 1.0),
        rotation=0.25 + math.pi / 2.0,
        mentioned=True,
    )
    assert rotated.level.regions[7][0] == box.AABB2D(
        min=(44.5, 1.0),
        max=(47.5, 4.0),
        mentioned=True,
    )
    assert rotated.ground_truth_trajectory == [(48.5, 1.0), (46.5, 2.0)]
    assert rotated.trajectory_keypoints == [
        (48.5, 1.0),
        (46.5, 2.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    assert rotated.start_direction_vector == (1.0, -0.0)
    assert meters_to_grid(48.5, 1.0) == (97.0, 2.0)


def test_relevant_semantic_boxes_rotates_leading_origin_but_preserves_zero_suffix():
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
    )
    relevant = box.RelevantSemanticBoxes(
        level_idx=0,
        level=level,
        instruction="",
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

    rotated = relevant.rotate_by_right_angle(1)

    assert rotated.trajectory_keypoints == [
        (49.5, 0.0),
        (49.5, 1.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]


def test_obb_distance_uses_rotation():
    rotated = box.OBB2D(
        center=(0.0, 0.0),
        half_extents=(2.0, 0.5),
        rotation=math.pi / 4.0,
    )

    assert box._point_to_obb_distance((2**0.5, 2**0.5), rotated) == pytest.approx(0.0)
    assert box._point_to_obb_distance((1.5, -1.5), rotated) > 1.0


def test_relevant_semantic_boxes_to_cognitive_map_scales_unmentioned_confidence():
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
    )
    level.objects[3].append(
        box.OBB2D(
            center=(0.0, 0.0),
            half_extents=(1.0, 1.0),
        )
    )
    level.objects[1].append(
        box.OBB2D(
            center=(1.0, 0.0),
            half_extents=(1.0, 1.0),
        )
    )

    relevant = box.SceneSemanticBoxes([level]).relevant_to(
        "walk to the table",
        ground_truth_trajectory=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        start_direction_vector=(0.0, 1.0),
        category_extractor=lambda instruction: ({3}, set()),
    )
    cognitive_map = relevant.to_cognitive_map()

    row, col = (0, 0)
    assert cognitive_map.grid[3, row, col] == 1.0
    assert cognitive_map.grid[1, row, col] == pytest.approx(box.IRRELEVANT_MULTIPLIER)
    assert cognitive_map.trajectory_keypoints == [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]


def test_relevant_to_rejects_one_selected_level_point():
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
    )

    with pytest.raises(ValueError, match="at least 2 selected-level points"):
        box.SceneSemanticBoxes([level]).relevant_to(
            "",
            ground_truth_trajectory=[(0.0, 0.0, 0.0)],
            start_direction_vector=(0.0, 1.0),
            category_extractor=lambda instruction: (set(), set()),
        )


def test_scene_semantic_boxes_has_no_direct_cognitive_map_shortcut():
    assert not hasattr(box.SceneSemanticBoxes, "to_cognitive_map")
    assert not hasattr(box.SceneSemanticBoxes, "first_encountered_level")
