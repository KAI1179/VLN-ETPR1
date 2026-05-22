from __future__ import annotations

from dataclasses import dataclass

import pytest
from magnum import Matrix4, Vector3
from pydantic import BaseModel

from prior import bbox as box


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
    monkeypatch.setattr(box, "Mp3dObjectCategory", FakeCategory)
    monkeypatch.setattr(box, "Mp3dRegionCategory", FakeCategory)

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
    assert levels[0].offset_x == 2.0
    assert levels[0].offset_z == 3.0
    assert levels[0].objects[3] == [
        box.OBB2D(
            id="0_0_0",
            center=(4.0, 6.0),
            half_extents=(1.0, 2.0),
            axes=((1.0, 0.0), (0.0, 1.0)),
        )
    ]
    assert levels[0].regions[7] == [
        box.AABB2D(id="0_0", min=(2.0, 3.0), max=(8.0, 11.0))
    ]


def test_scene_semantic_boxes_from_scene_sorts_levels_by_floor_y(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(box, "Mp3dObjectCategory", FakeCategory)
    monkeypatch.setattr(box, "Mp3dRegionCategory", FakeCategory)

    upper_region = FakeRegion(
        id="upper",
        category=FakeCategory(6),
        aabb=FakeAABB(center=Vector3(0.0, 4.0, 0.0), sizes=Vector3(2.0, 2.0, 2.0)),
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

    assert levels[0].regions[6][0].id == "lower"
    assert levels[0].range_y == [None, 3.0]
    assert levels[1].regions[6][0].id == "upper"
    assert levels[1].range_y == [3.0, None]


def test_scene_semantic_boxes_relevant_to_keeps_nearby_boxes_and_marks_mentions():
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
        offset_x=-5.0,
        offset_z=-5.0,
    )
    level.objects[3] = [
        box.OBB2D(
            id="near-mentioned-table",
            center=(0.0, 0.0),
            half_extents=(1.0, 1.0),
            axes=((1.0, 0.0), (0.0, 1.0)),
        ),
        box.OBB2D(
            id="far-mentioned-table",
            center=(10.0, 0.0),
            half_extents=(1.0, 1.0),
            axes=((1.0, 0.0), (0.0, 1.0)),
        ),
    ]
    level.objects[1] = [
        box.OBB2D(
            id="near-unmentioned-chair",
            center=(0.0, 2.0),
            half_extents=(1.0, 1.0),
            axes=((1.0, 0.0), (0.0, 1.0)),
        )
    ]
    level.regions[7] = [
        box.AABB2D(id="near-unmentioned-bathroom", min=(-2.0, -2.0), max=(2.0, 2.0))
    ]

    relevant = box.SceneSemanticBoxes([level]).relevant_to(
        "walk to the table",
        reference_path=[[0.0, 0.0, 0.0]],
        max_distance=1.5,
        category_extractor=lambda instruction: ({3}, set()),
    )

    assert [obb.id for obb in relevant.levels[0].objects[3]] == ["near-mentioned-table"]
    assert relevant.levels[0].objects[3][0].mentioned is True
    assert [obb.id for obb in relevant.levels[0].objects[1]] == [
        "near-unmentioned-chair"
    ]
    assert relevant.levels[0].objects[1][0].mentioned is False
    assert [region.id for region in relevant.levels[0].regions[7]] == [
        "near-unmentioned-bathroom"
    ]
    assert relevant.levels[0].regions[7][0].mentioned is False


def test_scene_semantic_boxes_from_scene_id_uses_level_wise_disk_cache(
    tmp_path, monkeypatch
):
    scene_id = "scene"
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
        offset_x=1.0,
        offset_z=2.0,
    )
    level.objects[3].append(
        box.OBB2D(
            id="cached-table",
            center=(3.0, 4.0),
            half_extents=(1.0, 1.0),
            axes=((1.0, 0.0), (0.0, 1.0)),
        )
    )
    cache_dir = tmp_path / scene_id
    cache_dir.mkdir()
    level.save(cache_dir / "0.npz")

    def fail_load(*args, **kwargs):
        raise AssertionError("scene should not load when cache exists")

    box._scene_semantic_boxes_from_scene_id.cache_clear()
    monkeypatch.setattr(box, "SEMANTIC_BOX_DIR", tmp_path)
    monkeypatch.setattr(box.SemanticScene, "load_mp3d_house", fail_load)

    scene_boxes = box.SceneSemanticBoxes.from_scene_id(scene_id)

    assert scene_boxes.levels[0].offset_x == 1.0
    assert scene_boxes.levels[0].offset_z == 2.0
    assert scene_boxes.levels[0].objects[3][0].id == "cached-table"


def test_level_semantic_boxes_saves_and_loads_json(tmp_path):
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[-1.0, 2.0],
        offset_x=1.0,
        offset_z=2.0,
    )
    level.objects[3].append(
        box.OBB2D(
            id="json-table",
            center=(3.0, 4.0),
            half_extents=(1.0, 2.0),
            axes=((1.0, 0.0), (0.0, 1.0)),
            mentioned=True,
        )
    )
    level.regions[7].append(
        box.AABB2D(
            id="json-bathroom",
            min=(2.0, 3.0),
            max=(8.0, 11.0),
            mentioned=False,
        )
    )

    path = tmp_path / "level.json"
    level.save(path)
    loaded = box.LevelSemanticBoxes.load(path)

    assert isinstance(level, BaseModel)
    assert path.read_text(encoding="utf-8").startswith("{")
    assert loaded == level


def test_scene_semantic_boxes_to_cognitive_map_scales_unmentioned_confidence():
    level = box.LevelSemanticBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
        offset_x=-2.0,
        offset_z=-2.0,
    )
    level.objects[3].append(
        box.OBB2D(
            id="mentioned-table",
            center=(0.0, 0.0),
            half_extents=(1.0, 1.0),
            axes=((1.0, 0.0), (0.0, 1.0)),
        )
    )
    level.objects[1].append(
        box.OBB2D(
            id="unmentioned-chair",
            center=(1.0, 0.0),
            half_extents=(1.0, 1.0),
            axes=((1.0, 0.0), (0.0, 1.0)),
        )
    )

    cognitive_map = box.SceneSemanticBoxes([level]).to_cognitive_map(
        "walk to the table",
        reference_path=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        start_direction_vector=(0.0, 1.0),
        category_extractor=lambda instruction: ({3}, set()),
    )

    row, col = (4, 4)
    assert cognitive_map.grid[3, row, col] == 1.0
    assert cognitive_map.grid[1, row, col] == pytest.approx(box.IRRELEVANT_MULTIPLIER)
    assert cognitive_map.positions[0] == (4.0, 4.0)
