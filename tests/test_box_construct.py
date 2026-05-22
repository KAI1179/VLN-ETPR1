from __future__ import annotations

from dataclasses import dataclass

import pytest
from magnum import Matrix4, Vector3

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


def test_construct_bounding_boxes_from_scene_groups_2d_boxes_by_mapped_category(
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

    levels = box.construct_bounding_boxes_from_scene(scene)

    assert len(levels) == 1
    assert levels[0].range_y == [None, None]
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


def test_construct_bounding_boxes_from_scene_sorts_levels_by_floor_y(
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

    levels = box.construct_bounding_boxes_from_scene(scene)

    assert levels[0].regions[6][0].id == "lower"
    assert levels[0].range_y == [None, 3.0]
    assert levels[1].regions[6][0].id == "upper"
    assert levels[1].range_y == [3.0, None]


def test_extract_relevant_bounding_boxes_keeps_nearby_boxes_and_marks_mentions():
    level = box.LevelBoundingBoxes(
        objects=[[] for _ in range(box.OBJECT_CATEGORIES)],
        regions=[[] for _ in range(box.REGION_CATEGORIES)],
        range_y=[None, None],
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

    relevant = box.extract_relevant_bounding_boxes(
        [level],
        "walk to the table",
        positions=[[0.0, 0.0, 0.0]],
        max_distance=1.5,
        category_extractor=lambda instruction: ({3}, set()),
    )

    assert [obb.id for obb in relevant[0].objects[3]] == ["near-mentioned-table"]
    assert relevant[0].objects[3][0].mentioned is True
    assert [obb.id for obb in relevant[0].objects[1]] == ["near-unmentioned-chair"]
    assert relevant[0].objects[1][0].mentioned is False
    assert [region.id for region in relevant[0].regions[7]] == [
        "near-unmentioned-bathroom"
    ]
    assert relevant[0].regions[7][0].mentioned is False
