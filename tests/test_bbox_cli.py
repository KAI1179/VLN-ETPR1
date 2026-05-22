from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
            self.reference_path = [[0.0, 0.0, 0.0]]

    @property
    def source(self) -> str:
        return self.split


def test_relevant_episode_mode_prints_only_relevant_boxes(monkeypatch, capsys):
    all_boxes = [
        bbox.LevelSemanticBoxes(
            objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
            regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
            range_y=[None, None],
        )
    ]
    relevant_boxes = [
        bbox.LevelSemanticBoxes(
            objects=[[] for _ in range(bbox.OBJECT_CATEGORIES)],
            regions=[[] for _ in range(bbox.REGION_CATEGORIES)],
            range_y=[None, None],
        )
    ]
    relevant_boxes[0].objects[3] = [
        bbox.OBB2D(
            id="table-1",
            center=(0.0, 0.0),
            half_extents=(1.0, 1.0),
            axes=((1.0, 0.0), (0.0, 1.0)),
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
    assert "Object table" in output
    assert "table-1 mentioned=True" in output
