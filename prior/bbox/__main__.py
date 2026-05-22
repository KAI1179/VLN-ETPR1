"""Print MP3D semantic bounding boxes."""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

from tap import Tap

from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES
from prior.vlnce import VLNCEEpisodeEntry

from . import (
    AABB2D,
    LevelSemanticBoxes,
    OBB2D,
    SceneSemanticBoxes,
)


class BoundingBoxArgs(Tap):
    scenes: List[str] = []
    """Scene ids to inspect when not using episode mode."""
    dataset: Optional[Literal["r2r", "rxr"]] = None
    """Dataset to use for relevant episode mode."""
    episode_id: Optional[int] = None
    """Episode id to use for relevant episode mode."""

    def configure(self) -> None:
        self.add_argument("scenes", nargs="*")


def parse_args(argv: Optional[Sequence[str]] = None) -> BoundingBoxArgs:
    return BoundingBoxArgs(underscores_to_dashes=True).parse_args(argv)


def _canonical_dataset(dataset: Literal["r2r", "rxr"]) -> Literal["R2R", "RxR"]:
    if dataset == "r2r":
        return "R2R"
    return "RxR"


def _find_episode(
    dataset: Literal["r2r", "rxr"],
    episode_id: int,
) -> VLNCEEpisodeEntry:
    canonical_dataset = _canonical_dataset(dataset)
    for episode in VLNCEEpisodeEntry.iter_from(canonical_dataset):
        if episode.episode_id == episode_id:
            return episode
    raise ValueError(f"Episode {episode_id} not found in {canonical_dataset}")


def _print_obb(box: OBB2D) -> None:
    print(
        f"      {box.id} mentioned={box.mentioned} "
        f"center={box.center} half_extents={box.half_extents} axes={box.axes}"
    )


def _print_aabb(box: AABB2D) -> None:
    print(f"      {box.id} mentioned={box.mentioned} {box.min} ~ {box.max}")


def _print_levels(levels: List[LevelSemanticBoxes]) -> None:
    for level_idx, level in enumerate(levels):
        print(f"  Level {level_idx} ({level.range_y})")

        for object_cat, boxes in enumerate(level.objects):
            if not boxes:
                continue
            object_name = MAPPED_OBJECT_NAMES[object_cat]
            print(f"    Object {object_name}")
            for box in boxes:
                _print_obb(box)

        for region_cat, boxes in enumerate(level.regions):
            if not boxes:
                continue
            region_name = MAPPED_REGION_NAMES[region_cat]
            print(f"    Region {region_name}")
            for box in boxes:
                _print_aabb(box)


def _print_scene_boxes(scene: str) -> None:
    print(f"Scene {scene}")
    _print_levels(SceneSemanticBoxes.from_scene_id(scene).levels)


def _print_relevant_episode_boxes(args: BoundingBoxArgs) -> None:
    assert args.dataset is not None
    assert args.episode_id is not None

    episode = _find_episode(args.dataset, args.episode_id)
    relevant_scene = SceneSemanticBoxes.from_scene_id(episode.scene_id).relevant_to(
        episode.instruction,
        episode.reference_path,
    )

    print(
        f"Relevant bounding boxes for {episode.dataset} episode "
        f"{episode.episode_id} ({episode.source})"
    )
    print(f"Scene {episode.scene_id}")
    _print_levels(relevant_scene.levels)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if (args.dataset is None) != (args.episode_id is None):
        raise ValueError("--dataset and --episode-id must be provided together")

    if args.dataset is not None:
        _print_relevant_episode_boxes(args)
        return

    for scene in args.scenes:
        _print_scene_boxes(scene)


if __name__ == "__main__":
    main()
