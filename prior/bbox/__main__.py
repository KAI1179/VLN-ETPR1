"""Print MP3D semantic bounding boxes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Sequence

from tap import Tap

from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES
from prior.vlnce import VLNCEEpisodeEntry

from . import (
    AABB2D,
    LevelSemanticBoxes,
    OBB2D,
    RelevantSemanticBoxes,
    SceneSemanticBoxes,
)


class BoundingBoxArgs(Tap):
    scenes: List[str] = []
    """Scene ids to inspect when not using episode mode."""
    dataset: Optional[Literal["r2r", "rxr"]] = None
    """Dataset to use for relevant episode mode."""
    episode_id: Optional[int] = None
    """Episode id to use for relevant episode mode."""
    split: Optional[Literal["train", "val_seen", "val_unseen"]] = None
    """Dataset split to disambiguate duplicate episode ids."""
    output: Optional[Path] = None
    """Optional JSON output path for relevant episode boxes."""

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
    split: Optional[Literal["train", "val_seen", "val_unseen"]] = None,
) -> VLNCEEpisodeEntry:
    canonical_dataset = _canonical_dataset(dataset)
    episodes = (
        VLNCEEpisodeEntry.iter_from(canonical_dataset)
        if split is None
        else VLNCEEpisodeEntry.iter_from(canonical_dataset, splits=[split])
    )
    matches = [episode for episode in episodes if episode.episode_id == episode_id]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        sources = ", ".join(episode.split for episode in matches)
        raise ValueError(
            f"Episode {episode_id} found in multiple splits: {sources}; pass --split"
        )
    raise ValueError(f"Episode {episode_id} not found in {canonical_dataset}")


def _print_obb(box: OBB2D) -> None:
    print(
        f"      mentioned={box.mentioned} center={box.center} "
        f"half_extents={box.half_extents} rotation={box.rotation}"
    )


def _print_aabb(box: AABB2D) -> None:
    print(f"      mentioned={box.mentioned} {box.min} ~ {box.max}")


def _print_level(level_idx: int, level: LevelSemanticBoxes) -> None:
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


def _print_levels(levels: List[LevelSemanticBoxes]) -> None:
    for level_idx, level in enumerate(levels):
        _print_level(level_idx, level)


def _export_relevant_json(relevant: RelevantSemanticBoxes, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    relevant.save_json(output)
    print(f"Wrote relevant boxes JSON file to {output}")


def _print_scene_boxes(scene: str) -> None:
    print(f"Scene {scene}")
    _print_levels(SceneSemanticBoxes.from_scene_id(scene).levels)


def _scene_boxes(scene: str) -> SceneSemanticBoxes:
    return SceneSemanticBoxes.from_scene_id(scene)


def _relevant_episode_boxes(
    args: BoundingBoxArgs,
) -> tuple[VLNCEEpisodeEntry, RelevantSemanticBoxes]:
    assert args.dataset is not None
    assert args.episode_id is not None

    episode = _find_episode(args.dataset, args.episode_id, split=args.split)
    scene_boxes = SceneSemanticBoxes.from_scene_id(episode.scene_id)
    relevant = scene_boxes.relevant_to(
        episode.instruction,
        episode.reference_path,
        episode.start_direction_vector,
    )
    return episode, relevant


def _print_relevant_episode_boxes(args: BoundingBoxArgs) -> None:
    episode, relevant = _relevant_episode_boxes(args)
    print(
        f"Relevant bounding boxes for {episode.dataset} episode "
        f"{episode.episode_id} ({episode.split})"
    )
    print(f"Scene {episode.scene_id}")
    _print_level(relevant.level_idx, relevant.level)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if (args.dataset is None) != (args.episode_id is None):
        raise ValueError("--dataset and --episode-id must be provided together")

    if args.dataset is not None:
        if args.output is not None:
            _, relevant = _relevant_episode_boxes(args)
            print(f"Level {relevant.level_idx} ({relevant.level.range_y})")
            _export_relevant_json(relevant, args.output)
        else:
            _print_relevant_episode_boxes(args)
        return

    if args.output is not None:
        raise ValueError("--output requires --dataset and --episode-id")

    for scene in args.scenes:
        _print_scene_boxes(scene)


if __name__ == "__main__":
    main()
