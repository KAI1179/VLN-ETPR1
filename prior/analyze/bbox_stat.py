"""Statistics."""

from sys import argv

from dataclasses import dataclass, field
from typing import Iterator, Tuple, Dict
from collections import defaultdict

from prior.bbox import (
    SceneSemanticBoxes,
    ObjectOBB2Ds,
    RegionAABB2Ds,
)
from prior.constants import MAX_DISTANCE_CELLS, CELL_SIZE
from prior.vlnce import DEFAULT_SPLITS, VLNCEEpisodeEntry


@dataclass
class DatasetStats:
    entries: int = 0
    """Number of entries in the dataset."""
    objects_mentioned: "defaultdict[int, int]" = field(
        default_factory=lambda: defaultdict(int)
    )
    """Distribution of explicitly mentioned relevant objects. (cnt -> freq)"""
    objects_unmentioned: "defaultdict[int, int]" = field(
        default_factory=lambda: defaultdict(int)
    )
    """Distribution of un-mentioned relevant objects. (cnt -> freq)"""
    regions_mentioned: "defaultdict[int, int]" = field(
        default_factory=lambda: defaultdict(int)
    )
    """Distribution of explicitly mentioned relevant regions. (cnt -> freq)"""
    regions_unmentioned: "defaultdict[int, int]" = field(
        default_factory=lambda: defaultdict(int)
    )
    """Distribution of un-mentioned relevant regions. (cnt -> freq)"""


def _show_dist(dist: Dict[int, int], indent=4):
    for k, v in sorted(dist.items()):
        print(f"{' ' * indent}{k}: {v}")


def _show_stats(stats: DatasetStats):
    print(f"  Entries: {stats.entries}")
    print("  Objects mentioned dist:")
    _show_dist(stats.objects_mentioned)
    print("  Objects unmentioned dist:")
    _show_dist(stats.objects_unmentioned)
    print("  Regions mentioned dist:")
    _show_dist(stats.regions_mentioned)
    print("  Regions unmentioned dist:")
    _show_dist(stats.regions_unmentioned)


def _count_boxes(boxes: "ObjectOBB2Ds | RegionAABB2Ds") -> Tuple[int, int]:
    """Counts relevant boxes - mentioned and unmentioned."""
    mentioned = 0
    unmentioned = 0
    for cat_boxes in boxes:
        for box in cat_boxes:
            if box.mentioned:
                mentioned += 1
            else:
                unmentioned += 1
    return mentioned, unmentioned


def get_stats(dataset: Iterator[VLNCEEpisodeEntry], radius: int) -> DatasetStats:
    stats = DatasetStats()
    for entry in dataset:
        scene_boxes = SceneSemanticBoxes.from_scene_id(entry.scene_id)
        relevant_boxes = scene_boxes.relevant_to(
            entry.instruction,
            entry.reference_path,
            entry.start_direction_vector,
            radius * CELL_SIZE,
        )
        objects_mentioned, objects_unmentioned = _count_boxes(
            relevant_boxes.level.objects
        )
        regions_mentioned, regions_unmentioned = _count_boxes(
            relevant_boxes.level.regions
        )

        stats.entries += 1
        stats.objects_mentioned[objects_mentioned] += 1
        stats.objects_unmentioned[objects_unmentioned] += 1
        stats.regions_mentioned[regions_mentioned] += 1
        stats.regions_unmentioned[regions_unmentioned] += 1

    return stats


def main():
    if len(argv) >= 2:
        radius = int(argv[1])
    else:
        radius = MAX_DISTANCE_CELLS
    print(f"Radius: {radius} cell(s)")

    for dataset in ("R2R", "RxR"):
        stats = get_stats(
            VLNCEEpisodeEntry.iter_from(dataset, splits=DEFAULT_SPLITS), radius
        )
        print(f"=== {dataset} Stats ===")
        _show_stats(stats)


if __name__ == "__main__":
    main()
