"""Export cognitive grid maps to NumPy arrays."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from prior import DATA_DIR
from prior.bbox import SceneSemanticBoxes
from prior.cognitive_map_generation import (
    DEFAULT_RADIUS_M,
    MapSource,
    build_cognitive_map,
    cache_paths,
    cache_root,
    map_cache_namespace,
)
from prior.trajectory import InsufficientTrajectoryPointsError
from prior.vlnce import DEFAULT_SPLITS, VLNCEEpisodeEntry

OUTPUT_DIR = DATA_DIR / "cognitive_maps"
LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VLN-CE cognitive-map caches."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root output directory for cognitive-map cache namespaces.",
    )
    parser.add_argument(
        "--map-source",
        choices=("bbox", "legacy"),
        default="bbox",
        help="Cognitive-map construction method.",
    )
    parser.add_argument(
        "--radius-m",
        type=float,
        default=DEFAULT_RADIUS_M,
        help="Path-neighborhood radius in meters.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help=(
            "Optional cache namespace under output-dir. Defaults to "
            "<map-source>_r<radius>, such as legacy_r2p5."
        ),
    )
    return parser.parse_args(argv)


def generate_cognitive_maps(
    entries: Iterable[Any],
    output_dir: Path = OUTPUT_DIR,
    map_source: MapSource = "bbox",
    radius_m: float = DEFAULT_RADIUS_M,
    namespace: Optional[str] = None,
) -> Tuple[int, int]:
    data = list(entries)
    cache_namespace = namespace or map_cache_namespace(map_source, radius_m)
    cache_root(output_dir, cache_namespace).mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0
    skipped_invalid: List[Tuple[str, str]] = []

    for i, entry in enumerate(data):
        scene_id = entry.scene_id
        dataset = entry.dataset
        episode_key = entry.unique_id
        print(
            f"[{dataset}] Processing episode {i + 1:0>5}/{len(data):0>5} "
            f"({episode_key}, scene {scene_id})",
            end="\r",
        )

        boxes_path, raster_path = cache_paths(
            output_dir,
            cache_namespace,
            scene_id,
            episode_key,
        )
        if boxes_path.exists() and raster_path.exists():
            print(
                f"[{dataset}] Cognitive map for episode {episode_key} in "
                f"scene {scene_id} already exists, skipping."
            )
            skipped += 1
            continue

        try:
            scene_boxes = SceneSemanticBoxes.from_scene_id(scene_id)
            relevant_boxes, cognitive_map = build_cognitive_map(
                scene_boxes,
                entry.instruction,
                entry.ground_truth_trajectory,
                entry.start_direction_vector,
                map_source,
                radius_m,
            )
            boxes_path.parent.mkdir(parents=True, exist_ok=True)
            raster_path.parent.mkdir(parents=True, exist_ok=True)
            relevant_boxes.save(boxes_path)
            cognitive_map.save(raster_path)
        except InsufficientTrajectoryPointsError as error:
            LOGGER.warning("skipping %s: %s", episode_key, error)
            skipped_invalid.append((episode_key, str(error)))
            skipped += 1
            continue

        generated += 1
        print(
            f"[{dataset}] Saved cognitive map for episode {episode_key} "
            f"in scene {scene_id}"
        )

    print(f"generated={generated} skipped={skipped}")
    if skipped_invalid:
        print(f"skipped_invalid_trajectory={len(skipped_invalid)}")
        for episode_key, reason in skipped_invalid:
            print(f"  {episode_key}: {reason}")
    return generated, skipped


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    namespace = args.namespace or map_cache_namespace(args.map_source, args.radius_m)
    generate_cognitive_maps(
        [
            *VLNCEEpisodeEntry.iter_from("R2R", splits=DEFAULT_SPLITS),
            *VLNCEEpisodeEntry.iter_from("RxR", splits=DEFAULT_SPLITS),
        ],
        args.output_dir,
        map_source=args.map_source,
        radius_m=args.radius_m,
        namespace=namespace,
    )


if __name__ == "__main__":
    main()
