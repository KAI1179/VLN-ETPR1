"""Export cognitive grid maps to NumPy arrays."""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import math
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from prior import DATA_DIR
from prior.bbox import LevelSemanticBoxes, SceneSemanticBoxes
from prior.bbox._rasterize import _rasterize_level_semantic_boxes
from prior.bbox._relevance import _first_usable_level_points
from prior.constants import (
    CELL_SIZE,
    COLS,
    MAX_DISTANCE_CELLS,
    OBJECT_CATEGORIES,
    REGION_CATEGORIES,
    ROWS,
)
from prior.grid_map import CognitiveGridMap
from prior.grid_map._cognitive import extract_categories
from prior.trajectory import (
    InsufficientTrajectoryPointsError,
    select_trajectory_keypoints,
)
from prior.vlnce import DEFAULT_SPLITS, VLNCEEpisodeEntry

OUTPUT_DIR = DATA_DIR / "cognitive_maps"
LOGGER = logging.getLogger(__name__)
DEFAULT_RADIUS_M = MAX_DISTANCE_CELLS * CELL_SIZE
MapSource = Literal["bbox", "legacy"]


def _radius_label(radius_m: float) -> str:
    return f"{radius_m:g}".replace(".", "p")


def map_cache_namespace(map_source: MapSource, radius_m: float) -> str:
    return f"{map_source}_r{_radius_label(radius_m)}"


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


def _cache_root(output_dir: Path, namespace: str) -> Path:
    return output_dir / namespace


def _cache_paths(
    output_dir: Path,
    namespace: str,
    scene_id: str,
    episode_key: str,
) -> tuple[Path, Path]:
    root = _cache_root(output_dir, namespace)
    return (
        root / "boxes" / scene_id / f"{episode_key}.npz",
        root / "raster" / scene_id / f"{episode_key}.npz",
    )


def _all_mentioned_level(level: LevelSemanticBoxes) -> LevelSemanticBoxes:
    return LevelSemanticBoxes(
        objects=[
            [replace(box, mentioned=True) for box in boxes]
            for boxes in level.objects
        ],
        regions=[
            [replace(box, mentioned=True) for box in boxes]
            for boxes in level.regions
        ],
        range_y=list(level.range_y),
    )


def _copy_legacy_path_neighborhood(
    source_grid: np.ndarray,
    cognitive_map: CognitiveGridMap,
    row: int,
    col: int,
    radius_cells: int,
    mentioned_objects: set[int],
    mentioned_regions: set[int],
) -> None:
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return

    up = max(0, row - radius_cells)
    left = max(0, col - radius_cells)
    down = min(ROWS - 1, row + radius_cells)
    right = min(COLS - 1, col + radius_cells)

    for layer_idx in range(OBJECT_CATEGORIES + REGION_CATEGORIES):
        source_slice = source_grid[layer_idx, up : down + 1, left : right + 1]
        if layer_idx < OBJECT_CATEGORIES:
            mentioned = layer_idx in mentioned_objects
        else:
            mentioned = (layer_idx - OBJECT_CATEGORIES) in mentioned_regions
        values = source_slice if mentioned else source_slice * 0.6
        target = cognitive_map.grid[layer_idx, up : down + 1, left : right + 1]
        np.maximum(target, values, out=target)


def _legacy_cognitive_map(
    scene_boxes: SceneSemanticBoxes,
    instruction: str,
    ground_truth_trajectory,
    start_direction_vector,
    radius_m: float,
) -> CognitiveGridMap:
    _, level, level_points = _first_usable_level_points(
        scene_boxes,
        ground_truth_trajectory,
    )
    full_grid = np.zeros(
        (OBJECT_CATEGORIES + REGION_CATEGORIES, ROWS, COLS),
        dtype=np.float32,
    )
    _rasterize_level_semantic_boxes(_all_mentioned_level(level), full_grid)

    cognitive_map = CognitiveGridMap()
    cognitive_map.range_y = list(level.range_y)
    cognitive_map.trajectory_keypoints = select_trajectory_keypoints(level_points)
    cognitive_map.start_direction_vector = start_direction_vector

    mentioned_objects, mentioned_regions = extract_categories(instruction)
    radius_cells = max(0, int(round(radius_m / CELL_SIZE)))
    for x, z in level_points:
        row = math.floor(x / CELL_SIZE)
        col = math.floor(z / CELL_SIZE)
        _copy_legacy_path_neighborhood(
            full_grid,
            cognitive_map,
            row,
            col,
            radius_cells,
            mentioned_objects,
            mentioned_regions,
        )
    return cognitive_map


def generate_cognitive_maps(
    entries: Iterable[Any],
    output_dir: Path = OUTPUT_DIR,
    map_source: MapSource = "bbox",
    radius_m: float = DEFAULT_RADIUS_M,
    namespace: Optional[str] = None,
) -> Tuple[int, int]:
    data = list(entries)
    cache_namespace = namespace or map_cache_namespace(map_source, radius_m)
    _cache_root(output_dir, cache_namespace).mkdir(parents=True, exist_ok=True)
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

        boxes_path, raster_path = _cache_paths(
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
            relevant_boxes = scene_boxes.relevant_to(
                entry.instruction,
                entry.ground_truth_trajectory,
                entry.start_direction_vector,
                max_distance=radius_m,
            )
            if map_source == "bbox":
                cognitive_map = relevant_boxes.to_cognitive_map()
            else:
                cognitive_map = _legacy_cognitive_map(
                    scene_boxes,
                    entry.instruction,
                    entry.ground_truth_trajectory,
                    entry.start_direction_vector,
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
