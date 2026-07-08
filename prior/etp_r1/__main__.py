"""Export cognitive grid maps for ETP-R1."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

from prior import DATA_DIR, VISUALIZATIONS_DIR
from prior.bbox import SceneSemanticBoxes
from prior.cognitive_map_generation import (
    DEFAULT_RADIUS_M,
    MapSource,
    MetadataSchema,
    build_cognitive_map,
    cache_paths,
    cache_root,
    map_cache_namespace,
    save_cognitive_map,
)
from prior.trajectory import InsufficientTrajectoryPointsError

from . import ANNOTATION_FILES, AnnotationEntry

OUTPUT_DIR = DATA_DIR / "cognitive_maps_etp_r1"
LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ETP-R1 cognitive-map caches."
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
            "gt.<map-source>.r<radius>.<metadata-schema>.v1."
        ),
    )
    parser.add_argument(
        "--metadata-schema",
        choices=("path5", "direction5"),
        default="path5",
        help="Map metadata contract to write.",
    )
    parser.add_argument("sampled_instr_ids", nargs="*")
    return parser.parse_args(argv)


def generate_cognitive_maps(
    annotation_file: str,
    sampled_instr_ids: Sequence[str],
    output_dir: Path = OUTPUT_DIR,
    map_source: MapSource = "bbox",
    radius_m: float = DEFAULT_RADIUS_M,
    metadata_schema: MetadataSchema = "path5",
    namespace: Optional[str] = None,
) -> Tuple[int, int]:
    """Generate cognitive maps for one ETP-R1 annotation file."""
    if map_source != "legacy" and metadata_schema == "direction5":
        raise ValueError("direction5 metadata is only supported for legacy maps")
    cache_namespace = namespace or map_cache_namespace(
        map_source, radius_m, metadata_schema
    )
    cache_root(output_dir, cache_namespace).mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0
    sampled_ids = set(sampled_instr_ids)

    for entry in AnnotationEntry.iter_from(annotation_file):
        instr_id = entry.instr_id
        if sampled_ids and instr_id not in sampled_ids:
            continue

        scene_id = entry.scan
        boxes_path, raster_path = cache_paths(
            output_dir,
            cache_namespace,
            scene_id,
            instr_id,
        )

        if boxes_path.exists() and raster_path.exists() and instr_id not in sampled_ids:
            print(
                f"[{annotation_file}] Cognitive map for instruction ID "
                f"{instr_id} in scene {scene_id} already exists, skipping."
            )
            skipped += 1
            continue

        ground_truth_trajectory = entry.positions()
        try:
            scene_boxes = SceneSemanticBoxes.from_scene_id(scene_id)
            relevant_boxes, cognitive_map = build_cognitive_map(
                scene_boxes,
                entry.instruction,
                ground_truth_trajectory,
                entry.start_direction_vector,
                map_source,
                radius_m,
            )
            boxes_path.parent.mkdir(parents=True, exist_ok=True)
            raster_path.parent.mkdir(parents=True, exist_ok=True)
            relevant_boxes.save(boxes_path)
            save_cognitive_map(
                cognitive_map,
                raster_path,
                metadata_schema,
                ground_truth_trajectory,
            )
        except InsufficientTrajectoryPointsError as error:
            LOGGER.warning("skipping %s: %s", instr_id, error)
            skipped += 1
            continue

        generated += 1
        selected_level = relevant_boxes.level_idx
        print(
            f"[{annotation_file}] Saved cognitive map for instruction ID "
            f"{instr_id} in scene {scene_id}",
            end="\r",
        )

        if sampled_ids:
            print("Instruction:", entry.instruction)
            vis_path = VISUALIZATIONS_DIR / "cognitive_maps" / scene_id
            vis_path.mkdir(parents=True, exist_ok=True)
            save_path_png = (
                vis_path / f"level_{selected_level}.instruction.{instr_id}.png"
            )
            cognitive_map.visualize(
                title=f"Cognitive - Scene {scene_id} - Level {selected_level}",
                save_path=save_path_png,
            )
            print(f"-> {save_path_png}")

    return generated, skipped


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    namespace = args.namespace or map_cache_namespace(
        args.map_source, args.radius_m, args.metadata_schema
    )
    cache_root(args.output_dir, namespace).mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0

    for annotation_file in ANNOTATION_FILES:
        file_generated, file_skipped = generate_cognitive_maps(
            annotation_file,
            args.sampled_instr_ids,
            args.output_dir,
            map_source=args.map_source,
            radius_m=args.radius_m,
            metadata_schema=args.metadata_schema,
            namespace=namespace,
        )
        generated += file_generated
        skipped += file_skipped

    print(f"generated={generated} skipped={skipped}")
    if skipped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
