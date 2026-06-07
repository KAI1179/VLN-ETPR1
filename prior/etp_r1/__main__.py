"""Export cognitive grid maps for ETP-R1."""

from __future__ import annotations

import logging
import sys
from typing import Optional, Sequence, Tuple

from prior import DATA_DIR, VISUALIZATIONS_DIR
from prior.bbox import SceneSemanticBoxes
from prior.trajectory import InsufficientTrajectoryPointsError

from . import ANNOTATION_FILES, AnnotationEntry

OUTPUT_DIR = DATA_DIR / "cognitive_maps_etp_r1"
LOGGER = logging.getLogger(__name__)


def generate_cognitive_maps(
    annotation_file: str,
    sampled_instr_ids: Sequence[str],
) -> Tuple[int, int]:
    """Generate cognitive maps for one ETP-R1 annotation file."""
    generated = 0
    skipped = 0
    sampled_ids = set(sampled_instr_ids)

    for entry in AnnotationEntry.iter_from(annotation_file):
        instr_id = entry.instr_id
        if sampled_ids and instr_id not in sampled_ids:
            continue

        scene_id = entry.scan
        scene_path = OUTPUT_DIR / scene_id
        scene_path.mkdir(parents=True, exist_ok=True)
        save_path = scene_path / f"{instr_id}.npz"

        if save_path.exists() and instr_id not in sampled_ids:
            print(
                f"[{annotation_file}] Cognitive map for instruction ID "
                f"{instr_id} in scene {scene_id} already exists, skipping."
            )
            skipped += 1
            continue

        ground_truth_trajectory = entry.positions()
        try:
            relevant_boxes = SceneSemanticBoxes.from_scene_id(scene_id).relevant_to(
                entry.instruction,
                ground_truth_trajectory,
                entry.start_direction_vector,
            )
            cognitive_map = relevant_boxes.to_cognitive_map()
            cognitive_map.save(save_path)
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
    sampled_instr_ids = list(sys.argv[1:] if argv is None else argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0

    for annotation_file in ANNOTATION_FILES:
        file_generated, file_skipped = generate_cognitive_maps(
            annotation_file,
            sampled_instr_ids,
        )
        generated += file_generated
        skipped += file_skipped

    print(f"generated={generated} skipped={skipped}")


if __name__ == "__main__":
    main()
