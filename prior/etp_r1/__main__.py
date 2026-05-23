"""Export cognitive grid maps for ETP-R1."""

from __future__ import annotations

from sys import argv
from prior import DATA_DIR, VISUALIZATIONS_DIR
from prior.bbox import SceneSemanticBoxes

from . import (
    ANNOTATION_FILES,
    AnnotationEntry,
)

OUTPUT_DIR = DATA_DIR / "cognitive_maps_etp_r1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLED_INSTR_IDS = argv[1:]


def generate_cognitive_map(annotation_file: str):
    """Generate cognitive map for given annotation file."""
    for entry in AnnotationEntry.iter_from(annotation_file):
        instr_id = entry.instr_id
        if len(SAMPLED_INSTR_IDS) > 0 and instr_id not in SAMPLED_INSTR_IDS:
            continue

        scene_id = entry.scan
        scene_path = OUTPUT_DIR / scene_id
        scene_path.mkdir(exist_ok=True)
        save_path = scene_path / f"{instr_id}.npz"

        if save_path.exists() and instr_id not in SAMPLED_INSTR_IDS:
            print(
                f"[{annotation_file}] Cognitive map for instruction ID {instr_id} in scene {scene_id} already exists, skipping."
            )
            continue

        positions = entry.positions()
        relevant_boxes = SceneSemanticBoxes.from_scene_id(scene_id).relevant_to(
            entry.instruction,
            positions,
            entry.start_direction_vector,
        )
        selected_level = relevant_boxes.level_idx
        cognitive_map = relevant_boxes.to_cognitive_map()
        # Save the non-empty cognitive map as a NumPy array
        cognitive_map.save(save_path)
        print(
            f"[{annotation_file}] Saved cognitive map for instruction ID {instr_id} in scene {scene_id}",
            end="\r",
        )

        # Sampled map
        if len(SAMPLED_INSTR_IDS) > 0:
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


def main():
    for annotation_file in ANNOTATION_FILES:
        generate_cognitive_map(annotation_file)


if __name__ == "__main__":
    main()
