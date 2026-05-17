from sys import argv
from typing import List

from prior import VISUALIZATIONS_DIR
from prior.vlnce import VLNCEEpisodeEntry

from . import CognitiveGridMap, GroundTruthGridMap


scene_id = argv[1] if len(argv) > 1 else "E9uDoFAP3SH"  # A large one: E9uDoFAP3SH
episode_id = argv[2] if len(argv) > 2 else 65  # An episode in that scene
cognitive_output_dir = VISUALIZATIONS_DIR / "cognitive_maps" / scene_id
cognitive_output_dir.mkdir(parents=True, exist_ok=True)

# Load R2R data

entry = next(
    (
        entry
        for entry in VLNCEEpisodeEntry.iter_from("R2R", splits=["train"])
        if entry.scene_id == scene_id and entry.episode_id == int(episode_id)
    ),
    None,
)
if entry is None:
    raise ValueError(
        f"Episode {episode_id} in scene {scene_id} not found in R2R train split"
    )

print(f"Instruction for episode {episode_id} in scene {scene_id}: {entry.instruction}")
print(f"Number of positions in the path: {len(entry.positions)}")

# Build ground truth grid maps

gt_grid_maps = GroundTruthGridMap.from_scene_id(scene_id)

for level, gt_grid_map in enumerate(gt_grid_maps):
    print(f"\nGround Truth Map - Level {level}")

    # Print text summary
    gt_grid_map.print_summary()

# Build cognitive grid maps

cognitive_maps: List[CognitiveGridMap] = []
for level, gt_grid_map in enumerate(gt_grid_maps):
    cognitive_map = gt_grid_map.to_cognitive_map(
        entry.instruction,
        entry.positions,
        start_direction_vector=entry.start_direction_vector,
    )
    cognitive_maps.append(cognitive_map)

    print(f"\nCognitive Map - Level {level} - Episode {episode_id}")

    # Print text summary
    cognitive_map.print_summary()

    # Save matplotlib visualization
    save_path = cognitive_output_dir / f"level_{level}.episode.{episode_id}.png"
    cognitive_map.visualize(
        title=f"Cognitive - Scene {scene_id} - Level {level}",
        save_path=save_path,
    )
    print(f"-> {save_path}")
