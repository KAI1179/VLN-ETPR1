"""Export cognitive grid maps to NumPy arrays."""

from typing import Optional

from prior import DATA_DIR
from prior.grid_map import CognitiveGridMap, GroundTruthGridMap
from prior.vlnce import DEFAULT_SPLITS, VLNCEEpisodeEntry

OUTPUT_DIR = DATA_DIR / "cognitive_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


data = [
    *VLNCEEpisodeEntry.iter_from("R2R", splits=DEFAULT_SPLITS),
    *VLNCEEpisodeEntry.iter_from("RxR", splits=DEFAULT_SPLITS),
]


for i, entry in enumerate(data):
    scene_id = entry.scene_id
    episode_id = entry.episode_id
    dataset = entry.dataset
    print(
        f"[{dataset}] Processing episode {i+1:0>5}/{len(data):0>5} (scene {scene_id})",
        end="\r",
    )

    # Create scene directory if it doesn't exist
    scene_output_dir = OUTPUT_DIR / scene_id
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    save_path = scene_output_dir / f"{dataset}_{episode_id}.npz"
    if save_path.exists():
        print(
            f"[{dataset}] Cognitive map for episode {episode_id} in scene {scene_id} already exists, skipping."
        )
        continue

    gt_grid_maps = GroundTruthGridMap.from_scene_id(scene_id)
    non_empty_map: Optional[CognitiveGridMap] = None

    for gt_grid_map in gt_grid_maps:
        cognitive_map = gt_grid_map.to_cognitive_map(
            entry.instruction,
            entry.positions,
            start_direction_vector=entry.start_direction_vector,
        )
        if not cognitive_map.is_empty():
            non_empty_map = cognitive_map
            break

    if non_empty_map is None:
        print(
            f"[{dataset}] No non-empty cognitive map found for episode {episode_id} in scene {scene_id}"
        )
    else:
        # Save the non-empty cognitive map as a NumPy array
        non_empty_map.save(save_path)
        print(
            f"[{dataset}] Saved cognitive map for episode {episode_id} in scene {scene_id}"
        )
