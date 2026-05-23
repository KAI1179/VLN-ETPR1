"""Export cognitive grid maps to NumPy arrays."""

from prior import DATA_DIR
from prior.bbox import SceneSemanticBoxes
from prior.vlnce import DEFAULT_SPLITS, VLNCEEpisodeEntry

OUTPUT_DIR = DATA_DIR / "cognitive_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


data = [
    *VLNCEEpisodeEntry.iter_from("R2R", splits=DEFAULT_SPLITS),
    *VLNCEEpisodeEntry.iter_from("RxR", splits=DEFAULT_SPLITS),
]


for i, entry in enumerate(data):
    scene_id = entry.scene_id
    dataset = entry.dataset
    episode_key = entry.unique_id
    print(
        f"[{dataset}] Processing episode {i + 1:0>5}/{len(data):0>5} ({episode_key}, scene {scene_id})",
        end="\r",
    )

    # Create scene directory if it doesn't exist
    scene_output_dir = OUTPUT_DIR / scene_id
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    save_path = scene_output_dir / f"{episode_key}.npz"
    if save_path.exists():
        print(
            f"[{dataset}] Cognitive map for episode {episode_key} in scene {scene_id} already exists, skipping."
        )
        continue

    cognitive_map = SceneSemanticBoxes.from_scene_id(scene_id).to_cognitive_map(
        entry.instruction,
        entry.reference_path,
        start_direction_vector=entry.start_direction_vector,
    )

    cognitive_map.save(save_path)
    print(
        f"[{dataset}] Saved cognitive map for episode {episode_key} in scene {scene_id}"
    )
