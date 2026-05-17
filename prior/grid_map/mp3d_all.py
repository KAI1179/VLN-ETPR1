from . import GroundTruthGridMap
from prior import MP3D_DIR, DATA_DIR

OUTPUT_DIR = DATA_DIR / "semantic_maps"

SCENE_IDS = map(lambda p: p.name, filter(lambda p: p.is_dir(), MP3D_DIR.iterdir()))

for scene_id in SCENE_IDS:
    save_path = OUTPUT_DIR / scene_id
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Processing scene {scene_id}...")

    maps = GroundTruthGridMap.from_scene_id(scene_id)

    for lvl, semantic_map in enumerate(maps):
        semantic_map.save(save_path / f"{lvl}.npz")
