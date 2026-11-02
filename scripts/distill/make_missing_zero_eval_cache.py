"""Keep available LLM maps; supply zero map inputs for missing eval episodes."""

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(REPO))

from prior.vlnce import VLNCEEpisodeEntry
from vlnce_baselines.models.etp_llm.navigation import (
    llm_cached_cognitive_map_to_tensors,
    llm_navigation_cache_report,
    llm_navigation_cognitive_map_raster_path,
    llm_navigation_status_path,
)

SOURCE = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
TARGET = SOURCE + "__missing-all-zero"
CACHE = REPO / "outputs/kd27000_missing_zero_cache"


def main():
    records = {}
    for split in ("val_seen", "val_unseen"):
        episodes = list(VLNCEEpisodeEntry.iter_from("R2R", splits=(split,)))
        ids = [str(ep.episode_id) for ep in episodes]
        source_report = llm_navigation_cache_report(
            "R2R", split, ids, require_boxes=False, model_key=SOURCE
        )
        missing = set(source_report.missing_episode_ids)
        for ep in episodes:
            path = llm_navigation_cognitive_map_raster_path(
                ep.scene_id, ep.unique_id, "R2R", split, cache_dir=CACHE, model_key=TARGET
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            if str(ep.episode_id) in missing:
                if not path.exists():
                    np.savez_compressed(
                        path,
                        grid=np.zeros((37, 100, 100), dtype=np.float32),
                        direction_vectors=np.zeros((5, 2), dtype=np.float32),
                        start_direction_vector=np.zeros(2, dtype=np.float32),
                        start_position=np.zeros(2, dtype=np.float32),
                        range_y=np.asarray([None, None], dtype=object),
                    )
                tensors = llm_cached_cognitive_map_to_tensors(
                    ep.scene_id, ep.unique_id, "R2R", split,
                    metadata_schema="direction5", cache_dir=CACHE, model_key=TARGET,
                )
                for name in ("grid", "map_trajectory_metadata", "start_direction_vector", "start_position"):
                    assert not tensors[name].any(), (ep.unique_id, name)
            else:
                source = llm_navigation_cognitive_map_raster_path(
                    ep.scene_id, ep.unique_id, "R2R", split, model_key=SOURCE
                ).resolve(strict=True)
                if not path.is_symlink():
                    path.symlink_to(source)
                assert path.resolve() == source
            status = llm_navigation_status_path(
                ep.scene_id, ep.unique_id, "R2R", split, cache_dir=CACHE, model_key=TARGET
            )
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text(json.dumps({
                "status": "complete", "source_model_key": SOURCE,
                "zero_filled": str(ep.episode_id) in missing,
            }) + "\n")
        report = llm_navigation_cache_report(
            "R2R", split, ids, require_boxes=False, cache_dir=CACHE, model_key=TARGET
        )
        assert report.available_count == len(episodes) and report.missing_count == 0
        records[split] = {"episodes": len(episodes), "zero_filled": sorted(missing, key=int)}
    (CACHE / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
