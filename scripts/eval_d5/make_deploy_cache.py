#!/usr/bin/env python3
"""Build the D5 no-LLM-output val_unseen cache and audit prompt metadata."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(REPO))

from prior.bbox import SceneSemanticBoxes  # noqa: E402
from prior.vlnce import VLNCEEpisodeEntry  # noqa: E402
from vlnce_baselines.models.etp_llm.navigation import (  # noqa: E402
    llm_cached_cognitive_map_to_tensors,
    llm_navigation_cognitive_map_raster_path,
    llm_navigation_status_path,
)

SOURCE_KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
TARGET_KEY = SOURCE_KEY + "__deploy-zero"
FIELDS = {"grid", "range_y", "direction_vectors", "start_direction_vector", "start_position"}


def main() -> None:
    items = list(VLNCEEpisodeEntry.iter_from("R2R", splits=("val_unseen",)))
    assert len(items) == 1839, len(items)
    records = []
    mismatched_direction = []
    mismatched_position = []
    missing_source = []
    for item in items:
        scene = item.scene_id
        cache_id = item.unique_id
        source = llm_navigation_cognitive_map_raster_path(
            scene, cache_id, "R2R", "val_unseen", model_key=SOURCE_KEY
        )
        relevant = SceneSemanticBoxes.from_scene_id(scene).relevant_to(
            item.instruction,
            item.ground_truth_trajectory,
            item.start_direction_vector,
            category_extractor=lambda _instruction: (set(), set()),
        )
        prompt_direction = np.asarray(item.start_direction_vector, dtype=np.float32)
        prompt_position = np.asarray(relevant.trajectory_keypoints[0], dtype=np.float32)
        if source.is_file():
            with np.load(source, allow_pickle=True) as data:
                arrays = {name: data[name] for name in data.files}
            assert set(arrays) == FIELDS, (source, set(arrays))
            if not np.array_equal(arrays["start_direction_vector"], prompt_direction):
                mismatched_direction.append(cache_id)
            if not np.array_equal(arrays["start_position"], prompt_position):
                mismatched_position.append(cache_id)
            range_y = arrays["range_y"]
        else:
            missing_source.append(cache_id)
            range_y = np.asarray([None, None], dtype=object)
        records.append((scene, cache_id, prompt_direction, prompt_position, range_y))

    print(json.dumps({
        "episodes": len(records),
        "source_missing": missing_source,
        "direction_mismatch": mismatched_direction,
        "position_mismatch": mismatched_position,
    }, indent=2))
    assert not mismatched_direction and not mismatched_position
    assert len(missing_source) == 6, missing_source

    target_dir = REPO / "data/llm_navigation" / TARGET_KEY / "r2r/val_unseen"
    assert not target_dir.exists(), target_dir
    for scene, cache_id, prompt_direction, prompt_position, range_y in records:
        raster = llm_navigation_cognitive_map_raster_path(
            scene, cache_id, "R2R", "val_unseen", model_key=TARGET_KEY
        )
        raster.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raster,
            grid=np.zeros((37, 100, 100), dtype=np.float32),
            range_y=range_y,
            direction_vectors=np.zeros((5, 2), dtype=np.float32),
            start_direction_vector=prompt_direction,
            start_position=prompt_position,
        )
        status = llm_navigation_status_path(
            scene, cache_id, "R2R", "val_unseen", model_key=TARGET_KEY
        )
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({
            "status": "complete", "dataset": "R2R", "split": "val_unseen",
            "scene_id": scene, "example_id": cache_id,
            "cognitive_map_raster_path": str(raster), "probe_mode": "deploy_zero",
        }) + "\n", encoding="utf-8")

    for scene, cache_id, _, _, _ in records[:20]:
        tensors = llm_cached_cognitive_map_to_tensors(
            scene, cache_id, "R2R", "val_unseen",
            metadata_schema="direction5", model_key=TARGET_KEY,
        )
        assert not tensors["grid"].any()
        assert not tensors["map_trajectory_metadata"].any()
    print(f"wrote {len(records)} deploy-zero caches under {target_dir}")


if __name__ == "__main__":
    main()
