#!/usr/bin/env python3
"""Create the matched DAgger control cache: zero predicted map fields."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(REPO))

from prior.vlnce import VLNCEEpisodeEntry  # noqa: E402
from vlnce_baselines.models.etp_llm.navigation import (  # noqa: E402
    llm_navigation_cognitive_map_raster_path,
    llm_navigation_status_path,
)

SOURCE_KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
TARGET_KEY = SOURCE_KEY + "__zero-all"


def main() -> None:
    target = REPO / "data/llm_navigation" / TARGET_KEY
    assert not target.exists(), target
    for split in ("train", "val_seen", "val_unseen"):
        count = 0
        for episode in VLNCEEpisodeEntry.iter_from("R2R", splits=(split,)):
            source = llm_navigation_cognitive_map_raster_path(
                episode.scene_id, episode.unique_id, "R2R", split,
                model_key=SOURCE_KEY,
            )
            if not source.is_file():
                continue  # Preserve the original available-episode set.
            with np.load(source, allow_pickle=True) as data:
                arrays = {key: data[key] for key in data.files}
            assert set(arrays) == {
                "grid", "range_y", "direction_vectors",
                "start_direction_vector", "start_position",
            }, source
            arrays["grid"] = np.zeros_like(arrays["grid"])
            arrays["direction_vectors"] = np.zeros_like(arrays["direction_vectors"])
            raster = llm_navigation_cognitive_map_raster_path(
                episode.scene_id, episode.unique_id, "R2R", split,
                model_key=TARGET_KEY,
            )
            raster.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raster, **arrays)
            status = llm_navigation_status_path(
                episode.scene_id, episode.unique_id, "R2R", split,
                model_key=TARGET_KEY,
            )
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text(json.dumps({
                "status": "complete", "dataset": "R2R", "split": split,
                "scene_id": episode.scene_id, "example_id": episode.unique_id,
                "cognitive_map_raster_path": str(raster),
                "probe_mode": "zero_all",
            }) + "\n", encoding="utf-8")
            count += 1
        print(f"{split}: {count} source-available maps copied with grid and directions zeroed")


if __name__ == "__main__":
    main()
