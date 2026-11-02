#!/usr/bin/env python3
"""CPU check for the D4 val_seen LLM cross-scene donor cache."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(ROOT))

from scripts.distill.make_raster_donor_cache import _load_episodes, _scene_of  # noqa: E402
from vlnce_baselines.models.etp_llm.navigation import (  # noqa: E402
    llm_cached_cognitive_map_to_tensors,
    llm_navigation_cache_report,
)

SOURCE = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
DONOR = SOURCE + "__raster-cross_scene-s0"


def main():
    report = llm_navigation_cache_report("R2R", "val_seen", require_boxes=False, model_key=DONOR)
    assert (report.available_count, report.missing_count) == (768, 10)
    checked = 0
    for episode in _load_episodes("val_seen"):
        if episode.episode_id in report.missing_episode_ids:
            continue
        scene = _scene_of(episode)
        name = f"R2R_val_seen_{episode.episode_id}.npz"
        own = ROOT / "data/llm_navigation" / SOURCE / "r2r/val_seen/cognitive_maps/raster" / scene / name
        moved = ROOT / "data/llm_navigation" / DONOR / "r2r/val_seen/cognitive_maps/raster" / scene / name
        with np.load(own, allow_pickle=True) as before, np.load(moved, allow_pickle=True) as after:
            for field in set(before.files) - {"grid"}:
                assert np.array_equal(before[field], after[field]), (name, field)
        llm_cached_cognitive_map_to_tensors(
            episode.scene_id, name[:-4], "R2R", "val_seen",
            metadata_schema="direction5", model_key=DONOR,
        )
        checked += 1
        if checked == 20:
            break
    assert checked == 20
    print(f"available={report.available_count} missing={report.missing_count} loader_checked={checked}")


if __name__ == "__main__":
    main()
