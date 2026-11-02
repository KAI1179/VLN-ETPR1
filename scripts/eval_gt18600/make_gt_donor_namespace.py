#!/usr/bin/env python3
"""Cross-scene GT raster donor: same assignment and grid-only swap as E2."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(ROOT))

from scripts.distill.make_raster_donor_cache import (  # noqa: E402
    _load_episodes,
    _scene_of,
    assign_donors,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import (  # noqa: E402
    cached_cognitive_map_to_tensors,
)

BASE = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps")
SOURCE = BASE / "gt.legacy.r1p5.direction5.v1"
TARGET = BASE / "gt.legacy.r1p5.direction5.donor_cross_scene.v1"
FIELDS = ("grid", "range_y", "direction_vectors", "start_direction_vector", "start_position")


def raster_path(namespace, episode):
    return namespace / "raster" / _scene_of(episode) / f"R2R_val_unseen_{episode.episode_id}.npz"


def main():
    episodes = _load_episodes("val_unseen")
    assert len(episodes) == 1839, len(episodes)
    donors = assign_donors(episodes, "cross_scene", 0)
    if TARGET.exists():
        raise FileExistsError(f"Refusing to modify existing namespace: {TARGET}")
    paths = [raster_path(SOURCE, episode) for episode in episodes]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} source maps; first: {missing[0]}")

    for episode in episodes:
        donor = donors[episode.episode_id]
        assert _scene_of(episode) != _scene_of(donor)
        own_path = raster_path(SOURCE, episode)
        donor_path = raster_path(SOURCE, donor)
        out_path = raster_path(TARGET, episode)
        with np.load(own_path, allow_pickle=True) as own, np.load(
            donor_path, allow_pickle=True
        ) as donor_map:
            arrays = {key: own[key] for key in own.files}
            assert set(FIELDS) <= set(arrays), own_path
            assert donor_map["grid"].shape == arrays["grid"].shape
            assert donor_map["grid"].dtype == arrays["grid"].dtype
            arrays["grid"] = donor_map["grid"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **arrays)

    output_files = list((TARGET / "raster").glob("*/*.npz"))
    assert len(output_files) == 1839, len(output_files)
    first = episodes[0]
    cached_cognitive_map_to_tensors(
        first.scene_id,
        f"R2R_val_unseen_{first.episode_id}",
        namespace=str(TARGET),
        metadata_schema="direction5",
    )
    print(f"namespace={TARGET} files={len(output_files)}")
    print("episode_id | donor_episode_id | donor_scene!=own_scene | donor fields")
    for episode in episodes[:20]:
        donor = donors[episode.episode_id]
        print(f"{episode.episode_id} | {donor.episode_id} | True | grid")


if __name__ == "__main__":
    main()
