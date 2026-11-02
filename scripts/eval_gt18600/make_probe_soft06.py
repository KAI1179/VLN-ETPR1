#!/usr/bin/env python3
"""Convert D0 LLM-map positive grid values from 1.0 to 0.6."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(ROOT))

from scripts.distill.make_raster_donor_cache import _load_episodes, _scene_of
from vlnce_baselines.models.etp_prior_gt.map_utils import cached_cognitive_map_to_tensors

SOURCE = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/llm.s2tagfree.as_priorgt.v1")
TARGET = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/probe.llm_soft06.v1")


def main() -> None:
    episodes = _load_episodes("val_unseen")
    assert len(episodes) == 1839
    assert not TARGET.exists(), TARGET
    sources = []
    for ep in episodes:
        scene = _scene_of(ep)
        name = f"R2R_val_unseen_{ep.episode_id}.npz"
        path = SOURCE / "raster" / scene / name
        assert path.is_file(), path
        sources.append((ep, path, TARGET / "raster" / scene / name))

    for _, source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        with np.load(source, allow_pickle=True) as file:
            arrays = {key: file[key] for key in file.files}
        grid = arrays["grid"].copy()
        grid[grid == 1.0] = 0.6
        arrays["grid"] = grid
        np.savez_compressed(target, **arrays)

    values = set()
    for ep, _, _ in sources[:20]:
        name = f"R2R_val_unseen_{ep.episode_id}"
        tensors = cached_cognitive_map_to_tensors(
            ep.scene_id, name, namespace=str(TARGET), metadata_schema="direction5"
        )
        values.update(np.unique(tensors["grid"].numpy()).tolist())
    assert values == {0.0, float(np.float32(0.6))}, values
    print(f"namespace={TARGET} files={len(sources)} first20_grid_values={sorted(values)}")


if __name__ == "__main__":
    main()
