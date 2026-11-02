#!/usr/bin/env python3
"""Expose val_unseen LLM rasters through a new PriorGT namespace."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(ROOT))

from scripts.distill.make_raster_donor_cache import _load_episodes, _scene_of
from vlnce_baselines.models.etp_prior_gt.map_utils import (
    cached_cognitive_map_to_tensors,
)

LLM = ROOT / "data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_unseen/cognitive_maps/raster"
GT = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster")
NAMESPACE = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/llm.s2tagfree.as_priorgt.v1")
MISSING = {"308", "349", "455", "804", "1318", "1331"}


def main() -> None:
    episodes = _load_episodes("val_unseen")
    assert len(episodes) == 1839
    sources = []
    missing = set()
    for episode in episodes:
        name = f"R2R_val_unseen_{episode.episode_id}.npz"
        scene = _scene_of(episode)
        source = LLM / scene / name
        gt = GT / scene / name
        assert gt.is_file(), gt
        if not source.is_file():
            missing.add(episode.episode_id)
        sources.append((episode, source, gt, NAMESPACE / "raster" / scene / name))
    assert missing == MISSING, missing
    assert not NAMESPACE.exists(), NAMESPACE

    for episode, source, gt, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            target.symlink_to(source)
        else:
            with np.load(gt, allow_pickle=True) as gt_map:
                arrays = {key: gt_map[key] for key in gt_map.files}
            arrays["grid"] = np.zeros_like(arrays["grid"])
            np.savez_compressed(target, **arrays)
            print(f"zero-grid placeholder: episode {episode.episode_id}")

    values = set()
    for episode, _, gt, target in sources[:20]:
        scene = episode.scene_id
        cache_id = f"R2R_val_unseen_{episode.episode_id}"
        got = cached_cognitive_map_to_tensors(
            scene, cache_id, namespace=str(NAMESPACE), metadata_schema="direction5"
        )
        reference = cached_cognitive_map_to_tensors(
            scene, cache_id, namespace=str(GT.parent), metadata_schema="direction5"
        )
        assert got.keys() == reference.keys()
        assert all(got[key].shape == reference[key].shape and got[key].dtype == reference[key].dtype for key in got)
        with np.load(target, allow_pickle=True) as source_map:
            with np.load(gt, allow_pickle=True) as gt_map:
                assert set(source_map.files) == set(gt_map.files)
                assert all(source_map[key].shape == gt_map[key].shape and source_map[key].dtype == gt_map[key].dtype for key in gt_map.files)
                values.update(np.unique(source_map["grid"]).tolist())
        print(f"checked: episode {episode.episode_id}")
    assert values == {0.0, 1.0}, values
    assert sum(1 for _, _, _, target in sources if target.is_file()) == 1839
    print(f"namespace={NAMESPACE} files=1839 links=1833 placeholders=6 sampled_grid_values={sorted(values)}")


if __name__ == "__main__":
    main()
