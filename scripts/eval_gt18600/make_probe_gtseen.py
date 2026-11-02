#!/usr/bin/env python3
"""Build non-causal final-GT-trajectory observation-mask oracle rasters."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(ROOT))

from scripts.distill.make_raster_donor_cache import _load_episodes, _scene_of
from vlnce_baselines.models.etp_prior_gt.map_utils import cached_cognitive_map_to_tensors

CACHE_ROOT = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps")
LLM = CACHE_ROOT / "llm.s2tagfree.as_priorgt.v1/raster"
GT = CACHE_ROOT / "gt.legacy.r1p5.direction5.v1/raster"
TRAJECTORIES = ROOT / "data/refiner/val_unseen"
TARGETS = {
    "llmunseen": CACHE_ROOT / "probe.gtseen_llmunseen.v1",
    "zero_unseen": CACHE_ROOT / "probe.gtseen_zero_unseen.v1",
}


def main() -> None:
    episodes = _load_episodes("val_unseen")
    assert len(episodes) == 1839
    assert all(not path.exists() for path in TARGETS.values()), TARGETS
    sources = []
    for ep in episodes:
        scene = _scene_of(ep)
        name = f"R2R_val_unseen_{ep.episode_id}.npz"
        llm = LLM / scene / name
        gt = GT / scene / name
        trajectory = TRAJECTORIES / scene / f"{ep.episode_id}_teacher.npz"
        assert llm.is_file() and gt.is_file(), ep.episode_id
        sources.append((ep, llm, gt, trajectory, scene, name))

    mask_missing = []
    observed_counts = []
    for ep, llm, gt, trajectory, scene, name in sources:
        with np.load(llm, allow_pickle=True) as file:
            arrays = {key: file[key] for key in file.files}
        with np.load(gt, allow_pickle=True) as file:
            gt_grid = file["grid"]
        if trajectory.is_file():
            with np.load(trajectory, allow_pickle=True) as file:
                mask = np.unpackbits(
                    file["steps"][-1]["observed"], count=10000
                ).reshape(100, 100).astype(bool)
            observed_counts.append(int(mask.sum()))
        else:
            mask = np.zeros((100, 100), dtype=bool)
            mask_missing.append(ep.episode_id)

        for mode, namespace in TARGETS.items():
            replacement = arrays["grid"] if mode == "llmunseen" else np.zeros_like(gt_grid)
            out = arrays.copy()
            out["grid"] = np.where(mask[None], gt_grid, replacement)
            target = namespace / "raster" / scene / name
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **out)

    assert len(observed_counts) == 1821 and len(mask_missing) == 18, mask_missing
    for namespace in TARGETS.values():
        for ep in episodes[:20]:
            cached_cognitive_map_to_tensors(
                ep.scene_id,
                f"R2R_val_unseen_{ep.episode_id}",
                namespace=str(namespace),
                metadata_schema="direction5",
            )
    print(f"mask_coverage={len(observed_counts)}/1839 mean_final_observed={np.mean(observed_counts):.2f}")
    print(f"mask_missing_ids={mask_missing}")
    for mode, namespace in TARGETS.items():
        count = len(list((namespace / "raster").glob("*/*.npz")))
        assert count == 1839
        print(f"mode={mode} namespace={namespace} files={count}")


if __name__ == "__main__":
    main()
