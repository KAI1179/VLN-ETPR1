#!/usr/bin/env python3
"""Build the two new D4 PriorGT val_seen probe namespaces (CPU only)."""

import json
from pathlib import Path
import sys
from typing import Literal

import numpy as np
from tap import Tap

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(ROOT))

from scripts.distill.make_raster_donor_cache import _load_episodes, _scene_of, assign_donors  # noqa: E402
from vlnce_baselines.models.etp_prior_gt.map_utils import cached_cognitive_map_to_tensors  # noqa: E402

BASE = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps")
GT = BASE / "gt.legacy.r1p5.direction5.v1"
LLM = ROOT / "data/llm_navigation/llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/r2r/val_seen/cognitive_maps/raster"
TARGETS = {
    "llm_as_priorgt": BASE / "llm.s2tagfree.as_priorgt.val_seen.v1",
    "gt_cross": BASE / "gt.legacy.r1p5.direction5.donor_cross_scene.val_seen.v1",
}
MISSING = {"11", "20", "99", "305", "365", "366", "419", "649", "736", "778"}


class Args(Tap):
    kind: Literal["llm_as_priorgt", "gt_cross"]


def paths(namespace: Path, episode):
    scene = _scene_of(episode)
    name = f"R2R_val_seen_{episode.episode_id}"
    return namespace / "raster" / scene / f"{name}.npz", scene, name


def save_status(target: Path, scene: str, name: str, payload: dict) -> None:
    status = target / "status" / scene / f"{name}.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({"status": "complete", **payload}, indent=2) + "\n")


def main() -> None:
    kind = Args().parse_args().kind
    target = TARGETS[kind]
    assert not target.exists(), target
    episodes = _load_episodes("val_seen")
    assert len(episodes) == 778, len(episodes)
    donors = assign_donors(episodes, "cross_scene", 0) if kind == "gt_cross" else None
    missing = set()

    for episode in episodes:
        own_gt, scene, name = paths(GT, episode)
        assert own_gt.is_file(), own_gt
        out, _, _ = paths(target, episode)
        out.parent.mkdir(parents=True, exist_ok=True)
        if kind == "llm_as_priorgt":
            source = LLM / scene / f"{name}.npz"
            if source.is_file():
                out.symlink_to(source)
            else:
                missing.add(str(episode.episode_id))
                with np.load(own_gt, allow_pickle=True) as file:
                    arrays = {key: file[key] for key in file.files}
                arrays["grid"] = np.zeros_like(arrays["grid"])
                np.savez_compressed(out, **arrays)
            save_status(target, scene, name, {"source": "llm" if source.is_file() else "zero_gt_metadata"})
        else:
            donor = donors[episode.episode_id]
            donor_gt, _, _ = paths(GT, donor)
            assert _scene_of(donor) != scene
            with np.load(own_gt, allow_pickle=True) as file, np.load(donor_gt, allow_pickle=True) as donor_file:
                arrays = {key: file[key] for key in file.files}
                arrays["grid"] = donor_file["grid"]
            np.savez_compressed(out, **arrays)
            save_status(target, scene, name, {"donor_episode_id": donor.episode_id,
                                              "donor_scene": _scene_of(donor)})

    if kind == "llm_as_priorgt":
        assert missing == MISSING, missing
    else:
        assert not missing
    files = list((target / "raster").glob("*/*.npz"))
    assert len(files) == 778, len(files)
    for episode in episodes[:20]:
        out, scene, name = paths(target, episode)
        own_gt, _, _ = paths(GT, episode)
        got = cached_cognitive_map_to_tensors(
            episode.scene_id, name, namespace=str(target), metadata_schema="direction5"
        )
        reference = cached_cognitive_map_to_tensors(
            episode.scene_id, name, namespace=str(GT), metadata_schema="direction5"
        )
        assert got.keys() == reference.keys()
        with np.load(out, allow_pickle=True) as file, np.load(own_gt, allow_pickle=True) as own:
            assert set(file.files) == set(own.files)
            assert file["grid"].shape == own["grid"].shape
            if kind == "gt_cross":
                for field in set(file.files) - {"grid"}:
                    assert np.array_equal(file[field], own[field]), (name, field)
    manifest = {"namespace": target.name, "split": "val_seen", "kind": kind,
                "episodes": len(files), "validated": 20, "missing_llm_ids": sorted(missing, key=int)}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
