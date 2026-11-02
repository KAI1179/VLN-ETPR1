#!/usr/bin/env python3
"""Build the D3 LLM region-only navigation cache on CPU."""

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(ROOT))

from prior import R2R_DIR
from vlnce_baselines.models.etp_llm.navigation import (
    llm_cached_cognitive_map_to_tensors,
    llm_navigation_cache_report,
)


CACHE_ROOT = ROOT / "data/llm_navigation"
SOURCE_KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
TARGET_KEY = "llm.region_only.s2tagfree.v1"
FIELDS = {"grid", "range_y", "direction_vectors", "start_direction_vector", "start_position"}


def episodes_for(split: str):
    dataset_split = "train_90" if split == "train" else split
    path = R2R_DIR / split / f"{dataset_split}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)["episodes"]


def main():
    target_root = CACHE_ROOT / TARGET_KEY
    assert not target_root.exists(), target_root
    manifest = {"model_key": TARGET_KEY, "source_model_key": SOURCE_KEY,
                "operation": "zero grid channels 0:27; preserve channels 27:37 and metadata",
                "splits": {}}
    for split in ("train", "val_seen", "val_unseen"):
        source = CACHE_ROOT / SOURCE_KEY / "r2r" / split / "cognitive_maps/raster"
        target = target_root / "r2r" / split
        selected = []
        missing = []
        for episode in episodes_for(split):
            scene = Path(episode["scene_id"]).stem
            episode_id = str(episode["episode_id"])
            name = f"R2R_{split}_{episode_id}"
            source_file = source / scene / f"{name}.npz"
            if source_file.is_file():
                selected.append((scene, name, source_file))
            else:
                missing.append(episode_id)
        for scene, name, source_file in selected:
            with np.load(source_file, allow_pickle=True) as old:
                arrays = {key: old[key] for key in old.files}
            assert arrays.keys() == FIELDS, source_file
            assert arrays["grid"].shape == (37, 100, 100), source_file
            arrays["grid"] = arrays["grid"].copy()
            arrays["grid"][:27] = 0
            dest = target / "cognitive_maps/raster" / scene / f"{name}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **arrays)
            status = target / "status" / scene / f"{name}.json"
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text(json.dumps({
                "status": "complete", "dataset": "R2R", "split": split,
                "scene_id": scene, "example_id": name,
                "cognitive_map_raster_path": str(dest),
            }, indent=2) + "\n")
        for scene, name, source_file in selected[:20]:
            dest = target / "cognitive_maps/raster" / scene / f"{name}.npz"
            with np.load(source_file, allow_pickle=True) as old, np.load(dest, allow_pickle=True) as new:
                assert old.files == new.files, name
                for field in FIELDS - {"grid"}:
                    assert np.array_equal(old[field], new[field]), (name, field)
                assert not np.any(new["grid"][:27]), name
                assert np.array_equal(old["grid"][27:], new["grid"][27:]), name
            tensors = llm_cached_cognitive_map_to_tensors(
                scene, name, "R2R", split, metadata_schema="direction5", model_key=TARGET_KEY
            )
            assert not np.any(tensors["grid"].numpy()[:27]), name
        report = llm_navigation_cache_report("R2R", split, require_boxes=False, model_key=TARGET_KEY)
        manifest["splits"][split] = {
            "selected_episodes": len(episodes_for(split)),
            "available": report.available_count,
            "missing": report.missing_count,
            "missing_ids": missing,
            "validated": min(20, len(selected)),
        }
        (target / "manifest.json").write_text(json.dumps({
            "cache_model_key": TARGET_KEY,
            "derived_from_model_key": SOURCE_KEY,
            "split": split,
            "operation": manifest["operation"],
            "available": report.available_count,
        }, indent=2) + "\n")
        assert report.available_count == len(selected), (split, report.available_count, len(selected))
        print(f"{split}: selected={len(episodes_for(split))} built={len(selected)} "
              f"cache_report available={report.available_count} missing={report.missing_count}", flush=True)
    manifest_path = target_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest={manifest_path} sha256={hashlib.sha256(manifest_path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
