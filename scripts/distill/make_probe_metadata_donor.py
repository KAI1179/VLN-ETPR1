#!/usr/bin/env python3
"""Build LLM-map direction-metadata probes."""

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np
from tap import Tap

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
SOURCE_KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
ROOT_CACHE = ROOT / "data/llm_navigation"
SOURCE = ROOT_CACHE / SOURCE_KEY / "r2r/val_unseen/cognitive_maps/raster"
ASSIGNMENTS = ROOT_CACHE / f"{SOURCE_KEY}__raster-cross_scene-s0/r2r/val_unseen/donor_assignment.csv"
GT_SOURCE = ROOT / "data/cognitive_maps/gt.legacy.r1p5.direction5.v1/raster"
MODES = {
    "metadata_only": ("direction_vectors", "start_direction_vector"),
    "full": ("grid", "range_y", "direction_vectors", "start_direction_vector", "start_position"),
    "gt_metadata": ("direction_vectors", "start_direction_vector"),
}


class Args(Tap):
    mode: Literal["metadata_only", "full", "gt_metadata"] = "gt_metadata"
    status_only: bool = False


def main() -> None:
    args = Args().parse_args()
    mode = args.mode
    rows = list(csv.DictReader(ASSIGNMENTS.open()))
    assert len(rows) == 1833 and len({row["episode_id"] for row in rows}) == 1833
    assert all(row["same_scene"] == "0" for row in rows)
    model_key = "probe.llm_gtmeta.v1" if mode == "gt_metadata" else f"probe.llm_{mode}_s0.v1"
    target_dir = ROOT_CACHE / model_key / "r2r/val_unseen"
    assert target_dir.exists() == args.status_only, target_dir

    for row in rows:
        own_name = f"R2R_val_unseen_{row['episode_id']}.npz"
        own_path = SOURCE / row["scene"] / own_name
        donor_path = (
            GT_SOURCE / row["scene"] / own_name
            if mode == "gt_metadata"
            else SOURCE / row["donor_scene"] / f"R2R_val_unseen_{row['donor_episode_id']}.npz"
        )
        with np.load(own_path, allow_pickle=True) as own_file:
            own = {key: own_file[key] for key in own_file.files}
        with np.load(donor_path, allow_pickle=True) as donor_file:
            donor = {key: donor_file[key] for key in donor_file.files}
        assert own.keys() == donor.keys() == set(MODES["full"])
        arrays = own.copy()
        arrays.update({field: donor[field] for field in MODES[mode]})
        target = target_dir / "cognitive_maps/raster" / row["scene"] / own_name
        if args.status_only:
            assert target.is_file(), target
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **arrays)
        status = target_dir / "status" / row["scene"] / f"R2R_val_unseen_{row['episode_id']}.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({
            "status": "complete",
            "dataset": "R2R",
            "split": "val_unseen",
            "scene_id": row["scene"],
            "example_id": f"R2R_val_unseen_{row['episode_id']}",
            "cognitive_map_raster_path": str(target),
            "probe_mode": mode,
        }, indent=2) + "\n")

    if mode != "gt_metadata" and not args.status_only:
        with ASSIGNMENTS.open() as source:
            (target_dir / "donor_assignment.csv").write_text(source.read())
    if not args.status_only:
        (target_dir / "manifest.json").write_text(json.dumps({
            "cache_model_key": model_key,
            "derived_from_model_key": SOURCE_KEY,
            "probe_mode": mode,
            "metadata_source": "gt.legacy.r1p5.direction5.v1" if mode == "gt_metadata" else "cross_scene_s0",
            "split": "val_unseen",
            "episodes": 1833,
            "replaced_fields": list(MODES[mode]),
        }, indent=2) + "\n")
    files = list((target_dir / "cognitive_maps/raster").glob("*/*.npz"))
    assert len(files) == 1833
    for row in rows[:20]:
        own_name = f"R2R_val_unseen_{row['episode_id']}.npz"
        donor_path = (
            GT_SOURCE / row["scene"] / own_name
            if mode == "gt_metadata"
            else SOURCE / row["donor_scene"] / f"R2R_val_unseen_{row['donor_episode_id']}.npz"
        )
        with np.load(SOURCE / row["scene"] / own_name, allow_pickle=True) as own, \
                np.load(donor_path, allow_pickle=True) as donor, \
                np.load(target_dir / "cognitive_maps/raster" / row["scene"] / own_name, allow_pickle=True) as got:
            for field in MODES["full"]:
                expected = donor[field] if field in MODES[mode] else own[field]
                assert np.array_equal(got[field], expected), (mode, row["episode_id"], field)
    print(f"model_key={model_key} files={len(files)} fields={MODES[mode]}")


if __name__ == "__main__":
    main()
