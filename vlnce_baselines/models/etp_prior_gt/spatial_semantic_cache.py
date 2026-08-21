"""Generate v2 spatial semantic supervision for the route-gated map updater."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from prior import MP3D_DIR

from .semantic_panorama import (
    build_panorama_semantic_simulator,
    panorama_spatial_semantic_label,
)


def _agent_pose(pose):
    return np.asarray((pose[3], pose[7], pose[11]), dtype=np.float32), np.asarray(
        (0.0, 0.0, 0.0, 1.0), dtype=np.float32
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--connectivity-dir", type=Path, default=Path("pretrain_src/datasets/R2R/connectivity"))
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--scans", nargs="*", default=())
    args = parser.parse_args()
    paths = (
        [args.connectivity_dir / f"{scan}_connectivity.json" for scan in args.scans]
        if args.scans
        else sorted(args.connectivity_dir.glob("*_connectivity.json"))
    )
    with h5py.File(args.output_file, "x") as output:
        output.attrs["schema"] = "pose-gated-spatial-semantics-v2"
        semantic_group = output.create_group("semantic")
        coverage_group = output.create_group("coverage")
        for connectivity_path in paths:
            scan = connectivity_path.name[: -len("_connectivity.json")]
            with connectivity_path.open() as source:
                viewpoints = json.load(source)
            simulator = build_panorama_semantic_simulator(
                MP3D_DIR / scan / f"{scan}.glb", args.gpu_device_id
            )
            try:
                simulator.reset()
                semantic_scene = simulator.semantic_annotations()
                for viewpoint in viewpoints:
                    if not viewpoint["included"]:
                        continue
                    position, rotation = _agent_pose(viewpoint["pose"])
                    observations = simulator.get_observations_at(
                        position=position, rotation=rotation, keep_agent_at_new_pose=True
                    )
                    labels, coverage = panorama_spatial_semantic_label(
                        observations, semantic_scene
                    )
                    key = f"{scan}_{viewpoint['image_id']}"
                    semantic_group.create_dataset(key, data=labels, compression="gzip")
                    coverage_group.create_dataset(key, data=coverage, compression="gzip")
            finally:
                simulator.close()


if __name__ == "__main__":
    main()
