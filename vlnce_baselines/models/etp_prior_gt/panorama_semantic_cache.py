"""Generate training-only 37-way semantic labels for MP3D viewpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
from prior import MP3D_DIR
from prior.constants import MAPPED_OBJECT_NAMES, MAPPED_REGION_NAMES
from vlnce_baselines.models.etp_prior_gt.semantic_panorama import (
    build_panorama_semantic_simulator,
    panorama_semantic_label,
)

@dataclass(frozen=True)
class PanoramaSemanticCacheArgs:
    output_file: Path
    connectivity_dir: Path
    gpu_device_id: int
    scans: Tuple[str, ...]
    limit_viewpoints: Optional[int]

    def validate(self) -> None:
        if self.gpu_device_id < 0:
            raise ValueError("gpu_device_id must be non-negative")
        if self.limit_viewpoints is not None and self.limit_viewpoints <= 0:
            raise ValueError("limit_viewpoints must be positive")
        if not self.connectivity_dir.is_dir():
            raise FileNotFoundError(self.connectivity_dir)
        if self.output_file.exists():
            raise FileExistsError(self.output_file)


def _agent_pose_from_connectivity(
    pose: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pose) < 12:
        raise ValueError(f"Connectivity pose must contain at least 12 values: {pose}")
    position = np.asarray((pose[3], pose[7], pose[11]), dtype=np.float32)
    if not np.isfinite(position).all():
        raise ValueError(f"Connectivity pose has a non-finite position: {pose}")
    rotation = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    return position, rotation


def _parse_args(argv: Optional[Sequence[str]]) -> PanoramaSemanticCacheArgs:
    parser = argparse.ArgumentParser(
        description="Generate OnlineFusion panorama semantic supervision"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/online_fusion/panorama_semantics_v1.hdf5"),
    )
    parser.add_argument(
        "--connectivity-dir",
        type=Path,
        default=Path("pretrain_src/datasets/R2R/connectivity"),
    )
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--scans", nargs="*", default=())
    parser.add_argument("--limit-viewpoints", type=int, default=None)
    namespace = parser.parse_args(list(argv) if argv is not None else None)
    args = PanoramaSemanticCacheArgs(
        output_file=namespace.output_file,
        connectivity_dir=namespace.connectivity_dir,
        gpu_device_id=namespace.gpu_device_id,
        scans=tuple(namespace.scans),
        limit_viewpoints=namespace.limit_viewpoints,
    )
    args.validate()
    return args


def _connectivity_files(args: PanoramaSemanticCacheArgs) -> Tuple[Path, ...]:
    if args.scans:
        paths = tuple(
            args.connectivity_dir / f"{scan}_connectivity.json"
            for scan in args.scans
        )
    else:
        paths = tuple(sorted(args.connectivity_dir.glob("*_connectivity.json")))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing connectivity files: {missing}")
    if not paths:
        raise FileNotFoundError(
            f"No connectivity files found under {args.connectivity_dir}"
        )
    return paths


def generate_panorama_semantic_cache(args: PanoramaSemanticCacheArgs) -> int:
    paths = _connectivity_files(args)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_file.with_suffix(args.output_file.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Stale temporary cache exists; inspect and remove it: {temporary}"
        )

    written = 0
    try:
        with h5py.File(temporary, "x") as output:
            output.attrs["schema"] = "online-fusion-panorama-semantics-v1"
            output.attrs["object_names"] = json.dumps(list(MAPPED_OBJECT_NAMES))
            output.attrs["region_names"] = json.dumps(list(MAPPED_REGION_NAMES))
            for connectivity_path in paths:
                scan = connectivity_path.name[: -len("_connectivity.json")]
                scene_path = MP3D_DIR / scan / f"{scan}.glb"
                if not scene_path.is_file():
                    raise FileNotFoundError(f"Missing MP3D scene: {scene_path}")
                with connectivity_path.open() as source:
                    viewpoints = json.load(source)
                simulator = build_panorama_semantic_simulator(
                    scene_path, args.gpu_device_id
                )
                try:
                    simulator.reset()
                    semantic_scene = simulator.semantic_annotations()
                    for viewpoint in viewpoints:
                        if not viewpoint.get("included", False):
                            continue
                        pose = viewpoint["pose"]
                        position, rotation = _agent_pose_from_connectivity(pose)
                        observations = simulator.get_observations_at(
                            position=position,
                            rotation=rotation,
                            keep_agent_at_new_pose=True,
                        )
                        if observations is None:
                            raise RuntimeError(
                                f"Failed to render {scan}/{viewpoint['image_id']}"
                            )
                        label = panorama_semantic_label(
                            observations,
                            semantic_scene,
                        )
                        key = f"{scan}_{viewpoint['image_id']}"
                        output.create_dataset(
                            key,
                            data=label,
                            compression="gzip",
                        )
                        written += 1
                        if (
                            args.limit_viewpoints is not None
                            and written >= args.limit_viewpoints
                        ):
                            break
                finally:
                    simulator.close()
                if (
                    args.limit_viewpoints is not None
                    and written >= args.limit_viewpoints
                ):
                    break
        temporary.replace(args.output_file)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    count = generate_panorama_semantic_cache(args)
    print(f"Wrote {count} viewpoint labels to {args.output_file}")
    return count


if __name__ == "__main__":
    main()
