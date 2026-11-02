#!/usr/bin/env python3
"""Build and verify D2 raster-only cognitive-map probes (CPU only)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
from typing import Literal

import numpy as np
from tap import Tap

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
sys.path.insert(0, str(ROOT))

from prior import R2R_DIR  # noqa: E402
from scripts.distill.make_raster_donor_cache import _load_episodes, _scene_of  # noqa: E402
from vlnce_baselines.models.etp_llm.navigation import (  # noqa: E402
    llm_cached_cognitive_map_to_tensors,
    llm_navigation_cache_report,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import (  # noqa: E402
    cached_cognitive_map_to_tensors,
)

LLM_ROOT = ROOT / "data/llm_navigation"
LLM_KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
GT_ROOT = Path("/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps")
GT_KEY = "gt.legacy.r1p5.direction5.v1"
MODES = (
    "llm_zero_grid", "llm_gtlegacy_grid", "llm_mean_grid", "llm_region_only",
    "gt_region_only", "gt_object_only",
)
FIELDS = {"grid", "range_y", "direction_vectors", "start_direction_vector", "start_position"}


class Args(Tap):
    mode: Literal[
        "llm_zero_grid", "llm_gtlegacy_grid", "llm_mean_grid", "llm_region_only",
        "gt_region_only", "gt_object_only",
    ]


def read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as file:
        arrays = {key: file[key] for key in file.files}
    assert set(arrays) == FIELDS, (path, arrays.keys())
    assert arrays["grid"].shape == (37, 100, 100), (path, arrays["grid"].shape)
    return arrays


def train90_mean() -> tuple[np.ndarray, int]:
    dataset = R2R_DIR / "train" / "train_90.json.gz"
    with gzip.open(dataset, "rt", encoding="utf-8") as file:
        episodes = json.load(file)["episodes"]
    total = np.zeros((37, 100, 100), dtype=np.float64)
    count = 0
    for ep in episodes:
        scene = Path(ep["scene_id"]).stem
        name = f"R2R_train_{ep['episode_id']}.npz"
        path = LLM_ROOT / LLM_KEY / "r2r/train/cognitive_maps/raster" / scene / name
        if path.is_file():
            total += read(path)["grid"]
            count += 1
    assert count == 9639, count
    return (total / count).astype(np.float32), count


def main() -> None:
    args = Args().parse_args()
    mode = args.mode
    llm = mode.startswith("llm_")
    key = f"probe.{mode}.v1"
    source = LLM_ROOT / LLM_KEY / "r2r/val_unseen/cognitive_maps/raster" if llm else GT_ROOT / GT_KEY / "raster"
    target_root = LLM_ROOT / key / "r2r/val_unseen" if llm else GT_ROOT / key
    assert not target_root.exists(), target_root
    episodes = _load_episodes("val_unseen")
    assert len(episodes) == 1839
    selected = []
    for ep in episodes:
        scene = _scene_of(ep)
        name = f"R2R_val_unseen_{ep.episode_id}.npz"
        path = source / scene / name
        if path.is_file():
            selected.append((ep, scene, name, path))
        elif not llm:
            raise FileNotFoundError(path)
    assert len(selected) == (1833 if llm else 1839), len(selected)
    mean = None
    train_count = None
    if mode == "llm_mean_grid":
        mean, train_count = train90_mean()
    for ep, scene, name, path in selected:
        arrays = read(path)
        grid = arrays["grid"].copy()
        if mode == "llm_zero_grid":
            grid.fill(0)
        elif mode == "llm_gtlegacy_grid":
            grid = read(GT_ROOT / GT_KEY / "raster" / scene / name)["grid"].copy()
        elif mode == "llm_mean_grid":
            assert mean is not None
            grid = mean
        elif mode.endswith("region_only"):
            grid[:27] = 0
        elif mode == "gt_object_only":
            grid[27:] = 0
        arrays["grid"] = grid
        dest = target_root / "cognitive_maps/raster" / scene / name if llm else target_root / "raster" / scene / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, **arrays)
        if llm:
            status = target_root / "status" / scene / name.replace(".npz", ".json")
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text(json.dumps({
                "status": "complete", "dataset": "R2R", "split": "val_unseen",
                "scene_id": scene, "example_id": name[:-4],
                "cognitive_map_raster_path": str(dest), "probe_mode": mode,
            }, indent=2) + "\n")
    for ep, scene, name, path in selected[:20]:
        dest = target_root / "cognitive_maps/raster" / scene / name if llm else target_root / "raster" / scene / name
        old, new = read(path), read(dest)
        for field in FIELDS - {"grid"}:
            assert np.array_equal(old[field], new[field]), (name, field)
        expected = old["grid"].copy()
        if mode == "llm_zero_grid":
            expected.fill(0)
        elif mode == "llm_gtlegacy_grid":
            expected = read(GT_ROOT / GT_KEY / "raster" / scene / name)["grid"]
        elif mode == "llm_mean_grid":
            expected = mean
        elif mode.endswith("region_only"):
            expected[:27] = 0
        else:
            expected[27:] = 0
        assert np.array_equal(new["grid"], expected), name
        tensors = (llm_cached_cognitive_map_to_tensors(
            ep.scene_id, name[:-4], "R2R", "val_unseen", metadata_schema="direction5", model_key=key
        ) if llm else cached_cognitive_map_to_tensors(
            ep.scene_id, name[:-4], namespace=str(target_root), metadata_schema="direction5"
        ))
        assert np.array_equal(tensors["grid"].numpy(), expected), name
        nonzero = set(np.flatnonzero(np.any(expected != 0, axis=(1, 2))).tolist())
        allowed = set(range(27, 37)) if mode.endswith("region_only") else set(range(27)) if mode == "gt_object_only" else set(range(37))
        if mode == "llm_zero_grid":
            allowed = set()
        assert nonzero <= allowed, (name, nonzero - allowed)
    payload = {"key": key, "mode": mode, "episodes": len(selected), "validated": 20}
    if llm:
        report = llm_navigation_cache_report("R2R", "val_unseen", require_boxes=False, model_key=key)
        assert (report.available_count, report.missing_count) == (1833, 6)
        payload.update(available=report.available_count, missing=report.missing_count)
    if mean is not None:
        payload.update(train_90_count=train_count, mean_grid_max=float(mean.max()),
                       nonzero_channel_cell_fraction=float(np.count_nonzero(mean) / mean.size),
                       nonzero_spatial_cell_fraction=float(np.mean(np.any(mean != 0, axis=0))),
                       channel_means=mean.mean(axis=(1, 2)).astype(float).tolist())
    (target_root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
