#!/usr/bin/env python3
"""Compare 121c369 online cognitive-map tensors with two raster caches."""

import json
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_DIR = REPO_ROOT / "reports/online_maps_121c369"
OUTPUT_PATH = REPO_ROOT / "reports/online_vs_cache_121c369.json"
CACHE_ROOTS = {
    "direction5": Path(
        "/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/"
        "gt.legacy.r1p5.direction5.v1/raster"
    ),
    "blurred": Path(
        "/data/xukai/etp-r1-snapshot/runtime-data/data/cognitive_maps/"
        "gt.legacy.r1p5.direction5.blurred.v1/raster"
    ),
}


def numeric_range(array):
    if array.size <= 64:
        return {"unique": np.unique(array).tolist()}
    return {"min": float(np.min(array)), "max": float(np.max(array))}


def exact(left, right) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)


def grid_transforms(grid):
    return {
        "transpose": grid.swapaxes(-2, -1),
        "flip_horizontal": np.flip(grid, axis=-1),
        "flip_vertical": np.flip(grid, axis=-2),
        "rot90_k1": np.rot90(grid, 1, axes=(-2, -1)),
        "rot90_k2": np.rot90(grid, 2, axes=(-2, -1)),
        "rot90_k3": np.rot90(grid, 3, axes=(-2, -1)),
        "transpose_plus_flip_horizontal": np.flip(grid.swapaxes(-2, -1), axis=-1),
    }


def grid_iou(left, right):
    if left.shape != right.shape:
        return None
    left_binary, right_binary = left > 0, right > 0
    intersections = np.logical_and(left_binary, right_binary).sum(axis=(-2, -1))
    unions = np.logical_or(left_binary, right_binary).sum(axis=(-2, -1))
    return [float(i / u) if u else 1.0 for i, u in zip(intersections, unions)]


def vector_relations(left, right):
    if left.shape != right.shape:
        return {
            "sorted_equal": False,
            "axis_swap_equal": False,
            "negated_equal": False,
            "axis_swap_and_negated_equal": False,
            "negate_x_equal": False,
            "negate_y_equal": False,
        }
    sorted_left = np.array(sorted(map(tuple, left.tolist())))
    sorted_right = np.array(sorted(map(tuple, right.tolist())))
    return {
        "sorted_equal": bool(np.array_equal(sorted_left, sorted_right)),
        "axis_swap_equal": bool(np.array_equal(left[..., ::-1], right)),
        "negated_equal": bool(np.array_equal(-left, right)),
        "axis_swap_and_negated_equal": bool(np.array_equal(-left[..., ::-1], right)),
        "negate_x_equal": bool(np.array_equal(left * [-1, 1], right)),
        "negate_y_equal": bool(np.array_equal(left * [1, -1], right)),
    }


def compare_field(name, online, cached):
    result = {
        "online": {"shape": list(online.shape), "dtype": str(online.dtype), "range": numeric_range(online)},
        "cache": {"shape": list(cached.shape), "dtype": str(cached.dtype), "range": numeric_range(cached)},
        "equal": exact(online, cached),
    }
    if online.shape == cached.shape:
        result["max_abs_diff"] = float(np.max(np.abs(online.astype(np.float64) - cached.astype(np.float64))))
    else:
        result["max_abs_diff"] = None
    if name == "grid":
        result["per_channel_iou"] = grid_iou(online, cached)
        matches = [key for key, value in grid_transforms(online).items() if exact(value, cached)]
        result["matching_transforms"] = matches
    elif name == "direction_vectors":
        result["online_values"] = online.tolist()
        result["cache_values"] = cached.tolist()
        result["difference"] = (online - cached).tolist() if online.shape == cached.shape else None
        result["relations"] = vector_relations(online, cached)
    else:
        result["online_values"] = online.tolist()
        result["cache_values"] = cached.tolist()
        result["difference"] = (online - cached).tolist() if online.shape == cached.shape else None
    return result


def summarize(results):
    summary = {}
    fields = ("grid", "direction_vectors", "start_direction_vector", "start_position")
    for cache_name, comparisons in results.items():
        available = [item for item in comparisons if item["cache_exists"]]
        field_equal = {
            field: sum(item["fields"][field]["equal"] for item in available)
            for field in fields
        }
        ious = [
            iou
            for item in available
            for iou in (item["fields"]["grid"]["per_channel_iou"] or [])
        ]
        transforms = Counter(
            transform
            for item in available
            for transform in item["fields"]["grid"]["matching_transforms"]
        )
        summary[cache_name] = {
            "episodes": len(comparisons),
            "cache_files_found": len(available),
            "fully_equal_count_by_field": field_equal,
            "grid_mean_channel_iou": float(np.mean(ious)) if ious else None,
            "grid_transform_distribution": dict(transforms),
        }
    return summary


def main() -> None:
    with (ONLINE_DIR / "index.json").open() as file:
        index = json.load(file)
    results = {name: [] for name in CACHE_ROOTS}
    fields = ("grid", "direction_vectors", "start_direction_vector", "start_position")
    for record in index:
        online_path = ONLINE_DIR / record["file"]
        with np.load(online_path, allow_pickle=False) as online_npz:
            online = {field: online_npz[field] for field in fields}
        scene = Path(record["scene_id"]).stem
        # The train_90 source subset is stored in the historical cache under
        # the parent R2R ``train`` split name, while its online dump keeps the
        # requested source split name.
        cache_filename = record["file"].replace("R2R_train_90_", "R2R_train_")
        for cache_name, cache_root in CACHE_ROOTS.items():
            cache_path = cache_root / scene / cache_filename
            item = {"episode": record, "cache_path": str(cache_path), "cache_exists": cache_path.exists()}
            if cache_path.exists():
                with np.load(cache_path, allow_pickle=True) as cache_npz:
                    item["fields"] = {field: compare_field(field, online[field], cache_npz[field]) for field in fields}
            else:
                item["fields"] = {}
            results[cache_name].append(item)
    payload = {"summary": summarize(results), "comparisons": results}
    with OUTPUT_PATH.open("w") as file:
        json.dump(payload, file, indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
