"""Build blurred cognitive-map cache namespaces from existing raster caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from tap import Tap

from prior import DATA_DIR

DEFAULT_SOURCE_NAMESPACE = "gt.legacy.r1p5.direction5.v1"
DEFAULT_TARGET_NAMESPACE = "gt.legacy.r1p5.direction5.blurred.v1"
DEFAULT_CACHE_ROOT = DATA_DIR / "cognitive_maps"


class BlurArgs(Tap):
    cache_root: Path = DEFAULT_CACHE_ROOT
    """Root cache directory containing cognitive-map namespaces."""
    source_namespace: str = DEFAULT_SOURCE_NAMESPACE
    """Existing namespace to process."""
    target_namespace: str = DEFAULT_TARGET_NAMESPACE
    """Derived namespace to write."""
    scale: int = 2
    """Spatial bottleneck scale. Use 2 for 100x100 -> 50x50 -> 100x100."""
    overwrite: bool = False
    """Overwrite existing target raster and boxes files."""


@dataclass(frozen=True)
class BlurSummary:
    raster_total: int
    raster_written: int
    raster_skipped: int
    boxes_written: int
    boxes_skipped: int
    boxes_missing: int


def blur_cognitive_map_grid(
    grid: NDArray[np.float32],
    scale: int = 2,
) -> NDArray[np.float32]:
    """Max-pool a grid by ``scale`` and nearest-neighbor upsample it back."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if grid.ndim != 3:
        raise ValueError(
            f"grid must have shape (channels, rows, cols), got {grid.shape}"
        )
    channels, rows, cols = grid.shape
    if rows % scale != 0 or cols % scale != 0:
        raise ValueError(
            f"grid shape {(channels, rows, cols)} is not divisible by scale {scale}"
        )
    pooled = grid.reshape(
        channels,
        rows // scale,
        scale,
        cols // scale,
        scale,
    ).max(axis=(2, 4))
    return np.repeat(np.repeat(pooled, scale, axis=1), scale, axis=2).astype(
        np.float32,
        copy=False,
    )


def blur_cognitive_map_file(
    source_path: Path,
    target_path: Path,
    scale: int = 2,
    overwrite: bool = False,
) -> bool:
    """Write one blurred raster file, preserving all non-grid arrays."""
    if target_path.exists() and not overwrite:
        return False
    with np.load(source_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    if "grid" not in payload:
        raise KeyError(f"Missing grid in {source_path}")
    payload["grid"] = blur_cognitive_map_grid(
        np.asarray(payload["grid"], dtype=np.float32),
        scale=scale,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_path, **payload)
    return True


def _copy_box_file(source_path: Path, target_path: Path, overwrite: bool) -> bool:
    if target_path.exists() and not overwrite:
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return True


def transform_cognitive_map_namespace(
    cache_root: Path,
    source_namespace: str = DEFAULT_SOURCE_NAMESPACE,
    target_namespace: str = DEFAULT_TARGET_NAMESPACE,
    scale: int = 2,
    overwrite: bool = False,
) -> BlurSummary:
    source_root = cache_root / source_namespace
    target_root = cache_root / target_namespace
    source_raster_root = source_root / "raster"
    source_boxes_root = source_root / "boxes"
    if not source_raster_root.is_dir():
        raise FileNotFoundError(
            f"Missing source raster namespace: {source_raster_root}"
        )

    raster_written = 0
    raster_skipped = 0
    boxes_written = 0
    boxes_skipped = 0
    boxes_missing = 0
    raster_paths = sorted(source_raster_root.glob("*/*.npz"))
    for raster_path in raster_paths:
        relative_path = raster_path.relative_to(source_raster_root)
        target_raster_path = target_root / "raster" / relative_path
        if blur_cognitive_map_file(
            raster_path,
            target_raster_path,
            scale=scale,
            overwrite=overwrite,
        ):
            raster_written += 1
        else:
            raster_skipped += 1

        boxes_path = source_boxes_root / relative_path
        if not boxes_path.is_file():
            boxes_missing += 1
            continue
        target_boxes_path = target_root / "boxes" / relative_path
        if _copy_box_file(boxes_path, target_boxes_path, overwrite=overwrite):
            boxes_written += 1
        else:
            boxes_skipped += 1

    return BlurSummary(
        raster_total=len(raster_paths),
        raster_written=raster_written,
        raster_skipped=raster_skipped,
        boxes_written=boxes_written,
        boxes_skipped=boxes_skipped,
        boxes_missing=boxes_missing,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = BlurArgs(underscores_to_dashes=True).parse_args(argv)
    summary = transform_cognitive_map_namespace(
        cache_root=args.cache_root,
        source_namespace=args.source_namespace,
        target_namespace=args.target_namespace,
        scale=args.scale,
        overwrite=args.overwrite,
    )
    print(
        "blurred_cognitive_maps: "
        f"source={args.source_namespace} target={args.target_namespace} "
        f"raster_total={summary.raster_total} "
        f"raster_written={summary.raster_written} "
        f"raster_skipped={summary.raster_skipped} "
        f"boxes_written={summary.boxes_written} "
        f"boxes_skipped={summary.boxes_skipped} "
        f"boxes_missing={summary.boxes_missing}"
    )


if __name__ == "__main__":
    main()
