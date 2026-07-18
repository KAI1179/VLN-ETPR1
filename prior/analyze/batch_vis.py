"""Render paired predicted and ground-truth cognitive-map samples."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from tap import Tap

from prior.bbox import RelevantSemanticBoxes
from prior.grid_map import BaseGridMap, CognitiveGridMap
from prior.trajectory import Point2D


class BatchVisualizationArgs(Tap):
    prediction_root: Path = Path(
        "data/llm_navigation/"
        "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree/"
        "r2r/val_unseen/cognitive_maps/raster"
    )
    # Use the same cache key under r2r/train to sample training predictions.
    """Prediction raster directory containing one directory per scene."""
    ground_truth_root: Path = Path(
        "data/cognitive_maps/gt.legacy.r1p5.direction5.blurred.v1"
    )
    """Ground-truth cache namespace containing raster and boxes directories."""
    output_root: Path = Path(
        "data/samples/llm-grid-s2.legacy.r1p5.direction5.comparison"
    )
    """Destination directory containing one directory per scene."""
    count: Optional[int] = None
    """Number of randomly sampled predictions; defaults to all predictions."""
    seed: int = 0
    """Random seed used when count is set."""


def _sample_prediction_paths(
    prediction_root: Path,
    count: Optional[int],
    seed: int,
) -> list[Path]:
    paths = sorted(prediction_root.glob("*/*.npz"))
    if count is None:
        return paths
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count > len(paths):
        raise ValueError(
            f"count {count} exceeds available prediction count {len(paths)}"
        )
    return sorted(random.Random(seed).sample(paths, count))


def _load_prediction_map(
    path: Path,
) -> tuple[BaseGridMap, Optional[Sequence[Point2D]]]:
    with np.load(path) as data:
        has_trajectory_keypoints = "trajectory_keypoints" in data.files
    if not has_trajectory_keypoints:
        return BaseGridMap.load(path), None
    cognitive_map = CognitiveGridMap.load(path)
    return cognitive_map, cognitive_map.trajectory_keypoints


def render_comparisons(
    prediction_root: Path,
    ground_truth_root: Path,
    output_root: Path,
    *,
    count: Optional[int] = None,
    seed: int = 0,
) -> int:
    if not prediction_root.is_dir():
        raise FileNotFoundError(f"Missing prediction root: {prediction_root}")
    raster_root = ground_truth_root / "raster"
    boxes_root = ground_truth_root / "boxes"
    if not raster_root.is_dir():
        raise FileNotFoundError(f"Missing ground-truth raster root: {raster_root}")
    if not boxes_root.is_dir():
        raise FileNotFoundError(f"Missing ground-truth boxes root: {boxes_root}")

    rendered = 0
    for prediction_path in _sample_prediction_paths(prediction_root, count, seed):
        scene_id = prediction_path.parent.name
        ground_truth_path = raster_root / scene_id / prediction_path.name
        boxes_path = boxes_root / scene_id / prediction_path.name
        if not ground_truth_path.is_file():
            raise FileNotFoundError(
                f"Missing paired ground-truth raster: {ground_truth_path}"
            )
        if not boxes_path.is_file():
            raise FileNotFoundError(f"Missing paired boxes: {boxes_path}")

        predicted_map, predicted_trajectory_keypoints = _load_prediction_map(
            prediction_path
        )
        ground_truth_map = BaseGridMap.load(ground_truth_path)
        boxes = RelevantSemanticBoxes.load(boxes_path)
        output_path = output_root / scene_id / f"{prediction_path.stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predicted_map.visualize_comparison(
            ground_truth_map,
            output_path,
            instruction=boxes.instruction,
            predicted_trajectory_keypoints=predicted_trajectory_keypoints,
            ground_truth_trajectory=boxes.ground_truth_trajectory,
            ground_truth_trajectory_keypoints=boxes.trajectory_keypoints,
            start_direction_vector=boxes.start_direction_vector,
        )
        rendered += 1

    return rendered


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = BatchVisualizationArgs(underscores_to_dashes=True).parse_args(argv)
    rendered = render_comparisons(
        args.prediction_root,
        args.ground_truth_root,
        args.output_root,
        count=args.count,
        seed=args.seed,
    )
    print(f"Rendered {rendered} comparison visualizations to {args.output_root}")


if __name__ == "__main__":
    main()
