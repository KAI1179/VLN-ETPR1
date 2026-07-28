"""Diagnose cardinal start-centred registration in a fixed LLM-Grid cache."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tap import Tap

from prior.analyze.llm_grid_registration import (
    direction_cosine,
    rotate_direction_vectors,
    score_warped_grid,
    select_best_angle,
    warp_grid_about_pivot,
)
from prior.constants import CELL_SIZE, MAPPED_OBJECT_NAMES, OBJECT_CATEGORIES
from prior.grid_map._cognitive import extract_categories
from vlnce_baselines.models.etp_llm.llm_grid_train import (
    GRID_SCALE,
    GRID_SHAPE,
    LLMGridDataset,
    LLMGridValidationError,
    load_llm_grid_examples,
    parse_grid_text,
)
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_prediction_path,
    llm_navigation_split_dir,
    validate_llm_navigation_manifest,
)


_BROAD_OBJECT_NAMES = frozenset(("void", "structure", "other", "free-space"))
_EXPECTED_EXAMPLES = 1_839
_EXPECTED_SCENES = 11
_DEFAULT_ANGLES = (0.0, 90.0, 180.0, 270.0)
_DEFAULT_CACHE_MODEL_KEY = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2"
_DEFAULT_COGNITIVE_MAP_NAMESPACE = "gt.legacy.r1p5.direction5.v1"
_DEFAULT_BOOTSTRAP_REPETITIONS = 10_000
_DEFAULT_BOOTSTRAP_SEED = 42


class FourWayDiagnosisArgs(Tap):
    """Inputs and outputs for the fixed epoch-2 R2R transform diagnosis."""

    cache_dir: Path = Path("data/llm_navigation")
    cache_model_key: str = _DEFAULT_CACHE_MODEL_KEY
    cognitive_map_namespace: str = _DEFAULT_COGNITIVE_MAP_NAMESPACE
    output_dir: Path = Path(
        "outputs/llm_grid_analysis/start_centered_registration_r2r_rxr_epoch2"
    )
    angles: Tuple[float, ...] = _DEFAULT_ANGLES
    bootstrap_repetitions: int = _DEFAULT_BOOTSTRAP_REPETITIONS
    bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED
    limit: Optional[int] = None
    quiet: bool = False


@dataclass(frozen=True)
class EpisodeTransformResult:
    """One retained R2R val_unseen episode scored over candidate rotations."""

    split: str
    scene_id: str
    example_id: str
    schema_valid: bool
    identity_iou: float
    best_iou: float
    delta_iou: float
    best_angle_degrees: float
    second_angle_degrees: float
    best_second_margin: float
    angle_ious: tuple[tuple[float, float], ...]
    cell_precision: float
    cell_recall: float
    cell_f1: float
    direction_cosine_before: float
    direction_cosine_after: float
    input_support: int
    warped_support: int
    in_frame_support: int
    out_of_frame_support: int
    target_support: int
    object_identity_iou: float
    object_best_iou: float
    region_identity_iou: float
    region_best_iou: float
    mentioned_identity_iou: float
    mentioned_best_iou: float
    unmentioned_identity_iou: float
    unmentioned_best_iou: float
    occupancy_identity_iou: float
    occupancy_best_iou: float
    broad_excluded_identity_iou: float
    broad_excluded_best_iou: float
    mentioned_target_support: int
    unmentioned_target_support: int


def start_pivot_for_scale(
    start_position_m: tuple[float, float], scale: int
) -> tuple[float, float]:
    """Convert level-local metres to the row/column grid pivot at ``scale``."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    if len(start_position_m) != 2:
        raise ValueError("start_position_m must contain exactly two coordinates")
    try:
        row_m = float(start_position_m[0])
        col_m = float(start_position_m[1])
    except (TypeError, ValueError) as error:
        raise ValueError("start_position_m coordinates must be finite") from error
    if not np.isfinite(row_m) or not np.isfinite(col_m):
        raise ValueError("start_position_m coordinates must be finite")
    cell_width_m = CELL_SIZE * scale
    return row_m / cell_width_m, col_m / cell_width_m


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _shared_angle_iou(
    predicted: NDArray[np.bool_],
    target: NDArray[np.bool_],
    pivot: tuple[float, float],
    best_angle: float,
) -> tuple[float, float]:
    identity = score_warped_grid(warp_grid_about_pivot(predicted, pivot, 0.0), target)
    selected = score_warped_grid(
        warp_grid_about_pivot(predicted, pivot, best_angle), target
    )
    return identity.iou, selected.iou


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p02_5": 0.0, "p50": 0.0, "p97_5": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p02_5": float(np.percentile(array, 2.5)),
        "p50": float(np.percentile(array, 50.0)),
        "p97_5": float(np.percentile(array, 97.5)),
    }


def _validate_episode_inputs(
    predicted_grid: NDArray[np.bool_],
    target_grid: NDArray[np.bool_],
    predicted_direction_vectors: NDArray[np.float32],
    target_direction_vectors: NDArray[np.float32],
) -> None:
    expected_channels = GRID_SHAPE[0]
    if predicted_grid.ndim != 3 or predicted_grid.shape[0] != expected_channels:
        raise ValueError(
            f"predicted_grid must have shape ({expected_channels}, H, W), "
            f"got {predicted_grid.shape}"
        )
    if target_grid.ndim != 3 or target_grid.shape[0] != expected_channels:
        raise ValueError(
            f"target_grid must have shape ({expected_channels}, H, W), "
            f"got {target_grid.shape}"
        )
    if predicted_grid.shape != target_grid.shape:
        raise ValueError("predicted_grid and target_grid must have the same shape")
    if predicted_direction_vectors.shape != (5, 2):
        raise ValueError("predicted_direction_vectors must have shape (5, 2)")
    if target_direction_vectors.shape != (5, 2):
        raise ValueError("target_direction_vectors must have shape (5, 2)")


def evaluate_episode(
    *,
    split: str,
    scene_id: str,
    example_id: str,
    schema_valid: bool,
    predicted_grid: NDArray[np.bool_],
    target_grid: NDArray[np.bool_],
    start_position_m: tuple[float, float],
    predicted_direction_vectors: NDArray[np.float32],
    target_direction_vectors: NDArray[np.float32],
    instruction: str,
    angles: tuple[float, ...],
) -> EpisodeTransformResult:
    """Score one retained episode with one all-channel selected angle."""
    _validate_episode_inputs(
        predicted_grid,
        target_grid,
        predicted_direction_vectors,
        target_direction_vectors,
    )
    if not schema_valid and (
        np.any(predicted_grid) or np.any(predicted_direction_vectors)
    ):
        raise ValueError(
            "schema_valid=False requires an empty grid and zero direction vectors"
        )
    pivot = start_pivot_for_scale(start_position_m, 2)
    best, scores = select_best_angle(predicted_grid, target_grid, pivot, angles)
    try:
        identity = next(score for score in scores if score.angle_degrees == 0.0)
    except StopIteration as error:
        raise ValueError("angles must contain identity 0.0") from error
    ordered = sorted(enumerate(scores), key=lambda item: (-item[1].raster.iou, item[0]))
    second = ordered[1][1] if len(ordered) > 1 else best
    raster = best.raster
    precision = _ratio(raster.intersection, raster.predicted_support)
    recall = _ratio(raster.intersection, raster.target_support)
    mentioned_objects, mentioned_regions = extract_categories(instruction)
    mentioned_channels = frozenset(mentioned_objects) | frozenset(
        OBJECT_CATEGORIES + category for category in mentioned_regions
    )
    channel_count = predicted_grid.shape[0]
    if target_grid.shape[0] != channel_count:
        raise ValueError("predicted_grid and target_grid must have the same channel count")
    if any(channel < 0 or channel >= channel_count for channel in mentioned_channels):
        raise ValueError("instruction mention channel is outside the supplied grid")
    object_mask = np.arange(channel_count) < OBJECT_CATEGORIES
    region_mask = ~object_mask
    mentioned_mask = np.asarray(
        [channel in mentioned_channels for channel in range(channel_count)], dtype=np.bool_
    )
    broad_object_channels = frozenset(
        index
        for index, name in enumerate(MAPPED_OBJECT_NAMES)
        if name in _BROAD_OBJECT_NAMES
    )
    broad_excluded_mask = np.asarray(
        [
            channel >= OBJECT_CATEGORIES or channel not in broad_object_channels
            for channel in range(channel_count)
        ],
        dtype=np.bool_,
    )
    object_identity_iou, object_best_iou = _shared_angle_iou(
        predicted_grid[object_mask], target_grid[object_mask], pivot, best.angle_degrees
    )
    region_identity_iou, region_best_iou = _shared_angle_iou(
        predicted_grid[region_mask], target_grid[region_mask], pivot, best.angle_degrees
    )
    mentioned_identity_iou, mentioned_best_iou = _shared_angle_iou(
        predicted_grid[mentioned_mask], target_grid[mentioned_mask], pivot, best.angle_degrees
    )
    unmentioned_identity_iou, unmentioned_best_iou = _shared_angle_iou(
        predicted_grid[~mentioned_mask], target_grid[~mentioned_mask], pivot, best.angle_degrees
    )
    occupancy_identity_iou, occupancy_best_iou = _shared_angle_iou(
        np.any(predicted_grid, axis=0, keepdims=True),
        np.any(target_grid, axis=0, keepdims=True),
        pivot,
        best.angle_degrees,
    )
    broad_excluded_identity_iou, broad_excluded_best_iou = _shared_angle_iou(
        predicted_grid[broad_excluded_mask],
        target_grid[broad_excluded_mask],
        pivot,
        best.angle_degrees,
    )
    return EpisodeTransformResult(
        split=split,
        scene_id=scene_id,
        example_id=example_id,
        schema_valid=schema_valid,
        identity_iou=identity.raster.iou,
        best_iou=best.raster.iou,
        delta_iou=best.raster.iou - identity.raster.iou,
        best_angle_degrees=best.angle_degrees,
        second_angle_degrees=second.angle_degrees,
        best_second_margin=best.raster.iou - second.raster.iou,
        angle_ious=tuple((score.angle_degrees, score.raster.iou) for score in scores),
        cell_precision=precision,
        cell_recall=recall,
        cell_f1=_ratio(2 * raster.intersection, raster.predicted_support + raster.target_support),
        direction_cosine_before=direction_cosine(
            predicted_direction_vectors, target_direction_vectors
        ),
        direction_cosine_after=direction_cosine(
            rotate_direction_vectors(predicted_direction_vectors, best.angle_degrees),
            target_direction_vectors,
        ),
        input_support=int(np.count_nonzero(predicted_grid)),
        warped_support=raster.predicted_support,
        in_frame_support=raster.in_frame_support,
        out_of_frame_support=raster.out_of_frame_support,
        target_support=raster.target_support,
        object_identity_iou=object_identity_iou,
        object_best_iou=object_best_iou,
        region_identity_iou=region_identity_iou,
        region_best_iou=region_best_iou,
        mentioned_identity_iou=mentioned_identity_iou,
        mentioned_best_iou=mentioned_best_iou,
        unmentioned_identity_iou=unmentioned_identity_iou,
        unmentioned_best_iou=unmentioned_best_iou,
        occupancy_identity_iou=occupancy_identity_iou,
        occupancy_best_iou=occupancy_best_iou,
        broad_excluded_identity_iou=broad_excluded_identity_iou,
        broad_excluded_best_iou=broad_excluded_best_iou,
        mentioned_target_support=int(np.count_nonzero(target_grid[mentioned_mask])),
        unmentioned_target_support=int(np.count_nonzero(target_grid[~mentioned_mask])),
    )


def _validate_args(args: FourWayDiagnosisArgs) -> None:
    if args.cache_model_key != _DEFAULT_CACHE_MODEL_KEY:
        raise ValueError(f"cache_model_key must be exactly {_DEFAULT_CACHE_MODEL_KEY!r}")
    if args.cognitive_map_namespace != _DEFAULT_COGNITIVE_MAP_NAMESPACE:
        raise ValueError(
            "cognitive_map_namespace must be exactly "
            f"{_DEFAULT_COGNITIVE_MAP_NAMESPACE!r}"
        )
    if tuple(args.angles) != _DEFAULT_ANGLES:
        raise ValueError(f"angles must be exactly {_DEFAULT_ANGLES}")
    if args.bootstrap_repetitions != _DEFAULT_BOOTSTRAP_REPETITIONS:
        raise ValueError(
            "bootstrap_repetitions must be exactly "
            f"{_DEFAULT_BOOTSTRAP_REPETITIONS}"
        )
    if args.bootstrap_seed != _DEFAULT_BOOTSTRAP_SEED:
        raise ValueError(f"bootstrap_seed must be exactly {_DEFAULT_BOOTSTRAP_SEED}")
    if args.limit is not None and (
        isinstance(args.limit, bool) or args.limit <= 0
    ):
        raise ValueError("limit must be absent or positive")
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    if (
        cache_dir == output_dir
        or cache_dir in output_dir.parents
        or output_dir in cache_dir.parents
    ):
        raise ValueError("output_dir and cache_dir must not overlap")


def _prediction_manifest_path(args: FourWayDiagnosisArgs) -> Path:
    split_dir = llm_navigation_split_dir(
        "R2R", "val_unseen", cache_dir=args.cache_dir, model_key=args.cache_model_key
    )
    validate_llm_navigation_manifest(
        split_dir,
        {
            "dataset": "R2R",
            "split": "val_unseen",
            "generator": "llm-grid",
            "scale": GRID_SCALE,
            "cache_model_key": args.cache_model_key,
        },
    )
    return split_dir / "manifest.json"


def _load_episode_rows(
    args: FourWayDiagnosisArgs,
) -> tuple[tuple[EpisodeTransformResult, ...], Path]:
    prediction_manifest_path = _prediction_manifest_path(args)
    loaded = load_llm_grid_examples(
        ("val_unseen",),
        limit_per_dataset=args.limit,
        quiet=args.quiet,
        cognitive_map_namespace=args.cognitive_map_namespace,
        datasets=("R2R",),
    )
    dataset = LLMGridDataset(loaded.examples, scale=GRID_SCALE)
    rows: list[EpisodeTransformResult] = []
    for example, item in zip(loaded.examples, dataset):
        if item["example_id"] != example.example_id:
            raise ValueError("LLM-Grid dataset order changed during transform diagnosis")
        prediction_path = llm_navigation_prediction_path(
            example.scene_id,
            example.example_id,
            "R2R",
            "val_unseen",
            cache_dir=args.cache_dir,
            model_key=args.cache_model_key,
        )
        schema_valid = True
        predicted_grid = np.zeros(GRID_SHAPE, dtype=np.bool_)
        predicted_directions = np.zeros((5, 2), dtype=np.float32)
        try:
            parsed = parse_grid_text(
                prediction_path.read_text(encoding="utf-8"), shape=GRID_SHAPE
            )
            predicted_grid = parsed.grid > 0
            predicted_directions = parsed.direction_vectors
        except (FileNotFoundError, LLMGridValidationError):
            schema_valid = False
        target_grid = item["target_grid"] > 0
        if target_grid.shape != GRID_SHAPE:
            raise ValueError(
                f"unexpected target shape for {example.example_id}: {target_grid.shape}"
            )
        rows.append(
            evaluate_episode(
                split="val_unseen",
                scene_id=example.scene_id,
                example_id=example.example_id,
                schema_valid=schema_valid,
                predicted_grid=predicted_grid,
                target_grid=target_grid,
                start_position_m=(
                    float(item["start_position"][0]),
                    float(item["start_position"][1]),
                ),
                predicted_direction_vectors=predicted_directions,
                target_direction_vectors=item["target_direction_vectors"],
                instruction=example.instruction,
                angles=tuple(args.angles),
            )
        )
    return tuple(rows), prediction_manifest_path


def _scene_bootstrap_multiplicities(
    rows: Sequence[EpisodeTransformResult], repetitions: int, seed: int
) -> tuple[tuple[str, ...], NDArray[np.int64]]:
    scene_ids = tuple(sorted({row.scene_id for row in rows}))
    if not scene_ids:
        raise ValueError("bootstrap requires at least one episode")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(scene_ids), size=(repetitions, len(scene_ids)))
    multiplicities = np.zeros((repetitions, len(scene_ids)), dtype=np.int64)
    for scene_index in range(len(scene_ids)):
        multiplicities[:, scene_index] = np.count_nonzero(
            draws == scene_index, axis=1
        )
    return scene_ids, multiplicities


def bootstrap_scene_delta(
    rows: Sequence[EpisodeTransformResult], repetitions: int, seed: int
) -> dict[str, object]:
    """Cluster-bootstrap episode-macro IoU deltas by scene with replacement."""
    if isinstance(repetitions, bool) or repetitions <= 0:
        raise ValueError("repetitions must be positive")
    scene_ids, multiplicities = _scene_bootstrap_multiplicities(rows, repetitions, seed)
    scene_sums = np.asarray(
        [sum(row.delta_iou for row in rows if row.scene_id == scene) for scene in scene_ids],
        dtype=np.float64,
    )
    scene_counts = np.asarray(
        [sum(row.scene_id == scene for row in rows) for scene in scene_ids],
        dtype=np.float64,
    )
    sampled = (multiplicities @ scene_sums) / (multiplicities @ scene_counts)
    return {
        "sampling_unit": "scene",
        "repetitions": repetitions,
        "seed": seed,
        "mean_delta": _mean([row.delta_iou for row in rows]),
        "bootstrap_mean_delta": float(np.mean(sampled)),
        "percentile_2_5": float(np.percentile(sampled, 2.5)),
        "percentile_97_5": float(np.percentile(sampled, 97.5)),
    }


def _metric_pair_summary(
    rows: Sequence[EpisodeTransformResult],
    select: Callable[[EpisodeTransformResult], tuple[float, float]],
) -> dict[str, float]:
    pairs = [select(row) for row in rows]
    identity = [pair[0] for pair in pairs]
    best = [pair[1] for pair in pairs]
    return {
        "identity_iou": _mean(identity),
        "best_iou": _mean(best),
        "delta_iou": _mean(best) - _mean(identity),
    }


def _support_summary(
    rows: Sequence[EpisodeTransformResult],
    select: Callable[[EpisodeTransformResult], int],
) -> dict[str, float]:
    values = [float(select(row)) for row in rows]
    return {"mean": _mean(values), "median": _median(values), "total": float(sum(values))}


def _summary(rows: Sequence[EpisodeTransformResult]) -> dict[str, object]:
    deltas = [row.delta_iou for row in rows]
    margins = [row.best_second_margin for row in rows]
    angle_counts = {
        str(angle): sum(row.best_angle_degrees == angle for row in rows)
        for angle in _DEFAULT_ANGLES
    }
    return {
        "examples": len(rows),
        "scenes": len({row.scene_id for row in rows}),
        "valid_examples": sum(row.schema_valid for row in rows),
        "invalid_examples": sum(not row.schema_valid for row in rows),
        "identity_episode_macro_iou": _mean([row.identity_iou for row in rows]),
        "best_episode_macro_iou": _mean([row.best_iou for row in rows]),
        "mean_paired_delta_iou": _mean(deltas),
        "median_paired_delta_iou": _median(deltas),
        "proportion_delta_above_0_01": _mean([delta > 0.01 for delta in deltas]),
        "proportion_delta_above_0_05": _mean([delta > 0.05 for delta in deltas]),
        "identity_rate": _mean([row.best_angle_degrees == 0.0 for row in rows]),
        "angle_counts": angle_counts,
        "best_second_margin": _quantiles(margins),
        "cell_metrics": {
            "precision": _mean([row.cell_precision for row in rows]),
            "recall": _mean([row.cell_recall for row in rows]),
            "f1": _mean([row.cell_f1 for row in rows]),
        },
        "direction_cosine": {
            "before": _mean([row.direction_cosine_before for row in rows]),
            "after": _mean([row.direction_cosine_after for row in rows]),
        },
        "sensitivity": {
            "object": _metric_pair_summary(
                rows, lambda row: (row.object_identity_iou, row.object_best_iou)
            ),
            "region": _metric_pair_summary(
                rows, lambda row: (row.region_identity_iou, row.region_best_iou)
            ),
            "mentioned": _metric_pair_summary(
                rows, lambda row: (row.mentioned_identity_iou, row.mentioned_best_iou)
            ),
            "unmentioned": _metric_pair_summary(
                rows,
                lambda row: (row.unmentioned_identity_iou, row.unmentioned_best_iou),
            ),
            "occupancy": _metric_pair_summary(
                rows, lambda row: (row.occupancy_identity_iou, row.occupancy_best_iou)
            ),
            "broad_excluded": _metric_pair_summary(
                rows,
                lambda row: (
                    row.broad_excluded_identity_iou,
                    row.broad_excluded_best_iou,
                ),
            ),
        },
        "support": {
            "input_support": _support_summary(rows, lambda row: row.input_support),
            "warped_support": _support_summary(rows, lambda row: row.warped_support),
            "in_frame_support": _support_summary(rows, lambda row: row.in_frame_support),
            "out_of_frame_support": _support_summary(
                rows, lambda row: row.out_of_frame_support
            ),
            "target_support": _support_summary(rows, lambda row: row.target_support),
        },
        "stop_go_gate_inputs": {
            "identity_rate": _mean([row.best_angle_degrees == 0.0 for row in rows]),
            "proportion_delta_above_0_01": _mean([delta > 0.01 for delta in deltas]),
            "proportion_delta_above_0_05": _mean([delta > 0.05 for delta in deltas]),
            "interpretation": "diagnostic registration evidence only; not a deployability claim",
        },
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[EpisodeTransformResult]) -> None:
    fieldnames = tuple(field.name for field in fields(EpisodeTransformResult))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            record["angle_ious"] = json.dumps(record["angle_ious"], separators=(",", ":"))
            writer.writerow(record)


def _write_angle_distribution_csv(
    path: Path, rows: Sequence[EpisodeTransformResult]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("angle_degrees", "episode_count"))
        for angle in _DEFAULT_ANGLES:
            writer.writerow(
                (int(angle), sum(row.best_angle_degrees == angle for row in rows))
            )


def _write_plots(output_dir: Path, rows: Sequence[EpisodeTransformResult]) -> None:
    deltas = [row.delta_iou for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(deltas, bins=30, color="#4878A8", edgecolor="white")
    for value, color in ((0.0, "#222222"), (0.01, "#F58518"), (0.05, "#E45756")):
        axis.axvline(value, color=color, linestyle="--", linewidth=1.2)
    axis.set_xlabel("Best minus identity category-aware IoU")
    axis.set_ylabel("Episodes")
    figure.tight_layout()
    figure.savefig(
        output_dir / "iou_delta_distribution.png", dpi=160, metadata={"Date": None}
    )
    plt.close(figure)

    angles = _DEFAULT_ANGLES
    counts = [sum(row.best_angle_degrees == angle for row in rows) for angle in angles]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([str(int(angle)) for angle in angles], counts, color="#54A24B")
    axis.set_xlabel("Selected rotation (degrees)")
    axis.set_ylabel("Episodes")
    figure.tight_layout()
    figure.savefig(
        output_dir / "angle_distribution.png", dpi=160, metadata={"Date": None}
    )
    plt.close(figure)


def _validate_rows(rows: Sequence[EpisodeTransformResult], expected_population: int) -> None:
    identities = [(row.scene_id, row.example_id) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate (scene_id, example_id) identity in diagnosis rows")
    if len(rows) != expected_population:
        raise ValueError(f"expected {expected_population} episodes, got {len(rows)}")


def _run_rows(
    args: FourWayDiagnosisArgs,
    rows: Sequence[EpisodeTransformResult],
    *,
    expected_population: int,
    prediction_manifest_path: Path,
) -> dict[str, object]:
    """Private synthetic runner with an injected population contract for tests."""
    _validate_args(args)
    _validate_rows(rows, expected_population)
    if args.limit is None and len({row.scene_id for row in rows}) != _EXPECTED_SCENES:
        raise ValueError(f"expected {_EXPECTED_SCENES} unique scenes for a full run")
    if not prediction_manifest_path.is_file():
        raise FileNotFoundError(
            f"prediction_manifest_path must be a file: {prediction_manifest_path}"
        )
    prediction_manifest_sha256 = _sha256_file(prediction_manifest_path)
    ordered_rows = tuple(sorted(rows, key=lambda row: (row.scene_id, row.example_id)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(ordered_rows)
    if args.limit is not None:
        summary["smoke"] = True
    bootstrap = bootstrap_scene_delta(
        ordered_rows, args.bootstrap_repetitions, args.bootstrap_seed
    )
    manifest = {
        "cache_dir": str(args.cache_dir),
        "prediction_manifest_path": str(prediction_manifest_path),
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "cache_model_key": args.cache_model_key,
        "cognitive_map_namespace": args.cognitive_map_namespace,
        "dataset": "R2R",
        "split": "val_unseen",
        "angles": list(_DEFAULT_ANGLES),
        "shape": list(GRID_SHAPE),
        "scale": GRID_SCALE,
        "cell_size_m": CELL_SIZE * GRID_SCALE,
        "bootstrap": {
            "sampling_unit": "scene",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
            "percentiles": [2.5, 97.5],
        },
        "population": expected_population,
        "smoke": args.limit is not None,
        "git_commit": _git_commit(),
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "manifest.json", manifest)
    _write_json(args.output_dir / "bootstrap.json", bootstrap)
    _write_csv(args.output_dir / "episodes.csv", ordered_rows)
    _write_angle_distribution_csv(
        args.output_dir / "angle_distribution.csv", ordered_rows
    )
    _write_plots(args.output_dir, ordered_rows)
    return {
        "summary": summary,
        "manifest": manifest,
        "bootstrap": bootstrap,
        "output_dir": args.output_dir,
    }


def run_diagnosis(args: FourWayDiagnosisArgs) -> dict[str, object]:
    """Replay the fixed R2R val_unseen cache and write deterministic artifacts."""
    _validate_args(args)
    rows, prediction_manifest_path = _load_episode_rows(args)
    expected_population = _EXPECTED_EXAMPLES if args.limit is None else args.limit
    return _run_rows(
        args,
        rows,
        expected_population=expected_population,
        prediction_manifest_path=prediction_manifest_path,
    )


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    args = FourWayDiagnosisArgs(underscores_to_dashes=True).parse_args(argv)
    result = run_diagnosis(args)
    if not args.quiet:
        print(f"Wrote diagnosis artifacts to {result['output_dir']}")
    return result


if __name__ == "__main__":
    main()
