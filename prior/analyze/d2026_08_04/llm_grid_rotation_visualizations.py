"""Render deterministic examples from the sealed cardinal-rotation diagnoses."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from tap import Tap

from prior.analyze.d2026_07_28.llm_grid_transform_crossfit import (
    CrossFitArgs,
    Direction,
    EpisodeCase,
    _load_episode_cases,
    crossfit_scores,
    score_angle_families,
)
from prior.analyze.llm_grid_registration import (
    RasterScore,
    SpatialBounds,
    WarpedGrid,
    score_warped_grid,
    warp_grid_about_pivot,
)
from prior.constants import OBJECT_CATEGORIES


_ANGLES = (0.0, 90.0, 180.0, 270.0)
_DIAGNOSIS_ROOT = Path(
    "outputs/llm_grid_analysis/start_centered_registration_r2r_rxr_epoch2"
)
_CROSSFIT_ROOT = Path(
    "outputs/llm_grid_analysis/start_centered_crossfit_controls_r2r_rxr_epoch2"
)
_SOURCE_HASHES = {
    _DIAGNOSIS_ROOT / "manifest.json": (
        "6b38bd67f1e58fe3e97ab6a40d79bcc4c9f5129ad151fb9391c84a89503d7c6c"
    ),
    _DIAGNOSIS_ROOT / "episodes.csv": (
        "6ed6e88319134db6c8e2ba4142c1c08163297b1212b2c03fb8be213bcd7c3fdd"
    ),
    _CROSSFIT_ROOT / "manifest.json": (
        "23895a96ffa11219b8297fccbc065ea5d4dfa284451953d60f4ab66e8faaac01"
    ),
    _CROSSFIT_ROOT / "crossfit_results.csv": (
        "6ade56cc30b636c78e01df461ca1f737fefe7fc4793772a130577271b6887205"
    ),
    _CROSSFIT_ROOT / "angle_scores.csv": (
        "682cdcd8600cd498554881bcd3d88c9012035e0db642bcc64cd8ae10af61b69e"
    ),
}
_EXPECTED_ORACLE_IDENTITIES = {
    0.0: ("TbHJrupSAjP", "R2R_val_unseen_1410"),
    90.0: ("zsNo4HB9uLZ", "R2R_val_unseen_1375"),
    180.0: ("X7HyMhZNoso", "R2R_val_unseen_114"),
    270.0: ("EU6Fwq7SyZv", "R2R_val_unseen_484"),
}
_EXPECTED_CROSSFIT_IDENTITIES = {
    "object_to_region": ("oLBMNvg9in8", "R2R_val_unseen_724"),
    "region_to_object": ("x8F5xyUWy9e", "R2R_val_unseen_1594"),
}
_ERROR_CMAP = mcolors.ListedColormap((
    "#FFFFFF",
    "#2CA02C",
    "#D62728",
    "#1F77B4",
    "#9467BD",
))


class RotationVisualizationArgs(Tap):
    """Fixed sealed sources and one experiment-only output directory."""

    output_dir: Path = Path(
        "outputs/llm_grid_analysis/start_centered_rotation_visualizations_2026_08_04"
    )
    quiet: bool = False


@dataclass(frozen=True)
class OracleSelectionRow:
    scene_id: str
    example_id: str
    identity_iou: float
    best_iou: float
    delta_iou: float
    best_angle_degrees: float
    second_angle_degrees: float
    best_second_margin: float
    input_support: int
    target_support: int
    angle_ious: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class CrossFitSelectionRow:
    scene_id: str
    example_id: str
    direction: str
    selected_angle_degrees: float
    selector_identity_iou: float
    selector_selected_iou: float
    selector_margin: float
    heldout_identity_iou: float
    heldout_selected_iou: float
    delta_iou: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_sources() -> None:
    for path, expected in _SOURCE_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing sealed source: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"sealed source hash mismatch for {path}: {actual}")


def _finite_float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {field!r} value") from error
    if not np.isfinite(value):
        raise ValueError(f"{field!r} must be finite")
    return value


def _nonnegative_int(row: Mapping[str, str], field: str) -> int:
    value = _finite_float(row, field)
    if value != int(value) or value < 0:
        raise ValueError(f"{field!r} must be a nonnegative integer")
    return int(value)


def _read_oracle_rows(path: Path) -> Tuple[OracleSelectionRow, ...]:
    result = []
    with path.open(encoding="utf-8", newline="") as file:
        for raw in csv.DictReader(file):
            if raw.get("schema_valid") != "True":
                continue
            try:
                parsed_angles = json.loads(raw["angle_ious"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError("invalid angle_ious JSON") from error
            angle_ious = tuple(
                (float(angle), float(iou)) for angle, iou in parsed_angles
            )
            if tuple(angle for angle, _ in angle_ious) != _ANGLES:
                raise ValueError("angle_ious must use the frozen cardinal order")
            result.append(
                OracleSelectionRow(
                    scene_id=raw["scene_id"],
                    example_id=raw["example_id"],
                    identity_iou=_finite_float(raw, "identity_iou"),
                    best_iou=_finite_float(raw, "best_iou"),
                    delta_iou=_finite_float(raw, "delta_iou"),
                    best_angle_degrees=_finite_float(raw, "best_angle_degrees"),
                    second_angle_degrees=_finite_float(raw, "second_angle_degrees"),
                    best_second_margin=_finite_float(raw, "best_second_margin"),
                    input_support=_nonnegative_int(raw, "input_support"),
                    target_support=_nonnegative_int(raw, "target_support"),
                    angle_ious=angle_ious,
                )
            )
    if len(result) != 1_830:
        raise ValueError(f"expected 1830 schema-valid oracle rows, got {len(result)}")
    identities = {(row.scene_id, row.example_id) for row in result}
    if len(identities) != len(result):
        raise ValueError("oracle rows contain duplicate identities")
    return tuple(result)


def select_oracle_rows(
    rows: Sequence[OracleSelectionRow],
) -> Tuple[OracleSelectionRow, ...]:
    """Select one strict, support-bearing median example for every best angle."""

    selected = []
    for angle in _ANGLES:
        group = tuple(
            row
            for row in rows
            if row.best_angle_degrees == angle
            and row.best_second_margin > 0.0
            and row.input_support > 0
            and row.target_support > 0
        )
        if not group:
            raise ValueError(f"no eligible strict-winner rows for {angle:g} degrees")
        field_values = (
            tuple(row.identity_iou for row in group)
            if angle == 0.0
            else tuple(row.delta_iou for row in group)
        )
        center = median(field_values)
        chosen = min(
            group,
            key=lambda row: (
                abs((row.identity_iou if angle == 0.0 else row.delta_iou) - center),
                row.scene_id,
                row.example_id,
            ),
        )
        selected.append(chosen)
    return tuple(selected)


def _read_crossfit_rows(path: Path) -> Tuple[CrossFitSelectionRow, ...]:
    result = []
    with path.open(encoding="utf-8", newline="") as file:
        for raw in csv.DictReader(file):
            if (
                raw.get("schema_valid") != "True"
                or raw.get("pivot_mode") != "true_start"
            ):
                continue
            result.append(
                CrossFitSelectionRow(
                    scene_id=raw["scene_id"],
                    example_id=raw["example_id"],
                    direction=raw["direction"],
                    selected_angle_degrees=_finite_float(raw, "selected_angle_degrees"),
                    selector_identity_iou=_finite_float(raw, "selector_identity_iou"),
                    selector_selected_iou=_finite_float(raw, "selector_selected_iou"),
                    selector_margin=_finite_float(raw, "selector_margin"),
                    heldout_identity_iou=_finite_float(raw, "heldout_identity_iou"),
                    heldout_selected_iou=_finite_float(raw, "heldout_selected_iou"),
                    delta_iou=_finite_float(raw, "delta_iou"),
                )
            )
    if len(result) != 3_660:
        raise ValueError(
            f"expected 3660 valid true-start cross-fit rows, got {len(result)}"
        )
    return tuple(result)


def select_crossfit_rows(
    rows: Sequence[CrossFitSelectionRow],
) -> Tuple[CrossFitSelectionRow, ...]:
    """Select median positive-transfer strict winners in both directions."""

    selected = []
    for direction in ("object_to_region", "region_to_object"):
        group = tuple(
            row
            for row in rows
            if row.direction == direction
            and row.selector_margin > 0.0
            and row.delta_iou > 0.0
            and row.selector_identity_iou > 0.0
            and row.selector_selected_iou > 0.0
            and row.heldout_identity_iou > 0.0
            and row.heldout_selected_iou > 0.0
        )
        if not group:
            raise ValueError(f"no positive-transfer strict winners for {direction}")
        center = median(tuple(row.delta_iou for row in group))
        selected.append(
            min(
                group,
                key=lambda row: (
                    abs(row.delta_iou - center),
                    row.scene_id,
                    row.example_id,
                ),
            )
        )
    return tuple(selected)


def _common_bounds(warps: Sequence[WarpedGrid]) -> SpatialBounds:
    if not warps:
        raise ValueError("common bounds require at least one warp")
    return SpatialBounds(
        min(0, *(warp.bounds.row_min for warp in warps)),
        max(50, *(warp.bounds.row_max for warp in warps)),
        min(0, *(warp.bounds.col_min for warp in warps)),
        max(50, *(warp.bounds.col_max for warp in warps)),
    )


def embed_warp(warp: WarpedGrid, bounds: SpatialBounds) -> NDArray[np.bool_]:
    """Embed a padded warp without clipping its out-of-frame support."""

    if (
        warp.bounds.row_min < bounds.row_min
        or warp.bounds.row_max > bounds.row_max
        or warp.bounds.col_min < bounds.col_min
        or warp.bounds.col_max > bounds.col_max
    ):
        raise ValueError("warp lies outside the requested common bounds")
    result = np.zeros((warp.grid.shape[0], bounds.rows, bounds.cols), dtype=np.bool_)
    row = warp.bounds.row_min - bounds.row_min
    col = warp.bounds.col_min - bounds.col_min
    result[:, row : row + warp.bounds.rows, col : col + warp.bounds.cols] = warp.grid
    if int(np.count_nonzero(result)) != warp.total_support:
        raise RuntimeError("embedding changed warp support")
    return result


def require_support_conservation(warps: Sequence[WarpedGrid]) -> None:
    """Reject any warp that loses or duplicates category-cell support."""

    if not warps:
        raise ValueError("support conservation requires at least one warp")
    expected = warps[0].input_support
    if any(
        warp.input_support != expected or warp.total_support != expected
        for warp in warps
    ):
        raise ValueError("rotation changed predicted support")


def _embed_target(
    target: NDArray[np.bool_], bounds: SpatialBounds
) -> NDArray[np.bool_]:
    result = np.zeros((target.shape[0], bounds.rows, bounds.cols), dtype=np.bool_)
    row = -bounds.row_min
    col = -bounds.col_min
    result[:, row : row + 50, col : col + 50] = target
    return result


def category_error_codes(
    prediction: NDArray[np.bool_], target: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    """Collapse channel-aware errors to a documented physical-cell display code."""

    if (
        prediction.shape != target.shape
        or prediction.dtype != np.bool_
        or target.dtype != np.bool_
    ):
        raise ValueError("prediction and target must be same-shape boolean grids")
    tp = np.any(prediction & target, axis=0)
    fp = np.any(prediction & ~target, axis=0)
    fn = np.any(~prediction & target, axis=0)
    result = np.zeros(tp.shape, dtype=np.uint8)
    result[tp & ~fp & ~fn] = 1
    result[fp & ~fn] = 2
    result[fn & ~fp] = 3
    result[fp & fn] = 4
    return result


def _setup_map_axis(ax: Axes, bounds: SpatialBounds, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlim(bounds.col_min, bounds.col_max)
    ax.set_ylim(bounds.row_max, bounds.row_min)
    ax.set_aspect("equal")
    ax.add_patch(
        mpatches.Rectangle((0, 0), 50, 50, fill=False, edgecolor="black", linewidth=1.2)
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def _draw_support(
    ax: Axes,
    grid: NDArray[np.bool_],
    bounds: SpatialBounds,
    pivot: Tuple[float, float],
    title: str,
) -> None:
    ax.imshow(
        np.any(grid, axis=0),
        cmap=mcolors.ListedColormap(("#FFFFFF", "#4C78A8")),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=(bounds.col_min, bounds.col_max, bounds.row_max, bounds.row_min),
    )
    _setup_map_axis(ax, bounds, title)
    ax.scatter((pivot[1],), (pivot[0],), marker="x", color="#FF7F0E", s=28, zorder=5)


def _draw_errors(
    ax: Axes,
    prediction: NDArray[np.bool_],
    target: NDArray[np.bool_],
    bounds: SpatialBounds,
    pivot: Tuple[float, float],
    title: str,
) -> None:
    ax.imshow(
        category_error_codes(prediction, target),
        cmap=_ERROR_CMAP,
        vmin=0,
        vmax=4,
        interpolation="nearest",
        extent=(bounds.col_min, bounds.col_max, bounds.row_max, bounds.row_min),
    )
    _setup_map_axis(ax, bounds, title)
    ax.scatter((pivot[1],), (pivot[0],), marker="x", color="#FF7F0E", s=28, zorder=5)


def _draw_angle_bars(
    ax: Axes, scores: Sequence[float], selected_angle: float, title: str
) -> None:
    colors = ("#F58518" if angle == selected_angle else "#BAB0AC" for angle in _ANGLES)
    ax.bar(
        tuple(str(int(angle)) for angle in _ANGLES), tuple(scores), color=tuple(colors)
    )
    ax.set_ylim(0.0, max(0.01, max(scores) * 1.18))
    ax.set_xlabel("Rotation (degrees)")
    ax.set_ylabel("Category-aware IoU")
    ax.set_title(title, fontsize=9)


def _oracle_figure(
    case: EpisodeCase, row: OracleSelectionRow, path: Path
) -> dict[str, object]:
    warps = tuple(
        warp_grid_about_pivot(case.predicted_grid, case.true_start_pivot, angle)
        for angle in _ANGLES
    )
    require_support_conservation(warps)
    scores = tuple(score_warped_grid(warp, case.target_grid) for warp in warps)
    recomputed = tuple(score.iou for score in scores)
    sealed = tuple(iou for _, iou in row.angle_ious)
    if not np.allclose(recomputed, sealed, rtol=0.0, atol=1e-12):
        raise ValueError(f"sealed oracle scores do not replay for {row.example_id}")
    selected_index = _ANGLES.index(row.best_angle_degrees)
    identity, selected = warps[0], warps[selected_index]
    bounds = _common_bounds((identity, selected))
    target = _embed_target(case.target_grid, bounds)
    original = embed_warp(identity, bounds)
    rotated = embed_warp(selected, bounds)

    figure, axes = plt.subplots(3, 3, figsize=(13, 12), squeeze=False)
    folds = ((0, OBJECT_CATEGORIES, "Objects"), (OBJECT_CATEGORIES, 37, "Regions"))
    for axis_row, (start, stop, label) in enumerate(folds):
        _draw_support(
            axes[axis_row, 0],
            target[start:stop],
            bounds,
            case.true_start_pivot,
            f"GT {label} spatial support (categories collapsed)",
        )
        _draw_support(
            axes[axis_row, 1],
            original[start:stop],
            bounds,
            case.true_start_pivot,
            f"Original {label} spatial support (0 deg; categories collapsed)",
        )
        _draw_support(
            axes[axis_row, 2],
            rotated[start:stop],
            bounds,
            case.true_start_pivot,
            f"Oracle-rotated {label} spatial support "
            f"({int(row.best_angle_degrees)} deg; categories collapsed)",
        )
    _draw_errors(
        axes[2, 0],
        original,
        target,
        bounds,
        case.true_start_pivot,
        "Original category-aware error",
    )
    _draw_errors(
        axes[2, 1],
        rotated,
        target,
        bounds,
        case.true_start_pivot,
        "Rotated category-aware error",
    )
    _draw_angle_bars(
        axes[2, 2],
        recomputed,
        row.best_angle_degrees,
        "GT-scored four-angle comparison",
    )
    figure.suptitle(
        f"{row.scene_id} / {row.example_id}: {row.identity_iou:.4f} -> {row.best_iou:.4f} "
        f"(delta {row.delta_iou:+.4f})\nGT-assisted oracle rotation; diagnostic only",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.015,
        "Error colors: green clean TP; red FP without FN; blue FN without FP; purple has both FP and FN. "
        f"Intersection {scores[0].intersection}->{scores[selected_index].intersection}; "
        f"prediction support {scores[0].predicted_support}->{scores[selected_index].predicted_support}; "
        f"out-of-frame {scores[selected_index].out_of_frame_support}.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(path, dpi=160, metadata={"Date": None})
    plt.close(figure)
    return {
        "kind": "all_channel_oracle",
        **asdict(row),
        "common_bounds": asdict(bounds),
        "intersections": [score.intersection for score in scores],
        "unions": [score.union for score in scores],
        "predicted_support": [score.predicted_support for score in scores],
        "out_of_frame_support": [score.out_of_frame_support for score in scores],
    }


def _crossfit_figure(
    case: EpisodeCase, row: CrossFitSelectionRow, path: Path
) -> dict[str, object]:
    family_scores = score_angle_families(
        case.predicted_grid, case.target_grid, case.true_start_pivot, _ANGLES
    )
    direction = Direction(row.direction)
    replay = crossfit_scores(family_scores, direction)
    expected = (
        row.selected_angle_degrees,
        row.selector_identity_iou,
        row.selector_selected_iou,
        row.heldout_identity_iou,
        row.heldout_selected_iou,
    )
    actual = (
        replay.selected_angle_degrees,
        replay.selector_identity_iou,
        replay.selector_selected_iou,
        replay.heldout_identity_iou,
        replay.heldout_selected_iou,
    )
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"sealed cross-fit scores do not replay for {row.example_id}")
    warps = tuple(
        warp_grid_about_pivot(case.predicted_grid, case.true_start_pivot, angle)
        for angle in _ANGLES
    )
    require_support_conservation(warps)
    selected_index = _ANGLES.index(row.selected_angle_degrees)
    bounds = _common_bounds((warps[0], warps[selected_index]))
    target = _embed_target(case.target_grid, bounds)
    original = embed_warp(warps[0], bounds)
    rotated = embed_warp(warps[selected_index], bounds)
    selector_slice = (
        slice(0, OBJECT_CATEGORIES)
        if direction is Direction.OBJECT_TO_REGION
        else slice(OBJECT_CATEGORIES, 37)
    )
    heldout_slice = (
        slice(OBJECT_CATEGORIES, 37)
        if direction is Direction.OBJECT_TO_REGION
        else slice(0, OBJECT_CATEGORIES)
    )
    selector_scores = tuple(
        score.object_score.iou
        if direction is Direction.OBJECT_TO_REGION
        else score.region_score.iou
        for score in family_scores
    )
    heldout_scores = tuple(
        score.region_score.iou
        if direction is Direction.OBJECT_TO_REGION
        else score.object_score.iou
        for score in family_scores
    )
    selector_label = "Objects" if direction is Direction.OBJECT_TO_REGION else "Regions"
    heldout_label = "Regions" if direction is Direction.OBJECT_TO_REGION else "Objects"

    figure, axes = plt.subplots(2, 3, figsize=(13, 8), squeeze=False)
    _draw_errors(
        axes[0, 0],
        original[selector_slice],
        target[selector_slice],
        bounds,
        case.true_start_pivot,
        f"{selector_label} selector: original",
    )
    _draw_errors(
        axes[0, 1],
        rotated[selector_slice],
        target[selector_slice],
        bounds,
        case.true_start_pivot,
        f"{selector_label} selector: selected",
    )
    _draw_angle_bars(
        axes[0, 2],
        selector_scores,
        row.selected_angle_degrees,
        f"{selector_label} selects angle",
    )
    _draw_errors(
        axes[1, 0],
        original[heldout_slice],
        target[heldout_slice],
        bounds,
        case.true_start_pivot,
        f"Held-out {heldout_label}: original",
    )
    _draw_errors(
        axes[1, 1],
        rotated[heldout_slice],
        target[heldout_slice],
        bounds,
        case.true_start_pivot,
        f"Held-out {heldout_label}: same angle",
    )
    _draw_angle_bars(
        axes[1, 2],
        heldout_scores,
        row.selected_angle_degrees,
        f"Held-out {heldout_label} scores",
    )
    figure.suptitle(
        f"{row.scene_id} / {row.example_id}: {selector_label} select {int(row.selected_angle_degrees)} deg; "
        f"held-out {heldout_label} {row.heldout_identity_iou:.4f}->{row.heldout_selected_iou:.4f}\n"
        "Representative among positive-score, positive-transfer strict winners; "
        "GT-assisted diagnostic only",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.92))
    figure.savefig(path, dpi=160, metadata={"Date": None})
    plt.close(figure)

    def serialize_family_score(raster: RasterScore) -> dict[str, object]:
        return {
            "intersection": raster.intersection,
            "union": raster.union,
            "predicted_support": raster.predicted_support,
            "target_support": raster.target_support,
            "in_frame_support": raster.in_frame_support,
            "out_of_frame_support": raster.out_of_frame_support,
            "iou": raster.iou,
        }

    return {
        "kind": "cross_family",
        **asdict(row),
        "common_bounds": asdict(bounds),
        "angles": list(_ANGLES),
        "object_scores": [
            serialize_family_score(score.object_score) for score in family_scores
        ],
        "region_scores": [
            serialize_family_score(score.region_score) for score in family_scores
        ],
    }


def _git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def run_visualization(args: RotationVisualizationArgs) -> dict[str, object]:
    """Validate sealed inputs, replay selected rows, and publish six figures."""

    _verify_sources()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite output directory: {args.output_dir}"
        )
    oracle_rows = select_oracle_rows(
        _read_oracle_rows(_DIAGNOSIS_ROOT / "episodes.csv")
    )
    crossfit_rows = select_crossfit_rows(
        _read_crossfit_rows(_CROSSFIT_ROOT / "crossfit_results.csv")
    )
    if {
        row.best_angle_degrees: (row.scene_id, row.example_id) for row in oracle_rows
    } != _EXPECTED_ORACLE_IDENTITIES:
        raise ValueError("oracle selection differs from the frozen expected identities")
    if {
        row.direction: (row.scene_id, row.example_id) for row in crossfit_rows
    } != _EXPECTED_CROSSFIT_IDENTITIES:
        raise ValueError(
            "cross-fit selection differs from the frozen expected identities"
        )

    loader_args = CrossFitArgs()
    loader_args.quiet = True
    cases, prediction_manifest_path = _load_episode_cases(loader_args)
    by_identity = {(case.scene_id, case.example_id): case for case in cases}
    parent = args.output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=parent))
    try:
        records = []
        for row in oracle_rows:
            filename = f"oracle-{int(row.best_angle_degrees):03d}-{row.example_id}.png"
            payload = _oracle_figure(
                by_identity[(row.scene_id, row.example_id)], row, staging / filename
            )
            records.append({**payload, "figure": filename})
        for row in crossfit_rows:
            filename = (
                f"crossfit-{row.direction.replace('_', '-')}-{row.example_id}.png"
            )
            payload = _crossfit_figure(
                by_identity[(row.scene_id, row.example_id)], row, staging / filename
            )
            records.append({**payload, "figure": filename})
        for record in records:
            figure_path = staging / str(record["figure"])
            record["figure_sha256"] = _sha256(figure_path)
            if figure_path.stat().st_size == 0:
                raise RuntimeError(f"renderer wrote an empty figure: {figure_path}")
        manifest: dict[str, object] = {
            "schema_version": "llm-grid-rotation-visualizations-v1",
            "git_commit": _git_commit(),
            "oracle_only_limitation": (
                "Angles use ground-truth raster cells; figures are diagnostic and not deployable performance."
            ),
            "crossfit_selection_limitation": (
                "Cross-family examples are median representatives among positive-score, "
                "positive-transfer strict winners, not all episodes."
            ),
            "source_sha256": {
                str(path): digest for path, digest in _SOURCE_HASHES.items()
            },
            "prediction_manifest": str(prediction_manifest_path),
            "prediction_manifest_sha256": _sha256(prediction_manifest_path),
            "selection": {
                "oracle": "strict support-bearing winner; 0 deg median identity IoU, nonzero median delta IoU; lexical tie",
                "crossfit": "true-start positive-score, positive-transfer strict winner; "
                "median held-out delta IoU; lexical tie",
            },
            "error_codes": {
                "0": "background",
                "1": "clean TP",
                "2": "has FP and no FN",
                "3": "has FN and no FP",
                "4": "has both FP and FN",
            },
            "records": records,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if not args.quiet:
        print(f"Wrote six sealed rotation figures to {args.output_dir}")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    args = RotationVisualizationArgs(underscores_to_dashes=True).parse_args(argv)
    return run_visualization(args)


if __name__ == "__main__":
    main()
