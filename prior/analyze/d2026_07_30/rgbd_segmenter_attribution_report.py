"""Deterministic quantitative report and figures for the sealed ESANet result."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, Union, cast

import numpy as np
from tap import Tap

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from prior.constants import MAPPED_OBJECT_COLORS, MAPPED_OBJECT_NAMES
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    NYU40_MAPPING_SHA256,
    PRIMARY_CATEGORY_INDICES,
    RAW_INDEX_SHA256,
    MappingEntry,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    AcceptedPrediction,
    ObservationPackageRow,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import RawFrameArrays
from prior.analyze.d2026_07_30.rgbd_segmenter_attribution import (
    CategoryCellRow,
    CategoryCellStatus,
    PixelContributorLedger,
    SemanticProjectionGeometry,
    _load_accepted_inputs,
    _repository_root,
    run_contributor_conservation,
    run_geometry_leakage_proof,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_esanet_benchmark import _git_commit
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames import _rename_noreplace

plt.switch_backend("Agg")

_OUTPUT_ROOT = (
    "data/rgbd_segmenter_benchmark/experiments/"
    "esanet-r34-nbt1d-scenenet-attribution-v1"
)
_DEPTH_BIN_NAMES = (
    "invalid",
    "(0,1]",
    "(1,2]",
    "(2,4]",
    "(4,6]",
    "(6,10)",
    "saturated",
)
_PROJECTION_STATE_NAMES = (
    "invalid_depth",
    "contributing",
    "out_of_bounds",
    "saturated",
)
_SELECTED_SCENE_RANKS = (0, 2, 5, 8, 10)
_DETAIL_VIEWS = (0, 3, 6, 9)
_EXPECTED_SELECTION = (
    (0, "2azQ1b91cZZ", "bd81990a0f88c2c285ac"),
    (9, "EU6Fwq7SyZv", "0f7d45c64591fd4f0a4f"),
    (28, "X7HyMhZNoso", "e41f277cbd7a7a13bbe3"),
    (41, "pLe4wQe7qrG", "5a4ba335d68aad27c13d"),
    (45, "zsNo4HB9uLZ", "26275e12b0798079ca3b"),
)
_OVERLAY_ALPHA = 0.60
_CANONICAL_RGB = np.asarray(
    [mcolors.to_rgb(color) for color in MAPPED_OBJECT_COLORS],
    dtype=np.float64,
)
_PIXEL_COLUMNS = (
    "ordinal",
    "observation_id",
    "scene_id",
    "view",
    "yaw_degrees",
    "depth_bin",
    "projection_state",
    "gt_canonical_index",
    "source_nyu40_index",
    "mapped_canonical_index",
    "mapping_kind",
    "pixel_count",
)
_CELL_COLUMNS = (
    "ordinal",
    "observation_id",
    "scene_id",
    "category",
    "row",
    "column",
    "status",
    "effect",
    "gt_pixel_support",
    "predicted_pixel_support",
    "correct_pixel_support",
    "gt_view_support",
    "predicted_view_support",
)
_CONTRIBUTOR_COLUMNS = (
    "ordinal",
    "observation_id",
    "scene_id",
    "category",
    "row",
    "column",
    "status",
    "effect",
    "view",
    "yaw_degrees",
    "depth_bin",
    "gt_canonical_index",
    "source_nyu40_index",
    "mapped_canonical_index",
    "mapping_kind",
    "supports_gt",
    "supports_prediction",
    "pixel_count",
)
_METRIC_COLUMNS = (
    "scope",
    "stratum_type",
    "stratum_value",
    "category",
    "category_name",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "iou",
    "f1",
)
_CsvValue = Union[str, int]
_CsvRow = Tuple[_CsvValue, ...]

__all__ = (
    "AttributionPublicationReport",
    "SegmenterAttributionReportArgs",
    "aggregate_pixel_rows",
    "main",
    "publish_attribution_report",
)


class SegmenterAttributionReportArgs(Tap):
    """Publish the frozen P7 attribution report into ignored experiment storage."""

    output: Path = Path(_OUTPUT_ROOT)


@dataclass(frozen=True)
class AttributionPublicationReport:
    schema_version: int
    observation_count: int
    view_count: int
    pixel_count: int
    contributing_pixel_count: int
    category_cell_count: int
    primary_tp_cell_count: int
    primary_fp_cell_count: int
    primary_fn_cell_count: int
    pixel_rows_sha256: str
    category_cells_sha256: str
    summary_sha256: str

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(asdict(self))


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(columns: Sequence[str], rows: Iterable[Sequence[object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _projection_states(ledger: PixelContributorLedger, view: int) -> np.ndarray:
    depth_bin = ledger.depth_bin[view]
    states = np.full(depth_bin.shape, 2, dtype=np.uint8)
    states[ledger.endpoint_valid[view] & ledger.in_bounds[view]] = 1
    states[depth_bin == 0] = 0
    states[depth_bin == 6] = 3
    return states


def aggregate_pixel_rows(
    ledger: PixelContributorLedger,
    mapping: Tuple[MappingEntry, ...],
) -> Tuple[_CsvRow, ...]:
    """Aggregate every stored pixel without losing the frozen attribution strata."""

    if len(mapping) != 40:
        raise ValueError("attribution requires the frozen 40-row mapping")
    mapped_lookup = np.asarray(
        tuple(
            -1 if entry.canonical_index is None else entry.canonical_index
            for entry in mapping
        ),
        dtype="<i2",
    )
    if not np.array_equal(mapped_lookup[ledger.source_labels], ledger.mapped_labels):
        raise ValueError("stored mapped labels differ from the frozen mapping")
    rows: list[_CsvRow] = []
    for view in range(12):
        depth_bin = ledger.depth_bin[view].ravel().astype(np.int64)
        state = _projection_states(ledger, view).ravel().astype(np.int64)
        gt = ledger.gt_labels[view].ravel().astype(np.int64)
        source = ledger.source_labels[view].ravel().astype(np.int64)
        mapped = ledger.mapped_labels[view].ravel().astype(np.int64)
        key = depth_bin
        key = key * len(_PROJECTION_STATE_NAMES) + state
        key = key * 28 + (gt + 1)
        key = key * 40 + source
        key = key * 28 + (mapped + 1)
        keys, counts = np.unique(key, return_counts=True)
        for encoded, count in zip(keys.tolist(), counts.tolist()):
            mapped_index = encoded % 28 - 1
            encoded //= 28
            source_index = encoded % 40
            encoded //= 40
            gt_index = encoded % 28 - 1
            encoded //= 28
            state_index = encoded % len(_PROJECTION_STATE_NAMES)
            depth_index = encoded // len(_PROJECTION_STATE_NAMES)
            rows.append(
                (
                    ledger.ordinal,
                    ledger.observation_id,
                    ledger.scene_id,
                    view,
                    int(ledger.sensor_yaw_degrees[view]),
                    _DEPTH_BIN_NAMES[depth_index],
                    _PROJECTION_STATE_NAMES[state_index],
                    gt_index,
                    source_index,
                    mapped_index,
                    mapping[source_index].kind.value,
                    count,
                )
            )
    return tuple(rows)


def aggregate_cell_contributors(
    ledger: PixelContributorLedger,
    category_cells: Sequence[CategoryCellRow],
    mapping: Tuple[MappingEntry, ...],
) -> Tuple[_CsvRow, ...]:
    """List lossless grouped contributors for every primary union category-cell."""

    projectable = (ledger.endpoint_valid & ledger.in_bounds).ravel()
    spatial = (
        ledger.target_cell_rc[..., 0].ravel().astype(np.int64) * 50
        + ledger.target_cell_rc[..., 1].ravel()
    )
    flat_indices = np.flatnonzero(projectable)
    order = np.argsort(spatial[flat_indices], kind="stable")
    sorted_spatial = spatial[flat_indices][order]
    sorted_indices = flat_indices[order]
    gt = ledger.gt_labels.ravel()
    source = ledger.source_labels.ravel()
    mapped = ledger.mapped_labels.ravel()
    depth_bin = ledger.depth_bin.ravel()
    output: list[_CsvRow] = []
    for cell in category_cells:
        start = len(output)
        spatial_key = cell.row * 50 + cell.column
        lower = int(np.searchsorted(sorted_spatial, spatial_key, side="left"))
        upper = int(np.searchsorted(sorted_spatial, spatial_key, side="right"))
        indices = sorted_indices[lower:upper]
        indices = indices[
            (gt[indices] == cell.category) | (mapped[indices] == cell.category)
        ]
        if not indices.size:
            raise ValueError("category cell has no relevant contributor")
        values = np.column_stack(
            (
                indices // (256 * 256),
                depth_bin[indices],
                gt[indices],
                source[indices],
                mapped[indices],
                gt[indices] == cell.category,
                mapped[indices] == cell.category,
            )
        )
        groups, counts = np.unique(values, axis=0, return_counts=True)
        for group, count in zip(groups, counts):
            view = int(group[0])
            source_index = int(group[3])
            output.append(
                (
                    ledger.ordinal,
                    ledger.observation_id,
                    ledger.scene_id,
                    cell.category,
                    cell.row,
                    cell.column,
                    cell.status.value,
                    cell.effect.value,
                    view,
                    int(ledger.sensor_yaw_degrees[view]),
                    _DEPTH_BIN_NAMES[int(group[1])],
                    int(group[2]),
                    source_index,
                    int(group[4]),
                    mapping[source_index].kind.value,
                    int(group[5]),
                    int(group[6]),
                    int(count),
                )
            )
        if (
            sum(
                int(row[17]) * int(row[15])
                for row in output[start:]
            )
            != cell.gt_pixel_support
            or sum(
                int(row[17]) * int(row[16])
                for row in output[start:]
            )
            != cell.predicted_pixel_support
            or sum(
                int(row[17]) * int(row[15]) * int(row[16])
                for row in output[start:]
            )
            != cell.correct_pixel_support
            or len({int(row[8]) for row in output[start:] if int(row[15])})
            != cell.gt_view_support
            or len({int(row[8]) for row in output[start:] if int(row[16])})
            != cell.predicted_view_support
        ):
            raise ValueError("grouped contributor support does not conserve")
    return tuple(output)


def _ratio(numerator: int, denominator: int) -> str:
    return "" if denominator == 0 else f"{numerator / denominator:.8f}"


def _metric_values(tp: int, fp: int, fn: int) -> Tuple[_CsvValue, ...]:
    return (
        tp,
        fp,
        fn,
        _ratio(tp, tp + fp),
        _ratio(tp, tp + fn),
        _ratio(tp, tp + fp + fn),
        _ratio(2 * tp, 2 * tp + fp + fn),
    )


def _pixel_metric_rows(pixel_rows: Sequence[_CsvRow]) -> Tuple[_CsvRow, ...]:
    strata: dict[Tuple[str, str, str, int], list[int]] = {}
    for row in pixel_rows:
        projection_state = str(row[6])
        scopes = ("all_stored",)
        if projection_state == "contributing":
            scopes += ("projectable_in_bounds",)
        dimensions = (
            ("overall", "all"),
            ("scene", str(row[2])),
            ("yaw", str(row[4])),
            ("depth_bin", str(row[5])),
            ("mapping_kind", str(row[10])),
        )
        gt = int(row[7])
        mapped = int(row[9])
        count = int(row[11])
        for scope in scopes:
            for stratum_type, stratum_value in dimensions:
                for category in PRIMARY_CATEGORY_INDICES:
                    values = strata.setdefault(
                        (scope, stratum_type, stratum_value, category),
                        [0, 0, 0],
                    )
                    values[0] += count * int(gt == category and mapped == category)
                    values[1] += count * int(gt != category and mapped == category)
                    values[2] += count * int(gt == category and mapped != category)
    return tuple(
        (
            scope,
            stratum_type,
            stratum_value,
            category,
            MAPPED_OBJECT_NAMES[category],
            *_metric_values(*values),
        )
        for (scope, stratum_type, stratum_value, category), values in sorted(
            strata.items()
        )
    )


def _grid_metric_rows(cell_rows: Sequence[_CsvRow]) -> Tuple[_CsvRow, ...]:
    strata: dict[Tuple[str, str, int], list[int]] = {}
    for row in cell_rows:
        category = int(row[3])
        status = str(row[6])
        for stratum_type, stratum_value in (
            ("overall", "all"),
            ("scene", str(row[2])),
        ):
            values = strata.setdefault(
                (stratum_type, stratum_value, category),
                [0, 0, 0],
            )
            values[0] += int(status == "TP")
            values[1] += int(status == "FP")
            values[2] += int(status == "FN")
    return tuple(
        (
            "primary_grid",
            stratum_type,
            stratum_value,
            category,
            MAPPED_OBJECT_NAMES[category],
            *_metric_values(*values),
        )
        for (stratum_type, stratum_value, category), values in sorted(strata.items())
    )


def _ledger_for(
    *,
    arrays: RawFrameArrays,
    observation: ObservationPackageRow,
    accepted_prediction: AcceptedPrediction,
) -> PixelContributorLedger:
    # These values originate from strict typed package readers. Keeping construction
    # here makes the report runner consume exactly the same ledger as P7.2.
    return PixelContributorLedger.build(
        ordinal=observation.ordinal,
        observation_id=observation.observation_id,
        scene_id=observation.scene_id,
        sensor_yaw_degrees=arrays.sensor_yaw_degrees,
        geometry=SemanticProjectionGeometry.from_raw_frame_arrays(arrays),
        gt_labels=arrays.object_categories,
        source_labels=accepted_prediction.prediction.source_labels,
        mapped_labels=accepted_prediction.prediction.mapped_labels,
    )


def _summary(
    *,
    pixel_rows: Sequence[_CsvRow],
    cell_rows: Sequence[_CsvRow],
) -> Mapping[str, object]:
    projection_counts: Counter[str] = Counter()
    mapping_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    effect_counts: Counter[str] = Counter()
    contributing_confusion: Counter[Tuple[int, int]] = Counter()
    primary = set(PRIMARY_CATEGORY_INDICES)
    for row in pixel_rows:
        count = cast(int, row[11])
        projection_counts[str(row[6])] += count
        mapping_counts[str(row[10])] += count
        depth_counts[str(row[5])] += count
        if row[6] == "contributing" and cast(int, row[7]) in primary:
            contributing_confusion[
                (cast(int, row[7]), cast(int, row[9]))
            ] += count
    for row in cell_rows:
        status_counts[str(row[6])] += 1
        effect_counts[str(row[7])] += 1
    top_confusions = sorted(
        (
            {
                "gt_canonical_index": gt,
                "mapped_canonical_index": mapped,
                "pixel_count": count,
            }
            for (gt, mapped), count in contributing_confusion.items()
            if gt != mapped
        ),
        key=lambda row: (
            -int(row["pixel_count"]),
            int(row["gt_canonical_index"]),
            int(row["mapped_canonical_index"]),
        ),
    )[:25]
    return {
        "schema_version": 1,
        "population": {
            "pixel_rows": "all stored pixels",
            "semantic_confusion": (
                "in-bounds, non-saturated endpoint pixels with primary GT"
            ),
            "category_cells": "primary target/prediction union",
        },
        "depth_bin_pixel_counts": dict(sorted(depth_counts.items())),
        "projection_state_pixel_counts": dict(sorted(projection_counts.items())),
        "mapping_kind_pixel_counts": dict(sorted(mapping_counts.items())),
        "category_cell_status_counts": dict(sorted(status_counts.items())),
        "category_cell_effect_counts": dict(sorted(effect_counts.items())),
        "top_primary_gt_mapped_pixel_errors": top_confusions,
    }


def _pascal_palette(size: int) -> np.ndarray:
    palette = np.zeros((size, 3), dtype=np.uint8)
    for index in range(size):
        value = index
        bit = 0
        while value:
            palette[index, 0] |= ((value >> 0) & 1) << (7 - bit)
            palette[index, 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[index, 2] |= ((value >> 2) & 1) << (7 - bit)
            bit += 1
            value >>= 3
    return palette


_RAW_RGB = _pascal_palette(41)[1:]


def _semantic_overlay(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rendered = rgb.astype(np.float64) / 255
    visible = labels >= 0
    rendered[visible] = (
        (1 - _OVERLAY_ALPHA) * rendered[visible]
        + _OVERLAY_ALPHA * _CANONICAL_RGB[labels[visible]]
    )
    return rendered


def _raw_labels_rgb(labels: np.ndarray) -> np.ndarray:
    return _RAW_RGB[labels]


def _depth_rgba(depth_m: np.ndarray) -> np.ndarray:
    rendered = plt.get_cmap("viridis")(np.clip(depth_m, 0, 10) / 10)
    rendered[depth_m == 0] = (0.8, 0.0, 0.8, 1.0)
    rendered[depth_m == 10] = (1.0, 1.0, 1.0, 1.0)
    return rendered


def _disagreement_rgba(gt: np.ndarray, mapped: np.ndarray) -> np.ndarray:
    primary_lookup = np.zeros(27, dtype=np.bool_)
    primary_lookup[np.asarray(PRIMARY_CATEGORY_INDICES)] = True
    gt_primary = (gt >= 0) & primary_lookup[np.maximum(gt, 0)]
    mapped_primary = (mapped >= 0) & primary_lookup[np.maximum(mapped, 0)]
    same = gt_primary & mapped_primary & (gt == mapped)
    substitution = gt_primary & mapped_primary & (gt != mapped)
    fn = gt_primary & ~mapped_primary
    fp = ~gt_primary & mapped_primary
    rgba = np.zeros((*gt.shape, 4), dtype=np.float64)
    rgba[same] = (0.0, 0.65, 0.0, 0.72)
    rgba[fn] = (0.1, 0.3, 1.0, 0.78)
    rgba[fp] = (1.0, 0.1, 0.1, 0.78)
    rgba[substitution] = (0.65, 0.1, 0.8, 0.82)
    return rgba


def _png_bytes(figure: Figure) -> bytes:
    stream = io.BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=120,
        metadata={"Date": None, "Software": "ETP-R1 P7"},
    )
    plt.close(figure)
    return stream.getvalue()


def _contact_figure(arrays: RawFrameArrays) -> bytes:
    figure, axes = plt.subplots(
        3,
        4,
        figsize=(12, 9),
        constrained_layout=True,
    )
    for view, axis in enumerate(axes.ravel()):
        axis.imshow(arrays.rgb[view], interpolation="nearest")
        axis.set_title(f"view {view} · yaw {int(arrays.sensor_yaw_degrees[view])}°")
        axis.set_axis_off()
    figure.suptitle("All 12 stored RGB views")
    return _png_bytes(figure)


def _detail_figure(
    arrays: RawFrameArrays,
    source_labels: np.ndarray,
    mapped_labels: np.ndarray,
    mapping: Tuple[MappingEntry, ...],
) -> bytes:
    figure, axes = plt.subplots(
        len(_DETAIL_VIEWS),
        6,
        figsize=(18, 14),
    )
    column_titles = (
        "RGB",
        "depth [0,10] m",
        "RGB + GT canonical",
        "RGB + mapped ESANet",
        "raw NYU40 argmax",
        "primary disagreement",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)
    for row, view in enumerate(_DETAIL_VIEWS):
        rgb = arrays.rgb[view]
        values = (
            rgb,
            _depth_rgba(arrays.depth_m[view]),
            _semantic_overlay(rgb, arrays.object_categories[view]),
            _semantic_overlay(rgb, mapped_labels[view]),
            _raw_labels_rgb(source_labels[view]),
            _disagreement_rgba(
                arrays.object_categories[view],
                mapped_labels[view],
            ),
        )
        for column, value in enumerate(values):
            axes[row, column].imshow(value, interpolation="nearest")
            axes[row, column].set_axis_off()
        axes[row, 0].text(
            0.02,
            0.96,
            f"view {view} · yaw {int(arrays.sensor_yaw_degrees[view])}°",
            color="white",
            fontsize=8,
            va="top",
            transform=axes[row, 0].transAxes,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
    figure.suptitle(
        "Invalid depth = magenta; saturated 10 m = white; "
        "TP/FN/FP/substitution = green/blue/red/purple"
    )
    canonical_indices = sorted(
        {
            int(value)
            for view in _DETAIL_VIEWS
            for labels in (
                arrays.object_categories[view],
                mapped_labels[view],
            )
            for value in np.unique(labels)
            if value >= 0
        }
    )
    source_indices = sorted(
        {
            int(value)
            for view in _DETAIL_VIEWS
            for value in np.unique(source_labels[view])
        }
    )
    handles = [
        mpatches.Patch(
            color=_CANONICAL_RGB[index],
            label=f"C{index} {MAPPED_OBJECT_NAMES[index]}",
        )
        for index in canonical_indices
    ]
    handles.extend(
        mpatches.Patch(
            color=_RAW_RGB[index].astype(np.float64) / 255,
            label=f"N{index} {mapping[index].source_name}",
        )
        for index in source_indices
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=8,
        fontsize=6,
        title="Active canonical (C) and raw NYU40 argmax (N) colors",
        title_fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.96))
    return _png_bytes(figure)


def _composite_grid(mask: np.ndarray) -> np.ndarray:
    colors = _CANONICAL_RGB[np.asarray(PRIMARY_CATEGORY_INDICES)]
    counts = np.count_nonzero(mask, axis=0)
    summed = np.einsum("chw,cd->hwd", mask, colors)
    rendered = np.ones((*counts.shape, 4), dtype=np.float64)
    occupied = counts > 0
    rendered[occupied, :3] = summed[occupied] / counts[occupied, None]
    rendered[~occupied, :3] = 0.08
    return rendered


def _rotate_forward(rotation_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = rotation_xyzw
    vector = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    q = np.asarray((x, y, z), dtype=np.float64)
    return (
        2 * np.dot(q, vector) * q
        + (w * w - np.dot(q, q)) * vector
        + 2 * w * np.cross(q, vector)
    )


def _annotate_grid_axis(axis: Axes, arrays: RawFrameArrays) -> None:
    start_rc = (
        arrays.start_position[[0, 2]] - arrays.target_origin_xz
    ) / 0.5
    forward = _rotate_forward(arrays.start_rotation_xyzw)
    axis.arrow(
        start_rc[1],
        start_rc[0],
        forward[2] * 3,
        forward[0] * 3,
        color="white",
        width=0.12,
        head_width=0.9,
        length_includes_head=True,
    )
    for position, rotation in zip(
        arrays.sensor_positions,
        arrays.sensor_rotations_xyzw,
    ):
        sensor_rc = (position[[0, 2]] - arrays.target_origin_xz) / 0.5
        ray = _rotate_forward(rotation)
        axis.arrow(
            sensor_rc[1],
            sensor_rc[0],
            ray[2] * 1.5,
            ray[0] * 1.5,
            color="cyan",
            width=0.04,
            head_width=0.45,
            alpha=0.5,
            length_includes_head=True,
        )
    axis.plot((1, 3), (48, 48), color="white", linewidth=3)
    axis.text(1, 47, "1 m", color="white", fontsize=7)
    axis.text(0.5, 1.5, "target origin", color="white", fontsize=6)
    axis.set_xlim(-0.5, 49.5)
    axis.set_ylim(49.5, -0.5)
    axis.set_xlabel("column = world z")
    axis.set_ylabel("row = world x")


def _grid_figure(
    arrays: RawFrameArrays,
    target: np.ndarray,
    prediction: np.ndarray,
) -> bytes:
    indices = np.asarray(PRIMARY_CATEGORY_INDICES)
    target_primary = target[indices]
    prediction_primary = prediction[indices]
    masks = (
        target_primary,
        prediction_primary,
        target_primary & prediction_primary,
        prediction_primary & ~target_primary,
        target_primary & ~prediction_primary,
    )
    titles = ("GT", "prediction", "TP", "FP", "FN")
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(18, 8),
        constrained_layout=True,
    )
    for column, (title, mask) in enumerate(zip(titles, masks)):
        axes[0, column].imshow(_composite_grid(mask), interpolation="nearest")
        axes[0, column].set_title(f"{title}: canonical color composite")
        image = axes[1, column].imshow(
            np.count_nonzero(mask, axis=0),
            interpolation="nearest",
            vmin=0,
            vmax=len(PRIMARY_CATEGORY_INDICES),
            cmap="magma",
        )
        axes[1, column].set_title(f"{title}: active category count")
        _annotate_grid_axis(axes[0, column], arrays)
        _annotate_grid_axis(axes[1, column], arrays)
        figure.colorbar(image, ax=axes[1, column], fraction=0.046)
    figure.suptitle(
        "50×50 grid · averaged canonical colors expose co-occupancy via count panels"
    )
    return _png_bytes(figure)


def _selected_ordinals(observations: Sequence[ObservationPackageRow]) -> Tuple[int, ...]:
    scene_ids = sorted({row.scene_id for row in observations})
    selected = tuple(
        min(
            row.ordinal
            for row in observations
            if row.scene_id == scene_ids[rank]
        )
        for rank in _SELECTED_SCENE_RANKS
    )
    identities = tuple(
        (
            observations[ordinal].ordinal,
            observations[ordinal].scene_id,
            observations[ordinal].observation_id,
        )
        for ordinal in selected
    )
    if identities != _EXPECTED_SELECTION:
        raise ValueError("frozen figure selection identities differ")
    return selected


def _publish_blobs(output: Path, blobs: Mapping[str, bytes]) -> None:
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() or staging.exists():
        raise ValueError("attribution output and staging paths must be absent")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    published = False
    try:
        for path, data in sorted(blobs.items()):
            destination = staging / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        for path, expected in blobs.items():
            actual = (staging / path).read_bytes()
            if actual != expected:
                raise ValueError(f"staged attribution file differs: {path}")
        descriptor = os.open(
            staging,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_descriptor = os.open(
            output.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            _rename_noreplace(
                parent_descriptor,
                staging.name,
                output.name,
            )
        finally:
            os.close(parent_descriptor)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def publish_attribution_report(
    output: Path,
) -> AttributionPublicationReport:
    """Strictly load sealed inputs and publish canonical attribution tables once."""

    if not isinstance(output, Path) or output.exists():
        raise ValueError("attribution output must be a new Path")
    root = _repository_root()
    producer_commit = _git_commit(root)
    leakage_report = run_geometry_leakage_proof()
    conservation_report = run_contributor_conservation()
    raw_values, _, accepted = _load_accepted_inputs(root)
    mapping = _mapping_tuple(root)
    selected_ordinals = _selected_ordinals(accepted.observations)
    pixel_rows: list[_CsvRow] = []
    cell_rows: list[_CsvRow] = []
    contributor_rows: list[_CsvRow] = []
    figure_data: dict[str, bytes] = {}
    for arrays, observation, accepted_prediction in zip(
        raw_values,
        accepted.observations,
        accepted.predictions,
    ):
        ledger = _ledger_for(
            arrays=arrays,
            observation=observation,
            accepted_prediction=accepted_prediction,
        )
        pixel_rows.extend(aggregate_pixel_rows(ledger, mapping))
        target, prediction = ledger.rebuild_grids()
        category_values = ledger.category_cells(
            target=target,
            prediction=prediction,
        )
        if observation.ordinal in selected_ordinals:
            basename = (
                f"{observation.ordinal:02d}-{observation.scene_id}-"
                f"{observation.observation_id}"
            )
            figure_data[f"figures/{basename}-contact.png"] = _contact_figure(arrays)
            figure_data[f"figures/{basename}-views.png"] = _detail_figure(
                arrays,
                accepted_prediction.prediction.source_labels,
                accepted_prediction.prediction.mapped_labels,
                mapping,
            )
            figure_data[f"figures/{basename}-grid.png"] = _grid_figure(
                arrays,
                target,
                prediction,
            )
        cell_rows.extend(
            (
                observation.ordinal,
                observation.observation_id,
                observation.scene_id,
                row.category,
                row.row,
                row.column,
                row.status.value,
                row.effect.value,
                row.gt_pixel_support,
                row.predicted_pixel_support,
                row.correct_pixel_support,
                row.gt_view_support,
                row.predicted_view_support,
            )
            for row in category_values
        )
        contributor_rows.extend(
            aggregate_cell_contributors(
                ledger,
                category_values,
                mapping,
            )
        )
    pixel_data = _csv_bytes(_PIXEL_COLUMNS, pixel_rows)
    cell_data = _csv_bytes(_CELL_COLUMNS, cell_rows)
    contributor_data = _csv_bytes(_CONTRIBUTOR_COLUMNS, contributor_rows)
    pixel_metrics_data = _csv_bytes(
        _METRIC_COLUMNS,
        _pixel_metric_rows(pixel_rows),
    )
    grid_metrics_data = _csv_bytes(
        _METRIC_COLUMNS,
        _grid_metric_rows(cell_rows),
    )
    summary_value = dict(_summary(pixel_rows=pixel_rows, cell_rows=cell_rows))
    summary_value["visualization"] = {
        "selected_scene_ranks": list(_SELECTED_SCENE_RANKS),
        "selected_ordinals": list(selected_ordinals),
        "detail_views": list(_DETAIL_VIEWS),
        "semantic_overlay_alpha": _OVERLAY_ALPHA,
        "depth_range_m": [0, 10],
        "pixel_metric_is_non_gating": True,
        "grid_yaw_depth_counts_are_nonadditive_incidence": True,
        "claims_nyu40_accuracy": False,
        "claims_calibration": False,
    }
    summary_data = _canonical_json_bytes(summary_value)
    status = Counter(str(row[6]) for row in cell_rows)
    report = AttributionPublicationReport(
        schema_version=1,
        observation_count=len(raw_values),
        view_count=len(raw_values) * 12,
        pixel_count=len(raw_values) * 12 * 256 * 256,
        contributing_pixel_count=sum(
            cast(int, row[11]) for row in pixel_rows if row[6] == "contributing"
        ),
        category_cell_count=len(cell_rows),
        primary_tp_cell_count=status[CategoryCellStatus.TP.value],
        primary_fp_cell_count=status[CategoryCellStatus.FP.value],
        primary_fn_cell_count=status[CategoryCellStatus.FN.value],
        pixel_rows_sha256=hashlib.sha256(pixel_data).hexdigest(),
        category_cells_sha256=hashlib.sha256(cell_data).hexdigest(),
        summary_sha256=hashlib.sha256(summary_data).hexdigest(),
    )
    if (
        report.pixel_count != 39_321_600
        or report.category_cell_count != 10_081
        or len(figure_data) != 15
        or (
            report.primary_tp_cell_count,
            report.primary_fp_cell_count,
            report.primary_fn_cell_count,
        )
        != (1222, 3961, 4898)
    ):
        raise ValueError("attribution publication does not conserve sealed inputs")
    blobs = {
        "pixel_attribution.csv": pixel_data,
        "pixel_metrics.csv": pixel_metrics_data,
        "grid_metrics.csv": grid_metrics_data,
        "category_cells.csv": cell_data,
        "cell_contributors.csv": contributor_data,
        "summary.json": summary_data,
        "report.json": report.canonical_bytes(),
        **figure_data,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "esanet-r34-nbt1d-scenenet-attribution-v1",
        "producer_git_commit": producer_commit,
        "sealed_evidence": {
            "raw_index_sha256": RAW_INDEX_SHA256,
            "nyu40_mapping_sha256": NYU40_MAPPING_SHA256,
            "candidate_manifest_sha256": leakage_report.candidate_manifest_sha256,
            "candidate_tree_sha256": leakage_report.candidate_tree_sha256,
            "p53_attestation_sha256": leakage_report.p53_attestation_sha256,
            "benchmark_attestation_sha256": (
                leakage_report.benchmark_attestation_sha256
            ),
            "p6_report_sha256": leakage_report.p6_report_sha256,
            "p6_pair_tree_sha256": leakage_report.p6_pair_tree_sha256,
            "p7_geometry_report": asdict(leakage_report),
            "p7_geometry_report_sha256": hashlib.sha256(
                leakage_report.canonical_bytes()
            ).hexdigest(),
            "p7_conservation_report": asdict(conservation_report),
            "p7_conservation_report_sha256": hashlib.sha256(
                conservation_report.canonical_bytes()
            ).hexdigest(),
        },
        "render_contract": {
            "selected_scene_ranks": list(_SELECTED_SCENE_RANKS),
            "selected_identities": [list(value) for value in _EXPECTED_SELECTION],
            "detail_views": list(_DETAIL_VIEWS),
            "overlay_alpha": _OVERLAY_ALPHA,
            "depth_range_m": [0, 10],
            "canonical_palette_sha256": hashlib.sha256(
                np.rint(_CANONICAL_RGB * 255).astype(np.uint8).tobytes()
            ).hexdigest(),
            "raw_nyu40_palette_sha256": hashlib.sha256(
                _RAW_RGB.tobytes()
            ).hexdigest(),
        },
        "files": [
            {
                "path": path,
                "byte_length": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for path, data in sorted(blobs.items())
        ],
    }
    blobs["manifest.json"] = _canonical_json_bytes(manifest)
    _publish_blobs(output, blobs)
    return report


def _mapping_tuple(root: Path) -> Tuple[MappingEntry, ...]:
    from prior.analyze.d2026_07_30.rgbd_semantic_projector import _mapping_authority

    return _mapping_authority(root).mapping


def main(argv: Sequence[str] | None = None) -> AttributionPublicationReport:
    args = SegmenterAttributionReportArgs(
        underscores_to_dashes=True
    ).parse_args(argv)
    report = publish_attribution_report(args.output)
    print(report.canonical_bytes().decode("utf-8"), end="")
    return report


if __name__ == "__main__":
    main()
