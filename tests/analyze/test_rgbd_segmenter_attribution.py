from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    RawFrameArrays,
)
from prior.analyze.d2026_07_30.rgbd_segmenter_attribution import (
    CategoryCellEffect,
    CategoryCellStatus,
    DepthBin,
    PixelContributorLedger,
    SemanticProjectionGeometry,
)
from prior.analyze.d2026_07_30.rgbd_semantic_projector import (
    project_mapped_labels_semantic_only,
)


def _raw_arrays() -> RawFrameArrays:
    return RawFrameArrays(
        schema_version=np.array(1, dtype="<i8"),
        rgb=np.zeros((12, 256, 256, 3), dtype=np.uint8),
        depth_m=np.zeros((12, 256, 256), dtype="<f4"),
        object_categories=np.full((12, 256, 256), -1, dtype="<i2"),
        region_categories=np.full((12, 256, 256), -1, dtype="<i2"),
        sensor_positions=np.zeros((12, 3), dtype="<f8"),
        sensor_rotations_xyzw=np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype="<f8"),
            (12, 1),
        ),
        sensor_yaw_degrees=np.arange(0, 360, 30, dtype="<i2"),
        sensor_hfov_degrees=np.array(90.0, dtype="<f8"),
        sensor_position_relative=np.array([0.0, 1.25, 0.0], dtype="<f8"),
        start_position=np.zeros(3, dtype="<f8"),
        start_rotation_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype="<f8"),
        target_origin_xz=np.array([-12.5, -12.5], dtype="<f8"),
        ego_observed_mask=np.zeros((50, 50), dtype=np.bool_),
        ego_free_mask=np.zeros((50, 50), dtype=np.bool_),
        target_observed_mask=np.zeros((50, 50), dtype=np.bool_),
        target_free_mask=np.zeros((50, 50), dtype=np.bool_),
    )


def _poison(arrays: RawFrameArrays) -> RawFrameArrays:
    return replace(
        arrays,
        object_categories=np.zeros((12, 256, 256), dtype="<i2"),
        region_categories=np.zeros((12, 256, 256), dtype="<i2"),
        ego_observed_mask=np.ones((50, 50), dtype=np.bool_),
        ego_free_mask=np.ones((50, 50), dtype=np.bool_),
        target_observed_mask=np.ones((50, 50), dtype=np.bool_),
        target_free_mask=np.ones((50, 50), dtype=np.bool_),
    )


def _assert_exact(expected: np.ndarray, actual: np.ndarray) -> None:
    assert expected.dtype == actual.dtype == np.dtype(np.bool_)
    assert expected.shape == actual.shape == (27, 50, 50)
    assert expected.flags.c_contiguous
    assert actual.flags.c_contiguous
    assert np.array_equal(actual, expected)
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def test_geometry_only_projection_excludes_poisoned_oracle_fields() -> None:
    arrays = _raw_arrays()
    arrays.depth_m[0, 127, 127] = 1.0
    mapped = np.full((12, 256, 256), -1, dtype="<i2")
    mapped[0, 127, 127] = 3
    poisoned = _poison(arrays)

    expected = project_mapped_labels_semantic_only(mapped, arrays)
    _assert_exact(
        expected,
        project_mapped_labels_semantic_only(mapped, poisoned),
    )
    _assert_exact(
        expected,
        SemanticProjectionGeometry.from_raw_frame_arrays(
            arrays
        ).project_mapped_labels(mapped),
    )
    _assert_exact(
        expected,
        SemanticProjectionGeometry.from_raw_frame_arrays(
            poisoned
        ).project_mapped_labels(mapped),
    )


def test_geometry_owns_exact_read_only_arrays() -> None:
    arrays = _raw_arrays()
    geometry = SemanticProjectionGeometry.from_raw_frame_arrays(arrays)
    original = geometry.depth_m.tobytes(order="C")
    arrays.depth_m[0, 0, 0] = 1.0

    assert geometry.depth_m.tobytes(order="C") == original
    assert not geometry.depth_m.flags.writeable
    assert not geometry.sensor_positions.flags.writeable
    assert not geometry.sensor_rotations_xyzw.flags.writeable
    assert not geometry.sensor_hfov_degrees.flags.writeable
    assert not geometry.target_origin_xz.flags.writeable


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("depth_m", np.zeros((12, 256, 255), dtype="<f4")),
        ("sensor_positions", np.zeros((12, 3), dtype="<f4")),
        (
            "sensor_rotations_xyzw",
            np.zeros((12, 4), dtype="<f8"),
        ),
        ("sensor_hfov_degrees", np.array(89.0, dtype="<f8")),
        ("target_origin_xz", np.array([np.nan, 0.0], dtype="<f8")),
    ),
)
def test_geometry_rejects_invalid_exact_contract(
    field: str,
    value: np.ndarray,
) -> None:
    arrays = _raw_arrays()
    with pytest.raises(ValueError, match=field.replace("_", " ")):
        SemanticProjectionGeometry.from_raw_frame_arrays(
            replace(arrays, **{field: value})
        )


def test_depth_bins_freeze_all_endpoint_boundaries() -> None:
    depth = np.array(
        [
            0.0,
            np.nextafter(np.float32(0), np.float32(1)),
            1.0,
            2.0,
            4.0,
            6.0,
            np.nextafter(np.float32(10), np.float32(0)),
            10.0,
        ],
        dtype="<f4",
    )
    assert PixelContributorLedger.depth_bins(depth).tolist() == [
        DepthBin.INVALID,
        DepthBin.UP_TO_1,
        DepthBin.UP_TO_1,
        DepthBin.UP_TO_2,
        DepthBin.UP_TO_4,
        DepthBin.UP_TO_6,
        DepthBin.UP_TO_10,
        DepthBin.SATURATED,
    ]


def test_contributor_ledger_rebuilds_multi_hot_grids_and_error_effects() -> None:
    arrays = _raw_arrays()
    arrays.depth_m[0, 127, 127:130] = 1.0
    arrays.object_categories[0, 127, 127:130] = np.array([1, 1, 2], dtype="<i2")
    source = np.zeros((12, 256, 256), dtype="<i2")
    mapped = np.full((12, 256, 256), -1, dtype="<i2")
    mapped[0, 127, 127:130] = np.array([1, -1, 3], dtype="<i2")
    geometry = SemanticProjectionGeometry.from_raw_frame_arrays(arrays)
    ledger = PixelContributorLedger.build(
        ordinal=0,
        observation_id="0" * 20,
        scene_id="scene",
        sensor_yaw_degrees=arrays.sensor_yaw_degrees,
        geometry=geometry,
        gt_labels=arrays.object_categories,
        source_labels=source,
        mapped_labels=mapped,
    )

    target, prediction = ledger.rebuild_grids()
    _assert_exact(geometry.project_oracle_labels(arrays.object_categories), target)
    _assert_exact(geometry.project_mapped_labels(mapped), prediction)

    rows = ledger.category_cells(target=target, prediction=prediction)
    status_effect = {(row.category, row.status, row.effect) for row in rows}
    assert (
        1,
        CategoryCellStatus.TP,
        CategoryCellEffect.RESCUED_TP,
    ) in status_effect
    assert (
        2,
        CategoryCellStatus.FN,
        CategoryCellEffect.FN_UNANIMOUS_WRONG_CLASS,
    ) in status_effect
    assert (
        3,
        CategoryCellStatus.FP,
        CategoryCellEffect.AMPLIFIED_FP,
    ) in status_effect
    assert any(
        row.category == 1
        and row.gt_pixel_support == 2
        and row.predicted_pixel_support == 1
        and row.correct_pixel_support == 1
        for row in rows
    )
