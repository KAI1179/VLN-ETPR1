from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    project_mapped_labels,
    project_oracle_target_labels,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    RawFrameArrays,
)
from prior.analyze.d2026_07_30.rgbd_semantic_projector import (
    project_mapped_labels_semantic_only,
    project_oracle_target_labels_semantic_only,
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


def _assert_exact(expected: np.ndarray, actual: np.ndarray) -> None:
    assert expected.dtype == actual.dtype == np.dtype(np.bool_)
    assert expected.shape == actual.shape == (27, 50, 50)
    assert expected.flags.c_contiguous
    assert actual.flags.c_contiguous
    assert np.array_equal(actual, expected)
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


@pytest.mark.parametrize(
    ("depth", "labels"),
    (
        ((), ()),
        (((0, 127, 127, 1.0),), ((0, 127, 127, 0),)),
        (((0, 127, 127, 10.0),), ((0, 127, 127, 4),)),
        (
            ((0, 127, 127, 1.0), (1, 127, 127, 1.0)),
            ((0, 127, 127, 2), (1, 127, 127, 3)),
        ),
    ),
)
def test_mapped_semantic_only_matches_synthetic_edges(
    depth: tuple[tuple[int, int, int, float], ...],
    labels: tuple[tuple[int, int, int, int], ...],
) -> None:
    arrays = _raw_arrays()
    mapped = np.full((12, 256, 256), -1, dtype="<i2")
    for view, row, col, value in depth:
        arrays.depth_m[view, row, col] = value
    for view, row, col, value in labels:
        mapped[view, row, col] = value
    _assert_exact(
        project_mapped_labels(mapped, arrays),
        project_mapped_labels_semantic_only(mapped, arrays),
    )


def test_oracle_semantic_only_keeps_other_and_drops_out_of_grid_hits() -> None:
    arrays = _raw_arrays()
    arrays.depth_m[0, 127, 127] = 1.0
    arrays.object_categories[0, 127, 127] = 16
    _assert_exact(
        project_oracle_target_labels(arrays),
        project_oracle_target_labels_semantic_only(arrays),
    )

    outside = replace(
        arrays,
        target_origin_xz=np.array([100.0, 100.0], dtype="<f8"),
    )
    _assert_exact(
        project_oracle_target_labels(outside),
        project_oracle_target_labels_semantic_only(outside),
    )


def test_mapped_semantic_only_rejects_oracle_only_other_label() -> None:
    arrays = _raw_arrays()
    mapped = np.full((12, 256, 256), -1, dtype="<i2")
    mapped[0, 127, 127] = 16
    with pytest.raises(ValueError, match="mapped labels"):
        project_mapped_labels_semantic_only(mapped, arrays)
