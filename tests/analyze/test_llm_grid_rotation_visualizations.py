from __future__ import annotations

import numpy as np
import pytest

from prior.analyze.d2026_08_04.llm_grid_rotation_visualizations import (
    CrossFitSelectionRow,
    OracleSelectionRow,
    category_error_codes,
    embed_warp,
    select_crossfit_rows,
    select_oracle_rows,
)
from prior.analyze.llm_grid_registration import SpatialBounds, WarpedGrid


def _oracle(
    angle: float, value: float, identity: float, example: str
) -> OracleSelectionRow:
    return OracleSelectionRow(
        scene_id="scene",
        example_id=example,
        identity_iou=identity,
        best_iou=identity + value,
        delta_iou=value,
        best_angle_degrees=angle,
        second_angle_degrees=(angle + 90.0) % 360.0,
        best_second_margin=0.1,
        input_support=1,
        target_support=1,
        angle_ious=((0.0, identity), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0)),
    )


def test_oracle_selection_uses_declared_medians_and_lexical_ties() -> None:
    rows = []
    for angle in (0.0, 90.0, 180.0, 270.0):
        rows.extend((
            _oracle(angle, 0.1, 0.1, f"{int(angle)}-a"),
            _oracle(angle, 0.2, 0.2, f"{int(angle)}-b"),
            _oracle(angle, 0.3, 0.3, f"{int(angle)}-c"),
        ))
    selected = select_oracle_rows(rows)
    assert tuple(row.example_id for row in selected) == (
        "0-b",
        "90-b",
        "180-b",
        "270-b",
    )


def test_crossfit_selection_is_median_positive_transfer() -> None:
    rows = []
    for direction in ("object_to_region", "region_to_object"):
        for index, delta in enumerate((0.1, 0.2, 0.3)):
            rows.append(
                CrossFitSelectionRow(
                    "scene",
                    f"{direction}-{index}",
                    direction,
                    90.0,
                    0.1,
                    0.2,
                    0.1,
                    0.1,
                    0.1 + delta,
                    delta,
                )
            )
    assert tuple(row.example_id for row in select_crossfit_rows(rows)) == (
        "object_to_region-1",
        "region_to_object-1",
    )


def test_embed_warp_preserves_out_of_frame_support() -> None:
    grid = np.zeros((1, 2, 2), dtype=np.bool_)
    grid[0, 0, 1] = True
    warp = WarpedGrid(grid, SpatialBounds(-1, 1, 50, 52), input_support=1)
    embedded = embed_warp(warp, SpatialBounds(-1, 50, 0, 52))
    assert np.count_nonzero(embedded) == 1
    assert embedded[0, 0, 51]


def test_category_error_codes_preserve_category_substitution() -> None:
    target = np.zeros((2, 1, 4), dtype=np.bool_)
    prediction = np.zeros_like(target)
    target[0, 0, 0] = prediction[0, 0, 0] = True
    prediction[0, 0, 1] = True
    target[0, 0, 2] = True
    target[0, 0, 3] = True
    prediction[1, 0, 3] = True
    assert category_error_codes(prediction, target).tolist() == [[1, 2, 3, 4]]


def test_embed_rejects_clipping_bounds() -> None:
    warp = WarpedGrid(
        np.zeros((1, 2, 2), dtype=np.bool_),
        SpatialBounds(-1, 1, 0, 2),
        input_support=0,
    )
    with pytest.raises(ValueError, match="outside"):
        embed_warp(warp, SpatialBounds(0, 50, 0, 50))
