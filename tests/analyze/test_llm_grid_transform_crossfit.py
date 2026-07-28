from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest

from prior.analyze.d2026_07_28.llm_grid_transform_crossfit import (
    Direction,
    EpisodeCase,
    FamilyAngleScore,
    PivotMode,
    crossfit_scores,
    score_angle_families,
)
from prior.analyze.llm_grid_registration import RasterScore, SpatialBounds


def family_score(
    angle: float, *, object_iou_tenths: int, region_iou_tenths: int
) -> FamilyAngleScore:
    bounds = SpatialBounds(0, 50, 0, 50)

    def raster(intersection: int) -> RasterScore:
        return RasterScore(
            intersection=intersection,
            union=10,
            predicted_support=intersection,
            target_support=10,
            in_frame_support=intersection,
            out_of_frame_support=0,
        )

    return FamilyAngleScore(
        angle_degrees=angle,
        bounds=bounds,
        object_input_support=object_iou_tenths,
        region_input_support=region_iou_tenths,
        object_score=raster(object_iou_tenths),
        region_score=raster(region_iou_tenths),
    )


@pytest.fixture
def canonical_grids() -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    predicted = np.zeros((37, 50, 50), dtype=np.bool_)
    return predicted, np.zeros_like(predicted)


def test_crossfit_selection_uses_only_declared_family() -> None:
    """Breaks if held-out scores independently select their own best angle."""
    scores = (
        family_score(0.0, object_iou_tenths=2, region_iou_tenths=8),
        family_score(90.0, object_iou_tenths=9, region_iou_tenths=1),
        family_score(180.0, object_iou_tenths=1, region_iou_tenths=1),
        family_score(270.0, object_iou_tenths=1, region_iou_tenths=1),
    )

    object_to_region = crossfit_scores(scores, Direction.OBJECT_TO_REGION)
    region_to_object = crossfit_scores(scores, Direction.REGION_TO_OBJECT)

    assert object_to_region.selected_angle_degrees == 90.0
    assert object_to_region.heldout_selected_iou == 0.1
    assert object_to_region.delta_iou == pytest.approx(-0.7)
    assert region_to_object.selected_angle_degrees == 0.0
    assert region_to_object.heldout_selected_iou == 0.2
    assert region_to_object.delta_iou == 0.0


def test_crossfit_tie_retains_declared_identity_angle() -> None:
    """Breaks if a tie changes the caller-declared identity-angle selection."""
    scores = (
        family_score(0.0, object_iou_tenths=0, region_iou_tenths=0),
        family_score(90.0, object_iou_tenths=0, region_iou_tenths=0),
        family_score(180.0, object_iou_tenths=0, region_iou_tenths=0),
        family_score(270.0, object_iou_tenths=0, region_iou_tenths=0),
    )

    result = crossfit_scores(scores, Direction.OBJECT_TO_REGION)

    assert result.selected_angle_degrees == 0.0
    assert result.selector_margin == 0.0
    assert result.delta_iou == 0.0


def test_score_angle_families_preserves_family_support_and_out_of_frame_pixels(
    canonical_grids: tuple[NDArray[np.bool_], NDArray[np.bool_]],
) -> None:
    """Breaks if family scoring rewarps slices or drops rotated false positives."""
    predicted, target = canonical_grids
    predicted[0, 10, 10] = True
    predicted[27, 12, 12] = True
    target[:] = predicted

    scores = score_angle_families(
        predicted,
        target,
        pivot=(0.5, 0.5),
        angles=(180.0, 0.0, 90.0),
    )

    assert tuple(score.angle_degrees for score in scores) == (180.0, 0.0, 90.0)
    assert scores[0].bounds == SpatialBounds(-49, 1, -49, 1)
    assert scores[1].bounds == SpatialBounds(0, 50, 0, 50)
    assert scores[0].object_input_support == 1
    assert scores[0].region_input_support == 1
    assert scores[0].object_score.out_of_frame_support == 1
    assert scores[0].region_score.out_of_frame_support == 1


def test_score_angle_families_rejects_malformed_grid_or_pivot(
    canonical_grids: tuple[NDArray[np.bool_], NDArray[np.bool_]],
) -> None:
    """Breaks if malformed grids reach geometry or hidden coercion paths."""
    predicted, target = canonical_grids

    with pytest.raises(ValueError, match="boolean"):
        score_angle_families(
            predicted.astype(np.float32), target, (1.0, 1.0), (0.0,)
        )
    with pytest.raises(ValueError, match="37"):
        score_angle_families(
            predicted[:36], target[:36], (1.0, 1.0), (0.0,)
        )
    with pytest.raises(ValueError, match="same spatial shape"):
        score_angle_families(
            predicted, target[:, :49], (1.0, 1.0), (0.0,)
        )
    with pytest.raises(ValueError, match="pivot"):
        score_angle_families(predicted, target, (np.nan, 1.0), (0.0,))


def test_crossfit_rejects_duplicate_or_missing_identity_angles() -> None:
    """Breaks if cross-fit output can lack an unambiguous baseline row."""
    with pytest.raises(ValueError, match="unique"):
        crossfit_scores(
            (
                family_score(0.0, object_iou_tenths=1, region_iou_tenths=1),
                family_score(0.0, object_iou_tenths=2, region_iou_tenths=2),
            ),
            Direction.OBJECT_TO_REGION,
        )
    with pytest.raises(ValueError, match="identity"):
        crossfit_scores(
            (
                family_score(90.0, object_iou_tenths=1, region_iou_tenths=1),
                family_score(180.0, object_iou_tenths=1, region_iou_tenths=1),
            ),
            Direction.OBJECT_TO_REGION,
        )


def test_episode_case_rejects_noncanonical_grid_or_empty_identity(
    canonical_grids: tuple[NDArray[np.bool_], NDArray[np.bool_]],
) -> None:
    """Breaks if records admit malformed data before grouped analysis."""
    predicted, target = canonical_grids
    assert PivotMode.TRUE_START.value == "true_start"
    valid = EpisodeCase(
        split="val_unseen",
        scene_id="scene",
        example_id="example",
        schema_valid=True,
        predicted_grid=predicted,
        target_grid=target,
        true_start_pivot=(1.0, 1.0),
    )
    assert valid.true_start_pivot == (1.0, 1.0)
    with pytest.raises(ValueError, match="split"):
        EpisodeCase(
            split="",
            scene_id="scene",
            example_id="example",
            schema_valid=True,
            predicted_grid=predicted,
            target_grid=target,
            true_start_pivot=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="boolean"):
        EpisodeCase(
            split="val_unseen",
            scene_id="scene",
            example_id="example",
            schema_valid=True,
            predicted_grid=predicted.astype(np.float32),
            target_grid=target,
            true_start_pivot=(1.0, 1.0),
        )
