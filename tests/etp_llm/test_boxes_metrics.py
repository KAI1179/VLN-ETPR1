import pytest

from vlnce_baselines.models.etp_llm.boxes_metrics import (
    category_aware_raster_metrics,
    category_f1,
    evaluate_llm_boxes_prediction,
)
from vlnce_baselines.models.etp_llm.boxes_schema import (
    ObjectBoxSpec,
    RegionBoxSpec,
    LLMBoxesSpec,
    spec_to_relevant_semantic_boxes,
)


def _spec(objects=None, regions=None):
    return LLMBoxesSpec(objects=objects or [], regions=regions or [])


def _object(category, center=(10.0, 10.0), half_extents=(1.0, 1.0)):
    return ObjectBoxSpec(
        category=category,
        center=center,
        half_extents=half_extents,
        rotation=0.0,
    )


def _region(category, min_point=(20.0, 20.0), max_point=(22.0, 22.0)):
    return RegionBoxSpec(category=category, min=min_point, max=max_point)


def _relevant(spec):
    return spec_to_relevant_semantic_boxes(
        spec,
        instruction="Go to the target.",
        level_idx=0,
        reference_path=[(0.0, 0.0)],
        start_direction_vector=(0.0, 1.0),
        category_extractor=lambda instruction: (set(), set()),
    )


def test_category_f1_counts_entity_type_and_category_multisets():
    pred = _spec(
        objects=[_object("chair"), _object("chair"), _object("table")],
        regions=[_region("circulation")],
    )
    target = _spec(
        objects=[_object("chair"), _object("table"), _object("table")],
        regions=[_region("living/social space")],
    )

    metrics = category_f1(pred, target)

    assert metrics == {
        "category_precision": pytest.approx(0.5),
        "category_recall": pytest.approx(0.5),
        "category_f1": pytest.approx(0.5),
    }


def test_category_aware_raster_metrics_macro_average_gt_present_channels():
    pred = _relevant(
        _spec(
            objects=[_object("chair")],
            regions=[_region("living/social space")],
        )
    )
    target = _relevant(
        _spec(
            objects=[_object("chair")],
            regions=[_region("living/social space")],
        )
    )

    metrics = category_aware_raster_metrics(pred, target)

    assert metrics == {
        "category_aware_raster_iou": pytest.approx(1.0),
        "category_aware_raster_recall": pytest.approx(1.0),
        "category_aware_raster_support": 2,
    }


def test_category_aware_raster_metrics_penalizes_wrong_category_geometry():
    pred = _relevant(_spec(objects=[_object("table")]))
    target = _relevant(_spec(objects=[_object("chair")]))

    metrics = category_aware_raster_metrics(pred, target)

    assert metrics["category_aware_raster_iou"] == pytest.approx(0.0)
    assert metrics["category_aware_raster_recall"] == pytest.approx(0.0)
    assert metrics["category_aware_raster_support"] == 1


def test_category_aware_raster_metrics_handles_empty_prediction():
    pred = _relevant(_spec())
    target = _relevant(_spec(objects=[_object("chair")]))

    metrics = category_aware_raster_metrics(pred, target)

    assert metrics["category_aware_raster_iou"] == pytest.approx(0.0)
    assert metrics["category_aware_raster_recall"] == pytest.approx(0.0)
    assert metrics["category_aware_raster_support"] == 1


def test_category_aware_raster_metrics_documents_empty_target_convention():
    pred = _relevant(_spec(objects=[_object("chair")]))
    target = _relevant(_spec())

    metrics = category_aware_raster_metrics(pred, target)

    assert metrics == {
        "category_aware_raster_iou": pytest.approx(0.0),
        "category_aware_raster_recall": pytest.approx(0.0),
        "category_aware_raster_support": 0,
    }


def test_category_aware_raster_metrics_documents_both_empty_convention():
    pred = _relevant(_spec())
    target = _relevant(_spec())

    metrics = category_aware_raster_metrics(pred, target)

    assert metrics == {
        "category_aware_raster_iou": pytest.approx(0.0),
        "category_aware_raster_recall": pytest.approx(0.0),
        "category_aware_raster_support": 0,
    }


def test_evaluate_llm_boxes_prediction_combines_category_and_raster_metrics():
    pred_spec = _spec(objects=[_object("chair")])
    target_spec = _spec(objects=[_object("chair")])

    metrics = evaluate_llm_boxes_prediction(
        pred_spec,
        target_spec,
        _relevant(pred_spec),
        _relevant(target_spec),
    )

    assert metrics == {
        "category_precision": pytest.approx(1.0),
        "category_recall": pytest.approx(1.0),
        "category_f1": pytest.approx(1.0),
        "category_aware_raster_iou": pytest.approx(1.0),
        "category_aware_raster_recall": pytest.approx(1.0),
        "category_aware_raster_support": 1,
    }
