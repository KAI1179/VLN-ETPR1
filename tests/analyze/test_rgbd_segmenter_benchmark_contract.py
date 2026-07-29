from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence, cast

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    DEFAULT_NYU40_MAPPING_PATH,
    RAW_FRAME_ROOT,
    RAW_GPU_UUID,
    RAW_INDEX_SHA256,
    RAW_MANIFEST_SHA256,
    RAW_PRODUCER_COMMIT,
    RAW_VALIDATOR_SOURCE_SHA256,
    BenchmarkEnvironmentAttestation,
    BenchmarkMetricSummary,
    MAX_ABSOLUTE_RESERVED_BYTES,
    MAX_LATENCY_P95_SECONDS,
    MIN_COVERED_CATEGORIES,
    MIN_MEAN_F1,
    MIN_MEAN_IOU,
    MIN_SUPPORT_COVERAGE,
    CandidateCommitment,
    CandidateGateResult,
    ConfusionCounts,
    ContrastEstimate,
    DeviceBatch,
    EndpointRow,
    GateStatus,
    LatencySummary,
    LicenseStatus,
    MappingEntry,
    MetricEndpoint,
    ObservationFailureCode,
    ObservationResult,
    ObservationMetrics,
    ObservationStatus,
    MappingKind,
    P53ValidationAttestation,
    P53ValidatorLaunch,
    PRIMARY_CATEGORY_INDICES,
    ProvenanceChecks,
    RobustnessEstimate,
    SegmenterInput,
    SpatialTransform,
    Prediction,
    PreparedHostBatch,
    ResourceMeasurement,
    StaticCoverage,
    TimingSample,
    TrustedCohort,
    aggregate_observation_metrics,
    capture_environment_sha256,
    canonical_json_bytes,
    estimate_scene_contrast,
    estimate_scene_robustness,
    evaluate_candidate_gates,
    iter_validated_raw_observations,
    inspect_visible_gpu,
    linear_quantile,
    load_nyu40_mapping,
    map_source_labels,
    project_mapped_labels,
    run_p53_validation_subprocess,
    restore_source_labels,
    scene_bootstrap_matrix,
    score_observation,
    transfer_prepared_host_batch,
    validate_source_logits,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import RawFrameArrays
from vlnce_baselines.models.etp_llm.llm_grid_oracle_cache import OracleSensorFrame

_RAW_INTEGRATION = pytest.mark.skipif(
    os.environ.get("ETP_R1_RUN_RGBD_RAW_INTEGRATION") != "1",
    reason="set ETP_R1_RUN_RGBD_RAW_INTEGRATION=1 for the sealed 166MB package",
)


def test_frozen_nyu40_mapping_has_exact_vocabulary_and_semantics() -> None:
    mapping = load_nyu40_mapping(DEFAULT_NYU40_MAPPING_PATH)

    assert len(mapping) == 40
    assert tuple(entry.source_index for entry in mapping) == tuple(range(40))
    assert mapping[0].source_name == "wall"
    assert mapping[23].source_name == "refridgerator"
    assert mapping[0].kind is MappingKind.DIAGNOSTIC
    assert mapping[0].canonical_index == 15
    assert mapping[1].canonical_index == 17
    assert mapping[13].kind is MappingKind.MANY_TO_ONE
    assert mapping[13].canonical_index == 3
    assert mapping[27].kind is MappingKind.IGNORED
    assert mapping[27].canonical_index is None
    assert all(entry.canonical_index != 16 for entry in mapping)
    assert hashlib.sha256(DEFAULT_NYU40_MAPPING_PATH.read_bytes()).hexdigest() == (
        "133ebb300e5dc01f6176e4eb01a48569c3eec635a90fea2b8a602b53e50bc395"
    )


def test_mapping_counts_coverage_and_shift_sensitive_rows_are_frozen() -> None:
    mapping = load_nyu40_mapping()

    assert sum(entry.kind is MappingKind.DIRECT for entry in mapping) == 13
    assert sum(entry.kind is MappingKind.SYNONYM for entry in mapping) == 3
    assert sum(entry.kind is MappingKind.MANY_TO_ONE for entry in mapping) == 3
    assert sum(entry.kind is MappingKind.DIAGNOSTIC for entry in mapping) == 6
    assert sum(entry.kind is MappingKind.IGNORED for entry in mapping) == 15
    assert (
        len({
            entry.canonical_index
            for entry in mapping
            if entry.kind
            in {MappingKind.DIRECT, MappingKind.SYNONYM, MappingKind.MANY_TO_ONE}
        })
        == 17
    )
    assert mapping[0].source_name == "wall"
    assert mapping[27].source_name == "shower curtain"
    assert mapping[38].source_name == "otherfurniture"
    assert MappingKind.UNMAPPED.value == "unmapped"


def test_nyu_mapping_loader_rejects_any_self_consistent_file_mutation(
    tmp_path: Path,
) -> None:
    mutated = tmp_path / "mapping.json"
    mutated.write_bytes(
        DEFAULT_NYU40_MAPPING_PATH.read_bytes().replace(b'"wall"', b'"walls"', 1)
    )
    with pytest.raises(ValueError, match="frozen file hash"):
        load_nyu40_mapping(mutated)


def test_segmenter_input_is_pinned_cpu_exact_and_frozen() -> None:
    value = SegmenterInput(
        rgb=torch.zeros((12, 256, 256, 3), dtype=torch.uint8, pin_memory=True),
        depth_m=torch.ones((12, 256, 256), dtype=torch.float32, pin_memory=True),
    )
    assert value.rgb.device.type == "cpu"
    assert value.depth_m.is_pinned()
    with pytest.raises(FrozenInstanceError):
        setattr(value, "rgb", value.rgb)

    with pytest.raises(ValueError, match="pinned"):
        SegmenterInput(
            rgb=torch.zeros((12, 256, 256, 3), dtype=torch.uint8),
            depth_m=torch.ones((12, 256, 256), dtype=torch.float32),
        )
    invalid_depth = torch.ones((12, 256, 256), dtype=torch.float32, pin_memory=True)
    invalid_depth[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        SegmenterInput(
            rgb=torch.zeros((12, 256, 256, 3), dtype=torch.uint8, pin_memory=True),
            depth_m=invalid_depth,
        )


def test_logits_and_mapping_fail_closed_on_schema_and_ranges() -> None:
    mapping = load_nyu40_mapping()
    logits = torch.zeros((12, 40, 7, 11), dtype=torch.float32)
    validate_source_logits(logits, source_class_count=40)
    with pytest.raises(ValueError, match="finite"):
        validate_source_logits(logits.fill_(float("inf")), source_class_count=40)
    with pytest.raises(ValueError, match="shape"):
        validate_source_logits(torch.zeros((12, 39, 7, 11)), source_class_count=40)

    labels = torch.zeros((12, 256, 256), dtype=torch.int64)
    labels[0, 0, :5] = torch.tensor([0, 1, 13, 27, 38])
    mapped = map_source_labels(labels, mapping)
    assert mapped.dtype is torch.int16
    assert mapped[0, 0, :5].tolist() == [15, 17, 3, -1, -1]
    with pytest.raises(ValueError, match="shape"):
        map_source_labels(labels[:, :, :5], mapping)
    with pytest.raises(ValueError, match="range"):
        map_source_labels(torch.full((12, 256, 256), 40), mapping)

    generic = MappingEntry(400, "future", MappingKind.UNMAPPED, None, None)
    assert generic.source_index == 400
    expanded = tuple(
        MappingEntry(index, f"source-{index}", MappingKind.UNMAPPED, None, None)
        for index in range(41)
    )
    expanded_labels = torch.full((12, 256, 256), 40, dtype=torch.int64)
    assert bool((map_source_labels(expanded_labels, expanded) == -1).all())
    with pytest.raises(ValueError, match="MappingKind"):
        MappingEntry(0, "bad", cast(MappingKind, "ignored"), None, None)


def test_spatial_transform_requires_exact_symmetric_padding_contract() -> None:
    transform = SpatialTransform.from_sizes(
        raw_height=256,
        raw_width=256,
        model_height=257,
        model_width=320,
    )
    assert (transform.resized_height, transform.resized_width) == (257, 257)
    assert (transform.pad_top, transform.pad_bottom) == (0, 0)
    assert (transform.pad_left, transform.pad_right) == (31, 32)
    with pytest.raises(ValueError, match="symmetric"):
        SpatialTransform(
            raw_height=256,
            raw_width=256,
            resized_height=257,
            resized_width=257,
            model_height=257,
            model_width=320,
            pad_top=0,
            pad_bottom=0,
            pad_left=30,
            pad_right=33,
        )


def test_logit_restore_unpads_poison_and_resolves_ties_before_mapping() -> None:
    transform = SpatialTransform.from_sizes(
        raw_height=256,
        raw_width=256,
        model_height=257,
        model_width=320,
    )
    logits = torch.zeros((12, 40, 257, 320), dtype=torch.float32)
    content = (
        slice(None),
        slice(None),
        slice(transform.pad_top, 257 - transform.pad_bottom),
        slice(transform.pad_left, 320 - transform.pad_right),
    )
    logits[content] = -1
    logits[:, 13, content[2], content[3]] = 1
    logits[:, 39, :, : transform.pad_left] = 1_000
    logits[:, 39, :, 320 - transform.pad_right :] = 1_000

    source = restore_source_labels(
        logits,
        spatial_transform=transform,
        source_class_count=40,
    )
    mapped = map_source_labels(source, load_nyu40_mapping())

    assert source.dtype is torch.int16
    assert mapped.dtype is torch.int16
    assert tuple(source.shape) == (12, 256, 256)
    assert bool((source == 13).all())
    assert bool((mapped == 3).all())

    tied_source = restore_source_labels(
        torch.zeros((12, 40, 1, 1)),
        spatial_transform=transform,
        source_class_count=40,
    )
    tied_mapped = map_source_labels(tied_source, load_nyu40_mapping())
    assert bool((tied_source == 0).all())
    assert bool((tied_mapped == 15).all())


def test_logit_restore_resizes_logits_before_argmax_and_maps_afterward() -> None:
    transform = SpatialTransform.from_sizes(
        raw_height=256,
        raw_width=256,
        model_height=256,
        model_width=256,
    )
    logits = torch.full((12, 40, 1, 2), -10.0)
    logits[:, 4, 0, 0] = 2
    logits[:, 5, 0, 0] = 0
    logits[:, 4, 0, 1] = 0
    logits[:, 5, 0, 1] = 1

    source = restore_source_labels(
        logits,
        spatial_transform=transform,
        source_class_count=40,
    )
    mapped = map_source_labels(source, load_nyu40_mapping())

    # Hard-label resize crosses at 128, while align_corners=False interpolated
    # logits cross near 149 (align_corners=True would cross near 170).
    assert source[0, 128, 140].item() == 4
    assert source[0, 128, 150].item() == 5
    assert source[0, 128, 190].item() == 5
    assert mapped[0, 128, 140].item() == 1
    assert mapped[0, 128, 190].item() == 5


def test_source_argmax_precedes_many_to_one_mapping_without_logit_merging() -> None:
    transform = SpatialTransform.from_sizes(
        raw_height=256,
        raw_width=256,
        model_height=256,
        model_width=256,
    )
    logits = torch.full((12, 40, 1, 1), -10.0)
    logits[:, 6] = 4
    logits[:, 13] = 4
    logits[:, 2] = 5

    source = restore_source_labels(
        logits,
        spatial_transform=transform,
        source_class_count=40,
    )
    mapped = map_source_labels(source, load_nyu40_mapping())

    # Source 6 and 13 both map to table (3), but their logits must not be
    # summed before source class 2 (cabinet, 19) wins argmax.
    assert bool((source == 2).all())
    assert bool((mapped == 19).all())


def test_logit_restore_rejects_malformed_transform_and_logits() -> None:
    valid = SpatialTransform.from_sizes(
        raw_height=256,
        raw_width=256,
        model_height=256,
        model_width=256,
    )
    wrong_raw_size = SpatialTransform.from_sizes(
        raw_height=128,
        raw_width=256,
        model_height=256,
        model_width=256,
    )
    logits = torch.zeros((12, 40, 1, 1))

    with pytest.raises(ValueError, match="shared 256x256"):
        restore_source_labels(
            logits,
            spatial_transform=wrong_raw_size,
            source_class_count=40,
        )
    with pytest.raises(ValueError, match="wrong shape"):
        restore_source_labels(
            torch.zeros((11, 40, 1, 1)),
            spatial_transform=valid,
            source_class_count=40,
        )
    with pytest.raises(ValueError, match="finite"):
        restore_source_labels(
            torch.full((12, 40, 1, 1), float("nan")),
            spatial_transform=valid,
            source_class_count=40,
        )


def test_spatial_coordinate_markers_survive_forward_and_inverse_round_trip() -> None:
    transform = SpatialTransform.from_sizes(
        raw_height=256,
        raw_width=256,
        model_height=257,
        model_width=320,
    )
    raw_logits = torch.full((12, 6, 256, 256), -1.0)
    raw_logits[:, 0] = 0
    raw_logits[:, 1, 0, 0] = 10
    raw_logits[:, 2, 0, 255] = 10
    raw_logits[:, 3, 128, 128] = 10
    raw_logits[:, 4, 255, 0] = 10
    raw_logits[:, 5, 255, 255] = 10
    resized = functional.interpolate(
        raw_logits,
        size=(transform.resized_height, transform.resized_width),
        mode="bilinear",
        align_corners=False,
    )
    padded = functional.pad(
        resized,
        (
            transform.pad_left,
            transform.pad_right,
            transform.pad_top,
            transform.pad_bottom,
        ),
        value=-1,
    )

    source = restore_source_labels(
        padded,
        spatial_transform=transform,
        source_class_count=6,
    )

    assert source[0, 0, 0].item() == 1
    assert source[0, 0, 255].item() == 2
    assert source[0, 128, 128].item() == 3
    assert source[0, 255, 0].item() == 4
    assert source[0, 255, 255].item() == 5
    assert source[0, 64, 64].item() == 0


def _projection_arrays() -> RawFrameArrays:
    depth = np.zeros((12, 256, 256), dtype="<f4")
    depth[0, 127, 127] = 1
    depth[1, 127, 127] = 1
    depth[2, 127, 127] = 10
    depth[3, 127, 127] = 0
    return RawFrameArrays(
        schema_version=np.asarray(1, dtype="<i8"),
        rgb=np.zeros((12, 256, 256, 3), dtype="|u1"),
        depth_m=depth,
        object_categories=np.full((12, 256, 256), -1, dtype="<i2"),
        region_categories=np.full((12, 256, 256), -1, dtype="<i2"),
        sensor_positions=np.zeros((12, 3), dtype="<f8"),
        sensor_rotations_xyzw=np.tile(np.asarray((0, 0, 0, 1), dtype="<f8"), (12, 1)),
        sensor_yaw_degrees=np.arange(0, 360, 30, dtype="<i2"),
        sensor_hfov_degrees=np.asarray(90, dtype="<f8"),
        sensor_position_relative=np.asarray((0, 1.25, 0), dtype="<f8"),
        start_position=np.zeros(3, dtype="<f8"),
        start_rotation_xyzw=np.asarray((0, 0, 0, 1), dtype="<f8"),
        target_origin_xz=np.asarray((-25, -26), dtype="<f8"),
        ego_observed_mask=np.zeros((50, 50), dtype="|b1"),
        ego_free_mask=np.zeros((50, 50), dtype="|b1"),
        target_observed_mask=np.eye(50, dtype="|b1"),
        target_free_mask=np.fliplr(np.eye(50, dtype="|b1")).copy(),
    )


def test_projection_reuses_all_poses_and_discards_candidate_geometry_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract as contract

    angles = np.deg2rad(np.arange(0, 360, 30, dtype="<f8"))
    arrays = replace(
        _projection_arrays(),
        sensor_positions=np.column_stack((
            np.arange(12, dtype="<f8"),
            np.arange(12, dtype="<f8") + 0.25,
            -np.arange(12, dtype="<f8"),
        )),
        sensor_rotations_xyzw=np.column_stack((
            np.zeros(12, dtype="<f8"),
            np.sin(angles / 2),
            np.zeros(12, dtype="<f8"),
            np.cos(angles / 2),
        )),
    )
    mapped = np.full((12, 256, 256), -1, dtype="<i2")
    seen: list[OracleSensorFrame] = []
    call_count = 0

    def projector(
        frames: Sequence[OracleSensorFrame],
        *,
        start_position: Sequence[float],
        start_rotation: Sequence[float],
        target_origin_xz: Sequence[float],
    ) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        assert call_count == 1, "projector called more than once"
        seen.extend(frames)
        assert tuple(start_position) == tuple(arrays.start_position)
        assert tuple(start_rotation) == tuple(arrays.start_rotation_xyzw)
        assert tuple(target_origin_xz) == tuple(arrays.target_origin_xz)
        semantic = np.zeros((37, 50, 50), dtype="|b1")
        semantic[1, 0, 0] = True
        return SimpleNamespace(
            target_semantic_grid=semantic,
            target_observed_mask=np.zeros((50, 50), dtype="|b1"),
            target_free_mask=np.zeros((50, 50), dtype="|b1"),
        )

    monkeypatch.setattr(contract, "project_oracle_frames", projector)
    projected = project_mapped_labels(mapped, arrays)

    assert call_count == 1
    assert len(seen) == 12
    for index, frame in enumerate(seen):
        assert np.array_equal(frame.depth_m, arrays.depth_m[index])
        assert np.shares_memory(frame.depth_m, arrays.depth_m)
        assert np.array_equal(frame.object_categories, mapped[index])
        assert np.shares_memory(frame.object_categories, mapped)
        assert bool((frame.region_categories == -1).all())
        assert frame.sensor_position == tuple(arrays.sensor_positions[index])
        assert frame.sensor_rotation == tuple(arrays.sensor_rotations_xyzw[index])
        assert frame.hfov_degrees == 90
    assert projected.shape == (27, 50, 50)
    assert projected[1, 0, 0]
    assert projected.sum() == 1
    assert np.array_equal(arrays.target_observed_mask, np.eye(50, dtype="|b1"))
    assert arrays.target_observed_mask[0, 0]
    assert not arrays.target_free_mask.all()


def test_projection_rejects_label_and_projector_schema_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract as contract

    arrays = _projection_arrays()
    valid = np.full((12, 256, 256), -1, dtype="<i2")
    invalid_values = (
        valid.astype("<i4"),
        valid[:, :, :255],
        np.full((12, 256, 256), 16, dtype="<i2"),
        np.asfortranarray(valid),
    )
    for invalid in invalid_values:
        with pytest.raises(ValueError, match="schema or range"):
            project_mapped_labels(invalid, arrays)

    def projector_returning(
        output: np.ndarray,
    ) -> object:
        def projector(
            frames: Sequence[OracleSensorFrame],
            *,
            start_position: Sequence[float],
            start_rotation: Sequence[float],
            target_origin_xz: Sequence[float],
        ) -> SimpleNamespace:
            del frames, start_position, start_rotation, target_origin_xz
            return SimpleNamespace(target_semantic_grid=output)

        return projector

    invalid_outputs = (
        np.zeros((27, 50, 50), dtype="|b1"),
        np.zeros((37, 50, 50), dtype="<i2"),
        np.zeros((37, 50, 50), dtype="|b1")[:, :, ::-1],
    )
    for invalid_output in invalid_outputs:
        monkeypatch.setattr(
            contract,
            "project_oracle_frames",
            projector_returning(invalid_output),
        )
        with pytest.raises(ValueError, match="projector returned"):
            project_mapped_labels(valid, arrays)


def test_projection_unions_views_and_ignores_zero_and_saturated_semantics() -> None:
    arrays = _projection_arrays()
    mapped = np.full((12, 256, 256), -1, dtype="<i2")
    mapped[0, 127, 127] = 1
    mapped[1, 127, 127] = 2
    mapped[2, 127, 127] = 3
    mapped[3, 127, 127] = 4

    projected = project_mapped_labels(mapped, arrays)

    assert projected[1, 24, 25]
    assert projected[2, 24, 25]
    assert not projected[3].any()
    assert not projected[4].any()
    assert projected.sum() == 2


def _grid(*entries: tuple[int, int, int]) -> np.ndarray:
    value = np.zeros((27, 50, 50), dtype="|b1")
    for category, row, column in entries:
        value[category, row, column] = True
    return value


def test_score_observation_rejects_dtype_shape_and_contiguity_mutations() -> None:
    target = _grid((1, 0, 0))
    invalid_predictions = (
        target.astype("|u1"),
        target[:, :, :49],
        target[:, :, ::-1],
    )

    for prediction in invalid_predictions:
        with pytest.raises(ValueError, match="C-contiguous bool"):
            score_observation(
                ordinal=0,
                scene_id="scene-a",
                prediction=prediction,
                target=target,
            )


def test_observation_metrics_use_exact_primary_channels_and_empty_rules() -> None:
    assert PRIMARY_CATEGORY_INDICES == (*range(1, 15), *range(18, 27))
    metrics = score_observation(
        ordinal=0,
        scene_id="scene-a",
        prediction=_grid((1, 0, 0), (15, 1, 1)),
        target=_grid((1, 0, 0), (2, 0, 1), (15, 1, 1)),
    )

    assert isinstance(metrics, ObservationMetrics)
    assert metrics.primary.counts == ConfusionCounts(tp=1, fp=0, fn=1)
    assert metrics.primary.iou == 0.5
    assert metrics.primary.f1 == pytest.approx(2 / 3)
    assert metrics.all_27.counts == ConfusionCounts(tp=2, fp=0, fn=1)
    assert metrics.per_category[1].counts == ConfusionCounts(tp=1, fp=0, fn=0)
    assert metrics.per_category[2].counts == ConfusionCounts(tp=0, fp=0, fn=1)
    assert metrics.per_category[15].counts == ConfusionCounts(tp=1, fp=0, fn=0)

    target_nonempty = score_observation(
        ordinal=1,
        scene_id="scene-a",
        prediction=_grid(),
        target=_grid((1, 0, 0)),
    ).primary
    assert target_nonempty.precision is None
    assert target_nonempty.recall == 0
    assert target_nonempty.iou == 0
    assert target_nonempty.f1 == 0
    assert target_nonempty.eligible

    both_empty = score_observation(
        ordinal=2,
        scene_id="scene-a",
        prediction=_grid(),
        target=_grid(),
    ).primary
    assert both_empty == MetricEndpoint(
        counts=ConfusionCounts(0, 0, 0),
        precision=None,
        recall=None,
        iou=None,
        f1=None,
        eligible=False,
    )

    false_positive = score_observation(
        ordinal=3,
        scene_id="scene-a",
        prediction=_grid((1, 0, 0)),
        target=_grid(),
    ).primary
    assert false_positive.precision == 0
    assert false_positive.recall is None
    assert false_positive.iou == 0
    assert false_positive.f1 == 0
    assert false_positive.eligible


def test_primary_flattens_categories_and_all_27_retains_only_diagnostics() -> None:
    diagnostic = score_observation(
        ordinal=0,
        scene_id="scene-a",
        prediction=_grid((15, 0, 0), (17, 0, 1)),
        target=_grid((0, 0, 2), (15, 0, 0), (16, 0, 3), (17, 0, 1)),
    )
    assert diagnostic.primary.counts == ConfusionCounts(0, 0, 0)
    assert not diagnostic.primary.eligible
    assert diagnostic.all_27.counts == ConfusionCounts(2, 0, 2)

    joint = score_observation(
        ordinal=1,
        scene_id="scene-a",
        prediction=_grid((1, 0, 0), (2, 1, 0)),
        target=_grid((1, 0, 0), (1, 0, 1), (1, 0, 2), (2, 1, 0)),
    )
    assert joint.primary.iou == 0.5
    assert (
        cast(float, joint.per_category[1].iou) + cast(float, joint.per_category[2].iou)
    ) / 2 == pytest.approx(2 / 3)


def test_mapping_is_post_argmax_and_many_to_one_remains_hard_label_mapping() -> None:
    labels = torch.zeros((12, 256, 256), dtype=torch.int16)
    labels[0, 0, :4] = torch.tensor((6, 13, 2, 31), dtype=torch.int16)

    mapped = map_source_labels(labels, load_nyu40_mapping())

    assert mapped[0, 0, :4].tolist() == [3, 3, 19, 19]


def test_metric_aggregation_separates_observation_pooled_scene_and_category() -> None:
    rows = (
        score_observation(
            ordinal=0,
            scene_id="scene-a",
            prediction=_grid((1, 0, 0)),
            target=_grid((1, 0, 0), (1, 0, 1)),
        ),
        score_observation(
            ordinal=1,
            scene_id="scene-a",
            prediction=_grid(),
            target=_grid((2, 0, 0)),
        ),
        score_observation(
            ordinal=2,
            scene_id="scene-b",
            prediction=_grid((1, 0, 0), (1, 0, 1)),
            target=_grid((1, 0, 0)),
        ),
    )

    summary = aggregate_observation_metrics(
        rows, expected_scenes=("scene-a", "scene-b")
    )

    assert isinstance(summary, BenchmarkMetricSummary)
    assert summary.primary.observation_count == 3
    assert summary.primary.eligible_observation_count == 3
    assert summary.primary.empty_both_count == 0
    assert summary.primary.target_empty_prediction_nonempty_count == 0
    assert summary.primary.target_nonempty_prediction_empty_count == 1
    assert summary.primary.mean_iou == pytest.approx(1 / 3)
    assert summary.primary.mean_f1 == pytest.approx((2 / 3 + 0 + 2 / 3) / 3)
    assert summary.primary.pooled.counts == ConfusionCounts(tp=2, fp=1, fn=2)
    assert summary.primary.pooled.iou == pytest.approx(2 / 5)
    assert summary.primary.scene_macro_iou == pytest.approx(3 / 8)
    assert summary.per_category[1].pooled.counts == ConfusionCounts(2, 1, 1)
    assert summary.per_category[2].pooled.counts == ConfusionCounts(0, 0, 1)


def test_scene_macro_is_null_when_any_expected_scene_has_no_eligible_row() -> None:
    rows = (
        score_observation(
            ordinal=0,
            scene_id="scene-a",
            prediction=_grid((1, 0, 0)),
            target=_grid((1, 0, 0)),
        ),
        score_observation(
            ordinal=1,
            scene_id="scene-b",
            prediction=_grid(),
            target=_grid(),
        ),
    )

    summary = aggregate_observation_metrics(
        rows, expected_scenes=("scene-a", "scene-b")
    )

    assert summary.primary.mean_iou == 1
    assert summary.primary.empty_both_count == 1
    assert summary.primary.scene_macro_iou is None
    assert summary.primary.scene_macro_f1 is None


def test_metric_aggregate_counts_each_empty_case_separately() -> None:
    rows = (
        score_observation(
            ordinal=0,
            scene_id="scene-a",
            prediction=_grid(),
            target=_grid(),
        ),
        score_observation(
            ordinal=1,
            scene_id="scene-a",
            prediction=_grid((1, 0, 0)),
            target=_grid(),
        ),
        score_observation(
            ordinal=2,
            scene_id="scene-a",
            prediction=_grid(),
            target=_grid((1, 0, 0)),
        ),
    )

    aggregate = aggregate_observation_metrics(
        rows, expected_scenes=("scene-a",)
    ).primary

    assert aggregate.empty_both_count == 1
    assert aggregate.target_empty_prediction_nonempty_count == 1
    assert aggregate.target_nonempty_prediction_empty_count == 1


def test_metric_aggregation_rejects_empty_duplicate_and_scene_mutations() -> None:
    scene_a = score_observation(
        ordinal=0,
        scene_id="scene-a",
        prediction=_grid(),
        target=_grid(),
    )
    scene_b = score_observation(
        ordinal=1,
        scene_id="scene-b",
        prediction=_grid(),
        target=_grid(),
    )

    with pytest.raises(ValueError, match="non-empty"):
        aggregate_observation_metrics((), expected_scenes=("scene-a",))
    with pytest.raises(ValueError, match="unique ordinals"):
        aggregate_observation_metrics(
            (scene_a, replace(scene_b, ordinal=0)),
            expected_scenes=("scene-a", "scene-b"),
        )
    with pytest.raises(ValueError, match="lexical order"):
        aggregate_observation_metrics(
            (scene_a, scene_b),
            expected_scenes=("scene-b", "scene-a"),
        )
    with pytest.raises(ValueError, match="exactly match"):
        aggregate_observation_metrics(
            (scene_a, scene_b),
            expected_scenes=("scene-a",),
        )


_STATISTIC_SCENES = tuple(f"scene-{index:02d}" for index in range(11))
_TRUSTED_COHORT = TrustedCohort(
    tuple(
        (
            ordinal,
            f"observation-{ordinal:02d}",
            _STATISTIC_SCENES[ordinal % 11],
        )
        for ordinal in range(50)
    )
)


def _endpoint_rows(
    *,
    value_scale: float = 0.1,
) -> tuple[EndpointRow, ...]:
    return tuple(
        EndpointRow(
            ordinal=ordinal,
            observation_id=f"observation-{ordinal:02d}",
            scene_id=_STATISTIC_SCENES[ordinal % 11],
            endpoint=(ordinal % 11) * value_scale,
        )
        for ordinal in range(50)
    )


def _cohort_mismatch(kind: str) -> tuple[EndpointRow, ...]:
    rows = list(_endpoint_rows(value_scale=0.1))
    if kind == "changed observation ID":
        rows[0] = replace(rows[0], observation_id="changed-observation")
    elif kind == "scene reassignment":
        rows[0] = replace(rows[0], scene_id=rows[1].scene_id)
    elif kind == "reorder with repaired ordinals":
        rows[0], rows[1] = replace(rows[1], ordinal=0), replace(rows[0], ordinal=1)
    elif kind == "self-consistent substitute":
        rows = [
            replace(row, observation_id=f"substitute-{row.ordinal:02d}") for row in rows
        ]
    else:
        raise AssertionError(f"unknown mismatch kind: {kind}")
    return tuple(rows)


def test_linear_quantile_and_exact_scene_bootstrap_matrix_are_frozen() -> None:
    assert linear_quantile((0.0, 10.0, 20.0, 30.0), 0.0) == 0
    assert linear_quantile((0.0, 10.0, 20.0, 30.0), 0.25) == 7.5
    assert linear_quantile((0.0, 10.0, 20.0, 30.0), 0.5) == 15
    assert linear_quantile((0.0, 10.0, 20.0, 30.0), 1.0) == 30

    matrix = scene_bootstrap_matrix()
    assert matrix.dtype == np.dtype("<i8")
    assert matrix.shape == (10_000, 11)
    assert matrix.flags.c_contiguous
    assert not matrix.flags.writeable
    assert matrix[0].tolist() == [7, 6, 10, 0, 10, 1, 6, 10, 5, 3, 6]
    assert matrix[1].tolist() == [1, 7, 7, 6, 3, 6, 2, 8, 0, 2, 10]
    assert matrix[-1].tolist() == [5, 7, 9, 0, 5, 8, 2, 1, 4, 8, 6]
    assert hashlib.sha256(matrix.tobytes(order="C")).hexdigest() == (
        "a89573633a8efd5dac5ffe03f292491a8aba5202ff2951f7ac186f07f2c07f99"
    )
    independently_generated = np.random.Generator(np.random.PCG64(20260728)).integers(
        0, 11, size=(10_000, 11), endpoint=False
    )
    independently_generated = np.ascontiguousarray(independently_generated, dtype="<i8")
    assert np.array_equal(matrix, independently_generated)

    mutated = matrix.copy()
    mutated[0, 0] = (mutated[0, 0] + 1) % 11
    with pytest.raises(ValueError, match="bootstrap matrix"):
        estimate_scene_robustness(
            _endpoint_rows(value_scale=0.1),
            trusted_cohort=_TRUSTED_COHORT,
            matrix=mutated,
        )


@pytest.mark.parametrize(
    "kind",
    (
        "changed observation ID",
        "scene reassignment",
        "reorder with repaired ordinals",
        "self-consistent substitute",
    ),
)
def test_statistics_bind_every_candidate_to_the_trusted_cohort(kind: str) -> None:
    mismatched = _cohort_mismatch(kind)
    trusted_rows = _endpoint_rows(value_scale=0.05)

    with pytest.raises(ValueError, match="trusted cohort"):
        estimate_scene_robustness(
            mismatched,
            trusted_cohort=_TRUSTED_COHORT,
        )
    with pytest.raises(ValueError, match="trusted cohort"):
        estimate_scene_contrast(
            mismatched,
            trusted_rows,
            trusted_cohort=_TRUSTED_COHORT,
        )
    with pytest.raises(ValueError, match="trusted cohort"):
        estimate_scene_contrast(
            trusted_rows,
            mismatched,
            trusted_cohort=_TRUSTED_COHORT,
        )


def test_scene_robustness_weights_all_rows_and_duplicate_scene_draws() -> None:
    rows = _endpoint_rows(value_scale=0.1)
    matrix = scene_bootstrap_matrix()
    scene_sums = np.asarray([
        sum(cast(float, row.endpoint) for row in rows if row.scene_id == scene)
        for scene in _STATISTIC_SCENES
    ])
    scene_counts = np.asarray([
        sum(row.scene_id == scene for row in rows) for scene in _STATISTIC_SCENES
    ])
    replicates = scene_sums[matrix].sum(axis=1) / scene_counts[matrix].sum(axis=1)
    leave_one_out = tuple(
        np.delete(scene_sums, index).sum() / np.delete(scene_counts, index).sum()
        for index in range(11)
    )

    estimate = estimate_scene_robustness(
        rows,
        trusted_cohort=_TRUSTED_COHORT,
        matrix=matrix,
    )

    assert isinstance(estimate, RobustnessEstimate)
    assert estimate.point_estimate == pytest.approx(scene_sums.sum() / 50)
    assert estimate.interval_low == pytest.approx(linear_quantile(replicates, 0.025))
    assert estimate.interval_high == pytest.approx(linear_quantile(replicates, 0.975))
    assert estimate.leave_one_scene_out_min == pytest.approx(min(leave_one_out))
    assert estimate.leave_one_scene_out_max == pytest.approx(max(leave_one_out))


def test_contrast_uses_separate_candidate_eligibility_and_ordered_direction() -> None:
    rows_a = list(_endpoint_rows(value_scale=0.1))
    rows_b = list(_endpoint_rows(value_scale=0.05))
    # Candidate A has a target-empty false positive (eligible zero); candidate
    # B is empty on that row. The opposite eligibility holds on row 1.
    rows_a[0] = replace(rows_a[0], endpoint=0.0)
    rows_b[0] = replace(rows_b[0], endpoint=None)
    rows_a[1] = replace(rows_a[1], endpoint=None)
    rows_b[1] = replace(rows_b[1], endpoint=0.0)

    estimate = estimate_scene_contrast(
        tuple(rows_a),
        tuple(rows_b),
        trusted_cohort=_TRUSTED_COHORT,
    )
    reverse = estimate_scene_contrast(
        tuple(rows_b),
        tuple(rows_a),
        trusted_cohort=_TRUSTED_COHORT,
    )

    assert isinstance(estimate, ContrastEstimate)
    matrix = scene_bootstrap_matrix()

    def candidate_arrays(
        rows: Sequence[EndpointRow],
    ) -> tuple[np.ndarray, np.ndarray]:
        sums = np.asarray([
            sum(
                row.endpoint
                for row in rows
                if row.scene_id == scene and row.endpoint is not None
            )
            for scene in _STATISTIC_SCENES
        ])
        counts = np.asarray([
            sum(row.scene_id == scene and row.endpoint is not None for row in rows)
            for scene in _STATISTIC_SCENES
        ])
        return sums, counts

    sums_a, counts_a = candidate_arrays(rows_a)
    sums_b, counts_b = candidate_arrays(rows_b)
    replicate_contrasts = sums_a[matrix].sum(axis=1) / counts_a[matrix].sum(
        axis=1
    ) - sums_b[matrix].sum(axis=1) / counts_b[matrix].sum(axis=1)
    loo_contrasts = np.asarray([
        np.delete(sums_a, index).sum() / np.delete(counts_a, index).sum()
        - np.delete(sums_b, index).sum() / np.delete(counts_b, index).sum()
        for index in range(11)
    ])
    expected_point = np.mean([
        row.endpoint for row in rows_a if row.endpoint is not None
    ]) - np.mean([row.endpoint for row in rows_b if row.endpoint is not None])
    assert estimate.point_estimate == pytest.approx(expected_point)
    assert estimate.interval_low == pytest.approx(
        linear_quantile(replicate_contrasts, 0.025)
    )
    assert estimate.interval_high == pytest.approx(
        linear_quantile(replicate_contrasts, 0.975)
    )
    assert estimate.leave_one_scene_out_min == pytest.approx(loo_contrasts.min())
    assert estimate.leave_one_scene_out_max == pytest.approx(loo_contrasts.max())
    intersection_delta = np.mean([
        row_a.endpoint - row_b.endpoint
        for row_a, row_b in zip(rows_a, rows_b)
        if row_a.endpoint is not None and row_b.endpoint is not None
    ])
    assert estimate.point_estimate != pytest.approx(intersection_delta)
    delta_sums = np.asarray([
        sum(
            row_a.endpoint - row_b.endpoint
            for row_a, row_b in zip(rows_a, rows_b)
            if row_a.scene_id == scene
            and row_a.endpoint is not None
            and row_b.endpoint is not None
        )
        for scene in _STATISTIC_SCENES
    ])
    delta_counts = np.asarray([
        sum(
            row_a.scene_id == scene
            and row_a.endpoint is not None
            and row_b.endpoint is not None
            for row_a, row_b in zip(rows_a, rows_b)
        )
        for scene in _STATISTIC_SCENES
    ])
    wrong_delta_replicates = delta_sums[matrix].sum(axis=1) / delta_counts[matrix].sum(
        axis=1
    )
    wrong_delta_loo = np.asarray([
        np.delete(delta_sums, index).sum() / np.delete(delta_counts, index).sum()
        for index in range(11)
    ])
    assert estimate.interval_low != pytest.approx(
        linear_quantile(wrong_delta_replicates, 0.025)
    )
    assert estimate.leave_one_scene_out_max != pytest.approx(wrong_delta_loo.max())
    assert reverse.point_estimate == pytest.approx(-estimate.point_estimate)
    assert reverse.interval_low == pytest.approx(-estimate.interval_high)
    assert reverse.interval_high == pytest.approx(-estimate.interval_low)
    assert reverse.leave_one_scene_out_min == pytest.approx(
        -estimate.leave_one_scene_out_max
    )
    assert reverse.leave_one_scene_out_max == pytest.approx(
        -estimate.leave_one_scene_out_min
    )

    mismatched = list(rows_b)
    mismatched[0] = replace(mismatched[0], observation_id="other")
    with pytest.raises(ValueError, match="identities"):
        estimate_scene_contrast(
            tuple(rows_a),
            tuple(mismatched),
            trusted_cohort=_TRUSTED_COHORT,
        )


def test_scene_statistics_fail_closed_on_zero_eligible_compositions() -> None:
    empty = tuple(replace(row, endpoint=None) for row in _endpoint_rows())
    with pytest.raises(ValueError, match="eligible"):
        estimate_scene_robustness(empty, trusted_cohort=_TRUSTED_COHORT)

    one_scene = tuple(
        replace(row, endpoint=0.5 if row.scene_id == _STATISTIC_SCENES[0] else None)
        for row in _endpoint_rows()
    )
    with pytest.raises(ValueError, match="eligible"):
        estimate_scene_robustness(one_scene, trusted_cohort=_TRUSTED_COHORT)
    with pytest.raises(ValueError, match="eligible"):
        estimate_scene_contrast(
            _endpoint_rows(value_scale=0.1),
            empty,
            trusted_cohort=_TRUSTED_COHORT,
        )


def _complete_gate_rows() -> tuple[EndpointRow, ...]:
    rows = list(_endpoint_rows(value_scale=0.01))
    rows[7] = replace(
        rows[7],
        status=ObservationStatus.FAILED,
        failure_code=ObservationFailureCode.INFERENCE_FAILURE,
    )
    return tuple(rows)


def _provenance_checks() -> ProvenanceChecks:
    return ProvenanceChecks(
        clean_experiment_commit=True,
        complete_command=True,
        raw_package_pinned=True,
        cohort_pinned=True,
        candidate_revision_pinned=True,
        checkpoint_hash_pinned=True,
        environment_pinned=True,
        mapping_pinned=True,
        projector_pinned=True,
        preprocessing_pinned=True,
        precision_pinned=True,
        permission_evidence_pinned=True,
    )


def _evaluate_gates(**overrides: object) -> CandidateGateResult:
    values: dict[str, object] = {
        "synthetic": False,
        "coverage": StaticCoverage(14, 80, 100),
        "trusted_cohort": _TRUSTED_COHORT,
        "rows": _complete_gate_rows(),
        "mean_iou": 0.15,
        "mean_f1": 0.25,
        "latency": LatencySummary(
            timing_comparable=True,
            unit="seconds",
            sample_count=100,
            p95_seconds=1.0,
        ),
        "resource": ResourceMeasurement(
            baseline_allocated_bytes=0,
            peak_allocated_bytes=MAX_ABSOLUTE_RESERVED_BYTES,
            baseline_reserved_bytes=0,
            peak_reserved_bytes=MAX_ABSOLUTE_RESERVED_BYTES,
        ),
        "code_license_status": LicenseStatus.PASS,
        "weight_license_status": LicenseStatus.PASS,
        "provenance": _provenance_checks(),
    }
    values.update(overrides)
    return evaluate_candidate_gates(**values)


def test_candidate_gates_use_exact_inclusive_thresholds_and_reserved_memory() -> None:
    assert (MIN_COVERED_CATEGORIES, MIN_SUPPORT_COVERAGE) == (14, 0.8)
    assert (MIN_MEAN_IOU, MIN_MEAN_F1) == (0.15, 0.25)
    assert MAX_LATENCY_P95_SECONDS == 1.0
    assert MAX_ABSOLUTE_RESERVED_BYTES == 16 * 1024**3
    assert _evaluate_gates().overall is GateStatus.PASS

    assert (
        _evaluate_gates(coverage=StaticCoverage(13, 80, 100)).coverage
        is GateStatus.FAIL
    )
    assert (
        _evaluate_gates(coverage=StaticCoverage(13, 80, 100)).quality is GateStatus.FAIL
    )
    assert (
        _evaluate_gates(coverage=StaticCoverage(14, 79, 100)).coverage
        is GateStatus.FAIL
    )
    assert _evaluate_gates(mean_iou=np.nextafter(0.15, 0.0)).quality is GateStatus.FAIL
    assert _evaluate_gates(mean_f1=np.nextafter(0.25, 0.0)).quality is GateStatus.FAIL
    assert (
        _evaluate_gates(
            latency=LatencySummary(
                timing_comparable=True,
                unit="seconds",
                sample_count=100,
                p95_seconds=np.nextafter(1.0, math.inf),
            )
        ).latency
        is GateStatus.FAIL
    )
    assert (
        _evaluate_gates(
            resource=ResourceMeasurement(
                baseline_allocated_bytes=MAX_ABSOLUTE_RESERVED_BYTES - 1,
                peak_allocated_bytes=MAX_ABSOLUTE_RESERVED_BYTES,
                baseline_reserved_bytes=MAX_ABSOLUTE_RESERVED_BYTES,
                peak_reserved_bytes=MAX_ABSOLUTE_RESERVED_BYTES + 1,
            )
        ).resource
        is GateStatus.FAIL
    )
    assert (
        _evaluate_gates(
            latency=LatencySummary(
                timing_comparable=False,
                unit="synthetic-tick",
                sample_count=0,
                p95_seconds=None,
            )
        ).latency
        is GateStatus.FAIL
    )


def test_candidate_gates_require_complete_rows_licenses_and_provenance() -> None:
    complete = _complete_gate_rows()
    assert _evaluate_gates(rows=complete).complete_rows is GateStatus.PASS
    assert _evaluate_gates(rows=complete[:-1]).complete_rows is GateStatus.FAIL
    assert _evaluate_gates(rows=complete[:-1]).quality is GateStatus.FAIL
    duplicate = list(complete)
    duplicate[-1] = replace(duplicate[-1], ordinal=48)
    assert _evaluate_gates(rows=tuple(duplicate)).complete_rows is GateStatus.FAIL
    assert _evaluate_gates(rows=complete + (complete[-1],)).complete_rows is (
        GateStatus.FAIL
    )
    reordered = list(complete)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert _evaluate_gates(rows=tuple(reordered)).complete_rows is GateStatus.FAIL

    for code, weight in (
        (LicenseStatus.FAIL, LicenseStatus.PASS),
        (LicenseStatus.PASS, LicenseStatus.FAIL),
        (LicenseStatus.FAIL, LicenseStatus.FAIL),
        (LicenseStatus.NOT_APPLICABLE, LicenseStatus.PASS),
        (LicenseStatus.PASS, LicenseStatus.NOT_APPLICABLE),
    ):
        assert (
            _evaluate_gates(
                code_license_status=code,
                weight_license_status=weight,
            ).license
            is GateStatus.FAIL
        )

    provenance = _provenance_checks()
    for field in fields(ProvenanceChecks):
        failed = replace(provenance, **{field.name: False})
        assert _evaluate_gates(provenance=failed).provenance is GateStatus.FAIL

    with pytest.raises(ValueError, match="synthetic"):
        _evaluate_gates(synthetic=True)


@pytest.mark.parametrize(
    "kind",
    (
        "changed observation ID",
        "scene reassignment",
        "reorder with repaired ordinals",
        "self-consistent substitute",
    ),
)
def test_gate_completeness_requires_exact_trusted_cohort(kind: str) -> None:
    result = _evaluate_gates(rows=_cohort_mismatch(kind))
    assert result.complete_rows is GateStatus.FAIL
    assert result.quality is GateStatus.FAIL
    assert result.overall is GateStatus.FAIL


def test_task3a_gate_scope_rejects_synthetic_candidates_structurally() -> None:
    # Task3B owns real timing/no-latency/GPU fields. Task4 owns the literal
    # manifest decision/field-absence checks; Task3A intentionally defines no
    # package schema for either concern.
    with pytest.raises(ValueError, match="synthetic"):
        _evaluate_gates(synthetic=True)


def test_gate_boundaries_reject_bool_nonfinite_and_invalid_latency_claims() -> None:
    with pytest.raises(ValueError, match="coverage"):
        StaticCoverage(cast(int, True), 80, 100)
    with pytest.raises(ValueError, match="endpoint"):
        EndpointRow(0, "observation", "scene", cast(float, True))
    with pytest.raises(ValueError, match="endpoint"):
        EndpointRow(0, "observation", "scene", float("nan"))
    with pytest.raises(ValueError, match="quality"):
        _evaluate_gates(mean_iou=True)
    with pytest.raises(ValueError, match="quality"):
        _evaluate_gates(mean_f1=float("nan"))
    with pytest.raises(ValueError, match="comparable latency"):
        LatencySummary(True, "milliseconds", 100, 1.0)
    with pytest.raises(ValueError, match="comparable latency"):
        LatencySummary(True, "seconds", 99, 1.0)
    with pytest.raises(ValueError, match="comparable latency"):
        LatencySummary(True, "seconds", 100, cast(float, True))
    with pytest.raises(ValueError, match="non-comparable"):
        LatencySummary(False, "synthetic-tick", 100, 1.0)
    with pytest.raises(ValueError, match="provenance"):
        replace(_provenance_checks(), complete_command=cast(bool, 1))


def test_runner_is_the_only_device_batch_producer_and_cpu_is_supported() -> None:
    transform = SpatialTransform.from_sizes(
        raw_height=256, raw_width=256, model_height=256, model_width=256
    )
    prepared = PreparedHostBatch((torch.ones(1),), transform)
    batch = transfer_prepared_host_batch(
        prepared, device=torch.device("cpu"), non_blocking=False
    )
    assert isinstance(batch, DeviceBatch)
    assert batch.tensors[0].device.type == "cpu"
    with pytest.raises(ValueError, match="benchmark runner"):
        DeviceBatch((torch.ones(1),), transform, object())


def _synthetic_commitment() -> CandidateCommitment:
    return CandidateCommitment(
        candidate_id="synthetic-contract",
        synthetic=True,
        repository_url="NOT_APPLICABLE",
        revision="NOT_APPLICABLE",
        checkpoint_id="NOT_APPLICABLE",
        checkpoint_url="NOT_APPLICABLE",
        checkpoint_sha256="NOT_APPLICABLE",
        source_dataset="NOT_APPLICABLE",
        code_license_status=LicenseStatus.NOT_APPLICABLE,
        weight_license_status=LicenseStatus.NOT_APPLICABLE,
        environment_lock_path="requirements.txt",
        environment_lock_sha256="a" * 64,
        rgb_units="uint8[0,255]",
        depth_units="float32-metres[0,10]",
        batch_views=12,
        source_vocabulary=("wall",),
        mapping_sha256="b" * 64,
        rgb_interpolation="bilinear-align-corners-false",
        rgb_coordinate_semantics="half-pixel",
        depth_interpolation="nearest-exact",
        depth_coordinate_semantics="half-pixel",
        normalization="none",
        invalid_depth_policy="preserve-zero",
        rgb_padding_value="0",
        depth_padding_value="0",
        precision_mode="float32",
    )


def _real_commitment() -> CandidateCommitment:
    return replace(
        _synthetic_commitment(),
        candidate_id="esanet-r34",
        synthetic=False,
        repository_url="https://example.test/repo",
        revision="c" * 40,
        checkpoint_id="nyuv2-r34",
        checkpoint_url="https://example.test/weights",
        checkpoint_sha256="d" * 64,
        source_dataset="NYUv2",
        code_license_status=LicenseStatus.PASS,
        weight_license_status=LicenseStatus.FAIL,
    )


def test_candidate_commitment_has_coherent_license_source_env_and_batch_fields() -> (
    None
):
    commitment = _synthetic_commitment()
    assert commitment.batch_views == 12
    with pytest.raises(ValueError, match="all be N/A"):
        replace(commitment, checkpoint_url="https://example.test/checkpoint")
    with pytest.raises(ValueError, match="twelve"):
        replace(commitment, batch_views=1)
    with pytest.raises(ValueError, match="input units"):
        replace(commitment, depth_units="millimetres")
    with pytest.raises(ValueError, match="real candidate"):
        replace(
            commitment,
            synthetic=False,
            repository_url="https://example.test/repo",
            revision="c" * 40,
            checkpoint_id="weights",
            checkpoint_url="https://example.test/weights",
            checkpoint_sha256="d" * 64,
            source_dataset="NYUv2",
        )
    assert _real_commitment().weight_license_status is LicenseStatus.FAIL


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_id", "../escape", "candidate ID"),
        ("candidate_id", "bad\x00id", "candidate ID"),
        ("repository_url", "NOT_APPLICABLE", "real candidate"),
        ("revision", "NOT_APPLICABLE", "real candidate"),
        ("checkpoint_id", "NOT_APPLICABLE", "real candidate"),
        ("checkpoint_url", "NOT_APPLICABLE", "real candidate"),
        ("checkpoint_sha256", "NOT_APPLICABLE", "real candidate"),
        ("source_dataset", "NOT_APPLICABLE", "real candidate"),
        ("environment_lock_path", "NOT_APPLICABLE", "lock path"),
        ("environment_lock_path", "../requirements.txt", "lock path"),
        ("environment_lock_path", "locks\\requirements.txt", "lock path"),
        ("environment_lock_path", "locks/\x00requirements.txt", "lock path"),
        ("environment_lock_sha256", "NOT_APPLICABLE", "lock hash"),
        ("rgb_interpolation", "NOT_APPLICABLE", "real candidate"),
        ("rgb_coordinate_semantics", "NOT_APPLICABLE", "real candidate"),
        ("depth_interpolation", "NOT_APPLICABLE", "real candidate"),
        ("depth_coordinate_semantics", "NOT_APPLICABLE", "real candidate"),
        ("normalization", "NOT_APPLICABLE", "real candidate"),
        ("invalid_depth_policy", "NOT_APPLICABLE", "real candidate"),
        ("rgb_padding_value", "NOT_APPLICABLE", "real candidate"),
        ("depth_padding_value", "NOT_APPLICABLE", "real candidate"),
        ("precision_mode", "NOT_APPLICABLE", "real candidate"),
        ("code_license_status", LicenseStatus.NOT_APPLICABLE, "real candidate"),
        ("weight_license_status", LicenseStatus.NOT_APPLICABLE, "real candidate"),
    ],
)
def test_real_candidate_rejects_unsafe_or_missing_provenance(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_real_commitment(), **{field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_forbids_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": value})


def _p53_attestation(root: Path) -> P53ValidationAttestation:
    return P53ValidationAttestation(
        python_executable=str(Path(sys.executable).resolve()),
        environment_sha256="a" * 64,
        gpu_device_id=0,
        gpu_uuid=RAW_GPU_UUID,
        producer_git_commit=RAW_PRODUCER_COMMIT,
        raw_root=str(root.resolve()),
        raw_manifest_sha256=RAW_MANIFEST_SHA256,
        raw_index_sha256=RAW_INDEX_SHA256,
        validator_source_sha256=RAW_VALIDATOR_SOURCE_SHA256,
    )


def _benchmark_attestation() -> BenchmarkEnvironmentAttestation:
    return BenchmarkEnvironmentAttestation(
        python_executable=str(Path(sys.executable).resolve()),
        environment_sha256=capture_environment_sha256(),
        visible_device_count=torch.cuda.device_count(),
        gpu_name="NOT_APPLICABLE",
        gpu_uuid="NOT_APPLICABLE",
        timing_comparable=False,
    )


def test_path_pinned_validator_launcher_is_canonical_and_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    attestation = _p53_attestation(root)
    seen: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def runner(
        args: Sequence[str],
        *,
        env: Mapping[str, str],
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        assert capture_output and not check and timeout == 9
        seen.append((tuple(args), dict(env)))
        return subprocess.CompletedProcess(args, 0, attestation.canonical_bytes(), b"")

    launch = P53ValidatorLaunch(
        python_executable=Path(sys.executable).resolve(),
        expected_environment_sha256="a" * 64,
        raw_root=root,
        timeout_seconds=9,
    )
    assert run_p53_validation_subprocess(launch, runner=runner) == attestation
    assert seen[0][0][0] == str(Path(sys.executable).resolve())
    assert "CUDA_VISIBLE_DEVICES" not in seen[0][1]

    def noisy_runner(
        args: Sequence[str],
        *,
        env: Mapping[str, str],
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del env, capture_output, check, timeout
        return subprocess.CompletedProcess(args, 0, attestation.canonical_bytes(), b"x")

    with pytest.raises(ValueError, match="stderr"):
        run_p53_validation_subprocess(launch, runner=noisy_runner)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("nonzero", "exit successfully"),
        ("signal", "exit successfully"),
        ("empty", "valid UTF-8 JSON"),
        ("multiple", "valid UTF-8 JSON"),
        ("malformed", "valid UTF-8 JSON"),
        ("environment", "launcher pins"),
        ("interpreter", "launcher pins"),
        ("root", "launcher pins"),
        ("producer", "launcher pins"),
        ("manifest", "launcher pins"),
        ("index", "launcher pins"),
        ("source", "identity"),
        ("bool-schema", "must be an integer"),
    ],
)
def test_validator_subprocess_rejects_every_untrusted_outcome(
    mode: str, message: str, tmp_path: Path
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    good = _p53_attestation(root)
    launch = P53ValidatorLaunch(
        python_executable=Path(sys.executable).resolve(),
        expected_environment_sha256="a" * 64,
        raw_root=root,
        timeout_seconds=9,
    )

    def runner(
        args: Sequence[str],
        *,
        env: Mapping[str, str],
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del env, capture_output, check, timeout
        returncode = -9 if mode == "signal" else 2 if mode == "nonzero" else 0
        if mode == "empty":
            output = b""
        elif mode == "multiple":
            output = good.canonical_bytes() * 2
        elif mode == "malformed":
            output = b"{"
        elif mode == "environment":
            output = replace(good, environment_sha256="e" * 64).canonical_bytes()
        elif mode == "interpreter":
            output = replace(good, python_executable="/bin/false").canonical_bytes()
        elif mode == "root":
            output = replace(good, raw_root="/tmp").canonical_bytes()
        elif mode == "producer":
            output = replace(good, producer_git_commit="1" * 40).canonical_bytes()
        elif mode == "manifest":
            output = replace(good, raw_manifest_sha256="2" * 64).canonical_bytes()
        elif mode == "index":
            output = replace(good, raw_index_sha256="3" * 64).canonical_bytes()
        elif mode in {"source", "bool-schema"}:
            raw = json.loads(good.canonical_bytes())
            raw["validator_source_sha256" if mode == "source" else "schema_version"] = (
                "f" * 64 if mode == "source" else True
            )
            output = canonical_json_bytes(raw)
        else:
            output = good.canonical_bytes()
        return subprocess.CompletedProcess(args, returncode, output, b"")

    with pytest.raises(ValueError, match=message):
        run_p53_validation_subprocess(launch, runner=runner)


def test_validator_subprocess_timeout_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    launch = P53ValidatorLaunch(
        python_executable=Path(sys.executable).resolve(),
        expected_environment_sha256="a" * 64,
        raw_root=root,
        timeout_seconds=9,
    )

    def runner(
        args: Sequence[str],
        *,
        env: Mapping[str, str],
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del env, capture_output, check
        raise subprocess.TimeoutExpired(args, timeout)

    with pytest.raises(ValueError, match="timed out"):
        run_p53_validation_subprocess(launch, runner=runner)


@_RAW_INTEGRATION
def test_real_raw_package_integration_binds_order_and_adapter_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = RAW_FRAME_ROOT.resolve()
    p53 = _p53_attestation(root)
    benchmark = _benchmark_attestation()
    monkeypatch.setattr(
        os,
        "listdir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reader must use bounded scandir")
        ),
    )
    ordinals: list[int] = []
    scenes: list[str] = []
    observations = iter_validated_raw_observations(
        root,
        p53_attestation=p53,
        expected_p53_attestation_sha256=p53.sha256,
        benchmark_attestation=benchmark,
        expected_benchmark_attestation_sha256=benchmark.sha256,
    )
    assert isinstance(observations, tuple)
    for observation in observations:
        ordinals.append(observation.row.ordinal)
        scenes.append(observation.row.scene_id)
        assert set(vars(observation.segmenter_input)) == {"rgb", "depth_m"}
        assert observation.segmenter_input.rgb.is_pinned()
    assert ordinals == list(range(50))
    assert scenes == sorted(scenes)

    with pytest.raises(ValueError, match="trusted launcher"):
        tuple(
            iter_validated_raw_observations(
                root,
                p53_attestation=p53,
                expected_p53_attestation_sha256="d" * 64,
                benchmark_attestation=benchmark,
                expected_benchmark_attestation_sha256=benchmark.sha256,
            )
        )
    with pytest.raises(ValueError, match="exactly one recorded RTX 3090"):
        BenchmarkEnvironmentAttestation(
            python_executable=str(Path(sys.executable).resolve()),
            environment_sha256="c" * 64,
            visible_device_count=2,
            gpu_name="NVIDIA GeForce RTX 3090",
            gpu_uuid="GPU-candidate",
            timing_comparable=True,
        )


def test_comparable_environment_verifies_same_count_gpu_name_and_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    attestation = BenchmarkEnvironmentAttestation(
        python_executable=str(Path(sys.executable).resolve()),
        environment_sha256=capture_environment_sha256(),
        visible_device_count=1,
        gpu_name="NVIDIA GeForce RTX 3090",
        gpu_uuid=RAW_GPU_UUID,
        timing_comparable=True,
    )
    attestation.require_current_process(
        gpu_inspector=lambda: ("NVIDIA GeForce RTX 3090", RAW_GPU_UUID)
    )
    with pytest.raises(ValueError, match="GPU identity"):
        attestation.require_current_process(
            gpu_inspector=lambda: ("NVIDIA GeForce RTX 3090", "GPU-changed")
        )
    with pytest.raises(ValueError, match="recorded RTX 3090"):
        replace(attestation, gpu_name="NVIDIA A100")


def test_gpu_inventory_requires_one_full_uuid_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def runner(
        args: Sequence[str],
        *,
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del capture_output, check, timeout
        stdout = b"4, NVIDIA GeForce RTX 3090, " + RAW_GPU_UUID.encode("ascii") + b"\n"
        return subprocess.CompletedProcess(args, 0, stdout, b"")

    monkeypatch.setattr(subprocess, "run", runner)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", RAW_GPU_UUID)
    assert inspect_visible_gpu() == ("NVIDIA GeForce RTX 3090", RAW_GPU_UUID)
    for invalid in ("4", f"{RAW_GPU_UUID},GPU-other", "GPU-missing"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", invalid)
        with pytest.raises(ValueError, match="UUID|index"):
            inspect_visible_gpu()


def test_result_timing_and_resource_schemas_reject_inconsistent_state() -> None:
    empty = np.full((12, 256, 256), -1, dtype="<i2")
    prediction = Prediction(empty.copy(), empty.copy())
    ObservationResult(
        ordinal=0,
        status=ObservationStatus.FAILED,
        failure_code=ObservationFailureCode.INFERENCE_FAILURE,
        prediction=prediction,
    )
    TimingSample(
        pass_index=2,
        ordinal=49,
        end_to_end=0.0,
        source_labels_sha256="a" * 64,
        mapped_labels_sha256="b" * 64,
        status=ObservationStatus.PASS,
        failure_code=None,
    )
    ResourceMeasurement(1, 3, 2, 4)
    with pytest.raises(ValueError, match="status"):
        ObservationResult(
            ordinal=0,
            status=ObservationStatus.PASS,
            failure_code=ObservationFailureCode.INFERENCE_FAILURE,
            prediction=prediction,
        )
    with pytest.raises(ValueError, match="empty"):
        ObservationResult(
            ordinal=0,
            status=ObservationStatus.FAILED,
            failure_code=ObservationFailureCode.INFERENCE_FAILURE,
            prediction=Prediction(np.zeros((12, 256, 256), dtype="<i2"), empty.copy()),
        )
    with pytest.raises(ValueError, match="inconsistent"):
        ResourceMeasurement(3, 2, 3, 2)
    with pytest.raises(ValueError, match="pass"):
        TimingSample(
            pass_index=cast(int, True),
            ordinal=0,
            end_to_end=0.0,
            source_labels_sha256="a" * 64,
            mapped_labels_sha256="b" * 64,
            status=ObservationStatus.PASS,
            failure_code=None,
        )


@_RAW_INTEGRATION
def test_reader_rejects_symlink_tree_after_accepting_pins(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    source = RAW_FRAME_ROOT.resolve()
    (root / "manifest.json").write_bytes((source / "manifest.json").read_bytes())
    (root / "index.jsonl").write_bytes((source / "index.jsonl").read_bytes())
    (root / "observations").symlink_to(
        source / "observations", target_is_directory=True
    )
    p53 = _p53_attestation(root)
    benchmark = _benchmark_attestation()
    with pytest.raises(ValueError, match="non-regular"):
        tuple(
            iter_validated_raw_observations(
                root,
                p53_attestation=p53,
                expected_p53_attestation_sha256=p53.sha256,
                benchmark_attestation=benchmark,
                expected_benchmark_attestation_sha256=benchmark.sha256,
            )
        )


def test_reader_rejects_oversized_file_before_reading_payload(tmp_path: Path) -> None:
    root = tmp_path / "oversized"
    root.mkdir()
    with (root / "manifest.json").open("wb") as stream:
        stream.truncate(56_532)
    p53 = _p53_attestation(root)
    benchmark = _benchmark_attestation()
    with pytest.raises(ValueError, match="wrong byte length"):
        iter_validated_raw_observations(
            root,
            p53_attestation=p53,
            expected_p53_attestation_sha256=p53.sha256,
            benchmark_attestation=benchmark,
            expected_benchmark_attestation_sha256=benchmark.sha256,
        )


@_RAW_INTEGRATION
def test_reader_rejects_missing_tree_entries(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    root.mkdir()
    source = RAW_FRAME_ROOT.resolve()
    (root / "manifest.json").write_bytes((source / "manifest.json").read_bytes())
    (root / "index.jsonl").write_bytes((source / "index.jsonl").read_bytes())
    (root / "observations").mkdir()
    p53 = _p53_attestation(root)
    benchmark = _benchmark_attestation()
    with pytest.raises(ValueError, match="missing, extra"):
        iter_validated_raw_observations(
            root,
            p53_attestation=p53,
            expected_p53_attestation_sha256=p53.sha256,
            benchmark_attestation=benchmark,
            expected_benchmark_attestation_sha256=benchmark.sha256,
        )


@_RAW_INTEGRATION
def test_reader_bounds_tree_entry_count_and_depth(tmp_path: Path) -> None:
    source = RAW_FRAME_ROOT.resolve()
    benchmark = _benchmark_attestation()
    for variant in ("count", "depth"):
        root = tmp_path / variant
        root.mkdir()
        (root / "manifest.json").write_bytes((source / "manifest.json").read_bytes())
        (root / "index.jsonl").write_bytes((source / "index.jsonl").read_bytes())
        observations = root / "observations"
        observations.mkdir()
        if variant == "count":
            for index in range(63):
                (observations / f"extra-{index}").write_bytes(b"x")
            message = "entry bound"
        else:
            (observations / "a").mkdir()
            (observations / "a" / "b").mkdir()
            (observations / "a" / "b" / "c").mkdir()
            (observations / "a" / "b" / "c" / "d").mkdir()
            message = "depth bound"
        p53 = _p53_attestation(root)
        with pytest.raises(ValueError, match=message):
            iter_validated_raw_observations(
                root,
                p53_attestation=p53,
                expected_p53_attestation_sha256=p53.sha256,
                benchmark_attestation=benchmark,
                expected_benchmark_attestation_sha256=benchmark.sha256,
            )

    root_link = tmp_path / "root-link"
    root_link.symlink_to(source, target_is_directory=True)
    linked_attestation = _p53_attestation(root_link)
    with pytest.raises(ValueError, match="raw root must not be a symlink"):
        iter_validated_raw_observations(
            root_link,
            p53_attestation=linked_attestation,
            expected_p53_attestation_sha256=linked_attestation.sha256,
            benchmark_attestation=benchmark,
            expected_benchmark_attestation_sha256=benchmark.sha256,
        )
