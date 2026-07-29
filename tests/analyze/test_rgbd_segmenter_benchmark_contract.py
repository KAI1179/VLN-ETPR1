from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
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
    CandidateCommitment,
    ConfusionCounts,
    DeviceBatch,
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
    SegmenterInput,
    SpatialTransform,
    Prediction,
    PreparedHostBatch,
    ResourceMeasurement,
    TimingSample,
    aggregate_observation_metrics,
    capture_environment_sha256,
    canonical_json_bytes,
    iter_validated_raw_observations,
    inspect_visible_gpu,
    load_nyu40_mapping,
    map_source_labels,
    project_mapped_labels,
    run_p53_validation_subprocess,
    restore_source_labels,
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
