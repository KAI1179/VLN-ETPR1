from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, cast

import numpy as np
import pytest
import torch

from prior.analyze.d2026_07_29 import rgbd_segmenter_synthetic as synthetic
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    RAW_GPU_UUID,
    RAW_INDEX_SHA256,
    RAW_MANIFEST_SHA256,
    RAW_PRODUCER_COMMIT,
    RAW_VALIDATOR_SOURCE_SHA256,
    BenchmarkEnvironmentAttestation,
    ComponentTimings,
    ObservationStatus,
    P53ValidationAttestation,
    Prediction,
    SegmenterInput,
    TimingSample,
    TrustedCohort,
    ValidatedRawObservation,
    logical_label_sha256,
    score_observation,
    transfer_prepared_host_batch,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    CandidateValidationAuthority,
    ObservationPackageRow,
    PredictionArtifact,
    SyntheticSuccessfulManifest,
    TimingPackageRow,
    encode_prediction_npz,
    parse_timings_bytes,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    FileRecord as RawFileRecord,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    IndexRow,
    RawFrameArrays,
)


def _input(value: int = 1) -> SegmenterInput:
    rgb = torch.full(
        (12, 256, 256, 3),
        value,
        dtype=torch.uint8,
        pin_memory=True,
    )
    depth = torch.full(
        (12, 256, 256),
        float(value),
        dtype=torch.float32,
        pin_memory=True,
    )
    return SegmenterInput(rgb=rgb, depth_m=depth)


def test_adapter_uses_exact_odd_padding_transform_and_zero_padding() -> None:
    prepared = synthetic.SyntheticSegmenterAdapter().preprocess_host(_input())

    assert prepared.spatial_transform.raw_height == 256
    assert prepared.spatial_transform.raw_width == 256
    assert prepared.spatial_transform.resized_height == 193
    assert prepared.spatial_transform.resized_width == 193
    assert prepared.spatial_transform.model_height == 193
    assert prepared.spatial_transform.model_width == 258
    assert (
        prepared.spatial_transform.pad_left,
        prepared.spatial_transform.pad_right,
    ) == (32, 33)
    rgb, depth = prepared.tensors
    assert rgb.shape == (12, 3, 193, 258)
    assert depth.shape == (12, 1, 193, 258)
    assert torch.count_nonzero(rgb[..., :32]) == 0
    assert torch.count_nonzero(rgb[..., -33:]) == 0
    assert torch.count_nonzero(depth[..., :32]) == 0
    assert torch.count_nonzero(depth[..., -33:]) == 0
    assert bool((rgb[..., 32:-33] > 0).all())
    assert bool((depth[..., 32:-33] > 0).all())


def test_adapter_boundary_and_logits_are_deterministic_and_input_sensitive() -> None:
    adapter = synthetic.SyntheticSegmenterAdapter()
    signature = inspect.signature(adapter.preprocess_host)
    assert tuple(signature.parameters) == ("value",)
    assert signature.parameters["value"].annotation in {
        "SegmenterInput",
        SegmenterInput,
    }
    assert tuple(inspect.signature(adapter.infer).parameters) == ("value",)

    first = transfer_prepared_host_batch(
        adapter.preprocess_host(_input(1)),
        device=torch.device("cpu"),
        non_blocking=False,
    )
    changed = transfer_prepared_host_batch(
        adapter.preprocess_host(_input(2)),
        device=torch.device("cpu"),
        non_blocking=False,
    )
    first_logits = adapter.infer(first)
    repeated_logits = adapter.infer(first)
    changed_logits = adapter.infer(changed)

    assert first_logits.shape == (12, 40, 97, 129)
    assert first_logits.dtype is torch.float32
    assert bool(torch.isfinite(first_logits).all())
    assert torch.equal(first_logits, repeated_logits)
    assert not torch.equal(first_logits, changed_logits)
    assert not vars(adapter)


def test_cli_has_no_configurable_experiment_inputs() -> None:
    assert synthetic.parse_args([])._annotations == {}
    with pytest.raises(SystemExit):
        synthetic.parse_args(["--destination", "/tmp/elsewhere"])


def test_main_rejects_a_masked_p53_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with pytest.raises(ValueError, match="unmasked P5.3"):
        synthetic.main([])


@pytest.mark.parametrize(
    "name", ("synthetic-contract-v1", ".synthetic-contract-v1.staging")
)
def test_early_preflight_refuses_existing_final_or_staging(
    tmp_path: Path,
    name: str,
) -> None:
    parent = tmp_path / "data" / "rgbd_segmenter_benchmark"
    parent.mkdir(parents=True)
    (parent / name).mkdir()
    destination = parent / "synthetic-contract-v1"

    with pytest.raises(FileExistsError, match="publication path already exists"):
        synthetic._require_publication_paths_absent(destination)


def _package_rows(
    root: Path,
) -> tuple[
    TrustedCohort,
    tuple[ObservationPackageRow, ...],
    tuple[TimingPackageRow, ...],
    tuple[tuple[str, PredictionArtifact], ...],
    tuple[np.ndarray, ...],
]:
    source = np.zeros((12, 256, 256), dtype="<i2")
    mapped = np.zeros((12, 256, 256), dtype="<i2")
    prediction = Prediction(source, mapped)
    artifact = encode_prediction_npz(prediction)
    target = np.ones((27, 50, 50), dtype=np.bool_)
    predicted = np.zeros_like(target)
    identities = tuple(
        (ordinal, f"{ordinal:020x}", f"scene-{ordinal % 11:02d}")
        for ordinal in range(50)
    )
    cohort = TrustedCohort(identities)
    rows = tuple(
        ObservationPackageRow(
            ordinal=ordinal,
            observation_id=observation_id,
            scene_id=scene_id,
            status=ObservationStatus.PASS,
            failure_code=None,
            prediction_path=(
                f"predictions/{scene_id}/{ordinal:02d}-{observation_id}.npz"
            ),
            prediction=artifact.file,
            prediction_members=artifact.members,
            metrics=score_observation(
                ordinal=ordinal,
                scene_id=scene_id,
                prediction=predicted,
                target=target,
            ),
        )
        for ordinal, observation_id, scene_id in identities
    )
    components = ComponentTimings(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    source_hash = logical_label_sha256(source)
    mapped_hash = logical_label_sha256(mapped)
    timings = tuple(
        TimingPackageRow(
            TimingSample(
                sequence_index=20 + (pass_index - 1) * 50 + ordinal,
                pass_index=pass_index,
                ordinal=ordinal,
                end_to_end=1.0,
                components=components,
                source_labels_sha256=source_hash,
                mapped_labels_sha256=mapped_hash,
                status=ObservationStatus.PASS,
                failure_code=None,
            )
        )
        for pass_index in (1, 2)
        for ordinal in range(50)
    )
    predictions = tuple((row.prediction_path, artifact) for row in rows)
    return (
        cohort,
        rows,
        timings,
        predictions,
        (target,) * 50,
    )


def test_manifest_is_canonical_53_file_nonclaim_and_binds_source_hashes() -> None:
    root = synthetic._repository_root()
    cohort, rows, timings, raw_predictions, targets = _package_rows(root)
    predictions = raw_predictions
    benchmark = BenchmarkEnvironmentAttestation(
        python_executable=str(Path(sys.executable).resolve(strict=True)),
        environment_sha256="a" * 64,
        visible_device_count=0,
        gpu_name="NOT_APPLICABLE",
        gpu_uuid="NOT_APPLICABLE",
        timing_comparable=False,
    )
    p53 = P53ValidationAttestation(
        python_executable=str(Path(sys.executable).resolve(strict=True)),
        environment_sha256="b" * 64,
        gpu_device_id=0,
        gpu_uuid=RAW_GPU_UUID,
        producer_git_commit=RAW_PRODUCER_COMMIT,
        raw_root=str((root / "data").absolute()),
        raw_manifest_sha256=RAW_MANIFEST_SHA256,
        raw_index_sha256=RAW_INDEX_SHA256,
        validator_source_sha256=RAW_VALIDATOR_SOURCE_SHA256,
    )
    commitment = synthetic._commitment(root)
    fields = synthetic._manifest_fields(
        root=root,
        p53=p53,
        benchmark=benchmark,
        commitment=commitment,
        commit="c" * 40,
        cohort=cohort,
        observations=rows,
        timings=timings,
        predictions=predictions,
        targets=targets,
    )
    manifest = SyntheticSuccessfulManifest(fields)

    assert fields["run_kind"] == "synthetic-contract"
    assert fields["decision"] == "NOT_APPLICABLE"
    assert fields["latency_claim"] is False
    assert "latency" not in fields
    assert "gates" not in fields
    assert "resource" not in fields
    timing = fields["timing_protocol"]
    assert isinstance(timing, dict)
    assert timing == {
        "backend": "deterministic-fake",
        "comparable": False,
        "measured_pass_count": 2,
        "measured_sample_count": 100,
        "unit": "synthetic-tick",
        "warmup_count": 20,
    }
    files = cast(Dict[str, object], fields["files"])
    predictions_section = cast(Dict[str, object], files["predictions"])
    assert predictions_section["file_count"] == 50
    assert 3 + cast(int, predictions_section["file_count"]) == 53
    sources = cast(Dict[str, object], fields["source_hashes"])
    adapter_source = cast(Dict[str, str], sources["adapter"])
    assert adapter_source["path"] == synthetic._ADAPTER_PATH
    assert (
        adapter_source["sha256"]
        == hashlib.sha256((root / synthetic._ADAPTER_PATH).read_bytes()).hexdigest()
    )
    assert manifest.canonical_bytes().endswith(b"\n")


def _raw_arrays() -> RawFrameArrays:
    return RawFrameArrays(
        schema_version=np.asarray(1, dtype="<i8"),
        rgb=np.zeros((12, 256, 256, 3), dtype="|u1"),
        depth_m=np.ones((12, 256, 256), dtype="<f4"),
        object_categories=np.full((12, 256, 256), -1, dtype="<i2"),
        region_categories=np.full((12, 256, 256), -1, dtype="<i2"),
        sensor_positions=np.zeros((12, 3), dtype="<f8"),
        sensor_rotations_xyzw=np.tile(np.asarray((0, 0, 0, 1), dtype="<f8"), (12, 1)),
        sensor_yaw_degrees=np.arange(0, 360, 30, dtype="<i2"),
        sensor_hfov_degrees=np.asarray(90, dtype="<f8"),
        sensor_position_relative=np.asarray((0, 1.25, 0), dtype="<f8"),
        start_position=np.zeros(3, dtype="<f8"),
        start_rotation_xyzw=np.asarray((0, 0, 0, 1), dtype="<f8"),
        target_origin_xz=np.asarray((-25, -25), dtype="<f8"),
        ego_observed_mask=np.zeros((50, 50), dtype="|b1"),
        ego_free_mask=np.zeros((50, 50), dtype="|b1"),
        target_observed_mask=np.zeros((50, 50), dtype="|b1"),
        target_free_mask=np.zeros((50, 50), dtype="|b1"),
    )


def test_build_artifacts_runs_exact_schedule_and_mints_canonical_package() -> None:
    root = synthetic._repository_root()
    segmenter_input = _input(1)
    arrays = _raw_arrays()
    raw = tuple(
        ValidatedRawObservation(
            row=IndexRow(
                artifact=f"frames/{ordinal:020x}.npz",
                cohort_row_sha256="1" * 64,
                members={},
                npz=RawFileRecord(1, "2" * 64),
                observation_id=f"{ordinal:020x}",
                oracle_artifact_sha256="3" * 64,
                ordinal=ordinal,
                scene_id=f"scene-{ordinal % 11:02d}",
            ),
            segmenter_input=segmenter_input,
            audit_arrays=arrays,
        )
        for ordinal in range(50)
    )
    benchmark = BenchmarkEnvironmentAttestation(
        python_executable=str(Path(sys.executable).resolve(strict=True)),
        environment_sha256="4" * 64,
        visible_device_count=0,
        gpu_name="NOT_APPLICABLE",
        gpu_uuid="NOT_APPLICABLE",
        timing_comparable=False,
    )
    p53 = P53ValidationAttestation(
        python_executable=str(Path(sys.executable).resolve(strict=True)),
        environment_sha256="5" * 64,
        gpu_device_id=0,
        gpu_uuid=RAW_GPU_UUID,
        producer_git_commit=RAW_PRODUCER_COMMIT,
        raw_root=str((root / "data").absolute()),
        raw_manifest_sha256=RAW_MANIFEST_SHA256,
        raw_index_sha256=RAW_INDEX_SHA256,
        validator_source_sha256=RAW_VALIDATOR_SOURCE_SHA256,
    )
    predicted = np.zeros((27, 50, 50), dtype=np.bool_)
    target = np.ones((27, 50, 50), dtype=np.bool_)

    artifacts, cohort = synthetic._build_artifacts(
        root=root,
        p53=p53,
        benchmark=benchmark,
        commitment=synthetic._commitment(root),
        commit="6" * 40,
        raw_observations=raw,
        prediction_projector=lambda _labels, _arrays: predicted,
        target_projector=lambda _arrays: target,
    )

    assert cohort.identities == tuple(
        (ordinal, f"{ordinal:020x}", f"scene-{ordinal % 11:02d}")
        for ordinal in range(50)
    )
    assert len(artifacts.files) == 53
    assert artifacts.validation.file_count == 53
    files = dict(artifacts.files)
    timings = parse_timings_bytes(files["timings.jsonl"])
    assert len(timings) == 100
    assert tuple(row.sample.sequence_index for row in timings) == tuple(range(20, 120))
    for ordinal in range(50):
        first = timings[ordinal].sample
        second = timings[50 + ordinal].sample
        assert (
            first.source_labels_sha256,
            first.mapped_labels_sha256,
            first.status,
            first.failure_code,
        ) == (
            second.source_labels_sha256,
            second.mapped_labels_sha256,
            second.status,
            second.failure_code,
        )
    assert artifacts.manifest.summary.timing_protocol.backend == "deterministic-fake"
    assert artifacts.manifest.summary.timing_protocol.comparable is False


def test_main_uses_typed_publication_then_fresh_endpoint_free_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    destination = root / synthetic._DESTINATION
    destination.parent.mkdir(parents=True)
    manifest = SimpleNamespace(canonical_bytes=lambda: b"manifest\n")
    artifacts = SimpleNamespace(manifest=manifest)
    record = SimpleNamespace(
        canonical_bytes=lambda: (
            b'{"observation_count":50,"synthetic_semantic_scores_exposed":false,'
            b'"validation_status":"PASS"}\n'
        )
    )
    published = SimpleNamespace(path=destination, validation=record)
    publisher_calls: list[tuple[Path, object, object]] = []
    fresh_calls: list[tuple[Path, object, bytes]] = []

    monkeypatch.setattr(synthetic, "_repository_root", lambda: root)
    monkeypatch.setattr(synthetic, "_git_commit", lambda _root: "d" * 40)
    monkeypatch.setattr(
        synthetic, "_benchmark_attestation", lambda: SimpleNamespace(sha256="a" * 64)
    )
    monkeypatch.setattr(
        synthetic,
        "_p53_launch",
        lambda _root: SimpleNamespace(raw_root=root / "raw"),
    )
    monkeypatch.setattr(
        synthetic,
        "run_p53_validation_subprocess",
        lambda _launch: SimpleNamespace(sha256="b" * 64),
    )
    monkeypatch.setattr(synthetic, "_commitment", lambda _root: object())
    monkeypatch.setattr(
        synthetic, "iter_validated_raw_observations", lambda *_args, **_kwargs: ()
    )
    monkeypatch.setattr(
        synthetic,
        "_build_artifacts",
        lambda **_kwargs: (artifacts, object()),
    )
    monkeypatch.setattr(
        synthetic,
        "_authority",
        lambda **_kwargs: object(),
    )

    def publish(path: Path, value: object, authority: object) -> object:
        publisher_calls.append((path, value, authority))
        return published

    monkeypatch.setattr(synthetic, "publish_successful_candidate_package", publish)
    monkeypatch.setattr(
        synthetic,
        "_fresh_validation",
        lambda path, authority, expected: (
            fresh_calls.append((path, authority, expected)) or expected
        ),
    )
    output = SimpleNamespace(write=lambda data: len(data))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=output))

    synthetic.main([])

    assert publisher_calls == [
        (destination.absolute(), artifacts, publisher_calls[0][2])
    ]
    expected = record.canonical_bytes()
    assert fresh_calls == [(destination, publisher_calls[0][2], expected)]
    assert b"iou" not in expected
    assert b"f1" not in expected


def test_post_publication_validation_uses_hardened_public_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = b'{"validation_status":"PASS"}\n'
    record = SimpleNamespace(canonical_bytes=lambda: data)
    authority = cast(CandidateValidationAuthority, object())
    calls: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        synthetic,
        "validate_candidate_package_fresh_process",
        lambda path, value: calls.append((path, value)) or record,
    )

    assert synthetic._fresh_validation(tmp_path, authority, data) == data
    assert calls == [(tmp_path, authority)]
    with pytest.raises(RuntimeError, match="fresh post-publication"):
        synthetic._fresh_validation(tmp_path, authority, b"different\n")


def test_main_never_retries_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    (root / synthetic._DESTINATION).parent.mkdir(parents=True)
    calls = 0
    monkeypatch.setattr(synthetic, "_repository_root", lambda: root)
    monkeypatch.setattr(synthetic, "_git_commit", lambda _root: "e" * 40)
    monkeypatch.setattr(
        synthetic, "_benchmark_attestation", lambda: SimpleNamespace(sha256="a" * 64)
    )
    monkeypatch.setattr(
        synthetic,
        "_p53_launch",
        lambda _root: SimpleNamespace(raw_root=root / "raw"),
    )
    monkeypatch.setattr(
        synthetic,
        "run_p53_validation_subprocess",
        lambda _launch: SimpleNamespace(sha256="b" * 64),
    )
    monkeypatch.setattr(synthetic, "_commitment", lambda _root: object())
    monkeypatch.setattr(
        synthetic, "iter_validated_raw_observations", lambda *_args, **_kwargs: ()
    )
    monkeypatch.setattr(
        synthetic,
        "_build_artifacts",
        lambda **_kwargs: (
            SimpleNamespace(
                manifest=SimpleNamespace(canonical_bytes=lambda: b"manifest\n")
            ),
            object(),
        ),
    )
    monkeypatch.setattr(synthetic, "_authority", lambda **_kwargs: object())

    def fail(*_args: object) -> object:
        nonlocal calls
        calls += 1
        raise FileExistsError("destination or staging exists")

    monkeypatch.setattr(synthetic, "publish_successful_candidate_package", fail)
    with pytest.raises(FileExistsError, match="destination or staging"):
        synthetic.main([])
    assert calls == 1
