from __future__ import annotations

import io
import json
import hashlib
import zipfile
from dataclasses import replace
from typing import Callable, Dict, Mapping, cast

import numpy as np
import pytest

from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    BOOTSTRAP_MATRIX_SHA256,
    RAW_GPU_UUID,
    RAW_VALIDATOR_SOURCE_SHA256,
    ComponentTimings,
    ConfusionCounts,
    MetricEndpoint,
    ObservationMetrics,
    ObservationStatus,
    Prediction,
    TimingSample,
    canonical_json_bytes,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    AbortedOomManifest,
    ArchivalCudaEvidence,
    FileRecord,
    ObservationPackageRow,
    RealSuccessfulManifest,
    SyntheticSuccessfulManifest,
    TimingPackageRow,
    canonical_observations_bytes,
    canonical_timings_bytes,
    encode_prediction_npz,
    parse_observations_bytes,
    parse_aborted_oom_manifest_bytes,
    parse_prediction_npz_bytes,
    parse_successful_manifest_bytes,
    parse_timings_bytes,
)

_HASH = "1" * 64
_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _prediction(value: int = 0) -> Prediction:
    return Prediction(
        source_labels=np.full((12, 256, 256), value, dtype="<i2"),
        mapped_labels=np.full((12, 256, 256), value, dtype="<i2"),
    )


def _empty_endpoint() -> MetricEndpoint:
    return MetricEndpoint.from_counts(ConfusionCounts(tp=0, fp=0, fn=0))


def _observation_rows() -> tuple[ObservationPackageRow, ...]:
    artifact = encode_prediction_npz(_prediction())
    endpoint = _empty_endpoint()
    return tuple(
        ObservationPackageRow(
            ordinal=ordinal,
            observation_id=f"{ordinal:020x}",
            scene_id=f"scene{ordinal // 5}",
            status=ObservationStatus.PASS,
            failure_code=None,
            prediction_path=(
                f"predictions/scene{ordinal // 5}/{ordinal:02d}-{ordinal:020x}.npz"
            ),
            prediction=artifact.file,
            prediction_members=artifact.members,
            metrics=ObservationMetrics(
                ordinal=ordinal,
                scene_id=f"scene{ordinal // 5}",
                primary=endpoint,
                all_27=endpoint,
                per_category=(endpoint,) * 27,
            ),
        )
        for ordinal in range(50)
    )


def _timing_rows() -> tuple[TimingPackageRow, ...]:
    components = ComponentTimings(
        preprocess=1.0,
        h2d=1.0,
        inference=1.0,
        device_postprocess=1.0,
        d2h=1.0,
        projection=1.0,
    )
    return tuple(
        TimingPackageRow(
            TimingSample(
                sequence_index=20 + (pass_index - 1) * 50 + ordinal,
                pass_index=pass_index,
                ordinal=ordinal,
                end_to_end=6.0,
                components=components,
                source_labels_sha256=_HASH,
                mapped_labels_sha256=_HASH,
                status=ObservationStatus.PASS,
                failure_code=None,
            )
        )
        for pass_index in (1, 2)
        for ordinal in range(50)
    )


def _candidate_commitment(*, synthetic: bool) -> dict[str, object]:
    return {
        "batch_views": 12,
        "candidate_id": "synthetic" if synthetic else "real",
        "checkpoint_id": "NOT_APPLICABLE" if synthetic else "checkpoint",
        "checkpoint_sha256": "NOT_APPLICABLE" if synthetic else _HASH,
        "checkpoint_url": (
            "NOT_APPLICABLE" if synthetic else "https://example.invalid/weights"
        ),
        "code_license_status": "NOT_APPLICABLE" if synthetic else "PASS",
        "depth_coordinate_semantics": "nearest-exact",
        "depth_interpolation": "nearest-exact",
        "depth_padding_value": "0",
        "depth_units": "float32-metres[0,10]",
        "environment_lock_path": "environment.lock",
        "environment_lock_sha256": _HASH,
        "invalid_depth_policy": "zero",
        "mapping_sha256": _HASH,
        "normalization": "none",
        "precision_mode": "float32",
        "repository_url": (
            "NOT_APPLICABLE" if synthetic else "https://example.invalid/repo"
        ),
        "revision": "NOT_APPLICABLE" if synthetic else _COMMIT,
        "rgb_coordinate_semantics": "align_corners=False",
        "rgb_interpolation": "bilinear",
        "rgb_padding_value": "0",
        "rgb_units": "uint8[0,255]",
        "source_dataset": "NOT_APPLICABLE" if synthetic else "NYU Depth v2",
        "source_vocabulary": [f"class-{index}" for index in range(40)],
        "synthetic": synthetic,
        "weight_license_status": "NOT_APPLICABLE" if synthetic else "PASS",
    }


def _embedded(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
        "value": value,
    }


def _dict_section(value: dict[str, object], name: str) -> dict[str, object]:
    section = value[name]
    assert isinstance(section, dict)
    return cast(Dict[str, object], section)


def _attestations(*, synthetic: bool) -> dict[str, object]:
    p53 = {
        "environment_sha256": _HASH,
        "gpu_device_id": 0,
        "gpu_uuid": RAW_GPU_UUID,
        "producer_git_commit": _COMMIT,
        "python_executable": "/usr/bin/python",
        "raw_index_sha256": _HASH,
        "raw_manifest_sha256": _HASH,
        "raw_root": "/sealed/raw",
        "schema_version": 1,
        "status": "PASS",
        "validator_source_sha256": RAW_VALIDATOR_SOURCE_SHA256,
    }
    benchmark = {
        "environment_sha256": _HASH,
        "gpu_name": "NOT_APPLICABLE" if synthetic else "NVIDIA GeForce RTX 3090",
        "gpu_uuid": "NOT_APPLICABLE" if synthetic else RAW_GPU_UUID,
        "python_executable": "/usr/bin/python",
        "timing_comparable": not synthetic,
        "visible_device_count": 0 if synthetic else 1,
    }
    return {"benchmark": _embedded(benchmark), "p53_validator": _embedded(p53)}


def _aggregate() -> dict[str, object]:
    return {
        "eligible_observation_count": 0,
        "empty_both_count": 50,
        "mean_f1": None,
        "mean_iou": None,
        "observation_count": 50,
        "pooled": {
            "counts": {"fn": 0, "fp": 0, "tp": 0},
            "eligible": False,
            "f1": None,
            "iou": None,
            "precision": None,
            "recall": None,
        },
        "scene_macro_f1": None,
        "scene_macro_iou": None,
        "target_empty_prediction_nonempty_count": 0,
        "target_nonempty_prediction_empty_count": 0,
    }


def _estimate() -> dict[str, object]:
    return {
        "interval_high": 0.0,
        "interval_low": 0.0,
        "leave_one_scene_out_max": 0.0,
        "leave_one_scene_out_min": 0.0,
        "point_estimate": 0.0,
        "replicate_count": 10_000,
    }


def _cuda_evidence() -> dict[str, object]:
    snapshot = {
        "clock_policy": "default",
        "compute_pids": [1234],
        "driver_version": "test",
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "gpu_uuid": RAW_GPU_UUID,
        "persistence_mode": "Disabled",
        "power_limit_watts": 350.0,
        "temperature_celsius": 40.0,
    }
    return {
        "after": dict(snapshot),
        "before": dict(snapshot),
        "historical_pid": 1234,
        "pytorch_cuda_alloc_conf": None,
    }


def _manifest(*, synthetic: bool) -> dict[str, object]:
    candidate_id = "synthetic" if synthetic else "real"
    result: dict[str, object] = {
        "attestations": _attestations(synthetic=synthetic),
        "candidate_commitment": _candidate_commitment(synthetic=synthetic),
        "candidate_id": candidate_id,
        "command": ["python", "-m", "benchmark"],
        "coverage": {
            "covered_category_count": 17,
            "covered_support_count": 5447,
            "support_ratio": 5447 / 6120,
            "total_support_count": 6120,
        },
        "files": {
            "observations": {"byte_length": 1, "row_count": 50, "sha256": _HASH},
            "predictions": {
                "file_count": 50,
                "total_byte_length": 1,
                "tree_sha256": _HASH,
            },
            "timings": {"byte_length": 1, "row_count": 100, "sha256": _HASH},
        },
        "metric_summary": {
            "all_27": _aggregate(),
            "per_category": [_aggregate() for _ in range(27)],
            "primary": _aggregate(),
        },
        "producer_commit_prefix": _COMMIT[:12],
        "producer_git_commit": _COMMIT,
        "raw_inputs": {
            "cohort_sha256": _HASH,
            "index_sha256": _HASH,
            "manifest_sha256": _HASH,
            "package_sha256": _HASH,
            "producer_git_commit": _COMMIT,
            "selection_sha256": _HASH,
        },
        "robustness": {"f1": _estimate(), "iou": _estimate()},
        "run_kind": "synthetic-contract" if synthetic else "candidate",
        "run_status": "success",
        "schema_version": 1,
        "source_hashes": {
            "constants": _HASH,
            "contract": _HASH,
            "mapping": _HASH,
            "package": _HASH,
            "projector": _HASH,
        },
        "statistics_protocol": {
            "bootstrap_matrix_sha256": BOOTSTRAP_MATRIX_SHA256,
            "numpy_version": np.__version__,
            "quantile_rule": "linear-h=(n-1)*p",
            "replicate_count": 10_000,
            "scene_count": 11,
            "seed": 20260728,
        },
        "timing_protocol": {
            "backend": "deterministic-fake" if synthetic else "official-cuda",
            "comparable": not synthetic,
            "measured_pass_count": 2,
            "measured_sample_count": 100,
            "unit": "synthetic-tick" if synthetic else "seconds",
            "warmup_count": 20,
        },
    }
    if synthetic:
        cast_sources = _dict_section(result, "source_hashes")
        cast_sources["synthetic_adapter"] = _HASH
        result["decision"] = "NOT_APPLICABLE"
        result["latency_claim"] = False
    else:
        result.update({
            "candidate_status": "PASS",
            "cold_start": {
                "checkpoint_byte_read_seconds": 1.0,
                "checkpoint_load_seconds": 2.0,
                "model_construction_seconds": 3.0,
            },
            "cuda_evidence": _cuda_evidence(),
            "gates": {
                "complete_rows": "PASS",
                "coverage": "PASS",
                "latency": "PASS",
                "license": "PASS",
                "overall": "PASS",
                "provenance": "PASS",
                "quality": "PASS",
                "resource": "PASS",
            },
            "latency": {
                "p50_seconds": 0.2,
                "p95_seconds": 0.3,
                "sample_count": 100,
                "total_seconds": 20.0,
                "views_per_second": 60.0,
            },
            "resource": {
                "baseline_allocated_bytes": 1,
                "baseline_reserved_bytes": 2,
                "peak_allocated_bytes": 3,
                "peak_reserved_bytes": 4,
            },
        })
    return result


def _aborted_manifest() -> dict[str, object]:
    real = _manifest(synthetic=False)
    return {
        "attestations": real["attestations"],
        "candidate_commitment": real["candidate_commitment"],
        "candidate_id": "real",
        "candidate_status": "FAIL_RESOURCE",
        "command": real["command"],
        "cuda_evidence": real["cuda_evidence"],
        "failure_code": "CUDA_OUT_OF_MEMORY",
        "failure_stage": "inference",
        "producer_commit_prefix": _COMMIT[:12],
        "producer_git_commit": _COMMIT,
        "raw_inputs": real["raw_inputs"],
        "resource_gate": "FAIL",
        "resource_snapshot": {
            "baseline_allocated_bytes": 1,
            "baseline_reserved_bytes": 2,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        },
        "run_kind": "candidate",
        "run_status": "aborted",
        "schema_version": 1,
        "source_hashes": real["source_hashes"],
    }


def _rewrite_archive(
    data: bytes,
    *,
    compression: int = zipfile.ZIP_STORED,
    reverse: bool = False,
    duplicate: bool = False,
    comment: bytes = b"",
) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        rows = [(info.filename, source.read(info)) for info in source.infolist()]
    if reverse:
        rows.reverse()
    if duplicate:
        rows.append(rows[0])
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as target:
        target.comment = comment
        for name, payload in rows:
            target.writestr(name, payload)
    return output.getvalue()


def _canonical_zip_with_member(data: bytes, name: str, payload: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        rows = [(info.filename, source.read(info)) for info in source.infolist()]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as target:
        for member_name, original in rows:
            info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            target.writestr(info, payload if member_name == name else original)
    return output.getvalue()


def _npy(value: np.ndarray, *, allow_pickle: bool = False) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, value, version=(1, 0), allow_pickle=allow_pickle)
    return output.getvalue()


def test_prediction_npz_is_deterministic_stored_and_round_trips() -> None:
    first = encode_prediction_npz(_prediction(3))
    second = encode_prediction_npz(_prediction(3))

    assert first == second
    assert tuple(first.members) == ("mapped_labels", "source_labels")
    with zipfile.ZipFile(io.BytesIO(first.data)) as archive:
        assert [info.filename for info in archive.infolist()] == [
            "mapped_labels.npy",
            "source_labels.npy",
        ]
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
        )
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()
        )

    parsed = parse_prediction_npz_bytes(
        first.data,
        expected_file=first.file,
        expected_members=first.members,
    )
    np.testing.assert_array_equal(parsed.prediction.source_labels, 3)
    np.testing.assert_array_equal(parsed.prediction.mapped_labels, 3)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: _rewrite_archive(data, compression=zipfile.ZIP_DEFLATED),
        lambda data: _rewrite_archive(data, reverse=True),
        lambda data: _rewrite_archive(data, duplicate=True),
        lambda data: _rewrite_archive(data, comment=b"forbidden"),
        lambda data: data + b"trailing",
    ],
)
def test_prediction_npz_rejects_noncanonical_zip(
    mutation: Callable[[bytes], bytes],
) -> None:
    data = encode_prediction_npz(_prediction()).data
    with pytest.raises(ValueError, match="prediction|ZIP|NPZ"):
        parse_prediction_npz_bytes(mutation(data))


def test_prediction_npz_rejects_wrong_expected_hashes() -> None:
    artifact = encode_prediction_npz(_prediction())
    with pytest.raises(ValueError, match="record mismatch"):
        parse_prediction_npz_bytes(
            artifact.data,
            expected_file=FileRecord(artifact.file.byte_length, "2" * 64),
        )
    changed_members = dict(artifact.members)
    changed_members["source_labels"] = replace(
        changed_members["source_labels"], array_sha256="2" * 64
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        parse_prediction_npz_bytes(
            artifact.data,
            expected_members=changed_members,
        )


@pytest.mark.parametrize(
    "payload",
    [
        _npy(np.zeros((12, 256, 255), dtype="<i2")),
        _npy(np.zeros((12, 256, 256), dtype="<i4")),
        _npy(np.array([object()], dtype=object), allow_pickle=True),
    ],
)
def test_prediction_npz_rejects_wrong_shape_dtype_and_pickle(payload: bytes) -> None:
    artifact = encode_prediction_npz(_prediction())
    mutated = _canonical_zip_with_member(artifact.data, "source_labels.npy", payload)
    with pytest.raises(ValueError, match="source_labels|NPY|prediction"):
        parse_prediction_npz_bytes(mutated)


def test_observation_jsonl_is_canonical_and_round_trips() -> None:
    rows = _observation_rows()
    data = canonical_observations_bytes(rows)
    assert parse_observations_bytes(data) == rows
    assert len(data.splitlines()) == 50


def test_observation_jsonl_rejects_extra_duplicate_noncanonical_and_reorder() -> None:
    data = canonical_observations_bytes(_observation_rows())
    first, *remaining = data.splitlines(keepends=True)
    extra = json.loads(first)
    extra["unexpected"] = True
    mutations = (
        canonical_json_bytes(extra) + b"".join(remaining),
        first.replace(b'{"failure_code":', b'{"status":"PASS","failure_code":', 1),
        first.replace(b'"failure_code":null', b'"failure_code":NaN', 1),
        first.replace(b'"failure_code":null', b'"failure_code":null ', 1),
        b"".join((remaining[0], first, *remaining[1:])),
        b"\xff" + data[1:],
    )
    for mutation in mutations:
        with pytest.raises(ValueError):
            parse_observations_bytes(mutation)


def test_timing_jsonl_is_canonical_and_round_trips() -> None:
    rows = _timing_rows()
    data = canonical_timings_bytes(rows)
    assert parse_timings_bytes(data) == rows
    assert len(data.splitlines()) == 100


def test_timing_jsonl_rejects_missing_extra_and_wrong_order() -> None:
    data = canonical_timings_bytes(_timing_rows())
    lines = data.splitlines(keepends=True)
    extra = json.loads(lines[0])
    extra["latency_claim"] = True
    for mutation in (
        b"".join(lines[:-1]),
        canonical_json_bytes(extra) + b"".join(lines[1:]),
        b"".join((lines[1], lines[0], *lines[2:])),
    ):
        with pytest.raises(ValueError):
            parse_timings_bytes(mutation)


@pytest.mark.parametrize("synthetic", [False, True])
def test_successful_manifests_have_separate_canonical_schemas(synthetic: bool) -> None:
    raw = _manifest(synthetic=synthetic)
    manifest = (
        SyntheticSuccessfulManifest(raw) if synthetic else RealSuccessfulManifest(raw)
    )
    parsed = parse_successful_manifest_bytes(manifest.canonical_bytes())
    assert parsed == manifest


def test_synthetic_manifest_rejects_real_claims_and_wrong_decision() -> None:
    raw = _manifest(synthetic=True)
    raw["latency"] = {
        "p50_seconds": 0.0,
        "p95_seconds": 0.0,
        "sample_count": 100,
        "total_seconds": 0.0,
        "views_per_second": 0.0,
    }
    with pytest.raises(ValueError, match="schema"):
        SyntheticSuccessfulManifest(raw)

    raw = _manifest(synthetic=True)
    raw["decision"] = "GO"
    with pytest.raises(ValueError, match="non-claim"):
        SyntheticSuccessfulManifest(raw)

    raw = _manifest(synthetic=True)
    del raw["latency_claim"]
    with pytest.raises(ValueError, match="schema"):
        SyntheticSuccessfulManifest(raw)

    raw = _manifest(synthetic=True)
    raw["latency_claim"] = True
    with pytest.raises(ValueError, match="non-claim"):
        SyntheticSuccessfulManifest(raw)


def test_real_manifest_rejects_selection_decision_and_synthetic_adapter() -> None:
    raw = _manifest(synthetic=False)
    raw["decision"] = "NOT_APPLICABLE"
    with pytest.raises(ValueError, match="schema"):
        RealSuccessfulManifest(raw)

    raw = _manifest(synthetic=False)
    raw["latency_claim"] = False
    with pytest.raises(ValueError, match="schema"):
        RealSuccessfulManifest(raw)

    raw = _manifest(synthetic=False)
    sources = _dict_section(raw, "source_hashes")
    sources["synthetic_adapter"] = _HASH
    with pytest.raises(ValueError, match="source_hashes"):
        RealSuccessfulManifest(raw)


def test_manifest_rejects_wrong_prefix_extra_fields_and_nonfinite() -> None:
    raw = _manifest(synthetic=True)
    raw["schema_version"] = True
    with pytest.raises(ValueError, match="literals"):
        SyntheticSuccessfulManifest(raw)

    raw = _manifest(synthetic=True)
    raw["producer_commit_prefix"] = _COMMIT[:13]
    with pytest.raises(ValueError, match="exactly the first 12"):
        SyntheticSuccessfulManifest(raw)

    raw = _manifest(synthetic=True)
    coverage = _dict_section(raw, "coverage")
    coverage["unexpected"] = 1
    with pytest.raises(ValueError, match="coverage"):
        SyntheticSuccessfulManifest(raw)

    raw = _manifest(synthetic=True)
    coverage = _dict_section(raw, "coverage")
    coverage["support_ratio"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        SyntheticSuccessfulManifest(raw)

    raw = _manifest(synthetic=True)
    statistics = _dict_section(raw, "statistics_protocol")
    statistics["bootstrap_matrix_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="frozen contract"):
        SyntheticSuccessfulManifest(raw)


def test_manifest_strict_reader_rejects_duplicate_invalid_utf8_and_whitespace() -> None:
    data = SyntheticSuccessfulManifest(_manifest(synthetic=True)).canonical_bytes()
    mutations = (
        data.replace(
            b'{"attestations":', b'{"run_kind":"candidate","attestations":', 1
        ),
        b"\xff" + data[1:],
        data[:-1] + b" \n",
    )
    for mutation in mutations:
        with pytest.raises(ValueError):
            parse_successful_manifest_bytes(mutation)


def test_manifest_and_row_own_immutable_copies() -> None:
    raw = _manifest(synthetic=True)
    manifest = SyntheticSuccessfulManifest(raw)
    coverage = _dict_section(raw, "coverage")
    coverage["covered_category_count"] = 0
    source_hashes = _dict_section(raw, "source_hashes")
    source_hashes["contract"] = "2" * 64
    assert parse_successful_manifest_bytes(manifest.canonical_bytes()) == manifest
    frozen_coverage = manifest.fields["coverage"]
    assert isinstance(frozen_coverage, Mapping)
    assert cast(Mapping[str, object], frozen_coverage)["covered_category_count"] == 17
    assert not hasattr(manifest.fields, "__setitem__")

    artifact = encode_prediction_npz(_prediction())
    mutable_members = dict(artifact.members)
    row = replace(_observation_rows()[0], prediction_members=mutable_members)
    mutable_members["source_labels"] = replace(
        mutable_members["source_labels"], array_sha256="2" * 64
    )
    assert row.prediction_members["source_labels"].array_sha256 != "2" * 64


def test_archival_cuda_evidence_uses_historical_not_current_pid() -> None:
    evidence = ArchivalCudaEvidence.from_json(_cuda_evidence())
    assert evidence.historical_pid == 1234
    assert evidence.to_json() == _cuda_evidence()

    drifted = _cuda_evidence()
    after = _dict_section(drifted, "after")
    after["driver_version"] = "different"
    with pytest.raises(ValueError, match="drifted"):
        ArchivalCudaEvidence.from_json(drifted)


def test_aborted_oom_manifest_is_distinct_canonical_manifest_only_schema() -> None:
    manifest = AbortedOomManifest(_aborted_manifest())
    assert parse_aborted_oom_manifest_bytes(manifest.canonical_bytes()) == manifest

    raw = _aborted_manifest()
    raw["decision"] = "NOT_APPLICABLE"
    with pytest.raises(ValueError, match="schema"):
        AbortedOomManifest(raw)

    raw = _aborted_manifest()
    raw["latency_claim"] = False
    with pytest.raises(ValueError, match="schema"):
        AbortedOomManifest(raw)

    raw = _aborted_manifest()
    raw["failure_stage"] = "retry-smaller-batch"
    with pytest.raises(ValueError, match="stage"):
        AbortedOomManifest(raw)

    for peak_name, baseline_name in (
        ("peak_allocated_bytes", "baseline_allocated_bytes"),
        ("peak_reserved_bytes", "baseline_reserved_bytes"),
    ):
        raw = _aborted_manifest()
        snapshot = _dict_section(raw, "resource_snapshot")
        snapshot[baseline_name] = 1
        snapshot[peak_name] = 0
        with pytest.raises(ValueError, match="below its baseline"):
            AbortedOomManifest(raw)
