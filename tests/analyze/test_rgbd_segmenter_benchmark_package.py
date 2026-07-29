from __future__ import annotations

import io
import hashlib
import json
import os
import socket
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Mapping, cast

import numpy as np
import pytest

from prior.analyze.d2026_07_29 import rgbd_segmenter_benchmark_contract
from prior.analyze.d2026_07_29 import rgbd_segmenter_benchmark_package as package
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    BOOTSTRAP_MATRIX_SHA256,
    RAW_GPU_UUID,
    RAW_VALIDATOR_SOURCE_SHA256,
    ComponentTimings,
    ConfusionCounts,
    EndpointRow,
    MappingEntry,
    MappingKind,
    MetricEndpoint,
    ObservationFailureCode,
    ObservationMetrics,
    ObservationStatus,
    Prediction,
    ProvenanceChecks,
    SegmenterInput,
    TimingSample,
    TrustedCohort,
    ValidatedRawObservation,
    aggregate_observation_metrics,
    canonical_json_bytes,
    compute_static_coverage,
    estimate_scene_robustness,
    logical_label_sha256,
    score_observation,
    summarize_latency,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    FileRecord as RawFileRecord,
    IndexRow,
    RawFrameArrays,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    AbortedOomManifest,
    ArchivalCudaEvidence,
    CandidateMappingAuthority,
    FileRecord,
    ObservationPackageRow,
    RealSuccessfulManifest,
    SyntheticSuccessfulManifest,
    TimingPackageRow,
    accept_candidate_package,
    canonical_observations_bytes,
    canonical_timings_bytes,
    encode_prediction_npz,
    file_tree_aggregate,
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
            "cohort_jsonl_sha256": _HASH,
            "raw_index_sha256": _HASH,
            "raw_manifest_sha256": _HASH,
            "raw_payload_tree_sha256": _HASH,
            "raw_producer_git_commit": _COMMIT,
            "selection_sha256": _HASH,
        },
        "robustness": {"f1": _estimate(), "iou": _estimate()},
        "run_kind": "synthetic-contract" if synthetic else "candidate",
        "run_status": "success",
        "schema_version": 1,
        "source_hashes": {
            "adapter": {"path": "candidate/adapter.py", "sha256": _HASH},
            "constants": {"path": "prior/constants.py", "sha256": _HASH},
            "contract": {
                "path": (
                    "prior/analyze/d2026_07_29/rgbd_segmenter_benchmark_contract.py"
                ),
                "sha256": _HASH,
            },
            "mapping": {
                "path": ("prior/analyze/d2026_07_29/rgbd_segmenter_nyu40_mapping.json"),
                "sha256": _HASH,
            },
            "package": {
                "path": (
                    "prior/analyze/d2026_07_29/rgbd_segmenter_benchmark_package.py"
                ),
                "sha256": _HASH,
            },
            "projector": {
                "path": ("vlnce_baselines/models/etp_llm/llm_grid_oracle_cache.py"),
                "sha256": _HASH,
            },
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
            "permission_evidence": {
                "code": {
                    "byte_length": 1,
                    "path": "LICENSE",
                    "root_role": "candidate_repository",
                    "sha256": _HASH,
                },
                "weights": {
                    "byte_length": 1,
                    "path": "WEIGHTS_LICENSE",
                    "root_role": "candidate_repository",
                    "sha256": _HASH,
                },
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
        "permission_evidence": real["permission_evidence"],
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


def test_real_manifest_rejects_selection_decision_and_extra_source_role() -> None:
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
    sources["synthetic_adapter"] = {"path": "old.py", "sha256": _HASH}
    with pytest.raises(ValueError, match="source_hashes"):
        RealSuccessfulManifest(raw)


def test_manifest_rejects_source_role_and_raw_hash_key_swaps() -> None:
    raw = _manifest(synthetic=True)
    sources = _dict_section(raw, "source_hashes")
    sources["constants"], sources["projector"] = (
        sources["projector"],
        sources["constants"],
    )
    with pytest.raises(ValueError, match="frozen role"):
        SyntheticSuccessfulManifest(raw)

    raw = _manifest(synthetic=True)
    raw_inputs = _dict_section(raw, "raw_inputs")
    raw_inputs["raw_package_sha256"] = raw_inputs.pop("raw_payload_tree_sha256")
    with pytest.raises(ValueError, match="raw_inputs"):
        SyntheticSuccessfulManifest(raw)


def test_manifest_rejects_backslash_source_and_evidence_paths() -> None:
    raw = _manifest(synthetic=True)
    sources = _dict_section(raw, "source_hashes")
    adapter = _dict_section(sources, "adapter")
    adapter["path"] = "candidate\\adapter.py"
    with pytest.raises(ValueError, match="relative POSIX"):
        SyntheticSuccessfulManifest(raw)

    raw = _manifest(synthetic=False)
    evidence = _dict_section(raw, "permission_evidence")
    weights = _dict_section(evidence, "weights")
    weights["path"] = "weights\\model.bin"
    with pytest.raises(ValueError, match="relative POSIX"):
        RealSuccessfulManifest(raw)


def test_permission_evidence_is_real_only_and_strict() -> None:
    raw = _manifest(synthetic=True)
    raw["permission_evidence"] = _manifest(synthetic=False)["permission_evidence"]
    with pytest.raises(ValueError, match="schema"):
        SyntheticSuccessfulManifest(raw)

    raw = _manifest(synthetic=False)
    evidence = _dict_section(raw, "permission_evidence")
    code = _dict_section(evidence, "code")
    code["root_role"] = "ambient_filesystem"
    with pytest.raises(ValueError, match="root_role"):
        RealSuccessfulManifest(raw)

    raw = _manifest(synthetic=False)
    evidence = _dict_section(raw, "permission_evidence")
    weights = _dict_section(evidence, "weights")
    weights["path"] = "../LICENSE"
    with pytest.raises(ValueError, match="relative POSIX"):
        RealSuccessfulManifest(raw)

    raw = _aborted_manifest()
    del raw["permission_evidence"]
    with pytest.raises(ValueError, match="schema"):
        AbortedOomManifest(raw)


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
    manifest: package.SuccessfulManifest
    if raw["run_kind"] == "candidate":
        manifest = RealSuccessfulManifest(raw)
    else:
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


@dataclass(frozen=True)
class _BenchmarkPackageFixture:
    root: Path
    manifest_sha256: str
    cohort: TrustedCohort
    authority: CandidateMappingAuthority


def _record(data: bytes) -> FileRecord:
    return FileRecord(
        byte_length=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _package_identities() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (ordinal, f"{ordinal:020x}", f"scene{ordinal % 11}") for ordinal in range(50)
    )


def _package_authority() -> CandidateMappingAuthority:
    return CandidateMappingAuthority(
        source_vocabulary=("wall",),
        mapping=(
            MappingEntry(
                source_index=0,
                source_name="wall",
                kind=MappingKind.DIAGNOSTIC,
                canonical_index=15,
                canonical_name="structure",
            ),
        ),
        mapping_sha256=_HASH,
    )


def _science_grid() -> np.ndarray:
    grid = np.zeros((27, 50, 50), dtype=np.bool_)
    grid[1, 0, 0] = True
    return grid


def _package_rows(
    artifact: package.PredictionArtifact,
    *,
    status: ObservationStatus = ObservationStatus.PASS,
    failure_code: ObservationFailureCode | None = None,
) -> tuple[ObservationPackageRow, ...]:
    grid = _science_grid()
    return tuple(
        ObservationPackageRow(
            ordinal=ordinal,
            observation_id=observation_id,
            scene_id=scene,
            status=status,
            failure_code=failure_code,
            prediction_path=(f"predictions/{scene}/{ordinal:02d}-{observation_id}.npz"),
            prediction=artifact.file,
            prediction_members=artifact.members,
            metrics=score_observation(
                ordinal=ordinal,
                scene_id=scene,
                prediction=grid,
                target=grid,
            ),
        )
        for ordinal, observation_id, scene in _package_identities()
    )


def _package_timings(
    rows: tuple[ObservationPackageRow, ...],
    prediction: Prediction,
) -> tuple[TimingPackageRow, ...]:
    source_hash = logical_label_sha256(prediction.source_labels)
    mapped_hash = logical_label_sha256(prediction.mapped_labels)
    result: list[TimingPackageRow] = []
    for pass_index in (1, 2):
        for row in rows:
            if row.status is ObservationStatus.PASS:
                components = ComponentTimings(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            else:
                components = ComponentTimings(1.0, 1.0, None, None, None, None)
            result.append(
                TimingPackageRow(
                    TimingSample(
                        sequence_index=20 + (pass_index - 1) * 50 + row.ordinal,
                        pass_index=pass_index,
                        ordinal=row.ordinal,
                        end_to_end=6.0,
                        components=components,
                        source_labels_sha256=source_hash,
                        mapped_labels_sha256=mapped_hash,
                        status=row.status,
                        failure_code=row.failure_code,
                    )
                )
            )
    return tuple(result)


def _write_package_metadata(
    root: Path,
    rows: tuple[ObservationPackageRow, ...],
    timings: tuple[TimingPackageRow, ...],
) -> str:
    observations_data = canonical_observations_bytes(rows)
    timings_data = canonical_timings_bytes(timings)
    (root / "observations.jsonl").write_bytes(observations_data)
    (root / "timings.jsonl").write_bytes(timings_data)
    manifest = _manifest(synthetic=True)
    commitment = _dict_section(manifest, "candidate_commitment")
    commitment["source_vocabulary"] = ["wall"]
    commitment["mapping_sha256"] = _HASH
    files = _dict_section(manifest, "files")
    files["observations"] = {
        "byte_length": len(observations_data),
        "row_count": 50,
        "sha256": hashlib.sha256(observations_data).hexdigest(),
    }
    files["timings"] = {
        "byte_length": len(timings_data),
        "row_count": 100,
        "sha256": hashlib.sha256(timings_data).hexdigest(),
    }
    aggregate = file_tree_aggregate(
        (row.prediction_path, row.prediction) for row in rows
    )
    files["predictions"] = {
        "file_count": aggregate.file_count,
        "total_byte_length": aggregate.total_byte_length,
        "tree_sha256": aggregate.tree_sha256,
    }
    cohort = TrustedCohort(_package_identities())
    metric_summary = aggregate_observation_metrics(
        tuple(row.metrics for row in rows),
        expected_scenes=cohort.scenes,
    )
    manifest["metric_summary"] = {
        "all_27": asdict(metric_summary.all_27),
        "per_category": [asdict(value) for value in metric_summary.per_category],
        "primary": asdict(metric_summary.primary),
    }
    grid = _science_grid()
    manifest["coverage"] = {
        **asdict(
            compute_static_coverage(
                _package_authority().mapping,
                (grid,) * 50,
            )
        ),
        "support_ratio": 0.0,
    }
    iou_rows = tuple(
        EndpointRow(
            ordinal=row.ordinal,
            observation_id=row.observation_id,
            scene_id=row.scene_id,
            endpoint=row.metrics.primary.iou,
            status=row.status,
            failure_code=row.failure_code,
        )
        for row in rows
    )
    f1_rows = tuple(
        replace(value, endpoint=row.metrics.primary.f1)
        for value, row in zip(iou_rows, rows)
    )
    manifest["robustness"] = {
        "iou": asdict(estimate_scene_robustness(iou_rows, trusted_cohort=cohort)),
        "f1": asdict(estimate_scene_robustness(f1_rows, trusted_cohort=cohort)),
    }
    data = SyntheticSuccessfulManifest(manifest).canonical_bytes()
    (root / "manifest.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def candidate_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> _BenchmarkPackageFixture:
    root = tmp_path_factory.mktemp("candidate-package") / "package"
    root.mkdir()
    prediction = Prediction(
        source_labels=np.zeros((12, 256, 256), dtype="<i2"),
        mapped_labels=np.full((12, 256, 256), 15, dtype="<i2"),
    )
    artifact = encode_prediction_npz(prediction)
    rows = _package_rows(artifact)
    for row in rows:
        path = root / row.prediction_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact.data)
    timings = _package_timings(rows, prediction)
    return _BenchmarkPackageFixture(
        root=root,
        manifest_sha256=_write_package_metadata(root, rows, timings),
        cohort=TrustedCohort(_package_identities()),
        authority=_package_authority(),
    )


def _accept(value: _BenchmarkPackageFixture) -> package.AcceptedCandidatePackage:
    return accept_candidate_package(
        value.root,
        expected_manifest_sha256=value.manifest_sha256,
        trusted_cohort=value.cohort,
        mapping_authority=value.authority,
    )


@pytest.fixture(scope="module")
def accepted_science_package(
    candidate_package: _BenchmarkPackageFixture,
) -> package.AcceptedCandidatePackage:
    return _accept(candidate_package)


def _science_raw_observations() -> tuple[ValidatedRawObservation, ...]:
    placeholder = np.empty(0)
    segmenter_input = cast(SegmenterInput, object())
    return tuple(
        ValidatedRawObservation(
            row=IndexRow(
                artifact=f"observations/{ordinal}.npz",
                cohort_row_sha256=_HASH,
                members={},
                npz=RawFileRecord(1, _HASH),
                observation_id=observation_id,
                oracle_artifact_sha256=_HASH,
                ordinal=ordinal,
                scene_id=scene_id,
            ),
            segmenter_input=segmenter_input,
            audit_arrays=RawFrameArrays(
                np.asarray(ordinal),
                *(placeholder for _ in range(16)),
            ),
        )
        for ordinal, observation_id, scene_id in _package_identities()
    )


def _accepted_with_manifest(
    accepted: package.AcceptedCandidatePackage,
    raw: dict[str, object],
    *,
    observations: tuple[ObservationPackageRow, ...] | None = None,
) -> package.AcceptedCandidatePackage:
    rows = accepted.observations if observations is None else observations
    observations_data = canonical_observations_bytes(rows)
    files = _dict_section(raw, "files")
    files["observations"] = {
        "byte_length": len(observations_data),
        "row_count": 50,
        "sha256": hashlib.sha256(observations_data).hexdigest(),
    }
    manifest: package.SuccessfulManifest
    if raw["run_kind"] == "candidate":
        manifest = RealSuccessfulManifest(raw)
    else:
        manifest = SyntheticSuccessfulManifest(raw)
    manifest_record = _record(manifest.canonical_bytes())
    complete_records = (
        ("manifest.json", manifest_record),
        ("observations.jsonl", _record(observations_data)),
        (
            "timings.jsonl",
            _record(canonical_timings_bytes(accepted.timings)),
        ),
    ) + tuple((prediction.path, prediction.file) for prediction in accepted.predictions)
    return package.AcceptedCandidatePackage(
        manifest=manifest,
        observations=rows,
        timings=accepted.timings,
        predictions=accepted.predictions,
        validation=package._package_validation_record(  # noqa: SLF001
            complete_records,
            manifest_record,
        ),
        _acceptance_token=package._ACCEPTANCE_TOKEN,  # noqa: SLF001
    )


def _passed_provenance() -> ProvenanceChecks:
    return ProvenanceChecks(*(True for _ in range(12)))


def _real_science_accepted(
    accepted: package.AcceptedCandidatePackage,
) -> package.AcceptedCandidatePackage:
    synthetic = cast(
        Dict[str, object],
        json.loads(accepted.manifest.canonical_bytes()),
    )
    raw = _manifest(synthetic=False)
    for section in ("coverage", "files", "metric_summary", "robustness"):
        raw[section] = synthetic[section]
    candidate = _dict_section(raw, "candidate_commitment")
    candidate["source_vocabulary"] = ["wall"]
    candidate["mapping_sha256"] = _HASH
    latency = summarize_latency(tuple(row.sample for row in accepted.timings))
    raw["latency"] = {
        "p50_seconds": latency.p50_seconds,
        "p95_seconds": latency.p95_seconds,
        "sample_count": latency.sample_count,
        "total_seconds": latency.total_seconds,
        "views_per_second": latency.views_per_second,
    }
    raw["gates"] = {
        "complete_rows": "PASS",
        "coverage": "FAIL",
        "latency": "FAIL",
        "license": "PASS",
        "overall": "FAIL",
        "provenance": "PASS",
        "quality": "FAIL",
        "resource": "PASS",
    }
    raw["candidate_status"] = "FAIL"
    return _accepted_with_manifest(accepted, raw)


def _preserve_files(root: Path, paths: tuple[str, ...]) -> dict[str, bytes]:
    return {path: (root / path).read_bytes() for path in paths}


def _restore_files(root: Path, values: Mapping[str, bytes]) -> None:
    for path, data in values.items():
        target = root / path
        if target.exists() or target.is_symlink():
            target.unlink()
        target.write_bytes(data)


def test_candidate_reader_accepts_target_free_package_and_owns_arrays(
    candidate_package: _BenchmarkPackageFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_calls = 0
    parser_arrays: list[np.ndarray] = []
    original_parser = package.parse_prediction_npz_bytes

    def forbidden_target(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal target_calls
        target_calls += 1
        raise AssertionError("target projector must not be called")

    def capturing_parser(
        data: bytes,
        *,
        expected_file: FileRecord | None = None,
        expected_members: Mapping[str, package.PredictionMember] | None = None,
    ) -> package.ParsedPrediction:
        parsed = original_parser(
            data,
            expected_file=expected_file,
            expected_members=expected_members,
        )
        if not parser_arrays:
            parser_arrays.append(parsed.prediction.source_labels)
        return parsed

    monkeypatch.setattr(
        rgbd_segmenter_benchmark_contract,
        "project_oracle_frames",
        forbidden_target,
    )
    monkeypatch.setattr(package, "parse_prediction_npz_bytes", capturing_parser)
    accepted = _accept(candidate_package)

    assert target_calls == 0
    assert not hasattr(accepted, "targets")
    assert accepted.validation.file_count == 53
    assert len(accepted.predictions) == 50
    with pytest.raises(ValueError):
        accepted.predictions[0].prediction.source_labels.setflags(write=True)
    parser_arrays[0][0, 0, 0] = 9
    assert accepted.predictions[0].prediction.source_labels[0, 0, 0] == 0
    with pytest.raises(ValueError, match="acceptance factory"):
        package.AcceptedCandidatePackage(
            manifest=accepted.manifest,
            observations=accepted.observations,
            timings=accepted.timings,
            predictions=accepted.predictions,
            validation=accepted.validation,
        )
    with pytest.raises(ValueError, match="acceptance factory"):
        replace(accepted, predictions=accepted.predictions[::-1])
    with pytest.raises(ValueError, match="acceptance factory"):
        replace(
            accepted,
            validation=replace(accepted.validation, tree_sha256="2" * 64),
        )


def test_scientific_recomputation_cannot_mint_authoritative_pass(
    candidate_package: _BenchmarkPackageFixture,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted = accepted_science_package
    grid = _science_grid()
    recomputed = package._recompute_accepted_candidate_science(  # noqa: SLF001
        accepted,
        raw_observations=_science_raw_observations(),
        trusted_cohort=candidate_package.cohort,
        mapping_authority=candidate_package.authority,
        oracle_projector=lambda _arrays: grid,
        prediction_projector=lambda _labels, _arrays: grid,
    )

    assert recomputed.metrics == accepted.manifest.summary.metrics
    assert recomputed.robustness == accepted.manifest.summary.robustness
    assert not hasattr(recomputed, "record")
    assert not hasattr(package, "validate_accepted_candidate_science")
    assert not hasattr(package, "ScientificValidationRecord")
    assert not hasattr(package, "ValidatedCandidateScience")


def test_scientific_validator_binds_all_identities_before_projection(
    candidate_package: _BenchmarkPackageFixture,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted = accepted_science_package
    projector_calls = 0

    def projector(*_args: object) -> np.ndarray:
        nonlocal projector_calls
        projector_calls += 1
        return _science_grid()

    with pytest.raises(ValueError, match="raw observations"):
        package._recompute_accepted_candidate_science(  # noqa: SLF001
            accepted,
            raw_observations=_science_raw_observations()[::-1],
            trusted_cohort=candidate_package.cohort,
            mapping_authority=candidate_package.authority,
            oracle_projector=projector,
            prediction_projector=projector,
        )
    assert projector_calls == 0


def test_scientific_validator_keeps_target_empty_false_positive_eligible(
    candidate_package: _BenchmarkPackageFixture,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted = accepted_science_package
    prediction = _science_grid()
    empty = np.zeros((27, 50, 50), dtype=np.bool_)
    targets = (empty,) + (_science_grid(),) * 49
    observations = tuple(
        replace(
            row,
            metrics=score_observation(
                ordinal=row.ordinal,
                scene_id=row.scene_id,
                prediction=prediction,
                target=targets[row.ordinal],
            ),
        )
        for row in accepted.observations
    )
    summary = aggregate_observation_metrics(
        tuple(row.metrics for row in observations),
        expected_scenes=candidate_package.cohort.scenes,
    )
    assert summary.primary.eligible_observation_count == 50
    assert summary.primary.target_empty_prediction_nonempty_count == 1
    raw = cast(Dict[str, object], json.loads(accepted.manifest.canonical_bytes()))
    raw["metric_summary"] = {
        "all_27": asdict(summary.all_27),
        "per_category": [asdict(value) for value in summary.per_category],
        "primary": asdict(summary.primary),
    }
    coverage = compute_static_coverage(candidate_package.authority.mapping, targets)
    raw["coverage"] = {
        **asdict(coverage),
        "support_ratio": coverage.support_ratio,
    }
    iou_rows = tuple(
        EndpointRow(
            ordinal=row.ordinal,
            observation_id=row.observation_id,
            scene_id=row.scene_id,
            endpoint=row.metrics.primary.iou,
            status=row.status,
            failure_code=row.failure_code,
        )
        for row in observations
    )
    f1_rows = tuple(
        replace(value, endpoint=row.metrics.primary.f1)
        for value, row in zip(iou_rows, observations)
    )
    raw["robustness"] = {
        "iou": asdict(
            estimate_scene_robustness(
                iou_rows,
                trusted_cohort=candidate_package.cohort,
            )
        ),
        "f1": asdict(
            estimate_scene_robustness(
                f1_rows,
                trusted_cohort=candidate_package.cohort,
            )
        ),
    }
    eligible = _accepted_with_manifest(
        accepted,
        raw,
        observations=observations,
    )

    recomputed = package._recompute_accepted_candidate_science(  # noqa: SLF001
        eligible,
        raw_observations=_science_raw_observations(),
        trusted_cohort=candidate_package.cohort,
        mapping_authority=candidate_package.authority,
        oracle_projector=lambda arrays: (
            empty if int(arrays.schema_version) == 0 else _science_grid()
        ),
        prediction_projector=lambda _labels, _arrays: prediction,
    )
    assert recomputed.metrics.primary.eligible_observation_count == 50
    assert recomputed.metrics.primary.target_empty_prediction_nonempty_count == 1


def test_scientific_endpoint_rows_preserve_failed_outcome_for_iou_and_f1(
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    original = accepted_science_package.observations[0]
    failed = replace(
        original,
        status=ObservationStatus.FAILED,
        failure_code=ObservationFailureCode.INFERENCE_FAILURE,
    )
    for endpoint in (
        lambda value: value.primary.iou,
        lambda value: value.primary.f1,
    ):
        row = package._endpoint_rows(  # noqa: SLF001
            (failed,),
            (failed.metrics,),
            endpoint=endpoint,
        )[0]
        assert row.status is ObservationStatus.FAILED
        assert row.failure_code is ObservationFailureCode.INFERENCE_FAILURE


@pytest.mark.parametrize(
    "section",
    ["row", "aggregate", "per_category", "coverage", "robustness", "f1"],
)
def test_scientific_validator_rejects_recomputed_claim_tamper(
    candidate_package: _BenchmarkPackageFixture,
    accepted_science_package: package.AcceptedCandidatePackage,
    section: str,
) -> None:
    accepted = accepted_science_package
    raw = cast(
        Dict[str, object],
        json.loads(accepted.manifest.canonical_bytes()),
    )
    observations: tuple[ObservationPackageRow, ...] | None = None
    if section == "row":
        first = accepted.observations[0]
        wrong = score_observation(
            ordinal=first.ordinal,
            scene_id=first.scene_id,
            prediction=np.zeros((27, 50, 50), dtype=np.bool_),
            target=_science_grid(),
        )
        observations = (replace(first, metrics=wrong),) + accepted.observations[1:]
    elif section == "aggregate":
        metric_summary = _dict_section(raw, "metric_summary")
        primary = _dict_section(metric_summary, "primary")
        primary["mean_iou"] = 0.5
    elif section == "per_category":
        metric_summary = _dict_section(raw, "metric_summary")
        per_category = metric_summary["per_category"]
        assert isinstance(per_category, list)
        category = per_category[1]
        assert isinstance(category, dict)
        cast(Dict[str, object], category)["mean_iou"] = 0.5
    elif section == "coverage":
        raw["coverage"] = {
            "covered_category_count": 1,
            "covered_support_count": 1,
            "support_ratio": 0.02,
            "total_support_count": 50,
        }
    else:
        robustness = _dict_section(raw, "robustness")
        estimate = _dict_section(
            robustness,
            "f1" if section == "f1" else "iou",
        )
        for key in (
            "interval_high",
            "interval_low",
            "leave_one_scene_out_max",
            "leave_one_scene_out_min",
            "point_estimate",
        ):
            estimate[key] = 0.5
    tampered = _accepted_with_manifest(
        accepted,
        raw,
        observations=observations,
    )

    expected_error = {
        "row": "observation",
        "aggregate": "metric summary",
        "per_category": "metric summary",
        "coverage": "coverage",
        "robustness": "robustness",
        "f1": "robustness",
    }[section]
    with pytest.raises(ValueError, match=expected_error):
        package._recompute_accepted_candidate_science(  # noqa: SLF001
            tampered,
            raw_observations=_science_raw_observations(),
            trusted_cohort=candidate_package.cohort,
            mapping_authority=candidate_package.authority,
            oracle_projector=lambda _arrays: _science_grid(),
            prediction_projector=lambda _labels, _arrays: _science_grid(),
        )


def test_manifest_rejects_stale_numpy_statistics_runtime() -> None:
    raw = _manifest(synthetic=True)
    statistics = _dict_section(raw, "statistics_protocol")
    statistics["numpy_version"] = "0.0.0"
    with pytest.raises(ValueError, match="statistics protocol"):
        SyntheticSuccessfulManifest(raw)


@pytest.mark.parametrize("section", ["latency", "resource", "gates"])
def test_real_scientific_validator_recomputes_latency_and_gates(
    candidate_package: _BenchmarkPackageFixture,
    accepted_science_package: package.AcceptedCandidatePackage,
    section: str,
) -> None:
    accepted = _real_science_accepted(accepted_science_package)
    raw = cast(Dict[str, object], json.loads(accepted.manifest.canonical_bytes()))
    if section == "latency":
        latency = _dict_section(raw, "latency")
        latency["p95_seconds"] = 7.0
    elif section == "resource":
        resource = _dict_section(raw, "resource")
        resource["peak_reserved_bytes"] = 32 * 1024**3
    else:
        gates = _dict_section(raw, "gates")
        gates["latency"] = "PASS"
    tampered = _accepted_with_manifest(accepted, raw)

    with pytest.raises(ValueError, match="latency|gates"):
        package._recompute_accepted_candidate_science(  # noqa: SLF001
            tampered,
            raw_observations=_science_raw_observations(),
            trusted_cohort=candidate_package.cohort,
            mapping_authority=candidate_package.authority,
            provenance=_passed_provenance(),
            oracle_projector=lambda _arrays: _science_grid(),
            prediction_projector=lambda _labels, _arrays: _science_grid(),
        )


def test_real_scientific_validator_accepts_exact_recomputation(
    candidate_package: _BenchmarkPackageFixture,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    recomputed = package._recompute_accepted_candidate_science(  # noqa: SLF001
        _real_science_accepted(accepted_science_package),
        raw_observations=_science_raw_observations(),
        trusted_cohort=candidate_package.cohort,
        mapping_authority=candidate_package.authority,
        provenance=_passed_provenance(),
        oracle_projector=lambda _arrays: _science_grid(),
        prediction_projector=lambda _labels, _arrays: _science_grid(),
    )
    assert recomputed.latency is not None
    assert recomputed.gates is not None
    assert not hasattr(recomputed, "record")


def test_file_tree_aggregate_uses_canonical_sorted_records() -> None:
    records = (
        ("z/file", FileRecord(2, "2" * 64)),
        ("a/file", FileRecord(1, "1" * 64)),
    )
    aggregate = file_tree_aggregate(records)
    expected = canonical_json_bytes([
        {"byte_length": 1, "path": "a/file", "sha256": "1" * 64},
        {"byte_length": 2, "path": "z/file", "sha256": "2" * 64},
    ])
    assert aggregate.file_count == 2
    assert aggregate.total_byte_length == 3
    assert aggregate.tree_sha256 == hashlib.sha256(expected).hexdigest()
    assert file_tree_aggregate(reversed(records)) == aggregate


def test_file_tree_aggregate_rejects_backslash_path() -> None:
    with pytest.raises(ValueError, match="relative POSIX"):
        file_tree_aggregate((("directory\\file", FileRecord(1, "1" * 64)),))


def test_accepted_prediction_defensively_owns_immutable_labels() -> None:
    source = np.zeros((12, 256, 256), dtype="<i2")
    mapped = np.zeros((12, 256, 256), dtype="<i2")
    prediction = Prediction(source_labels=source, mapped_labels=mapped)
    artifact = encode_prediction_npz(prediction)
    source.setflags(write=False)
    mapped.setflags(write=False)

    accepted = package.AcceptedPrediction(
        path="predictions/scene/observation.npz",
        prediction=prediction,
        file=artifact.file,
        members=artifact.members,
    )

    source.setflags(write=True)
    mapped.setflags(write=True)
    source[0, 0, 0] = 1
    mapped[0, 0, 0] = 2
    assert accepted.prediction.source_labels[0, 0, 0] == 0
    assert accepted.prediction.mapped_labels[0, 0, 0] == 0
    with pytest.raises(ValueError):
        accepted.prediction.source_labels.setflags(write=True)
    with pytest.raises(ValueError):
        accepted.prediction.mapped_labels.setflags(write=True)
    with pytest.raises(ValueError, match="metadata is not canonical"):
        package.AcceptedPrediction(
            path="predictions/scene/observation.npz",
            prediction=prediction,
            file=FileRecord(artifact.file.byte_length, "2" * 64),
            members=artifact.members,
        )
    wrong_members = dict(artifact.members)
    wrong_members["source_labels"] = replace(
        wrong_members["source_labels"],
        array_sha256="2" * 64,
    )
    with pytest.raises(ValueError, match="metadata is not canonical"):
        package.AcceptedPrediction(
            path="predictions/scene/observation.npz",
            prediction=prediction,
            file=artifact.file,
            members=wrong_members,
        )


def test_candidate_reader_requires_external_hash_cohort_and_mapping_authority(
    candidate_package: _BenchmarkPackageFixture,
) -> None:
    with pytest.raises(ValueError, match="record mismatch"):
        accept_candidate_package(
            candidate_package.root,
            expected_manifest_sha256="2" * 64,
            trusted_cohort=candidate_package.cohort,
            mapping_authority=candidate_package.authority,
        )
    wrong_vocabulary = CandidateMappingAuthority(
        source_vocabulary=("ceiling",),
        mapping=(
            MappingEntry(
                0,
                "ceiling",
                MappingKind.DIAGNOSTIC,
                17,
                "free-space",
            ),
        ),
        mapping_sha256=_HASH,
    )
    with pytest.raises(ValueError, match="authority mismatch"):
        accept_candidate_package(
            candidate_package.root,
            expected_manifest_sha256=candidate_package.manifest_sha256,
            trusted_cohort=candidate_package.cohort,
            mapping_authority=wrong_vocabulary,
        )
    identities = list(candidate_package.cohort.identities)
    identities[0], identities[1] = identities[1], identities[0]
    identities = [
        (ordinal, observation_id, scene)
        for ordinal, (_, observation_id, scene) in enumerate(identities)
    ]
    wrong_cohort = TrustedCohort(tuple(identities))
    with pytest.raises(ValueError, match="trusted cohort|exact tree"):
        accept_candidate_package(
            candidate_package.root,
            expected_manifest_sha256=candidate_package.manifest_sha256,
            trusted_cohort=wrong_cohort,
            mapping_authority=candidate_package.authority,
        )


def test_candidate_reader_rejects_tree_entry_types_and_sizes(
    candidate_package: _BenchmarkPackageFixture,
) -> None:
    root = candidate_package.root
    extra = root / "extra"
    extra.write_text("x")
    try:
        with pytest.raises(ValueError, match="exact tree"):
            _accept(candidate_package)
    finally:
        extra.unlink()

    target = (
        root / _package_rows(encode_prediction_npz(_prediction()))[0].prediction_path
    )
    original = target.read_bytes()
    target.unlink()
    target.symlink_to("missing")
    try:
        with pytest.raises(ValueError, match="entry type|exact tree"):
            _accept(candidate_package)
    finally:
        target.unlink()
        target.write_bytes(original)

    second = root / (f"predictions/scene1/01-{1:020x}.npz")
    target.unlink()
    os.link(second, target)
    try:
        with pytest.raises(ValueError, match="hard-linked"):
            _accept(candidate_package)
    finally:
        target.unlink()
        target.write_bytes(original)

    target.unlink()
    os.mkfifo(target)
    try:
        with pytest.raises(ValueError, match="entry type"):
            _accept(candidate_package)
    finally:
        target.unlink()
        target.write_bytes(original)

    unix_socket = socket.socket(socket.AF_UNIX)
    socket_path = root / "socket"
    unix_socket.bind(str(socket_path))
    try:
        with pytest.raises(ValueError, match="entry type"):
            _accept(candidate_package)
    finally:
        unix_socket.close()
        socket_path.unlink()

    target.unlink()
    with target.open("wb") as stream:
        stream.truncate(4 * 1024 * 1024 + 1)
    try:
        with pytest.raises(ValueError, match="metadata"):
            _accept(candidate_package)
    finally:
        target.unlink()
        target.write_bytes(original)


def test_candidate_reader_rejects_empty_deep_large_and_relative_trees(
    candidate_package: _BenchmarkPackageFixture,
) -> None:
    root = candidate_package.root
    empty = root / "empty"
    empty.mkdir()
    try:
        with pytest.raises(ValueError, match="empty directory|exact tree"):
            _accept(candidate_package)
    finally:
        empty.rmdir()

    deep = root / "too" / "deep" / "forbidden"
    deep.mkdir(parents=True)
    (deep / "entry").write_text("x")
    try:
        with pytest.raises(ValueError, match="depth|too many|exact tree"):
            _accept(candidate_package)
    finally:
        (deep / "entry").unlink()
        deep.rmdir()
        deep.parent.rmdir()
        deep.parent.parent.rmdir()

    extras = tuple(root / f"extra-{index}" for index in range(20))
    for path in extras:
        path.write_text("x")
    try:
        with pytest.raises(ValueError, match="too many"):
            _accept(candidate_package)
    finally:
        for path in extras:
            path.unlink()

    with pytest.raises(ValueError, match="absolute"):
        accept_candidate_package(
            Path("relative/package"),
            expected_manifest_sha256=candidate_package.manifest_sha256,
            trusted_cohort=candidate_package.cohort,
            mapping_authority=candidate_package.authority,
        )


@pytest.mark.parametrize(
    ("source_value", "mapped_value", "status", "failure_code", "message"),
    [
        (1, 15, ObservationStatus.PASS, None, "out of range"),
        (0, 17, ObservationStatus.PASS, None, "authoritative mapping"),
        (
            0,
            15,
            ObservationStatus.FAILED,
            ObservationFailureCode.INFERENCE_FAILURE,
            "all -1",
        ),
    ],
)
def test_candidate_reader_rejects_prediction_semantic_drift(
    candidate_package: _BenchmarkPackageFixture,
    source_value: int,
    mapped_value: int,
    status: ObservationStatus,
    failure_code: ObservationFailureCode | None,
    message: str,
) -> None:
    root = candidate_package.root
    rows = parse_observations_bytes((root / "observations.jsonl").read_bytes())
    first_path = rows[0].prediction_path
    preserved = _preserve_files(
        root,
        ("manifest.json", "observations.jsonl", "timings.jsonl", first_path),
    )
    prediction = Prediction(
        np.full((12, 256, 256), source_value, dtype="<i2"),
        np.full((12, 256, 256), mapped_value, dtype="<i2"),
    )
    artifact = encode_prediction_npz(prediction)
    changed_rows = (
        replace(
            rows[0],
            status=status,
            failure_code=failure_code,
            prediction=artifact.file,
            prediction_members=artifact.members,
        ),
    ) + rows[1:]
    timings = _package_timings(changed_rows, prediction)
    (root / first_path).write_bytes(artifact.data)
    expected_manifest = _write_package_metadata(root, changed_rows, timings)
    try:
        with pytest.raises(ValueError, match=message):
            accept_candidate_package(
                root,
                expected_manifest_sha256=expected_manifest,
                trusted_cohort=candidate_package.cohort,
                mapping_authority=candidate_package.authority,
            )
    finally:
        _restore_files(root, preserved)


def test_candidate_reader_rejects_timing_and_manifest_aggregate_mismatch(
    candidate_package: _BenchmarkPackageFixture,
) -> None:
    root = candidate_package.root
    rows = parse_observations_bytes((root / "observations.jsonl").read_bytes())
    timings = parse_timings_bytes((root / "timings.jsonl").read_bytes())
    preserved = _preserve_files(root, ("manifest.json", "timings.jsonl"))
    changed = (
        TimingPackageRow(replace(timings[0].sample, source_labels_sha256="2" * 64)),
    ) + timings[1:]
    expected_manifest = _write_package_metadata(root, rows, changed)
    try:
        with pytest.raises(ValueError, match="timing row"):
            accept_candidate_package(
                root,
                expected_manifest_sha256=expected_manifest,
                trusted_cohort=candidate_package.cohort,
                mapping_authority=candidate_package.authority,
            )
    finally:
        _restore_files(root, preserved)

    manifest = _manifest(synthetic=True)
    commitment = _dict_section(manifest, "candidate_commitment")
    commitment["source_vocabulary"] = ["wall"]
    commitment["mapping_sha256"] = _HASH
    files = _dict_section(manifest, "files")
    observations_data = (root / "observations.jsonl").read_bytes()
    timings_data = (root / "timings.jsonl").read_bytes()
    files["observations"] = {
        "byte_length": len(observations_data),
        "row_count": 50,
        "sha256": hashlib.sha256(observations_data).hexdigest(),
    }
    files["timings"] = {
        "byte_length": len(timings_data),
        "row_count": 100,
        "sha256": hashlib.sha256(timings_data).hexdigest(),
    }
    files["predictions"] = {
        "file_count": 50,
        "total_byte_length": sum(row.prediction.byte_length for row in rows),
        "tree_sha256": "2" * 64,
    }
    bad_manifest = SyntheticSuccessfulManifest(manifest).canonical_bytes()
    original_manifest = (root / "manifest.json").read_bytes()
    (root / "manifest.json").write_bytes(bad_manifest)
    try:
        with pytest.raises(ValueError, match="tree aggregate"):
            accept_candidate_package(
                root,
                expected_manifest_sha256=hashlib.sha256(bad_manifest).hexdigest(),
                trusted_cohort=candidate_package.cohort,
                mapping_authority=candidate_package.authority,
            )
    finally:
        (root / "manifest.json").write_bytes(original_manifest)


def test_candidate_reader_rejects_intermediate_directory_swap(
    candidate_package: _BenchmarkPackageFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    original_open = package.os.open
    scene_open_count = 0

    def swapping_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal scene_open_count
        if path == "scene0":
            scene_open_count += 1
            if scene_open_count == 2:
                return original_open(attacker, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(package.os, "open", swapping_open)
    with pytest.raises(ValueError, match="directory changed|unsafe"):
        _accept(candidate_package)


def test_candidate_reader_rejects_root_binding_and_final_content_races(
    candidate_package: _BenchmarkPackageFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = candidate_package.root
    moved = root.with_name(f"{root.name}-moved")
    original_read = package._read_file_at
    swapped = False

    def swapping_root(
        root_descriptor: int,
        path: str,
        *,
        expected_fingerprint: package._Fingerprint,
        expected_directories: Mapping[str, package._Fingerprint],
        expected_record: FileRecord | None = None,
    ) -> tuple[bytes, FileRecord]:
        nonlocal swapped
        result = original_read(
            root_descriptor,
            path,
            expected_fingerprint=expected_fingerprint,
            expected_directories=expected_directories,
            expected_record=expected_record,
        )
        if not swapped:
            root.rename(moved)
            root.mkdir()
            swapped = True
        return result

    monkeypatch.setattr(package, "_read_file_at", swapping_root)
    try:
        with pytest.raises(ValueError, match="tree changed|ancestor binding"):
            _accept(candidate_package)
    finally:
        root.rmdir()
        moved.rename(root)

    monkeypatch.setattr(package, "_read_file_at", original_read)
    manifest_reads = 0
    manifest_path = root / "manifest.json"
    original_manifest = manifest_path.read_bytes()

    def mutating_final_read(
        root_descriptor: int,
        path: str,
        *,
        expected_fingerprint: package._Fingerprint,
        expected_directories: Mapping[str, package._Fingerprint],
        expected_record: FileRecord | None = None,
    ) -> tuple[bytes, FileRecord]:
        nonlocal manifest_reads
        result = original_read(
            root_descriptor,
            path,
            expected_fingerprint=expected_fingerprint,
            expected_directories=expected_directories,
            expected_record=expected_record,
        )
        if path == "manifest.json":
            manifest_reads += 1
            if manifest_reads == 2:
                manifest_path.write_bytes(original_manifest + b" ")
        return result

    monkeypatch.setattr(package, "_read_file_at", mutating_final_read)
    try:
        with pytest.raises(ValueError, match="final recapture"):
            _accept(candidate_package)
    finally:
        manifest_path.write_bytes(original_manifest)
