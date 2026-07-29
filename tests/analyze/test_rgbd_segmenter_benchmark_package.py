from __future__ import annotations

import ctypes
import errno
import io
import hashlib
import inspect
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple, cast

import numpy as np
import pytest
import torch

from prior.analyze.d2026_07_29 import rgbd_segmenter_benchmark_contract
from prior.analyze.d2026_07_29 import rgbd_segmenter_benchmark_package as package
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    BOOTSTRAP_MATRIX_SHA256,
    NYU40_MAPPING_SHA256,
    RAW_GPU_UUID,
    RAW_INDEX_SHA256,
    RAW_MANIFEST_SHA256,
    RAW_PRODUCER_COMMIT,
    RAW_VALIDATOR_SOURCE_SHA256,
    BenchmarkEnvironmentAttestation,
    ComponentTimings,
    ConfusionCounts,
    EndpointRow,
    MappingEntry,
    MappingKind,
    MetricEndpoint,
    ObservationFailureCode,
    ObservationMetrics,
    ObservationStatus,
    P53ValidationAttestation,
    P53ValidatorLaunch,
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
    load_nyu40_mapping,
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
    CandidateValidationAuthority,
    FileRecord,
    ObservationPackageRow,
    RealSuccessfulManifest,
    RealProvenanceAuthority,
    SyntheticSuccessfulManifest,
    TimingPackageRow,
    TrustedArtifactAuthority,
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


def _public_validation_fixture(
    tmp_path: Path,
    accepted: package.AcceptedCandidatePackage,
) -> tuple[
    package.AcceptedCandidatePackage,
    CandidateValidationAuthority,
    P53ValidationAttestation,
]:
    benchmark_root = Path(package.__file__).resolve(strict=True).parents[3]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    real_run = accepted.manifest.summary.run_kind is package.RunKind.REAL
    python_executable = Path(sys.executable).resolve(strict=True)
    p53 = P53ValidationAttestation(
        python_executable=str(python_executable),
        environment_sha256=_HASH,
        gpu_device_id=0,
        gpu_uuid=RAW_GPU_UUID,
        producer_git_commit=RAW_PRODUCER_COMMIT,
        raw_root=str(raw_root),
        raw_manifest_sha256=RAW_MANIFEST_SHA256,
        raw_index_sha256=RAW_INDEX_SHA256,
        validator_source_sha256=RAW_VALIDATOR_SOURCE_SHA256,
    )
    benchmark = BenchmarkEnvironmentAttestation(
        python_executable=str(python_executable),
        environment_sha256=_HASH,
        visible_device_count=1 if real_run else 0,
        gpu_name="NVIDIA GeForce RTX 3090" if real_run else "NOT_APPLICABLE",
        gpu_uuid=RAW_GPU_UUID if real_run else "NOT_APPLICABLE",
        timing_comparable=real_run,
    )
    raw = cast(
        Dict[str, object],
        json.loads(accepted.manifest.canonical_bytes()),
    )
    raw["attestations"] = {
        "benchmark": _embedded(
            cast(Dict[str, object], json.loads(benchmark.canonical_bytes()))
        ),
        "p53_validator": _embedded(
            cast(Dict[str, object], json.loads(p53.canonical_bytes()))
        ),
    }
    raw["raw_inputs"] = {
        "cohort_jsonl_sha256": package._COHORT_JSONL_SHA256,  # noqa: SLF001
        "raw_index_sha256": RAW_INDEX_SHA256,
        "raw_manifest_sha256": RAW_MANIFEST_SHA256,
        "raw_payload_tree_sha256": package._RAW_PAYLOAD_TREE_SHA256,  # noqa: SLF001
        "raw_producer_git_commit": RAW_PRODUCER_COMMIT,
        "selection_sha256": package._SELECTION_SHA256,  # noqa: SLF001
    }
    frozen_mapping = load_nyu40_mapping()
    frozen_vocabulary = tuple(entry.source_name for entry in frozen_mapping)
    candidate = _dict_section(raw, "candidate_commitment")
    candidate["mapping_sha256"] = NYU40_MAPPING_SHA256
    candidate["source_vocabulary"] = list(frozen_vocabulary)
    coverage = compute_static_coverage(frozen_mapping, (_science_grid(),) * 50)
    raw["coverage"] = {
        **asdict(coverage),
        "support_ratio": coverage.support_ratio,
    }
    if real_run:
        assert accepted.manifest.summary.real is not None
        metric_summary = accepted.manifest.summary.metrics
        assert metric_summary.primary.mean_iou is not None
        assert metric_summary.primary.mean_f1 is not None
        iou_rows = tuple(
            EndpointRow(
                ordinal=row.ordinal,
                observation_id=row.observation_id,
                scene_id=row.scene_id,
                endpoint=row.metrics.primary.iou,
                status=row.status,
                failure_code=row.failure_code,
            )
            for row in accepted.observations
        )
        gates = package.evaluate_candidate_gates(
            synthetic=False,
            coverage=coverage,
            trusted_cohort=TrustedCohort(_package_identities()),
            rows=iou_rows,
            mean_iou=metric_summary.primary.mean_iou,
            mean_f1=metric_summary.primary.mean_f1,
            latency=accepted.manifest.summary.real.latency,
            resource=accepted.manifest.summary.real.resource,
            code_license_status=accepted.manifest.summary.candidate.code_license_status,
            weight_license_status=(
                accepted.manifest.summary.candidate.weight_license_status
            ),
            provenance=_passed_provenance(),
        )
        raw["gates"] = {
            name: getattr(gates, name).value
            for name in (
                "complete_rows",
                "coverage",
                "latency",
                "license",
                "overall",
                "provenance",
                "quality",
                "resource",
            )
        }
        raw["candidate_status"] = gates.overall.value
    source_hashes = _dict_section(raw, "source_hashes")
    for value in source_hashes.values():
        assert isinstance(value, dict)
        cast(Dict[str, object], value)["sha256"] = _HASH
    mapping_source = source_hashes["mapping"]
    assert isinstance(mapping_source, dict)
    cast(Dict[str, object], mapping_source)["sha256"] = NYU40_MAPPING_SHA256
    bound = _accepted_with_manifest(accepted, raw)
    command = bound.manifest.fields["command"]
    assert isinstance(command, tuple)
    adapter = cast(Mapping[str, object], bound.manifest.fields["source_hashes"])[
        "adapter"
    ]
    assert isinstance(adapter, Mapping)
    real_authority: RealProvenanceAuthority | None = None
    if real_run:
        candidate_root = tmp_path / "candidate"
        checkpoint_root = tmp_path / "checkpoints"
        candidate_root.mkdir()
        checkpoint_root.mkdir()
        code_permission = TrustedArtifactAuthority(
            root_role="candidate_repository",
            path="LICENSE",
            file=FileRecord(1, _HASH),
        )
        weight_permission = TrustedArtifactAuthority(
            root_role="candidate_repository",
            path="WEIGHTS_LICENSE",
            file=FileRecord(1, _HASH),
        )
        real_authority = RealProvenanceAuthority(
            candidate_repository_root=candidate_root,
            checkpoint_root=checkpoint_root,
            checkpoint_path="model.pt",
            checkpoint_file=FileRecord(1, _HASH),
            code_permission=code_permission,
            weight_permission=weight_permission,
        )
    authority = CandidateValidationAuthority(
        expected_manifest_sha256=bound.validation.manifest.sha256,
        p53_launch=P53ValidatorLaunch(
            python_executable=python_executable,
            expected_environment_sha256=_HASH,
            raw_root=raw_root,
            timeout_seconds=30.0,
        ),
        benchmark_attestation=benchmark,
        expected_benchmark_attestation_sha256=benchmark.sha256,
        expected_command=cast(Tuple[str, ...], command),
        benchmark_repository_root=benchmark_root,
        expected_producer_commit=_COMMIT,
        candidate=bound.manifest.summary.candidate,
        mapping=CandidateMappingAuthority(
            source_vocabulary=frozen_vocabulary,
            mapping=frozen_mapping,
            mapping_sha256=NYU40_MAPPING_SHA256,
        ),
        adapter_path=cast(str, cast(Mapping[str, object], adapter)["path"]),
        environment_lock=TrustedArtifactAuthority(
            root_role="benchmark_repository",
            path=bound.manifest.summary.candidate.environment_lock_path,
            file=FileRecord(1, _HASH),
        ),
        real=real_authority,
    )
    return bound, authority, p53


def _install_public_validation_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    accepted: package.AcceptedCandidatePackage,
    authority: CandidateValidationAuthority,
    p53: P53ValidationAttestation,
    events: list[str],
) -> None:
    raw = _science_raw_observations()

    def inspect_git(root: Path) -> package.GitRepositoryState:
        events.append("git")
        if root == authority.benchmark_repository_root:
            return package.GitRepositoryState(_COMMIT, True)
        assert authority.real is not None
        assert root == authority.real.candidate_repository_root
        return package.GitRepositoryState(
            _COMMIT,
            True,
            authority.candidate.repository_url,
        )

    def cohort_index(
        root: Path,
        *,
        p53_attestation: P53ValidationAttestation,
    ) -> TrustedCohort:
        assert root == authority.p53_launch.raw_root
        assert p53_attestation == p53
        events.append("cohort-index")
        return TrustedCohort(_package_identities())

    def load_raw(
        root: Path,
        **kwargs: object,
    ) -> tuple[ValidatedRawObservation, ...]:
        assert root == authority.p53_launch.raw_root
        assert kwargs["p53_attestation"] == p53
        events.append("raw")
        return raw

    def accept(
        root: Path,
        *,
        expected_manifest_sha256: str,
        trusted_cohort: TrustedCohort,
        mapping_authority: CandidateMappingAuthority,
    ) -> package.AcceptedCandidatePackage:
        del root
        assert expected_manifest_sha256 == authority.expected_manifest_sha256
        assert trusted_cohort.identities == _package_identities()
        assert mapping_authority == authority.mapping
        events.append("b1")
        return accepted

    def source_record(root: Path, path: str) -> FileRecord:
        allowed_roots = {authority.benchmark_repository_root}
        if authority.real is not None:
            allowed_roots.update({
                authority.real.candidate_repository_root,
                authority.real.checkpoint_root,
            })
        assert root in allowed_roots
        assert path
        events.append("source")
        sha256 = (
            NYU40_MAPPING_SHA256
            if path.endswith("rgbd_segmenter_nyu40_mapping.json")
            else (
                RAW_VALIDATOR_SOURCE_SHA256
                if path.endswith("rgbd_segmenter_raw_frame_package.py")
                else _HASH
            )
        )
        return FileRecord(1, sha256)

    def oracle(_arrays: RawFrameArrays) -> np.ndarray:
        events.append("oracle")
        return _science_grid()

    monkeypatch.setattr(
        package,
        "run_p53_validation_subprocess",
        lambda launch: events.append("p53") or p53,
    )
    monkeypatch.setattr(package, "iter_validated_raw_observations", load_raw)
    monkeypatch.setattr(package, "_trusted_cohort_from_raw_index", cohort_index)
    monkeypatch.setattr(package, "accept_candidate_package", accept)
    monkeypatch.setattr(package, "_inspect_git_repository", inspect_git)
    monkeypatch.setattr(package, "_trusted_file_record", source_record)
    monkeypatch.setattr(
        package,
        "load_nyu40_mapping",
        lambda _path: authority.mapping.mapping,
    )
    monkeypatch.setattr(package, "project_oracle_target_labels", oracle)
    monkeypatch.setattr(
        package,
        "project_mapped_labels",
        lambda _labels, _arrays: _science_grid(),
    )
    monkeypatch.setattr(
        BenchmarkEnvironmentAttestation,
        "require_current_process",
        lambda self: None,
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


def test_public_validator_mints_only_endpoint_free_factory_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted, authority, p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    events: list[str] = []
    _install_public_validation_fakes(
        monkeypatch,
        accepted=accepted,
        authority=authority,
        p53=p53,
        events=events,
    )

    validated = package.validate_candidate_package(
        tmp_path / "untrusted-package",
        authority=authority,
    )

    assert (
        events.index("p53")
        < events.index("cohort-index")
        < events.index("b1")
        < events.index("raw")
        < events.index("oracle")
    )
    assert events.count("p53") == 2
    assert events.count("cohort-index") == 2
    assert events.count("b1") == 2
    assert validated.validation == accepted.validation
    assert validated.record.run_kind is package.RunKind.SYNTHETIC
    assert validated.record.candidate_id == authority.candidate.candidate_id
    assert validated.record.synthetic_semantic_scores_exposed is False
    record = json.loads(validated.record.canonical_bytes())
    assert "metrics" not in record
    assert "coverage" not in record
    assert "robustness" not in record
    with pytest.raises(ValueError, match="validator"):
        replace(validated.record, validation_status="PASS")
    with pytest.raises(ValueError, match="validator"):
        replace(validated, validation=validated.validation)


def test_public_validator_signature_has_no_injectable_science_authority() -> None:
    signature = inspect.signature(package.validate_candidate_package)
    assert tuple(signature.parameters) == ("root", "authority")
    assert signature.parameters["authority"].kind is inspect.Parameter.KEYWORD_ONLY
    forbidden = {
        "raw_observations",
        "provenance",
        "oracle_projector",
        "prediction_projector",
        "runtime",
    }
    assert forbidden.isdisjoint(signature.parameters)


def test_public_validator_rejects_stale_embedded_p53_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted, authority, p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    raw = cast(
        Dict[str, object],
        json.loads(accepted.manifest.canonical_bytes()),
    )
    stale = replace(p53, environment_sha256="2" * 64)
    attestations = _dict_section(raw, "attestations")
    attestations["p53_validator"] = _embedded(
        cast(Dict[str, object], json.loads(stale.canonical_bytes()))
    )
    stale_accepted = _accepted_with_manifest(accepted, raw)
    stale_authority = replace(
        authority,
        expected_manifest_sha256=stale_accepted.validation.manifest.sha256,
    )
    _install_public_validation_fakes(
        monkeypatch,
        accepted=stale_accepted,
        authority=stale_authority,
        p53=p53,
        events=[],
    )

    with pytest.raises(ValueError, match="manifest differs"):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=stale_authority,
        )


def test_public_validator_rejects_mapping_tuple_not_bound_to_frozen_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted, authority, p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    altered = replace(
        authority.mapping.mapping[0],
        canonical_name=authority.mapping.mapping[1].canonical_name,
        canonical_index=authority.mapping.mapping[1].canonical_index,
    )
    altered_mapping = CandidateMappingAuthority(
        source_vocabulary=authority.mapping.source_vocabulary,
        mapping=(altered,) + authority.mapping.mapping[1:],
        mapping_sha256=NYU40_MAPPING_SHA256,
    )
    altered_authority = replace(authority, mapping=altered_mapping)
    _install_public_validation_fakes(
        monkeypatch,
        accepted=accepted,
        authority=altered_authority,
        p53=p53,
        events=[],
    )
    monkeypatch.setattr(
        package,
        "load_nyu40_mapping",
        lambda _path: authority.mapping.mapping,
    )

    with pytest.raises(ValueError, match="mapping source contents"):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=altered_authority,
        )


def test_public_validator_rejects_unrelated_benchmark_clone(
    tmp_path: Path,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    _accepted, authority, _p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    unrelated = tmp_path / "unrelated-clone"
    unrelated.mkdir()
    unrelated_authority = replace(
        authority,
        benchmark_repository_root=unrelated,
    )

    with pytest.raises(ValueError, match="runtime .* source"):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=unrelated_authority,
        )


@pytest.mark.parametrize(
    "module_name",
    ["_constants_module", "_raw_package_module"],
)
def test_public_validator_rejects_relocated_runtime_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
    module_name: str,
) -> None:
    _accepted, authority, _p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    relocated = tmp_path / f"{module_name}.py"
    relocated.write_text("# relocated\n")
    module = getattr(package, module_name)
    monkeypatch.setattr(module, "__file__", str(relocated))

    with pytest.raises(ValueError, match="outside benchmark authority root"):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=authority,
        )


def test_public_validator_rejects_command_and_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted, authority, p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    wrong_command = replace(
        authority,
        expected_command=authority.expected_command + ("--extra",),
    )
    _install_public_validation_fakes(
        monkeypatch,
        accepted=accepted,
        authority=wrong_command,
        p53=p53,
        events=[],
    )
    with pytest.raises(ValueError, match="manifest differs"):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=wrong_command,
        )

    _install_public_validation_fakes(
        monkeypatch,
        accepted=accepted,
        authority=authority,
        p53=p53,
        events=[],
    )

    def wrong_source(_root: Path, path: str) -> FileRecord:
        sha256 = (
            RAW_VALIDATOR_SOURCE_SHA256
            if path.endswith("rgbd_segmenter_raw_frame_package.py")
            else (
                NYU40_MAPPING_SHA256
                if path.endswith("rgbd_segmenter_nyu40_mapping.json")
                else "2" * 64
            )
        )
        return FileRecord(1, sha256)

    monkeypatch.setattr(package, "_trusted_file_record", wrong_source)
    with pytest.raises(ValueError, match="source hash differs"):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=authority,
        )


def test_public_validator_rejects_final_git_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted, authority, p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    _install_public_validation_fakes(
        monkeypatch,
        accepted=accepted,
        authority=authority,
        p53=p53,
        events=[],
    )
    calls = 0

    def drifting_git(_root: Path) -> package.GitRepositoryState:
        nonlocal calls
        calls += 1
        return package.GitRepositoryState(_COMMIT, calls == 1)

    monkeypatch.setattr(package, "_inspect_git_repository", drifting_git)
    with pytest.raises(ValueError, match="Git state"):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=authority,
        )


def test_public_real_validator_accepts_but_keeps_deployability_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    real = _real_science_accepted(accepted_science_package)
    accepted, authority, p53 = _public_validation_fixture(tmp_path, real)
    events: list[str] = []
    _install_public_validation_fakes(
        monkeypatch,
        accepted=accepted,
        authority=authority,
        p53=p53,
        events=events,
    )

    validated = package.validate_candidate_package(
        tmp_path / "untrusted-package",
        authority=authority,
    )

    assert validated.record.validation_status == "PASS"
    assert validated.record.candidate_status == "FAIL"
    assert validated.record.run_kind is package.RunKind.REAL
    assert events.count("b1") == 2


@pytest.mark.parametrize(
    "fault",
    [
        "checkpoint",
        "environment_lock",
        "code_permission_root",
        "code_permission_bytes",
        "weight_permission_root",
        "weight_permission_bytes",
        "candidate_commit",
        "candidate_origin",
        "cuda_identity",
        "final_checkpoint_drift",
    ],
)
def test_public_real_validator_rejects_provenance_perturbations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
    fault: str,
) -> None:
    real = _real_science_accepted(accepted_science_package)
    accepted, authority, p53 = _public_validation_fixture(tmp_path, real)
    assert authority.real is not None
    if fault in {"code_permission_bytes", "code_permission_root"}:
        authority = replace(
            authority,
            real=replace(
                authority.real,
                code_permission=replace(
                    authority.real.code_permission,
                    root_role=(
                        "benchmark_repository"
                        if fault == "code_permission_root"
                        else authority.real.code_permission.root_role
                    ),
                    file=(
                        FileRecord(2, _HASH)
                        if fault == "code_permission_bytes"
                        else authority.real.code_permission.file
                    ),
                ),
            ),
        )
    elif fault in {"weight_permission_bytes", "weight_permission_root"}:
        authority = replace(
            authority,
            real=replace(
                authority.real,
                weight_permission=replace(
                    authority.real.weight_permission,
                    root_role=(
                        "benchmark_repository"
                        if fault == "weight_permission_root"
                        else authority.real.weight_permission.root_role
                    ),
                    file=(
                        FileRecord(2, _HASH)
                        if fault == "weight_permission_bytes"
                        else authority.real.weight_permission.file
                    ),
                ),
            ),
        )
    elif fault == "cuda_identity":
        raw = cast(
            Dict[str, object],
            json.loads(accepted.manifest.canonical_bytes()),
        )
        cuda = _dict_section(raw, "cuda_evidence")
        for name in ("before", "after"):
            snapshot = _dict_section(cuda, name)
            snapshot["gpu_uuid"] = "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"
        accepted = _accepted_with_manifest(accepted, raw)
        authority = replace(
            authority,
            expected_manifest_sha256=accepted.validation.manifest.sha256,
        )
    _install_public_validation_fakes(
        monkeypatch,
        accepted=accepted,
        authority=authority,
        p53=p53,
        events=[],
    )
    original_file_record = package._trusted_file_record  # noqa: SLF001
    checkpoint_calls = 0

    def perturbed_file_record(root: Path, path: str) -> FileRecord:
        nonlocal checkpoint_calls
        record = original_file_record(root, path)
        if fault == "checkpoint" and path == "model.pt":
            return FileRecord(1, "2" * 64)
        if fault == "environment_lock" and path == "environment.lock":
            return FileRecord(1, "2" * 64)
        if fault == "final_checkpoint_drift" and path == "model.pt":
            checkpoint_calls += 1
            if checkpoint_calls == 2:
                return FileRecord(1, "2" * 64)
        return record

    monkeypatch.setattr(package, "_trusted_file_record", perturbed_file_record)
    if fault in {"candidate_commit", "candidate_origin"}:
        original_git = package._inspect_git_repository  # noqa: SLF001
        assert authority.real is not None
        candidate_repository_root = authority.real.candidate_repository_root

        def wrong_git_identity(root: Path) -> package.GitRepositoryState:
            state = original_git(root)
            if root == candidate_repository_root:
                return replace(
                    state,
                    commit=("f" * 40 if fault == "candidate_commit" else state.commit),
                    origin_url=(
                        "https://example.invalid/wrong"
                        if fault == "candidate_origin"
                        else state.origin_url
                    ),
                )
            return state

        monkeypatch.setattr(package, "_inspect_git_repository", wrong_git_identity)

    with pytest.raises(ValueError):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=authority,
        )


def test_trusted_file_record_rejects_final_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    target = root / "artifact"
    moved = root / "moved"
    target.write_bytes(b"trusted")
    original_read = package.os.read
    swapped = False

    def swapping_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        data = original_read(descriptor, count)
        if not swapped:
            target.rename(moved)
            target.write_bytes(b"trusted")
            swapped = True
        return data

    monkeypatch.setattr(package.os, "read", swapping_read)
    with pytest.raises(ValueError, match="binding changed|changed while reading"):
        package._trusted_file_record(root, "artifact")  # noqa: SLF001


def test_trusted_file_record_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    os.mkfifo(root / "artifact")
    with pytest.raises(ValueError, match="bounded unique file"):
        package._trusted_file_record(root, "artifact")  # noqa: SLF001


def test_public_validator_rejects_package_mutation_after_first_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted, authority, p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    _install_public_validation_fakes(
        monkeypatch,
        accepted=accepted,
        authority=authority,
        p53=p53,
        events=[],
    )
    calls = 0

    def mutating_accept(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("candidate package mutated after B1")
        return accepted

    monkeypatch.setattr(package, "accept_candidate_package", mutating_accept)
    with pytest.raises(ValueError, match="mutated after B1"):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=authority,
        )
    assert calls == 2


def test_public_validator_rejects_final_benchmark_attestation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted, authority, p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    events: list[str] = []
    _install_public_validation_fakes(
        monkeypatch,
        accepted=accepted,
        authority=authority,
        p53=p53,
        events=events,
    )

    def drifted(_self: BenchmarkEnvironmentAttestation) -> None:
        raise ValueError("benchmark environment drifted")

    monkeypatch.setattr(
        BenchmarkEnvironmentAttestation,
        "require_current_process",
        drifted,
    )
    with pytest.raises(ValueError, match="environment drifted"):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=authority,
        )
    assert events.count("b1") == 2


def test_public_validator_rejects_final_p53_attestation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    accepted, authority, p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    events: list[str] = []
    _install_public_validation_fakes(
        monkeypatch,
        accepted=accepted,
        authority=authority,
        p53=p53,
        events=events,
    )
    calls = 0

    def drifting_p53(_launch: P53ValidatorLaunch) -> P53ValidationAttestation:
        nonlocal calls
        calls += 1
        return p53 if calls == 1 else replace(p53, environment_sha256="2" * 64)

    monkeypatch.setattr(package, "run_p53_validation_subprocess", drifting_p53)
    with pytest.raises(ValueError, match="P5.3 attestation changed"):
        package.validate_candidate_package(
            tmp_path / "untrusted-package",
            authority=authority,
        )
    assert calls == 2
    assert events.count("b1") == 1


def _write_fake_git_metadata(root: Path, revision: str = _COMMIT) -> None:
    git_directory = root / ".git"
    git_directory.mkdir()
    (git_directory / "objects" / "info").mkdir(parents=True)
    (git_directory / "info").mkdir()
    (git_directory / "refs" / "heads").mkdir(parents=True)
    (git_directory / "HEAD").write_text(f"{revision}\n")
    (git_directory / "config").write_text("[core]\n\tbare = false\n")
    (git_directory / "info" / "exclude").write_text("")


def _initialize_real_git_repository(root: Path) -> None:
    root.mkdir()
    subprocess.run(("/usr/bin/git", "init", "-q"), cwd=root, check=True)
    subprocess.run(
        ("/usr/bin/git", "config", "user.email", "validator@example.invalid"),
        cwd=root,
        check=True,
    )
    subprocess.run(
        ("/usr/bin/git", "config", "user.name", "Validator"),
        cwd=root,
        check=True,
    )


def _git_blob_id(data: bytes) -> str:
    digest = hashlib.new("sha1", usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


def _git_tree_bytes(entries: Mapping[str, Tuple[str, bytes]]) -> bytes:
    return b"".join(
        f"{mode} blob {_git_blob_id(data)}\t{path}\x00".encode()
        for path, (mode, data) in sorted(entries.items())
    )


def test_git_inspection_normalizes_subprocess_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _write_fake_git_metadata(root)

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise subprocess.CalledProcessError(1, ("git",))

    monkeypatch.setattr(package.subprocess, "run", fail)
    with pytest.raises(ValueError, match="inspection failed"):
        package._inspect_git_repository(root)  # noqa: SLF001


def test_git_inspection_anchors_repo_fd_and_ignores_process_poisoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _write_fake_git_metadata(root)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker"))
    poisoned_index = str(tmp_path / "attacker-index")
    monkeypatch.setenv("GIT_INDEX_FILE", poisoned_index)
    monkeypatch.setenv("PATH", str(tmp_path / "fake-bin"))
    monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "attacker.so"))
    calls: list[tuple[tuple[str, ...], Mapping[str, str], float, tuple[int, ...]]] = []

    def completed(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = cast(Mapping[str, str], kwargs["env"])
        timeout = cast(float, kwargs["timeout"])
        passed = cast(Tuple[int, ...], kwargs["pass_fds"])
        calls.append((args, environment, timeout, passed))
        stdout = (
            b"remote.origin.url\nhttps://example.invalid/repository\x00"
            if "config" in args
            else b""
        )
        return subprocess.CompletedProcess(args, 0, stdout, b"")

    monkeypatch.setattr(package.subprocess, "run", completed)
    state = package._inspect_git_repository(root)  # noqa: SLF001
    assert state == package.GitRepositoryState(
        _COMMIT,
        True,
        "https://example.invalid/repository",
    )
    assert len(calls) == 2
    base_environment_keys = {
        "GIT_ATTR_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_SYSTEM",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OPTIONAL_LOCKS",
        "GIT_TERMINAL_PROMPT",
        "HOME",
        "LANG",
        "LC_ALL",
    }
    assert set(calls[0][1]) == base_environment_keys
    assert all(
        set(environment) == base_environment_keys for _, environment, _, _ in calls
    )
    assert all(
        environment["GIT_OPTIONAL_LOCKS"] == "0" for _, environment, _, _ in calls
    )
    assert all(args[0] == "/usr/bin/git" for args, _, _, _ in calls)
    assert all(str(root) not in args for args, _, _, _ in calls)
    assert all(
        any("/proc/self/fd/" in value for value in args) for args, _, _, _ in calls
    )
    assert len(calls[0][3]) == 1
    assert all(len(passed) == 3 for _, _, _, passed in calls[1:])
    assert all(
        any(f"/proc/self/fd/{passed[0]}" in value for value in args)
        for args, _, _, passed in calls[1:]
    )
    assert (
        "core.fsmonitor=false" in calls[1][0]
        and "core.hooksPath=/dev/null" in calls[1][0]
        and "diff.external=" in calls[1][0]
        and "core.excludesFile=/dev/null" in calls[1][0]
        and "core.ignoreCase=false" in calls[1][0]
    )
    commands = [
        next(name for name in ("config", "ls-tree", "check-ignore") if name in args)
        for args, _, _, _ in calls
    ]
    assert commands == ["config", "ls-tree"]
    assert all("GIT_INDEX_FILE" not in environment for _, environment, _, _ in calls)
    assert all(
        poisoned_index not in environment.values() for _, environment, _, _ in calls
    )
    assert all(
        not any(
            forbidden in args
            for forbidden in ("status", "read-tree", "ls-files", "diff", "hash-object")
        )
        for args, _, _, _ in calls
    )
    assert all(
        timeout == package._GIT_INSPECTION_TIMEOUT_SECONDS  # noqa: SLF001
        for _, _, timeout, _ in calls
    )


@pytest.mark.parametrize(
    ("fault", "expected_clean"),
    [
        ("clean", True),
        ("same-size-content", False),
        ("executable-bit", False),
        ("executable-owner-bit-removed", False),
        ("regular-nonowner-execute", True),
        ("symlink-target", False),
        ("untracked-file", False),
        ("missing-file", False),
        ("ignored-file", True),
        ("ignored-spaced-file", True),
        ("nested-ignored-file", True),
        ("info-excluded-file", False),
        ("extra-empty-directory", True),
    ],
)
def test_git_inspection_uses_no_filter_real_worktree_check(
    tmp_path: Path,
    fault: str,
    expected_clean: bool,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(
        ("/usr/bin/git", "init", "-q"),
        cwd=root,
        check=True,
    )
    subprocess.run(
        ("/usr/bin/git", "config", "user.email", "validator@example.invalid"),
        cwd=root,
        check=True,
    )
    subprocess.run(
        ("/usr/bin/git", "config", "user.name", "Validator"),
        cwd=root,
        check=True,
    )
    (root / "bin").mkdir()
    executable = b"#!/bin/sh\n"
    (root / "bin" / "run").write_bytes(executable)
    (root / "bin" / "run").chmod(0o755)
    original = b"original"
    (root / "tracked.txt").write_bytes(original)
    os.symlink("tracked.txt", root / "link")
    (root / ".gitignore").write_text("ignored.txt\npotted plant.pt\n")
    (root / "nested").mkdir()
    (root / "nested" / ".gitignore").write_text("nested-ignored.txt\n")
    subprocess.run(
        ("/usr/bin/git", "add", "."),
        cwd=root,
        check=True,
    )
    subprocess.run(
        ("/usr/bin/git", "commit", "-q", "-m", "fixture"),
        cwd=root,
        check=True,
    )

    if fault == "same-size-content":
        (root / "tracked.txt").write_bytes(b"mutated!")
    elif fault == "executable-bit":
        (root / "tracked.txt").chmod(0o755)
    elif fault == "executable-owner-bit-removed":
        (root / "bin" / "run").chmod(0o601)
    elif fault == "regular-nonowner-execute":
        (root / "tracked.txt").chmod(0o601)
    elif fault == "symlink-target":
        (root / "link").unlink()
        os.symlink("bin/run", root / "link")
    elif fault == "untracked-file":
        (root / "extra.txt").write_text("extra")
    elif fault == "missing-file":
        (root / "tracked.txt").unlink()
    elif fault == "ignored-file":
        (root / "ignored.txt").write_text("ignored")
    elif fault == "ignored-spaced-file":
        (root / "potted plant.pt").write_text("ignored")
    elif fault == "nested-ignored-file":
        (root / "nested" / "nested-ignored.txt").write_text("ignored")
    elif fault == "info-excluded-file":
        (root / ".git" / "info" / "exclude").write_text("private.txt\n")
        (root / "private.txt").write_text("must remain dirty")
    elif fault == "extra-empty-directory":
        (root / "empty").mkdir()

    if fault == "info-excluded-file":
        with pytest.raises(ValueError):
            package._inspect_git_repository(root)  # noqa: SLF001
        return
    state = package._inspect_git_repository(root)  # noqa: SLF001
    assert state.clean is expected_clean


@pytest.mark.parametrize(
    ("ignored", "hardlinked", "expected_clean"),
    [
        (True, False, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_git_inspection_does_not_read_oversized_untracked_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ignored: bool,
    hardlinked: bool,
    expected_clean: bool,
) -> None:
    root = tmp_path / "repository"
    _initialize_real_git_repository(root)
    (root / ".gitignore").write_text("ignored-*.bin\n")
    subprocess.run(("/usr/bin/git", "add", ".gitignore"), cwd=root, check=True)
    subprocess.run(
        ("/usr/bin/git", "commit", "-q", "-m", "fixture"),
        cwd=root,
        check=True,
    )

    filename = "ignored-large.bin" if ignored else "unignored-large.bin"
    target = root / filename
    with target.open("wb") as stream:
        stream.truncate(package._MAX_TRUSTED_ARTIFACT_BYTES + 1)  # noqa: SLF001
    guarded_names = {filename}
    if hardlinked:
        alias = root / "ignored-large-alias.bin"
        os.link(target, alias)
        guarded_names.add(alias.name)

    real_open = os.open

    def guarded_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fsdecode(path) in guarded_names:
            raise AssertionError("untracked file must not be opened")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(package.os, "open", guarded_open)
    state = package._inspect_git_repository(root)  # noqa: SLF001
    assert state.clean is expected_clean


def test_git_inspection_rejects_oversized_tracked_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    _initialize_real_git_repository(root)
    tracked = root / "tracked.txt"
    tracked.write_bytes(b"tracked")
    subprocess.run(("/usr/bin/git", "add", "tracked.txt"), cwd=root, check=True)
    subprocess.run(
        ("/usr/bin/git", "commit", "-q", "-m", "fixture"),
        cwd=root,
        check=True,
    )
    monkeypatch.setattr(
        package,
        "_MAX_TRUSTED_ARTIFACT_BYTES",
        tracked.stat().st_size - 1,
    )

    with pytest.raises(ValueError, match="Git worktree file is unsafe: tracked.txt"):
        package._inspect_git_repository(root)  # noqa: SLF001


def test_git_inspection_prunes_trusted_ignored_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    _initialize_real_git_repository(root)
    (root / ".gitignore").write_text("ignored-dir/\n")
    subprocess.run(("/usr/bin/git", "add", ".gitignore"), cwd=root, check=True)
    subprocess.run(
        ("/usr/bin/git", "commit", "-q", "-m", "fixture"),
        cwd=root,
        check=True,
    )
    ignored = root / "ignored-dir"
    ignored.mkdir()
    with (ignored / "huge.bin").open("wb") as stream:
        stream.truncate(package._MAX_TRUSTED_ARTIFACT_BYTES + 1)  # noqa: SLF001
    os.mkfifo(ignored / "pipe")
    unreadable = ignored / "unreadable"
    unreadable.mkdir()
    unreadable.chmod(0)

    real_open = os.open
    real_listdir = os.listdir

    def guarded_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fsdecode(path) == ignored.name:
            raise AssertionError("trusted ignored directory must not be opened")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def guarded_listdir(descriptor: int) -> list[str]:
        if Path(f"/proc/self/fd/{descriptor}").resolve() == ignored:
            raise AssertionError("trusted ignored directory must not be listed")
        return real_listdir(descriptor)

    monkeypatch.setattr(package.os, "open", guarded_open)
    monkeypatch.setattr(package.os, "listdir", guarded_listdir)
    try:
        state = package._inspect_git_repository(root)  # noqa: SLF001
    finally:
        unreadable.chmod(0o700)
    assert state.clean


def test_git_inspection_never_prunes_directory_with_tracked_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    _initialize_real_git_repository(root)
    (root / ".gitignore").write_text("ignored-dir/\n")
    tracked = root / "ignored-dir" / "tracked.txt"
    tracked.parent.mkdir()
    tracked.write_text("original")
    subprocess.run(
        ("/usr/bin/git", "add", ".gitignore"),
        cwd=root,
        check=True,
    )
    subprocess.run(
        ("/usr/bin/git", "add", "-f", "ignored-dir/tracked.txt"),
        cwd=root,
        check=True,
    )
    subprocess.run(
        ("/usr/bin/git", "commit", "-q", "-m", "fixture"),
        cwd=root,
        check=True,
    )
    tracked.write_text("mutated")
    opened_tracked = False
    real_open = os.open

    def recording_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal opened_tracked
        if os.fsdecode(path) == tracked.name:
            opened_tracked = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(package.os, "open", recording_open)
    state = package._inspect_git_repository(root)  # noqa: SLF001
    assert not state.clean
    assert opened_tracked


def test_git_inspection_descends_into_unignored_directory(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _initialize_real_git_repository(root)
    subprocess.run(
        ("/usr/bin/git", "commit", "--allow-empty", "-q", "-m", "fixture"),
        cwd=root,
        check=True,
    )
    visible = root / "visible"
    visible.mkdir()
    (visible / "extra.txt").write_text("extra")

    state = package._inspect_git_repository(root)  # noqa: SLF001
    assert not state.clean


def test_git_inspection_does_not_trust_untracked_ignore_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    _initialize_real_git_repository(root)
    subprocess.run(
        ("/usr/bin/git", "commit", "--allow-empty", "-q", "-m", "fixture"),
        cwd=root,
        check=True,
    )
    (root / ".gitignore").write_text("ignored-dir/\n")
    ignored = root / "ignored-dir"
    ignored.mkdir()
    (ignored / "payload.txt").write_text("payload")
    opened_ignored = False
    real_open = os.open

    def recording_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal opened_ignored
        if os.fsdecode(path) == ignored.name:
            opened_ignored = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(package.os, "open", recording_open)
    state = package._inspect_git_repository(root)  # noqa: SLF001
    assert not state.clean
    assert opened_ignored


def test_git_inspection_treats_negated_paths_as_unignored(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _initialize_real_git_repository(root)
    (root / ".gitignore").write_text("ignored-dir/*\n!ignored-dir/keep.txt\n")
    subprocess.run(("/usr/bin/git", "add", ".gitignore"), cwd=root, check=True)
    subprocess.run(
        ("/usr/bin/git", "commit", "-q", "-m", "fixture"),
        cwd=root,
        check=True,
    )
    ignored = root / "ignored-dir"
    ignored.mkdir()
    (ignored / "hidden.txt").write_text("ignored")
    (ignored / "keep.txt").write_text("unignored")

    state = package._inspect_git_repository(root)  # noqa: SLF001
    assert not state.clean


def test_git_inspection_rejects_pruned_directory_fingerprint_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    _initialize_real_git_repository(root)
    (root / ".gitignore").write_text("ignored-dir/\n")
    subprocess.run(("/usr/bin/git", "add", ".gitignore"), cwd=root, check=True)
    subprocess.run(
        ("/usr/bin/git", "commit", "-q", "-m", "fixture"),
        cwd=root,
        check=True,
    )
    ignored = root / "ignored-dir"
    ignored.mkdir()
    real_check_ignore = package._run_bounded_git_check_ignore  # noqa: SLF001
    directory_queries = 0

    def mutating_check_ignore(
        command: tuple[str, ...],
        *,
        input_bytes: bytes,
        environment: Mapping[str, str],
        passed_descriptors: Tuple[int, ...],
        output_limit: int,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal directory_queries
        if input_bytes == b"ignored-dir\x00":
            directory_queries += 1
            if directory_queries == 2:
                (ignored / "late.txt").write_text("late")
        return real_check_ignore(
            command,
            input_bytes=input_bytes,
            environment=environment,
            passed_descriptors=passed_descriptors,
            output_limit=output_limit,
        )

    monkeypatch.setattr(
        package,
        "_run_bounded_git_check_ignore",
        mutating_check_ignore,
    )
    with pytest.raises(ValueError, match="directory changed|worktree changed"):
        package._inspect_git_repository(root)  # noqa: SLF001
    assert directory_queries == 2


def test_git_check_ignore_output_budget_remains_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = b".gitignore\x001\x00ignored\x00ignored\x00"

    def completed(
        command: tuple[str, ...],
        *,
        input_bytes: bytes,
        environment: Mapping[str, str],
        passed_descriptors: Tuple[int, ...],
        output_limit: int,
    ) -> subprocess.CompletedProcess[bytes]:
        del input_bytes, environment, passed_descriptors, output_limit
        return subprocess.CompletedProcess(command, 0, output, b"")

    monkeypatch.setattr(package, "_run_bounded_git_check_ignore", completed)
    monkeypatch.setattr(package, "_MAX_GIT_IGNORE_BYTES", len(output) - 1)
    with pytest.raises(ValueError, match="output budget"):
        package._classify_git_ignored_paths(  # noqa: SLF001
            ("ignored",),
            git_prefix=("/usr/bin/git",),
            git_environment={},
            passed_descriptors=(),
            budget=package._GitIgnoreBudget(),  # noqa: SLF001
        )


def test_git_check_ignore_streaming_output_cap() -> None:
    command = (
        sys.executable,
        "-c",
        "import sys;sys.stdin.buffer.read();sys.stdout.buffer.write(b'x'*1024)",
    )
    with pytest.raises(ValueError, match="output budget"):
        package._run_bounded_git_check_ignore(  # noqa: SLF001
            command,
            input_bytes=b"path\x00",
            environment={},
            passed_descriptors=(),
            output_limit=10,
        )


def test_git_check_ignore_rejects_path_budget_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("path budget must fail before subprocess")

    monkeypatch.setattr(package, "_MAX_GIT_IGNORE_PATHS", 1)
    monkeypatch.setattr(package, "_run_bounded_git_check_ignore", forbidden)
    with pytest.raises(ValueError, match="path budget"):
        package._classify_git_ignored_paths(  # noqa: SLF001
            ("one", "two"),
            git_prefix=("/usr/bin/git",),
            git_environment={},
            passed_descriptors=(),
            budget=package._GitIgnoreBudget(),  # noqa: SLF001
        )


def test_git_inspection_does_not_reopen_mutated_filter_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _write_fake_git_metadata(root)
    git_directory = root / ".git"
    config_path = git_directory / "config"
    commands: list[str] = []

    def completed(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        command = next(
            name for name in ("config", "ls-tree", "check-ignore") if name in args
        )
        commands.append(command)
        if command == "config":
            config_path.write_text('[filter "attack"]\n\tclean = /bin/false\n')
            stdout = b"core.bare\nfalse\x00"
        else:
            stdout = b""
        return subprocess.CompletedProcess(args, 0, stdout, b"")

    monkeypatch.setattr(package.subprocess, "run", completed)
    with pytest.raises(ValueError, match="metadata file changed"):
        package._inspect_git_repository(root)  # noqa: SLF001
    assert commands == ["config", "ls-tree"]
    assert all(
        command not in commands for command in ("diff", "checkout", "hash-object")
    )


def test_git_inspection_recaptures_tracked_gitignore_after_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _write_fake_git_metadata(root)
    ignore_bytes = b"ignored.txt\n"
    (root / ".gitignore").write_bytes(ignore_bytes)
    (root / "ignored.txt").write_text("ignored")
    tree = _git_tree_bytes({".gitignore": ("100644", ignore_bytes)})
    mutations = 0

    def completed(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        if "config" in args:
            stdout = b""
        else:
            stdout = tree
        return subprocess.CompletedProcess(args, 0, stdout, b"")

    def check_ignore(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal mutations
        del kwargs
        mutations += 1
        (root / ".gitignore").write_bytes(b"ignored.bin\n")
        time.sleep(0.01)
        (root / ".gitignore").write_bytes(ignore_bytes)
        stdout = b".gitignore\x001\x00ignored.txt\x00ignored.txt\x00"
        return subprocess.CompletedProcess(command, 0, stdout, b"")

    monkeypatch.setattr(package.subprocess, "run", completed)
    monkeypatch.setattr(package, "_run_bounded_git_check_ignore", check_ignore)
    with pytest.raises(ValueError, match="worktree changed"):
        package._inspect_git_repository(root)  # noqa: SLF001
    assert mutations == 1


def test_git_inspection_rejects_pruned_ignore_decision_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _write_fake_git_metadata(root)
    ignore_bytes = b"ignored-dir/\n"
    (root / ".gitignore").write_bytes(ignore_bytes)
    (root / "ignored-dir").mkdir()
    tree = _git_tree_bytes({".gitignore": ("100644", ignore_bytes)})
    decisions = 0

    def completed(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        if "config" in args:
            stdout = b""
        else:
            stdout = tree
        return subprocess.CompletedProcess(args, 0, stdout, b"")

    def check_ignore(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal decisions
        del kwargs
        decisions += 1
        pattern = b"ignored-dir/" if decisions == 1 else b"ignored*/"
        stdout = b".gitignore\x001\x00" + pattern + b"\x00ignored-dir\x00"
        return subprocess.CompletedProcess(command, 0, stdout, b"")

    monkeypatch.setattr(package.subprocess, "run", completed)
    monkeypatch.setattr(package, "_run_bounded_git_check_ignore", check_ignore)
    with pytest.raises(ValueError, match="worktree changed"):
        package._inspect_git_repository(root)  # noqa: SLF001
    assert decisions == 2


def test_git_inspection_recaptures_tracked_gitignore_ancestor_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _write_fake_git_metadata(root)
    ignore_bytes = b"ignored.txt\n"
    nested = root / "nested"
    nested.mkdir()
    (nested / ".gitignore").write_bytes(ignore_bytes)
    (nested / "ignored.txt").write_text("ignored")
    tree = _git_tree_bytes({"nested/.gitignore": ("100644", ignore_bytes)})
    mutations = 0

    def completed(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        if "config" in args:
            stdout = b""
        else:
            stdout = tree
        return subprocess.CompletedProcess(args, 0, stdout, b"")

    def check_ignore(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal mutations
        del kwargs
        mutations += 1
        saved = tmp_path / "saved-nested"
        replacement = tmp_path / "replacement-nested"
        nested.rename(saved)
        replacement.mkdir()
        replacement.rename(nested)
        nested.rename(replacement)
        saved.rename(nested)
        nested.chmod(0o700)
        time.sleep(0.01)
        nested.chmod(0o755)
        stdout = b"nested/.gitignore\x001\x00ignored.txt\x00nested/ignored.txt\x00"
        return subprocess.CompletedProcess(command, 0, stdout, b"")

    monkeypatch.setattr(package.subprocess, "run", completed)
    monkeypatch.setattr(package, "_run_bounded_git_check_ignore", check_ignore)
    with pytest.raises(
        ValueError, match="worktree changed|directory changed|root changed"
    ):
        package._inspect_git_repository(root)  # noqa: SLF001
    assert mutations == 1


@pytest.mark.parametrize(
    "output",
    [
        (
            b".gitignore\x001\x00ignored.txt\x00ignored.txt\x00"
            b".gitignore\x002\x00ignored.txt\x00ignored.txt\x00"
        ),
        b"/outside/.gitignore\x001\x00ignored.txt\x00ignored.txt\x00",
    ],
)
def test_git_check_ignore_rejects_malformed_or_unsafe_decisions(output: bytes) -> None:
    with pytest.raises(ValueError):
        package._parse_git_check_ignore(output)  # noqa: SLF001


def test_git_check_ignore_accepts_exact_spaced_paths() -> None:
    output = b".gitignore\x002\x00potted plant.pt\x00data/reference/potted plant.pt\x00"
    assert package._parse_git_check_ignore(output) == {  # noqa: SLF001
        "data/reference/potted plant.pt": (".gitignore", "potted plant.pt")
    }


def test_git_check_ignore_preserves_negated_decision() -> None:
    output = b".gitignore\x002\x00!keep.txt\x00keep.txt\x00"
    assert package._parse_git_check_ignore(output) == {  # noqa: SLF001
        "keep.txt": (".gitignore", "!keep.txt")
    }


def test_git_inspection_rejects_head_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _write_fake_git_metadata(root)
    other_commit = "f" * 40

    def completed(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        if "ls-tree" in args:
            (root / ".git" / "HEAD").write_text(f"{other_commit}\n")
        stdout = b""
        return subprocess.CompletedProcess(args, 0, stdout, b"")

    monkeypatch.setattr(package.subprocess, "run", completed)
    with pytest.raises(ValueError, match="metadata file changed"):
        package._inspect_git_repository(root)  # noqa: SLF001


def test_git_tree_rejects_submodules() -> None:
    submodule = f"160000 commit {_COMMIT}\tdependency\x00".encode()
    with pytest.raises(ValueError, match="submodules"):
        package._parse_git_tree(submodule)  # noqa: SLF001


@pytest.mark.parametrize(
    "path",
    [
        "potted plant.pt",
        " leading.pt",
        "trailing.pt ",
        "unicode-雪.pt",
        "back\\slash.pt",
        "line\nbreak.pt",
        "tab\tfile.pt",
        "control-\x01.pt",
    ],
)
def test_git_tree_accepts_nul_delimited_posix_names(path: str) -> None:
    tree = package._parse_git_tree(  # noqa: SLF001
        _git_tree_bytes({path: ("100644", b"content")})
    )
    assert tuple(tree) == (path,)


@pytest.mark.parametrize(
    "raw_path",
    [
        b"",
        b".",
        b"/absolute",
        b"../outside",
        b"nested/../outside",
        b"nested/./file",
        b"nested//file",
        b"nested/file/",
        b"\xff",
    ],
)
def test_git_tree_rejects_unsafe_or_invalid_paths(raw_path: bytes) -> None:
    prefix = f"100644 blob {_git_blob_id(b'content')}\t".encode()
    with pytest.raises(ValueError):
        package._parse_git_tree(prefix + raw_path + b"\x00")  # noqa: SLF001


@pytest.mark.parametrize("path", ["contains\x00nul", "\udcff"])
def test_git_path_rejects_unrepresentable_names(path: str) -> None:
    with pytest.raises(ValueError):
        package._git_relative_path(path)  # noqa: SLF001


@pytest.mark.parametrize(
    "output",
    [
        b"../.gitignore\x001\x00ignored\x00ignored\x00",
        b".gitignore\x001\x00ignored\x00../outside\x00",
        b".gitignore\x001\x00ignored\x00nested//file\x00",
        b".gitignore\x001\x00ignored\x00invalid-\xff\x00",
    ],
)
def test_git_check_ignore_rejects_unsafe_or_invalid_paths(output: bytes) -> None:
    with pytest.raises(ValueError):
        package._parse_git_check_ignore(output)  # noqa: SLF001


def test_git_inspection_rejects_root_path_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    moved = tmp_path / "repository-moved"
    root.mkdir()
    _write_fake_git_metadata(root)
    captured: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

    def swapping_completed(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        passed = cast(Tuple[int, ...], kwargs["pass_fds"])
        captured.append((args, passed))
        if len(captured) == 1:
            root.rename(moved)
            root.mkdir()
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(package.subprocess, "run", swapping_completed)
    try:
        with pytest.raises(ValueError, match="root changed|ancestor binding"):
            package._inspect_git_repository(root)  # noqa: SLF001
        assert all(str(root) not in args for args, _ in captured)
        assert len(captured[0][1]) == 1
        assert all(len(passed) == 3 for _, passed in captured[1:])
        assert all(
            any(f"/proc/self/fd/{passed[0]}" in value for value in args)
            for args, passed in captured[1:]
        )
    finally:
        if moved.exists():
            root.rmdir()
            moved.rename(root)


@pytest.mark.parametrize(
    "key",
    [
        "filter.attack.clean",
        "include.path",
        "includeif.gitdir:/tmp.path",
        "core.worktree",
        "core.attributesfile",
        "extensions.objectformat",
        "extensions.partialclone",
        "extensions.worktreeconfig",
        "core.fsmonitor",
        "core.hookspath",
        "diff.external",
    ],
)
def test_git_inspection_rejects_unsafe_local_config_before_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _write_fake_git_metadata(root)
    commands: list[str] = []

    def unsafe_config(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        command = next(
            name for name in ("config", "ls-tree", "check-ignore") if name in args
        )
        commands.append(command)
        return subprocess.CompletedProcess(
            args,
            0,
            f"{key}\nmalicious\x00".encode(),
            b"",
        )

    monkeypatch.setattr(package.subprocess, "run", unsafe_config)
    with pytest.raises(ValueError, match="unsafe local Git config"):
        package._inspect_git_repository(root)  # noqa: SLF001
    assert commands == ["config"]


def test_git_inspection_rejects_git_directory_file_and_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_root = tmp_path / "file-root"
    file_root.mkdir()
    (file_root / ".git").write_text("gitdir: elsewhere\n")
    with pytest.raises(ValueError, match="anchored .git directory"):
        package._inspect_git_repository(file_root)  # noqa: SLF001

    root = tmp_path / "repository"
    root.mkdir()
    git_directory = root / ".git"
    moved = root / ".git-moved"
    _write_fake_git_metadata(root)
    calls = 0

    def swapping_git_directory(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            git_directory.rename(moved)
            git_directory.mkdir()
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(package.subprocess, "run", swapping_git_directory)
    try:
        with pytest.raises(ValueError, match="changed"):
            package._inspect_git_repository(root)  # noqa: SLF001
    finally:
        git_directory.rmdir()
        moved.rename(git_directory)


def test_real_provenance_authority_supports_external_checkpoint_root(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    checkpoint_root = tmp_path / "checkpoints"
    candidate_root.mkdir()
    checkpoint_root.mkdir()
    record = FileRecord(1, _HASH)
    permission = TrustedArtifactAuthority(
        root_role="candidate_repository",
        path="LICENSE",
        file=record,
    )
    real = RealProvenanceAuthority(
        candidate_repository_root=candidate_root,
        checkpoint_root=checkpoint_root,
        checkpoint_path="model.pt",
        checkpoint_file=record,
        code_permission=permission,
        weight_permission=permission,
    )
    assert real.checkpoint_root == checkpoint_root


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


def _publication_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted: package.AcceptedCandidatePackage,
) -> tuple[
    package.SuccessfulPackageArtifacts,
    CandidateValidationAuthority,
    package.CandidateValidationRecord,
]:
    bound, authority, _ = _public_validation_fixture(tmp_path, accepted)
    artifacts = package.SuccessfulPackageArtifacts.from_accepted(bound)
    record = package.CandidateValidationRecord(
        schema_version=1,
        validation_status="PASS",
        candidate_status="NOT_APPLICABLE",
        run_kind=package.RunKind.SYNTHETIC,
        candidate_id=authority.candidate.candidate_id,
        producer_git_commit=authority.expected_producer_commit,
        p53_attestation_sha256=artifacts.manifest.summary.p53_attestation.sha256,
        benchmark_attestation_sha256=(authority.expected_benchmark_attestation_sha256),
        synthetic_semantic_scores_exposed=False,
        observation_count=50,
        package=artifacts.validation,
        _validation_token=package._VALIDATION_TOKEN,  # noqa: SLF001
    )
    provenance = object()
    monkeypatch.setattr(
        package,
        "_capture_publication_provenance",
        lambda _fields, _authority: provenance,
    )
    monkeypatch.setattr(
        package,
        "_run_publication_child",
        lambda _mode, _descriptor, _authority: record.canonical_bytes(),
    )
    return artifacts, authority, record


def test_successful_publisher_atomically_writes_exact_canonical_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, record = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    destination = tmp_path / "published"

    published = package.publish_successful_candidate_package(
        destination,
        artifacts,
        authority,
    )

    assert published == package.PublishedCandidatePackage(destination, record)
    assert not (tmp_path / ".published.staging").exists()
    files = tuple(
        sorted(
            str(path.relative_to(destination))
            for path in destination.rglob("*")
            if path.is_file()
        )
    )
    assert files == tuple(sorted(path for path, _ in artifacts.files))
    assert all(
        path.stat().st_mode & 0o777 == 0o600
        for path in destination.rglob("*")
        if path.is_file()
    )
    assert all(
        path.stat().st_mode & 0o777 == 0o700
        for path in (
            destination,
            *tuple(path for path in destination.rglob("*") if path.is_dir()),
        )
    )


@pytest.mark.parametrize("occupied", ["published", ".published.staging"])
def test_successful_publisher_preserves_preexisting_paths_before_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
    occupied: str,
) -> None:
    artifacts, authority, _ = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    existing = tmp_path / occupied
    existing.mkdir()
    marker = existing / "keep"
    marker.write_text("owned elsewhere")

    def unexpected_artifact_build(
        _artifacts: package.SuccessfulPackageArtifacts,
    ) -> tuple[tuple[str, bytes], ...]:
        raise AssertionError("artifacts built before collision preflight")

    monkeypatch.setattr(
        package,
        "_successful_artifact_bytes",
        unexpected_artifact_build,
    )

    with pytest.raises(FileExistsError):
        package.publish_successful_candidate_package(
            tmp_path / "published",
            artifacts,
            authority,
        )

    assert marker.read_text() == "owned elsewhere"


def test_successful_publisher_losing_rename_race_preserves_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, _ = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    original = package._rename_noreplace  # noqa: SLF001

    def lose_race(
        parent_descriptor: int,
        staging_name: str,
        destination_name: str,
    ) -> None:
        if destination_name == "published":
            os.mkdir(destination_name, dir_fd=parent_descriptor)
        original(parent_descriptor, staging_name, destination_name)

    monkeypatch.setattr(package, "_rename_noreplace", lose_race)
    destination = tmp_path / "published"
    with pytest.raises(FileExistsError):
        package.publish_successful_candidate_package(
            destination,
            artifacts,
            authority,
        )

    assert destination.is_dir()
    assert tuple(destination.iterdir()) == ()
    assert not (tmp_path / ".published.staging").exists()


def test_successful_publisher_preserves_final_on_postrename_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, record = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    renamed = False
    original_rename = package._rename_noreplace  # noqa: SLF001
    original_fsync = package.os.fsync

    def tracked_rename(
        parent_descriptor: int,
        staging_name: str,
        destination_name: str,
    ) -> None:
        nonlocal renamed
        original_rename(parent_descriptor, staging_name, destination_name)
        renamed = True

    def failing_fsync(descriptor: int) -> None:
        if renamed:
            raise OSError("parent fsync failed")
        original_fsync(descriptor)

    monkeypatch.setattr(package, "_rename_noreplace", tracked_rename)
    monkeypatch.setattr(package.os, "fsync", failing_fsync)
    destination = tmp_path / "published"

    with pytest.raises(package.PublicationDurabilityUncertain) as raised:
        package.publish_successful_candidate_package(
            destination,
            artifacts,
            authority,
        )

    assert raised.value.final_path == destination
    assert raised.value.validation == record
    assert destination.is_dir()
    assert not (tmp_path / ".published.staging").exists()


def test_successful_publisher_fsyncs_files_then_directories_postorder_then_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, _ = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    calls: list[str] = []
    original_fsync = package.os.fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(os.readlink(f"/proc/self/fd/{descriptor}"))
        original_fsync(descriptor)

    monkeypatch.setattr(package.os, "fsync", record_fsync)
    package.publish_successful_candidate_package(
        tmp_path / "published",
        artifacts,
        authority,
    )

    staging = tmp_path / ".published.staging"
    artifact_paths = [str(staging / path) for path, _ in artifacts.files]
    directory_paths = sorted(
        {
            str(parent)
            for path, _ in artifacts.files
            for parent in Path(path).parents
            if parent != Path(".")
        },
        key=lambda value: (len(Path(value).parts), value),
    )
    expected = (
        artifact_paths
        + [str(staging / path) for path in reversed(directory_paths)]
        + [str(staging), str(tmp_path)]
    )
    assert calls == expected


def test_successful_publisher_rejects_postchild_tree_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, record = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )

    def mutate_after_validation(
        _mode: str,
        descriptor: int,
        _authority: CandidateValidationAuthority,
    ) -> bytes:
        manifest = Path(f"/proc/self/fd/{descriptor}/manifest.json")
        manifest.write_bytes(manifest.read_bytes() + b" ")
        return record.canonical_bytes()

    monkeypatch.setattr(package, "_run_publication_child", mutate_after_validation)
    destination = tmp_path / "published"
    with pytest.raises(ValueError, match="changed"):
        package.publish_successful_candidate_package(
            destination,
            artifacts,
            authority,
        )

    assert not destination.exists()
    assert not (tmp_path / ".published.staging").exists()


def test_successful_publisher_rejects_parent_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, record = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    parent = tmp_path / "ancestor"
    parent.mkdir()
    moved = tmp_path / "ancestor-moved"

    def swap_parent(
        _mode: str,
        _descriptor: int,
        _authority: CandidateValidationAuthority,
    ) -> bytes:
        parent.rename(moved)
        parent.mkdir()
        return record.canonical_bytes()

    monkeypatch.setattr(package, "_run_publication_child", swap_parent)
    try:
        with pytest.raises(ValueError, match="ancestor binding"):
            package.publish_successful_candidate_package(
                parent / "published",
                artifacts,
                authority,
            )
        assert tuple(parent.iterdir()) == ()
    finally:
        parent.rmdir()
        moved.rename(parent)


def test_successful_publisher_rejects_child_p53_hash_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, record = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    raw = cast(Dict[str, object], json.loads(record.canonical_bytes()))
    raw["p53_attestation_sha256"] = "2" * 64
    monkeypatch.setattr(
        package,
        "_run_publication_child",
        lambda _mode, _descriptor, _authority: canonical_json_bytes(raw),
    )
    with pytest.raises(ValueError, match="differs"):
        package.publish_successful_candidate_package(
            tmp_path / "published",
            artifacts,
            authority,
        )


def test_successful_publisher_rejects_postchild_provenance_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, record = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    captures = iter((object(), object()))
    monkeypatch.setattr(
        package,
        "_capture_publication_provenance",
        lambda _fields, _authority: next(captures),
    )
    monkeypatch.setattr(
        package,
        "_run_publication_child",
        lambda _mode, _descriptor, _authority: record.canonical_bytes(),
    )
    with pytest.raises(ValueError, match="provenance"):
        package.publish_successful_candidate_package(
            tmp_path / "published",
            artifacts,
            authority,
        )


def test_publication_provenance_recaptures_runtime_p53_and_raw_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    bound, authority, p53 = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    changed_p53 = replace(p53, environment_sha256="2" * 64)
    p53_values = iter((p53, changed_p53))
    events: list[str] = []
    monkeypatch.setattr(
        BenchmarkEnvironmentAttestation,
        "require_current_process",
        lambda _self: events.append("runtime"),
    )
    monkeypatch.setattr(
        package,
        "_require_runtime_source_bindings",
        lambda _authority: events.append("sources"),
    )
    monkeypatch.setattr(
        package,
        "run_p53_validation_subprocess",
        lambda _launch: next(p53_values),
    )
    monkeypatch.setattr(
        package,
        "_trusted_cohort_from_raw_index",
        lambda _root, **_kwargs: (
            events.append("raw") or TrustedCohort(_package_identities())
        ),
    )
    monkeypatch.setattr(
        package,
        "_inspect_git_repository",
        lambda _root: package.GitRepositoryState(_COMMIT, True),
    )
    source_records = {
        cast(str, value["path"]): FileRecord(1, cast(str, value["sha256"]))
        for value in cast(
            Mapping[str, Mapping[str, object]],
            bound.manifest.fields["source_hashes"],
        ).values()
    }
    monkeypatch.setattr(
        package,
        "_trusted_file_record",
        lambda _root, path: source_records[path],
    )
    monkeypatch.setattr(
        package,
        "_capture_environment_lock",
        lambda _authority: authority.environment_lock.file,
    )

    before = package._capture_publication_provenance(  # noqa: SLF001
        bound.manifest.fields,
        authority,
    )
    after = package._capture_publication_provenance(  # noqa: SLF001
        bound.manifest.fields,
        authority,
    )

    assert before.p53 == p53
    assert after.p53 == changed_p53
    assert before != after
    assert events == ["runtime", "sources", "raw"] * 2


def test_successful_publisher_cleans_owned_staging_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, _ = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    original = package._write_publication_file  # noqa: SLF001
    writes = 0

    def fail_second_write(descriptor: int, name: str, data: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected write failure")
        original(descriptor, name, data)

    monkeypatch.setattr(package, "_write_publication_file", fail_second_write)
    with pytest.raises(OSError, match="injected"):
        package.publish_successful_candidate_package(
            tmp_path / "published",
            artifacts,
            authority,
        )

    assert not (tmp_path / "published").exists()
    assert not (tmp_path / ".published.staging").exists()


def test_successful_publisher_preserves_primary_and_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, _ = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )

    def fail_write(_descriptor: int, _name: str, _data: bytes) -> None:
        raise OSError("primary failure")

    def fail_cleanup(*_args: object) -> None:
        raise OSError("cleanup failure")

    monkeypatch.setattr(
        package,
        "_write_publication_file",
        fail_write,
    )
    monkeypatch.setattr(
        package,
        "_remove_owned_staging",
        fail_cleanup,
    )

    with pytest.raises(package.PublicationCleanupError) as raised:
        package.publish_successful_candidate_package(
            tmp_path / "published",
            artifacts,
            authority,
        )

    assert str(raised.value.primary) == "primary failure"
    assert str(raised.value.cleanup) == "cleanup failure"


def test_publication_file_write_retries_eintr_and_partial_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    original_write = package.os.write
    calls = 0

    def interrupted_partial_write(file_descriptor: int, data: memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        return original_write(file_descriptor, data[: max(1, len(data) // 2)])

    monkeypatch.setattr(package.os, "write", interrupted_partial_write)
    try:
        package._write_publication_file(  # noqa: SLF001
            descriptor,
            "artifact",
            b"partial-write-payload",
        )
    finally:
        os.close(descriptor)
    assert (tmp_path / "artifact").read_bytes() == b"partial-write-payload"
    assert calls > 2


def test_publication_file_close_failure_preserves_write_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    original_close = package.os.close
    artifact_descriptor: int | None = None

    def fail_write(descriptor: int, _data: memoryview) -> int:
        nonlocal artifact_descriptor
        artifact_descriptor = descriptor
        raise OSError("write primary")

    def fail_artifact_close(descriptor: int) -> None:
        if descriptor == artifact_descriptor:
            raise OSError("artifact close")
        original_close(descriptor)

    monkeypatch.setattr(package.os, "write", fail_write)
    monkeypatch.setattr(package.os, "close", fail_artifact_close)
    try:
        with pytest.raises(package.PublicationCleanupError) as raised:
            package._write_publication_file(  # noqa: SLF001
                parent_descriptor,
                "artifact",
                b"payload",
            )
        assert str(raised.value.primary) == "write primary"
        assert isinstance(raised.value.cleanup, package._DescriptorCloseError)  # noqa: SLF001
        assert len(raised.value.cleanup.failures) == 1
    finally:
        if artifact_descriptor is not None:
            original_close(artifact_descriptor)
        original_close(parent_descriptor)


def test_publication_directory_closes_are_exhaustive_and_preserve_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, _ = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )

    def fail_artifact_write(
        _descriptor: int,
        _name: str,
        _data: bytes,
    ) -> None:
        raise OSError("artifact primary")

    monkeypatch.setattr(
        package,
        "_write_publication_file",
        fail_artifact_write,
    )
    original_close = package.os.close
    attempted: list[int] = []
    failed: list[int] = []
    expected_directory_count = len({
        str(parent)
        for path, _ in artifacts.files
        for parent in Path(path).parents
        if parent != Path(".")
    })

    def fail_first_two_directory_closes(descriptor: int) -> None:
        if len(attempted) < expected_directory_count:
            try:
                target = os.readlink(f"/proc/self/fd/{descriptor}")
            except OSError:
                target = ""
            if "/.published.staging/predictions" in target:
                attempted.append(descriptor)
                if len(failed) < 2:
                    failed.append(descriptor)
                    raise OSError(f"directory close {len(failed)}")
        original_close(descriptor)

    monkeypatch.setattr(package.os, "close", fail_first_two_directory_closes)
    try:
        with pytest.raises(package.PublicationCleanupError) as raised:
            package.publish_successful_candidate_package(
                tmp_path / "published",
                artifacts,
                authority,
            )
        assert str(raised.value.primary) == "artifact primary"
        assert isinstance(raised.value.cleanup, package._DescriptorCloseError)  # noqa: SLF001
        assert len(raised.value.cleanup.failures) == 2
        assert len(attempted) == expected_directory_count
    finally:
        for descriptor in failed:
            original_close(descriptor)


def test_cleanup_isolates_and_preserves_raced_staging_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, _ = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    original_file_write = package._write_publication_file  # noqa: SLF001

    def fail_write(_descriptor: int, _name: str, _data: bytes) -> None:
        raise OSError("primary")

    monkeypatch.setattr(
        package,
        "_write_publication_file",
        fail_write,
    )
    original = package._rename_noreplace  # noqa: SLF001
    moved = tmp_path / "owned-moved"

    def swap_before_isolation(
        parent_descriptor: int,
        staging_name: str,
        destination_name: str,
    ) -> None:
        if destination_name.startswith(".publication-cleanup-"):
            os.rename(
                staging_name,
                moved.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.mkdir(staging_name, dir_fd=parent_descriptor)
            replacement = os.open(
                staging_name,
                os.O_RDONLY | os.O_DIRECTORY,
                dir_fd=parent_descriptor,
            )
            try:
                original_file_write(
                    replacement,
                    "keep",
                    b"replacement",
                )
            finally:
                os.close(replacement)
        original(parent_descriptor, staging_name, destination_name)

    monkeypatch.setattr(package, "_rename_noreplace", swap_before_isolation)
    with pytest.raises(package.PublicationCleanupError):
        package.publish_successful_candidate_package(
            tmp_path / "published",
            artifacts,
            authority,
        )
    replacement = tmp_path / ".published.staging"
    assert (replacement / "keep").read_bytes() == b"replacement"
    assert moved.is_dir()


def test_cleanup_parent_fsync_failure_composes_with_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, _ = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )

    def fail_write(_descriptor: int, _name: str, _data: bytes) -> None:
        raise OSError("primary")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("cleanup fsync")

    monkeypatch.setattr(package, "_write_publication_file", fail_write)
    monkeypatch.setattr(package.os, "fsync", fail_fsync)
    with pytest.raises(package.PublicationCleanupError) as raised:
        package.publish_successful_candidate_package(
            tmp_path / "published",
            artifacts,
            authority,
        )
    assert str(raised.value.primary) == "primary"
    assert str(raised.value.cleanup) == "cleanup fsync"
    assert not (tmp_path / ".published.staging").exists()


def test_publication_collects_all_final_descriptor_close_failures_with_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    artifacts, authority, _ = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    staging_descriptor: int | None = None

    def malformed_child(
        _mode: str,
        descriptor: int,
        _authority: CandidateValidationAuthority,
    ) -> bytes:
        nonlocal staging_descriptor
        staging_descriptor = descriptor
        return b"{}"

    monkeypatch.setattr(package, "_run_publication_child", malformed_child)
    original_close = package.os.close
    final_closing = False
    failed: list[int] = []
    attempted: list[int] = []

    def failing_close(descriptor: int) -> None:
        nonlocal final_closing
        if descriptor == staging_descriptor:
            final_closing = True
        if final_closing:
            attempted.append(descriptor)
            if len(failed) < 2:
                failed.append(descriptor)
                raise OSError(f"injected close failure {len(failed)}")
        original_close(descriptor)

    monkeypatch.setattr(package.os, "close", failing_close)
    try:
        with pytest.raises(package.PublicationCleanupError) as raised:
            package.publish_successful_candidate_package(
                tmp_path / "published",
                artifacts,
                authority,
            )
        assert isinstance(raised.value.primary, ValueError)
        assert isinstance(raised.value.cleanup, package._DescriptorCloseError)  # noqa: SLF001
        assert len(raised.value.cleanup.failures) == 2
        assert len(attempted) > len(failed)
    finally:
        for descriptor in failed:
            original_close(descriptor)


def test_publication_name_rejects_staging_name_over_name_max(
    tmp_path: Path,
) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        name_max = os.fpathconf(descriptor, "PC_NAME_MAX")
        destination = tmp_path / ("a" * (name_max - len("..staging") + 1))
        with pytest.raises(ValueError, match="NAME_MAX"):
            package._publication_names(destination, descriptor)  # noqa: SLF001
    finally:
        os.close(descriptor)


def test_publication_authority_canonical_json_codec_round_trips(
    tmp_path: Path,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    _, authority, _ = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    encoded = canonical_json_bytes(
        package._publication_authority_json(authority)  # noqa: SLF001
    )
    decoded = package._publication_authority_from_json(  # noqa: SLF001
        json.loads(encoded)
    )
    assert decoded == authority


@pytest.mark.parametrize(
    "loader_path",
    [
        None,
        "",
        "/hostile/loader",
        "/proc/self/fd/",
        "/proc/self/fd/-1",
        "/proc/self/fd/01",
        "/proc/self/fd/1:/hostile/loader",
        "/proc/123/fd/1",
    ],
)
def test_publication_child_loader_rebinding_rejects_malformed_environment(
    loader_path: Optional[str],
) -> None:
    descriptor = os.open(
        Path(package.__file__).parent,
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        environment = dict(os.environ)
        if loader_path is None:
            environment.pop("LD_LIBRARY_PATH", None)
        else:
            environment["LD_LIBRARY_PATH"] = loader_path
        result = subprocess.run(
            (
                sys.executable,
                "-c",
                "import os,stat\n"
                + package._publication_loader_bootstrap(  # noqa: SLF001
                    descriptor
                ),
            ),
            capture_output=True,
            check=False,
            env=environment,
            pass_fds=(descriptor,),
            timeout=30,
        )
    finally:
        os.close(descriptor)

    assert result.returncode != 0
    assert b"publication loader environment is invalid" in result.stderr


def test_publication_child_loader_rebinding_rejects_unusable_descriptor(
    tmp_path: Path,
) -> None:
    ordinary_file = tmp_path / "loader"
    ordinary_file.touch()
    descriptor = os.open(ordinary_file, os.O_RDONLY)
    try:
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = f"/proc/self/fd/{descriptor}"
        result = subprocess.run(
            (
                sys.executable,
                "-c",
                "import os,stat\n"
                + package._publication_loader_bootstrap(  # noqa: SLF001
                    descriptor
                ),
            ),
            capture_output=True,
            check=False,
            env=environment,
            pass_fds=(descriptor,),
            timeout=30,
        )
    finally:
        os.close(descriptor)

    assert result.returncode != 0
    assert b"publication loader descriptor is not a directory" in result.stderr


def test_publication_child_rebinds_loader_authority_for_nested_subprocess(
    tmp_path: Path,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    _, authority, _ = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    runtime_loader = authority.runtime_loader
    assert runtime_loader is not None
    (
        actual_loader,
        interpreter_descriptor,
        loader_descriptor,
    ) = package._open_runtime_loader(  # noqa: SLF001
        Path(authority.benchmark_attestation.python_executable)
    )
    assert actual_loader == runtime_loader
    loader_metadata = os.fstat(loader_descriptor)
    nested_script = (
        "import os,sys\n"
        "loader_path=os.environ['LD_LIBRARY_PATH']\n"
        f"expected=f'/proc/{{os.getppid()}}/fd/{loader_descriptor}'\n"
        "assert loader_path == expected, (loader_path,expected)\n"
        "metadata=os.stat(loader_path)\n"
        f"assert (metadata.st_dev,metadata.st_ino)=="
        f"({loader_metadata.st_dev},{loader_metadata.st_ino})\n"
        "sys.stdout.write(loader_path)\n"
    )
    child_script = (
        "import contextlib,os,stat,subprocess,sys\n"
        + package._publication_loader_bootstrap(  # noqa: SLF001
            loader_descriptor
        )
        + "with open(os.devnull,'w') as import_errors:\n"
        "    with contextlib.redirect_stderr(import_errors):\n"
        "        import prior.analyze.d2026_07_29."
        "rgbd_segmenter_benchmark_package\n"
        "os.environ['LD_LIBRARY_PATH']=_bound_loader\n"
        "result=subprocess.run("
        f"({authority.benchmark_attestation.python_executable!r},'-c',"
        f"{nested_script!r}),"
        "check=True,capture_output=True,env=dict(os.environ),timeout=30)\n"
        "sys.stdout.buffer.write(result.stdout)\n"
    )
    try:
        result = package._run_bounded_publication_process(  # noqa: SLF001
            (
                f"/proc/self/fd/{interpreter_descriptor}",
                "-c",
                child_script,
            ),
            request=b"",
            environment={
                "HOME": "/nonexistent",
                "LANG": "C",
                "LC_ALL": "C",
                "LD_LIBRARY_PATH": f"/proc/self/fd/{loader_descriptor}",
                "PATH": "/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(authority.benchmark_repository_root),
                "PYTHONWARNINGS": "ignore",
            },
            timeout_seconds=60.0,
            working_directory=authority.benchmark_repository_root,
            inherited_descriptors=(
                interpreter_descriptor,
                loader_descriptor,
            ),
        )
    finally:
        package._close_descriptors(  # noqa: SLF001
            (interpreter_descriptor, loader_descriptor)
        )

    assert result.returncode == 0
    assert result.stderr == b""
    assert result.stdout.startswith(b"/proc/")
    assert result.stdout.endswith(f"/fd/{loader_descriptor}".encode("ascii"))


def test_fresh_publication_child_uses_canonical_request_and_minimal_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    _, authority, _ = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    captured: dict[str, object] = {}

    def run(
        command: tuple[str, ...],
        *,
        request: bytes,
        environment: Mapping[str, str],
        timeout_seconds: float,
        working_directory: Path,
        inherited_descriptors: tuple[int, ...],
    ) -> subprocess.CompletedProcess[bytes]:
        captured["command"] = command
        captured["input"] = request
        captured["env"] = environment
        captured["timeout"] = timeout_seconds
        captured["working_directory"] = working_directory
        captured["inherited_descriptors"] = inherited_descriptors
        return subprocess.CompletedProcess(command, 0, b"validated", b"")

    monkeypatch.setattr(package, "_run_bounded_publication_process", run)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/hostile/loader")
    monkeypatch.setenv("LD_PRELOAD", "/hostile/preload.so")
    monkeypatch.setenv("PYTHONHOME", "/hostile/python")
    monkeypatch.setenv("PYTHONSTARTUP", "/hostile/startup.py")
    try:
        assert (
            package._run_publication_child(  # noqa: SLF001
                "successful",
                descriptor,
                authority,
            )
            == b"validated"
        )
    finally:
        os.close(descriptor)

    command = cast(Tuple[str, ...], captured["command"])
    request = cast(bytes, captured["input"])
    environment = cast(Dict[str, str], captured["env"])
    assert Path(command[0]).is_absolute()
    assert command[1] == "-c"
    assert command[0].startswith("/proc/self/fd/")
    assert "pickle" not in command[2]
    assert canonical_json_bytes(json.loads(request)) == request
    assert json.loads(request)["root"] == str(staging)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONPATH"] == str(authority.benchmark_repository_root)
    inherited_descriptors = cast(
        Tuple[int, ...],
        captured["inherited_descriptors"],
    )
    assert len(inherited_descriptors) == 2
    assert command[0] == f"/proc/self/fd/{inherited_descriptors[0]}"
    assert environment["LD_LIBRARY_PATH"] == (
        f"/proc/self/fd/{inherited_descriptors[1]}"
    )
    assert environment["LD_LIBRARY_PATH"] != "/hostile/loader"
    assert not {
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONSTARTUP",
    } & set(environment)
    assert captured["timeout"] == 3600.0
    assert captured["working_directory"] == authority.benchmark_repository_root


def test_runtime_loader_supports_real_fresh_child_from_poisoned_parent_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    _, authority, _ = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    runtime_loader = authority.runtime_loader
    assert runtime_loader is not None
    (
        actual_loader,
        interpreter_descriptor,
        loader_descriptor,
    ) = package._open_runtime_loader(  # noqa: SLF001
        Path(authority.benchmark_attestation.python_executable)
    )
    assert actual_loader == runtime_loader
    request = canonical_json_bytes(
        package._publication_authority_json(authority)  # noqa: SLF001
    )
    script = (
        "import contextlib,json,os,sys\n"
        "with open(os.devnull,'w') as import_errors:\n"
        "    with contextlib.redirect_stderr(import_errors):\n"
        "        from prior.analyze.d2026_07_29."
        "rgbd_segmenter_benchmark_package import _publication_authority_from_json\n"
        "authority=_publication_authority_from_json("
        "json.loads(sys.stdin.buffer.read()))\n"
        "sys.stdout.write(authority.runtime_loader.library.sha256)\n"
    )
    poison = tmp_path / "poison"
    (poison / "prior").mkdir(parents=True)
    (poison / "prior" / "__init__.py").write_text(
        "raise RuntimeError('hostile cwd prior imported')\n"
    )
    monkeypatch.chdir(poison)
    try:
        result = package._run_bounded_publication_process(  # noqa: SLF001
            (
                f"/proc/self/fd/{interpreter_descriptor}",
                "-c",
                script,
            ),
            request=request,
            environment={
                "HOME": "/nonexistent",
                "LANG": "C",
                "LC_ALL": "C",
                "LD_LIBRARY_PATH": f"/proc/self/fd/{loader_descriptor}",
                "PATH": "/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(authority.benchmark_repository_root),
                "PYTHONWARNINGS": "ignore",
            },
            timeout_seconds=60.0,
            working_directory=authority.benchmark_repository_root,
            inherited_descriptors=(
                interpreter_descriptor,
                loader_descriptor,
            ),
        )
    finally:
        package._close_descriptors(  # noqa: SLF001
            (interpreter_descriptor, loader_descriptor)
        )

    assert result.returncode == 0
    assert result.stderr == b""
    assert result.stdout.endswith(runtime_loader.library.sha256.encode("ascii"))


@pytest.mark.parametrize("directory_name", [".published.staging", "published"])
def test_public_fresh_validator_rebinds_exact_local_record_and_ignores_hostile_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
    directory_name: str,
) -> None:
    original_run_child = package._run_publication_child  # noqa: SLF001
    artifacts, authority, expected = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    monkeypatch.setattr(package, "_run_publication_child", original_run_child)
    root = tmp_path / directory_name
    root.mkdir(mode=0o700)
    for relative, data in artifacts.files:
        path = root / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o600)
    for directory in root.rglob("*"):
        if directory.is_dir():
            directory.chmod(0o700)
    captured_environment: Mapping[str, str] | None = None

    def run(
        command: tuple[str, ...],
        *,
        request: bytes,
        environment: Mapping[str, str],
        timeout_seconds: float,
        working_directory: Path,
        inherited_descriptors: tuple[int, ...],
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal captured_environment
        del request, timeout_seconds, working_directory, inherited_descriptors
        captured_environment = environment
        return subprocess.CompletedProcess(
            command,
            0,
            expected.canonical_bytes(),
            b"",
        )

    monkeypatch.setattr(package, "_run_bounded_publication_process", run)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/hostile/loader")
    monkeypatch.setenv("LD_PRELOAD", "/hostile/preload.so")

    actual = package.validate_candidate_package_fresh_process(root, authority)

    assert actual == expected
    assert captured_environment is not None
    assert captured_environment["LD_LIBRARY_PATH"].startswith("/proc/self/fd/")
    assert captured_environment["LD_LIBRARY_PATH"] != "/hostile/loader"
    assert "LD_PRELOAD" not in captured_environment


def test_runtime_loader_rejects_symlinked_interpreter(
    tmp_path: Path,
) -> None:
    (tmp_path / "runtime" / "bin").mkdir(parents=True)
    (tmp_path / "runtime" / "lib").mkdir()
    interpreter = tmp_path / "runtime" / "bin" / "python"
    interpreter.symlink_to(Path(sys.executable).resolve(strict=True))

    with pytest.raises(ValueError, match="attestation failed"):
        package._open_runtime_loader(interpreter)  # noqa: SLF001


def test_runtime_loader_rejects_world_writable_path(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "runtime"
    (prefix / "bin").mkdir(parents=True)
    loader = prefix / "lib"
    loader.mkdir()
    interpreter = prefix / "bin" / "python"
    interpreter.write_bytes(Path(sys.executable).resolve(strict=True).read_bytes())
    interpreter.chmod(0o755)
    loader.chmod(0o777)

    with pytest.raises(ValueError, match="world-writable"):
        package._open_runtime_loader(interpreter)  # noqa: SLF001


def test_fresh_publication_child_rejects_runtime_loader_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    _, authority, _ = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    original_open = package._open_runtime_loader  # noqa: SLF001

    def drifted(
        interpreter: Path,
    ) -> tuple[package.RuntimeLoaderAttestation, int, int]:
        attestation, interpreter_descriptor, loader_descriptor = original_open(
            interpreter
        )
        return (
            replace(attestation, inode=attestation.inode + 1),
            interpreter_descriptor,
            loader_descriptor,
        )

    monkeypatch.setattr(package, "_open_runtime_loader", drifted)
    try:
        with pytest.raises(ValueError, match="differs from authority"):
            package._run_publication_child(  # noqa: SLF001
                "successful",
                staging_descriptor,
                authority,
            )
    finally:
        os.close(staging_descriptor)


def test_fresh_publication_child_executes_held_interpreter_and_rejects_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    _, authority, _ = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    runtime_loader = authority.runtime_loader
    assert runtime_loader is not None
    benchmark_root = authority.benchmark_repository_root
    with tempfile.TemporaryDirectory(
        prefix=".runtime-swap-",
        dir=benchmark_root,
    ) as temporary:
        prefix = Path(temporary)
        (prefix / "bin").mkdir()
        (prefix / "lib").mkdir()
        interpreter = prefix / "bin" / "python"
        interpreter.write_bytes(
            Path(authority.benchmark_attestation.python_executable).read_bytes()
        )
        interpreter.chmod(0o755)
        library_target = runtime_loader.library_target
        (prefix / "lib" / library_target).write_bytes(
            (runtime_loader.path / library_target).read_bytes()
        )
        (prefix / "lib" / library_target).chmod(0o755)
        (prefix / "lib" / "libstdc++.so.6").symlink_to(library_target)
        benchmark = replace(
            authority.benchmark_attestation,
            python_executable=str(interpreter),
        )
        fake_authority = replace(
            authority,
            benchmark_attestation=benchmark,
            expected_benchmark_attestation_sha256=benchmark.sha256,
            runtime_loader=None,
        )
        expected_loader = fake_authority.runtime_loader
        assert expected_loader is not None
        staging = tmp_path / "staging-swap"
        staging.mkdir()
        staging_descriptor = os.open(
            staging,
            os.O_RDONLY | os.O_DIRECTORY,
        )

        def swap_at_popen(
            command: tuple[str, ...],
            *,
            request: bytes,
            environment: Mapping[str, str],
            timeout_seconds: float,
            working_directory: Path,
            inherited_descriptors: tuple[int, ...],
        ) -> subprocess.CompletedProcess[bytes]:
            del request, environment, timeout_seconds
            assert working_directory == benchmark_root
            assert len(inherited_descriptors) == 2
            held_interpreter = inherited_descriptors[0]
            assert command[0] == f"/proc/self/fd/{held_interpreter}"
            assert os.fstat(held_interpreter).st_ino == (
                expected_loader.interpreter_inode
            )
            interpreter.rename(interpreter.with_name("python.attested"))
            interpreter.write_bytes(b"#!/bin/sh\nexit 99\n")
            interpreter.chmod(0o755)
            return subprocess.CompletedProcess(command, 0, b"validated", b"")

        monkeypatch.setattr(
            package,
            "_run_bounded_publication_process",
            swap_at_popen,
        )
        try:
            with pytest.raises(
                ValueError,
                match="publication runtime authority drifted",
            ):
                package._run_publication_child(  # noqa: SLF001
                    "successful",
                    staging_descriptor,
                    fake_authority,
                )
        finally:
            os.close(staging_descriptor)


@pytest.mark.parametrize(
    ("outcome", "returncode", "stdout", "stderr"),
    [
        ("timeout", 0, b"", b""),
        ("nonzero", 1, b"record", b""),
        ("signal", -9, b"record", b""),
        ("stderr", 0, b"record", b"diagnostic"),
        ("empty", 0, b"", b""),
    ],
)
def test_fresh_publication_child_rejects_process_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
    outcome: str,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
) -> None:
    _, authority, _ = _public_validation_fixture(
        tmp_path,
        accepted_science_package,
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)

    def run(
        command: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        if outcome == "timeout":
            raise ValueError("fresh-process publication validation timed out")
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(package, "_run_bounded_publication_process", run)
    try:
        with pytest.raises(ValueError, match="fresh-process"):
            package._run_publication_child(  # noqa: SLF001
                "successful",
                descriptor,
                authority,
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("descriptor", [1, 2])
def test_bounded_publication_child_terminates_on_output_cap(
    descriptor: int,
) -> None:
    script = (
        "import os\n"
        "chunk=b'x'*4096\n"
        f"fd={descriptor}\n"
        "for _ in range(1024): os.write(fd,chunk)\n"
    )
    with pytest.raises(ValueError, match="hard cap"):
        package._run_bounded_publication_process(  # noqa: SLF001
            (str(Path(sys.executable).resolve(strict=True)), "-c", script),
            request=b"",
            environment={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
            },
            timeout_seconds=10.0,
            working_directory=Path(package.__file__).resolve(strict=True).parents[3],
        )


@pytest.mark.parametrize("payload", [b"{}", b"not-json", b"{}\n", b"{}{}"])
def test_successful_publisher_rejects_malformed_or_extra_child_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
    payload: bytes,
) -> None:
    artifacts, authority, _ = _publication_fixture(
        tmp_path,
        monkeypatch,
        accepted_science_package,
    )
    monkeypatch.setattr(
        package,
        "_run_publication_child",
        lambda _mode, _descriptor, _authority: payload,
    )
    with pytest.raises(ValueError):
        package.publish_successful_candidate_package(
            tmp_path / "published",
            artifacts,
            authority,
        )


@pytest.mark.parametrize("error_number", [errno.ENOSYS, errno.EINVAL])
def test_rename_noreplace_has_no_unsupported_kernel_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    class RenameAt2:
        argtypes: object = None
        restype: object = None

        def __call__(self, *_args: object) -> int:
            ctypes.set_errno(error_number)
            return -1

    class Libc:
        renameat2 = RenameAt2()

    monkeypatch.setattr(package.ctypes, "CDLL", lambda *_args, **_kwargs: Libc())
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError) as raised:
            package._rename_noreplace(  # noqa: SLF001
                descriptor,
                "source",
                "destination",
            )
    finally:
        os.close(descriptor)
    assert raised.value.errno == error_number


def test_aborted_oom_factory_and_publisher_require_exact_real_cuda_oom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_science_package: package.AcceptedCandidatePackage,
) -> None:
    real = _real_science_accepted(accepted_science_package)
    bound, candidate_authority, p53 = _public_validation_fixture(tmp_path, real)
    assert bound.manifest.summary.real is not None
    authority = package.AbortedOomPublicationAuthority(
        candidate_authority,
        bound.manifest.summary.real.cuda_evidence,
    )
    provenance = object()
    monkeypatch.setattr(
        package,
        "run_p53_validation_subprocess",
        lambda _launch: p53,
    )
    monkeypatch.setattr(
        package,
        "_capture_publication_provenance",
        lambda _fields, _authority: provenance,
    )
    monkeypatch.setattr(
        package,
        "_trusted_file_record",
        lambda _root, _path: FileRecord(1, _HASH),
    )
    allocator_events: list[str] = []
    monkeypatch.setattr(
        package.torch.cuda,
        "memory_allocated",
        lambda: allocator_events.append("baseline-allocated") or 1,
    )
    monkeypatch.setattr(
        package.torch.cuda,
        "memory_reserved",
        lambda: allocator_events.append("baseline-reserved") or 2,
    )
    monkeypatch.setattr(
        package.torch.cuda,
        "reset_peak_memory_stats",
        lambda: allocator_events.append("reset"),
    )
    monkeypatch.setattr(
        package.torch.cuda,
        "max_memory_allocated",
        lambda: allocator_events.append("peak-allocated") or 3,
    )
    monkeypatch.setattr(
        package.torch.cuda,
        "max_memory_reserved",
        lambda: allocator_events.append("peak-reserved") or 4,
    )

    def committed_oom(*_args: object, **_kwargs: object) -> object:
        allocator_events.append("run")

        def raise_oom() -> object:
            raise torch.cuda.OutOfMemoryError("expected")

        return rgbd_segmenter_benchmark_contract._measure_stage(  # noqa: SLF001
            cast(package.TimingBackend, object()),
            package.TimingStage.INFERENCE,
            raise_oom,
            measured=False,
            device_stage=False,
        )

    monkeypatch.setattr(package, "run_timed_benchmark", committed_oom)
    aborted = package.run_timed_benchmark_for_publication(
        (),
        trusted_cohort=cast(TrustedCohort, object()),
        adapter=cast(package.SegmenterAdapter, object()),
        backend=cast(package.TimingBackend, object()),
        authority=authority,
    )
    assert isinstance(aborted, package.AbortedOomPackage)
    assert allocator_events == [
        "baseline-allocated",
        "baseline-reserved",
        "reset",
        "run",
        "peak-allocated",
        "peak-reserved",
    ]
    with pytest.raises(ValueError, match="factory"):
        replace(aborted)

    def fabricated_oom(*_args: object, **_kwargs: object) -> object:
        raise torch.cuda.OutOfMemoryError("fabricated outside contract")

    monkeypatch.setattr(package, "run_timed_benchmark", fabricated_oom)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="fabricated"):
        package.run_timed_benchmark_for_publication(
            (),
            trusted_cohort=cast(TrustedCohort, object()),
            adapter=cast(package.SegmenterAdapter, object()),
            backend=cast(package.TimingBackend, object()),
            authority=authority,
        )

    def prewrapped_oom(*_args: object, **_kwargs: object) -> object:
        def operation() -> object:
            try:
                raise torch.cuda.OutOfMemoryError("fabricated cause")
            except torch.cuda.OutOfMemoryError as cause:
                raise package.BenchmarkCudaOutOfMemory(
                    package.TimingStage.INFERENCE
                ) from cause

        return rgbd_segmenter_benchmark_contract._measure_stage(  # noqa: SLF001
            cast(package.TimingBackend, object()),
            package.TimingStage.INFERENCE,
            operation,
            measured=False,
            device_stage=False,
        )

    monkeypatch.setattr(package, "run_timed_benchmark", prewrapped_oom)
    with pytest.raises(ValueError, match="origin"):
        package.run_timed_benchmark_for_publication(
            (),
            trusted_cohort=cast(TrustedCohort, object()),
            adapter=cast(package.SegmenterAdapter, object()),
            backend=cast(package.TimingBackend, object()),
            authority=authority,
        )

    manifest = aborted.manifest
    monkeypatch.setattr(
        package,
        "_run_publication_child",
        lambda _mode, _descriptor, _authority: manifest.canonical_bytes(),
    )
    published = package.publish_aborted_oom_package(
        tmp_path,
        aborted,
        authority,
    )
    assert published.name == (
        f"{candidate_authority.candidate.candidate_id}-aborted-oom-"
        f"{candidate_authority.expected_producer_commit[:12]}"
    )
    assert tuple(path.name for path in published.iterdir()) == ("manifest.json",)
    assert (published / "manifest.json").read_bytes() == manifest.canonical_bytes()
