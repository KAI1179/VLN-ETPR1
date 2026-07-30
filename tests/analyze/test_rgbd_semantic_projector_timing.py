from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, cast

import numpy as np
import pytest

from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    CudaDeviceEvidence,
    OfficialCudaEvidence,
    ObservationResult,
    ObservationStatus,
    Prediction,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    AcceptedCandidatePackage,
)
from prior.analyze.d2026_07_30 import rgbd_semantic_projector_timing as timing
from prior.analyze.d2026_07_30.rgbd_semantic_projector import EquivalenceReport


def _prediction(value: int = -1) -> Prediction:
    labels = np.full((12, 256, 256), value, dtype="<i2")
    return Prediction(source_labels=labels.copy(), mapped_labels=labels.copy())


def _accepted(
    predictions: tuple[Prediction, ...],
) -> AcceptedCandidatePackage:
    observations = tuple(
        SimpleNamespace(
            ordinal=ordinal,
            status=ObservationStatus.PASS,
            failure_code=None,
        )
        for ordinal in range(50)
    )
    values = tuple(
        SimpleNamespace(prediction=prediction) for prediction in predictions
    )
    return cast(
        AcceptedCandidatePackage,
        SimpleNamespace(observations=observations, predictions=values),
    )


def _results(
    predictions: tuple[Prediction, ...],
) -> tuple[ObservationResult, ...]:
    return tuple(
        ObservationResult(
            ordinal=ordinal,
            status=ObservationStatus.PASS,
            failure_code=None,
            prediction=prediction,
        )
        for ordinal, prediction in enumerate(predictions)
    )


def test_frozen_output_identity_is_exact_and_ordered() -> None:
    predictions = tuple(_prediction() for _ in range(50))
    accepted = _accepted(predictions)
    digest = timing._require_frozen_output_identity(
        _results(predictions),
        accepted,
    )
    assert len(digest) == 64

    changed = list(predictions)
    changed_labels = changed[23].mapped_labels.copy()
    changed_labels[0, 0, 0] = 2
    changed[23] = replace(changed[23], mapped_labels=changed_labels)
    with pytest.raises(ValueError, match="frozen observation 23"):
        timing._require_frozen_output_identity(
            _results(tuple(changed)),
            accepted,
        )


def test_publish_report_is_no_overwrite(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "data/rgbd_segmenter_benchmark"
    benchmark_root.mkdir(parents=True)
    data = b'{"schema_version":1}\n'

    timings = b'{"sequence_index":20}\n'
    report = timing._publish_report(tmp_path, data, timings)
    assert report.read_bytes() == data
    assert report.with_name("timings.jsonl").read_bytes() == timings
    assert report.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        timing._publish_report(tmp_path, data, timings)


def test_destination_preflight_rejects_existing_experiment(tmp_path: Path) -> None:
    destination = (
        tmp_path
        / "data/rgbd_segmenter_benchmark/experiments"
        / timing._EXPERIMENT_ID
    )
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="experiment directory"):
        timing._require_destination_absent(tmp_path)


def test_timing_report_is_canonical_json() -> None:
    summary = timing.DistributionSummary(1.0, 0.9, 1.2)
    snapshot = CudaDeviceEvidence(
        gpu_name="NVIDIA GeForce RTX 3090",
        gpu_uuid="GPU-test",
        driver_version="test",
        clock_policy="test",
        persistence_mode="Enabled",
        power_limit_watts=300.0,
        temperature_celsius=50.0,
        compute_pids=(os.getpid(),),
    )
    evidence = OfficialCudaEvidence(
        before=snapshot,
        after=snapshot,
        current_pid=os.getpid(),
        pytorch_cuda_alloc_conf=None,
    )
    report = timing.TimingReport(
        schema_version=1,
        experiment_id="test",
        command=("python", "-m", "test"),
        producer_git_commit="a" * 40,
        source_candidate_manifest_sha256="b" * 64,
        environment_sha256="c" * 64,
        benchmark_attestation_sha256="d" * 64,
        p53_attestation_sha256="2" * 64,
        projector_sha256="e" * 64,
        equivalence_pair_tree_sha256="1" * 64,
        segmenter_label_identity_tree_sha256="f" * 64,
        timing_sample_tree_sha256="0" * 64,
        timing_file_byte_length=10,
        timing_file_sha256="3" * 64,
        warmup_count=20,
        measured_pass_count=2,
        measured_sample_count=100,
        p50_seconds=0.1,
        p95_seconds=0.2,
        total_seconds=10.0,
        views_per_second=120.0,
        components=(("projection", summary),),
        cuda_evidence=evidence,
    )
    first = report.canonical_bytes()
    assert report.canonical_bytes() == first
    assert json.loads(first)["components"]["projection"]["mean_seconds"] == 1.0


def test_frozen_equivalence_rejects_any_gate_drift() -> None:
    accepted = EquivalenceReport(
        schema_version=1,
        mapped_grid_count=50,
        oracle_grid_count=50,
        comparison_count=100,
        pair_tree_sha256=timing._EQUIVALENCE_PAIR_TREE_SHA256,
    )
    timing._require_frozen_equivalence(accepted)
    with pytest.raises(ValueError, match="frozen gate"):
        timing._require_frozen_equivalence(
            replace(accepted, oracle_grid_count=49)
        )


def test_equivalence_subprocess_removes_cuda_timing_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "python"
    executable.write_bytes(b"python")
    captured: dict[str, object] = {}

    def run(*args: object, **kwargs: object) -> SimpleNamespace:
        captured["args"] = args
        captured["environment"] = kwargs["env"]
        expected = EquivalenceReport(
            schema_version=1,
            mapped_grid_count=50,
            oracle_grid_count=50,
            comparison_count=100,
            pair_tree_sha256=timing._EQUIVALENCE_PAIR_TREE_SHA256,
        )
        return SimpleNamespace(stdout=expected.canonical_bytes())

    monkeypatch.setattr(timing.subprocess, "run", run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setenv("PYTHONPATH", "test")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "test")
    timing._run_frozen_equivalence_subprocess(executable)
    environment = cast(Dict[str, str], captured["environment"])
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert "PYTHONPATH" not in environment
    assert "PYTORCH_CUDA_ALLOC_CONF" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_post_run_state_rejects_projector_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projector = tmp_path / timing._PROJECTOR_PATH
    projector.parent.mkdir(parents=True)
    projector.write_bytes(b"before")
    monkeypatch.setattr(timing, "capture_environment_sha256", lambda: "e" * 64)
    monkeypatch.setattr(timing.official, "_git_commit", lambda _root: "c" * 40)
    timing._require_post_run_state(
        tmp_path,
        commit="c" * 40,
        environment_sha256="e" * 64,
        projector_data=b"before",
    )
    projector.write_bytes(b"after")
    with pytest.raises(ValueError, match="projector changed"):
        timing._require_post_run_state(
            tmp_path,
            commit="c" * 40,
            environment_sha256="e" * 64,
            projector_data=b"before",
        )


def test_cli_has_no_experiment_overrides() -> None:
    timing.SemanticProjectorTimingArgs(underscores_to_dashes=True).parse_args(())
    with pytest.raises(SystemExit):
        timing.SemanticProjectorTimingArgs(
            underscores_to_dashes=True
        ).parse_args(("--device", "cuda:0"))
