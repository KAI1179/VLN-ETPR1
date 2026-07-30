"""Official P6.1 timing of the exact-equivalent semantic-only projector."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from tap import Tap

from prior.analyze.d2026_07_29 import rgbd_segmenter_esanet_benchmark as official
from prior.analyze.d2026_07_29 import rgbd_segmenter_benchmark_package as package
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    OfficialCudaEvidence,
    OfficialCudaTimingBackend,
    ObservationResult,
    TimedBenchmarkInput,
    TimedBenchmarkRun,
    TrustedCohort,
    capture_environment_sha256,
    iter_validated_raw_observations,
    logical_label_sha256,
    run_p53_validation_subprocess,
    run_timed_benchmark,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    AcceptedCandidatePackage,
    TimingPackageRow,
    accept_candidate_package,
    canonical_timings_bytes,
)
from prior.analyze.d2026_07_30.rgbd_semantic_projector import (
    EquivalenceReport,
    project_mapped_labels_semantic_only,
)

_EXPERIMENT_ID = "esanet-r34-nbt1d-scenenet-semantic-projector-timing-v1"
_DESTINATION = (
    Path("data/rgbd_segmenter_benchmark/experiments") / _EXPERIMENT_ID
)
_PUBLISHED_CANDIDATE = Path(
    "data/rgbd_segmenter_benchmark/esanet-r34-nbt1d-scenenet-v1"
)
_PUBLISHED_MANIFEST_SHA256 = (
    "030c27239ad88af4e8c637cfd0e18b725ff018ea71f134513332a73169c14aa2"
)
_PROJECTOR_PATH = "prior/analyze/d2026_07_30/rgbd_semantic_projector.py"
_MODULE = "prior.analyze.d2026_07_30.rgbd_semantic_projector_timing"
_EQUIVALENCE_PAIR_TREE_SHA256 = (
    "7e7be242e6d46d0eadbdac450bb9b8e277e8fb238f95be3845ef322d6d7b3118"
)
_COMPONENT_NAMES = (
    "preprocess",
    "h2d",
    "inference",
    "device_postprocess",
    "d2h",
    "projection",
)

__all__ = ("SemanticProjectorTimingArgs", "TimingReport", "main")


class SemanticProjectorTimingArgs(Tap):
    """The sealed P6.1 official timing runner has no configurable inputs."""


@dataclass(frozen=True)
class DistributionSummary:
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float


@dataclass(frozen=True)
class TimingReport:
    schema_version: int
    experiment_id: str
    command: Tuple[str, ...]
    producer_git_commit: str
    source_candidate_manifest_sha256: str
    environment_sha256: str
    benchmark_attestation_sha256: str
    p53_attestation_sha256: str
    projector_sha256: str
    equivalence_pair_tree_sha256: str
    segmenter_label_identity_tree_sha256: str
    timing_sample_tree_sha256: str
    timing_file_byte_length: int
    timing_file_sha256: str
    warmup_count: int
    measured_pass_count: int
    measured_sample_count: int
    p50_seconds: float
    p95_seconds: float
    total_seconds: float
    views_per_second: float
    components: Tuple[Tuple[str, DistributionSummary], ...]
    cuda_evidence: OfficialCudaEvidence

    def canonical_bytes(self) -> bytes:
        value = asdict(self)
        value["components"] = {
            name: asdict(summary) for name, summary in self.components
        }
        return (
            json.dumps(
                value,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")


def _repository_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[3]


def _require_launch_environment() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or re.fullmatch(r"GPU-[A-Za-z0-9-]+", visible) is None:
        raise ValueError("P6.1 timing requires one visible GPU UUID")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ValueError("P6.1 timing requires PYTHONNOUSERSITE=1")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise ValueError("P6.1 timing requires PYTHONDONTWRITEBYTECODE=1")
    if os.environ.get("PYTHONPATH") is not None:
        raise ValueError("P6.1 timing requires PYTHONPATH absent")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is not None:
        raise ValueError("P6.1 timing requires allocator override absent")
    return visible


def _require_destination_absent(root: Path) -> None:
    benchmark_root = (root / "data/rgbd_segmenter_benchmark").resolve(strict=True)
    experiments = benchmark_root / "experiments"
    if experiments.is_symlink():
        raise ValueError("experiment publication parent cannot be a symlink")
    if experiments.exists() and (
        experiments.resolve(strict=True) != experiments or not experiments.is_dir()
    ):
        raise ValueError("experiment publication parent must be a canonical directory")
    destination = experiments / _EXPERIMENT_ID
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"experiment directory already exists: {destination}")


def _timed_inputs(
    observations: Sequence[official.ValidatedRawObservation],
) -> Tuple[Tuple[TimedBenchmarkInput, ...], TrustedCohort]:
    values = tuple(
        TimedBenchmarkInput(
            ordinal=value.row.ordinal,
            observation_id=value.row.observation_id,
            scene_id=value.row.scene_id,
            segmenter_input=value.segmenter_input,
            raw_arrays=value.audit_arrays,
        )
        for value in observations
    )
    cohort = TrustedCohort(
        tuple(
            (value.ordinal, value.observation_id, value.scene_id)
            for value in values
        )
    )
    return values, cohort


def _require_frozen_output_identity(
    results: Sequence[ObservationResult],
    accepted: AcceptedCandidatePackage,
) -> str:
    values = tuple(results)
    if len(values) != 50:
        raise ValueError("P6.1 requires exactly 50 canonical outputs")
    lines: list[bytes] = []
    for result, observation, prediction in zip(
        values,
        accepted.observations,
        accepted.predictions,
    ):
        expected = prediction.prediction
        source_sha256 = logical_label_sha256(result.prediction.source_labels)
        mapped_sha256 = logical_label_sha256(result.prediction.mapped_labels)
        if (
            result.ordinal != observation.ordinal
            or result.status is not observation.status
            or result.failure_code is not observation.failure_code
            or result.prediction.source_labels.dtype
            != expected.source_labels.dtype
            or result.prediction.mapped_labels.dtype
            != expected.mapped_labels.dtype
            or result.prediction.source_labels.shape
            != expected.source_labels.shape
            or result.prediction.mapped_labels.shape
            != expected.mapped_labels.shape
            or not np.array_equal(
                result.prediction.source_labels,
                expected.source_labels,
            )
            or not np.array_equal(
                result.prediction.mapped_labels,
                expected.mapped_labels,
            )
            or source_sha256
            != logical_label_sha256(expected.source_labels)
            or mapped_sha256
            != logical_label_sha256(expected.mapped_labels)
        ):
            raise ValueError(
                f"P6.1 output differs from frozen observation {observation.ordinal}"
            )
        failure = "" if result.failure_code is None else result.failure_code.value
        lines.append(
            (
                f"{result.ordinal:02d} {result.status.value} {failure} "
                f"{source_sha256} {mapped_sha256}\n"
            ).encode("ascii")
        )
    return hashlib.sha256(b"".join(lines)).hexdigest()


def _distribution(values: Sequence[float]) -> DistributionSummary:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (100,) or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("P6.1 component timing sample is invalid")
    return DistributionSummary(
        mean_seconds=float(np.mean(array)),
        p50_seconds=float(np.quantile(array, 0.50, method="linear")),
        p95_seconds=float(np.quantile(array, 0.95, method="linear")),
    )


def _require_frozen_equivalence(report: EquivalenceReport) -> None:
    if (
        report.schema_version != 1
        or report.mapped_grid_count != 50
        or report.oracle_grid_count != 50
        or report.comparison_count != 100
        or report.pair_tree_sha256 != _EQUIVALENCE_PAIR_TREE_SHA256
    ):
        raise ValueError("P6.0 exact-equivalence evidence differs from the frozen gate")


def _run_frozen_equivalence_subprocess(python_executable: Path) -> None:
    expected = EquivalenceReport(
        schema_version=1,
        mapped_grid_count=50,
        oracle_grid_count=50,
        comparison_count=100,
        pair_tree_sha256=_EQUIVALENCE_PAIR_TREE_SHA256,
    )
    _require_frozen_equivalence(expected)
    environment = os.environ.copy()
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (
            str(python_executable.resolve(strict=True)),
            "-m",
            "prior.analyze.d2026_07_30.rgbd_semantic_projector",
        ),
        cwd=_repository_root(),
        env=environment,
        capture_output=True,
        check=True,
        timeout=300.0,
    )
    if completed.stdout != expected.canonical_bytes():
        raise ValueError("P6.0 subprocess output differs from the frozen gate")


def _require_post_run_state(
    root: Path,
    *,
    commit: str,
    environment_sha256: str,
    projector_data: bytes,
) -> None:
    if capture_environment_sha256() != environment_sha256:
        raise ValueError("benchmark environment changed during P6.1 timing")
    if official._git_commit(root) != commit:
        raise ValueError("experiment Git commit changed during P6.1 timing")
    if (root / _PROJECTOR_PATH).read_bytes() != projector_data:
        raise ValueError("semantic projector changed during P6.1 timing")


def _canonical_timing_bytes(run: TimedBenchmarkRun) -> bytes:
    return canonical_timings_bytes(
        tuple(TimingPackageRow(sample) for sample in run.samples)
    )


def _timing_tree_sha256(run: TimedBenchmarkRun) -> str:
    lines = []
    for sample in run.samples:
        value = {
            "components": asdict(sample.components),
            "end_to_end": sample.end_to_end,
            "failure_code": (
                None if sample.failure_code is None else sample.failure_code.value
            ),
            "mapped_labels_sha256": sample.mapped_labels_sha256,
            "ordinal": sample.ordinal,
            "pass_index": sample.pass_index,
            "sequence_index": sample.sequence_index,
            "source_labels_sha256": sample.source_labels_sha256,
            "status": sample.status.value,
        }
        lines.append(
            (
                json.dumps(
                    value,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        )
    return hashlib.sha256(b"".join(lines)).hexdigest()


def _build_report(
    *,
    run: TimedBenchmarkRun,
    commit: str,
    environment_sha256: str,
    benchmark_attestation_sha256: str,
    p53_attestation_sha256: str,
    projector_sha256: str,
    segmenter_label_identity_tree_sha256: str,
    timing_data: bytes,
) -> TimingReport:
    if (
        run.protocol.backend != "official-cuda"
        or not run.protocol.comparable
        or run.latency is None
        or run.official_cuda_evidence is None
        or run.latency.p50_seconds is None
        or run.latency.p95_seconds is None
        or run.latency.total_seconds is None
        or run.latency.views_per_second is None
    ):
        raise ValueError("P6.1 report requires an official comparable CUDA run")
    components: list[Tuple[str, DistributionSummary]] = []
    for name in _COMPONENT_NAMES:
        values = tuple(getattr(sample.components, name) for sample in run.samples)
        if any(value is None for value in values):
            raise ValueError(f"P6.1 run lacks complete {name} timings")
        components.append(
            (name, _distribution(tuple(float(value) for value in values)))
        )
    return TimingReport(
        schema_version=1,
        experiment_id=_EXPERIMENT_ID,
        command=(str(Path(sys.executable).resolve(strict=True)), "-m", _MODULE),
        producer_git_commit=commit,
        source_candidate_manifest_sha256=_PUBLISHED_MANIFEST_SHA256,
        environment_sha256=environment_sha256,
        benchmark_attestation_sha256=benchmark_attestation_sha256,
        p53_attestation_sha256=p53_attestation_sha256,
        projector_sha256=projector_sha256,
        equivalence_pair_tree_sha256=_EQUIVALENCE_PAIR_TREE_SHA256,
        segmenter_label_identity_tree_sha256=(
            segmenter_label_identity_tree_sha256
        ),
        timing_sample_tree_sha256=_timing_tree_sha256(run),
        timing_file_byte_length=len(timing_data),
        timing_file_sha256=hashlib.sha256(timing_data).hexdigest(),
        warmup_count=20,
        measured_pass_count=2,
        measured_sample_count=100,
        p50_seconds=run.latency.p50_seconds,
        p95_seconds=run.latency.p95_seconds,
        total_seconds=run.latency.total_seconds,
        views_per_second=run.latency.views_per_second,
        components=tuple(components),
        cuda_evidence=run.official_cuda_evidence,
    )


def _publish_report(root: Path, report_data: bytes, timing_data: bytes) -> Path:
    benchmark_root = (root / "data/rgbd_segmenter_benchmark").resolve(strict=True)
    benchmark_descriptor = os.open(
        benchmark_root,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    experiments_descriptor: Optional[int] = None
    staging_descriptor: Optional[int] = None
    staging_name = f".{_EXPERIMENT_ID}.staging"
    published = False
    try:
        try:
            os.mkdir("experiments", mode=0o755, dir_fd=benchmark_descriptor)
            os.fsync(benchmark_descriptor)
        except FileExistsError:
            pass
        experiments_descriptor = os.open(
            "experiments",
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=benchmark_descriptor,
        )
        for name in (_EXPERIMENT_ID, staging_name):
            try:
                os.stat(name, dir_fd=experiments_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise FileExistsError(f"experiment publication path exists: {name}")
        os.mkdir(staging_name, mode=0o700, dir_fd=experiments_descriptor)
        staging_descriptor = os.open(
            staging_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=experiments_descriptor,
        )
        for name, data in (
            ("report.json", report_data),
            ("timings.jsonl", timing_data),
        ):
            file_descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                0o400,
                dir_fd=staging_descriptor,
            )
            try:
                with os.fdopen(file_descriptor, "wb", closefd=False) as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
        os.fsync(staging_descriptor)
        staging_info = os.fstat(staging_descriptor)
        named_info = os.stat(
            staging_name,
            dir_fd=experiments_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(named_info.st_mode)
            or (named_info.st_dev, named_info.st_ino)
            != (staging_info.st_dev, staging_info.st_ino)
        ):
            raise RuntimeError("experiment staging directory identity changed")
        package._rename_noreplace(
            experiments_descriptor,
            staging_name,
            _EXPERIMENT_ID,
        )
        os.fsync(experiments_descriptor)
        published = True
    finally:
        if not published and staging_descriptor is not None:
            for name in ("report.json", "timings.jsonl"):
                try:
                    os.unlink(name, dir_fd=staging_descriptor)
                except FileNotFoundError:
                    pass
            if experiments_descriptor is not None:
                try:
                    named_info = os.stat(
                        staging_name,
                        dir_fd=experiments_descriptor,
                        follow_symlinks=False,
                    )
                    staging_info = os.fstat(staging_descriptor)
                    if (named_info.st_dev, named_info.st_ino) == (
                        staging_info.st_dev,
                        staging_info.st_ino,
                    ):
                        os.rmdir(staging_name, dir_fd=experiments_descriptor)
                except OSError:
                    pass
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if experiments_descriptor is not None:
            os.close(experiments_descriptor)
        os.close(benchmark_descriptor)
    return (
        benchmark_root
        / "experiments"
        / _EXPERIMENT_ID
        / "report.json"
    )


def main(argv: Optional[Sequence[str]] = None) -> TimingReport:
    SemanticProjectorTimingArgs(underscores_to_dashes=True).parse_args(argv)
    visible = _require_launch_environment()
    root = _repository_root()
    official._require_dedicated_interpreter(root)
    _require_destination_absent(root)
    paths = official._candidate_paths(root)
    official.audit_candidate_assets(paths)
    environment_sha256 = capture_environment_sha256()
    if environment_sha256 != official._COMPLETE_ENVIRONMENT_SHA256:
        raise ValueError("complete benchmark environment differs from pinned digest")
    commit = official._git_commit(root)
    projector_data = (root / _PROJECTOR_PATH).read_bytes()
    projector_sha256 = hashlib.sha256(projector_data).hexdigest()
    benchmark = official._benchmark_attestation(environment_sha256)
    launch = official._p53_launch(root)
    p53 = run_p53_validation_subprocess(launch)
    _run_frozen_equivalence_subprocess(launch.python_executable)
    snapshot_inspector = official._CudaSnapshotInspector(visible)
    raw_observations = iter_validated_raw_observations(
        launch.raw_root,
        p53_attestation=p53,
        expected_p53_attestation_sha256=p53.sha256,
        benchmark_attestation=benchmark,
        expected_benchmark_attestation_sha256=benchmark.sha256,
    )
    inputs, cohort = _timed_inputs(raw_observations)
    accepted = accept_candidate_package(
        root / _PUBLISHED_CANDIDATE,
        expected_manifest_sha256=_PUBLISHED_MANIFEST_SHA256,
        trusted_cohort=cohort,
        mapping_authority=official._mapping_authority(root),
    )
    official._activate_source(paths)
    official.verify_upstream_preprocessing()
    official._require_loaded_source_origins(paths)
    commitment = official._commitment(root)
    adapter, _ = official._load_model(paths)
    run = run_timed_benchmark(
        inputs,
        trusted_cohort=cohort,
        commitment=commitment,
        adapter=adapter,
        mapping=official._mapping_authority(root).mapping,
        backend=OfficialCudaTimingBackend(
            expected_gpu_uuid=benchmark.gpu_uuid,
            snapshot_inspector=snapshot_inspector,
        ),
        projector=project_mapped_labels_semantic_only,
    )
    output_identity = _require_frozen_output_identity(
        run.canonical_results,
        accepted,
    )
    official._require_loaded_source_origins(paths)
    official.audit_candidate_assets(paths)
    _require_post_run_state(
        root,
        commit=commit,
        environment_sha256=environment_sha256,
        projector_data=projector_data,
    )
    timing_data = _canonical_timing_bytes(run)
    report = _build_report(
        run=run,
        commit=commit,
        environment_sha256=environment_sha256,
        benchmark_attestation_sha256=benchmark.sha256,
        p53_attestation_sha256=p53.sha256,
        projector_sha256=projector_sha256,
        segmenter_label_identity_tree_sha256=output_identity,
        timing_data=timing_data,
    )
    _publish_report(root, report.canonical_bytes(), timing_data)
    sys.stdout.buffer.write(report.canonical_bytes())
    return report


if __name__ == "__main__":
    main()
