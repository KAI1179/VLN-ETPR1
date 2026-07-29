"""Publish the pinned P5.6 ESANet benchmark package."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import site
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from tap import Tap

from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    BOOTSTRAP_MATRIX_SHA256,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SCENE_COUNT,
    BOOTSTRAP_SEED,
    NYU40_MAPPING_SHA256,
    RAW_FRAME_ROOT,
    RAW_INDEX_SHA256,
    RAW_MANIFEST_SHA256,
    RAW_PRODUCER_COMMIT,
    BenchmarkEnvironmentAttestation,
    CandidateCommitment,
    CudaDeviceEvidence,
    EndpointRow,
    LicenseStatus,
    OfficialCudaEvidence,
    OfficialCudaTimingBackend,
    P53ValidationAttestation,
    P53ValidatorLaunch,
    ProvenanceChecks,
    RawFrameArrays,
    ResourceMeasurement,
    TimedBenchmarkInput,
    TimedBenchmarkRun,
    TrustedCohort,
    ValidatedRawObservation,
    aggregate_observation_metrics,
    capture_environment_sha256,
    compute_static_coverage,
    estimate_scene_robustness,
    evaluate_candidate_gates,
    inspect_visible_gpu,
    iter_validated_raw_observations,
    load_nyu40_mapping,
    project_mapped_labels,
    project_oracle_target_labels,
    run_p53_validation_subprocess,
    score_observation,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    AbortedOomPackage,
    AbortedOomPublicationAuthority,
    ArchivalCudaDeviceEvidence,
    ArchivalCudaEvidence,
    CandidateMappingAuthority,
    CandidateValidationAuthority,
    FileRecord,
    ObservationPackageRow,
    PredictionArtifact,
    RealProvenanceAuthority,
    RealSuccessfulManifest,
    SuccessfulPackageArtifacts,
    TimingPackageRow,
    TrustedArtifactAuthority,
    accept_candidate_package,
    canonical_observations_bytes,
    canonical_timings_bytes,
    encode_prediction_npz,
    file_tree_aggregate,
    publish_successful_candidate_package,
    run_timed_benchmark_for_publication,
    validate_candidate_package_fresh_process,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_esanet import (
    ESANetPaths,
    ESANetSegmenterAdapter,
    audit_candidate_assets,
    build_esanet,
    load_checkpoint_strict,
    verify_upstream_preprocessing,
)

_ADAPTER_PATH = "prior/analyze/d2026_07_29/rgbd_segmenter_esanet.py"
_CONTRACT_PATH = "prior/analyze/d2026_07_29/rgbd_segmenter_benchmark_contract.py"
_PACKAGE_PATH = "prior/analyze/d2026_07_29/rgbd_segmenter_benchmark_package.py"
_MAPPING_PATH = "prior/analyze/d2026_07_29/rgbd_segmenter_nyu40_mapping.json"
_PROJECTOR_PATH = "vlnce_baselines/models/etp_llm/llm_grid_oracle_cache.py"
_CONSTANTS_PATH = "prior/constants.py"
_ENVIRONMENT_LOCK_PATH = (
    "prior/analyze/d2026_07_29/rgbd_segmenter_esanet_requirements.txt"
)
_WEIGHT_PERMISSION_PATH = (
    "docs/superpowers/specs/2026-07-29-esanet-weight-risk-exception.md"
)
_DESTINATION = Path("data/rgbd_segmenter_benchmark/esanet-r34-nbt1d-scenenet-v1")
_MODULE = "prior.analyze.d2026_07_29.rgbd_segmenter_esanet_benchmark"
_REPOSITORY_URL = "https://github.com/TUI-NICR/ESANet.git"
_REVISION = "820c5bb633e49e69dcd075d4330165bb540a0cc9"
_CHECKPOINT_ID = "nyuv2/r34_NBt1D_scenenet.pth"
_CHECKPOINT_URL = "https://drive.google.com/uc?id=1w_Qa8AWUC6uHzQamwu-PAqA7P00hgl8w"
_CHECKPOINT_SHA256 = "6b84f77dee42739fd3c5dd9e6b278450fa14e6977e2ec1609060eb6eb05cf456"
_COHORT_JSONL_SHA256 = (
    "89ae70f3e489fa702c66110f9bd9e7666ba0e16a9fbc3ac20aaa91a95adef0ce"
)
_SELECTION_SHA256 = "32a7adddf32291f059eb1045e63e077a693ca64d6fdf6e7418b54dd5d3644cb2"
_RAW_PAYLOAD_TREE_SHA256 = (
    "6b9c48460493bf8f50aeed408173c95210426723fb0c845324e87cfd4dad1c46"
)
_VENV_RELATIVE = Path(
    "data/rgbd_segmenter_benchmark/candidates/esanet-r34-nbt1d-scenenet/"
    "environment/esanet-benchmark-v1"
)
_COMPLETE_ENVIRONMENT_SHA256 = (
    "a0342fd075394f31c9ee3d77f64278c6189b2137f88a9c47d12f09f8f6fd925c"
)
_P53_ENVIRONMENT_SHA256 = (
    "35be66f579798cbdee8c4717da06ed7c4449fe0094580e8829b0acc330e483b7"
)
_LOCKED_RUNTIME_DISTRIBUTIONS = {
    "pandas": "2.0.3",
    "pytz": "2025.2",
    "tzdata": "2025.3",
}
_RUNTIME_LOADER_ALIAS = "libstdc++.so.6"
_RUNTIME_LOADER_TARGET = "libstdc++.so.6.0.34"
_RUNTIME_LOADER_BYTE_LENGTH = 21_295_144
_RUNTIME_LOADER_SHA256 = (
    "9581ad615b7c073423f57b69a3b148a89f8ea76fc909124211f9007909b807a6"
)

__all__ = ("ESANetBenchmarkArgs", "main", "parse_args")


class ESANetBenchmarkArgs(Tap):
    """The official P5.6 publisher has no configurable experiment inputs."""


def parse_args(argv: Optional[Sequence[str]] = None) -> ESANetBenchmarkArgs:
    return ESANetBenchmarkArgs(underscores_to_dashes=True).parse_args(argv)


@dataclass(frozen=True)
class _ColdStart:
    model_construction_seconds: float
    checkpoint_byte_read_seconds: float
    checkpoint_load_seconds: float


def _repository_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[3]


def _record(path: Path) -> FileRecord:
    data = path.read_bytes()
    return FileRecord(len(data), hashlib.sha256(data).hexdigest())


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ("/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"),
        capture_output=True,
        check=True,
        timeout=10.0,
    )
    value = completed.stdout.decode("ascii").strip()
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("experiment Git commit is invalid")
    status = subprocess.run(
        (
            "/usr/bin/git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ),
        capture_output=True,
        check=True,
        timeout=10.0,
    )
    if status.stdout or status.stderr:
        raise ValueError("experiment Git repository must be clean")
    return value


def _command() -> Tuple[str, ...]:
    return (str(Path(sys.executable).resolve(strict=True)), "-m", _MODULE)


def _require_publication_paths_absent(destination: Path) -> None:
    if not destination.is_absolute():
        raise ValueError("ESANet publication destination must be absolute")
    parent = destination.parent
    if parent.resolve(strict=True) != parent:
        raise ValueError("ESANet publication parent must be canonical")
    descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        for name in (destination.name, f".{destination.name}.staging"):
            try:
                os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise FileExistsError(f"publication path already exists: {name}")
    finally:
        os.close(descriptor)


def _p53_launch(root: Path) -> P53ValidatorLaunch:
    python_executable = Path(sys.base_prefix).absolute() / "bin/python3.8"
    executable_info = python_executable.lstat()
    if (
        python_executable.resolve(strict=True) != python_executable
        or not stat.S_ISREG(executable_info.st_mode)
        or stat.S_ISLNK(executable_info.st_mode)
        or stat.S_IMODE(executable_info.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ValueError("P5.3 requires the canonical base Python 3.8 interpreter")
    return P53ValidatorLaunch(
        python_executable=python_executable,
        expected_environment_sha256=_P53_ENVIRONMENT_SHA256,
        raw_root=(root / RAW_FRAME_ROOT).resolve(strict=True),
        timeout_seconds=300.0,
    )


def _benchmark_attestation(
    environment_sha256: str,
) -> BenchmarkEnvironmentAttestation:
    gpu_name, gpu_uuid = inspect_visible_gpu()
    return BenchmarkEnvironmentAttestation(
        python_executable=str(Path(sys.executable).resolve(strict=True)),
        environment_sha256=environment_sha256,
        visible_device_count=torch.cuda.device_count(),
        gpu_name=gpu_name,
        gpu_uuid=gpu_uuid,
        timing_comparable=True,
    )


def _candidate_paths(root: Path) -> ESANetPaths:
    paths = ESANetPaths.default()
    site_packages = root / _VENV_RELATIVE / "lib/python3.8/site-packages"
    return ESANetPaths(
        repository_root=paths.repository_root,
        candidate_root=paths.candidate_root,
        source_root=paths.source_root,
        archive=paths.archive,
        checkpoint=paths.checkpoint,
        overlay=site_packages,
        requirements=paths.requirements,
    )


def _require_runtime_loader(venv: Path) -> None:
    def stable_identity(value: os.stat_result) -> Tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    loader_directory = venv / "lib"
    if loader_directory.resolve(strict=True) != loader_directory:
        raise ValueError("runtime-loader directory must be canonical")
    directory_descriptor = os.open(
        loader_directory,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    target_descriptor: Optional[int] = None
    try:
        alias_info = os.stat(
            _RUNTIME_LOADER_ALIAS,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISLNK(alias_info.st_mode)
            or os.readlink(
                _RUNTIME_LOADER_ALIAS,
                dir_fd=directory_descriptor,
            )
            != _RUNTIME_LOADER_TARGET
        ):
            raise ValueError("runtime-loader alias differs from the pinned target")
        target_descriptor = os.open(
            _RUNTIME_LOADER_TARGET,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
        before = os.fstat(target_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o555
            or before.st_nlink != 1
            or before.st_size != _RUNTIME_LOADER_BYTE_LENGTH
        ):
            raise ValueError("runtime-loader target metadata differs")
        digest = hashlib.sha256()
        while chunk := os.read(target_descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != _RUNTIME_LOADER_SHA256 or stable_identity(
            os.fstat(target_descriptor)
        ) != stable_identity(before):
            raise ValueError("runtime-loader target bytes differ")
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        os.close(directory_descriptor)


def _require_dedicated_interpreter(root: Path) -> None:
    venv = (root / _VENV_RELATIVE).absolute()
    expected = (root / _VENV_RELATIVE / "bin/python").absolute()
    expected_site = (venv / "lib/python3.8/site-packages").absolute()
    info = expected.lstat()
    if (
        Path(sys.executable).absolute() != expected
        or expected.resolve(strict=True) != expected
        or Path(sys.prefix).absolute() != venv
        or expected_site
        not in tuple(Path(value).absolute() for value in site.getsitepackages())
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ValueError("P5.6 requires the pinned copy-based experiment interpreter")
    _require_runtime_loader(venv)
    for name, version in _LOCKED_RUNTIME_DISTRIBUTIONS.items():
        distribution = importlib.metadata.distribution(name)
        if (
            distribution.version != version
            or Path(distribution.locate_file("")).resolve(strict=True) != expected_site
        ):
            raise ValueError(f"locked {name} distribution origin differs")
        module = importlib.import_module(name)
        module_path = getattr(module, "__file__", None)
        if (
            not isinstance(module_path, str)
            or expected_site not in Path(module_path).resolve(strict=True).parents
        ):
            raise ValueError(f"loaded {name} module origin differs")


def _require_loaded_source_origins(paths: ESANetPaths) -> None:
    expected_package = (paths.source_root / "src").resolve(strict=True)
    loaded = False
    for name, module in tuple(sys.modules.items()):
        if name != "src" and not name.startswith("src."):
            continue
        loaded = True
        module_path = getattr(module, "__file__", None)
        if isinstance(module_path, str):
            resolved = Path(module_path).resolve(strict=True)
            if (
                expected_package != resolved
                and expected_package not in resolved.parents
            ):
                raise ValueError(f"loaded upstream module origin differs: {name}")
            continue
        namespace_paths = getattr(module, "__path__", ())
        resolved_paths = tuple(
            Path(value).resolve(strict=True) for value in namespace_paths
        )
        if resolved_paths != (expected_package,):
            raise ValueError(f"loaded upstream namespace origin differs: {name}")
    if not loaded:
        raise ValueError("no upstream ESANet source modules were loaded")


def _activate_source(paths: ESANetPaths) -> None:
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise ValueError("generic src package is already loaded")
    if importlib.util.find_spec("src") is not None:
        raise ValueError("a competing generic src package is importable")
    source = str(paths.source_root)
    if source in sys.path:
        raise ValueError("ESANet source root was already active")
    sys.path.insert(0, source)
    importlib.invalidate_caches()


def _mapping_authority(root: Path) -> CandidateMappingAuthority:
    mapping = load_nyu40_mapping(root / _MAPPING_PATH)
    return CandidateMappingAuthority(
        source_vocabulary=tuple(entry.source_name for entry in mapping),
        mapping=mapping,
        mapping_sha256=NYU40_MAPPING_SHA256,
    )


def _commitment(root: Path) -> CandidateCommitment:
    mapping = _mapping_authority(root)
    return CandidateCommitment(
        candidate_id="esanet-r34-nbt1d-scenenet-v1",
        synthetic=False,
        repository_url=_REPOSITORY_URL,
        revision=_REVISION,
        checkpoint_id=_CHECKPOINT_ID,
        checkpoint_url=_CHECKPOINT_URL,
        checkpoint_sha256=_CHECKPOINT_SHA256,
        source_dataset="NYUv2-40",
        code_license_status=LicenseStatus.PASS,
        weight_license_status=LicenseStatus.PASS,
        environment_lock_path=_ENVIRONMENT_LOCK_PATH,
        environment_lock_sha256=_record(root / _ENVIRONMENT_LOCK_PATH).sha256,
        rgb_units="uint8[0,255]",
        depth_units="float32-metres[0,10]",
        batch_views=12,
        source_vocabulary=mapping.source_vocabulary,
        mapping_sha256=mapping.mapping_sha256,
        rgb_interpolation="identity-native-256x256",
        rgb_coordinate_semantics="pixel-centres-unchanged",
        depth_interpolation="identity-native-256x256",
        depth_coordinate_semantics="pixel-centres-unchanged",
        normalization=(
            "rgb-imagenet;depth-uint16-mm-mean-2841.94941272766-std-1417.2594281672277"
        ),
        invalid_depth_policy="quantized-zero-restored-to-zero",
        rgb_padding_value="none-native-geometry",
        depth_padding_value="none-native-geometry",
        precision_mode="float32",
    )


def _permission(value: TrustedArtifactAuthority) -> Mapping[str, object]:
    return {
        "byte_length": value.file.byte_length,
        "path": value.path,
        "root_role": value.root_role,
        "sha256": value.file.sha256,
    }


def _authority(
    *,
    root: Path,
    paths: ESANetPaths,
    manifest_sha256: str,
    p53_launch: P53ValidatorLaunch,
    benchmark: BenchmarkEnvironmentAttestation,
    commitment: CandidateCommitment,
    commit: str,
) -> CandidateValidationAuthority:
    code_permission = TrustedArtifactAuthority(
        root_role="candidate_repository",
        path="LICENSE",
        file=_record(paths.source_root / "LICENSE"),
    )
    weight_permission = TrustedArtifactAuthority(
        root_role="benchmark_repository",
        path=_WEIGHT_PERMISSION_PATH,
        file=_record(root / _WEIGHT_PERMISSION_PATH),
    )
    checkpoint_root = paths.checkpoint.parent.parent.resolve(strict=True)
    return CandidateValidationAuthority(
        expected_manifest_sha256=manifest_sha256,
        p53_launch=p53_launch,
        benchmark_attestation=benchmark,
        expected_benchmark_attestation_sha256=benchmark.sha256,
        expected_command=_command(),
        benchmark_repository_root=root,
        expected_producer_commit=commit,
        candidate=commitment,
        mapping=_mapping_authority(root),
        adapter_path=_ADAPTER_PATH,
        environment_lock=TrustedArtifactAuthority(
            root_role="benchmark_repository",
            path=_ENVIRONMENT_LOCK_PATH,
            file=_record(root / _ENVIRONMENT_LOCK_PATH),
        ),
        real=RealProvenanceAuthority(
            candidate_repository_root=paths.source_root.resolve(strict=True),
            checkpoint_root=checkpoint_root,
            checkpoint_path=str(paths.checkpoint.relative_to(checkpoint_root)),
            checkpoint_file=_record(paths.checkpoint),
            code_permission=code_permission,
            weight_permission=weight_permission,
        ),
    )


def _cuda_snapshot(expected_uuid: str) -> CudaDeviceEvidence:
    query = subprocess.run(
        (
            "nvidia-smi",
            f"--id={expected_uuid}",
            "--query-gpu=name,uuid,driver_version,persistence_mode,power.limit,"
            "temperature.gpu,clocks.applications.graphics,"
            "clocks.applications.memory",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        check=False,
        timeout=10.0,
    )
    if query.returncode != 0 or query.stderr:
        raise ValueError("unable to capture CUDA device policy")
    rows = list(csv.reader(query.stdout.decode("utf-8").splitlines()))
    if len(rows) != 1 or len(rows[0]) != 8:
        raise ValueError("CUDA device policy query returned an invalid row")
    (
        gpu_name,
        gpu_uuid,
        driver,
        persistence,
        power,
        temperature,
        graphics_clock,
        memory_clock,
    ) = (field.strip() for field in rows[0])
    processes = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        check=False,
        timeout=10.0,
    )
    if processes.returncode != 0 or processes.stderr:
        raise ValueError("unable to capture CUDA compute processes")
    process_rows = [
        tuple(field.strip() for field in row)
        for row in csv.reader(processes.stdout.decode("utf-8").splitlines())
        if row
    ]
    if any(len(row) != 2 for row in process_rows):
        raise ValueError("CUDA compute-process query returned an invalid row")
    try:
        pids = tuple(
            sorted(int(row[1]) for row in process_rows if row[0] == expected_uuid)
        )
        return CudaDeviceEvidence(
            gpu_name=gpu_name,
            gpu_uuid=gpu_uuid,
            driver_version=driver,
            clock_policy=f"graphics={graphics_clock};memory={memory_clock}",
            persistence_mode=persistence,
            power_limit_watts=float(power),
            temperature_celsius=float(temperature),
            compute_pids=pids,
        )
    except ValueError as error:
        raise ValueError("CUDA device evidence contains invalid values") from error


def _archival_cuda_evidence(
    before: CudaDeviceEvidence,
    after: CudaDeviceEvidence,
) -> ArchivalCudaEvidence:
    def convert(value: CudaDeviceEvidence) -> ArchivalCudaDeviceEvidence:
        return ArchivalCudaDeviceEvidence(
            gpu_name=value.gpu_name,
            gpu_uuid=value.gpu_uuid,
            driver_version=value.driver_version,
            clock_policy=value.clock_policy,
            persistence_mode=value.persistence_mode,
            power_limit_watts=value.power_limit_watts,
            temperature_celsius=value.temperature_celsius,
            compute_pids=value.compute_pids,
        )

    return ArchivalCudaEvidence(
        before=convert(before),
        after=convert(after),
        historical_pid=os.getpid(),
        pytorch_cuda_alloc_conf=None,
    )


def _attestation_envelope(data: bytes) -> Mapping[str, object]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("attestation must be a JSON object")
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "value": cast(Mapping[str, object], value),
    }


def _candidate_json(value: CandidateCommitment) -> Mapping[str, object]:
    result = asdict(value)
    result["code_license_status"] = value.code_license_status.value
    result["weight_license_status"] = value.weight_license_status.value
    result["source_vocabulary"] = list(value.source_vocabulary)
    return result


def _cuda_json(value: OfficialCudaEvidence) -> Mapping[str, object]:
    def snapshot(item: CudaDeviceEvidence) -> Mapping[str, object]:
        result = asdict(item)
        result["compute_pids"] = list(item.compute_pids)
        return result

    return {
        "after": snapshot(value.after),
        "before": snapshot(value.before),
        "historical_pid": value.current_pid,
        "pytorch_cuda_alloc_conf": value.pytorch_cuda_alloc_conf,
    }


def _endpoint_rows(
    observations: Sequence[ObservationPackageRow],
    *,
    endpoint: str,
) -> Tuple[EndpointRow, ...]:
    return tuple(
        EndpointRow(
            ordinal=row.ordinal,
            observation_id=row.observation_id,
            scene_id=row.scene_id,
            endpoint=cast(Optional[float], getattr(row.metrics.primary, endpoint)),
            status=row.status,
            failure_code=row.failure_code,
        )
        for row in observations
    )


def _passed_provenance() -> ProvenanceChecks:
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


def _manifest_fields(
    *,
    root: Path,
    paths: ESANetPaths,
    p53: P53ValidationAttestation,
    benchmark: BenchmarkEnvironmentAttestation,
    commitment: CandidateCommitment,
    commit: str,
    cohort: TrustedCohort,
    observations: Tuple[ObservationPackageRow, ...],
    timings: Tuple[TimingPackageRow, ...],
    predictions: Tuple[Tuple[str, PredictionArtifact], ...],
    targets: Tuple[np.ndarray, ...],
    run: TimedBenchmarkRun,
    cold_start: _ColdStart,
    resource: ResourceMeasurement,
) -> Mapping[str, object]:
    if run.latency is None or run.official_cuda_evidence is None:
        raise ValueError("real ESANet run lacks official CUDA evidence")
    observation_data = canonical_observations_bytes(observations)
    timing_data = canonical_timings_bytes(timings)
    prediction_tree = file_tree_aggregate(
        (path, artifact.file) for path, artifact in predictions
    )
    metrics = aggregate_observation_metrics(
        tuple(row.metrics for row in observations),
        expected_scenes=cohort.scenes,
    )
    if metrics.primary.mean_iou is None or metrics.primary.mean_f1 is None:
        raise ValueError("real ESANet run has no eligible quality endpoints")
    coverage = compute_static_coverage(
        load_nyu40_mapping(root / _MAPPING_PATH), targets
    )
    iou_rows = _endpoint_rows(observations, endpoint="iou")
    gates = evaluate_candidate_gates(
        synthetic=False,
        coverage=coverage,
        trusted_cohort=cohort,
        rows=iou_rows,
        mean_iou=metrics.primary.mean_iou,
        mean_f1=metrics.primary.mean_f1,
        latency=run.latency,
        resource=resource,
        code_license_status=commitment.code_license_status,
        weight_license_status=commitment.weight_license_status,
        provenance=_passed_provenance(),
    )
    code_permission = TrustedArtifactAuthority(
        "candidate_repository", "LICENSE", _record(paths.source_root / "LICENSE")
    )
    weight_permission = TrustedArtifactAuthority(
        "benchmark_repository",
        _WEIGHT_PERMISSION_PATH,
        _record(root / _WEIGHT_PERMISSION_PATH),
    )
    source_paths = {
        "adapter": _ADAPTER_PATH,
        "constants": _CONSTANTS_PATH,
        "contract": _CONTRACT_PATH,
        "mapping": _MAPPING_PATH,
        "package": _PACKAGE_PATH,
        "projector": _PROJECTOR_PATH,
    }
    latency = run.latency
    return {
        "attestations": {
            "benchmark": _attestation_envelope(benchmark.canonical_bytes()),
            "p53_validator": _attestation_envelope(p53.canonical_bytes()),
        },
        "candidate_commitment": _candidate_json(commitment),
        "candidate_id": commitment.candidate_id,
        "candidate_status": gates.overall.value,
        "cold_start": asdict(cold_start),
        "command": list(_command()),
        "coverage": {**asdict(coverage), "support_ratio": coverage.support_ratio},
        "cuda_evidence": _cuda_json(run.official_cuda_evidence),
        "files": {
            "observations": {
                "byte_length": len(observation_data),
                "row_count": 50,
                "sha256": hashlib.sha256(observation_data).hexdigest(),
            },
            "predictions": asdict(prediction_tree),
            "timings": {
                "byte_length": len(timing_data),
                "row_count": 100,
                "sha256": hashlib.sha256(timing_data).hexdigest(),
            },
        },
        "gates": {name: getattr(gates, name).value for name in vars(gates)},
        "latency": {
            "p50_seconds": latency.p50_seconds,
            "p95_seconds": latency.p95_seconds,
            "sample_count": latency.sample_count,
            "total_seconds": latency.total_seconds,
            "views_per_second": latency.views_per_second,
        },
        "metric_summary": {
            "all_27": asdict(metrics.all_27),
            "per_category": [asdict(value) for value in metrics.per_category],
            "primary": asdict(metrics.primary),
        },
        "permission_evidence": {
            "code": _permission(code_permission),
            "weights": _permission(weight_permission),
        },
        "producer_commit_prefix": commit[:12],
        "producer_git_commit": commit,
        "raw_inputs": {
            "cohort_jsonl_sha256": _COHORT_JSONL_SHA256,
            "raw_index_sha256": RAW_INDEX_SHA256,
            "raw_manifest_sha256": RAW_MANIFEST_SHA256,
            "raw_payload_tree_sha256": _RAW_PAYLOAD_TREE_SHA256,
            "raw_producer_git_commit": RAW_PRODUCER_COMMIT,
            "selection_sha256": _SELECTION_SHA256,
        },
        "resource": asdict(resource),
        "robustness": {
            "f1": asdict(
                estimate_scene_robustness(
                    _endpoint_rows(observations, endpoint="f1"),
                    trusted_cohort=cohort,
                )
            ),
            "iou": asdict(estimate_scene_robustness(iou_rows, trusted_cohort=cohort)),
        },
        "run_kind": "candidate",
        "run_status": "success",
        "schema_version": 1,
        "source_hashes": {
            role: {"path": path, "sha256": _record(root / path).sha256}
            for role, path in source_paths.items()
        },
        "statistics_protocol": {
            "bootstrap_matrix_sha256": BOOTSTRAP_MATRIX_SHA256,
            "numpy_version": np.__version__,
            "quantile_rule": "linear-h=(n-1)*p",
            "replicate_count": BOOTSTRAP_REPLICATES,
            "scene_count": BOOTSTRAP_SCENE_COUNT,
            "seed": BOOTSTRAP_SEED,
        },
        "timing_protocol": {
            "backend": "official-cuda",
            "comparable": True,
            "measured_pass_count": 2,
            "measured_sample_count": 100,
            "unit": "seconds",
            "warmup_count": 20,
        },
    }


def _load_model(paths: ESANetPaths) -> Tuple[ESANetSegmenterAdapter, _ColdStart]:
    start = time.perf_counter()
    model = build_esanet(device=torch.device("cuda:0"))
    torch.cuda.synchronize()
    construction_seconds = time.perf_counter() - start

    start = time.perf_counter()
    checkpoint_data = paths.checkpoint.read_bytes()
    byte_read_seconds = time.perf_counter() - start
    if hashlib.sha256(checkpoint_data).hexdigest() != _CHECKPOINT_SHA256:
        raise ValueError("checkpoint bytes differ from the committed digest")

    start = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+b") as accepted_checkpoint:
        accepted_checkpoint.write(checkpoint_data)
        accepted_checkpoint.flush()
        os.fchmod(accepted_checkpoint.fileno(), stat.S_IRUSR)
        accepted_checkpoint.seek(0)
        load_checkpoint_strict(
            model,
            Path(f"/proc/self/fd/{accepted_checkpoint.fileno()}"),
        )
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    return ESANetSegmenterAdapter(model), _ColdStart(
        model_construction_seconds=construction_seconds,
        checkpoint_byte_read_seconds=byte_read_seconds,
        checkpoint_load_seconds=load_seconds,
    )


def _build_artifacts(
    *,
    root: Path,
    paths: ESANetPaths,
    p53: P53ValidationAttestation,
    benchmark: BenchmarkEnvironmentAttestation,
    commitment: CandidateCommitment,
    commit: str,
    raw_observations: Sequence[ValidatedRawObservation],
    snapshot_inspector: Callable[[], CudaDeviceEvidence],
    oom_authority: AbortedOomPublicationAuthority,
    prediction_projector: Callable[[np.ndarray, RawFrameArrays], np.ndarray] = (
        project_mapped_labels
    ),
    target_projector: Callable[[RawFrameArrays], np.ndarray] = (
        project_oracle_target_labels
    ),
) -> Tuple[SuccessfulPackageArtifacts | AbortedOomPackage, TrustedCohort]:
    validated = tuple(raw_observations)
    timed_inputs = tuple(
        TimedBenchmarkInput(
            ordinal=value.row.ordinal,
            observation_id=value.row.observation_id,
            scene_id=value.row.scene_id,
            segmenter_input=value.segmenter_input,
            raw_arrays=value.audit_arrays,
        )
        for value in validated
    )
    cohort = TrustedCohort(
        tuple(
            (value.ordinal, value.observation_id, value.scene_id)
            for value in timed_inputs
        )
    )
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    adapter, cold_start = _load_model(paths)
    backend = OfficialCudaTimingBackend(
        expected_gpu_uuid=benchmark.gpu_uuid,
        snapshot_inspector=snapshot_inspector,
    )
    run = run_timed_benchmark_for_publication(
        timed_inputs,
        trusted_cohort=cohort,
        adapter=adapter,
        backend=backend,
        authority=oom_authority,
    )
    if isinstance(run, AbortedOomPackage):
        return run, cohort
    resource = ResourceMeasurement(
        baseline_allocated_bytes=baseline_allocated,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        baseline_reserved_bytes=baseline_reserved,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    targets = tuple(target_projector(value.raw_arrays) for value in timed_inputs)
    prediction_artifacts: list[Tuple[str, PredictionArtifact]] = []
    observation_rows: list[ObservationPackageRow] = []
    for value, result, target in zip(timed_inputs, run.canonical_results, targets):
        prediction_grid = prediction_projector(
            result.prediction.mapped_labels, value.raw_arrays
        )
        artifact = encode_prediction_npz(result.prediction)
        path = (
            f"predictions/{value.scene_id}/"
            f"{value.ordinal:02d}-{value.observation_id}.npz"
        )
        prediction_artifacts.append((path, artifact))
        observation_rows.append(
            ObservationPackageRow(
                ordinal=value.ordinal,
                observation_id=value.observation_id,
                scene_id=value.scene_id,
                status=result.status,
                failure_code=result.failure_code,
                prediction_path=path,
                prediction=artifact.file,
                prediction_members=artifact.members,
                metrics=score_observation(
                    ordinal=value.ordinal,
                    scene_id=value.scene_id,
                    prediction=prediction_grid,
                    target=target,
                ),
            )
        )
    observations = tuple(observation_rows)
    timings = tuple(TimingPackageRow(sample) for sample in run.samples)
    predictions = tuple(prediction_artifacts)
    manifest = RealSuccessfulManifest(
        _manifest_fields(
            root=root,
            paths=paths,
            p53=p53,
            benchmark=benchmark,
            commitment=commitment,
            commit=commit,
            cohort=cohort,
            observations=observations,
            timings=timings,
            predictions=predictions,
            targets=targets,
            run=run,
            cold_start=cold_start,
            resource=resource,
        )
    )
    manifest_sha256 = hashlib.sha256(manifest.canonical_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="rgbd-esanet-package-") as directory:
        package_root = Path(directory) / "package"
        package_root.mkdir(mode=0o700)
        for path, data in (
            ("manifest.json", manifest.canonical_bytes()),
            ("observations.jsonl", canonical_observations_bytes(observations)),
            ("timings.jsonl", canonical_timings_bytes(timings)),
            *((path, artifact.data) for path, artifact in predictions),
        ):
            destination = package_root / path
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            destination.write_bytes(data)
        accepted = accept_candidate_package(
            package_root,
            expected_manifest_sha256=manifest_sha256,
            trusted_cohort=cohort,
            mapping_authority=_mapping_authority(root),
        )
        return SuccessfulPackageArtifacts.from_accepted(accepted), cohort


def _fresh_validation(
    destination: Path,
    authority: CandidateValidationAuthority,
    expected_record: bytes,
) -> bytes:
    actual = validate_candidate_package_fresh_process(destination, authority)
    data = actual.canonical_bytes()
    if data != expected_record:
        raise RuntimeError("fresh post-publication validation failed")
    return data


def _require_publishable_success(
    artifacts: SuccessfulPackageArtifacts | AbortedOomPackage,
) -> SuccessfulPackageArtifacts:
    if isinstance(artifacts, AbortedOomPackage):
        raise RuntimeError(
            "CUDA OOM aborted the benchmark; no package was published because "
            "truthful post-run CUDA evidence is unavailable"
        )
    return artifacts


def main(argv: Optional[Sequence[str]] = None) -> None:
    parse_args(argv)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or re.fullmatch(r"GPU-[A-Za-z0-9-]+", visible) is None:
        raise ValueError("ESANet publication requires one visible GPU UUID")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ValueError("ESANet publication requires PYTHONNOUSERSITE=1")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise ValueError("ESANet publication requires PYTHONDONTWRITEBYTECODE=1")
    if os.environ.get("PYTHONPATH") is not None:
        raise ValueError("ESANet publication requires PYTHONPATH absent")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is not None:
        raise ValueError("ESANet publication requires allocator override absent")
    root = _repository_root()
    _require_dedicated_interpreter(root)
    destination = (root / _DESTINATION).absolute()
    _require_publication_paths_absent(destination)
    paths = _candidate_paths(root)
    audit_candidate_assets(paths)
    environment_sha256 = capture_environment_sha256()
    if environment_sha256 != _COMPLETE_ENVIRONMENT_SHA256:
        raise ValueError("complete benchmark environment differs from pinned digest")
    commit = _git_commit(root)
    benchmark = _benchmark_attestation(environment_sha256)
    launch = _p53_launch(root)
    p53 = run_p53_validation_subprocess(launch)
    observations = iter_validated_raw_observations(
        launch.raw_root,
        p53_attestation=p53,
        expected_p53_attestation_sha256=p53.sha256,
        benchmark_attestation=benchmark,
        expected_benchmark_attestation_sha256=benchmark.sha256,
    )
    _activate_source(paths)
    verify_upstream_preprocessing()
    _require_loaded_source_origins(paths)
    commitment = _commitment(root)
    preflight_snapshot = _cuda_snapshot(benchmark.gpu_uuid)
    oom_candidate_authority = _authority(
        root=root,
        paths=paths,
        manifest_sha256="0" * 64,
        p53_launch=launch,
        benchmark=benchmark,
        commitment=commitment,
        commit=commit,
    )
    oom_authority = AbortedOomPublicationAuthority(
        candidate=oom_candidate_authority,
        cuda_evidence=_archival_cuda_evidence(
            preflight_snapshot,
            preflight_snapshot,
        ),
    )
    artifacts, _ = _build_artifacts(
        root=root,
        paths=paths,
        p53=p53,
        benchmark=benchmark,
        commitment=commitment,
        commit=commit,
        raw_observations=observations,
        snapshot_inspector=lambda: _cuda_snapshot(benchmark.gpu_uuid),
        oom_authority=oom_authority,
    )
    _require_loaded_source_origins(paths)
    audit_candidate_assets(paths)
    if capture_environment_sha256() != environment_sha256:
        raise ValueError("benchmark environment changed during upstream execution")
    artifacts = _require_publishable_success(artifacts)
    authority = _authority(
        root=root,
        paths=paths,
        manifest_sha256=hashlib.sha256(
            artifacts.manifest.canonical_bytes()
        ).hexdigest(),
        p53_launch=launch,
        benchmark=benchmark,
        commitment=commitment,
        commit=commit,
    )
    published = publish_successful_candidate_package(
        destination,
        artifacts,
        authority,
    )
    record = published.validation.canonical_bytes()
    _fresh_validation(published.path, authority, record)
    sys.stdout.buffer.write(record)


if __name__ == "__main__":
    main()
