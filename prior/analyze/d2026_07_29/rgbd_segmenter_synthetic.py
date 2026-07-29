"""Publish the fixed P5.4 synthetic RGB-D segmenter contract package."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as functional
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
    DeterministicFakeTimingBackend,
    DeviceBatch,
    EndpointRow,
    LicenseStatus,
    P53ValidationAttestation,
    P53ValidatorLaunch,
    PreparedHostBatch,
    RawFrameArrays,
    SegmenterInput,
    SpatialTransform,
    TimedBenchmarkInput,
    TrustedCohort,
    ValidatedRawObservation,
    aggregate_observation_metrics,
    capture_environment_sha256,
    compute_static_coverage,
    estimate_scene_robustness,
    iter_validated_raw_observations,
    load_nyu40_mapping,
    project_mapped_labels,
    project_oracle_target_labels,
    run_p53_validation_subprocess,
    run_timed_benchmark,
    score_observation,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    CandidateMappingAuthority,
    CandidateValidationAuthority,
    FileRecord,
    ObservationPackageRow,
    PredictionArtifact,
    SuccessfulPackageArtifacts,
    SyntheticSuccessfulManifest,
    TimingPackageRow,
    TrustedArtifactAuthority,
    accept_candidate_package,
    canonical_observations_bytes,
    canonical_timings_bytes,
    encode_prediction_npz,
    file_tree_aggregate,
    publish_successful_candidate_package,
    validate_candidate_package_fresh_process,
)

_ADAPTER_PATH = "prior/analyze/d2026_07_29/rgbd_segmenter_synthetic.py"
_CONTRACT_PATH = "prior/analyze/d2026_07_29/rgbd_segmenter_benchmark_contract.py"
_PACKAGE_PATH = "prior/analyze/d2026_07_29/rgbd_segmenter_benchmark_package.py"
_MAPPING_PATH = "prior/analyze/d2026_07_29/rgbd_segmenter_nyu40_mapping.json"
_PROJECTOR_PATH = "vlnce_baselines/models/etp_llm/llm_grid_oracle_cache.py"
_CONSTANTS_PATH = "prior/constants.py"
_ENVIRONMENT_LOCK_PATH = "pyproject.toml"
_DESTINATION = Path("data/rgbd_segmenter_benchmark/synthetic-contract-v1")
_COHORT_JSONL_SHA256 = (
    "89ae70f3e489fa702c66110f9bd9e7666ba0e16a9fbc3ac20aaa91a95adef0ce"
)
_SELECTION_SHA256 = "32a7adddf32291f059eb1045e63e077a693ca64d6fdf6e7418b54dd5d3644cb2"
_RAW_PAYLOAD_TREE_SHA256 = (
    "6b9c48460493bf8f50aeed408173c95210426723fb0c845324e87cfd4dad1c46"
)
_MODEL_SIZE = (193, 258)
_LOGIT_SIZE = (97, 129)
_MODULE = "prior.analyze.d2026_07_29.rgbd_segmenter_synthetic"

__all__ = ("SyntheticSegmenterAdapter", "SyntheticArgs", "parse_args", "main")


class SyntheticSegmenterAdapter:
    """Deterministic RGB/depth-only adapter used to prove the harness contract."""

    def preprocess_host(self, value: SegmenterInput) -> PreparedHostBatch:
        if not isinstance(value, SegmenterInput):
            raise ValueError("synthetic adapter requires SegmenterInput")
        transform = SpatialTransform.from_sizes(
            raw_height=256,
            raw_width=256,
            model_height=_MODEL_SIZE[0],
            model_width=_MODEL_SIZE[1],
        )
        rgb = value.rgb.permute(0, 3, 1, 2).to(dtype=torch.float32)
        depth = value.depth_m.unsqueeze(1)
        rgb = functional.interpolate(
            rgb,
            size=(transform.resized_height, transform.resized_width),
            mode="bilinear",
            align_corners=False,
        )
        depth = functional.interpolate(
            depth,
            size=(transform.resized_height, transform.resized_width),
            mode="nearest-exact",
        )
        padding = (
            transform.pad_left,
            transform.pad_right,
            transform.pad_top,
            transform.pad_bottom,
        )
        return PreparedHostBatch(
            tensors=(
                functional.pad(rgb, padding, value=0.0),
                functional.pad(depth, padding, value=0.0),
            ),
            spatial_transform=transform,
        )

    def infer(self, value: DeviceBatch) -> torch.Tensor:
        if not isinstance(value, DeviceBatch) or len(value.tensors) != 2:
            raise ValueError("synthetic adapter requires the runner device batch")
        rgb, depth = value.tensors
        if (
            rgb.shape != (12, 3, *_MODEL_SIZE)
            or depth.shape != (12, 1, *_MODEL_SIZE)
            or rgb.dtype is not torch.float32
            or depth.dtype is not torch.float32
        ):
            raise ValueError("synthetic prepared tensors differ from commitment")
        rgb_low = functional.interpolate(
            rgb, size=_LOGIT_SIZE, mode="bilinear", align_corners=False
        )
        depth_low = functional.interpolate(
            depth, size=_LOGIT_SIZE, mode="nearest-exact"
        )
        signal = torch.remainder(
            torch.floor(
                rgb_low[:, :1] * 0.125
                + rgb_low[:, 1:2] * 0.0625
                + rgb_low[:, 2:3] * 0.03125
                + depth_low * 4.0
            ).to(dtype=torch.int64),
            40,
        )
        classes = torch.arange(40, device=signal.device, dtype=torch.int64).view(
            1, 40, 1, 1
        )
        return -(classes - signal).abs().to(dtype=torch.float32)


class SyntheticArgs(Tap):
    """The P5.4 command intentionally has no configurable experiment inputs."""


def parse_args(argv: Optional[Sequence[str]] = None) -> SyntheticArgs:
    return SyntheticArgs(underscores_to_dashes=True).parse_args(argv)


def _repository_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[3]


def _require_publication_paths_absent(destination: Path) -> None:
    if not destination.is_absolute():
        raise ValueError("synthetic publication destination must be absolute")
    parent = destination.parent
    if parent.resolve(strict=True) != parent:
        raise ValueError("synthetic publication parent must be canonical")
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
    return value


def _command() -> Tuple[str, ...]:
    return (str(Path(sys.executable).resolve(strict=True)), "-m", _MODULE)


def _benchmark_attestation() -> BenchmarkEnvironmentAttestation:
    return BenchmarkEnvironmentAttestation(
        python_executable=str(Path(sys.executable).resolve(strict=True)),
        environment_sha256=capture_environment_sha256(),
        visible_device_count=torch.cuda.device_count(),
        gpu_name="NOT_APPLICABLE",
        gpu_uuid="NOT_APPLICABLE",
        timing_comparable=False,
    )


def _p53_launch(root: Path) -> P53ValidatorLaunch:
    return P53ValidatorLaunch(
        python_executable=Path(sys.executable).resolve(strict=True),
        expected_environment_sha256=capture_environment_sha256(),
        raw_root=(root / RAW_FRAME_ROOT).resolve(strict=True),
        timeout_seconds=300.0,
    )


def _commitment(root: Path) -> CandidateCommitment:
    mapping = load_nyu40_mapping(root / _MAPPING_PATH)
    return CandidateCommitment(
        candidate_id="synthetic-contract-v1",
        synthetic=True,
        repository_url="NOT_APPLICABLE",
        revision="NOT_APPLICABLE",
        checkpoint_id="NOT_APPLICABLE",
        checkpoint_url="NOT_APPLICABLE",
        checkpoint_sha256="NOT_APPLICABLE",
        source_dataset="NOT_APPLICABLE",
        code_license_status=LicenseStatus.NOT_APPLICABLE,
        weight_license_status=LicenseStatus.NOT_APPLICABLE,
        environment_lock_path=_ENVIRONMENT_LOCK_PATH,
        environment_lock_sha256=_record(root / _ENVIRONMENT_LOCK_PATH).sha256,
        rgb_units="uint8[0,255]",
        depth_units="float32-metres[0,10]",
        batch_views=12,
        source_vocabulary=tuple(entry.source_name for entry in mapping),
        mapping_sha256=NYU40_MAPPING_SHA256,
        rgb_interpolation="bilinear",
        rgb_coordinate_semantics="align_corners=False",
        depth_interpolation="nearest-exact",
        depth_coordinate_semantics="nearest-exact",
        normalization="none",
        invalid_depth_policy="zero",
        rgb_padding_value="0",
        depth_padding_value="0",
        precision_mode="float32",
    )


def _mapping_authority(root: Path) -> CandidateMappingAuthority:
    mapping = load_nyu40_mapping(root / _MAPPING_PATH)
    return CandidateMappingAuthority(
        source_vocabulary=tuple(entry.source_name for entry in mapping),
        mapping=mapping,
        mapping_sha256=NYU40_MAPPING_SHA256,
    )


def _authority(
    *,
    root: Path,
    manifest_sha256: str,
    p53_launch: P53ValidatorLaunch,
    benchmark: BenchmarkEnvironmentAttestation,
    commitment: CandidateCommitment,
    commit: str,
) -> CandidateValidationAuthority:
    environment = _record(root / _ENVIRONMENT_LOCK_PATH)
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
            file=environment,
        ),
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


def _manifest_fields(
    *,
    root: Path,
    p53: P53ValidationAttestation,
    benchmark: BenchmarkEnvironmentAttestation,
    commitment: CandidateCommitment,
    commit: str,
    cohort: TrustedCohort,
    observations: Tuple[ObservationPackageRow, ...],
    timings: Tuple[TimingPackageRow, ...],
    predictions: Tuple[Tuple[str, PredictionArtifact], ...],
    targets: Tuple[np.ndarray, ...],
) -> Mapping[str, object]:
    observation_data = canonical_observations_bytes(observations)
    timing_data = canonical_timings_bytes(timings)
    prediction_tree = file_tree_aggregate(
        (path, artifact.file) for path, artifact in predictions
    )
    metrics = aggregate_observation_metrics(
        tuple(row.metrics for row in observations),
        expected_scenes=cohort.scenes,
    )
    coverage = compute_static_coverage(
        load_nyu40_mapping(root / _MAPPING_PATH), targets
    )
    source_paths = {
        "adapter": _ADAPTER_PATH,
        "constants": _CONSTANTS_PATH,
        "contract": _CONTRACT_PATH,
        "mapping": _MAPPING_PATH,
        "package": _PACKAGE_PATH,
        "projector": _PROJECTOR_PATH,
    }
    return {
        "attestations": {
            "benchmark": _attestation_envelope(benchmark.canonical_bytes()),
            "p53_validator": _attestation_envelope(p53.canonical_bytes()),
        },
        "candidate_commitment": _candidate_json(commitment),
        "candidate_id": commitment.candidate_id,
        "command": list(_command()),
        "coverage": {**asdict(coverage), "support_ratio": coverage.support_ratio},
        "decision": "NOT_APPLICABLE",
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
        "latency_claim": False,
        "metric_summary": {
            "all_27": asdict(metrics.all_27),
            "per_category": [asdict(value) for value in metrics.per_category],
            "primary": asdict(metrics.primary),
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
        "robustness": {
            "f1": asdict(
                estimate_scene_robustness(
                    _endpoint_rows(observations, endpoint="f1"),
                    trusted_cohort=cohort,
                )
            ),
            "iou": asdict(
                estimate_scene_robustness(
                    _endpoint_rows(observations, endpoint="iou"),
                    trusted_cohort=cohort,
                )
            ),
        },
        "run_kind": "synthetic-contract",
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
            "backend": "deterministic-fake",
            "comparable": False,
            "measured_pass_count": 2,
            "measured_sample_count": 100,
            "unit": "synthetic-tick",
            "warmup_count": 20,
        },
    }


def _build_artifacts(
    *,
    root: Path,
    p53: P53ValidationAttestation,
    benchmark: BenchmarkEnvironmentAttestation,
    commitment: CandidateCommitment,
    commit: str,
    raw_observations: Sequence[ValidatedRawObservation],
    prediction_projector: Callable[[np.ndarray, RawFrameArrays], np.ndarray] = (
        project_mapped_labels
    ),
    target_projector: Callable[[RawFrameArrays], np.ndarray] = (
        project_oracle_target_labels
    ),
) -> Tuple[SuccessfulPackageArtifacts, TrustedCohort]:
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
    mapping = load_nyu40_mapping(root / _MAPPING_PATH)
    run = run_timed_benchmark(
        timed_inputs,
        trusted_cohort=cohort,
        commitment=commitment,
        adapter=SyntheticSegmenterAdapter(),
        mapping=mapping,
        backend=DeterministicFakeTimingBackend(),
        projector=prediction_projector,
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
    manifest = SyntheticSuccessfulManifest(
        _manifest_fields(
            root=root,
            p53=p53,
            benchmark=benchmark,
            commitment=commitment,
            commit=commit,
            cohort=cohort,
            observations=observations,
            timings=timings,
            predictions=predictions,
            targets=targets,
        )
    )
    manifest_sha256 = hashlib.sha256(manifest.canonical_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="rgbd-synthetic-package-") as directory:
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parse_args(argv)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError(
            "synthetic contract publication requires the unmasked P5.3 environment"
        )
    root = _repository_root()
    _require_publication_paths_absent((root / _DESTINATION).absolute())
    commit = _git_commit(root)
    benchmark = _benchmark_attestation()
    launch = _p53_launch(root)
    p53 = run_p53_validation_subprocess(launch)
    commitment = _commitment(root)
    observations = iter_validated_raw_observations(
        launch.raw_root,
        p53_attestation=p53,
        expected_p53_attestation_sha256=p53.sha256,
        benchmark_attestation=benchmark,
        expected_benchmark_attestation_sha256=benchmark.sha256,
    )
    artifacts, _ = _build_artifacts(
        root=root,
        p53=p53,
        benchmark=benchmark,
        commitment=commitment,
        commit=commit,
        raw_observations=observations,
    )
    manifest_sha256 = hashlib.sha256(artifacts.manifest.canonical_bytes()).hexdigest()
    authority = _authority(
        root=root,
        manifest_sha256=manifest_sha256,
        p53_launch=launch,
        benchmark=benchmark,
        commitment=commitment,
        commit=commit,
    )
    published = publish_successful_candidate_package(
        (root / _DESTINATION).absolute(),
        artifacts,
        authority,
    )
    record = published.validation.canonical_bytes()
    _fresh_validation(published.path, authority, record)
    sys.stdout.buffer.write(record)


if __name__ == "__main__":
    main()
