from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, cast

import pytest
import torch

from prior.analyze.d2026_07_29 import rgbd_segmenter_esanet_benchmark as benchmark
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    CudaDeviceEvidence,
    LicenseStatus,
    OfficialCudaEvidence,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_package import (
    CandidateValidationAuthority,
)


def test_cli_has_no_experiment_overrides() -> None:
    benchmark.parse_args(())
    with pytest.raises(SystemExit):
        benchmark.parse_args(("--device", "cuda:0"))


def test_commitment_binds_narrow_user_approved_weight_permission() -> None:
    value = benchmark._commitment(Path.cwd())
    assert value.synthetic is False
    assert value.candidate_id == "esanet-r34-nbt1d-scenenet-v1"
    assert value.revision == benchmark._REVISION
    assert value.checkpoint_sha256 == benchmark._CHECKPOINT_SHA256
    assert value.code_license_status is LicenseStatus.PASS
    assert value.weight_license_status is LicenseStatus.PASS
    assert value.batch_views == 12
    assert value.precision_mode == "float32"


def test_candidate_paths_replace_target_overlay_with_intrinsic_venv_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = SimpleNamespace(
        repository_root=tmp_path,
        candidate_root=tmp_path / "candidate",
        source_root=tmp_path / "candidate/source",
        archive=tmp_path / "candidate/archive",
        checkpoint=tmp_path / "candidate/checkpoint",
        overlay=tmp_path / "candidate/old-overlay",
        requirements=tmp_path / "requirements",
    )
    monkeypatch.setattr(benchmark.ESANetPaths, "default", lambda: default)
    value = benchmark._candidate_paths(tmp_path)
    assert value.overlay == (
        tmp_path / benchmark._VENV_RELATIVE / "lib/python3.8/site-packages"
    )
    assert value.source_root == default.source_root
    assert value.checkpoint == default.checkpoint


def test_p53_launch_uses_distinct_frozen_base_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path.cwd()
    base = Path(sys.base_prefix).resolve(strict=True)
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.setattr(sys, "executable", "/candidate/venv/bin/python")
    launch = benchmark._p53_launch(root)
    assert launch.python_executable == base / "bin/python3.8"
    assert launch.python_executable != Path(sys.executable)
    assert launch.expected_environment_sha256 == benchmark._P53_ENVIRONMENT_SHA256
    assert launch.expected_environment_sha256 != benchmark._COMPLETE_ENVIRONMENT_SHA256


def test_dedicated_interpreter_requires_copy_not_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / benchmark._VENV_RELATIVE / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python")
    executable.chmod(0o555)
    loader = executable.parent.parent / "lib/libstdc++.so.6.0.34"
    loader.parent.mkdir()
    loader.write_bytes(b"loader")
    loader.chmod(0o555)
    (loader.parent / "libstdc++.so.6").symlink_to(loader.name)
    monkeypatch.setattr(benchmark, "_RUNTIME_LOADER_BYTE_LENGTH", len(b"loader"))
    monkeypatch.setattr(
        benchmark,
        "_RUNTIME_LOADER_SHA256",
        benchmark.hashlib.sha256(b"loader").hexdigest(),
    )
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(sys, "prefix", str(executable.parent.parent))
    monkeypatch.setattr(
        benchmark.site,
        "getsitepackages",
        lambda: [
            str(tmp_path / benchmark._VENV_RELATIVE / "lib/python3.8/site-packages")
        ],
    )
    distribution = SimpleNamespace(
        version="2.0.3",
        locate_file=lambda _path: (
            tmp_path / benchmark._VENV_RELATIVE / "lib/python3.8/site-packages"
        ),
    )
    monkeypatch.setattr(
        benchmark.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            version=benchmark._LOCKED_RUNTIME_DISTRIBUTIONS[name],
            locate_file=distribution.locate_file,
        ),
    )
    monkeypatch.setattr(
        benchmark.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            __file__=str(
                tmp_path
                / benchmark._VENV_RELATIVE
                / "lib/python3.8/site-packages"
                / name
                / "__init__.py"
            )
        ),
    )
    for name in benchmark._LOCKED_RUNTIME_DISTRIBUTIONS:
        package = (
            tmp_path / benchmark._VENV_RELATIVE / "lib/python3.8/site-packages" / name
        )
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
    benchmark._require_dedicated_interpreter(tmp_path)

    executable.unlink()
    executable.symlink_to("/bin/true")
    with pytest.raises(ValueError, match="copy-based"):
        benchmark._require_dedicated_interpreter(tmp_path)


def test_runtime_loader_requires_exact_relative_alias_and_immutable_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv = tmp_path / "venv"
    loader = venv / "lib/libstdc++.so.6.0.34"
    loader.parent.mkdir(parents=True)
    loader.write_bytes(b"loader")
    loader.chmod(0o555)
    alias = loader.parent / "libstdc++.so.6"
    alias.symlink_to(loader.name)
    monkeypatch.setattr(benchmark, "_RUNTIME_LOADER_BYTE_LENGTH", len(b"loader"))
    monkeypatch.setattr(
        benchmark,
        "_RUNTIME_LOADER_SHA256",
        benchmark.hashlib.sha256(b"loader").hexdigest(),
    )
    benchmark._require_runtime_loader(venv)

    alias.unlink()
    alias.symlink_to(loader)
    with pytest.raises(ValueError, match="alias differs"):
        benchmark._require_runtime_loader(venv)
    alias.unlink()
    alias.symlink_to(loader.name)
    loader.chmod(0o755)
    with pytest.raises(ValueError, match="metadata differs"):
        benchmark._require_runtime_loader(venv)


def test_source_activation_rejects_preloaded_and_competing_src(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = SimpleNamespace(source_root=tmp_path / "source")
    monkeypatch.setitem(sys.modules, "src", SimpleNamespace())
    with pytest.raises(ValueError, match="already loaded"):
        benchmark._activate_source(cast(benchmark.ESANetPaths, paths))
    monkeypatch.delitem(sys.modules, "src")
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda _name: object())
    with pytest.raises(ValueError, match="competing"):
        benchmark._activate_source(cast(benchmark.ESANetPaths, paths))


def test_loaded_source_origins_reject_competing_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "checkout"
    (expected / "src").mkdir(parents=True)
    good = expected / "src/model.py"
    good.touch()
    bad = tmp_path / "other/model.py"
    bad.parent.mkdir()
    bad.touch()
    paths = cast(benchmark.ESANetPaths, SimpleNamespace(source_root=expected))
    monkeypatch.setitem(
        sys.modules,
        "src",
        SimpleNamespace(__file__=None, __path__=[str(expected / "src")]),
    )
    monkeypatch.setitem(sys.modules, "src.model", SimpleNamespace(__file__=str(good)))
    benchmark._require_loaded_source_origins(paths)
    monkeypatch.setitem(sys.modules, "src.model", SimpleNamespace(__file__=str(bad)))
    with pytest.raises(ValueError, match="origin differs"):
        benchmark._require_loaded_source_origins(paths)


def test_cuda_snapshot_captures_exact_device_policy_and_compute_pids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu = subprocess.CompletedProcess(
        (),
        0,
        (
            b"NVIDIA GeForce RTX 3090, GPU-test, 550.1, Disabled, "
            b"350.00, 45, 1695, 9751\n"
        ),
        b"",
    )
    processes = subprocess.CompletedProcess(
        (), 0, b"GPU-other, 7\nGPU-test, 123\n", b""
    )
    responses = iter((gpu, processes))
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *_a, **_k: next(responses))
    value = benchmark._cuda_snapshot("GPU-test")
    assert value == CudaDeviceEvidence(
        gpu_name="NVIDIA GeForce RTX 3090",
        gpu_uuid="GPU-test",
        driver_version="550.1",
        clock_policy="graphics=1695;memory=9751",
        persistence_mode="Disabled",
        power_limit_watts=350.0,
        temperature_celsius=45.0,
        compute_pids=(123,),
    )


@pytest.mark.parametrize(
    "response",
    (
        subprocess.CompletedProcess((), 1, b"", b"failure"),
        subprocess.CompletedProcess((), 0, b"short,row\n", b""),
    ),
)
def test_cuda_snapshot_rejects_failed_or_malformed_policy_query(
    monkeypatch: pytest.MonkeyPatch,
    response: subprocess.CompletedProcess[bytes],
) -> None:
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *_a, **_k: response)
    with pytest.raises(ValueError, match="CUDA device policy"):
        benchmark._cuda_snapshot("GPU-test")


def test_cuda_snapshot_inspector_binds_namespace_pid_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def snapshot(
        pids: tuple[int, ...],
        *,
        temperature: float = 45.0,
    ) -> CudaDeviceEvidence:
        return CudaDeviceEvidence(
            "NVIDIA GeForce RTX 3090",
            "GPU-test",
            "550.1",
            "graphics=1695;memory=9751",
            "Disabled",
            350.0,
            temperature,
            pids,
        )

    raw = iter((
        snapshot(()),
        snapshot(()),
        snapshot((3_281_897,), temperature=45.5),
        snapshot((3_281_897,), temperature=46.0),
        snapshot((3_281_897,), temperature=46.5),
        snapshot((3_281_897,), temperature=47.0),
    ))
    monkeypatch.setattr(benchmark, "_cuda_snapshot", lambda _uuid: next(raw))
    inspector = benchmark._CudaSnapshotInspector("GPU-test")
    first = inspector()
    second = inspector()
    assert first.compute_pids == (os.getpid(),)
    assert second.compute_pids == (os.getpid(),)
    assert first.temperature_celsius == 46.0
    assert second.temperature_celsius == 47.0
    with pytest.raises(ValueError, match="exactly two calls"):
        inspector()

    preexisting = iter((snapshot(()), snapshot((99,))))
    monkeypatch.setattr(
        benchmark,
        "_cuda_snapshot",
        lambda _uuid: next(preexisting),
    )
    with pytest.raises(ValueError, match="pre-existing"):
        benchmark._CudaSnapshotInspector("GPU-test")


@pytest.mark.parametrize(
    ("call_snapshots", "message"),
    (
        (((7,), ()), "exactly one"),
        (((7,), (7, 8)), "exactly one"),
        (((7,), (7,), (8,), (8,)), "process drifted"),
    ),
)
def test_cuda_snapshot_inspector_rejects_empty_extra_and_pid_drift(
    monkeypatch: pytest.MonkeyPatch,
    call_snapshots: tuple[tuple[int, ...], ...],
    message: str,
) -> None:
    def snapshot(pids: tuple[int, ...]) -> CudaDeviceEvidence:
        return CudaDeviceEvidence(
            "NVIDIA GeForce RTX 3090",
            "GPU-test",
            "550.1",
            "graphics=1695;memory=9751",
            "Disabled",
            350.0,
            45.0,
            pids,
        )

    raw = iter((
        snapshot(()),
        snapshot(()),
        *(snapshot(pids) for pids in call_snapshots),
    ))
    monkeypatch.setattr(benchmark, "_cuda_snapshot", lambda _uuid: next(raw))
    inspector = benchmark._CudaSnapshotInspector("GPU-test")
    if len(call_snapshots) == 4:
        inspector()
    with pytest.raises(ValueError, match=message):
        inspector()


def test_cuda_snapshot_inspector_rejects_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = CudaDeviceEvidence(
        "NVIDIA GeForce RTX 3090",
        "GPU-test",
        "550.1",
        "graphics=1695;memory=9751",
        "Disabled",
        350.0,
        45.0,
        (),
    )
    monkeypatch.setattr(benchmark, "_cuda_snapshot", lambda _uuid: snapshot)
    inspector = benchmark._CudaSnapshotInspector("GPU-test")
    creator_pid = os.getpid()
    monkeypatch.setattr(benchmark.os, "getpid", lambda: creator_pid + 1)
    with pytest.raises(ValueError, match="process boundary"):
        inspector()


def test_cuda_json_converts_live_evidence_to_archival_schema() -> None:
    before = CudaDeviceEvidence(
        "NVIDIA GeForce RTX 3090",
        "GPU-test",
        "550.1",
        "default",
        "Disabled",
        350.0,
        45.0,
        (os.getpid(),),
    )
    evidence = OfficialCudaEvidence(
        before,
        CudaDeviceEvidence(
            "NVIDIA GeForce RTX 3090",
            "GPU-test",
            "550.1",
            "default",
            "Disabled",
            350.0,
            46.0,
            (os.getpid(),),
        ),
        os.getpid(),
        None,
    )
    value = benchmark._cuda_json(evidence)
    assert value["historical_pid"] == os.getpid()
    assert value["pytorch_cuda_alloc_conf"] is None
    assert cast(Mapping[str, object], value["before"])["compute_pids"] == [os.getpid()]


def test_checkpoint_loader_consumes_the_measured_digest_bound_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    data = b"committed checkpoint bytes"
    checkpoint.write_bytes(data)
    paths = cast(benchmark.ESANetPaths, SimpleNamespace(checkpoint=checkpoint))
    model = torch.nn.Identity().eval()
    loaded: list[bytes] = []
    monkeypatch.setattr(
        benchmark,
        "_CHECKPOINT_SHA256",
        benchmark.hashlib.sha256(data).hexdigest(),
    )
    monkeypatch.setattr(benchmark, "build_esanet", lambda **_kwargs: model)
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        benchmark,
        "load_checkpoint_strict",
        lambda _model, path: loaded.append(path.read_bytes()),
    )
    adapter, cold = benchmark._load_model(paths)
    assert isinstance(adapter, benchmark.ESANetSegmenterAdapter)
    assert loaded == [data]
    assert cold.checkpoint_byte_read_seconds >= 0
    assert cold.checkpoint_load_seconds >= 0


def test_direct_timed_cuda_oom_propagates_without_packaging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fatal = torch.cuda.OutOfMemoryError("fatal timing OOM")
    monkeypatch.setattr(benchmark, "TrustedCohort", lambda _values: object())
    monkeypatch.setattr(
        benchmark,
        "_load_model",
        lambda _paths: (object(), object()),
    )
    monkeypatch.setattr(
        benchmark,
        "OfficialCudaTimingBackend",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        benchmark,
        "_mapping_authority",
        lambda _root: SimpleNamespace(mapping=()),
    )
    monkeypatch.setattr(benchmark.torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(benchmark.torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(benchmark.torch.cuda, "reset_peak_memory_stats", lambda: None)

    def raise_oom(*_args: object, **_kwargs: object) -> object:
        raise fatal

    monkeypatch.setattr(benchmark, "run_timed_benchmark", raise_oom)
    with pytest.raises(torch.cuda.OutOfMemoryError) as raised:
        benchmark._build_artifacts(
            root=tmp_path,
            paths=cast(benchmark.ESANetPaths, object()),
            p53=cast(benchmark.P53ValidationAttestation, object()),
            benchmark=cast(
                benchmark.BenchmarkEnvironmentAttestation,
                SimpleNamespace(gpu_uuid="GPU-test"),
            ),
            commitment=cast(benchmark.CandidateCommitment, object()),
            commit="0" * 40,
            raw_observations=(),
            snapshot_inspector=lambda: cast(benchmark.CudaDeviceEvidence, object()),
        )
    assert raised.value is fatal


def test_fresh_validation_requires_byte_identical_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = cast(CandidateValidationAuthority, object())
    monkeypatch.setattr(
        benchmark,
        "validate_candidate_package_fresh_process",
        lambda *_args: SimpleNamespace(canonical_bytes=lambda: b"record\n"),
    )
    assert benchmark._fresh_validation(tmp_path, authority, b"record\n") == b"record\n"
    with pytest.raises(RuntimeError, match="fresh post-publication"):
        benchmark._fresh_validation(tmp_path, authority, b"different\n")


def _publisher_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)


def test_main_rejects_existing_destination_before_candidate_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publisher_environment(monkeypatch)
    destination = tmp_path / benchmark._DESTINATION
    destination.parent.mkdir(parents=True)
    destination.mkdir()
    monkeypatch.setattr(benchmark, "_repository_root", lambda: tmp_path)
    gpu_queried = False

    def inspect_gpu(_uuid: str) -> object:
        nonlocal gpu_queried
        gpu_queried = True
        return object()

    monkeypatch.setattr(benchmark, "_CudaSnapshotInspector", inspect_gpu)
    monkeypatch.setattr(benchmark, "_require_dedicated_interpreter", lambda _root: None)
    called = False

    def audit(_paths: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(benchmark, "audit_candidate_assets", audit)
    with pytest.raises(FileExistsError, match="already exists"):
        benchmark.main(())
    assert called is False
    assert gpu_queried is False


def test_main_publishes_once_then_fresh_validates_before_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publisher_environment(monkeypatch)
    destination = tmp_path / benchmark._DESTINATION
    destination.parent.mkdir(parents=True)
    paths = cast(benchmark.ESANetPaths, SimpleNamespace())
    p53 = SimpleNamespace(sha256="1" * 64)
    environment = SimpleNamespace(
        sha256="2" * 64,
        gpu_uuid="GPU-test",
    )
    manifest = SimpleNamespace(canonical_bytes=lambda: b"manifest\n")
    artifacts = SimpleNamespace(manifest=manifest)
    authority = object()
    record = b'{"validation_status":"PASS"}\n'
    validation = SimpleNamespace(canonical_bytes=lambda: record)
    published = SimpleNamespace(path=destination, validation=validation)
    events: list[str] = []

    monkeypatch.setattr(benchmark, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        benchmark,
        "_CudaSnapshotInspector",
        lambda _uuid: events.append("gpu") or object(),
    )
    monkeypatch.setattr(
        benchmark, "_require_dedicated_interpreter", lambda _root: events.append("venv")
    )
    monkeypatch.setattr(benchmark, "_candidate_paths", lambda _root: paths)
    monkeypatch.setattr(
        benchmark, "audit_candidate_assets", lambda _paths: events.append("audit")
    )
    monkeypatch.setattr(benchmark, "capture_environment_sha256", lambda: "3" * 64)
    monkeypatch.setattr(benchmark, "_COMPLETE_ENVIRONMENT_SHA256", "3" * 64)
    monkeypatch.setattr(benchmark, "_git_commit", lambda _root: "4" * 40)
    monkeypatch.setattr(benchmark, "_benchmark_attestation", lambda _hash: environment)
    launch = SimpleNamespace(raw_root=tmp_path / "raw")
    monkeypatch.setattr(benchmark, "_p53_launch", lambda _root: launch)
    monkeypatch.setattr(
        benchmark,
        "run_p53_validation_subprocess",
        lambda _launch: events.append("p53") or p53,
    )
    monkeypatch.setattr(
        benchmark,
        "iter_validated_raw_observations",
        lambda *_a, **_k: events.append("raw") or (),
    )
    monkeypatch.setattr(
        benchmark, "_activate_source", lambda _paths: events.append("src")
    )
    monkeypatch.setattr(
        benchmark,
        "verify_upstream_preprocessing",
        lambda: events.append("preprocess"),
    )
    monkeypatch.setattr(
        benchmark,
        "_require_loaded_source_origins",
        lambda _paths: events.append("origins"),
    )
    monkeypatch.setattr(benchmark, "_commitment", lambda _root: object())
    monkeypatch.setattr(
        benchmark,
        "_build_artifacts",
        lambda **_kwargs: (events.append("run") or artifacts, object()),
    )
    monkeypatch.setattr(
        benchmark,
        "_authority",
        lambda **_kwargs: events.append("authority") or authority,
    )

    def publish(*_args: object) -> object:
        events.append("publish")
        return published

    monkeypatch.setattr(benchmark, "publish_successful_candidate_package", publish)
    monkeypatch.setattr(
        benchmark,
        "_fresh_validation",
        lambda *_args: events.append("fresh") or record,
    )
    output = SimpleNamespace(buffer=io.BytesIO())
    monkeypatch.setattr(sys, "stdout", output)
    benchmark.main(())
    assert output.buffer.getvalue() == record
    assert events == [
        "venv",
        "audit",
        "p53",
        "gpu",
        "raw",
        "src",
        "preprocess",
        "origins",
        "run",
        "origins",
        "audit",
        "authority",
        "publish",
        "fresh",
    ]


def test_interpreter_permission_gate_rejects_writable_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / benchmark._VENV_RELATIVE / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python")
    executable.chmod(stat.S_IRWXU | stat.S_IWGRP)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(sys, "prefix", str(executable.parent.parent))
    with pytest.raises(ValueError, match="copy-based"):
        benchmark._require_dedicated_interpreter(tmp_path)
