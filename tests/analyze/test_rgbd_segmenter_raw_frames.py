from __future__ import annotations

import hashlib
import io
import math
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping, Optional, Sequence, Union
import zipfile

import numpy as np
import pytest

from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    FileRecord,
    MemberMetadata,
    RawFrameArrays,
    strict_read_bytes,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames import (
    ORACLE_ARTIFACT_ROOT,
    SCENE_DATASET_ROOT,
    CollectionObservation,
    PinnedOracle,
    SceneBundle,
    SnapshotFile,
    build_scene_simulator,
    load_collection_inputs,
    load_pinned_oracle_after_render,
    render_raw_frame_artifact,
    replay_and_require_exact,
    run_first_row_smoke,
    snapshot_scene_bundle,
)
from prior.constants import MAPPED_OBJECT_NAMES, OBJECT_MAPPING, REGION_MAPPING
from vlnce_baselines.models.etp_llm.llm_grid_oracle_cache import (
    OracleSensorFrame,
    project_oracle_frames,
)


def _oracle_bytes() -> bytes:
    semantic = np.zeros((37, 50, 50), dtype=np.bool_)
    observed = np.zeros((50, 50), dtype=np.bool_)
    free = np.zeros((50, 50), dtype=np.bool_)
    observed[0, 0] = True
    free[0, 0] = True
    semantic[0, 0, 0] = True
    output = io.BytesIO()
    np.savez_compressed(
        output,
        schema_version=np.asarray(1, dtype=np.int64),
        grid_scale=np.asarray(2, dtype=np.int64),
        cell_size_m=np.asarray(1.0, dtype=np.float32),
        ego_semantic_grid=semantic,
        ego_observed_mask=observed,
        ego_free_mask=free,
        target_semantic_grid=semantic,
        target_observed_mask=observed,
        target_free_mask=free,
        start_position=np.asarray((1.0, 2.0), dtype=np.float32),
        start_direction=np.asarray((0.0, 1.0), dtype=np.float32),
    )
    return output.getvalue()


def _observation(*, artifact_sha256: str) -> CollectionObservation:
    return CollectionObservation(
        artifact_path=Path("observations/scene/0123456789abcdefabcd.npz"),
        artifact_sha256=artifact_sha256,
        example_ids=("R2R_val_unseen_1",),
        observation_id="0123456789abcdefabcd",
        scene_id="scene",
        start_position=(1.0, 2.0, 3.0),
        start_rotation=(0.0, 0.0, 0.0, 1.0),
    )


def test_strict_read_bytes_uses_single_nofollow_regular_file_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.bin"
    payload = b"accepted buffer"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    calls: list[int] = []
    original_open = os.open

    def checked_open(
        name: Union[str, bytes, os.PathLike[str], os.PathLike[bytes]],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: Optional[int] = None,
    ) -> int:
        if str(name) == path.name:
            calls.append(flags)
        return original_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", checked_open)

    assert strict_read_bytes(path, expected, "source") == payload
    assert calls == [os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW]


def test_strict_read_bytes_rejects_symlink_and_hash_drift(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"target")
    symlink = tmp_path / "link"
    symlink.symlink_to(target)

    with pytest.raises(ValueError, match="regular file"):
        strict_read_bytes(symlink, hashlib.sha256(b"target").hexdigest(), "source")
    with pytest.raises(ValueError, match="SHA-256"):
        strict_read_bytes(target, "0" * 64, "source")
    directory_link = tmp_path / "linked-directory"
    directory_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="regular file"):
        strict_read_bytes(
            directory_link / "target",
            hashlib.sha256(b"target").hexdigest(),
            "source",
        )


def test_strict_read_bytes_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="regular file"):
        strict_read_bytes(fifo, "0" * 64, "source")


def test_collection_inputs_reproduce_the_sealed_50_observations_without_oracle_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_oracles: list[Path] = []
    original_open = os.open

    def checked_open(
        name: Union[str, bytes, os.PathLike[str], os.PathLike[bytes]],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: Optional[int] = None,
    ) -> int:
        if str(name).endswith(".npz"):
            opened_oracles.append(Path(str(name)))
        return original_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", checked_open)

    inputs = load_collection_inputs()

    assert len(inputs.observations) == 50
    assert (
        tuple(sorted({item.scene_id for item in inputs.observations})) == inputs.scenes
    )
    assert len(inputs.scenes) == 11
    assert sum(len(item.example_ids) for item in inputs.observations) == 255
    assert {
        scene: sum(item.scene_id == scene for item in inputs.observations)
        for scene in inputs.scenes
    } == {
        "2azQ1b91cZZ": 8,
        "8194nk5LbLH": 1,
        "EU6Fwq7SyZv": 4,
        "QUCTc6BB5sX": 8,
        "TbHJrupSAjP": 7,
        "X7HyMhZNoso": 4,
        "Z6MFQCViBuw": 4,
        "oLBMNvg9in8": 5,
        "pLe4wQe7qrG": 1,
        "x8F5xyUWy9e": 3,
        "zsNo4HB9uLZ": 5,
    }
    assert opened_oracles == []
    assert all(
        item.artifact_path.parts[:1] == ("observations",)
        for item in inputs.observations
    )


def test_collection_input_rejects_alias_and_artifact_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = load_collection_inputs()
    cohort = Path("data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1/cohort.jsonl")
    changed = cohort.read_bytes().replace(b"R2R_val_unseen_1", b"R2R_val_unseen_9", 1)

    def changed_reader(path: Path, expected_sha256: str, label: str) -> bytes:
        if path == cohort:
            return changed
        return strict_read_bytes(path, expected_sha256, label)

    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.strict_read_bytes",
        changed_reader,
    )
    with pytest.raises(ValueError, match="SHA-256|alias"):
        load_collection_inputs()

    assert inputs.observations


def test_collection_input_hashes_but_never_parses_the_raw_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    original_reader = raw_frames.strict_read_bytes
    raw_reads: list[tuple[Path, str]] = []

    def opaque_raw_reader(path: Path, expected_sha256: str, label: str) -> bytes:
        if path == raw_frames.RAW_SPLIT_PATH:
            raw_reads.append((path, expected_sha256))
            return b"not a gzip or JSON payload"
        return original_reader(path, expected_sha256, label)

    monkeypatch.setattr(raw_frames, "strict_read_bytes", opaque_raw_reader)

    assert len(load_collection_inputs().observations) == 50
    assert raw_reads == [
        (
            raw_frames.RAW_SPLIT_PATH,
            "6140b46759fe332ee96aa849d4bb64e1c1829b8f65acee127355040a6ee23484",
        )
    ]


def test_pinned_oracle_is_hashed_and_parsed_from_one_accepted_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _oracle_bytes()
    source = tmp_path / "observations" / "scene"
    source.mkdir(parents=True)
    artifact = source / "0123456789abcdefabcd.npz"
    artifact.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    observation = _observation(artifact_sha256=expected)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.ORACLE_ARTIFACT_ROOT",
        tmp_path,
    )

    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    original_reader = raw_frames.strict_read_bytes
    original_open = os.open
    oracle_reads = 0
    oracle_opens = 0

    def counted_reader(path: Path, expected_sha256: str, label: str) -> bytes:
        nonlocal oracle_reads
        if label == "oracle artifact":
            oracle_reads += 1
        return original_reader(path, expected_sha256, label)

    def one_oracle_open(
        name: Union[str, bytes, os.PathLike[str], os.PathLike[bytes]],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: Optional[int] = None,
    ) -> int:
        nonlocal oracle_opens
        if str(name) == artifact.name:
            oracle_opens += 1
            if oracle_opens > 1:
                raise AssertionError("oracle artifact reopened")
        return original_open(name, flags, mode, dir_fd=dir_fd)

    from vlnce_baselines.models.etp_llm.llm_grid_evidence import GridEvidence

    def forbidden_core_loader(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("path-reopening core loader was called")

    monkeypatch.setattr(raw_frames, "strict_read_bytes", counted_reader)
    monkeypatch.setattr(os, "open", one_oracle_open)
    monkeypatch.setattr(GridEvidence, "load", forbidden_core_loader)
    parsed = load_pinned_oracle_after_render(observation)

    assert parsed.sha256 == expected
    assert parsed.ego_semantic_grid.dtype == np.bool_
    assert parsed.target_semantic_grid[0, 0, 0]
    assert oracle_reads == 1
    assert oracle_opens == 1
    monkeypatch.setattr(os, "open", original_open)
    artifact.write_bytes(payload + b"drift")
    with pytest.raises(ValueError, match="SHA-256"):
        load_pinned_oracle_after_render(observation)


def test_pinned_oracle_rejects_reordered_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(_oracle_bytes())) as source, zipfile.ZipFile(
        output, "w"
    ) as target:
        for name in reversed(source.namelist()):
            target.writestr(name, source.read(name))
    root = tmp_path / "observations" / "scene"
    root.mkdir(parents=True)
    payload = output.getvalue()
    (root / "0123456789abcdefabcd.npz").write_bytes(payload)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.ORACLE_ARTIFACT_ROOT",
        tmp_path,
    )

    with pytest.raises(ValueError, match="11-member"):
        load_pinned_oracle_after_render(
            _observation(artifact_sha256=hashlib.sha256(payload).hexdigest())
        )


def test_snapshot_scene_bundle_copies_exact_immutable_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    scene = "scene"
    source = source_root / scene
    source.mkdir(parents=True)
    expected = {
        f"{scene}.glb": b"glb",
        f"{scene}.house": b"house",
        f"{scene}_semantic.ply": b"ply",
        f"{scene}.navmesh": b"navmesh",
    }
    for name, payload in expected.items():
        (source / name).write_bytes(payload)
    private_sibling = tmp_path / "private"
    private_sibling.mkdir(mode=0o700)
    private_sibling.chmod(0o700)
    fsyncs: list[int] = []
    opens: list[tuple[str, int]] = []
    original_fsync = os.fsync
    original_open = os.open

    def tracked_fsync(descriptor: int) -> None:
        fsyncs.append(descriptor)
        original_fsync(descriptor)

    def tracked_open(
        name: Union[str, bytes, os.PathLike[str], os.PathLike[bytes]],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: Optional[int] = None,
    ) -> int:
        if str(name) in expected:
            opens.append((str(name), flags))
        return original_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.SCENE_DATASET_ROOT",
        source_root,
    )

    bundle = snapshot_scene_bundle(scene, private_sibling)

    assert tuple(bundle.files) == ("glb", "house", "semantic_ply", "navmesh")
    for role, record in bundle.files.items():
        assert record.path.read_bytes() == expected[record.path.name]
        assert record.path.stat().st_mode & 0o777 == 0o444
        assert record.byte_length == len(expected[record.path.name])
        assert record.sha256 == hashlib.sha256(expected[record.path.name]).hexdigest()
    assert all(
        source.joinpath(name).read_bytes() == payload
        for name, payload in expected.items()
    )
    assert len(fsyncs) >= 6
    assert any(
        flags == os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW
        for _, flags in opens
    )
    assert any(
        flags & os.O_WRONLY and flags & os.O_EXCL and flags & os.O_NOFOLLOW
        for _, flags in opens
    )
    with pytest.raises(FileExistsError):
        snapshot_scene_bundle(scene, private_sibling)


def test_snapshot_rejects_symlink_source_and_private_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source = source_root / "scene"
    source.mkdir(parents=True)
    for name in ("scene.glb", "scene.house", "scene_semantic.ply", "scene.navmesh"):
        (source / name).write_bytes(b"x")
    (source / "scene.glb").unlink()
    (source / "scene.glb").symlink_to(source / "scene.house")
    private_sibling = tmp_path / "private"
    private_sibling.mkdir(mode=0o700)
    private_sibling.chmod(0o700)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.SCENE_DATASET_ROOT",
        source_root,
    )

    with pytest.raises(ValueError, match="regular file"):
        snapshot_scene_bundle("scene", private_sibling)
    with pytest.raises(ValueError, match="scene_id"):
        snapshot_scene_bundle("../escape", private_sibling)


def test_snapshot_rejects_intermediate_symlinks_and_nonregular_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source = source_root / "scene"
    source.mkdir(parents=True)
    for name in ("scene.glb", "scene.house", "scene_semantic.ply", "scene.navmesh"):
        (source / name).write_bytes(b"x")
    private_sibling = tmp_path / "private"
    private_sibling.mkdir(mode=0o700)
    private_sibling.chmod(0o700)
    source_link = tmp_path / "source-link"
    source_link.symlink_to(source_root, target_is_directory=True)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.SCENE_DATASET_ROOT",
        source_link,
    )
    with pytest.raises(ValueError, match="real directory"):
        snapshot_scene_bundle("scene", private_sibling)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.SCENE_DATASET_ROOT",
        source_root,
    )
    private_link = tmp_path / "private-link"
    private_link.symlink_to(private_sibling, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        snapshot_scene_bundle("scene", private_link)
    (source / "scene.glb").unlink()
    os.mkfifo(source / "scene.glb")
    with pytest.raises(ValueError, match="regular file"):
        snapshot_scene_bundle("scene", private_sibling)


def test_snapshot_rejects_shared_or_package_staging_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source = source_root / "scene"
    source.mkdir(parents=True)
    for name in ("scene.glb", "scene.house", "scene_semantic.ply", "scene.navmesh"):
        (source / name).write_bytes(b"x")
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.SCENE_DATASET_ROOT",
        source_root,
    )
    with pytest.raises(ValueError, match="process-private"):
        snapshot_scene_bundle("scene", shared)
    package_root = Path("data/rgbd_segmenter_benchmark/r2r-val-unseen-50-raw-v1")
    with pytest.raises(ValueError, match="package staging"):
        snapshot_scene_bundle("scene", package_root)
    with pytest.raises(ValueError, match="package staging"):
        snapshot_scene_bundle("scene", package_root / "private")


def test_snapshot_checks_destination_file_state_before_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source = source_root / "scene"
    source.mkdir(parents=True)
    for name in ("scene.glb", "scene.house", "scene_semantic.ply", "scene.navmesh"):
        (source / name).write_bytes(b"asset")
    private_sibling = tmp_path / "private"
    private_sibling.mkdir(mode=0o700)
    private_sibling.chmod(0o700)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.SCENE_DATASET_ROOT",
        source_root,
    )
    original_open = os.open
    original_fstat = os.fstat
    destination_descriptor: list[int] = []

    def tracked_open(
        name: Union[str, bytes, os.PathLike[str], os.PathLike[bytes]],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: Optional[int] = None,
    ) -> int:
        descriptor = original_open(name, flags, mode, dir_fd=dir_fd)
        if flags & os.O_WRONLY:
            destination_descriptor.append(descriptor)
        return descriptor

    def changed_destination_state(descriptor: int) -> os.stat_result:
        state = original_fstat(descriptor)
        if descriptor in destination_descriptor:
            values = list(state)
            values[6] += 1
            return os.stat_result(values)
        return state

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "fstat", changed_destination_state)

    with pytest.raises(ValueError, match="copy verification"):
        snapshot_scene_bundle("scene", private_sibling)
    assert not (private_sibling / "scene").exists()


def test_snapshot_detects_source_mutation_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source = source_root / "scene"
    source.mkdir(parents=True)
    for name in ("scene.glb", "scene.house", "scene_semantic.ply", "scene.navmesh"):
        (source / name).write_bytes(b"asset")
    private_sibling = tmp_path / "private"
    private_sibling.mkdir(mode=0o700)
    private_sibling.chmod(0o700)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.SCENE_DATASET_ROOT",
        source_root,
    )
    original_read = os.read
    mutated = False

    def mutating_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        data = original_read(descriptor, size)
        if data == b"asset" and not mutated:
            mutated = True
            (source / "scene.glb").write_bytes(b"mutated-source")
        return data

    monkeypatch.setattr(os, "read", mutating_read)

    with pytest.raises(ValueError, match="changed while copied"):
        snapshot_scene_bundle("scene", private_sibling)
    assert mutated
    assert not (private_sibling / "scene").exists()


def test_snapshot_rolls_back_late_role_failure_and_allows_clean_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source = source_root / "scene"
    source.mkdir(parents=True)
    for name in ("scene.glb", "scene.house", "scene_semantic.ply"):
        (source / name).write_bytes(b"x")
    private_sibling = tmp_path / "private"
    private_sibling.mkdir(mode=0o700)
    private_sibling.chmod(0o700)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.SCENE_DATASET_ROOT",
        source_root,
    )

    with pytest.raises(ValueError, match="regular file"):
        snapshot_scene_bundle("scene", private_sibling)
    assert not (private_sibling / "scene").exists()
    (source / "scene.navmesh").write_bytes(b"x")
    assert snapshot_scene_bundle("scene", private_sibling).files["navmesh"].path.exists()


def test_snapshot_surfaces_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source = source_root / "scene"
    source.mkdir(parents=True)
    for name in ("scene.glb", "scene.house", "scene_semantic.ply"):
        (source / name).write_bytes(b"x")
    private_sibling = tmp_path / "private"
    private_sibling.mkdir(mode=0o700)
    private_sibling.chmod(0o700)
    monkeypatch.setattr(
        "prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames.SCENE_DATASET_ROOT",
        source_root,
    )
    original_unlink = os.unlink

    def failed_unlink(name: str, *, dir_fd: Optional[int] = None) -> None:
        if name == "scene.glb":
            raise OSError("cleanup denied")
        original_unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", failed_unlink)

    with pytest.raises(RuntimeError, match="cleanup"):
        snapshot_scene_bundle("scene", private_sibling)


def test_public_roots_are_the_experiment_pinned_locations() -> None:
    assert ORACLE_ARTIFACT_ROOT == Path(
        "data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen"
    )
    assert SCENE_DATASET_ROOT == Path("data/scene_datasets/mp3d")


def _bundle(tmp_path: Path) -> SceneBundle:
    scene_root = tmp_path / "scene"
    scene_root.mkdir()
    names = {
        "glb": "scene.glb",
        "house": "scene.house",
        "semantic_ply": "scene_semantic.ply",
        "navmesh": "scene.navmesh",
    }
    files = {}
    for role, name in names.items():
        path = scene_root / name
        path.write_bytes(role.encode())
        files[role] = SnapshotFile(
            path=path,
            byte_length=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return SceneBundle(scene_id="scene", files=MappingProxyType(files))


def test_build_scene_simulator_uses_snapshot_and_canonical_36_sensor_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from habitat import sims

    captured = []
    sentinel = object()

    def fake_make_sim(*, id_sim: str, config: object) -> object:
        captured.append((id_sim, config))
        return sentinel

    monkeypatch.setattr(sims, "make_sim", fake_make_sim)
    bundle = _bundle(tmp_path)

    assert build_scene_simulator(bundle) is sentinel

    assert len(captured) == 1
    simulator_type, config = captured[0]
    assert simulator_type == "Sim-v0"
    assert config.SCENE == str(bundle.files["glb"].path)
    assert config.HABITAT_SIM_V0.GPU_DEVICE_ID == 0
    expected_names = [
        f"{kind}_{yaw:03d}"
        for yaw in range(0, 360, 30)
        for kind in ("RGB", "DEPTH", "SEMANTIC")
    ]
    assert config.AGENT_0.SENSORS == expected_names
    for yaw in range(0, 360, 30):
        for kind in ("RGB", "DEPTH", "SEMANTIC"):
            sensor = getattr(config, f"{kind}_{yaw:03d}")
            assert sensor.UUID == f"{kind.lower()}_{yaw:03d}"
            expected_type = {
                "RGB": "HabitatSimRGBSensor",
                "DEPTH": "HabitatSimDepthSensor",
                "SEMANTIC": "HabitatSimSemanticSensor",
            }[kind]
            assert sensor.TYPE == expected_type
            assert sensor.WIDTH == sensor.HEIGHT == 256
            assert sensor.HFOV == 90.0
            assert sensor.POSITION == [0.0, 1.25, 0.0]
            assert sensor.ORIENTATION == [0.0, math.radians(yaw), 0.0]
            if kind == "DEPTH":
                assert sensor.MIN_DEPTH == 0.0
                assert sensor.MAX_DEPTH == 10.0
                assert sensor.NORMALIZE_DEPTH is False


class _Category:
    def __init__(self, index: int) -> None:
        self.value = index

    def index(self, mapping: Optional[str] = None) -> int:
        if mapping is not None:
            assert mapping == "mpcat40"
        return self.value


class _AABB:
    def __init__(self, minimum: tuple[float, float, float]) -> None:
        self.center = np.asarray(minimum, dtype=np.float64) + 0.5
        self.sizes = np.ones(3, dtype=np.float64)


def _semantic_scene() -> SimpleNamespace:
    region = SimpleNamespace(category=_Category(2), aabb=_AABB((-4.0, 1.0, -6.0)))
    objects = [
        None,
        SimpleNamespace(
            id="semantic_1",
            category=_Category(0),
            region=region,
        ),
        SimpleNamespace(
            id="semantic_2",
            category=_Category(1),
            region=None,
        ),
        SimpleNamespace(
            id="semantic_3",
            category=_Category(len(OBJECT_MAPPING)),
            region=SimpleNamespace(category=None),
        ),
    ]
    levels = [
        SimpleNamespace(
            aabb=_AABB((-10.0, 0.0, -20.0)),
            regions=[region],
        ),
        SimpleNamespace(
            aabb=_AABB((30.0, 5.0, 40.0)),
            regions=[],
        ),
    ]
    return SimpleNamespace(objects=objects, levels=levels)


def _quaternion(yaw_degrees: int, *, scale: float = 1.0) -> np.ndarray:
    half = math.radians(yaw_degrees) / 2.0
    return np.asarray(
        [0.0, math.sin(half) * scale, 0.0, math.cos(half) * scale],
        dtype=np.float64,
    )


def _fake_simulator(
    *,
    scene: Optional[SimpleNamespace] = None,
    mutate: Optional[Callable[[dict[str, np.ndarray], SimpleNamespace], None]] = None,
) -> SimpleNamespace:
    observations: dict[str, np.ndarray] = {}
    sensor_states = {}
    start = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    for index, yaw in enumerate(range(0, 360, 30)):
        rgb = np.full((256, 256, 3), index, dtype=np.uint8)
        depth = np.zeros((256, 256), dtype=np.float32)
        depth[index, index] = np.float32(index + 1) / np.float32(2)
        semantic = np.full((256, 256), 1 + index % 4, dtype=np.int32)
        observations[f"rgb_{yaw:03d}"] = rgb
        observations[f"depth_{yaw:03d}"] = (
            depth[..., None] if index % 2 else depth
        )
        observations[f"semantic_{yaw:03d}"] = semantic
        position = start + np.asarray([index / 10.0, 1.25, -index / 20.0])
        rotation = _quaternion(yaw, scale=1.0 + 4e-8)
        for kind in ("rgb", "depth", "semantic"):
            sensor_states[f"{kind}_{yaw:03d}"] = SimpleNamespace(
                position=position.copy(),
                rotation=rotation.copy(),
            )
    state = SimpleNamespace(
        position=start.copy(),
        rotation=_quaternion(0, scale=1.0 + 4e-8),
        sensor_states=sensor_states,
    )
    if mutate is not None:
        mutate(observations, state)
    calls = []

    def get_observations_at(
        *,
        position: list[float],
        rotation: list[float],
        keep_agent_at_new_pose: bool,
    ) -> dict[str, np.ndarray]:
        calls.append((position, rotation, keep_agent_at_new_pose))
        return observations

    closed = []
    return SimpleNamespace(
        close=lambda: closed.append(True),
        closed=closed,
        get_agent_state=lambda: state,
        get_observations_at=get_observations_at,
        observation_calls=calls,
        semantic_annotations=lambda: scene or _semantic_scene(),
    )


def _rendered_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RawFrameArrays, SimpleNamespace, list[tuple[OracleSensorFrame, ...]]]:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    simulator = _fake_simulator()
    projected_frames = []
    real_projector = project_oracle_frames

    def projector_spy(
        frames: list[OracleSensorFrame] | tuple[OracleSensorFrame, ...],
        *,
        start_position: Sequence[float],
        start_rotation: Sequence[float],
        target_origin_xz: Sequence[float],
    ) -> object:
        projected_frames.append(tuple(frames))
        return real_projector(
            frames,
            start_position=start_position,
            start_rotation=start_rotation,
            target_origin_xz=target_origin_xz,
        )

    monkeypatch.setattr(raw_frames, "project_oracle_frames", projector_spy)
    arrays = render_raw_frame_artifact(
        simulator,
        _observation(artifact_sha256="0" * 64),
    )
    return arrays, simulator, projected_frames


def test_render_uses_exact_modalities_mapping_origin_and_authoritative_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays, simulator, projected_frames = _rendered_arrays(monkeypatch)

    assert simulator.observation_calls == [
        ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], True)
    ]
    assert arrays.rgb.dtype == np.uint8
    assert [arrays.rgb[index, 0, 0, 0] for index in range(12)] == list(range(12))
    assert arrays.depth_m.dtype == np.float32
    assert arrays.object_categories.dtype == np.int16
    assert arrays.region_categories.dtype == np.int16
    assert arrays.object_categories[0, 0, 0] == 0
    assert arrays.object_categories[1, 0, 0] == OBJECT_MAPPING[1]
    assert arrays.object_categories[2, 0, 0] == MAPPED_OBJECT_NAMES.index("other")
    assert arrays.object_categories[3, 0, 0] == -1
    assert arrays.region_categories[0, 0, 0] == REGION_MAPPING[2]
    assert arrays.region_categories[1, 0, 0] == -1
    assert np.array_equal(
        arrays.sensor_positions[:, 0],
        np.asarray([1.0 + index / 10.0 for index in range(12)]),
    )
    assert np.array_equal(arrays.sensor_rotations_xyzw[1], _quaternion(30))
    assert np.array_equal(arrays.target_origin_xz, np.asarray([-10.0, -20.0]))
    assert len(projected_frames) == 1
    assert [frame.depth_m[frame_index, frame_index] for frame_index, frame in enumerate(projected_frames[0])] == [
        np.float32(index + 1) / np.float32(2) for index in range(12)
    ]
    assert [
        frame.sensor_position for frame in projected_frames[0]
    ] == [tuple(position) for position in arrays.sensor_positions]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda observations, _state: observations.__setitem__(
                "rgb_000", np.zeros((256, 256, 4), dtype=np.uint8)
            ),
            "rgb_000",
        ),
        (
            lambda observations, _state: observations.__setitem__(
                "rgb_000", np.zeros((256, 256, 3), dtype=np.float32)
            ),
            "rgb_000",
        ),
        (
            lambda observations, _state: observations.__setitem__(
                "depth_000", np.zeros((256, 256, 1, 1), dtype=np.float32)
            ),
            "depth_000",
        ),
        (
            lambda observations, _state: observations.__setitem__(
                "depth_000", np.zeros((256, 256), dtype=np.float64)
            ),
            "depth_000",
        ),
        (
            lambda observations, _state: observations["depth_000"].__setitem__(
                (0, 0), np.nan
            ),
            "depth_000",
        ),
        (
            lambda observations, _state: observations["depth_000"].__setitem__(
                (0, 0), -0.1
            ),
            "depth_000",
        ),
        (
            lambda observations, _state: observations["depth_000"].__setitem__(
                (0, 0), 10.1
            ),
            "depth_000",
        ),
        (
            lambda observations, _state: observations.__setitem__(
                "semantic_000", np.zeros((256, 256), dtype=np.float32)
            ),
            "semantic_000",
        ),
        (
            lambda observations, _state: observations.__setitem__(
                "semantic_000", np.zeros((256, 256, 1), dtype=np.int32)
            ),
            "semantic_000",
        ),
    ],
)
def test_render_rejects_modality_coercion(
    mutation: Callable[[dict[str, np.ndarray], SimpleNamespace], None],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        render_raw_frame_artifact(
            _fake_simulator(mutate=mutation),
            _observation(artifact_sha256="0" * 64),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda _observations, state: setattr(
            state.sensor_states["rgb_000"], "position", np.asarray([9.0, 9.0, 9.0])
        ),
        lambda _observations, state: setattr(
            state.sensor_states["semantic_000"], "rotation", _quaternion(30)
        ),
        lambda _observations, state: setattr(
            state.sensor_states["depth_000"],
            "rotation",
            _quaternion(0, scale=1.0 + 6e-8),
        ),
        lambda _observations, state: setattr(
            state.sensor_states["depth_000"],
            "rotation",
            _quaternion(0).astype(np.float32),
        ),
        lambda _observations, state: setattr(
            state.sensor_states["depth_000"],
            "rotation",
            np.asarray([0.0, np.nan, 0.0, 1.0], dtype=np.float64),
        ),
        lambda _observations, state: setattr(
            state, "position", np.asarray([1.0, 2.0, 3.0 + 1e-15])
        ),
        lambda _observations, state: setattr(state, "rotation", _quaternion(30)),
    ],
)
def test_render_rejects_pose_drift(
    mutation: Callable[[dict[str, np.ndarray], SimpleNamespace], None],
) -> None:
    with pytest.raises(ValueError, match="pose|position|rotation|quaternion|extrinsics"):
        render_raw_frame_artifact(
            _fake_simulator(mutate=mutation),
            _observation(artifact_sha256="0" * 64),
        )


def test_render_accepts_sign_equivalent_agent_rotation_and_norm_boundary() -> None:
    def mutate(_observations: dict[str, np.ndarray], state: SimpleNamespace) -> None:
        state.rotation = -_quaternion(0)
        for kind in ("rgb", "depth", "semantic"):
            state.sensor_states[f"{kind}_000"].rotation = _quaternion(
                0, scale=1.0 + 5e-8
            )

    arrays = render_raw_frame_artifact(
        _fake_simulator(mutate=mutate),
        _observation(artifact_sha256="0" * 64),
    )

    assert np.array_equal(arrays.sensor_rotations_xyzw[0], _quaternion(0))


def test_render_rejects_bad_semantic_suffix_and_region_mapping() -> None:
    bad_suffix = _semantic_scene()
    bad_suffix.objects.append(
        SimpleNamespace(id="not_an_integer", category=_Category(1), region=None)
    )
    with pytest.raises(ValueError, match="semantic object ID"):
        render_raw_frame_artifact(
            _fake_simulator(scene=bad_suffix),
            _observation(artifact_sha256="0" * 64),
        )

    bad_region = _semantic_scene()
    bad_region.objects[1].region.category = _Category(len(REGION_MAPPING))
    with pytest.raises(ValueError, match="region category"):
        render_raw_frame_artifact(
            _fake_simulator(scene=bad_region),
            _observation(artifact_sha256="0" * 64),
        )


def test_level_origin_selects_below_and_between_floors_and_rejects_nan() -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    scene = _semantic_scene()
    assert raw_frames._target_origin_at_start(scene, -5.0) == (-10.0, -20.0)
    assert raw_frames._target_origin_at_start(scene, 4.0) == (-10.0, -20.0)
    assert raw_frames._target_origin_at_start(scene, 6.0) == (30.0, 40.0)
    scene.levels[0].aabb.center[0] = np.nan
    with pytest.raises(ValueError, match="AABB"):
        raw_frames._target_origin_at_start(scene, 2.0)


def _pinned_from_arrays(arrays: RawFrameArrays) -> PinnedOracle:
    frames = tuple(
        OracleSensorFrame(
            depth_m=arrays.depth_m[index],
            object_categories=arrays.object_categories[index],
            region_categories=arrays.region_categories[index],
            sensor_position=tuple(arrays.sensor_positions[index]),
            sensor_rotation=tuple(arrays.sensor_rotations_xyzw[index]),
        )
        for index in range(12)
    )
    evidence = project_oracle_frames(
        frames,
        start_position=tuple(float(value) for value in arrays.start_position),
        start_rotation=tuple(float(value) for value in arrays.start_rotation_xyzw),
        target_origin_xz=tuple(float(value) for value in arrays.target_origin_xz),
    )
    return PinnedOracle(
        sha256="0" * 64,
        ego_semantic_grid=evidence.ego_semantic_grid,
        ego_observed_mask=evidence.ego_observed_mask,
        ego_free_mask=evidence.ego_free_mask,
        target_semantic_grid=evidence.target_semantic_grid,
        target_observed_mask=evidence.target_observed_mask,
        target_free_mask=evidence.target_free_mask,
        start_position=tuple(np.asarray(evidence.start_position, dtype=np.float32)),
        start_direction=tuple(np.asarray(evidence.start_direction, dtype=np.float32)),
    )


def test_replay_uses_12_public_frames_once_and_requires_all_six_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    arrays, _, _ = _rendered_arrays(monkeypatch)
    oracle = _pinned_from_arrays(arrays)
    calls = []
    real_projector = project_oracle_frames

    def projector_spy(
        frames: list[OracleSensorFrame] | tuple[OracleSensorFrame, ...],
        *,
        start_position: Sequence[float],
        start_rotation: Sequence[float],
        target_origin_xz: Sequence[float],
    ) -> object:
        calls.append(tuple(frames))
        return real_projector(
            frames,
            start_position=start_position,
            start_rotation=start_rotation,
            target_origin_xz=target_origin_xz,
        )

    monkeypatch.setattr(raw_frames, "project_oracle_frames", projector_spy)

    replay_and_require_exact(
        arrays,
        _observation(artifact_sha256="0" * 64),
        oracle,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 12
    assert [frame.depth_m[index, index] for index, frame in enumerate(calls[0])] == [
        np.float32(index + 1) / np.float32(2) for index in range(12)
    ]
    for field in (
        "ego_semantic_grid",
        "ego_observed_mask",
        "ego_free_mask",
        "target_semantic_grid",
        "target_observed_mask",
        "target_free_mask",
    ):
        changed = np.array(getattr(oracle, field), copy=True)
        changed.flat[0] = ~changed.flat[0]
        changed_oracle = {
            "ego_semantic_grid": replace(oracle, ego_semantic_grid=changed),
            "ego_observed_mask": replace(oracle, ego_observed_mask=changed),
            "ego_free_mask": replace(oracle, ego_free_mask=changed),
            "target_semantic_grid": replace(oracle, target_semantic_grid=changed),
            "target_observed_mask": replace(oracle, target_observed_mask=changed),
            "target_free_mask": replace(oracle, target_free_mask=changed),
        }[field]
        with pytest.raises(ValueError, match=field):
            replay_and_require_exact(
                arrays,
                _observation(artifact_sha256="0" * 64),
                changed_oracle,
            )


def test_replay_requires_exact_stored_pose_and_float32_oracle_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays, _, _ = _rendered_arrays(monkeypatch)
    with pytest.raises(ValueError, match="stored start pose"):
        replay_and_require_exact(
            replace(
                arrays,
                start_position=arrays.start_position
                + np.asarray([1e-15, 0.0, 0.0], dtype=np.float64),
            ),
            _observation(artifact_sha256="0" * 64),
            _pinned_from_arrays(arrays),
        )
    changed_origin = arrays.target_origin_xz + np.asarray(
        [0.123456789, 0.987654321], dtype=np.float64
    )
    changed = replace(arrays, target_origin_xz=changed_origin)
    oracle = _pinned_from_arrays(changed)

    replay_and_require_exact(
        changed,
        _observation(artifact_sha256="0" * 64),
        oracle,
    )

    raw_target_start = changed.start_position[[0, 2]] - changed_origin
    assert not np.array_equal(
        raw_target_start,
        np.asarray(oracle.start_position, dtype=np.float64),
    )
    assert np.array_equal(
        raw_target_start.astype(np.float32),
        np.asarray(oracle.start_position, dtype=np.float32),
    )


def test_first_row_smoke_orders_events_closes_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames
    from prior.analyze.d2026_07_29 import rgbd_segmenter_raw_frame_package as package

    smoke_root = tmp_path / "smoke"
    final_root = tmp_path / "final"
    snapshot_source = tmp_path / "snapshot-source"
    snapshot_source.mkdir()
    simulator = _fake_simulator()
    observation = _observation(artifact_sha256="0" * 64)
    arrays = render_raw_frame_artifact(simulator, observation)
    oracle = _pinned_from_arrays(arrays)
    simulator = _fake_simulator()
    events = []
    snapshot_roots = []
    reloaded_buffers = []

    def record(name: str, function: Callable[..., object]) -> Callable[..., object]:
        def wrapped(*args: object, **kwargs: object) -> object:
            events.append(name)
            return function(*args, **kwargs)

        return wrapped

    def fake_snapshot(_scene: str, root: Path) -> SceneBundle:
        snapshot_roots.append(root)
        (root / "snapshot-marker").write_bytes(b"snapshot")
        return _bundle(snapshot_source)

    monkeypatch.setattr(raw_frames, "SMOKE_PACKAGE_ROOT", smoke_root)
    monkeypatch.setattr(raw_frames, "FINAL_PACKAGE_ROOT", final_root)
    monkeypatch.setattr(
        raw_frames,
        "load_collection_inputs",
        record(
            "metadata",
            lambda: SimpleNamespace(observations=(observation,), scenes=("scene",)),
        ),
    )
    monkeypatch.setattr(
        raw_frames,
        "snapshot_scene_bundle",
        record("snapshot", fake_snapshot),
    )
    monkeypatch.setattr(
        raw_frames,
        "build_scene_simulator",
        record("simulator", lambda _bundle: simulator),
    )
    monkeypatch.setattr(
        raw_frames,
        "render_raw_frame_artifact",
        record("render", raw_frames.render_raw_frame_artifact),
    )
    monkeypatch.setattr(
        raw_frames,
        "encode_raw_frame_npz",
        record("encode", package.encode_raw_frame_npz),
    )
    original_write = Path.write_bytes

    def tracked_write(path: Path, data: bytes) -> int:
        if path.suffix == ".npz":
            events.append("write")
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", tracked_write)
    def tracked_reload(path: Path, expected_sha256: str, label: str) -> bytes:
        events.append("reload")
        accepted = package.strict_read_bytes(path, expected_sha256, label)
        reloaded_buffers.append(accepted)
        return accepted

    def tracked_parse(
        data: bytes,
        *,
        expected_members: Mapping[str, MemberMetadata] | None = None,
        expected_npz: FileRecord | None = None,
    ) -> object:
        events.append("parse")
        assert data is reloaded_buffers[0]
        return package.parse_raw_frame_npz_bytes(
            data,
            expected_members=expected_members,
            expected_npz=expected_npz,
        )

    monkeypatch.setattr(raw_frames, "strict_read_bytes", tracked_reload)
    monkeypatch.setattr(raw_frames, "parse_raw_frame_npz_bytes", tracked_parse)
    monkeypatch.setattr(
        raw_frames,
        "load_pinned_oracle_after_render",
        record("oracle", lambda _observation: oracle),
    )
    monkeypatch.setattr(
        raw_frames,
        "replay_and_require_exact",
        record("replay", raw_frames.replay_and_require_exact),
    )

    report = run_first_row_smoke()

    assert events == [
        "metadata",
        "snapshot",
        "simulator",
        "render",
        "encode",
        "write",
        "reload",
        "parse",
        "oracle",
        "replay",
    ]
    assert simulator.closed == [True]
    assert report == (
        b'{"control":"PASS","observation_id":"0123456789abcdefabcd",'
        b'"replay":"PASS","scene_id":"scene","schema_version":1}\n'
    )
    assert capsys.readouterr().out.encode() == report
    assert not smoke_root.exists()
    assert len(snapshot_roots) == 1
    assert not snapshot_roots[0].exists()
    assert not final_root.exists()


@pytest.mark.parametrize(
    ("failure_event", "simulator_exists"),
    [
        ("snapshot", False),
        ("simulator", False),
        ("render", True),
        ("encode", True),
        ("write", True),
        ("reload", True),
        ("parse", True),
        ("oracle", True),
        ("replay", True),
    ],
)
def test_first_row_smoke_closes_once_on_each_failure(
    failure_event: str,
    simulator_exists: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    smoke_root = tmp_path / "smoke"
    source = tmp_path / "source"
    source.mkdir()
    simulator = _fake_simulator()
    observation = _observation(artifact_sha256="0" * 64)
    arrays = render_raw_frame_artifact(_fake_simulator(), observation)
    oracle = _pinned_from_arrays(arrays)
    snapshot_roots = []
    monkeypatch.setattr(raw_frames, "SMOKE_PACKAGE_ROOT", smoke_root)
    monkeypatch.setattr(
        raw_frames,
        "load_collection_inputs",
        lambda: SimpleNamespace(observations=(observation,), scenes=("scene",)),
    )
    def fake_snapshot(_scene: str, root: Path) -> SceneBundle:
        snapshot_roots.append(root)
        (root / "snapshot-marker").write_bytes(b"snapshot")
        return _bundle(source)

    monkeypatch.setattr(raw_frames, "snapshot_scene_bundle", fake_snapshot)
    monkeypatch.setattr(raw_frames, "build_scene_simulator", lambda _bundle: simulator)
    monkeypatch.setattr(
        raw_frames,
        "load_pinned_oracle_after_render",
        lambda _observation: oracle,
    )
    target = {
        "render": "render_raw_frame_artifact",
        "encode": "encode_raw_frame_npz",
        "parse": "parse_raw_frame_npz_bytes",
        "oracle": "load_pinned_oracle_after_render",
        "replay": "replay_and_require_exact",
    }.get(failure_event)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(failure_event)

    if failure_event == "snapshot":
        def fail_snapshot(_scene: str, root: Path) -> None:
            snapshot_roots.append(root)
            (root / "snapshot-marker").write_bytes(b"snapshot")
            fail()

        monkeypatch.setattr(raw_frames, "snapshot_scene_bundle", fail_snapshot)
    elif failure_event == "simulator":
        monkeypatch.setattr(raw_frames, "build_scene_simulator", fail)
    elif failure_event == "write":
        original_write = Path.write_bytes

        def fail_npz_write(path: Path, data: bytes) -> int:
            if path.suffix == ".npz":
                fail()
            return original_write(path, data)

        monkeypatch.setattr(Path, "write_bytes", fail_npz_write)
    elif failure_event == "reload":
        monkeypatch.setattr(raw_frames, "strict_read_bytes", fail)
    else:
        assert target is not None
        monkeypatch.setattr(raw_frames, target, fail)

    with pytest.raises(RuntimeError, match=failure_event):
        run_first_row_smoke()

    assert simulator.closed == ([True] if simulator_exists else [])
    assert not smoke_root.exists()
    assert all(not root.exists() for root in snapshot_roots)


def test_first_row_smoke_cleans_first_directory_when_second_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    smoke_root = tmp_path / "smoke"
    created = []
    real_mkdtemp = tempfile.mkdtemp

    def fail_second_mkdtemp(
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[Union[str, os.PathLike[str]]] = None,
    ) -> str:
        if created:
            raise OSError("second creation")
        path = real_mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        created.append(Path(path))
        return path

    monkeypatch.setattr(raw_frames, "SMOKE_PACKAGE_ROOT", smoke_root)
    monkeypatch.setattr(
        raw_frames,
        "load_collection_inputs",
        lambda: SimpleNamespace(
            observations=(_observation(artifact_sha256="0" * 64),),
            scenes=("scene",),
        ),
    )
    monkeypatch.setattr(raw_frames.tempfile, "mkdtemp", fail_second_mkdtemp)

    with pytest.raises(OSError, match="second creation"):
        run_first_row_smoke()

    assert len(created) == 1
    assert not created[0].exists()
    assert not smoke_root.exists()


def test_smoke_cleanup_attempts_snapshot_removal_after_staging_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    smoke_root = tmp_path / "smoke"
    staging = smoke_root / "staging"
    snapshots = tmp_path / "snapshots"
    staging.mkdir(parents=True)
    snapshots.mkdir()
    removals = []
    original_rmtree = shutil.rmtree

    def fail_first(path: Path) -> None:
        removals.append(path)
        if path == staging:
            raise OSError("staging cleanup")
        original_rmtree(path)

    monkeypatch.setattr(raw_frames, "SMOKE_PACKAGE_ROOT", smoke_root)
    monkeypatch.setattr(raw_frames.shutil, "rmtree", fail_first)

    with pytest.raises(RuntimeError, match="cleanup"):
        raw_frames._cleanup_smoke_paths(staging, snapshots)

    assert removals == [staging, snapshots]
    assert not snapshots.exists()
