from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
import time
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Callable, Mapping, Optional, Sequence, Union
import zipfile

import numpy as np
import pytest

from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    FileRecord,
    IndexRow,
    MemberMetadata,
    RawFrameArrays,
    SourceRecord,
    canonical_index_bytes,
    encode_raw_frame_npz,
    strict_read_bytes,
    validate_raw_frame_directory,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames import (
    ORACLE_ARTIFACT_ROOT,
    SCENE_DATASET_ROOT,
    CollectionObservation,
    EnvironmentCapture,
    InstalledDistribution,
    ManifestSources,
    PinnedOracle,
    SceneAssetCommitment,
    SceneAssetFile,
    SceneBundle,
    SnapshotFile,
    build_manifest,
    build_scene_simulator,
    collect_attempt,
    load_collection_inputs,
    load_pinned_oracle_after_render,
    render_raw_frame_artifact,
    replay_and_require_exact,
    run_first_per_scene_smoke,
    run_first_row_smoke,
    snapshot_scene_bundle,
)
from prior.constants import MAPPED_OBJECT_NAMES, OBJECT_MAPPING, REGION_MAPPING
from vlnce_baselines.models.etp_llm.llm_grid_oracle_cache import (
    OracleSensorFrame,
    project_oracle_frames,
)


def _abrupt_spawn_worker(*_args: object) -> None:
    os._exit(17)


def _large_array_spawn_worker(connection: Connection, *_args: object) -> None:
    small = np.asarray(0, dtype=np.uint8)
    connection.send(
        (
            "success",
            RawFrameArrays(
                schema_version=small,
                rgb=np.zeros(10 * 1024 * 1024, dtype=np.uint8),
                depth_m=small,
                object_categories=small,
                region_categories=small,
                sensor_positions=small,
                sensor_rotations_xyzw=small,
                sensor_yaw_degrees=small,
                sensor_hfov_degrees=small,
                sensor_position_relative=small,
                start_position=small,
                start_rotation_xyzw=small,
                target_origin_xz=small,
                ego_observed_mask=small,
                ego_free_mask=small,
                target_observed_mask=small,
                target_free_mask=small,
            ),
        )
    )
    connection.close()


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
        cohort_row_sha256="1" * 64,
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
    sentinel = SimpleNamespace()

    def fake_make_sim(*, id_sim: str, config: object) -> object:
        captured.append((id_sim, config))
        agent = getattr(config, "AGENT_0")
        sensor_types = {
            "HabitatSimRGBSensor": "COLOR",
            "HabitatSimDepthSensor": "DEPTH",
            "HabitatSimSemanticSensor": "SEMANTIC",
        }
        specifications = [
            SimpleNamespace(
                uuid=getattr(config, name).UUID,
                sensor_type=SimpleNamespace(
                    name=sensor_types[getattr(config, name).TYPE]
                ),
                sensor_subtype=SimpleNamespace(name="PINHOLE"),
                resolution=np.asarray(
                    [getattr(config, name).HEIGHT, getattr(config, name).WIDTH],
                    dtype=np.int32,
                ),
                position=np.asarray(getattr(config, name).POSITION, dtype=np.float32),
                orientation=np.asarray(
                    getattr(config, name).ORIENTATION, dtype=np.float32
                ),
                parameters={"hfov": str(getattr(config, name).HFOV)},
            )
            for name in agent.SENSORS
        ]
        sentinel.sim_config = SimpleNamespace(
            agents=[SimpleNamespace(sensor_specifications=specifications)]
        )
        sentinel.sensor_suite = SimpleNamespace(
            sensors={
                getattr(config, name).UUID: SimpleNamespace(
                    config=SimpleNamespace(
                        MIN_DEPTH=getattr(config, name).MIN_DEPTH,
                        MAX_DEPTH=getattr(config, name).MAX_DEPTH,
                        NORMALIZE_DEPTH=getattr(config, name).NORMALIZE_DEPTH,
                    )
                )
                for name in agent.SENSORS
                if getattr(config, name).TYPE == "HabitatSimDepthSensor"
            }
        )
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


@pytest.mark.parametrize("drift", ["type", "subtype"])
def test_build_scene_simulator_rejects_resolved_sensor_type_drift(
    drift: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import habitat
    import habitat_sim

    config = habitat.get_config()
    config.defrost()
    if drift == "type":
        config.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimDepthSensor"
        monkeypatch.setattr(habitat, "get_config", lambda: config)
    else:
        monkeypatch.setattr(
            habitat_sim,
            "SensorSpec",
            lambda: SimpleNamespace(sensor_subtype=object()),
        )

    with pytest.raises(ValueError, match="sensor (name and type|subtype)"):
        build_scene_simulator(_bundle(tmp_path))


@pytest.mark.parametrize(
    "drift",
    [
        "resolution",
        "hfov",
        "position",
        "orientation",
        "min_depth_m",
        "max_depth_m",
        "normalize_depth",
    ],
)
def test_build_scene_simulator_rejects_post_construction_sensor_spec_drift(
    drift: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from habitat import sims

    closed = []

    def fake_make_sim(*, id_sim: str, config: object) -> object:
        del id_sim
        agent = getattr(config, "AGENT_0")
        sensor_types = {
            "HabitatSimRGBSensor": "COLOR",
            "HabitatSimDepthSensor": "DEPTH",
            "HabitatSimSemanticSensor": "SEMANTIC",
        }
        specifications = [
            SimpleNamespace(
                uuid=getattr(config, name).UUID,
                sensor_type=SimpleNamespace(
                    name=sensor_types[getattr(config, name).TYPE]
                ),
                sensor_subtype=SimpleNamespace(name="PINHOLE"),
                resolution=np.asarray(
                    [getattr(config, name).HEIGHT, getattr(config, name).WIDTH],
                    dtype=np.int32,
                ),
                position=np.asarray(getattr(config, name).POSITION, dtype=np.float32),
                orientation=np.asarray(
                    getattr(config, name).ORIENTATION, dtype=np.float32
                ),
                parameters={"hfov": str(getattr(config, name).HFOV)},
            )
            for name in agent.SENSORS
        ]
        sensors = {
            getattr(config, name).UUID: SimpleNamespace(
                config=SimpleNamespace(
                    MIN_DEPTH=getattr(config, name).MIN_DEPTH,
                    MAX_DEPTH=getattr(config, name).MAX_DEPTH,
                    NORMALIZE_DEPTH=getattr(config, name).NORMALIZE_DEPTH,
                )
            )
            for name in agent.SENSORS
            if getattr(config, name).TYPE == "HabitatSimDepthSensor"
        }
        specification = specifications[0]
        if drift == "resolution":
            specification.resolution[0] = 257
        elif drift == "hfov":
            specification.parameters["hfov"] = "91.0"
        elif drift == "position":
            specification.position[1] = 1.5
        elif drift == "orientation":
            specification.orientation[1] = 0.1
        else:
            depth = sensors["depth_000"].config
            setattr(
                depth,
                {
                    "min_depth_m": "MIN_DEPTH",
                    "max_depth_m": "MAX_DEPTH",
                    "normalize_depth": "NORMALIZE_DEPTH",
                }[drift],
                {
                    "min_depth_m": 0.1,
                    "max_depth_m": 9.0,
                    "normalize_depth": True,
                }[drift],
            )
        return SimpleNamespace(
            close=lambda: closed.append(True),
            sensor_suite=SimpleNamespace(sensors=sensors),
            sim_config=SimpleNamespace(
                agents=[SimpleNamespace(sensor_specifications=specifications)]
            ),
        )

    monkeypatch.setattr(sims, "make_sim", fake_make_sim)

    with pytest.raises(ValueError, match="simulator (sensor|depth)"):
        build_scene_simulator(_bundle(tmp_path))

    assert closed == [True]


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
            _quaternion(
                0,
                scale=float(
                    np.nextafter(
                        np.float64(1.0 + 2**-23),
                        np.float64(np.inf),
                    ),
                ),
            ),
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


def test_render_accepts_sign_equivalent_agent_and_habitat_float32_precision() -> None:
    def mutate(_observations: dict[str, np.ndarray], state: SimpleNamespace) -> None:
        state.rotation = -_quaternion(0)
        for kind in ("rgb", "depth", "semantic"):
            state.sensor_states[f"{kind}_000"].rotation = _quaternion(
                0, scale=1.0 + 7.99814713e-8
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


def _source(path: str, marker: str) -> SourceRecord:
    return SourceRecord(path=path, data=marker.encode())


def _manifest_sources() -> ManifestSources:
    return ManifestSources(
        collector_source=_source(
            "prior/analyze/d2026_07_29/rgbd_segmenter_raw_frames.py", "collector"
        ),
        package_source=_source(
            "prior/analyze/d2026_07_29/rgbd_segmenter_raw_frame_package.py",
            "package",
        ),
        asset_roles=_source(
            "prior/analyze/d2026_07_29/rgbd_segmenter_asset_roles.json",
            '{\n'
            '  "algorithm": "single-role-omission-first-row-per-scene-v1",\n'
            '  "auxiliary": [\n'
            '    "navmesh"\n'
            "  ],\n"
            '  "required": [\n'
            '    "glb",\n'
            '    "house",\n'
            '    "semantic_ply"\n'
            "  ],\n"
            '  "schema_version": 1\n'
            "}\n",
        ),
        cohort_manifest=_source(
            "data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1/manifest.json",
            "cohort manifest",
        ),
        cohort_jsonl=_source(
            "data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1/cohort.jsonl",
            "cohort rows",
        ),
        evidence_manifest=_source(
            "data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen/manifest.json",
            "evidence manifest",
        ),
        evidence_index=_source(
            "data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen/index.jsonl",
            "evidence index",
        ),
        raw_split=_source(
            "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/"
            "val_unseen.json.gz",
            "raw split",
        ),
        projector_source=_source(
            "vlnce_baselines/models/etp_llm/llm_grid_oracle_cache.py",
            "projector",
        ),
        mapping_source=_source("prior/constants.py", "mapping"),
    )


def _environment() -> EnvironmentCapture:
    return EnvironmentCapture(
        python_version="3.8.20",
        python_implementation="CPython",
        platform="Linux-test",
        numpy_version="1.24.3",
        zlib_version="1.2.13",
        habitat_version="0.1.7",
        habitat_sim_version="0.1.7",
        cuda_runtime_version="11.8",
        nvidia_driver_version="535.0",
        gpu_name="Fake GPU",
        gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
        gpu_device_id=0,
        installed_distributions=(
            InstalledDistribution(name="habitat-lab", version="0.1.7"),
            InstalledDistribution(name="numpy", version="1.24.3"),
        ),
    )


def _manifest_rows() -> tuple[IndexRow, ...]:
    encoded = encode_raw_frame_npz(
        render_raw_frame_artifact(
            _fake_simulator(),
            _observation(artifact_sha256="0" * 64),
        )
    )
    record = FileRecord(
        byte_length=len(encoded.data),
        sha256=hashlib.sha256(encoded.data).hexdigest(),
    )
    scenes = tuple(
        scene
        for scene, count in zip(
            (f"scene-{index:02d}" for index in range(11)),
            (5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4),
        )
        for _ in range(count)
    )
    return tuple(
        IndexRow(
            artifact=f"observations/{scenes[ordinal]}/"
            f"{ordinal:02d}-{ordinal:020x}.npz",
            cohort_row_sha256=f"{ordinal + 1:064x}",
            members=encoded.members,
            npz=record,
            observation_id=f"{ordinal:020x}",
            oracle_artifact_sha256=f"{ordinal + 101:064x}",
            ordinal=ordinal,
            scene_id=scenes[ordinal],
        )
        for ordinal in range(50)
    )


def _scene_assets() -> Mapping[str, SceneAssetCommitment]:
    result = {}
    for scene_index in range(11):
        scene = f"scene-{scene_index:02d}"
        files = {}
        for role, suffix, required in (
            ("glb", ".glb", True),
            ("house", ".house", True),
            ("navmesh", ".navmesh", False),
            ("semantic_ply", "_semantic.ply", True),
        ):
            payload = f"{scene}:{role}".encode()
            files[role] = SceneAssetFile(
                path=f"{scene}/{scene}{suffix}",
                required=required,
                byte_length=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        result[scene] = SceneAssetCommitment.from_files(files)
    return MappingProxyType(result)


def test_manifest_is_canonical_and_binds_every_exact_commitment() -> None:
    rows = _manifest_rows()
    index = canonical_index_bytes(rows)
    manifest_bytes = build_manifest(
        git_commit="a" * 40,
        sources=_manifest_sources(),
        scene_assets=_scene_assets(),
        environment=_environment(),
        index_bytes=index,
        rows=rows,
    )

    assert manifest_bytes.endswith(b"\n")
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == (
        json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    )
    assert set(manifest) == {
        "schema_version",
        "collection_id",
        "collection",
        "cohort",
        "evidence",
        "sensor",
        "scene_assets",
        "environment",
        "replay",
        "files",
    }
    assert manifest["schema_version"] == 1
    assert manifest["collection_id"] == "r2r-val-unseen-50-raw-v1"
    assert set(manifest["collection"]) == {
        "git_commit",
        "collector_source",
        "package_source",
        "asset_roles",
        "command",
        "gpu_device_id",
    }
    assert manifest["collection"]["command"] == (
        "python -m prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames"
    )
    assert manifest["collection"]["gpu_device_id"] == 0
    assert manifest["collection"]["collector_source"] == {
        "path": "prior/analyze/d2026_07_29/rgbd_segmenter_raw_frames.py",
        "byte_length": 9,
        "sha256": "0736fd5b7cc7ab7dfe821d3a17f93f2634497770232486155c9c881321c4d22c",
    }
    assert manifest["cohort"] == {
        "cohort_id": "r2r-val-unseen-50-v1",
        "directory": "data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1",
        "manifest_sha256": (
            "d71f04f102d80df3799e5fea76162147ad76c060c82ccbca88813d8b14a0b191"
        ),
        "cohort_jsonl_sha256": (
            "89ae70f3e489fa702c66110f9bd9e7666ba0e16a9fbc3ac20aaa91a95adef0ce"
        ),
        "selection_sha256": (
            "32a7adddf32291f059eb1045e63e077a693ca64d6fdf6e7418b54dd5d3644cb2"
        ),
        "sealing_git_commit": "80780e5a8a3736dc666b8dd861c00ce55f4685d8",
        "observation_count": 50,
        "scene_count": 11,
    }
    assert set(manifest["evidence"]) == {
        "root",
        "evidence_key",
        "dataset",
        "split",
        "manifest",
        "index",
        "raw_split",
        "projector_source",
        "mapping_source",
        "mapping_sha256",
    }
    assert manifest["evidence"]["mapping_sha256"] == (
        "0aedb9a63f9e12d919fa46a0be144b43262d26c9eb50fe7f374f128685a2b47d"
    )
    assert set(manifest["sensor"]) == {"config", "config_sha256"}
    sensor = manifest["sensor"]["config"]
    assert set(sensor) == {
        "views",
        "yaw_degrees",
        "height",
        "width",
        "hfov_degrees",
        "position",
        "min_depth_m",
        "max_depth_m",
        "normalize_depth",
        "rgb_channel_order",
        "depth_units",
        "orientation_rule",
        "camera_pose_authority",
        "resolved_specs",
    }
    assert sensor["views"] == 12
    assert sensor["yaw_degrees"] == list(range(0, 360, 30))
    assert len(sensor["resolved_specs"]) == 36
    assert [record["uuid"] for record in sensor["resolved_specs"]] == sorted(
        record["uuid"] for record in sensor["resolved_specs"]
    )
    assert set(sensor["resolved_specs"][0]) == {
        "uuid",
        "modality",
        "habitat_sensor_type",
        "habitat_sensor_subtype",
        "resolution",
        "hfov_degrees",
        "position",
        "orientation",
        "min_depth_m",
        "max_depth_m",
        "normalize_depth",
    }
    depth = next(
        record
        for record in sensor["resolved_specs"]
        if record["uuid"] == "depth_000"
    )
    assert depth == {
        "uuid": "depth_000",
        "modality": "depth",
        "habitat_sensor_type": "DEPTH",
        "habitat_sensor_subtype": "PINHOLE",
        "resolution": [256, 256],
        "hfov_degrees": 90.0,
        "position": [0.0, 1.25, 0.0],
        "orientation": [0.0, 0.0, 0.0],
        "min_depth_m": 0.0,
        "max_depth_m": 10.0,
        "normalize_depth": False,
    }
    rgb = next(
        record for record in sensor["resolved_specs"] if record["uuid"] == "rgb_000"
    )
    assert rgb["habitat_sensor_type"] == "COLOR"
    assert (
        rgb["min_depth_m"],
        rgb["max_depth_m"],
        rgb["normalize_depth"],
    ) == (None, None, None)
    assert manifest["sensor"]["config_sha256"] == hashlib.sha256(
        json.dumps(sensor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert set(manifest["scene_assets"]) == {
        "source_root",
        "role_classification_algorithm",
        "required_roles",
        "auxiliary_roles",
        "scenes",
    }
    assert manifest["scene_assets"]["required_roles"] == [
        "glb",
        "house",
        "semantic_ply",
    ]
    assert manifest["scene_assets"]["auxiliary_roles"] == ["navmesh"]
    assert len(manifest["scene_assets"]["scenes"]) == 11
    assert set(manifest["environment"]) == {
        "python_version",
        "python_implementation",
        "platform",
        "numpy_version",
        "zlib_version",
        "habitat_version",
        "habitat_sim_version",
        "cuda_runtime_version",
        "nvidia_driver_version",
        "gpu_name",
        "gpu_uuid",
        "gpu_device_id",
        "installed_distributions",
        "installed_distributions_sha256",
    }
    assert manifest["replay"] == {
        "observation_count": 50,
        "passed_count": 50,
        "scene_count": 11,
        "array_equal_fields": [
            "ego_free_mask",
            "ego_observed_mask",
            "ego_semantic_grid",
            "target_free_mask",
            "target_observed_mask",
            "target_semantic_grid",
        ],
        "metadata_comparison": (
            "cast-replayed-values-to-float32-then-array-equal"
        ),
    }
    assert set(manifest["files"]) == {"index", "artifacts", "payload"}
    assert manifest["files"]["index"] == {
        "path": "index.jsonl",
        "byte_length": len(index),
        "row_count": 50,
        "sha256": hashlib.sha256(index).hexdigest(),
    }
    assert manifest["files"]["artifacts"]["file_count"] == 50
    assert manifest["files"]["payload"]["file_count"] == 51


def test_tracked_asset_roles_bind_same_buffer_content_and_manifest_sha() -> None:
    path = "prior/analyze/d2026_07_29/rgbd_segmenter_asset_roles.json"
    data = Path(path).read_bytes()
    expected = (
        b"{\n"
        b'  "algorithm": "single-role-omission-first-row-per-scene-v1",\n'
        b'  "auxiliary": [\n'
        b'    "navmesh"\n'
        b"  ],\n"
        b'  "required": [\n'
        b'    "glb",\n'
        b'    "house",\n'
        b'    "semantic_ply"\n'
        b"  ],\n"
        b'  "schema_version": 1\n'
        b"}\n"
    )
    digest = hashlib.sha256(data).hexdigest()
    sources = replace(
        _manifest_sources(),
        asset_roles=SourceRecord(path=path, data=data),
    )
    rows = _manifest_rows()
    manifest = json.loads(
        build_manifest(
            git_commit="a" * 40,
            sources=sources,
            scene_assets=_scene_assets(),
            environment=_environment(),
            index_bytes=canonical_index_bytes(rows),
            rows=rows,
        )
    )

    assert data == expected
    assert digest == "a2cb167a71fb710f11c686589614c3e414c4f877ad83d6444d0e868983628b94"
    assert manifest["collection"]["asset_roles"] == {
        "path": path,
        "byte_length": len(data),
        "sha256": digest,
    }


def test_collection_input_preserves_same_buffer_cohort_row_hash() -> None:
    inputs = load_collection_inputs()
    cohort = Path(
        "data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1/cohort.jsonl"
    ).read_bytes()

    assert [item.cohort_row_sha256 for item in inputs.observations] == [
        hashlib.sha256(line).hexdigest() for line in cohort.splitlines()
    ]


def test_cohort_hashes_each_accepted_line_without_json_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    root = Path("data/rgbd_segmenter_benchmark/r2r-val-unseen-50-v1")
    manifest = (root / "manifest.json").read_bytes()
    cohort_data = (root / "cohort.jsonl").read_bytes()
    accepted_lines = cohort_data.splitlines()

    class AcceptedCohort(bytes):
        def splitlines(self, keepends: bool = False) -> list[bytes]:
            assert keepends is False
            return accepted_lines

    cohort = AcceptedCohort(cohort_data)
    real_dumps = raw_frames.json.dumps
    real_sha256 = raw_frames.hashlib.sha256
    calls = 0
    hashed_buffers = []

    def count_dumps(
        value: object,
        *,
        sort_keys: bool = False,
        separators: Optional[tuple[str, str]] = None,
    ) -> str:
        nonlocal calls
        calls += 1
        return real_dumps(
            value,
            sort_keys=sort_keys,
            separators=separators,
        )

    monkeypatch.setattr(raw_frames.json, "dumps", count_dumps)
    monkeypatch.setattr(
        raw_frames.hashlib,
        "sha256",
        lambda data=b"": (
            hashed_buffers.append(data),
            real_sha256(data),
        )[1],
    )
    _, _, hashes = raw_frames._parse_cohort(manifest, cohort)

    assert calls == 50
    assert len(hashed_buffers) == len(accepted_lines)
    assert all(
        actual is accepted
        for actual, accepted in zip(hashed_buffers, accepted_lines)
    )
    assert hashes == tuple(
        hashlib.sha256(line).hexdigest() for line in cohort.splitlines()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "collection_source",
        "cohort_directory",
        "evidence_root",
        "sensor_fixed",
        "sensor_wrong_type",
        "sensor_spec",
        "scene_algorithm",
        "scene_path",
        "scene_required",
        "scene_bundle",
        "environment_distribution",
        "environment_type",
        "replay_fields",
    ],
)
def test_manifest_rejects_nested_schema_and_asset_role_mutations(
    mutation: str, tmp_path: Path,
) -> None:
    rows = _manifest_rows()
    index = canonical_index_bytes(rows)
    assets = dict(_scene_assets())
    first = assets["scene-00"]
    files = dict(first.files)
    files["navmesh"] = replace(files["navmesh"], required=True)
    with pytest.raises(ValueError, match="required"):
        SceneAssetCommitment.from_files(files)

    manifest = json.loads(
        build_manifest(
            git_commit="a" * 40,
            sources=_manifest_sources(),
            scene_assets=_scene_assets(),
            environment=_environment(),
            index_bytes=index,
            rows=rows,
        )
    )
    if mutation == "collection_source":
        manifest["collection"]["collector_source"]["path"] = "wrong.py"
    elif mutation == "cohort_directory":
        manifest["cohort"]["directory"] = "wrong"
    elif mutation == "evidence_root":
        manifest["evidence"]["root"] = "wrong"
    elif mutation == "sensor_fixed":
        manifest["sensor"]["config"]["width"] = 257
        config = manifest["sensor"]["config"]
        manifest["sensor"]["config_sha256"] = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    elif mutation == "sensor_wrong_type":
        manifest["sensor"]["config"]["width"] = 256.0
        config = manifest["sensor"]["config"]
        manifest["sensor"]["config_sha256"] = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    elif mutation == "sensor_spec":
        manifest["sensor"]["config"]["resolved_specs"][0]["habitat_sensor_type"] = (
            "COLOR"
        )
        config = manifest["sensor"]["config"]
        manifest["sensor"]["config_sha256"] = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    elif mutation == "scene_algorithm":
        manifest["scene_assets"]["role_classification_algorithm"] = "wrong"
    elif mutation == "scene_path":
        manifest["scene_assets"]["scenes"]["scene-00"]["files"]["glb"]["path"] = (
            "scene-00/wrong.glb"
        )
    elif mutation == "scene_required":
        manifest["scene_assets"]["scenes"]["scene-00"]["files"]["navmesh"][
            "required"
        ] = True
    elif mutation == "scene_bundle":
        manifest["scene_assets"]["scenes"]["scene-00"]["bundle_sha256"] = "0" * 64
    elif mutation == "environment_distribution":
        manifest["environment"]["installed_distributions"][0]["name"] = "Habitat_Lab"
        distributions = manifest["environment"]["installed_distributions"]
        manifest["environment"]["installed_distributions_sha256"] = hashlib.sha256(
            json.dumps(
                distributions, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    elif mutation == "environment_type":
        manifest["environment"]["gpu_name"] = 7
    elif mutation == "replay_fields":
        manifest["replay"]["array_equal_fields"].reverse()
    mutated = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "manifest.json").write_bytes(mutated)

    with pytest.raises(ValueError, match="manifest"):
        validate_raw_frame_directory(root, expected_manifest=mutated)


def test_manifest_accepts_sorted_unique_distribution_name_version_pairs() -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package as package

    rows = _manifest_rows()
    index = canonical_index_bytes(rows)
    manifest = json.loads(
        build_manifest(
            git_commit="a" * 40,
            sources=_manifest_sources(),
            scene_assets=_scene_assets(),
            environment=_environment(),
            index_bytes=index,
            rows=rows,
        )
    )
    distributions = manifest["environment"]["installed_distributions"]
    distributions.append(
        {
            "name": distributions[0]["name"],
            "version": distributions[0]["version"] + ".post1",
        }
    )
    distributions.sort(key=lambda item: (item["name"], item["version"]))
    manifest["environment"]["installed_distributions_sha256"] = hashlib.sha256(
        json.dumps(distributions, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    mutated = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"

    package._parse_manifest_bytes(mutated)


def test_validator_accepts_one_complete_structurally_valid_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package as package

    artifact_data = b"accepted-artifact"
    artifact_record = FileRecord(
        byte_length=len(artifact_data),
        sha256=hashlib.sha256(artifact_data).hexdigest(),
    )
    rows = tuple(replace(row, npz=artifact_record) for row in _manifest_rows())
    index = canonical_index_bytes(rows)
    manifest = build_manifest(
        git_commit="a" * 40,
        sources=_manifest_sources(),
        scene_assets=_scene_assets(),
        environment=_environment(),
        index_bytes=index,
        rows=rows,
    )
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "index.jsonl").write_bytes(index)
    (root / "manifest.json").write_bytes(manifest)
    for row in rows:
        artifact = root / row.artifact
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(artifact_data)
    monkeypatch.setattr(
        package,
        "parse_raw_frame_npz_bytes",
        lambda *_args, **_kwargs: None,
    )

    validate_raw_frame_directory(root, expected_manifest=manifest)


def _attempt_observations() -> tuple[CollectionObservation, ...]:
    counts = (5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4)
    observations = []
    ordinal = 0
    for scene_index, count in enumerate(counts):
        for _ in range(count):
            observations.append(
                CollectionObservation(
                    artifact_path=Path(
                        f"observations/scene-{scene_index:02d}/{ordinal:020x}.npz"
                    ),
                    artifact_sha256=f"{ordinal + 101:064x}",
                    cohort_row_sha256=f"{ordinal + 1:064x}",
                    example_ids=(f"R2R_val_unseen_{ordinal}",),
                    observation_id=f"{ordinal:020x}",
                    scene_id=f"scene-{scene_index:02d}",
                    start_position=(float(ordinal), 0.0, 0.0),
                    start_rotation=(0.0, 0.0, 0.0, 1.0),
                )
            )
            ordinal += 1
    return tuple(observations)


def _bundle_for_scene(root: Path, scene: str) -> SceneBundle:
    directory = root / scene
    directory.mkdir()
    files = {}
    for role, suffix in (
        ("glb", ".glb"),
        ("house", ".house"),
        ("navmesh", ".navmesh"),
        ("semantic_ply", "_semantic.ply"),
    ):
        path = directory / f"{scene}{suffix}"
        path.write_bytes(f"{scene}:{role}".encode())
        files[role] = SnapshotFile(
            path=path,
            byte_length=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return SceneBundle(scene_id=scene, files=MappingProxyType(files))


def test_collect_attempt_keeps_sealed_order_one_simulator_per_scene_and_no_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    observations = _attempt_observations()
    scenes = tuple(f"scene-{index:02d}" for index in range(11))
    attempt = tmp_path / "attempt"
    snapshots = tmp_path / "snapshots"
    events: list[tuple[str, object]] = []
    closed: list[str] = []
    encoded_members = encode_raw_frame_npz(
        render_raw_frame_artifact(
            _fake_simulator(), _observation(artifact_sha256="0" * 64)
        )
    ).members
    captures = iter(
        [
            ("a" * 40, _manifest_sources(), _environment()),
            ("a" * 40, _manifest_sources(), _environment()),
            ("a" * 40, _manifest_sources(), _environment()),
        ]
    )

    class FakeSimulator:
        def __init__(self, scene: str) -> None:
            self.scene = scene

        def close(self) -> None:
            closed.append(self.scene)

    def fake_snapshot(scene: str, root: Path) -> SceneBundle:
        events.append(("snapshot", scene))
        return _bundle_for_scene(root, scene)

    def fake_render(
        simulator: FakeSimulator, observation: CollectionObservation
    ) -> SimpleNamespace:
        assert simulator.scene == observation.scene_id
        events.append(("render", observation.observation_id))
        return SimpleNamespace(observation_id=observation.observation_id)

    def fake_encode(arrays: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            data=f"npz:{arrays.observation_id}".encode(),
            members=encoded_members,
        )

    def fake_parse(data: bytes, **_kwargs: object) -> SimpleNamespace:
        identifier = data.decode().split(":", 1)[1]
        events.append(("parse", identifier))
        return SimpleNamespace(
            arrays=SimpleNamespace(observation_id=identifier),
            members=encoded_members,
        )

    def fake_oracle(observation: CollectionObservation) -> SimpleNamespace:
        events.append(("oracle", observation.observation_id))
        return SimpleNamespace()

    def fake_replay(
        arrays: SimpleNamespace,
        observation: CollectionObservation,
        _oracle: object,
    ) -> None:
        assert arrays.observation_id == observation.observation_id
        events.append(("replay", observation.observation_id))

    monkeypatch.setattr(
        raw_frames,
        "load_collection_inputs",
        lambda: SimpleNamespace(observations=observations, scenes=scenes),
    )
    monkeypatch.setattr(raw_frames, "_capture_attempt_state", lambda: next(captures))
    monkeypatch.setattr(raw_frames, "snapshot_scene_bundle", fake_snapshot)
    monkeypatch.setattr(
        raw_frames,
        "build_scene_simulator",
        lambda bundle: FakeSimulator(bundle.scene_id),
    )
    monkeypatch.setattr(raw_frames, "render_raw_frame_artifact", fake_render)
    monkeypatch.setattr(raw_frames, "encode_raw_frame_npz", fake_encode)
    monkeypatch.setattr(raw_frames, "parse_raw_frame_npz_bytes", fake_parse)
    monkeypatch.setattr(raw_frames, "load_pinned_oracle_after_render", fake_oracle)
    monkeypatch.setattr(raw_frames, "replay_and_require_exact", fake_replay)
    monkeypatch.setattr(raw_frames, "_require_bundle_unchanged", lambda _bundle: None)
    monkeypatch.setattr(
        raw_frames,
        "_require_original_asset_commitments",
        lambda _assets: None,
    )
    monkeypatch.setattr(
        raw_frames,
        "validate_raw_frame_directory",
        lambda root, *, expected_manifest: events.append(("validate", root)),
    )

    manifest = collect_attempt(attempt, snapshots)

    assert [value for name, value in events if name == "render"] == [
        observation.observation_id for observation in observations
    ]
    assert [value for name, value in events if name == "snapshot"] == list(scenes)
    assert closed == list(scenes)
    for observation in observations:
        identifier = observation.observation_id
        positions = {
            name: events.index((name, identifier))
            for name in ("render", "parse", "oracle", "replay")
        }
        assert positions["render"] < positions["parse"] < positions["oracle"]
        assert positions["oracle"] < positions["replay"]
    assert (attempt / "index.jsonl").read_bytes() == canonical_index_bytes(
        tuple(
            IndexRow(
                artifact=f"observations/{item.scene_id}/"
                f"{ordinal:02d}-{item.observation_id}.npz",
                cohort_row_sha256=item.cohort_row_sha256,
                members=encoded_members,
                npz=FileRecord(
                    byte_length=len(f"npz:{item.observation_id}".encode()),
                    sha256=hashlib.sha256(
                        f"npz:{item.observation_id}".encode()
                    ).hexdigest(),
                ),
                observation_id=item.observation_id,
                oracle_artifact_sha256=item.artifact_sha256,
                ordinal=ordinal,
                scene_id=item.scene_id,
            )
            for ordinal, item in enumerate(observations)
        )
    )
    assert (attempt / "manifest.json").read_bytes() == manifest
    assert events[-1] == ("validate", attempt)
    assert not snapshots.exists()

    with pytest.raises(FileExistsError):
        collect_attempt(attempt, snapshots)


def test_collect_attempt_closes_and_removes_partial_index_on_replay_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    observations = _attempt_observations()
    attempt = tmp_path / "attempt"
    snapshots = tmp_path / "snapshots"
    simulator = SimpleNamespace(close=lambda: None)
    closed = []
    simulator.close = lambda: closed.append(True)
    members = encode_raw_frame_npz(
        render_raw_frame_artifact(
            _fake_simulator(), _observation(artifact_sha256="0" * 64)
        )
    ).members
    monkeypatch.setattr(
        raw_frames,
        "load_collection_inputs",
        lambda: SimpleNamespace(
            observations=observations,
            scenes=tuple(f"scene-{index:02d}" for index in range(11)),
        ),
    )
    monkeypatch.setattr(
        raw_frames,
        "_capture_attempt_state",
        lambda: ("a" * 40, _manifest_sources(), _environment()),
    )
    monkeypatch.setattr(
        raw_frames,
        "snapshot_scene_bundle",
        lambda scene, root: _bundle_for_scene(root, scene),
    )
    monkeypatch.setattr(raw_frames, "build_scene_simulator", lambda _bundle: simulator)
    monkeypatch.setattr(
        raw_frames, "render_raw_frame_artifact", lambda _sim, obs: obs
    )
    monkeypatch.setattr(
        raw_frames,
        "encode_raw_frame_npz",
        lambda obs: SimpleNamespace(
            data=f"npz:{obs.observation_id}".encode(), members=members
        ),
    )
    monkeypatch.setattr(
        raw_frames,
        "parse_raw_frame_npz_bytes",
        lambda _data, **_kwargs: SimpleNamespace(
            arrays=SimpleNamespace(), members=members
        ),
    )
    monkeypatch.setattr(
        raw_frames, "load_pinned_oracle_after_render", lambda _obs: SimpleNamespace()
    )

    def fail_replay(*_args: object) -> None:
        raise RuntimeError("replay failed")

    monkeypatch.setattr(raw_frames, "replay_and_require_exact", fail_replay)
    monkeypatch.setattr(raw_frames, "_require_bundle_unchanged", lambda _bundle: None)

    with pytest.raises(RuntimeError, match="replay failed"):
        collect_attempt(attempt, snapshots)

    assert closed == [True]
    assert not attempt.exists()
    assert not snapshots.exists()


def test_full_attempt_requires_tracked_asset_roles_before_creating_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    attempt = tmp_path / "attempt"
    snapshots = tmp_path / "snapshots"
    paths = dict(raw_frames._SOURCE_PATHS)
    paths["asset_roles"] = (tmp_path / "missing-asset-roles.json").as_posix()
    monkeypatch.setattr(raw_frames, "_SOURCE_PATHS", paths)

    with pytest.raises(FileNotFoundError, match="asset_roles"):
        collect_attempt(attempt, snapshots)

    assert not attempt.exists()
    assert not snapshots.exists()


def test_collect_attempt_rejects_final_and_smoke_roots_before_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    final = tmp_path / "final"
    smoke = tmp_path / "smoke"
    snapshots = tmp_path / "snapshots"
    monkeypatch.setattr(raw_frames, "FINAL_PACKAGE_ROOT", final)
    monkeypatch.setattr(raw_frames, "SMOKE_PACKAGE_ROOT", smoke)

    def fail_preflight() -> None:
        raise AssertionError("forbidden roots must fail before preflight")

    monkeypatch.setattr(
        raw_frames,
        "_capture_attempt_state",
        fail_preflight,
    )

    for root in (final, final / "child", smoke, smoke / "child"):
        with pytest.raises(ValueError, match="forbidden"):
            collect_attempt(root, snapshots)
        assert not root.exists()
        assert not snapshots.exists()

    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    for root in (alias / "final", alias / "smoke"):
        with pytest.raises(ValueError, match="resolves inside"):
            collect_attempt(root, snapshots)
        assert not final.exists()
        assert not smoke.exists()

    with pytest.raises(ValueError, match="forbidden"):
        collect_attempt(tmp_path / "attempt", final / "snapshot")
    with pytest.raises(ValueError, match="overlap"):
        collect_attempt(tmp_path / "attempt", tmp_path / "attempt" / "snapshot")
    with pytest.raises(ValueError, match="lexical"):
        collect_attempt(tmp_path / "safe" / ".." / "attempt", snapshots)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="absent"):
        collect_attempt(existing, snapshots)


def test_dynamic_source_reader_uses_one_nofollow_regular_file_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    monkeypatch.chdir(tmp_path)
    Path("source.py").write_bytes(b"source")
    Path("alias.py").symlink_to("source.py")

    record = raw_frames._read_source("source.py")

    assert record.data == b"source"
    with pytest.raises(ValueError, match="regular"):
        raw_frames._read_source("alias.py")

    Path("real").mkdir()
    Path("real/nested.py").write_bytes(b"nested")
    Path("linked").symlink_to("real", target_is_directory=True)
    with pytest.raises(ValueError, match="stable regular"):
        raw_frames._read_source("linked/nested.py")


def test_dynamic_source_reader_detects_ctime_only_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    monkeypatch.chdir(tmp_path)
    source = Path("source.py")
    source.write_bytes(b"before")
    original = source.stat()
    real_read = raw_frames.os.read
    mutated = False

    def mutate_after_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        data = real_read(descriptor, size)
        if data and not mutated:
            mutated = True
            time.sleep(0.01)
            source.write_bytes(b"after!")
            os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
        return data

    monkeypatch.setattr(raw_frames.os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="changed while read"):
        raw_frames._read_source("source.py")


def test_dynamic_source_reader_closes_rejected_leaf_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    monkeypatch.chdir(tmp_path)
    Path("source.py").write_bytes(b"source")
    real_open = raw_frames.os.open
    real_close = raw_frames.os.close
    real_fstat = raw_frames.os.fstat
    leaf_descriptors = []
    closed = []

    def track_open(
        path: Union[str, bytes, os.PathLike[str], os.PathLike[bytes]],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: Optional[int] = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "source.py":
            leaf_descriptors.append(descriptor)
        return descriptor

    def non_regular_leaf(descriptor: int) -> os.stat_result:
        result = real_fstat(descriptor)
        if descriptor in leaf_descriptors:
            values = list(result)
            values[0] = stat.S_IFDIR | 0o700
            return os.stat_result(values)
        return result

    def track_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(raw_frames.os, "open", track_open)
    monkeypatch.setattr(raw_frames.os, "fstat", non_regular_leaf)
    monkeypatch.setattr(raw_frames.os, "close", track_close)

    with pytest.raises(ValueError, match="changed before read"):
        raw_frames._read_source("source.py")

    assert leaf_descriptors[0] in closed


def test_close_still_rehashes_bundle_when_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    events = []

    class CloseFailure:
        def close(self) -> None:
            events.append("close")
            raise RuntimeError("x")

    simulator = CloseFailure()
    monkeypatch.setattr(
        raw_frames,
        "_require_bundle_unchanged",
        lambda _bundle: events.append("rehash"),
    )

    with pytest.raises(RuntimeError, match="x"):
        raw_frames._close_and_require_bundle_unchanged(
            simulator,
            SceneBundle(scene_id="scene", files=MappingProxyType({})),
        )

    assert events == ["close", "rehash"]


def test_final_original_asset_rehash_detects_post_scene_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    root = tmp_path / "mp3d"
    scene = "scene-00"
    source = root / scene
    source.mkdir(parents=True)
    commitment = _scene_assets()[scene]
    for record in commitment.files.values():
        role = record.path.rsplit("/", 1)[1]
        payload_role = next(
            candidate
            for candidate in ("glb", "house", "navmesh", "semantic_ply")
            if record.sha256
            == hashlib.sha256(f"{scene}:{candidate}".encode()).hexdigest()
        )
        (source / role).write_bytes(f"{scene}:{payload_role}".encode())
    monkeypatch.setattr(raw_frames, "SCENE_DATASET_ROOT", root)

    raw_frames._require_original_asset_commitments({scene: commitment})
    (source / f"{scene}.glb").write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256"):
        raw_frames._require_original_asset_commitments({scene: commitment})


@pytest.mark.parametrize(
    "drift",
    ["head", "source", "environment", "gpu", "asset"],
)
def test_attempt_drift_matrix_cleans_all_private_state(
    drift: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    observations = _attempt_observations()
    scenes = tuple(f"scene-{index:02d}" for index in range(11))
    attempt = tmp_path / "attempt"
    snapshots = tmp_path / "snapshots"
    sources = _manifest_sources()
    environment = _environment()
    base = ("a" * 40, sources, environment)
    changed = {
        "head": ("b" * 40, sources, environment),
        "source": (
            "a" * 40,
            replace(
                sources,
                collector_source=_source(
                    "prior/analyze/d2026_07_29/rgbd_segmenter_raw_frames.py",
                    "changed",
                ),
            ),
            environment,
        ),
        "environment": (
            "a" * 40,
            sources,
            replace(environment, platform="changed"),
        ),
        "gpu": (
            "a" * 40,
            sources,
            replace(environment, gpu_uuid="GPU-changed"),
        ),
        "asset": base,
    }[drift]
    captures = iter((base, base, changed))
    closed = []
    members = _manifest_rows()[0].members

    class Simulator:
        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(
        raw_frames,
        "load_collection_inputs",
        lambda: SimpleNamespace(observations=observations, scenes=scenes),
    )
    monkeypatch.setattr(raw_frames, "_capture_attempt_state", lambda: next(captures))
    monkeypatch.setattr(
        raw_frames,
        "snapshot_scene_bundle",
        lambda scene, root: _bundle_for_scene(root, scene),
    )
    monkeypatch.setattr(raw_frames, "build_scene_simulator", lambda _bundle: Simulator())
    monkeypatch.setattr(
        raw_frames, "render_raw_frame_artifact", lambda *_args: object()
    )
    monkeypatch.setattr(
        raw_frames,
        "encode_raw_frame_npz",
        lambda _arrays: SimpleNamespace(data=b"npz", members=members),
    )
    monkeypatch.setattr(
        raw_frames,
        "parse_raw_frame_npz_bytes",
        lambda *_args, **_kwargs: SimpleNamespace(arrays=object(), members=members),
    )
    monkeypatch.setattr(
        raw_frames, "load_pinned_oracle_after_render", lambda _observation: object()
    )
    monkeypatch.setattr(raw_frames, "replay_and_require_exact", lambda *_args: None)
    monkeypatch.setattr(raw_frames, "_require_bundle_unchanged", lambda _bundle: None)
    if drift == "asset":
        monkeypatch.setattr(
            raw_frames,
            "_require_original_asset_commitments",
            lambda _assets: (_ for _ in ()).throw(ValueError("asset changed")),
        )
    else:
        monkeypatch.setattr(
            raw_frames, "_require_original_asset_commitments", lambda _assets: None
        )
    monkeypatch.setattr(
        raw_frames, "validate_raw_frame_directory", lambda *_args, **_kwargs: None
    )

    with pytest.raises(ValueError, match="changed"):
        collect_attempt(attempt, snapshots)

    assert len(closed) == 11
    assert not attempt.exists()
    assert not snapshots.exists()


def test_simulator_constructor_failure_rehashes_bundle_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    observations = _attempt_observations()
    scenes = tuple(f"scene-{index:02d}" for index in range(11))
    rehashed = []
    monkeypatch.setattr(
        raw_frames,
        "load_collection_inputs",
        lambda: SimpleNamespace(observations=observations, scenes=scenes),
    )
    monkeypatch.setattr(
        raw_frames,
        "_capture_attempt_state",
        lambda: ("a" * 40, _manifest_sources(), _environment()),
    )
    monkeypatch.setattr(
        raw_frames,
        "snapshot_scene_bundle",
        lambda scene, root: _bundle_for_scene(root, scene),
    )
    monkeypatch.setattr(
        raw_frames,
        "build_scene_simulator",
        lambda _bundle: (_ for _ in ()).throw(RuntimeError("constructor")),
    )
    monkeypatch.setattr(
        raw_frames,
        "_require_bundle_unchanged",
        lambda bundle: rehashed.append(bundle.scene_id),
    )
    attempt = tmp_path / "attempt"
    snapshots = tmp_path / "snapshots"

    with pytest.raises(RuntimeError, match="constructor"):
        collect_attempt(attempt, snapshots)

    assert rehashed == ["scene-00"]
    assert not attempt.exists()
    assert not snapshots.exists()


def test_omission_spawn_contains_native_exit_and_revalidates_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    staging = tmp_path / "staging"
    snapshots = tmp_path / "snapshots"
    bundle_root = tmp_path / "bundle"
    staging.mkdir()
    snapshots.mkdir()
    bundle_root.mkdir()
    bundle = _bundle(bundle_root)
    validated = []
    render_spawned = raw_frames._render_in_spawned_process
    monkeypatch.setattr(
        raw_frames, "snapshot_scene_bundle", lambda _scene, _root: bundle
    )
    monkeypatch.setattr(
        raw_frames,
        "_render_in_spawned_process",
        lambda live_bundle, observation, omitted_role: (
            render_spawned(
                live_bundle,
                observation,
                omitted_role,
                worker=_abrupt_spawn_worker,
            )
        ),
    )
    monkeypatch.setattr(
        raw_frames,
        "_require_variant_assets",
        lambda _bundle, omitted: validated.append(omitted),
    )

    assert (
        raw_frames._run_smoke_variant(
            _observation(artifact_sha256="0" * 64),
            staging,
            snapshots,
            "glb",
        )
        is False
    )
    assert validated == ["glb"]
    raw_frames._cleanup_smoke_paths(staging, snapshots)
    assert not staging.exists()
    assert not snapshots.exists()

    control_root = tmp_path / "control"
    control_root.mkdir()
    with pytest.raises(RuntimeError, match="control"):
        render_spawned(
            _bundle(control_root),
            _observation(artifact_sha256="0" * 64),
            None,
            worker=_abrupt_spawn_worker,
        )

    (control_root / "success").mkdir()
    arrays = render_spawned(
        _bundle(control_root / "success"),
        _observation(artifact_sha256="0" * 64),
        None,
        worker=_large_array_spawn_worker,
    )
    assert arrays is not None
    assert arrays.rgb.nbytes == 10 * 1024 * 1024


@pytest.mark.parametrize("failure", ["encode", "reload", "parse", "oracle"])
def test_omission_smoke_propagates_nonclassification_failures(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    staging = tmp_path / "staging"
    snapshots = tmp_path / "snapshots"
    bundle_root = tmp_path / "bundle"
    staging.mkdir()
    snapshots.mkdir()
    bundle_root.mkdir()
    bundle = _bundle(bundle_root)
    monkeypatch.setattr(
        raw_frames, "snapshot_scene_bundle", lambda _scene, _root: bundle
    )
    monkeypatch.setattr(
        raw_frames,
        "_render_in_spawned_process",
        lambda _bundle, _observation, _omitted: object(),
    )
    monkeypatch.setattr(
        raw_frames,
        "encode_raw_frame_npz",
        lambda _arrays: SimpleNamespace(data=b"raw", members={}),
    )
    monkeypatch.setattr(raw_frames, "strict_read_bytes", lambda *_args: b"raw")
    monkeypatch.setattr(
        raw_frames,
        "parse_raw_frame_npz_bytes",
        lambda *_args, **_kwargs: SimpleNamespace(arrays=object()),
    )
    monkeypatch.setattr(
        raw_frames, "load_pinned_oracle_after_render", lambda _observation: object()
    )
    monkeypatch.setattr(raw_frames, "replay_and_require_exact", lambda *_args: None)
    monkeypatch.setattr(raw_frames, "_require_variant_assets", lambda *_args: None)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(failure)

    target = {
        "encode": "encode_raw_frame_npz",
        "reload": "strict_read_bytes",
        "parse": "parse_raw_frame_npz_bytes",
        "oracle": "load_pinned_oracle_after_render",
    }[failure]
    monkeypatch.setattr(raw_frames, target, fail)

    with pytest.raises(RuntimeError, match=failure):
        raw_frames._run_smoke_variant(
            _observation(artifact_sha256="0" * 64),
            staging,
            snapshots,
            "navmesh",
        )


def test_first_per_scene_smoke_classifies_55_disposable_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames as raw_frames

    observations = _attempt_observations()
    smoke_root = tmp_path / "smoke"
    final_root = tmp_path / "final"
    calls = []

    def fake_variant(
        observation: CollectionObservation,
        _staging: Path,
        _snapshots: Path,
        omitted_role: str | None,
    ) -> bool:
        calls.append((observation.scene_id, omitted_role))
        return omitted_role not in {"glb", "house", "semantic_ply"}

    monkeypatch.setattr(raw_frames, "SMOKE_PACKAGE_ROOT", smoke_root)
    monkeypatch.setattr(raw_frames, "FINAL_PACKAGE_ROOT", final_root)
    monkeypatch.setattr(
        raw_frames,
        "load_collection_inputs",
        lambda: SimpleNamespace(
            observations=observations,
            scenes=tuple(f"scene-{index:02d}" for index in range(11)),
        ),
    )
    monkeypatch.setattr(raw_frames, "_run_smoke_variant", fake_variant)

    report = run_first_per_scene_smoke()

    expected_calls = [
        (f"scene-{index:02d}", role)
        for index in range(11)
        for role in (None, "glb", "house", "navmesh", "semantic_ply")
    ]
    assert calls == expected_calls
    expected = {
        "classification": {
            "algorithm": "single-role-omission-first-row-per-scene-v1",
            "auxiliary": ["navmesh"],
            "required": ["glb", "house", "semantic_ply"],
            "schema_version": 1,
        },
        "control_passed_count": 11,
        "scene_count": 11,
        "schema_version": 1,
    }
    assert report == json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"
    assert capsys.readouterr().out.encode() == report
    assert not smoke_root.exists()
    assert not final_root.exists()
