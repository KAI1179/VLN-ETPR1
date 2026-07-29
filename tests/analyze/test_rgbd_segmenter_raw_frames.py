from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Optional, Union
import zipfile

import numpy as np
import pytest

from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    strict_read_bytes,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frames import (
    ORACLE_ARTIFACT_ROOT,
    SCENE_DATASET_ROOT,
    CollectionObservation,
    load_collection_inputs,
    load_pinned_oracle_after_render,
    snapshot_scene_bundle,
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
