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
    assert calls == [os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW]


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

    parsed = load_pinned_oracle_after_render(observation)

    assert parsed.sha256 == expected
    assert parsed.ego_semantic_grid.dtype == np.bool_
    assert parsed.target_semantic_grid[0, 0, 0]
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
        flags == os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW for _, flags in opens
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
    private_sibling.mkdir()
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
    private_sibling.mkdir()
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


def test_public_roots_are_the_experiment_pinned_locations() -> None:
    assert ORACLE_ARTIFACT_ROOT == Path(
        "data/llm_grid_oracle_evidence/oracle-t0-v1/r2r/val_unseen"
    )
    assert SCENE_DATASET_ROOT == Path("data/scene_datasets/mp3d")
