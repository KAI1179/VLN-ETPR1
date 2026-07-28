from __future__ import annotations

import hashlib
import errno
import gzip
import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import prior.analyze.d2026_07_28.rgbd_segmenter_cohort as cohort_module
from prior.analyze.d2026_07_28.rgbd_segmenter_cohort import (
    OFFICIAL_SCENE_COUNTS,
    OFFICIAL_SCENE_QUOTAS,
    SELECTION_DOMAIN,
    CanonicalObservation,
    build_cohort,
    build_cohort_from_sources,
    hamilton_apportion,
    publish_cohort,
    reconstruct_source_observations,
    select_observations,
    validate_cohort_directory,
)


def _observation(
    *,
    observation_id: str = "0123456789abcdefabcd",
    scene_id: str = "2azQ1b91cZZ",
    artifact_sha256: str = "a" * 64,
    aliases: tuple[str, ...] = ("R2R_val_unseen_1",),
    x: float = 1.0,
) -> CanonicalObservation:
    return CanonicalObservation(
        artifact_sha256=artifact_sha256,
        example_ids=aliases,
        observation_id=observation_id,
        scene_id=scene_id,
        start_position=(x, 2.0, 3.0),
        start_rotation=(0.0, 0.0, 0.0, 1.0),
    )


def test_canonical_observation_has_exact_stable_json_bytes() -> None:
    observation = _observation(aliases=("R2R_val_unseen_1", "R2R_val_unseen_7"))

    assert observation.canonical_row_bytes() == (
        b'{"artifact_sha256":"'
        + b"a" * 64
        + b'","example_ids":["R2R_val_unseen_1","R2R_val_unseen_7"],'
        b'"observation_id":"0123456789abcdefabcd","scene_id":"2azQ1b91cZZ",'
        b'"start_position":[1.0,2.0,3.0],'
        b'"start_rotation":[0.0,0.0,0.0,1.0]}'
    )
    selection_key = json.loads(observation.canonical_selection_key_bytes())
    assert "artifact_sha256" not in selection_key
    assert (
        observation.selection_digest()
        == hashlib.sha256(
            SELECTION_DOMAIN + observation.canonical_selection_key_bytes()
        ).digest()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_sha256", "A" * 64),
        ("artifact_sha256", "a" * 63),
        ("artifact_sha256", 1),
        ("observation_id", "g" * 20),
        ("observation_id", "0" * 19),
        ("observation_id", 1),
        ("scene_id", "../badscene"),
        ("scene_id", "short"),
        ("scene_id", 1),
        ("example_ids", ("R2R_val_seen_1",)),
        ("example_ids", ("R2R_val_unseen_01",)),
        ("example_ids", (1,)),
        (
            "example_ids",
            ("R2R_val_unseen_2", "R2R_val_unseen_1"),
        ),
        (
            "example_ids",
            ("R2R_val_unseen_1", "R2R_val_unseen_1"),
        ),
        ("example_ids", ()),
        ("start_position", (0.0, math.nan, 0.0)),
        ("start_position", (0.0, 0.0)),
        ("start_rotation", (0.0, 0.0, math.inf, 1.0)),
        ("start_rotation", (0.0, 0.0, 1.0)),
        ("start_rotation", (0.0, 0.0, 0.0, 2.0)),
    ],
)
def test_canonical_observation_rejects_invalid_values(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        replace(_observation(), **{field: value})


def test_canonical_observation_rejects_extra_fields() -> None:
    fields = {
        "artifact_sha256": "a" * 64,
        "example_ids": ("R2R_val_unseen_1",),
        "observation_id": "0" * 20,
        "scene_id": "2azQ1b91cZZ",
        "start_position": (0.0, 0.0, 0.0),
        "start_rotation": (0.0, 0.0, 0.0, 1.0),
        "extra": "forbidden",
    }

    with pytest.raises(TypeError):
        getattr(cohort_module, "CanonicalObservation")(**fields)


def test_hamilton_apportion_reproduces_frozen_official_quotas() -> None:
    assert hamilton_apportion(OFFICIAL_SCENE_COUNTS, 50) == dict(OFFICIAL_SCENE_QUOTAS)
    assert sum(OFFICIAL_SCENE_QUOTAS.values()) == 50


def test_hamilton_apportion_uses_lexical_remainder_ties() -> None:
    assert hamilton_apportion({"scene-c": 1, "scene-a": 1, "scene-b": 1}, 2) == {
        "scene-a": 1,
        "scene-b": 1,
        "scene-c": 0,
    }


@pytest.mark.parametrize(
    ("populations", "target"),
    [
        ({}, 1),
        ({"scene": 0}, 1),
        ({"scene": True}, 1),
        ({"scene": 1}, 0),
        ({"scene": 1}, 2),
    ],
)
def test_hamilton_apportion_rejects_invalid_population(
    populations: dict[str, int], target: int
) -> None:
    with pytest.raises(ValueError):
        hamilton_apportion(populations, target)


def test_select_observations_is_stratified_stable_and_artifact_independent() -> None:
    observations = (
        _observation(
            observation_id="00000000000000000001",
            artifact_sha256="1" * 64,
            aliases=("R2R_val_unseen_1",),
        ),
        _observation(
            observation_id="00000000000000000002",
            artifact_sha256="2" * 64,
            aliases=("R2R_val_unseen_2",),
        ),
        _observation(
            observation_id="00000000000000000003",
            artifact_sha256="3" * 64,
            aliases=("R2R_val_unseen_3",),
            scene_id="8194nk5LbLH",
        ),
        _observation(
            observation_id="00000000000000000004",
            artifact_sha256="4" * 64,
            aliases=("R2R_val_unseen_4",),
            scene_id="8194nk5LbLH",
        ),
    )
    quotas = {"2azQ1b91cZZ": 1, "8194nk5LbLH": 1}

    selected = select_observations(reversed(observations), quotas)
    selected_again = select_observations(observations, quotas)
    changed_hashes = tuple(
        replace(observation, artifact_sha256="f" * 64) for observation in observations
    )
    selected_changed_hashes = select_observations(changed_hashes, quotas)

    assert selected == selected_again
    assert [item.scene_id for item in selected] == [
        "2azQ1b91cZZ",
        "8194nk5LbLH",
    ]
    assert [item.observation_id for item in selected] == [
        "00000000000000000001",
        "00000000000000000003",
    ]
    assert [item.observation_id for item in selected] == [
        item.observation_id for item in selected_changed_hashes
    ]


def test_select_observations_rejects_population_inconsistency() -> None:
    observation = _observation()

    with pytest.raises(ValueError, match="duplicate observation"):
        select_observations((observation, observation), {"2azQ1b91cZZ": 1})
    with pytest.raises(ValueError, match="unknown scene"):
        select_observations((observation,), {"2azQ1b91cZZ": 1, "8194nk5LbLH": 0})
    with pytest.raises(ValueError, match="quota"):
        select_observations((observation,), {"2azQ1b91cZZ": 2})
    with pytest.raises(ValueError, match="quota"):
        getattr(cohort_module, "select_observations")((observation,), {1: 1})


def test_build_cohort_has_exact_order_bytes_and_distinct_digests() -> None:
    observations = (
        _observation(
            observation_id="00000000000000000002",
            artifact_sha256="2" * 64,
            aliases=("R2R_val_unseen_2",),
        ),
        _observation(
            observation_id="00000000000000000001",
            artifact_sha256="1" * 64,
            aliases=("R2R_val_unseen_1",),
        ),
        _observation(
            observation_id="00000000000000000003",
            artifact_sha256="3" * 64,
            aliases=("R2R_val_unseen_3",),
            scene_id="8194nk5LbLH",
        ),
    )

    cohort = build_cohort(observations, target_count=2)
    expected_jsonl = (
        b'{"artifact_sha256":"' + b"1" * 64 + b'","example_ids":["R2R_val_unseen_1"],'
        b'"observation_id":"00000000000000000001","scene_id":"2azQ1b91cZZ",'
        b'"start_position":[1.0,2.0,3.0],'
        b'"start_rotation":[0.0,0.0,0.0,1.0]}\n'
        b'{"artifact_sha256":"' + b"3" * 64 + b'","example_ids":["R2R_val_unseen_3"],'
        b'"observation_id":"00000000000000000003","scene_id":"8194nk5LbLH",'
        b'"start_position":[1.0,2.0,3.0],'
        b'"start_rotation":[0.0,0.0,0.0,1.0]}\n'
    )

    assert cohort.scene_quotas == {"2azQ1b91cZZ": 1, "8194nk5LbLH": 1}
    assert cohort.cohort_jsonl == expected_jsonl
    assert (
        cohort.selection_sha256
        == "272064e93350e31e5683dd45a31ece1181a034a8c996c7c64df3e3ec5223e40f"
    )
    assert (
        cohort.cohort_sha256
        == "34c1ef9706fce79b259386e0b223450eeb955786e4a0e227027fb378b93da8c9"
    )
    assert tuple(item.scene_id for item in cohort.observations) == (
        "2azQ1b91cZZ",
        "8194nk5LbLH",
    )


def _source_buffers(
    episodes: list[dict[str, object]],
    *,
    artifact_sha256: str = "a" * 64,
) -> tuple[bytes, bytes, bytes]:
    raw_split = gzip.compress(json.dumps({"episodes": episodes}).encode())
    parsed = cohort_module.parse_start_episodes(raw_split)
    records = [
        {
            "artifact_sha256": artifact_sha256,
            "example_id": episode.example_id,
            "observation_id": episode.observation_id,
            "scene_id": episode.scene_id,
        }
        for episode in parsed
    ]
    records.sort(key=lambda record: record["example_id"])
    index = b"".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        for record in records
    )
    manifest = {
        "cell_size_m": 1.0,
        "dataset": "R2R",
        "ego_frame": "start-centered heading-normalized",
        "evidence_key": "oracle-t0-v1",
        "example_count": len(records),
        "grid_scale": 2,
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "mask_shape": [50, 50],
        "observation_count": len({record["observation_id"] for record in records}),
        "provenance": {},
        "schema_version": 1,
        "semantic_shape": [37, 50, 50],
        "split": "val_unseen",
        "target_frame": "level-local world-aligned",
    }
    return (
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode(),
        index,
        raw_split,
    )


def _raw_episode(
    episode_id: int,
    *,
    scene_id: str = "mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb",
    position: list[float] | None = None,
    rotation: list[float] | None = None,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "start_position": position or [1.0, 2.0, 3.0],
        "start_rotation": rotation or [0.0, 0.0, 0.0, 1.0],
        "instruction": {"ignored": True},
    }


def test_parse_start_episodes_matches_float64_and_episode_34_contract() -> None:
    episode = _raw_episode(
        34,
        scene_id="mp3d/TbHJrupSAjP/TbHJrupSAjP.glb",
        position=[1.1446900367736816, 3.4365439414978027, -9.352140426635742],
        rotation=[-0.0, 0.8660254037844385, -0.0, -0.5000000000000002],
    )
    raw_split = gzip.compress(json.dumps({"episodes": [episode]}).encode())

    parsed = cohort_module.parse_start_episodes(raw_split)

    assert len(parsed) == 1
    assert parsed[0].example_id == "R2R_val_unseen_34"
    assert parsed[0].scene_id == "TbHJrupSAjP"
    assert parsed[0].observation_id == "409975fb26c71c8b50ec"
    assert parsed[0].start_position == (
        1.1446900367736816,
        3.4365439414978027,
        -9.352140426635742,
    )
    assert math.isclose(
        math.sqrt(sum(value * value for value in parsed[0].start_rotation)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_parse_start_episodes_normalizes_quaternion_within_exact_tolerance() -> None:
    raw_split = gzip.compress(
        json.dumps(
            {
                "episodes": [
                    _raw_episode(1, rotation=[0.0, 0.0, 0.0, 0.99995])
                ]
            }
        ).encode()
    )

    parsed = cohort_module.parse_start_episodes(raw_split)

    assert parsed[0].start_rotation == (0.0, 0.0, 0.0, 1.0)


def test_parse_start_episodes_ignores_raw_top_level_metadata() -> None:
    raw_split = gzip.compress(
        json.dumps(
            {"episodes": [_raw_episode(1)], "metadata": {"ignored": True}}
        ).encode()
    )

    parsed = cohort_module.parse_start_episodes(raw_split)

    assert parsed[0].example_id == "R2R_val_unseen_1"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("start_position", [0.0, 1.0], "shape"),
        ("start_position", [0.0, math.inf, 1.0], "finite"),
        ("start_rotation", [0.0, 0.0, 0.0, 0.9998], "unit quaternion"),
    ],
)
def test_parse_start_episodes_rejects_invalid_vectors(
    field: str, value: object, message: str
) -> None:
    episode = _raw_episode(1)
    episode[field] = value

    with pytest.raises(ValueError, match=message):
        cohort_module.parse_start_episodes(
            gzip.compress(json.dumps({"episodes": [episode]}).encode())
        )


def test_reconstruct_source_observations_couples_aliases_and_artifacts() -> None:
    episodes = [
        _raw_episode(2),
        _raw_episode(1),
        _raw_episode(3, position=[4.0, 5.0, 6.0]),
    ]
    manifest, index, raw_split = _source_buffers(episodes)

    observations = reconstruct_source_observations(manifest, index, raw_split)
    by_aliases = {observation.example_ids: observation for observation in observations}

    assert len(observations) == 2
    sibling_aliases = ("R2R_val_unseen_1", "R2R_val_unseen_2")
    assert by_aliases[sibling_aliases].artifact_sha256 == "a" * 64
    assert ("R2R_val_unseen_3",) in by_aliases


def test_reconstruct_source_observations_rejects_observation_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episodes = [_raw_episode(1), _raw_episode(2, position=[4.0, 5.0, 6.0])]
    manifest, index, raw_split = _source_buffers(episodes)
    monkeypatch.setattr(
        cohort_module.StartEpisode,
        "observation_id",
        property(lambda _self: "0" * 20),
    )

    with pytest.raises(RuntimeError, match="observation ID collision"):
        reconstruct_source_observations(manifest, index, raw_split)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("alias", "alias"),
        ("observation", "observation"),
        ("scene", "scene"),
        ("artifact", "artifact"),
        ("extra_index_field", "fields"),
        ("extra_manifest_field", "fields"),
        ("manifest_index_hash", "index SHA-256"),
        ("manifest_count", "example_count"),
    ],
)
def test_reconstruct_source_observations_fails_closed(
    mutation: str, message: str
) -> None:
    manifest_bytes, index_bytes, raw_split = _source_buffers([_raw_episode(1)])
    manifest = json.loads(manifest_bytes)
    records = [json.loads(line) for line in index_bytes.splitlines()]
    if mutation == "alias":
        records[0]["example_id"] = "R2R_val_unseen_9"
    elif mutation == "observation":
        records[0]["observation_id"] = "0" * 20
    elif mutation == "scene":
        records[0]["scene_id"] = "8194nk5LbLH"
    elif mutation == "artifact":
        records.append({**records[0], "example_id": "R2R_val_unseen_2", "artifact_sha256": "b" * 64})
        manifest["example_count"] = 2
    elif mutation == "extra_index_field":
        records[0]["extra"] = 1
    elif mutation == "extra_manifest_field":
        manifest["extra"] = 1
    elif mutation == "manifest_index_hash":
        manifest["index_sha256"] = "0" * 64
    elif mutation == "manifest_count":
        manifest["example_count"] = 2
    index_bytes = b"".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        for record in records
    )
    if mutation not in {"manifest_index_hash"}:
        manifest["index_sha256"] = hashlib.sha256(index_bytes).hexdigest()
    manifest_bytes = json.dumps(
        manifest, separators=(",", ":"), sort_keys=True
    ).encode()

    with pytest.raises(ValueError, match=message):
        reconstruct_source_observations(manifest_bytes, index_bytes, raw_split)


def test_build_cohort_from_sources_hashes_each_exact_buffer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, index, raw_split = _source_buffers([_raw_episode(1)])
    paths = {
        "manifest": tmp_path / "manifest.json",
        "index": tmp_path / "index.jsonl",
        "raw": tmp_path / "split.json.gz",
    }
    for key, content in zip(paths, (manifest, index, raw_split)):
        paths[key].write_bytes(content)
    monkeypatch.setattr(cohort_module, "EVIDENCE_MANIFEST_PATH", paths["manifest"])
    monkeypatch.setattr(cohort_module, "EVIDENCE_INDEX_PATH", paths["index"])
    monkeypatch.setattr(cohort_module, "RAW_SPLIT_PATH", paths["raw"])
    monkeypatch.setattr(
        cohort_module, "EVIDENCE_MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest()
    )
    monkeypatch.setattr(
        cohort_module, "EVIDENCE_INDEX_SHA256", hashlib.sha256(index).hexdigest()
    )
    monkeypatch.setattr(
        cohort_module, "RAW_SPLIT_SHA256", hashlib.sha256(raw_split).hexdigest()
    )
    monkeypatch.setattr(cohort_module, "EXPECTED_EXAMPLE_COUNT", 1)
    monkeypatch.setattr(cohort_module, "EXPECTED_OBSERVATION_COUNT", 1)
    monkeypatch.setattr(cohort_module, "EXPECTED_SCENE_COUNT", 1)
    monkeypatch.setattr(cohort_module, "TARGET_COUNT", 1)
    monkeypatch.setattr(cohort_module, "OFFICIAL_SCENE_COUNTS", {"2azQ1b91cZZ": 1})
    monkeypatch.setattr(cohort_module, "OFFICIAL_SCENE_QUOTAS", {"2azQ1b91cZZ": 1})
    reads: list[Path] = []
    original_read_bytes = Path.read_bytes

    def recording_read_bytes(path: Path) -> bytes:
        reads.append(path)
        content = original_read_bytes(path)
        path.unlink()
        return content

    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)

    def fail_npz_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("NPZ payload must not be read")

    monkeypatch.setattr(cohort_module.np, "load", fail_npz_read)

    artifacts = build_cohort_from_sources(git_commit="1" * 40)

    assert reads == [paths["manifest"], paths["index"], paths["raw"]]
    assert not any(path.exists() for path in paths.values())
    assert len(artifacts.cohort.observations) == 1
    assert json.loads(artifacts.manifest_json)["git_commit"] == "1" * 40


def test_build_cohort_from_sources_rejects_a_pinned_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_manifest = tmp_path / "manifest.json"
    bad_manifest.write_bytes(b"{}")
    monkeypatch.setattr(cohort_module, "EVIDENCE_MANIFEST_PATH", bad_manifest)
    monkeypatch.setattr(cohort_module, "EVIDENCE_MANIFEST_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="manifest SHA-256"):
        build_cohort_from_sources(git_commit="1" * 40)


def test_real_sources_reconstruct_frozen_population_and_episode_34() -> None:
    artifacts = build_cohort_from_sources(git_commit="1" * 40)
    manifest = json.loads(artifacts.manifest_json)
    episode_34 = next(
        observation
        for observation in artifacts.population
        if "R2R_val_unseen_34" in observation.example_ids
    )

    assert len(artifacts.population) == 393
    assert sum(len(item.example_ids) for item in artifacts.population) == 1839
    assert len({item.scene_id for item in artifacts.population}) == 11
    assert artifacts.cohort.scene_quotas == OFFICIAL_SCENE_QUOTAS
    assert episode_34.observation_id == "409975fb26c71c8b50ec"
    assert manifest["population"]["scene_observation_counts"] == dict(
        OFFICIAL_SCENE_COUNTS
    )


def _cohort_artifacts(git_commit: str = "1" * 40) -> cohort_module.CohortArtifacts:
    observation = _observation()
    cohort = build_cohort((observation,), target_count=1)
    manifest = {
        "cohort_id": cohort_module.COHORT_ID,
        "files": {
            "cohort.jsonl": {
                "byte_length": len(cohort.cohort_jsonl),
                "row_count": 1,
                "sha256": cohort.cohort_sha256,
            }
        },
        "git_commit": git_commit,
        "population": {
            "example_count": 1,
            "observation_count": 1,
            "scene_count": 1,
            "scene_observation_counts": {"2azQ1b91cZZ": 1},
        },
        "schema_version": 1,
        "selection": {
            "algorithm": cohort_module.ALGORITHM,
            "domain_hex": cohort_module.SELECTION_DOMAIN.hex(),
            "scene_quotas": {"2azQ1b91cZZ": 1},
            "scene_selected_counts": {"2azQ1b91cZZ": 1},
            "selected_example_count": 1,
            "selected_observation_count": 1,
            "selected_scene_count": 1,
            "selection_sha256": cohort.selection_sha256,
            "target_observation_count": 1,
        },
        "source": {
            "dataset": cohort_module.DATASET,
            "evidence_index": cohort_module.EVIDENCE_INDEX_PATH.as_posix(),
            "evidence_index_sha256": cohort_module.EVIDENCE_INDEX_SHA256,
            "evidence_key": cohort_module.EVIDENCE_KEY,
            "evidence_manifest": cohort_module.EVIDENCE_MANIFEST_PATH.as_posix(),
            "evidence_manifest_sha256": cohort_module.EVIDENCE_MANIFEST_SHA256,
            "evidence_root": cohort_module.EVIDENCE_ROOT.as_posix(),
            "raw_split": cohort_module.RAW_SPLIT_PATH.as_posix(),
            "raw_split_sha256": cohort_module.RAW_SPLIT_SHA256,
            "split": cohort_module.SPLIT,
        },
    }
    manifest_json = (
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode()
    return cohort_module.CohortArtifacts((observation,), cohort, manifest_json)


def _write_cohort_directory(
    output_dir: Path, artifacts: cohort_module.CohortArtifacts
) -> None:
    output_dir.mkdir()
    (output_dir / "cohort.jsonl").write_bytes(artifacts.cohort_jsonl)
    (output_dir / "manifest.json").write_bytes(artifacts.manifest_json)


def test_validate_cohort_directory_rebuilds_exact_expected_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _cohort_artifacts()
    output_dir = tmp_path / "cohort"
    _write_cohort_directory(output_dir, artifacts)
    calls: list[str] = []

    def rebuild(*, git_commit: str) -> cohort_module.CohortArtifacts:
        calls.append(git_commit)
        return artifacts

    monkeypatch.setattr(cohort_module, "build_cohort_from_sources", rebuild)

    validate_cohort_directory(output_dir, expected_git_commit="1" * 40)

    assert calls == ["1" * 40]


@pytest.mark.parametrize(
    "mutation",
    [
        "cohort_bytes",
        "row_order",
        "manifest_unknown",
        "manifest_missing",
        "manifest_count",
        "manifest_source_hash",
        "manifest_commit",
        "extra_file",
        "missing_file",
        "symlink_member",
    ],
)
def test_validate_cohort_directory_rejects_every_package_drift(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _cohort_artifacts()
    output_dir = tmp_path / "cohort"
    _write_cohort_directory(output_dir, artifacts)
    monkeypatch.setattr(
        cohort_module,
        "build_cohort_from_sources",
        lambda *, git_commit: artifacts,
    )
    if mutation == "cohort_bytes":
        (output_dir / "cohort.jsonl").write_bytes(artifacts.cohort_jsonl + b" ")
    elif mutation == "row_order":
        row = artifacts.cohort_jsonl[:-1]
        (output_dir / "cohort.jsonl").write_bytes(row + b"\n" + row + b"\n")
    elif mutation.startswith("manifest_"):
        manifest = json.loads(artifacts.manifest_json)
        if mutation == "manifest_unknown":
            manifest["unknown"] = True
        elif mutation == "manifest_missing":
            del manifest["selection"]
        elif mutation == "manifest_count":
            manifest["files"]["cohort.jsonl"]["row_count"] = 2
        elif mutation == "manifest_source_hash":
            manifest["source"]["raw_split_sha256"] = "0" * 64
        elif mutation == "manifest_commit":
            manifest["git_commit"] = "2" * 40
        (output_dir / "manifest.json").write_bytes(
            (
                json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True)
                + "\n"
            ).encode()
        )
    elif mutation == "extra_file":
        (output_dir / "extra").write_bytes(b"")
    elif mutation == "missing_file":
        (output_dir / "manifest.json").unlink()
    elif mutation == "symlink_member":
        (output_dir / "manifest.json").unlink()
        (output_dir / "manifest.json").symlink_to(tmp_path / "elsewhere")

    with pytest.raises((ValueError, FileNotFoundError)):
        validate_cohort_directory(output_dir, expected_git_commit="1" * 40)


def test_validator_requires_external_expected_commit_before_source_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _cohort_artifacts()
    output_dir = tmp_path / "cohort"
    _write_cohort_directory(output_dir, artifacts)

    def fail_build(*, git_commit: str) -> cohort_module.CohortArtifacts:
        raise AssertionError(
            f"invalid expected commit reached source rebuild: {git_commit}"
        )

    monkeypatch.setattr(
        cohort_module,
        "build_cohort_from_sources",
        fail_build,
    )

    with pytest.raises(ValueError, match="expected_git_commit"):
        validate_cohort_directory(output_dir, expected_git_commit="bad")


def test_validator_rejects_wrong_valid_external_commit_and_output_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _cohort_artifacts()
    actual = tmp_path / "actual"
    _write_cohort_directory(actual, artifacts)
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing", target_is_directory=True)
    monkeypatch.setattr(
        cohort_module,
        "build_cohort_from_sources",
        lambda *, git_commit: _cohort_artifacts(git_commit),
    )

    with pytest.raises(ValueError, match="manifest.json"):
        validate_cohort_directory(actual, expected_git_commit="2" * 40)
    with pytest.raises(ValueError, match="real directory"):
        validate_cohort_directory(linked, expected_git_commit="1" * 40)
    with pytest.raises(ValueError, match="real directory"):
        validate_cohort_directory(broken, expected_git_commit="1" * 40)


def test_publish_cohort_validates_temp_rechecks_git_and_atomically_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _cohort_artifacts()
    output_dir = tmp_path / "cohort"
    events: list[str] = []
    monkeypatch.setattr(
        cohort_module,
        "build_cohort_from_sources",
        lambda *, git_commit: artifacts,
    )
    monkeypatch.setattr(
        cohort_module,
        "_preflight_git_commit",
        lambda: (events.append("preflight"), "1" * 40)[1],
    )
    original_validate = cohort_module.validate_cohort_directory

    def validate(path: Path, *, expected_git_commit: str) -> None:
        events.append("validate")
        original_validate(path, expected_git_commit=expected_git_commit)

    monkeypatch.setattr(cohort_module, "validate_cohort_directory", validate)
    original_rename = cohort_module._rename_noreplace

    def rename(source: Path, destination: Path) -> None:
        events.append("rename")
        original_rename(source, destination)

    monkeypatch.setattr(cohort_module, "_rename_noreplace", rename)

    publish_cohort(output_dir, artifacts)

    assert events == ["validate", "preflight", "rename"]
    assert (output_dir / "cohort.jsonl").read_bytes() == artifacts.cohort_jsonl
    assert (output_dir / "manifest.json").read_bytes() == artifacts.manifest_json
    assert tuple(tmp_path.glob(".cohort.tmp-*")) == ()


def test_publish_cohort_race_preserves_existing_destination_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _cohort_artifacts()
    output_dir = tmp_path / "cohort"
    monkeypatch.setattr(
        cohort_module,
        "build_cohort_from_sources",
        lambda *, git_commit: artifacts,
    )
    monkeypatch.setattr(
        cohort_module, "_preflight_git_commit", lambda: "1" * 40
    )
    original_rename = cohort_module._rename_noreplace

    def race(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "winner").write_bytes(b"preserve")
        original_rename(source, destination)

    monkeypatch.setattr(cohort_module, "_rename_noreplace", race)

    with pytest.raises(FileExistsError):
        publish_cohort(output_dir, artifacts)

    assert (output_dir / "winner").read_bytes() == b"preserve"
    assert tuple(tmp_path.glob(".cohort.tmp-*")) == ()


@pytest.mark.parametrize("failure", ["validate", "dirty", "head"])
def test_publish_cohort_failure_removes_only_temporary_directory(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _cohort_artifacts()
    output_dir = tmp_path / "cohort"
    if failure == "validate":
        monkeypatch.setattr(
            cohort_module,
            "validate_cohort_directory",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid")),
        )
    else:
        monkeypatch.setattr(
            cohort_module,
            "build_cohort_from_sources",
            lambda *, git_commit: artifacts,
        )
        if failure == "dirty":
            monkeypatch.setattr(
                cohort_module,
                "_preflight_git_commit",
                lambda: (_ for _ in ()).throw(RuntimeError("dirty")),
            )
        else:
            monkeypatch.setattr(
                cohort_module, "_preflight_git_commit", lambda: "2" * 40
            )

    with pytest.raises((RuntimeError, ValueError)):
        publish_cohort(output_dir, artifacts)

    assert not output_dir.exists()
    assert tuple(tmp_path.glob(".cohort.tmp-*")) == ()


def test_post_rename_fsync_failure_preserves_published_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _cohort_artifacts()
    output_dir = tmp_path / "cohort"
    monkeypatch.setattr(
        cohort_module,
        "build_cohort_from_sources",
        lambda *, git_commit: artifacts,
    )
    monkeypatch.setattr(
        cohort_module, "_preflight_git_commit", lambda: "1" * 40
    )
    calls = 0
    original_fsync = cohort_module._fsync_directory

    def fsync(directory: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("parent fsync failed")
        original_fsync(directory)

    monkeypatch.setattr(cohort_module, "_fsync_directory", fsync)

    with pytest.raises(OSError, match="parent fsync"):
        publish_cohort(output_dir, artifacts)

    assert output_dir.is_dir()
    assert (output_dir / "cohort.jsonl").read_bytes() == artifacts.cohort_jsonl
    assert tuple(tmp_path.glob(".cohort.tmp-*")) == ()


def test_rename_noreplace_uses_linux_abi_and_preserves_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    calls: list[tuple[object, ...]] = []

    class FakeRenameAt2:
        argtypes: object = None
        restype: object = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            cohort_module.ctypes.set_errno(errno.EEXIST)
            return -1

    renameat2 = FakeRenameAt2()
    monkeypatch.setattr(
        cohort_module.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(renameat2=renameat2),
    )

    with pytest.raises(FileExistsError):
        cohort_module._rename_noreplace(source, destination)

    assert calls == [
        (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    ]
    assert source.is_dir()
    assert destination.is_dir()


def test_fixed_cli_rejects_scientific_and_path_overrides() -> None:
    for override in (
        ["--output-dir", "elsewhere"],
        ["--target-count", "1"],
        ["--source", "elsewhere"],
        ["--overwrite"],
        ["--resume"],
        ["--limit", "1"],
    ):
        with pytest.raises(SystemExit):
            cohort_module.CohortArgs(underscores_to_dashes=True).parse_args(override)


def test_run_requires_absent_output_then_clean_git_before_source_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "official"
    monkeypatch.setattr(cohort_module, "OFFICIAL_OUTPUT_DIR", output_dir)
    events: list[str] = []
    monkeypatch.setattr(
        cohort_module,
        "_preflight_git_commit",
        lambda: (events.append("preflight"), "1" * 40)[1],
    )
    monkeypatch.setattr(
        cohort_module,
        "build_cohort_from_sources",
        lambda **_kwargs: (events.append("source"), _cohort_artifacts())[1],
    )
    monkeypatch.setattr(
        cohort_module,
        "publish_cohort",
        lambda *_args: events.append("publish"),
    )

    output_dir.mkdir()
    with pytest.raises(FileExistsError):
        cohort_module._run()
    assert events == []

    output_dir.rmdir()
    output_dir.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(FileExistsError):
        cohort_module._run()
    assert events == []

    output_dir.unlink()
    monkeypatch.setattr(
        cohort_module,
        "_preflight_git_commit",
        lambda: (_ for _ in ()).throw(RuntimeError("dirty")),
    )
    with pytest.raises(RuntimeError, match="dirty"):
        cohort_module._run()
    assert events == []


def test_run_records_exact_clean_head_and_publishes_matching_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "official"
    monkeypatch.setattr(cohort_module, "OFFICIAL_OUTPUT_DIR", output_dir)
    commits: list[str] = []
    artifacts = _cohort_artifacts("a" * 40)
    monkeypatch.setattr(
        cohort_module, "_preflight_git_commit", lambda: "a" * 40
    )

    def build(*, git_commit: str) -> cohort_module.CohortArtifacts:
        commits.append(git_commit)
        return artifacts

    monkeypatch.setattr(cohort_module, "build_cohort_from_sources", build)
    monkeypatch.setattr(
        cohort_module,
        "publish_cohort",
        lambda _path, actual: commits.append(
            json.loads(actual.manifest_json)["git_commit"]
        ),
    )

    cohort_module._run()

    assert commits == ["a" * 40, "a" * 40]


def test_clean_git_preflight_runs_status_before_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def completed(
        command: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        stdout = "" if command[1] == "status" else "c" * 40 + "\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(cohort_module.subprocess, "run", completed)

    assert cohort_module._preflight_git_commit() == "c" * 40
    assert calls == [
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        ("git", "rev-parse", "HEAD"),
    ]
