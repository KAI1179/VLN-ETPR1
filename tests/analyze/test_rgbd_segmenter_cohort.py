from __future__ import annotations

import hashlib
import gzip
import json
import math
from dataclasses import replace
from pathlib import Path

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
    reconstruct_source_observations,
    select_observations,
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
