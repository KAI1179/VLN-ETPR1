from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

import pytest

import prior.analyze.d2026_07_28.rgbd_segmenter_cohort as cohort_module
from prior.analyze.d2026_07_28.rgbd_segmenter_cohort import (
    OFFICIAL_SCENE_COUNTS,
    OFFICIAL_SCENE_QUOTAS,
    SELECTION_DOMAIN,
    CanonicalObservation,
    build_cohort,
    hamilton_apportion,
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
