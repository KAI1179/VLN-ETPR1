import json
from pathlib import Path

import numpy as np
import pytest

from vlnce_baselines.models.etp_llm.llm_grid_evidence import (
    EvidenceAssignmentKind,
    EvidenceAssignments,
    EvidenceEpisode,
    GridEvidence,
    GridEvidenceIndex,
)


def _evidence(*, target_channel: int = 1) -> GridEvidence:
    ego_semantic = np.zeros((37, 50, 50), dtype=np.bool_)
    ego_observed = np.zeros((50, 50), dtype=np.bool_)
    ego_free = np.zeros((50, 50), dtype=np.bool_)
    ego_semantic[1, 25, 25] = True
    ego_observed[25, 24:26] = True
    ego_free[25, 24] = True

    target_semantic = np.zeros((37, 50, 50), dtype=np.bool_)
    target_observed = np.zeros((50, 50), dtype=np.bool_)
    target_free = np.zeros((50, 50), dtype=np.bool_)
    target_semantic[target_channel, 4, 5] = True
    target_observed[4, 4:6] = True
    target_free[4, 4] = True
    return GridEvidence(
        ego_semantic_grid=ego_semantic,
        ego_observed_mask=ego_observed,
        ego_free_mask=ego_free,
        target_semantic_grid=target_semantic,
        target_observed_mask=target_observed,
        target_free_mask=target_free,
        start_position=(4.0, 5.0),
        start_direction=(0.0, 1.0),
    )


def _write_observation(
    root: Path,
    key: str,
    scene_id: str,
    observation_id: str,
    *,
    target_channel: int = 1,
) -> None:
    _evidence(target_channel=target_channel).save(
        GridEvidenceIndex.observation_path(
            root,
            key,
            "R2R",
            "val_unseen",
            scene_id,
            observation_id,
        )
    )


def _index(tmp_path: Path) -> GridEvidenceIndex:
    key = "oracle-v1"
    episodes = (
        EvidenceEpisode("R2R_val_unseen_1a", "scene-a", "obs-a1"),
        EvidenceEpisode("R2R_val_unseen_1b", "scene-a", "obs-a1"),
        EvidenceEpisode("R2R_val_unseen_2", "scene-a", "obs-a2"),
        EvidenceEpisode("R2R_val_unseen_3", "scene-b", "obs-b1"),
        EvidenceEpisode("R2R_val_unseen_4", "scene-b", "obs-b2"),
    )
    for index, (scene_id, observation_id) in enumerate((
        ("scene-a", "obs-a1"),
        ("scene-a", "obs-a2"),
        ("scene-b", "obs-b1"),
        ("scene-b", "obs-b2"),
    )):
        _write_observation(
            tmp_path,
            key,
            scene_id,
            observation_id,
            target_channel=index + 1,
        )
    return GridEvidenceIndex.create(
        tmp_path,
        key,
        "R2R",
        "val_unseen",
        episodes,
        provenance={"observation_time": "t0", "sensor": "semantic+depth"},
    )


def test_grid_evidence_round_trip_is_pickle_free_and_strict(tmp_path: Path) -> None:
    path = tmp_path / "evidence.npz"
    expected = _evidence()

    expected.save(path)
    actual = GridEvidence.load(path)

    np.testing.assert_array_equal(actual.ego_semantic_grid, expected.ego_semantic_grid)
    np.testing.assert_array_equal(
        actual.target_observed_mask, expected.target_observed_mask
    )
    np.testing.assert_array_equal(
        actual.ego_unknown_mask, np.logical_not(expected.ego_observed_mask)
    )
    assert actual.start_position == (4.0, 5.0)
    with np.load(path, allow_pickle=False) as artifact:
        assert all(artifact[name].dtype != object for name in artifact.files)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda values: values["ego_free_mask"].__setitem__((0, 0), True),
            "free cells must be observed",
        ),
        (
            lambda values: values["ego_semantic_grid"].__setitem__((1, 0, 0), True),
            "semantic cells must be observed",
        ),
    ],
)
def test_grid_evidence_rejects_free_unknown_semantic_conflicts(
    mutate,
    message: str,
) -> None:
    evidence = _evidence()
    values = {
        field: np.array(getattr(evidence, field), copy=True)
        for field in (
            "ego_semantic_grid",
            "ego_observed_mask",
            "ego_free_mask",
            "target_semantic_grid",
            "target_observed_mask",
            "target_free_mask",
        )
    }
    mutate(values)

    with pytest.raises(ValueError, match=message):
        GridEvidence(
            **values,
            start_position=evidence.start_position,
            start_direction=evidence.start_direction,
        )


def test_grid_evidence_allows_region_semantics_on_free_cells() -> None:
    evidence = _evidence()
    ego_free = evidence.ego_free_mask.copy()
    ego_free[25, 25] = True

    GridEvidence(
        ego_semantic_grid=evidence.ego_semantic_grid,
        ego_observed_mask=evidence.ego_observed_mask,
        ego_free_mask=ego_free,
        target_semantic_grid=evidence.target_semantic_grid,
        target_observed_mask=evidence.target_observed_mask,
        target_free_mask=evidence.target_free_mask,
        start_position=evidence.start_position,
        start_direction=evidence.start_direction,
    )


def test_prompt_block_uses_only_target_frame_and_excludes_ids() -> None:
    evidence = _evidence(target_channel=2)

    block = evidence.prompt_block()

    assert '"frame":"level-local world-aligned"' in block
    assert '"door":[[4,5,5]]' in block
    assert '"free_runs":[[4,4,4]]' in block
    assert '"chair"' not in block
    assert "scene-secret" not in block
    assert "observation-secret" not in block


def test_index_deduplicates_observations_and_pins_hashes(tmp_path: Path) -> None:
    index = _index(tmp_path)

    assert index.example_count == 5
    assert index.observation_count == 4
    assert index.episode("R2R_val_unseen_1b").observation_id == "obs-a1"
    assert index.load_evidence("R2R_val_unseen_2").target_semantic_grid[2, 4, 5]
    assert index.load_observation("obs-a2").target_semantic_grid[2, 4, 5]
    split_dir = GridEvidenceIndex.split_dir(tmp_path, "oracle-v1", "R2R", "val_unseen")
    manifest = json.loads((split_dir / "manifest.json").read_text())
    assert manifest["index_sha256"] == index.index_sha256
    assert manifest["observation_count"] == 4

    artifact_path = GridEvidenceIndex.observation_path(
        tmp_path,
        "oracle-v1",
        "R2R",
        "val_unseen",
        "scene-a",
        "obs-a2",
    )
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        index.load_evidence("R2R_val_unseen_2")


def test_index_rejects_tampered_index_and_extra_manifest_fields(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    split_dir = GridEvidenceIndex.split_dir(tmp_path, "oracle-v1", "R2R", "val_unseen")
    index_path = split_dir / "index.jsonl"
    index_path.write_text(index_path.read_text() + "\n")
    with pytest.raises(ValueError, match="index SHA-256 mismatch"):
        GridEvidenceIndex.load(tmp_path, "oracle-v1", "R2R", "val_unseen")

    index = _index(tmp_path)
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest has invalid fields"):
        GridEvidenceIndex.load(tmp_path, "oracle-v1", "R2R", "val_unseen")
    assert index.manifest_sha256


@pytest.mark.parametrize(
    "kind",
    ["matched", "null", "within-scene", "global"],
)
def test_assignments_are_deterministic_and_observation_level(
    tmp_path: Path,
    kind: EvidenceAssignmentKind,
) -> None:
    index = _index(tmp_path)

    first = EvidenceAssignments.build(index, kind, seed=42)
    second = EvidenceAssignments.build(index, kind, seed=42)

    assert first == second
    assert first.donor_for("R2R_val_unseen_1a") == first.donor_for("R2R_val_unseen_1b")
    scenes = {episode.observation_id: episode.scene_id for episode in index.episodes}
    for entry in first.entries:
        donor = entry.donor_observation_id
        if kind == "matched":
            assert donor == entry.observation_id
        elif kind == "null":
            assert donor is None
        elif kind == "within-scene":
            assert donor is not None
            assert donor != entry.observation_id
            assert scenes[donor] == scenes[entry.observation_id]
        else:
            assert donor is not None
            assert donor != entry.observation_id
            assert scenes[donor] != scenes[entry.observation_id]


def test_assignment_round_trip_validates_content_hash(tmp_path: Path) -> None:
    index = _index(tmp_path)
    assignments = EvidenceAssignments.build(index, "global", seed=7)
    path = tmp_path / "assignments.jsonl"

    digest = assignments.save(path)

    assert EvidenceAssignments.load(path, index, expected_sha256=digest) == assignments
    path.write_text(path.read_text().replace('"seed":7', '"seed":8'))
    with pytest.raises(ValueError, match="assignment SHA-256 mismatch"):
        EvidenceAssignments.load(path, index, expected_sha256=digest)


def test_assignment_rejects_impossible_scene_controls(tmp_path: Path) -> None:
    key = "singleton"
    _write_observation(tmp_path, key, "scene-a", "obs-a")
    index = GridEvidenceIndex.create(
        tmp_path,
        key,
        "R2R",
        "val_unseen",
        [EvidenceEpisode("example-a", "scene-a", "obs-a")],
        provenance={},
    )

    with pytest.raises(ValueError, match="requires two observations"):
        EvidenceAssignments.build(index, "within-scene", seed=0)
    with pytest.raises(ValueError, match="at least two scenes"):
        EvidenceAssignments.build(index, "global", seed=0)
