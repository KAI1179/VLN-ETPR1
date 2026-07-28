"""Deterministic pure selection for the preregistered RGB-D benchmark cohort."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import re
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Tuple

import numpy as np

from vlnce_baselines.models.etp_llm.llm_grid_oracle_cache import (
    StartEpisode,
    group_start_episodes,
)

SELECTION_DOMAIN = b"etp-r1:rgbd-segmenter-benchmark:cohort-v1\0"
COHORT_ID = "r2r-val-unseen-50-v1"
ALGORITHM = "hamilton-scene-proportional-sha256-selection-key-v1"
EVIDENCE_KEY = "oracle-t0-v1"
DATASET = "R2R"
SPLIT = "val_unseen"
EVIDENCE_ROOT = Path("data/llm_grid_oracle_evidence")
EVIDENCE_MANIFEST_PATH = (
    EVIDENCE_ROOT / EVIDENCE_KEY / "r2r" / SPLIT / "manifest.json"
)
EVIDENCE_INDEX_PATH = EVIDENCE_MANIFEST_PATH.with_name("index.jsonl")
RAW_SPLIT_PATH = Path(
    "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz"
)
EVIDENCE_MANIFEST_SHA256 = (
    "be2d25890c01234dd1bb8191af6c1d87121330991299287b635205f2ebf0327a"
)
EVIDENCE_INDEX_SHA256 = (
    "0acc9d18aea6d369db4b0a73f8ab3eebbe8fab51257b34614d1a7c407443acb9"
)
RAW_SPLIT_SHA256 = (
    "6140b46759fe332ee96aa849d4bb64e1c1829b8f65acee127355040a6ee23484"
)
EXPECTED_EXAMPLE_COUNT = 1839
EXPECTED_OBSERVATION_COUNT = 393
EXPECTED_SCENE_COUNT = 11
TARGET_COUNT = 50

OFFICIAL_SCENE_COUNTS: Mapping[str, int] = MappingProxyType({
    "2azQ1b91cZZ": 63,
    "8194nk5LbLH": 7,
    "EU6Fwq7SyZv": 33,
    "QUCTc6BB5sX": 64,
    "TbHJrupSAjP": 55,
    "X7HyMhZNoso": 33,
    "Z6MFQCViBuw": 28,
    "oLBMNvg9in8": 41,
    "pLe4wQe7qrG": 6,
    "x8F5xyUWy9e": 21,
    "zsNo4HB9uLZ": 42,
})
OFFICIAL_SCENE_QUOTAS: Mapping[str, int] = MappingProxyType({
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
})

_LOWER_HEX_20 = re.compile(r"[0-9a-f]{20}")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")
_LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}")
_SCENE_ID = re.compile(r"[A-Za-z0-9]{11}")
_EXAMPLE_ID = re.compile(r"R2R_val_unseen_(?:0|[1-9][0-9]*)")
_EVIDENCE_MANIFEST_KEYS = {
    "schema_version",
    "evidence_key",
    "dataset",
    "split",
    "semantic_shape",
    "mask_shape",
    "grid_scale",
    "cell_size_m",
    "ego_frame",
    "target_frame",
    "example_count",
    "observation_count",
    "index_sha256",
    "provenance",
}
_EVIDENCE_INDEX_KEYS = {
    "example_id",
    "scene_id",
    "observation_id",
    "artifact_sha256",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _finite_floats(value: Tuple[float, ...], *, length: int, name: str) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != length
        or any(not isinstance(item, float) or not math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{name} must contain exactly {length} finite floats")


@dataclass(frozen=True)
class CanonicalObservation:
    """One strictly validated source observation."""

    artifact_sha256: str
    example_ids: Tuple[str, ...]
    observation_id: str
    scene_id: str
    start_position: Tuple[float, float, float]
    start_rotation: Tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact_sha256, str)
            or _LOWER_HEX_64.fullmatch(self.artifact_sha256) is None
        ):
            raise ValueError("artifact_sha256 must be 64 lowercase hex characters")
        if (
            not isinstance(self.observation_id, str)
            or _LOWER_HEX_20.fullmatch(self.observation_id) is None
        ):
            raise ValueError("observation_id must be 20 lowercase hex characters")
        if (
            not isinstance(self.scene_id, str)
            or _SCENE_ID.fullmatch(self.scene_id) is None
        ):
            raise ValueError("scene_id must be an 11-character MP3D scene ID")
        if (
            not isinstance(self.example_ids, tuple)
            or not self.example_ids
            or any(
                not isinstance(item, str) or _EXAMPLE_ID.fullmatch(item) is None
                for item in self.example_ids
            )
            or tuple(sorted(self.example_ids)) != self.example_ids
            or len(set(self.example_ids)) != len(self.example_ids)
        ):
            raise ValueError("example_ids must be unique sorted R2R val_unseen aliases")
        _finite_floats(self.start_position, length=3, name="start_position")
        _finite_floats(self.start_rotation, length=4, name="start_rotation")
        norm = math.sqrt(sum(value * value for value in self.start_rotation))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError("start_rotation must be a unit quaternion")

    def canonical_row_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "artifact_sha256": self.artifact_sha256,
            "example_ids": self.example_ids,
            "observation_id": self.observation_id,
            "scene_id": self.scene_id,
            "start_position": self.start_position,
            "start_rotation": self.start_rotation,
        })

    def canonical_selection_key_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "example_ids": self.example_ids,
            "observation_id": self.observation_id,
            "scene_id": self.scene_id,
            "start_position": self.start_position,
            "start_rotation": self.start_rotation,
        })

    def selection_digest(self) -> bytes:
        return sha256(SELECTION_DOMAIN + self.canonical_selection_key_bytes()).digest()


@dataclass(frozen=True)
class BuiltCohort:
    """Selected observations and their exact serialized package payload."""

    observations: Tuple[CanonicalObservation, ...]
    scene_quotas: Mapping[str, int]
    cohort_jsonl: bytes
    selection_sha256: str
    cohort_sha256: str


@dataclass(frozen=True)
class CohortArtifacts:
    """In-memory official package reconstructed from its pinned sources."""

    population: Tuple[CanonicalObservation, ...]
    cohort: BuiltCohort
    manifest_json: bytes

    @property
    def cohort_jsonl(self) -> bytes:
        return self.cohort.cohort_jsonl


def hamilton_apportion(
    scene_populations: Mapping[str, int], target_count: int
) -> dict[str, int]:
    """Apportion a target exactly, breaking equal remainder ties lexically."""

    if (
        not scene_populations
        or isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or target_count <= 0
    ):
        raise ValueError("target_count and scene populations must be positive")
    if any(
        not isinstance(scene, str)
        or not scene
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        for scene, count in scene_populations.items()
    ):
        raise ValueError("scene populations must have positive integer counts")

    population_count = sum(scene_populations.values())
    if target_count > population_count:
        raise ValueError("target_count exceeds the population")

    exact = {
        scene: Fraction(target_count * count, population_count)
        for scene, count in scene_populations.items()
    }
    quotas = {
        scene: value.numerator // value.denominator for scene, value in exact.items()
    }
    remaining = target_count - sum(quotas.values())
    ranked = sorted(
        exact,
        key=lambda scene: (-(exact[scene] - quotas[scene]), scene),
    )
    for scene in ranked[:remaining]:
        quotas[scene] += 1
    return {scene: quotas[scene] for scene in sorted(quotas)}


def select_observations(
    observations: Iterable[CanonicalObservation],
    scene_quotas: Mapping[str, int],
) -> Tuple[CanonicalObservation, ...]:
    """Select the digest-smallest observations within every scene."""

    population = tuple(observations)
    if not population:
        raise ValueError("observation population must not be empty")
    if not scene_quotas or any(
        not isinstance(scene_id, str)
        or isinstance(quota, bool)
        or not isinstance(quota, int)
        for scene_id, quota in scene_quotas.items()
    ):
        raise ValueError("scene quotas must map scene strings to integers")

    observation_ids = [item.observation_id for item in population]
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("duplicate observation ID")
    aliases = [alias for item in population for alias in item.example_ids]
    if len(set(aliases)) != len(aliases):
        raise ValueError("duplicate example alias")

    by_scene: dict[str, list[CanonicalObservation]] = {}
    for observation in population:
        by_scene.setdefault(observation.scene_id, []).append(observation)
    population_scenes = set(by_scene)
    quota_scenes = set(scene_quotas)
    if quota_scenes - population_scenes:
        raise ValueError("scene quotas contain an unknown scene")
    if population_scenes - quota_scenes:
        raise ValueError("scene quota is missing")

    selected = []
    for scene_id in sorted(by_scene):
        quota = scene_quotas[scene_id]
        if (
            isinstance(quota, bool)
            or not isinstance(quota, int)
            or quota < 0
            or quota > len(by_scene[scene_id])
        ):
            raise ValueError(f"invalid quota for scene {scene_id}")
        ranked = sorted(
            by_scene[scene_id],
            key=lambda item: (item.selection_digest(), item.observation_id),
        )
        selected.extend(ranked[:quota])
    if not selected:
        raise ValueError("scene quotas select no observations")
    return tuple(selected)


def build_cohort(
    observations: Iterable[CanonicalObservation], *, target_count: int
) -> BuiltCohort:
    """Build the exact JSONL and hashes for a deterministic cohort."""

    population = tuple(observations)
    scene_counts: dict[str, int] = {}
    for observation in population:
        scene_counts[observation.scene_id] = (
            scene_counts.get(observation.scene_id, 0) + 1
        )
    quotas = hamilton_apportion(scene_counts, target_count)
    selected = select_observations(population, quotas)
    rows = tuple(observation.canonical_row_bytes() for observation in selected)
    cohort_jsonl = b"".join(row + b"\n" for row in rows)
    return BuiltCohort(
        observations=selected,
        scene_quotas=MappingProxyType(quotas),
        cohort_jsonl=cohort_jsonl,
        selection_sha256=sha256(b"".join(rows)).hexdigest(),
        cohort_sha256=sha256(cohort_jsonl).hexdigest(),
    )


def _finite_vector(value: object, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def parse_start_episodes(raw_split: bytes) -> Tuple[StartEpisode, ...]:
    """Parse permitted episode fields from already-hashed compressed bytes."""

    try:
        payload = json.loads(gzip.decompress(raw_split))
    except (EOFError, OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("raw split must be valid gzip-compressed JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("raw split must be an object containing an episodes list")
    raw_payload: dict[str, object] = {
        str(key): value for key, value in payload.items()
    }
    raw_episodes = raw_payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("raw split episodes must be a non-empty list")

    episodes = []
    for index, raw_episode in enumerate(raw_episodes):
        if not isinstance(raw_episode, dict) or any(
            not isinstance(key, str) for key in raw_episode
        ):
            raise ValueError(f"raw split episodes[{index}] must be an object")
        episode: dict[str, object] = {
            str(key): value for key, value in raw_episode.items()
        }
        required = {"episode_id", "scene_id", "start_position", "start_rotation"}
        missing = required - set(episode)
        if missing:
            raise ValueError(
                f"raw split episodes[{index}] is missing {sorted(missing)[0]}"
            )
        position = _finite_vector(
            episode["start_position"],
            size=3,
            name=f"raw split episodes[{index}].start_position",
        )
        rotation = _finite_vector(
            episode["start_rotation"],
            size=4,
            name=f"raw split episodes[{index}].start_rotation",
        )
        norm = float(np.linalg.norm(rotation))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError(
                f"raw split episodes[{index}].start_rotation must be a unit "
                f"quaternion, got norm {norm}"
            )
        rotation = rotation / norm
        episodes.append(
            StartEpisode(
                dataset=DATASET,
                split=SPLIT,
                episode_id=str(episode["episode_id"]),
                scene_id=Path(str(episode["scene_id"])).stem,
                start_position=(
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                ),
                start_rotation=(
                    float(rotation[0]),
                    float(rotation[1]),
                    float(rotation[2]),
                    float(rotation[3]),
                ),
            )
        )
    aliases = [episode.example_id for episode in episodes]
    if len(set(aliases)) != len(aliases):
        raise ValueError("raw split contains a duplicate example alias")
    return tuple(episodes)


def _parse_evidence_manifest(manifest_bytes: bytes) -> Mapping[str, Any]:
    try:
        manifest = json.loads(manifest_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("evidence manifest must be valid UTF-8 JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != _EVIDENCE_MANIFEST_KEYS:
        raise ValueError("evidence manifest has invalid fields")
    expected: Mapping[str, object] = {
        "schema_version": 1,
        "evidence_key": EVIDENCE_KEY,
        "dataset": DATASET,
        "split": SPLIT,
        "semantic_shape": [37, 50, 50],
        "mask_shape": [50, 50],
        "grid_scale": 2,
        "cell_size_m": 1.0,
        "ego_frame": "start-centered heading-normalized",
        "target_frame": "level-local world-aligned",
    }
    mismatches = {
        key: (manifest[key], value)
        for key, value in expected.items()
        if type(manifest[key]) is not type(value) or manifest[key] != value
    }
    if mismatches:
        raise ValueError(f"evidence manifest contract mismatch: {mismatches}")
    for field in ("example_count", "observation_count"):
        value = manifest[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"evidence manifest {field} must be a positive integer")
    if (
        not isinstance(manifest["index_sha256"], str)
        or _LOWER_HEX_64.fullmatch(manifest["index_sha256"]) is None
    ):
        raise ValueError("evidence manifest index_sha256 is invalid")
    if not isinstance(manifest["provenance"], dict):
        raise ValueError("evidence manifest provenance must be an object")
    return manifest


def _parse_evidence_index(index_bytes: bytes) -> Tuple[Mapping[str, str], ...]:
    if not index_bytes or not index_bytes.endswith(b"\n"):
        raise ValueError("evidence index must be non-empty and end with a newline")
    records = []
    for line_number, line in enumerate(index_bytes.splitlines(), start=1):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(
                f"evidence index line {line_number} must be valid UTF-8 JSON"
            ) from error
        if not isinstance(record, dict) or set(record) != _EVIDENCE_INDEX_KEYS:
            raise ValueError(f"evidence index line {line_number} has invalid fields")
        if any(not isinstance(record[key], str) for key in _EVIDENCE_INDEX_KEYS):
            raise ValueError(f"evidence index line {line_number} has invalid values")
        if _EXAMPLE_ID.fullmatch(record["example_id"]) is None:
            raise ValueError(f"evidence index line {line_number} has invalid alias")
        if _SCENE_ID.fullmatch(record["scene_id"]) is None:
            raise ValueError(f"evidence index line {line_number} has invalid scene")
        if _LOWER_HEX_20.fullmatch(record["observation_id"]) is None:
            raise ValueError(
                f"evidence index line {line_number} has invalid observation ID"
            )
        if _LOWER_HEX_64.fullmatch(record["artifact_sha256"]) is None:
            raise ValueError(
                f"evidence index line {line_number} has invalid artifact SHA-256"
            )
        records.append(record)
    aliases = [record["example_id"] for record in records]
    if aliases != sorted(aliases) or len(set(aliases)) != len(aliases):
        raise ValueError("evidence index aliases must be unique and lexically sorted")
    return tuple(records)


def reconstruct_source_observations(
    manifest_bytes: bytes,
    index_bytes: bytes,
    raw_split_bytes: bytes,
) -> Tuple[CanonicalObservation, ...]:
    """Join strict evidence records to start episodes without opening paths."""

    manifest = _parse_evidence_manifest(manifest_bytes)
    index_sha256 = sha256(index_bytes).hexdigest()
    if manifest["index_sha256"] != index_sha256:
        raise ValueError("evidence manifest/index SHA-256 mismatch")
    records = _parse_evidence_index(index_bytes)
    if manifest["example_count"] != len(records):
        raise ValueError("evidence manifest example_count mismatch")
    observation_count = len({record["observation_id"] for record in records})
    if manifest["observation_count"] != observation_count:
        raise ValueError("evidence manifest observation_count mismatch")

    artifact_contract: dict[str, Tuple[str, str]] = {}
    for record in records:
        contract = (record["scene_id"], record["artifact_sha256"])
        previous = artifact_contract.setdefault(record["observation_id"], contract)
        if previous != contract:
            raise ValueError(
                "inconsistent observation scene or artifact hash for "
                f"{record['observation_id']}"
            )

    episodes = parse_start_episodes(raw_split_bytes)
    groups = group_start_episodes(episodes)
    records_by_alias = {record["example_id"]: record for record in records}
    episode_aliases = {episode.example_id for episode in episodes}
    if set(records_by_alias) != episode_aliases:
        raise ValueError("raw split/evidence index alias mismatch")

    observations = []
    for group in groups:
        aliases = tuple(sorted(episode.example_id for episode in group.episodes))
        group_records = tuple(records_by_alias[alias] for alias in aliases)
        if any(
            record["observation_id"] != group.observation_id
            for record in group_records
        ):
            raise ValueError(
                f"raw/index observation mismatch for {group.observation_id}"
            )
        if any(record["scene_id"] != group.scene_id for record in group_records):
            raise ValueError(f"raw/index scene mismatch for {group.observation_id}")
        artifact_hashes = {
            record["artifact_sha256"] for record in group_records
        }
        if len(artifact_hashes) != 1:
            raise ValueError(
                f"raw/index artifact mismatch for {group.observation_id}"
            )
        observations.append(
            CanonicalObservation(
                artifact_sha256=next(iter(artifact_hashes)),
                example_ids=aliases,
                observation_id=group.observation_id,
                scene_id=group.scene_id,
                start_position=group.start_position,
                start_rotation=group.start_rotation,
            )
        )
    if len(observations) != observation_count:
        raise ValueError("raw/index observation population mismatch")
    return tuple(sorted(observations, key=lambda item: item.observation_id))


def _read_pinned_bytes(path: Path, expected_sha256: str, name: str) -> bytes:
    content = path.read_bytes()
    if sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"{name} SHA-256 mismatch: {path}")
    return content


def _manifest_bytes(
    population: Tuple[CanonicalObservation, ...],
    cohort: BuiltCohort,
    git_commit: str,
) -> bytes:
    scene_counts = {
        scene_id: sum(item.scene_id == scene_id for item in population)
        for scene_id in sorted({item.scene_id for item in population})
    }
    selected_scene_counts = {
        scene_id: sum(item.scene_id == scene_id for item in cohort.observations)
        for scene_id in sorted({item.scene_id for item in cohort.observations})
    }
    manifest = {
        "cohort_id": COHORT_ID,
        "files": {
            "cohort.jsonl": {
                "byte_length": len(cohort.cohort_jsonl),
                "row_count": len(cohort.observations),
                "sha256": cohort.cohort_sha256,
            }
        },
        "git_commit": git_commit,
        "population": {
            "example_count": sum(len(item.example_ids) for item in population),
            "observation_count": len(population),
            "scene_count": len(scene_counts),
            "scene_observation_counts": scene_counts,
        },
        "schema_version": 1,
        "selection": {
            "algorithm": ALGORITHM,
            "domain_hex": SELECTION_DOMAIN.hex(),
            "scene_quotas": dict(cohort.scene_quotas),
            "scene_selected_counts": selected_scene_counts,
            "selected_example_count": sum(
                len(item.example_ids) for item in cohort.observations
            ),
            "selected_observation_count": len(cohort.observations),
            "selected_scene_count": len(selected_scene_counts),
            "selection_sha256": cohort.selection_sha256,
            "target_observation_count": TARGET_COUNT,
        },
        "source": {
            "dataset": DATASET,
            "evidence_index": EVIDENCE_INDEX_PATH.as_posix(),
            "evidence_index_sha256": EVIDENCE_INDEX_SHA256,
            "evidence_key": EVIDENCE_KEY,
            "evidence_manifest": EVIDENCE_MANIFEST_PATH.as_posix(),
            "evidence_manifest_sha256": EVIDENCE_MANIFEST_SHA256,
            "evidence_root": EVIDENCE_ROOT.as_posix(),
            "raw_split": RAW_SPLIT_PATH.as_posix(),
            "raw_split_sha256": RAW_SPLIT_SHA256,
            "split": SPLIT,
        },
    }
    return (
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_cohort_from_sources(*, git_commit: str) -> CohortArtifacts:
    """Reconstruct the frozen population and build the exact cohort package."""

    if not isinstance(git_commit, str) or _LOWER_HEX_40.fullmatch(git_commit) is None:
        raise ValueError("git_commit must be 40 lowercase hex characters")
    manifest_bytes = _read_pinned_bytes(
        EVIDENCE_MANIFEST_PATH,
        EVIDENCE_MANIFEST_SHA256,
        "evidence manifest",
    )
    index_bytes = _read_pinned_bytes(
        EVIDENCE_INDEX_PATH,
        EVIDENCE_INDEX_SHA256,
        "evidence index",
    )
    raw_split_bytes = _read_pinned_bytes(
        RAW_SPLIT_PATH,
        RAW_SPLIT_SHA256,
        "raw split",
    )
    population = reconstruct_source_observations(
        manifest_bytes,
        index_bytes,
        raw_split_bytes,
    )
    scene_counts = {
        scene_id: sum(item.scene_id == scene_id for item in population)
        for scene_id in sorted({item.scene_id for item in population})
    }
    if sum(len(item.example_ids) for item in population) != EXPECTED_EXAMPLE_COUNT:
        raise ValueError("source example population drift")
    if len(population) != EXPECTED_OBSERVATION_COUNT:
        raise ValueError("source observation population drift")
    if len(scene_counts) != EXPECTED_SCENE_COUNT:
        raise ValueError("source scene population drift")
    if scene_counts != dict(OFFICIAL_SCENE_COUNTS):
        raise ValueError("source scene observation populations drift")
    cohort = build_cohort(population, target_count=TARGET_COUNT)
    if cohort.scene_quotas != OFFICIAL_SCENE_QUOTAS:
        raise ValueError("source cohort quotas drift")
    return CohortArtifacts(
        population=population,
        cohort=cohort,
        manifest_json=_manifest_bytes(population, cohort, git_commit),
    )
